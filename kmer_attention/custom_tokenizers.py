from transformers import PreTrainedTokenizer
import torch
import torch.nn.functional as F
from itertools import product
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "../training"))
from train_utils import mlm_mask

class GenomicTokenizer(PreTrainedTokenizer):
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2, "+": 3, "-": 4,
                      "A": 5, "C": 6, "T": 7, "G": 8, "5mC": 9}
        super().__init__(
            pad_token="[PAD]",
            mask_token="[MASK]",
            unk_token="[UNK]",
            additional_special_tokens=["+", "-"]
        )

    def get_vocab(self):
        return self.vocab
    
    def _tokenize(self, text):
        return list(text.upper())

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["[UNK]"])

    def _convert_id_to_token(self, index):
        return {v: k for k, v in self.vocab.items()}.get(index, "[UNK]")

#############################
class KmerFreq:
    def __init__(self,token_ids : list, k : int):
        self.token_ids = token_ids
        self.k = k
        self.kmers = [" ".join(map(str, kmer)) for kmer in product(self.token_ids, repeat=self.k)]

def n_slices(lengths : torch.Tensor, win : int, overlap : float) -> torch.Tensor:
    step = win - round(overlap*win)
    n_full = torch.clamp(lengths - win, min=0) // step + 1             
    last_end = (n_full - 1) * step + win                               
    return n_full + ((lengths - last_end) >= win/2).long()             

class SLKT: # short-length k-mer transform
    def __init__(self,tokenizer : PreTrainedTokenizer, k : int, win_size : int, overlap : float = 0.3,
                ids_tok_to_skip : list | None = None, with_masking : bool = False):
        assert win_size >= k, "window has to contain at least one k-mer"
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.ids_to_skip = ids_tok_to_skip

        self.k = k
        self.win_size = win_size
        self.overlap = overlap
        self.step = win_size - round(overlap*win_size)

        # PAD and ids_tok_to_skip are removed from the sequence and excluded from the k-mer vocab
        all_ids = list(tokenizer.get_vocab().values())
        removed = {self.pad_token_id} | set(ids_tok_to_skip or [])
        if with_masking:
            removed.discard(tokenizer.mask_token_id)
        self.removed_ids = torch.tensor(sorted(removed))
        token_ids = [v for v in all_ids if v not in removed]
        self.kmerfreq = KmerFreq(token_ids,k)

        # k-mer → base-V integer (same order as self.kmerfreq.kmers)
        self.V = len(token_ids)
        self.lut = torch.full((max(all_ids) + 1,), -1, dtype=torch.long)     # id → vocab position, -1 = removed
        self.lut[torch.tensor(token_ids)] = torch.arange(self.V)
        self.pow = self.V ** torch.arange(k - 1, -1, -1)                     # [V^(k-1), ..., V, 1]

    def transform(self, seq : torch.Tensor) -> torch.Tensor:
        # seq: (L,) token ids of a single example → (nslices, V^k) per-window frequencies
        win, step, k = self.win_size, self.step, self.k
        lut, pow = self.lut.to(seq.device), self.pow.to(seq.device)
        
        ids = seq[~torch.isin(seq, self.removed_ids.to(seq.device))]
        L = ids.numel()
        nslices = int(n_slices(torch.tensor(L), win, self.overlap))

        # pad (with PAD) or crop so that unfold yields exactly nslices windows of length win
        tot = (nslices - 1) * step + win
        padded = F.pad(ids, (0, max(tot - L, 0)), value=self.pad_token_id)[:tot]

        kmers = padded.unfold(0, win, step).unfold(1, k, 1)                  # (nslices, win-k+1, k)
        idx = lut[kmers]                                                     # -1 on tail PAD
        keep = (idx >= 0).all(-1)                                            # (nslices, win-k+1)
        codes = (idx.clamp(min=0) * pow).sum(-1)                             # (nslices, win-k+1)

        counts = torch.zeros(nslices, self.V ** k, device=seq.device)
        counts.scatter_add_(1, codes, keep.float())

        # remainder < win/2 (cropped above): extend the last window to the end of the sequence
        if L > tot:
            last = ids[(nslices - 1) * step:].unfold(0, k, 1)                # (n_last_kmers, k)
            last_codes = (lut[last] * pow).sum(-1)
            counts[-1] = torch.bincount(last_codes, minlength=self.V ** k).float()

        return counts / counts.sum(-1, keepdim=True).clamp(min=1)

@dataclass(frozen=True)
class SLKTargs:
    k : int
    win_size : int
    overlap : float 

class SLKT_tokenizer:
    def __init__(self, base_tokenizer : PreTrainedTokenizer, slkts_args : list[SLKTargs], with_masking : bool = False):
        self.base_tokenizer = base_tokenizer
        self.with_masking = with_masking

        self.slkts_args = slkts_args
        self.slkts = [SLKT(tokenizer = self.base_tokenizer,
                          ids_tok_to_skip=self.base_tokenizer.all_special_ids,
                          with_masking=with_masking,
                          **vars(slkt_args)) for slkt_args in self.slkts_args]

    def __getattr__(self, name): # NB: assignments are not delegated: set attributes on base_tokenizer directly
        if name == "base_tokenizer":             
          raise AttributeError(name)
        return getattr(self.base_tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.base_tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self.base_tokenizer)

    def encode_kmers(self, seq: list) -> dict:
        # seq: token list (e.g. ["A", "C", "5mC"])
        # feats: one (nslices, nkmers) tensor per branch; length: tokens left after removing PAD/specials
        x = torch.tensor(self.convert_tokens_to_ids(seq), dtype=torch.long)
        L = int((~torch.isin(x, self.slkts[0].removed_ids)).sum())
        return {"feats": [slkt.transform(x) for slkt in self.slkts], "length": L, "seq": seq}

    def encode_kmers_withMLM(self, seq : list[str], mlm_prob : float = 0.15):
        assert self.with_masking, "MLM needs with_masking=True, otherwise SLKT drops [MASK] tokens"
        seq_ids = torch.tensor(self.convert_tokens_to_ids(seq), dtype=torch.long)
        input_ids, labels = mlm_mask(seq_ids, self.base_tokenizer, mlm_prob=mlm_prob)

        # compact labels to the same coordinates as the windows (removed tokens dropped, [MASK] kept)
        keep = ~torch.isin(input_ids, self.slkts[0].removed_ids)
        input_ids, labels = input_ids[keep], labels[keep]

        return {"feats": [slkt.transform(input_ids) for slkt in self.slkts],
                "length": int(keep.sum()), "labels": labels}

