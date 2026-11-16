import torch
from torch.nn.utils.rnn import pad_sequence
from custom_tokenizers import SLKT_tokenizer


def pad_kmer_examples(examples : list[dict]) -> dict:
    # examples: list of encode_kmers(_withMLM) outputs → per-channel padded feats (all-zero padding windows) and key masks
    feats, attn_masks, nslices = [], [], []
    for b in range(len(examples[0]["feats"])):
        branch = [ex["feats"][b] for ex in examples]
        n = torch.tensor([f.shape[0] for f in branch])
        x = pad_sequence(branch, batch_first=True)                  # (B, S_max, nkmers)
        valid = torch.arange(x.shape[1])[None, :] < n[:, None]     # (B, S_max)
        feats.append(x)
        attn_masks.append(valid[:, None, :])                        # (B, 1, S_max)
        nslices.append(n)
    return {"feats": feats, "attn_mask": attn_masks, "nslices": nslices,
            "length": torch.tensor([ex["length"] for ex in examples])}


class SLKTCollator:
    # raw token lists → k-mer batch (no masking)
    def __init__(self, tokenizer : SLKT_tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seqs : list[list[str]]) -> dict:
        return pad_kmer_examples([self.tokenizer.encode_kmers(s) for s in seqs])


class DataCollatorMLMforKmerAttention:
    # raw token lists → masked k-mer batch + labels (B, L_max), -100 where nothing to predict
    def __init__(self, tokenizer : SLKT_tokenizer, mlm_probability : float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, seqs : list[list[str]]) -> dict:
        examples = [self.tokenizer.encode_kmers_withMLM(s, mlm_prob=self.mlm_probability) for s in seqs]
        batch = pad_kmer_examples(examples)
        batch["labels"] = pad_sequence([ex["labels"] for ex in examples], batch_first=True, padding_value=-100)
        return batch
