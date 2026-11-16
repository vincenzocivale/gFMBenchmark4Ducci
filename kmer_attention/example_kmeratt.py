import random
import torch
from torch.utils.data import Dataset, DataLoader

from ntv3_tokenizer_asHF import NTv3Tokenizer
from custom_tokenizers import SLKT_tokenizer, SLKTargs
from custom_collators import SLKTCollator
from kmer_attention import KmerAttention, MultiChannelKmerAttention, kmer_att_mean_pooling


class SeqDataset(Dataset):
    # returns raw token lists; tokenization happens in the collator
    def __init__(self, seqs : list[str]):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return list(self.seqs[i])


def to_device(batch : dict, device : str) -> dict:
    return {k: [t.to(device) for t in v] if isinstance(v, list) else v.to(device) for k, v in batch.items()}


def run(channels : list[SLKTargs], seqs : list[str], device : str):
    tokenizer = SLKT_tokenizer(NTv3Tokenizer(), channels)
    nkmers = [len(slkt.kmerfreq.kmers) for slkt in tokenizer.slkts]
    loader = DataLoader(SeqDataset(seqs), batch_size=4, collate_fn=SLKTCollator(tokenizer))
    batch = to_device(next(iter(loader)), device)

    if len(channels) == 1:
        model = KmerAttention(nkmers[0], de=128, dk=64).to(device)
        out = model(batch["feats"][0], batch["attn_mask"][0])            # (B, S, de)
        pooled = kmer_att_mean_pooling(out, batch["attn_mask"][0])       # (B, de)
        print("KmerAttention out:", tuple(out.shape))
    else:
        model = MultiChannelKmerAttention(nkmers, de=128, dk=64).to(device)
        out = model(batch["feats"], batch["attn_mask"])                  # list of (B, S_c, de)
        pooled = kmer_att_mean_pooling(out, batch["attn_mask"])          # (B, de)
        print("MultiChannelKmerAttention out:", [tuple(o.shape) for o in out])

    print("pooled:", tuple(pooled.shape))


if __name__ == "__main__":
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = ["".join(random.choice("ACGT") for _ in range(random.randint(200, 2000))) for _ in range(8)]

    run([SLKTargs(k=4, win_size=64, overlap=0)], seqs, device)
    run([SLKTargs(k=4, win_size=64, overlap=0), SLKTargs(k=3, win_size=256, overlap=0.5)], seqs, device)
