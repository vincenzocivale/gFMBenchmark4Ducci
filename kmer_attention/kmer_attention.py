import torch
import torch.nn as nn
import torch.nn.functional as F


class KmerAttention(nn.Module):
    def __init__(self, nkmers : int, de : int, dk : int):
        super().__init__()
        self.embed = nn.Linear(nkmers, de)
        self.norm = nn.LayerNorm(de)
        self.Wq = nn.Linear(de, dk)
        self.Wk = nn.Linear(de, dk)
        self.Wv = nn.Linear(de, dk)
        self.Wo = nn.Linear(dk, de)

    def forward(self, x : torch.Tensor, attn_mask : torch.Tensor) -> torch.Tensor:
        # x: (B, S, nkmers), attn_mask: (B, 1, S) bool, True = real window
        x = self.embed(x)                                   # (B, S, de)
        h = self.norm(x)
        h = F.scaled_dot_product_attention(self.Wq(h), self.Wk(h), self.Wv(h), attn_mask=attn_mask)
        return x + self.Wo(h)                               # (B, S, de)


class MultiChannelKmerAttention(nn.Module):
    def __init__(self, nkmers : list[int], de : int, dk : int):
        super().__init__()
        self.channels = nn.ModuleList([KmerAttention(n, de, dk) for n in nkmers])

    def forward(self, feats : list[torch.Tensor], attn_mask : list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.channels):
            raise ValueError(f"got {len(feats)} channels, model has {len(self.channels)}")
        return [channel(x, m) for channel, x, m in zip(self.channels, feats, attn_mask)]


def kmer_att_mean_pooling(out : torch.Tensor | list[torch.Tensor], 
                          attn_mask : torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    # out: (B, S, de) or list of them (one per channel); attn_mask: (B, 1, S) bool or list of them
    # → (B, de); with multiple channels the per-channel means are averaged too (quick hack for tests)
    if not isinstance(out, list):
        out, attn_mask = [out], [attn_mask]

    pooled = []
    for x, mask in zip(out, attn_mask):
        valid = mask[:, 0, :, None].to(x.dtype)                         # (B, S, 1)
        pooled.append((x * valid).sum(1) / valid.sum(1).clamp(min=1))   # (B, de)

    return torch.stack(pooled).mean(0)                                  # (n_channels, B, de) → (B, de)
