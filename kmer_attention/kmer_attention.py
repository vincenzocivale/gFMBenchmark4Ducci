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


def windows_to_positions(logits : torch.Tensor, nslices : torch.Tensor, length : torch.Tensor,
                         win : int, step : int) -> torch.Tensor:
    # logits: (B, S, P, V) per inner window position → (B, L_max, V), averaged over the windows covering each position
    # window j starts at j*step; non-last windows cover win tokens, the last real window always ends at length
    B, S, P, V = logits.shape
    dev = logits.device
    nslices, length = nslices.to(dev), length.to(dev)
    L_max = int(length.max())

    j = torch.arange(S, device=dev)
    i = torch.arange(P, device=dev)
    pos = j[:, None] * step + i[None, :]                                                # (S, P)

    is_last = j[None, :] == nslices[:, None] - 1                                        # (B, S)
    win_len = torch.where(is_last, length[:, None] - j[None, :] * step, win)            # (B, S)
    valid = (j[None, :] < nslices[:, None])[..., None] & (i[None, None, :] < win_len[..., None])   # (B, S, P)

    flat = (torch.arange(B, device=dev)[:, None, None] * L_max + pos[None])[valid]
    summed = torch.zeros(B * L_max, V, device=dev, dtype=logits.dtype).index_add_(0, flat, logits[valid])
    count = torch.zeros(B * L_max, device=dev, dtype=logits.dtype).index_add_(0, flat, torch.ones_like(flat, dtype=logits.dtype))
    return (summed / count.clamp(min=1)[:, None]).view(B, L_max, V)


class KmerAttentionMLMHead(nn.Module):
    # to be stacked after a KmerAttention: kmer_att is only read for de, it is not stored
    def __init__(self, kmer_att : KmerAttention, slkt_args, vocab_size : int):
        # slkt_args: the SLKTargs used by the tokenizer for this channel (win_size, overlap)
        super().__init__()
        self.win = slkt_args.win_size
        self.step = slkt_args.win_size - round(slkt_args.overlap * slkt_args.win_size)
        self.P = self.win + self.win // 2                   # extended last window is < 1.5 * win
        self.vocab_size = vocab_size
        self.head = nn.Linear(kmer_att.embed.out_features, self.P * vocab_size)

    def forward(self, out : torch.Tensor, nslices : torch.Tensor, length : torch.Tensor) -> torch.Tensor:
        # out: (B, S, de) KmerAttention output → (B, L_max, V)
        logits = self.head(out).view(*out.shape[:2], self.P, self.vocab_size)   # (B, S, P, V)
        return windows_to_positions(logits, nslices, length, self.win, self.step)


class MultiChannelMLMHead(nn.Module):
    # to be stacked after a MultiChannelKmerAttention: one head per channel, logits averaged over channels
    def __init__(self, mc_kmer_att : MultiChannelKmerAttention, slkts_args : list, vocab_size : int):
        super().__init__()
        if len(slkts_args) != len(mc_kmer_att.channels):
            raise ValueError(f"got {len(slkts_args)} SLKTargs, model has {len(mc_kmer_att.channels)} channels")
        self.heads = nn.ModuleList([KmerAttentionMLMHead(ch, a, vocab_size)
                                    for ch, a in zip(mc_kmer_att.channels, slkts_args)])

    def forward(self, outs : list[torch.Tensor], nslices : list[torch.Tensor], length : torch.Tensor) -> torch.Tensor:
        # outs: MultiChannelKmerAttention output, list of (B, S_c, de) → (B, L_max, V)
        logits = [h(o, n, length) for h, o, n in zip(self.heads, outs, nslices)]
        return torch.stack(logits).mean(0)
