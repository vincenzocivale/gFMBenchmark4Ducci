# K-mer attention

Attention over windows of a sequence, where each window is represented by its k-mer frequency vector.
Full example: [example_kmeratt.py](example_kmeratt.py).

## Pipeline

```
sequence → SLKT_tokenizer.encode_kmers → SLKTCollator → (MultiChannel)KmerAttention → kmer_att_mean_pooling
```

| Component | File | Input → output |
|---|---|---|
| `NTv3Tokenizer` | `ntv3_tokenizer_asHF.py` | HF tokenizer, same vocab and ids as NTv3 |
| `SLKTargs` | `custom_tokenizers.py` | one channel config: `k`, `win_size`, `overlap` |
| `SLKT_tokenizer` | `custom_tokenizers.py` | wraps a HF tokenizer; one example → per-channel k-mer frequencies |
| `SLKTCollator` | `custom_collators.py` | list of examples → padded batch + masks |
| `KmerAttention` | `kmer_attention.py` | one channel: `(B, S, nkmers)` → `(B, S, de)` |
| `MultiChannelKmerAttention` | `kmer_attention.py` | list of channels → list of `(B, S_c, de)` |
| `kmer_att_mean_pooling` | `kmer_attention.py` | masked mean over windows (and channels) → `(B, de)` |

## Tokenizer

```python
tokenizer = SLKT_tokenizer(NTv3Tokenizer(), [SLKTargs(k=4, win_size=64), SLKTargs(k=3, win_size=256, overlap=0.5)])
ex = tokenizer.encode_kmers(list("ACGT..."))
# ex["feats"]:  list, one (nslices, V^k) tensor per channel; each row sums to 1
# ex["length"]: tokens left after removing PAD and special tokens
```

- Any HF tokenizer can be wrapped; everything not defined by `SLKT_tokenizer` is delegated to it.
- PAD and all special tokens are removed before windowing; the k-mer vocab is built from the remaining tokens (`V` = 5 for `NTv3Tokenizer`: A, T, C, G, N).
- Windows: full windows with step `win_size * (1 - overlap)`, plus a tail window if the remainder is ≥ `win_size / 2`.
- k-mer order: `tokenizer.slkts[c].kmerfreq.kmers`.

**Notes**
- Without `include_softmasked_tokens`, lowercase bases become `<unk>` (as in NTv3) and SLKT drops them, since `<unk>` is a special token. E.g. a length-700 sequence with half lowercase characters gave `length` 335. If your data has lowercase bases, either enable soft-masked tokens or uppercase the sequences first.
- `all_special_ids` includes CLS, BOS and EOS, so SLKT drops them even if enabled. This is the intended behaviour for k-mer frequencies.

## Collator

```python
loader = DataLoader(dataset, batch_size=64, collate_fn=SLKTCollator())   # dataset returns encode_kmers outputs
```

Output (lists have one element per channel):

| Key | Shape | Content |
|---|---|---|
| `feats` | list of `(B, S_max_c, nkmers_c)` | frequencies; padding windows are all zeros |
| `attn_mask` | list of `(B, 1, S_max_c)` bool | `True` = real window (masks keys) |
| `nslices` | list of `(B,)` | real windows per example |
| `length` | `(B,)` | tokens per example after removals |

Move lists to the device element by element. No MLM masking yet.

## Model

```python
model = MultiChannelKmerAttention(nkmers=[625, 125], de=128, dk=64)
out = model(batch["feats"], batch["attn_mask"])           # list of (B, S_c, de)
x = kmer_att_mean_pooling(out, batch["attn_mask"])        # (B, de)
```

- `KmerAttention`: linear embedding `nkmers → de`, then a pre-LN single-head self-attention block over windows with a residual connection.
- `nkmers` per channel: `len(tokenizer.slkts[c].kmerfreq.kmers)`.
- Outputs on padding windows are meaningless: ignore them via `attn_mask` / `nslices`.
- `kmer_att_mean_pooling` averages real windows per channel, then averages channels (quick solution for tests; all channels weigh the same).
