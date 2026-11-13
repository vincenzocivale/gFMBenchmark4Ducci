import torch
from torch.nn.utils.rnn import pad_sequence


class SLKTCollator: #TODO: implement masking, decide how, rigth now attnmask is always true apart from padding windows
    # examples: list of encode_kmers outputs → per-branch padded feats and key masks
    def __call__(self, examples : list[dict]) -> dict:
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
#as now, it pads to 0 to the max lenght of kmer tensor 