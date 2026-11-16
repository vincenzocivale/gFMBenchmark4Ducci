import torch

def mlm_mask(ids, tokenizer, mlm_prob=0.15, mask_prob=0.8, random_prob=0.1, generator=None):
    # ids: LongTensor (L,), no padding
    ids = ids.clone()
    labels = ids.clone()

    special = torch.isin(ids, torch.tensor(tokenizer.all_special_ids))
    probs = torch.full(ids.shape, mlm_prob)
    probs[special] = 0.0
    masked = torch.bernoulli(probs, generator=generator).bool()
    labels[~masked] = -100

    # mask_prob of masked positions -> <mask>
    replaced = torch.bernoulli(torch.full(ids.shape, mask_prob), generator=generator).bool() & masked
    ids[replaced] = tokenizer.mask_token_id

    # random_prob of masked positions -> random non-special token
    rnd_scaled = random_prob / (1 - mask_prob)
    randomized = torch.bernoulli(torch.full(ids.shape, rnd_scaled), generator=generator).bool() & masked & ~replaced
    candidates = torch.tensor([i for i in range(len(tokenizer)) if i not in tokenizer.all_special_ids])
    ids[randomized] = candidates[torch.randint(len(candidates), (int(randomized.sum()),), generator=generator)]

    # remaining masked positions stay unchanged
    return ids, labels
