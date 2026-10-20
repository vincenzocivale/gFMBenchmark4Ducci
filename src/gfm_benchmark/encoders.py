"""Lazy backends: the baseline never imports a GPU framework."""

from itertools import product
from typing import Protocol

import numpy as np

from .data import reverse_complement


class Encoder(Protocol):
    def encode(self, sequences: list[str]) -> np.ndarray:
        """Return a finite float matrix [sequences, features]."""
        ...


class KmerEncoder:
    def __init__(self, k=6, canonical=True, **kwargs):
        if not isinstance(k, int) or not 1 <= k <= 10:
            raise ValueError("k must be an integer between 1 and 10")
        self.k, self.canonical = k, canonical
        words = ("".join(chars) for chars in product("ACGT", repeat=k))
        self.vocabulary = sorted({self.key(word) for word in words})
        self.index = {word: i for i, word in enumerate(self.vocabulary)}

    def key(self, word):
        return min(word, reverse_complement(word)) if self.canonical else word

    def encode(self, sequences):
        result = np.zeros((len(sequences), len(self.index)), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            for start in range(len(sequence) - self.k + 1):
                word = sequence[start:start + self.k]
                if "N" not in word:
                    result[row, self.index[self.key(word)]] += 1
        totals = result.sum(axis=1, keepdims=True)
        return result / np.maximum(totals, 1)


class HuggingFaceEncoder:
    """Last hidden state mean pooling, excluding padding and special tokens."""
    def __init__(self, model_id, revision=None, device="cpu", max_tokens=512,
                 trust_remote_code=False, dnabert2=False, **kwargs):
        import torch
        from transformers import AutoModel, AutoTokenizer
        options = dict(revision=revision, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        if dnabert2:
            from transformers.models.bert.configuration_bert import BertConfig
            options["config"] = BertConfig.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, **options).to(device).eval()
        self.torch, self.device, self.max_tokens = torch, device, max_tokens

    def encode(self, sequences):
        tokens = self.tokenizer(sequences, padding=True, truncation=False,
                                return_special_tokens_mask=True, return_tensors="pt")
        special = tokens.pop("special_tokens_mask").to(self.device)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        if tokens["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Token budget exceeded; reduce common window_bp (no silent truncation)")
        mask = tokens["attention_mask"].bool() & ~special.bool()
        with self.torch.inference_mode():
            output = self.model(**tokens)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return pooled.cpu().numpy()


class NTv3Encoder:
    """Official JAX backend; species condition is explicit for post-training."""
    def __init__(self, model_name, layer=10, species=None, posttrained=False,
                 use_bfloat16=True, **kwargs):
        from nucleotide_transformer_v3 import pretrained
        if posttrained and not species:
            raise ValueError("NTv3 post requires an explicit species condition")
        loader = (pretrained.get_posttrained_ntv3_model if posttrained
                  else pretrained.get_pretrained_ntv3_model)
        self.model, self.tokenizer, self.config = loader(
            model_name=model_name, embeddings_layers_to_save=(layer,), use_bfloat16=use_bfloat16)
        if not 1 <= layer <= self.config.num_layers:
            raise ValueError(f"layer must be between 1 and {self.config.num_layers}")
        self.layer, self.species, self.posttrained = layer, species, posttrained

    def encode(self, sequences):
        # Process singly to avoid introducing padding from other samples.
        result = []
        multiple = 2 ** self.config.num_downsamples
        for sequence in sequences:
            # Native N padding is needed by the convolutional architecture.
            padded = sequence + "N" * (-len(sequence) % multiple)
            tokens = self.tokenizer.batch_np_tokenize([padded])
            options = {"species_tokens": self.model.encode_species(self.species)} if self.posttrained else {}
            output = self.model(tokens=tokens, **options)
            hidden = np.asarray(output[f"embeddings_{self.layer}"], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("Unexpected NTv3 embedding shape")
            # Transformer positions summarize bins of `multiple` bases.
            weights = np.clip(len(sequence) - np.arange(hidden.shape[1]) * multiple, 0, multiple)
            if not weights.sum():
                raise ValueError("Unexpected NTv3 embedding shape")
            result.append(np.average(hidden[0], axis=0, weights=weights))
        return np.asarray(result, dtype=np.float32)


class Evo2Encoder:
    def __init__(self, model_name="evo2_7b", layer="blocks.28", device="cuda:0", **kwargs):
        import torch
        from evo2 import Evo2
        self.model = Evo2(model_name)
        self.model.model.to(device).eval()
        self.torch, self.device, self.layer = torch, device, layer

    def encode(self, sequences):
        rows = []
        for sequence in sequences:
            ids = self.torch.tensor([self.model.tokenizer.tokenize(sequence)], device=self.device)
            with self.torch.inference_mode():
                _, embeddings = self.model(ids, return_embeddings=True, layer_names=[self.layer])
                rows.append(embeddings[self.layer].float().mean(dim=1).cpu().numpy()[0])
        return np.asarray(rows, dtype=np.float32)


REGISTRY = {"kmer": KmerEncoder, "hf": HuggingFaceEncoder,
            "ntv3": NTv3Encoder, "evo2": Evo2Encoder}


def create_encoder(config):
    options = dict(config)
    backend = options.pop("backend")
    try:
        factory = REGISTRY[backend]
    except KeyError:
        raise ValueError(f"Unknown backend {backend!r}; available: {sorted(REGISTRY)}") from None
    return factory(**options)
