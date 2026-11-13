import json
import os
from transformers import PreTrainedTokenizer


class NTv3Tokenizer(PreTrainedTokenizer):
    # HF version of the NTv3 NucleotideTokenizer: same vocab, ids and behaviour (1 token per base)
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    SPECIAL_TOKENS = ["<unk>", "<pad>", "<mask>", "<cls>", "<eos>", "<bos>"]
    STANDARD_TOKENS = ["A", "T", "C", "G", "N"]
    SOFTMASKED_TOKENS = ["a", "t", "c", "g", "n"]

    def __init__(self, include_softmasked_tokens : bool = False, prepend_cls_token : bool = False,
                 prepend_bos_token : bool = False, append_eos_token : bool = False,
                 vocab_file : str | None = None, **kwargs):
        # vocab_file is ignored: the vocab is fully determined by include_softmasked_tokens
        if prepend_cls_token and prepend_bos_token:
            raise ValueError("Cannot prepend both BOS and CLS token, must choose only one")

        tokens = self.SPECIAL_TOKENS + self.STANDARD_TOKENS
        if include_softmasked_tokens:
            tokens += self.SOFTMASKED_TOKENS
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.ids_to_tokens = {i: tok for tok, i in self.vocab.items()}

        self.include_softmasked_tokens = include_softmasked_tokens
        self.prepend_cls_token = prepend_cls_token
        self.prepend_bos_token = prepend_bos_token
        self.append_eos_token = append_eos_token

        for name, tok in zip(["unk_token", "pad_token", "mask_token", "cls_token", "eos_token", "bos_token"],
                             self.SPECIAL_TOKENS):
            kwargs.setdefault(name, tok)
        super().__init__(include_softmasked_tokens=include_softmasked_tokens,
                         prepend_cls_token=prepend_cls_token,
                         prepend_bos_token=prepend_bos_token,
                         append_eos_token=append_eos_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict:
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text : str) -> list[str]:
        return [c for c in text if not c.isspace()]

    def _convert_token_to_id(self, token : str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def _convert_id_to_token(self, index : int) -> str:
        return self.ids_to_tokens.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens : list[str]) -> str:
        return "".join(tokens)

    def _prefix_suffix(self) -> tuple[list[int], list[int]]:
        prefix = [self.cls_token_id] if self.prepend_cls_token else [self.bos_token_id] if self.prepend_bos_token else []
        suffix = [self.eos_token_id] if self.append_eos_token else []
        return prefix, suffix

    def build_inputs_with_special_tokens(self, token_ids_0 : list[int], token_ids_1 : list[int] | None = None) -> list[int]:
        prefix, suffix = self._prefix_suffix()
        return prefix + token_ids_0 + (token_ids_1 or []) + suffix

    def get_special_tokens_mask(self, token_ids_0 : list[int], token_ids_1 : list[int] | None = None,
                                already_has_special_tokens : bool = False) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        prefix, suffix = self._prefix_suffix()
        return [1] * len(prefix) + [0] * (len(token_ids_0) + len(token_ids_1 or [])) + [1] * len(suffix)

    def save_vocabulary(self, save_directory : str, filename_prefix : str | None = None) -> tuple[str]:
        path = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)
        return (path,)
