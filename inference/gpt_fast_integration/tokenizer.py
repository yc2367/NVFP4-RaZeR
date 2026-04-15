import os
import json
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import Dict

class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


class HFTokenizerJsonWrapper(TokenizerInterface):
    """Wrapper around HuggingFace `tokenizers` JSON files (e.g. Qwen tokenizer.json)."""

    def __init__(self, tokenizer_json_path: Path, checkpoint_dir: Path | None = None):
        super().__init__(tokenizer_json_path)
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HuggingFace tokenizer.json support requires the `tokenizers` package. "
                "Install it (e.g. `pip install tokenizers`) or provide a `tokenizer.model` instead."
            ) from e

        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

        self._bos_id = None
        self._eos_id = None

        # Prefer model config if present (common in HF repos)
        if checkpoint_dir is not None:
            cfg_path = Path(checkpoint_dir) / "config.json"
            if cfg_path.is_file():
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                self._bos_id = cfg.get("bos_token_id", None)
                self._eos_id = cfg.get("eos_token_id", None)

        # Fall back to common Qwen special tokens.
        if self._bos_id is None:
            self._bos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self._eos_id is None:
            self._eos_id = self.tokenizer.token_to_id("<|im_end|>")

        if self._bos_id is None or self._eos_id is None:
            raise RuntimeError(
                f"Could not determine BOS/EOS ids from {tokenizer_json_path}. "
                "Expected config.json with bos_token_id/eos_token_id or standard Qwen special tokens."
            )

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def bos_id(self):
        return int(self._bos_id)

    def eos_id(self):
        return int(self._eos_id)

def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.
    
    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    model_name_str = str(model_name).lower()
    tokenizer_path = Path(tokenizer_model_path)

    if tokenizer_path.name.endswith("tokenizer.json"):
        checkpoint_dir = Path(model_name).parent if hasattr(model_name, "parent") else None
        return HFTokenizerJsonWrapper(tokenizer_path, checkpoint_dir=checkpoint_dir)

    if "llama-3" in model_name_str:
        return TiktokenWrapper(tokenizer_model_path)

    return SentencePieceWrapper(tokenizer_model_path)
