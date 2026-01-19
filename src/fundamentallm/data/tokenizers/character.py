"""Character-level tokenizer implementation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer with special token support."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, min_frequency: int = 1) -> None:
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.min_frequency = min_frequency
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self._trained = False

    def train(self, texts: List[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)
        # Filter by min_frequency and ensure deterministic ordering
        vocab_chars = sorted([ch for ch, freq in counts.items() if freq >= self.min_frequency])
        tokens = self.special_tokens + vocab_chars
        self.char_to_id = {ch: idx for idx, ch in enumerate(tokens)}
        self.id_to_char = {idx: ch for ch, idx in self.char_to_id.items()}
        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        self._require_trained()
        tokens = [self.char_to_id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        self._require_trained()
        pieces: List[str] = []
        special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
        for tok in tokens:
            if skip_special_tokens and tok in special_ids:
                if tok == self.unk_token_id:
                    pieces.append(self.UNK_TOKEN)
                continue
            char = self.id_to_char.get(tok, self.UNK_TOKEN)
            pieces.append(char)
        return "".join(pieces)

    def save(self, path: Path) -> None:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_frequency": self.min_frequency,
            "char_to_id": self.char_to_id,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls(min_frequency=int(data.get("min_frequency", 1)))
        tokenizer.special_tokens = list(data["special_tokens"])
        tokenizer.char_to_id = {str(k): int(v) for k, v in data["char_to_id"].items()}
        tokenizer.id_to_char = {idx: ch for ch, idx in tokenizer.char_to_id.items()}
        tokenizer._trained = True
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.char_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.char_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.char_to_id[self.EOS_TOKEN]
