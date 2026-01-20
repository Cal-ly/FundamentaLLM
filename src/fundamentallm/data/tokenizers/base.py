"""Abstract base class for tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List


class BaseTokenizer(ABC):
    """Base interface for all tokenizers."""

    _trained: bool = False

    @abstractmethod
    def train(self, texts: List[str]) -> None:  # pragma: no cover - interface only
        """Learn tokenization from the provided corpus."""

    @abstractmethod
    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> List[int]:  # pragma: no cover - interface only
        """Convert a string to token ids."""

    @abstractmethod
    def decode(
        self, tokens: List[int], skip_special_tokens: bool = True
    ) -> str:  # pragma: no cover - interface only
        """Convert token ids back to text."""

    def batch_encode(
        self, texts: Iterable[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        """Encode multiple texts in a single call."""
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def batch_decode(
        self, sequences: Iterable[Iterable[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode multiple token sequences in a single call."""
        return [
            self.decode(list(seq), skip_special_tokens=skip_special_tokens) for seq in sequences
        ]

    @abstractmethod
    def save(self, path: Path) -> None:  # pragma: no cover - interface only
        """Serialize tokenizer state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTokenizer":  # pragma: no cover - interface only
        """Load tokenizer state from disk."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:  # pragma: no cover - interface only
        """Return vocabulary size."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:  # pragma: no cover - interface only
        """Return the padding token id."""

    @property
    @abstractmethod
    def unk_token_id(self) -> int:  # pragma: no cover - interface only
        """Return the unknown token id."""

    @property
    @abstractmethod
    def bos_token_id(self) -> int:  # pragma: no cover - interface only
        """Return the beginning-of-sequence token id."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int:  # pragma: no cover - interface only
        """Return the end-of-sequence token id."""

    def _require_trained(self) -> None:
        """Ensure tokenizer has been trained before use."""
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before encoding or decoding.")
