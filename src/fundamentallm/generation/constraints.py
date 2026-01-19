"""Generation constraints such as stop sequences."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

from fundamentallm.data.tokenizers.base import BaseTokenizer


class StopSequenceConstraint:
    """Stop generation when specified token sequences appear."""

    def __init__(self, stop_sequences: Sequence[str], tokenizer: BaseTokenizer) -> None:
        cleaned = [seq for seq in stop_sequences if seq]
        if not cleaned:
            raise ValueError("stop_sequences must contain at least one non-empty string")
        self.stop_sequences = list(cleaned)
        self._encoded: List[Tuple[int, ...]] = [tuple(tokenizer.encode(seq)) for seq in self.stop_sequences]

    def is_met(self, tokens: Iterable[int] | torch.Tensor) -> Tuple[bool, str | None]:
        """Return True when the generated tokens end with a stop sequence."""
        sequence = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        for raw, encoded in zip(self.stop_sequences, self._encoded):
            if not encoded:
                continue
            if len(sequence) >= len(encoded) and sequence[-len(encoded) :] == list(encoded):
                return True, raw
        return False, None
