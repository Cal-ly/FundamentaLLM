"""Dataset utilities for language modeling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Create (input, target) pairs for autoregressive language modeling."""

    def __init__(
        self, token_ids: torch.Tensor, sequence_length: int, stride: Optional[int] = None
    ) -> None:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        self.token_ids = token_ids.long()
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

    def __len__(self) -> int:  # pragma: no cover - trivial
        total_tokens = self.token_ids.size(0)
        if total_tokens <= self.sequence_length:
            return 0
        return max(0, (total_tokens - self.sequence_length - 1) // self.stride + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        start = idx * self.stride
        end = start + self.sequence_length + 1
        window = self.token_ids[start:end]
        input_ids = window[:-1]
        target_ids = window[1:]
        return input_ids, target_ids
