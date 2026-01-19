"""Abstract base classes for models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all FundamentaLLM models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover - interface only
        """Run the forward pass."""

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def save(self, path: Path) -> None:  # pragma: no cover - interface only
        """Persist model weights to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":  # pragma: no cover - interface only
        """Restore model from disk."""
