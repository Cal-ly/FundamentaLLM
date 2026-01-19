"""Sampling strategies for text generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


def _prepare_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Normalize logits shape to [batch, vocab]."""
    if logits.dim() not in (1, 2):
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")
    squeezed = logits.dim() == 1
    return (logits.unsqueeze(0) if squeezed else logits), squeezed


def _restore_shape(tokens: torch.Tensor, squeezed: bool) -> torch.Tensor:
    """Return tokens to original dimensionality."""
    return tokens.squeeze(0) if squeezed else tokens


class Sampler(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token IDs from logits."""


class GreedySampler(Sampler):
    """Greedy decoding - always pick the most likely token."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits_2d, squeezed = _prepare_logits(logits)
        tokens = torch.argmax(logits_2d, dim=-1)
        return _restore_shape(tokens, squeezed)


class TemperatureSampler(Sampler):
    """Temperature-scaled sampling."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits_2d, squeezed = _prepare_logits(logits)
        scaled_logits = logits_2d / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return _restore_shape(sampled, squeezed)


class TopKSampler(Sampler):
    """Top-k sampling - restrict choices to the k highest logits."""

    def __init__(self, k: int = 50, temperature: float = 1.0) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.k = int(k)
        self.temperature = float(temperature)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits_2d, squeezed = _prepare_logits(logits)
        vocab_size = logits_2d.size(-1)
        k = min(self.k, vocab_size)

        top_k_vals, top_k_idx = torch.topk(logits_2d, k, dim=-1)
        filtered = torch.full_like(logits_2d, float("-inf"))
        filtered.scatter_(-1, top_k_idx, top_k_vals)

        scaled_logits = filtered / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return _restore_shape(sampled, squeezed)


class TopPSampler(Sampler):
    """Top-p (nucleus) sampling - keep tokens until cumulative prob exceeds p."""

    def __init__(self, p: float = 0.95, temperature: float = 1.0) -> None:
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.p = float(p)
        self.temperature = float(temperature)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits_2d, squeezed = _prepare_logits(logits)
        scaled_logits = logits_2d / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumsum > self.p
        cutoff_mask[..., 0] = False  # always keep most probable token
        sorted_probs = torch.where(cutoff_mask, torch.zeros_like(sorted_probs), sorted_probs)

        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        sampled = torch.multinomial(filtered, num_samples=1).squeeze(-1)
        return _restore_shape(sampled, squeezed)


def create_sampler(strategy: str, **kwargs) -> Sampler:
    """Factory for creating samplers by name."""
    name = strategy.lower()
    if name == "greedy":
        return GreedySampler()
    if name == "temperature":
        return TemperatureSampler(kwargs.get("temperature", 1.0))
    if name == "top_k":
        return TopKSampler(
            k=kwargs.get("k", 50),
            temperature=kwargs.get("temperature", 1.0),
        )
    if name == "top_p":
        return TopPSampler(
            p=kwargs.get("p", 0.95),
            temperature=kwargs.get("temperature", 1.0),
        )
    raise ValueError(f"Unknown sampling strategy: {strategy}")
