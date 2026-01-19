"""Tests for sampling strategies."""

from __future__ import annotations

import torch
import pytest

from fundamentallm.generation.sampling import (
    GreedySampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
    create_sampler,
)


def test_greedy_sampler_returns_argmax():
    logits = torch.tensor([[-1.0, 0.5, 2.0]])
    sampler = GreedySampler()
    token = sampler.sample(logits)
    assert token.item() == 2


def test_temperature_sampler_respects_temperature():
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.0]])

    cold_sampler = TemperatureSampler(temperature=0.1)
    cold_samples = torch.stack([cold_sampler.sample(logits) for _ in range(10)])
    assert torch.all(cold_samples == cold_samples[0])  # distribution collapses to argmax

    torch.manual_seed(1)
    hot_sampler = TemperatureSampler(temperature=5.0)
    hot_samples = torch.stack([hot_sampler.sample(logits) for _ in range(30)])
    assert torch.any(hot_samples != hot_samples[0])  # higher entropy distribution


def test_top_k_sampler_masks_non_top_tokens():
    torch.manual_seed(42)
    logits = torch.tensor([[0.1, 2.0, 0.5, 3.0]])
    sampler = TopKSampler(k=2)

    samples = torch.stack([sampler.sample(logits) for _ in range(20)])
    allowed = {1, 3}
    assert all(int(tok) in allowed for tok in samples)


def test_top_p_sampler_respects_cumulative_probability():
    torch.manual_seed(7)
    logits = torch.tensor([[5.0, 2.0, 1.0, 0.0]])
    sampler = TopPSampler(p=0.99)

    samples = torch.stack([sampler.sample(logits) for _ in range(40)])
    allowed = {0, 1}  # top two tokens keep cumulative prob under 0.99
    assert all(int(tok) in allowed for tok in samples)


def test_sampler_factory():
    assert isinstance(create_sampler("greedy"), GreedySampler)
    assert isinstance(create_sampler("temperature", temperature=0.5), TemperatureSampler)
    assert isinstance(create_sampler("top_k", k=5), TopKSampler)
    assert isinstance(create_sampler("top_p", p=0.8), TopPSampler)

    with pytest.raises(ValueError):
        create_sampler("unknown")
