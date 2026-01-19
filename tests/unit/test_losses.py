"""Tests for language modeling loss."""

import torch

from fundamentallm.training.losses import LanguageModelingLoss, compute_loss


def test_loss_masks_padding():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, -100], [3, -1, 4]])

    loss_fn = LanguageModelingLoss()
    loss = loss_fn(logits, targets, reduction="mean")

    # Masked positions should not contribute
    losses = loss_fn(logits, targets, reduction="none")
    assert losses.shape == (2, 3)
    assert torch.allclose(losses[0, -1], torch.tensor(0.0))
    assert torch.allclose(losses[1, 1], torch.tensor(0.0))
    assert loss.item() > 0


def test_label_smoothing_reduces_confidence():
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 4)
    targets = torch.tensor([[0, 1]])

    base = compute_loss(logits, targets, reduction="mean", label_smoothing=0.0)
    smoothed = compute_loss(logits, targets, reduction="mean", label_smoothing=0.2)

    assert smoothed.item() != base.item()
    # Smoothed loss should be bounded and finite
    assert torch.isfinite(smoothed)


def test_per_sample_reduction():
    torch.manual_seed(1)
    logits = torch.randn(2, 2, 3, requires_grad=True)
    targets = torch.tensor([[1, 2], [0, 1]])

    loss_fn = LanguageModelingLoss(reduction="batch")
    per_sample = loss_fn(logits, targets)
    assert per_sample.shape == (2,)


def test_gradient_flow():
    logits = torch.randn(2, 2, 4, requires_grad=True)
    targets = torch.tensor([[0, 1], [2, 3]])

    loss = LanguageModelingLoss()(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
