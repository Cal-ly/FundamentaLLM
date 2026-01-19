"""Loss functions for language model training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelingLoss(nn.Module):
    """Cross-entropy loss tailored for autoregressive language modeling.

    Supports label smoothing, masking of padding/ignored targets, and per-sample
    reduction. The expected shapes are logits of ``[batch, seq_len, vocab]`` and
    integer targets of ``[batch, seq_len]``.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if label_smoothing < 0 or label_smoothing >= 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if reduction not in {"none", "mean", "sum", "batch", "batchmean"}:
            raise ValueError("reduction must be one of none, mean, sum, batch")

        self.label_smoothing = float(label_smoothing)
        self.ignore_index = ignore_index
        self.reduction = "batch" if reduction == "batchmean" else reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the language modeling loss.

        Args:
            logits: Tensor of shape ``[batch, seq_len, vocab]``.
            targets: Tensor of shape ``[batch, seq_len]``.
            reduction: Optional override for reduction (``none``, ``mean``,
                ``sum``, ``batch``/``batchmean``).

        Returns:
            Loss tensor reduced according to ``reduction``.
        """
        if logits.dim() != 3 or targets.dim() != 2:
            raise ValueError("logits must be [batch, seq_len, vocab] and targets [batch, seq_len]")

        batch, seq_len, vocab_size = logits.shape
        reduction = reduction or self.reduction

        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        ignore_mask = (targets_flat == self.ignore_index) | (targets_flat == -1)
        safe_targets = targets_flat.clone()
        safe_targets[ignore_mask] = 0  # avoid invalid indices during loss computation

        losses = F.cross_entropy(
            logits_flat,
            safe_targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # Zero-out ignored positions
        mask = (~ignore_mask).to(losses.dtype)
        losses = losses * mask

        if reduction == "none":
            return losses.view(batch, seq_len)

        if reduction == "sum":
            return losses.sum()

        if reduction == "mean":
            denom = mask.sum().clamp_min(1.0)
            return losses.sum() / denom

        if reduction == "batch":
            losses = losses.view(batch, seq_len)
            mask_2d = mask.view(batch, seq_len)
            denom = mask_2d.sum(dim=1).clamp_min(1.0)
            return (losses * mask_2d).sum(dim=1) / denom

        raise ValueError(f"Unsupported reduction: {reduction}")


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Utility wrapper for quick loss computation."""
    loss_fn = LanguageModelingLoss(
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    return loss_fn(logits, targets)
