"""Normalization layers: LayerNorm, RMSNorm."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization.

    Normalizes each sample independently across the feature dimension.
    Reference implementation of layer normalization from "Layer Normalization"
    (Ba et al., 2016).

    Args:
        d_model: Feature dimension.
        eps: Small constant for numerical stability. Default: 1e-6.

    Shape:
        - Input: [*, d_model]
        - Output: [*, d_model]

    Example:
        >>> norm = LayerNorm(512)
        >>> x = torch.randn(2, 32, 512)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 32, 512])
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize LayerNorm."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scale and shift (gamma and beta)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape [*, d_model].

        Returns:
            Normalized tensor of same shape.
        """
        # Compute mean and variance over the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.weight * x_normalized + self.bias


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Modern alternative to LayerNorm with better computational efficiency
    and training stability. Used in models like PaLM, LLaMA, and GPT-3.

    From "Root Mean Square Layer Normalization" (Zhang & Savarese, 2019).

    Formula:
        RMSNorm(x) = x / RMS(x) * γ
        where RMS(x) = √(1/d * Σ x_i²)

    Args:
        d_model: Feature dimension.
        eps: Small constant for numerical stability. Default: 1e-6.

    Shape:
        - Input: [*, d_model]
        - Output: [*, d_model]

    Example:
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(2, 32, 512)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([2, 32, 512])

    Note:
        RMSNorm has only one learnable parameter (weight/gamma), making it
        more efficient than LayerNorm which has weight and bias. No bias
        parameter by design.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scale (gamma only, no bias)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape [*, d_model].

        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS (root mean square) over the last dimension
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return (x / rms) * self.weight
