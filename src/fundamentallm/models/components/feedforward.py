"""Feed-forward network component."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network.

    Two linear transformations with an activation function in between.
    Applied to each position separately and identically.

    Architecture:
        Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    Formula:
        FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
        (Using GELU instead of ReLU for smoothness)

    Args:
        d_model: Feature dimension (input and output).
        d_ff: Hidden dimension (intermediate). Default: 4 * d_model.
        dropout: Dropout probability. Default: 0.1.
        activation: Activation function ("gelu" or "relu"). Default: "gelu".

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]

    Example:
        >>> ffn = FeedForwardNetwork(d_model=512)
        >>> x = torch.randn(2, 32, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 32, 512])

    Note:
        The default d_ff is 4*d_model as in the original Transformer paper.
        Some models use d_ff = 8*d_model or other multiples for more capacity.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize FeedForwardNetwork."""
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        # First linear transformation: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Activation function
        self.activation: nn.GELU | nn.ReLU
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}. " f"Choose from: 'gelu', 'relu'")

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Second linear transformation: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward network.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        # [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        x = self.linear1(x)

        # Activation
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        # [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        x = self.linear2(x)

        return x
