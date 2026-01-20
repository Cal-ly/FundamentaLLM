"""Positional encoding layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding.

    Learnable position embeddings, similar to token embeddings.
    Each position gets its own learned embedding vector.

    This approach is simple and allows the model to learn position
    representations specific to the task.

    Args:
        d_model: Feature dimension.
        max_seq_len: Maximum sequence length to encode. Default: 2048.
        dropout: Dropout probability. Default: 0.1.

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]

    Example:
        >>> pos_enc = LearnedPositionalEncoding(d_model=512)
        >>> x = torch.randn(2, 32, 512)
        >>> x_encoded = x + pos_enc(x)
        >>> x_encoded.shape
        torch.Size([2, 32, 512])
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize LearnedPositionalEncoding."""
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learnable embeddings for each position
        self.embeddings = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Positional encoding tensor of shape [batch, seq_len, d_model].
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # Generate position indices
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        # Get position embeddings
        pos_enc = self.embeddings(positions)  # [seq_len, d_model]

        # Expand to batch dimension
        pos_enc = pos_enc.unsqueeze(0)  # [1, seq_len, d_model]

        return self.dropout(pos_enc)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, non-learnable).

    Uses fixed sinusoidal functions to encode position information.
    Based on the original Transformer paper "Attention Is All You Need"
    (Vaswani et al., 2017).

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This approach is:
    - Non-learnable (always the same)
    - Extrapolatable to longer sequences
    - Provides relative position information

    Args:
        d_model: Feature dimension.
        max_seq_len: Maximum sequence length. Default: 2048.
        dropout: Dropout probability. Default: 0.1.

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]

    Example:
        >>> pos_enc = SinusoidalPositionalEncoding(d_model=512)
        >>> x = torch.randn(2, 32, 512)
        >>> x_encoded = x + pos_enc(x)
        >>> x_encoded.shape
        torch.Size([2, 32, 512])
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SinusoidalPositionalEncoding."""
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Register as buffer so it moves with the model but doesn't get trained
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices [0, 1, 2, ...]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Dimension indices in log space [0, 1, ..., d_model-1]
        # Formula: 10000^(2i/d_model)
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(dim_indices * -(math.log(10000.0) / d_model))

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].

        Returns:
            Positional encoding tensor of shape [batch, seq_len, d_model].
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # Get pre-computed positional encoding
        # [seq_len, d_model] -> [1, seq_len, d_model]
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)

        return self.dropout(pos_enc)


def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_seq_len: int = 2048,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function for positional encodings.

    Args:
        encoding_type: Type of encoding ("learned" or "sinusoidal").
        d_model: Feature dimension.
        max_seq_len: Maximum sequence length. Default: 2048.
        dropout: Dropout probability. Default: 0.1.

    Returns:
        Positional encoding module.

    Raises:
        ValueError: If encoding_type is not recognized.

    Example:
        >>> pos_enc = create_positional_encoding("sinusoidal", d_model=512)
        >>> pos_enc = create_positional_encoding("learned", d_model=512)
    """
    encoding_type = encoding_type.lower()

    if encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_seq_len, dropout)
    elif encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. " f"Choose from: 'learned', 'sinusoidal'"
        )
