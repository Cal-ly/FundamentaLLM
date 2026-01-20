"""Transformer model: complete decoder-only architecture."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from fundamentallm.config import TransformerConfig
from fundamentallm.models.base import BaseModel
from fundamentallm.models.components import (
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    RMSNorm,
    create_positional_encoding,
)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization.

    Architecture (Pre-Norm variant):
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> [Add residual] ->
        x -> LayerNorm -> FFN -> Dropout -> [Add residual]

    Pre-normalization is more stable than post-normalization and allows
    for better gradient flow during training.

    Args:
        d_model: Feature dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension. Default: 4 * d_model.
        dropout: Dropout probability. Default: 0.1.
        norm_type: Normalization type ("rmsnorm" or "layernorm"). Default: "rmsnorm".
        causal: Whether to use causal attention mask. Default: True.

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]

    Example:
        >>> block = TransformerBlock(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 32, 512)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 32, 512])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        causal: bool = True,
    ) -> None:
        """Initialize TransformerBlock."""
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Pre-normalization (before attention)
        self.norm1: RMSNorm | LayerNorm
        self.norm2: RMSNorm | LayerNorm
        if norm_type.lower() == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_type.lower() == "layernorm":
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type}. " f"Choose from: 'rmsnorm', 'layernorm'"
            )

        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        )

        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation="gelu",
        )

        # Dropout for residual paths
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        # Pre-norm attention with residual
        normed = self.norm1(x)
        attn_output = self.attention(normed, normed, normed, attention_mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output

        # Pre-norm FFN with residual
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output

        return x


class Transformer(BaseModel):
    """Decoder-only transformer language model.

    A complete transformer architecture for language modeling with:
    - Token embeddings
    - Positional encodings
    - Stack of transformer blocks
    - Output projection (with weight tying)

    This is the core model used for next-token prediction.

    Architecture:
        Input tokens -> Embeddings + PositionalEncoding ->
        [TransformerBlocks] -> LayerNorm -> Output Projection -> Logits

    Args:
        config: TransformerConfig with all hyperparameters.

    Shape:
        - Input: [batch, seq_len] (token indices)
        - Output: [batch, seq_len, vocab_size] (logits)

    Example:
        >>> from fundamentallm.config import TransformerConfig
        >>> config = TransformerConfig(
        ...     vocab_size=10000,
        ...     d_model=512,
        ...     num_layers=6,
        ...     num_heads=8,
        ... )
        >>> model = Transformer(config)
        >>> input_ids = torch.randint(0, 10000, (2, 32))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 32, 10000])
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize Transformer."""
        super().__init__()  # Call nn.Module.__init__ instead of BaseModel.__init__(config)

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_layers = config.num_layers

        # Token embedding (learned)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.positional_encoding = create_positional_encoding(
            encoding_type=config.pos_encoding,
            d_model=config.d_model,
            max_seq_len=config.sequence_length,
            dropout=config.dropout,
        )

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # Feed-forward dimension
        d_ff = config.d_model * config.ffn_expansion

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=d_ff,
                    dropout=config.dropout,
                    norm_type="rmsnorm",  # Use RMSNorm by default
                    causal=True,  # Always use causal for language modeling
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final normalization
        self.output_norm = RMSNorm(config.d_model)

        # Output projection (projects d_model -> vocab_size)
        # Weight tying: share with token embedding for efficiency
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            input_ids: Token indices of shape [batch, seq_len].
            attention_mask: Optional attention mask.

        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Add positional encoding
        pos_enc = self.positional_encoding(x)
        x = x + pos_enc
        x = self.embed_dropout(x)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, x.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final normalization
        x = self.output_norm(x)

        # Output projection -> logits
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        return logits

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        torch.save(self.state_dict(), str(path))

    @classmethod
    def load(cls, path: Path) -> "Transformer":
        """Load model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded Transformer instance.
        """
        state_dict = torch.load(str(path), map_location="cpu")
        # Create model instance with proper config
        # This is a simplified loader - in practice, config should be saved with model
        model = cls(
            config=TransformerConfig(
                vocab_size=state_dict["token_embedding.weight"].shape[0],
                sequence_length=2048,  # Default - should be saved with model
                d_model=state_dict["token_embedding.weight"].shape[1],
                num_heads=8,  # Default - should be saved in config
                num_layers=len(
                    [k for k in state_dict.keys() if "layers." in k and ".attn.qkv" in k]
                ),
                dropout=0.1,
                ffn_expansion=4,
                pos_encoding="sinusoidal",
            ),
        )
        model.load_state_dict(state_dict)
        return model

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights (GPT-2 style).

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _create_causal_mask(
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask.

        Prevents attention to future positions. Lower triangular matrix of ones.

        Args:
            seq_len: Sequence length.
            device: Device for the mask.

        Returns:
            Causal mask of shape [1, 1, seq_len, seq_len].
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
