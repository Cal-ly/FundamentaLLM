"""Multi-head attention mechanism."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.
    
    Implements scaled dot-product attention with multiple heads in parallel.
    This allows the model to attend to different representation subspaces
    at different positions.
    
    Formula (single head):
        Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
    
    Multi-head:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)·W^O
        where head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
    
    Args:
        d_model: Feature dimension (must be divisible by num_heads).
        num_heads: Number of attention heads. Default: 8.
        dropout: Dropout probability for attention weights. Default: 0.1.
        causal: If True, apply causal mask (only attend to past/present).
                Default: True (for language modeling).
    
    Shape:
        - Query: [batch, seq_len, d_model]
        - Key: [batch, seq_len, d_model]
        - Value: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]
    
    Example:
        >>> attention = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 32, 512)
        >>> output = attention(x, x, x)  # Self-attention
        >>> output.shape
        torch.Size([2, 32, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        """Initialize MultiHeadAttention."""
        super().__init__()
        
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        
        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.
        
        Args:
            query: Query tensor of shape [batch, seq_len_q, d_model].
            key: Key tensor of shape [batch, seq_len_k, d_model].
            value: Value tensor of shape [batch, seq_len_v, d_model].
            attention_mask: Optional mask of shape [batch, seq_len_q, seq_len_k]
                          or [batch, 1, seq_len_q, seq_len_k].
                          Mask value 0 means "don't attend", 1 means "attend".
        
        Returns:
            Output tensor of shape [batch, seq_len_q, d_model].
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_v, d_model]
        
        # Split into multiple heads
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        # -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q·K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch, num_heads, seq_len_q, seq_len_k]
        
        # Apply causal mask if requested
        if self.causal:
            causal_mask = self._create_causal_mask(seq_len_q, seq_len_k, device=scores.device)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        
        # Apply provided attention mask
        if attention_mask is not None:
            # Reshape mask to [batch, 1, seq_len_q, seq_len_k] if needed
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # [batch, num_heads, seq_len_q, d_k]
        
        # Concatenate heads
        # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, num_heads, d_k]
        # -> [batch, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output
    
    @staticmethod
    def _create_causal_mask(
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask.
        
        Prevents attention to future positions. Lower triangular matrix of ones.
        
        Args:
            seq_len_q: Query sequence length.
            seq_len_k: Key sequence length.
            device: Device for the mask.
        
        Returns:
            Causal mask of shape [seq_len_q, seq_len_k] with 1s in lower triangle
            and 0s in upper triangle.
        """
        # Create lower triangular matrix
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=device))
        return mask
