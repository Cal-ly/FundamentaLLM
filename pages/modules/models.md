# Models Module

Deep dive into the transformer model implementation in FundamentaLLM.

## Overview

The models module (`src/fundamentallm/models/`) implements the complete transformer architecture from scratch, designed for educational clarity.

```
models/
├── base.py           ← Base model interface
├── transformer.py    ← Main transformer model
└── components/
    ├── attention.py      ← Multi-head attention
    ├── feedforward.py    ← Position-wise FFN
    ├── embeddings.py     ← Token + positional embeddings
    └── normalization.py  ← Layer normalization
```

## Architecture Overview

```
Input IDs (batch, seq_len)
    ↓
TokenEmbedding + PositionalEncoding
    ↓
TransformerBlock × N
  ├─ MultiHeadAttention
  ├─ LayerNorm + Residual
  ├─ FeedForward
  └─ LayerNorm + Residual
    ↓
Output Layer (vocab_size)
    ↓
Logits (batch, seq_len, vocab_size)
```

## Core Components

### 1. Token Embedding

**File:** `components/embeddings.py`

**Purpose:** Convert token IDs to dense vectors.

```python
class TokenEmbedding(nn.Module):
    """
    Maps token IDs to continuous vectors.
    
    Args:
        vocab_size: Number of unique tokens
        d_model: Embedding dimension
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model)
        # Why: Keeps embedding magnitude consistent with positional encoding
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Why scaling?** Prevents positional encodings from dominating the embedding values.

**Example:**
```python
vocab_size = 256  # Character-level
d_model = 128
emb = TokenEmbedding(vocab_size, d_model)

# Input: [72, 101, 108, 108, 111]  ("Hello")
# Output: tensor of shape [5, 128]
```

### 2. Positional Encoding

**File:** `components/embeddings.py`

**Purpose:** Add position information to embeddings.

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]
```

**Why sinusoidal?**
- No parameters to learn
- Can extrapolate to longer sequences
- Different frequencies capture different scales

**Visualization:**
```
Position 0:  [0.00, 1.00, 0.00, 1.00, ...]
Position 1:  [0.84, 0.54, 0.01, 1.00, ...]
Position 2:  [0.91, -0.42, 0.02, 1.00, ...]
...

Each position has unique pattern
```

### 3. Multi-Head Attention

**File:** `components/attention.py`

**Purpose:** Core attention mechanism.

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # output: (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        return self.out_linear(output)
```

**Key points:**
- Splits d_model into num_heads × head_dim
- Each head learns different attention patterns
- Scaled by √head_dim to prevent gradient vanishing
- Causal mask prevents looking at future tokens

### 4. Feed-Forward Network

**File:** `components/feedforward.py`

**Purpose:** Position-wise transformation.

```python
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 × d_model)
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Smooth activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project back
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x
```

**Why GELU?** Smooth, differentiable, works better than ReLU for transformers.

**Why expand then contract?** Gives model more capacity to learn complex functions.

### 5. Layer Normalization

**File:** `components/normalization.py`

**Purpose:** Stabilize training by normalizing layer outputs.

```python
class LayerNorm(nn.Module):
    """
    Layer normalization.
    
    Normalizes across feature dimension (not batch dimension).
    
    Args:
        d_model: Feature dimension
        eps: Small constant for numerical stability
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift (learnable parameters)
        return self.gamma * x_norm + self.beta
```

**LayerNorm vs BatchNorm:**
- LayerNorm: Normalizes across features (works for sequences)
- BatchNorm: Normalizes across batch (doesn't work well for variable-length sequences)

## Transformer Block

**File:** `transformer.py`

Combines all components:

```python
class TransformerBlock(nn.Module):
    """
    Single transformer block: Self-Attention + FFN.
    
    Pre-norm architecture:
      x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward with residual
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return x
```

**Pre-norm vs Post-norm:**
- **Pre-norm** (used here): Normalize before sub-layer, more stable
- **Post-norm**: Normalize after sub-layer, original transformer design

## Full Transformer Model

**File:** `transformer.py`

```python
class Transformer(nn.Module):
    """
    Complete transformer language model.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Standard practice
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input token IDs (batch_size, seq_len)
            mask: Causal mask (optional)
            
        Returns:
            Logits (batch_size, seq_len, vocab_size)
        """
        # Embed and add positional encoding
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = self._generate_causal_mask(seq_len, x.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to vocabulary
        logits = self.output_layer(x)
        
        return logits
    
    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        Generate lower-triangular mask for causal attention.
        
        Returns mask where mask[i, j] = 1 if j <= i else 0
        """
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

## Model Configuration

**Typical configurations:**

### Small Model
```python
model = Transformer(
    vocab_size=256,
    d_model=128,
    num_layers=4,
    num_heads=4,
    dropout=0.1
)
# ~500K parameters
```

### Medium Model
```python
model = Transformer(
    vocab_size=256,
    d_model=256,
    num_layers=6,
    num_heads=8,
    dropout=0.1
)
# ~2M parameters
```

### Large Model
```python
model = Transformer(
    vocab_size=256,
    d_model=512,
    num_layers=12,
    num_heads=16,
    dropout=0.2
)
# ~15M parameters
```

## Design Decisions

### Why Pre-Norm?
More stable training, especially for deep networks. Gradients flow better through residual connections.

### Why GELU?
Smoother than ReLU, works better for transformers. Used in GPT, BERT, etc.

### Why Character-Level?
Educational clarity - no complex tokenizer needed. Every text can be processed.

### Why Sinusoidal Positional Encoding?
No parameters to learn, generalizes to any length, interpretable frequency bands.

## Usage Example

```python
from fundamentallm.models import Transformer

# Create model
model = Transformer(
    vocab_size=256,
    d_model=256,
    num_layers=6,
    num_heads=8
)

# Input: batch of token IDs
input_ids = torch.randint(0, 256, (2, 100))  # (batch=2, seq_len=100)

# Forward pass
logits = model(input_ids)  # (2, 100, 256)

# Get predictions for next token
next_token_logits = logits[:, -1, :]  # (2, 256)
next_token = torch.argmax(next_token_logits, dim=-1)  # (2,)
```

## Performance Considerations

### Memory Usage
- Attention: O(n²d) where n=seq_len, d=d_model
- Parameters: O(Ld²) where L=num_layers

### Computation
- Forward pass: O(n²d + nd²) per layer
- Most expensive: Attention (quadratic in sequence length)

### Optimization Tips
- Use mixed precision (FP16)
- Gradient checkpointing for very deep models
- Optimize batch size for GPU utilization

## Further Reading

- [Transformer Architecture](../concepts/transformers.md) - Theory
- [Attention Mechanism](../concepts/attention.md) - Deep dive
- [Training Guide](../guide/training.md) - How to train
- Original paper: "Attention is All You Need" (Vaswani et al., 2017)
