# Phase 3: Model Architecture

**Objective:** Implement transformer components and complete transformer model with attention, normalization, embeddings, and proper initialization.

**Status:** Planning

**Dependencies:** Phase 1 (Core Infrastructure) ✅, Phase 2 (Data Pipeline) ✅

**Estimated Timeline:** 3-4 days

---

## Overview

Phase 3 builds the neural network core of FundamentaLLM. This includes:
- Multi-head attention mechanism with causal masking
- Normalization layers (RMSNorm, LayerNorm)
- Positional encoding variants (learned, sinusoidal, RoPE)
- Feed-forward networks
- Complete transformer blocks with pre-normalization
- Full transformer model with weight tying
- Model registry for extensibility
- Proper weight initialization (GPT-2 style)

This is the most mathematically complex phase.

---

## Architecture Overview

```
Input Tokens
    ↓
Token Embeddings + Positional Encoding
    ↓
[Transformer Block × N layers]
    ├── Multi-Head Attention (with causal mask)
    ├── Residual Connection + Layer Norm
    ├── Feed-Forward Network
    └── Residual Connection + Layer Norm
    ↓
Output Layer Norm
    ↓
Linear Projection → Logits
    ↓
Vocab Distribution
```

---

## Files to Create

### Core Components

```
src/fundamentallm/models/
├── __init__.py                     # Model exports
├── base.py                         # BaseModel (from Phase 1, refine)
├── transformer.py                  # Complete Transformer implementation
├── components/
│   ├── __init__.py
│   ├── attention.py                # MultiHeadAttention
│   ├── embeddings.py               # PositionalEncoding variants
│   ├── feedforward.py              # FeedForward network
│   └── normalization.py            # RMSNorm, LayerNorm
└── registry.py                     # Model registry pattern
```

### Testing

```
tests/
├── unit/
│   ├── test_attention.py           # Attention mechanism tests
│   ├── test_embeddings.py          # Embedding tests
│   ├── test_normalization.py       # Normalization tests
│   ├── test_models.py              # Full model tests
│   └── test_components.py          # Component integration
└── integration/
    └── test_model_forward_pass.py  # End-to-end forward pass
```

---

## Detailed Tasks

### Task 3.1: Normalization Layers

**Objective:** Implement RMSNorm and LayerNorm for stable training

**File:** `src/fundamentallm/models/components/normalization.py`

**Class 1: LayerNorm**

Purpose: Standard layer normalization

Parameters:
- `dim: int` - Feature dimension
- `eps: float = 1e-6` - Small constant for numerical stability
- `bias: bool = True` - Include learnable bias

Implementation:
```python
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over last dimension
        # Formula: (x - mean) / sqrt(var + eps) * weight + bias
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + (self.bias if self.bias is not None else 0)
```

**Class 2: RMSNorm**

Purpose: Root Mean Square normalization (better performance than LayerNorm)

Key Difference:
- Simpler than LayerNorm (no mean subtraction)
- Uses RMS instead of variance
- Empirically better for transformers

Implementation:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: x / (RMS + eps) * weight
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**Design Decisions:**
- RMSNorm as default (modern practice)
- LayerNorm as alternative
- Both should produce similar shapes
- eps=1e-6 for stability (smaller than LayerNorm's 1e-5)

**Success Criteria:**
- ✅ Both classes inherit from nn.Module
- ✅ Forward pass preserves input shape
- ✅ Learnable parameters correct
- ✅ Numerical stability with small values
- ✅ Can be used in nn.Sequential

---

### Task 3.2: Attention Mechanism

**Objective:** Implement scaled dot-product attention with multi-head support

**File:** `src/fundamentallm/models/components/attention.py`

**Class: MultiHeadAttention**

Parameters:
```python
def __init__(
    self,
    config: TransformerConfig,
    # OR
    d_model: int,
    num_heads: int,
    dropout: float = 0.1
)
```

**Key Components:**

1. **Linear projections for Q, K, V:**
   - `W_q: Linear(d_model, d_model)`
   - `W_k: Linear(d_model, d_model)`
   - `W_v: Linear(d_model, d_model)`
   - Output projection: `W_o: Linear(d_model, d_model)`

2. **Scaled Dot-Product Attention:**
   ```
   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
   where d_k = d_model / num_heads
   ```

3. **Multi-head mechanism:**
   - Split attention into num_heads parallel heads
   - Each head has dimension d_k = d_model / num_heads
   - Concatenate and project output

**Implementation Pseudo-code:**

```python
def forward(
    self,
    x: torch.Tensor,                    # [batch, seq_len, d_model]
    mask: Optional[torch.Tensor] = None # [batch, seq_len, seq_len] or [1, 1, seq_len, seq_len]
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    
    # Project inputs
    Q = self.W_q(x)  # [batch, seq_len, d_model]
    K = self.W_k(x)  # [batch, seq_len, d_model]
    V = self.W_v(x)  # [batch, seq_len, d_model]
    
    # Reshape for multi-head
    Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    # [batch, num_heads, seq_len, d_k]
    
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / sqrt(self.d_k)
    # [batch, num_heads, seq_len, seq_len]
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -inf)
    
    # Softmax
    attn_weights = softmax(scores, dim=-1)
    # [batch, num_heads, seq_len, seq_len]
    
    # Apply attention to values
    attn_output = attn_weights @ V
    # [batch, num_heads, seq_len, d_k]
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_model)
    
    # Output projection
    output = self.W_o(attn_output)
    return output
```

**Causal Masking:**

For language modeling, we need causal mask (can't attend to future):
```
Mask pattern for seq_len=4:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]

Where 1 = attend, 0 = don't attend
```

**Implementation of causal mask:**
```python
@staticmethod
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create lower triangular attention mask for causal attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

**Design Notes:**
- Use `@` operator for matrix multiplication (clear)
- Use `contiguous()` after transpose before view
- Dropout on attention weights (not outputs)
- Support both batch 4D mask and single 2D mask

**Success Criteria:**
- ✅ Output shape: [batch, seq_len, d_model]
- ✅ Causal mask prevents future attention
- ✅ Multiple heads work correctly
- ✅ Numerically stable (no NaNs)
- ✅ Gradients flow properly

---

### Task 3.3: Positional Encodings

**Objective:** Implement position encoding variants

**File:** `src/fundamentallm/models/components/embeddings.py`

**Class 1: Learned Positional Encoding**

Simplest approach: learnable embeddings per position

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.embedding(pos_ids)  # [1, seq_len, d_model]
        return x + pos_emb
```

**Class 2: Sinusoidal Positional Encoding**

Fixed, non-learnable positional encoding (original Transformer)

Formula:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        
        # Compute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * 
            -(math.log(10000.0) / d_model)
        )  # [d_model/2]
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
```

**Class 3: RoPE (Rotary Position Embedding)**

Modern approach: rotate embedding vectors by position

Note: RoPE is more complex, can be implemented in Phase 3.1 extensions or Phase 5.
For MVP, focus on learned and sinusoidal.

**Embedding Factory Function:**

```python
def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_seq_len: int
) -> nn.Module:
    """Create positional encoding from config."""
    if encoding_type == "learned":
        return LearnedPositionalEncoding(max_seq_len, d_model)
    elif encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_seq_len)
    else:
        raise ValueError(f"Unknown encoding: {encoding_type}")
```

**Design Notes:**
- Learned: Fast, flexible, learnable
- Sinusoidal: No parameters, works at inference for longer sequences
- Both add to embeddings (additive)
- Must preserve input shape

**Success Criteria:**
- ✅ Both implementations preserve shape
- ✅ Sinusoidal non-learnable
- ✅ Learned has trainable parameters
- ✅ Can handle variable sequence lengths

---

### Task 3.4: Feed-Forward Network

**Objective:** Implement position-wise feed-forward network

**File:** `src/fundamentallm/models/components/feedforward.py`

**Class: FeedForwardNetwork**

Purpose: Non-linear transformation in each transformer layer

Architecture:
```
Input [batch, seq_len, d_model]
  ↓
Linear (d_model → d_ff)  where d_ff = d_model * ffn_expansion
  ↓
Activation (GELU)
  ↓
Dropout
  ↓
Linear (d_ff → d_model)
  ↓
Output [batch, seq_len, d_model]
```

Implementation:
```python
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

**Design Decisions:**
- GELU activation (modern, slightly non-linear)
- Dropout after activation (common pattern)
- d_ff = d_model * 4 (typical configuration, parameterizable)
- Applied position-wise (same across sequence positions)

**Success Criteria:**
- ✅ Output shape matches input shape
- ✅ Expansion dimension correct
- ✅ Activation works as expected
- ✅ Dropout applied correctly

---

### Task 3.5: Transformer Block

**Objective:** Combine attention and feed-forward with residual connections and normalization

**File:** `src/fundamentallm/models/transformer.py` (or separate file)

**Class: TransformerBlock**

Purpose: Single transformer layer with pre-normalization

Architecture (Pre-Norm variant):
```
Input x
  ↓
LayerNorm
  ↓
MultiHeadAttention + Residual
  ↓
LayerNorm
  ↓
FeedForward + Residual
  ↓
Output
```

Implementation:
```python
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForwardNetwork(
            d_model=config.d_model,
            d_ff=config.d_model * config.ffn_expansion,
            dropout=config.dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attention(self.norm1(x), mask)
        
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.norm2(x))
        
        return x
```

**Key Design: Pre-Normalization**

Why pre-norm?
- ✅ Better gradient flow
- ✅ More stable training
- ✅ No need for output layer norm in block

Alternative (Post-Norm):
```
LN(Attn(x) + x)  # Post-norm (older)
x + Attn(LN(x))  # Pre-norm (modern, better)
```

**Success Criteria:**
- ✅ Output shape: [batch, seq_len, d_model]
- ✅ Residual connections work
- ✅ Pre-norm applied correctly
- ✅ Mask propagates to attention

---

### Task 3.6: Complete Transformer Model

**Objective:** Assemble full transformer model for language modeling

**File:** `src/fundamentallm/models/transformer.py`

**Class: Transformer**

Purpose: Complete language model

Architecture:
```
Input Tokens [batch, seq_len]
  ↓
Token Embeddings
  ↓
Positional Encoding
  ↓
LayerNorm (RMSNorm)
  ↓
[Transformer Block × num_layers]
  ↓
Output LayerNorm
  ↓
Linear Projection → Logits [batch, seq_len, vocab_size]
```

Implementation outline:
```python
@ModelRegistry.register("transformer")
class Transformer(BaseModel):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = create_positional_encoding(
            config.pos_encoding,
            config.d_model,
            config.sequence_length
        )
        
        # Input normalization and dropout
        self.embed_norm = RMSNorm(config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.output_norm = RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share embeddings with output projection
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len, seq_len] or None
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

**Key Features:**

1. **Weight Tying**: Output embedding matrix = input embedding matrix
   - Reduces parameters ~33%
   - Improves training efficiency
   - Common in language models

2. **Causal Masking**: Only attend to current and past positions
   - Essential for autoregressive generation
   - Prevents information leakage

3. **GPT-2 Initialization**: 
   - Small stddev (0.02) for stable initialization
   - Helps with training large models

**Success Criteria:**
- ✅ Output shape: [batch, seq_len, vocab_size]
- ✅ Can instantiate from TransformerConfig
- ✅ Forward pass works without errors
- ✅ Gradients flow to all parameters
- ✅ Parameter count reasonable

---

### Task 3.7: Model Registry

**Objective:** Implement extensible model factory pattern

**File:** `src/fundamentallm/models/registry.py`

**Class: ModelRegistry**

Purpose: Register and instantiate models dynamically

```python
class ModelRegistry:
    """Registry for model architectures."""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register model class."""
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: TransformerConfig) -> BaseModel:
        """Create model from registry."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' not registered. Available: {available}"
            )
        return cls._registry[name](config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if model is registered."""
        return name in cls._registry
```

**Usage:**

```python
# In transformer.py
@ModelRegistry.register("transformer")
class Transformer(BaseModel):
    ...

# To create model
model = ModelRegistry.create("transformer", config)

# To list available models
models = ModelRegistry.list_models()
```

**Design Benefits:**
- ✅ Easy to add new models
- ✅ No hardcoded imports needed
- ✅ Extensible for future variants
- ✅ CLI can support multiple models

**Success Criteria:**
- ✅ Can register models with decorator
- ✅ Can create models from registry
- ✅ Can list registered models
- ✅ Raises clear errors for unknown models

---

### Task 3.8: Component Tests

**Objective:** Unit tests for all components

**File:** `tests/unit/test_attention.py`

**Tests for MultiHeadAttention:**
- ✅ Output shape correct
- ✅ Causal mask works
- ✅ Multiple heads work
- ✅ Gradients flow
- ✅ Self-attention mechanism

**File:** `tests/unit/test_embeddings.py`

**Tests for positional encodings:**
- ✅ LearnedPositionalEncoding trainable
- ✅ SinusoidalPositionalEncoding non-trainable
- ✅ Both preserve shape
- ✅ Sinusoidal values correct
- ✅ Handle different sequence lengths

**File:** `tests/unit/test_normalization.py`

**Tests for normalization:**
- ✅ RMSNorm output shape
- ✅ LayerNorm output shape
- ✅ Numerical stability
- ✅ Learnable weights

**File:** `tests/unit/test_models.py`

**Tests for complete model:**
- ✅ Model instantiation
- ✅ Forward pass shape
- ✅ Parameter count
- ✅ Device placement
- ✅ Causal mask effect
- ✅ Gradient flow
- ✅ Generate logits in vocab range

**File:** `tests/integration/test_model_forward_pass.py`

**Integration tests:**
- ✅ End-to-end forward pass
- ✅ Backward pass works
- ✅ Loss computation
- ✅ Different batch sizes
- ✅ Different sequence lengths

**Success Criteria:**
- ✅ All tests pass
- ✅ Coverage > 85%
- ✅ Edge cases handled

---

### Task 3.9: Documentation and Examples

**Objective:** Document architecture and provide usage examples

**Docstring Examples:**

In each class/function:
```python
def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
    """Apply attention.
    
    Args:
        x: Input tensor of shape [batch, seq_len, d_model]
        mask: Optional attention mask of shape [batch, seq_len, seq_len]
    
    Returns:
        Output tensor of shape [batch, seq_len, d_model]
    
    Example:
        >>> attention = MultiHeadAttention(config)
        >>> x = torch.randn(2, 32, 512)  # batch=2, seq_len=32, d_model=512
        >>> output = attention(x)
        >>> output.shape
        torch.Size([2, 32, 512])
    """
```

**Usage Notes:**

Add comment-based documentation in transformer.py:
```python
# FundamentaLLM Transformer
#
# A standard decoder-only transformer with:
# - Pre-normalization (better training stability)
# - RMSNorm for efficiency
# - Learned or sinusoidal positional encoding
# - Weight tying (shared embeddings)
# - Causal attention mask
# - GPT-2 style initialization
#
# For educational purposes, emphasis on clarity over optimization.
```

**Success Criteria:**
- ✅ All classes have docstrings
- ✅ All functions have docstrings
- ✅ Type hints complete
- ✅ Examples provided

---

## Implementation Checklist

- [ ] Create normalization layers (Task 3.1)
  - [ ] LayerNorm implementation
  - [ ] RMSNorm implementation
- [ ] Create attention mechanism (Task 3.2)
  - [ ] MultiHeadAttention implementation
  - [ ] Causal mask creation
  - [ ] Scaled dot-product attention
- [ ] Create positional encodings (Task 3.3)
  - [ ] LearnedPositionalEncoding
  - [ ] SinusoidalPositionalEncoding
  - [ ] Encoding factory function
- [ ] Create feed-forward network (Task 3.4)
- [ ] Create transformer block (Task 3.5)
- [ ] Create complete transformer (Task 3.6)
  - [ ] Embedding layer
  - [ ] Blocks stack
  - [ ] Output projection
  - [ ] Weight tying
  - [ ] Weight initialization
  - [ ] Causal mask handling
- [ ] Create model registry (Task 3.7)
- [ ] Create component tests (Task 3.8)
  - [ ] Attention tests
  - [ ] Embedding tests
  - [ ] Normalization tests
  - [ ] Model tests
  - [ ] Integration tests
- [ ] Add documentation (Task 3.9)

---

## Success Criteria for Phase 3

1. **Component Implementation**
   - ✅ All components (attention, norm, embeddings, FFN) work
   - ✅ Shapes preserved through pipeline
   - ✅ Numerical stability maintained

2. **Transformer Model**
   - ✅ Can instantiate from TransformerConfig
   - ✅ Forward pass works: [batch, seq_len] → [batch, seq_len, vocab_size]
   - ✅ Causal masking prevents future attention
   - ✅ Weight tying reduces parameters
   - ✅ Proper initialization

3. **Model Registry**
   - ✅ Can register models
   - ✅ Can instantiate from registry
   - ✅ Can list available models

4. **Testing**
   - ✅ Component unit tests pass
   - ✅ Integration tests pass
   - ✅ Coverage > 85%
   - ✅ Edge cases handled

5. **Documentation**
   - ✅ All classes documented
   - ✅ Type hints complete
   - ✅ Usage examples provided

---

## Next Phase Dependency

Phase 3 must be complete before starting Phase 4 (Training System).

Phase 4 will:
- Use Transformer from Phase 3
- Train model on data from Phase 2
- Optimize with components from Phase 4

---

## Mathematical Background

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)·W^O
where head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
```

### RMSNorm
```
RMSNorm(x) = x / RMS(x)·γ
where RMS(x) = √(1/d * Σ x_i^2)
```

### Feed-Forward
```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
(or GELU instead of ReLU)
```

---

## Performance Considerations

- **Memory**: O(seq_len^2) for attention (quadratic with sequence length)
- **Computation**: O(seq_len * d_model * num_heads) per layer
- **Parameters**: ~d_model^2 * num_layers

For MVP:
- seq_len=256, d_model=512, num_heads=8, num_layers=6
- ~83M parameters (manageable)

---

## Extension Points (Future Phases)

- RoPE (Rotary Position Embeddings)
- Different attention patterns (sparse, linear)
- Multi-query attention
- Flash attention for efficiency
- Mixture of Experts
