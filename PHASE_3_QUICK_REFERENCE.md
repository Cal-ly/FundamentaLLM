# Phase 3 Quick Reference Guide

## 🔥 Phase 3 Complete: Model Architecture (112 tests ✅)

### Component Quick Access

| Component | File | Tests | Key Class |
|-----------|------|-------|-----------|
| Normalization | `components/normalization.py` | 20 | `LayerNorm`, `RMSNorm` |
| Attention | `components/attention.py` | 16 | `MultiHeadAttention` |
| Embeddings | `components/embeddings.py` | 28 | `LearnedPositionalEncoding`, `SinusoidalPositionalEncoding` |
| Feed-Forward | `components/feedforward.py` | 20 | `FeedForwardNetwork` |
| Transformer | `transformer.py` | 28 | `TransformerBlock`, `Transformer` |

### Quick Usage Examples

#### 1. Normalization
```python
from fundamentallm.models.components import RMSNorm, LayerNorm

# Modern norm (default, efficient)
norm = RMSNorm(d_model=512)
x = torch.randn(2, 32, 512)
output = norm(x)  # [2, 32, 512]

# Reference norm
norm = LayerNorm(d_model=512)
output = norm(x)  # [2, 32, 512]
```

#### 2. Attention
```python
from fundamentallm.models.components import MultiHeadAttention

attn = MultiHeadAttention(d_model=512, num_heads=8, causal=True)
x = torch.randn(2, 32, 512)
output = attn(x, x, x)  # Self-attention: [2, 32, 512]
```

#### 3. Positional Encoding
```python
from fundamentallm.models.components import create_positional_encoding

# Learned (trainable)
pos_enc = create_positional_encoding("learned", d_model=512)
x = torch.randn(2, 32, 512)
pos = pos_enc(x)  # [1, 32, 512]

# Sinusoidal (fixed, extrapolatable)
pos_enc = create_positional_encoding("sinusoidal", d_model=512)
pos = pos_enc(x)  # [1, 32, 512]
```

#### 4. Feed-Forward
```python
from fundamentallm.models.components import FeedForwardNetwork

ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
x = torch.randn(2, 32, 512)
output = ffn(x)  # [2, 32, 512]
```

#### 5. Complete Transformer
```python
from fundamentallm.config import TransformerConfig
from fundamentallm.models.transformer import Transformer

config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
)

model = Transformer(config)

# Forward pass
input_ids = torch.randint(0, 10000, (2, 32))
logits = model(input_ids)  # [2, 32, 10000]

# Generation step
next_token_logits = logits[0, -1, :]  # Last position logits
next_token = next_token_logits.argmax().item()
```

### Architecture Visualization

```
Input Tokens [batch, seq_len]
    ↓
Token Embedding → [batch, seq_len, d_model]
    ↓
+ Positional Encoding
    ↓
[ Transformer Block × num_layers ]
│  ├─ RMSNorm
│  ├─ MultiHeadAttention
│  ├─ Residual Add
│  ├─ RMSNorm
│  ├─ FeedForward
│  └─ Residual Add
    ↓
Output Normalization
    ↓
Linear Projection → [batch, seq_len, vocab_size]
    ↓
Output Logits
```

### Configuration Options

```python
TransformerConfig(
    vocab_size: int,              # Vocabulary size
    d_model: int = 512,           # Model dimension
    num_heads: int = 8,           # Attention heads
    num_layers: int = 6,          # Number of blocks
    sequence_length: int = 256,   # Max sequence length
    dropout: float = 0.1,         # Dropout probability
    ffn_expansion: int = 4,       # d_ff = d_model * ffn_expansion
    pos_encoding: str = "learned" # "learned" or "sinusoidal"
)
```

### Performance Notes

- **RMSNorm**: Faster than LayerNorm, 50% fewer parameters
- **Causal Attention**: O(seq_len²) memory, requires full sequence at inference
- **Weight Tying**: Reduces vocab projection parameters
- **Pre-norm**: Better gradient flow, more stable training
- **Learned pos_enc**: More flexible but doesn't extrapolate
- **Sinusoidal pos_enc**: Fixed, extrapolates to longer sequences

### Common Patterns

#### 1. Next-Token Prediction
```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
next_logits = logits[0, -1, :]  # Last token
next_token = next_logits.argmax().item()
```

#### 2. Training Setup
```python
model = Transformer(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    logits = model(batch['input_ids'])
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch['target_ids'].view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 3. Saving/Loading
```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model = Transformer(config)
model.load_state_dict(torch.load("model.pt"))
```

### Test Coverage

- **Unit Tests**: All components tested individually (112 tests)
- **Shape Tests**: All operations preserve expected shapes
- **Gradient Tests**: All layers support backpropagation
- **Determinism Tests**: Eval mode is deterministic
- **Integration Tests**: End-to-end generation pipeline
- **Edge Cases**: Different batch/sequence sizes, long sequences

### Next Steps

Ready for Phase 4:
- Training system (optimizer, scheduler, trainer)
- Loss computation and learning dynamics
- Model evaluation metrics
- Generation strategies (sampling, beam search)

---
**Total Tests:** 130 (18 Phase 2 + 112 Phase 3)
**Status:** ✅ All passing
**Coverage:** 100% for all model components
