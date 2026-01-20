# Positional Encoding

Learn how transformers encode position information to understand sequence order.

## The Problem

Transformers process all tokens in parallel, unlike RNNs which process sequentially. This is great for speed, but creates a problem:

**The model has no sense of position.**

```
"The cat sat" and "sat cat The"
```

Look identical to a transformer without positional encoding! Both are just three tokens being processed simultaneously.

## The Solution

**Positional Encoding:** Add position information to the input embeddings.

```
Token Embedding:      [0.2, 0.5, 0.3, ...]
Position Encoding:  + [0.0, 1.0, 0.0, ...]
Final Representation: [0.2, 1.5, 0.3, ...]
```

Now each token "knows" where it is in the sequence.

## Two Approaches

### 1. Learned Positional Embeddings

**What:** Train position embeddings like token embeddings.

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)

# Use
pos_ids = torch.arange(seq_len)
pos_emb = self.pos_embedding(pos_ids)
x = token_emb + pos_emb
```

**Pros:**
- ✅ Model learns optimal encodings for the task
- ✅ Simple to implement

**Cons:**
- ❌ Fixed maximum length (can't extrapolate)
- ❌ More parameters to learn

**Used by:** BERT, GPT

### 2. Sinusoidal Positional Encoding

**What:** Use fixed sinusoidal patterns (no learning required).

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**FundamentaLLM uses sinusoidal encoding** for its mathematical elegance.

## Sinusoidal Encoding Explained

### The Formula

For position $pos$ and dimension $i$:
- **Even dimensions (0, 2, 4, ...)**: Use sine
- **Odd dimensions (1, 3, 5, ...)**: Use cosine

The frequency decreases as dimension increases:
- Low dimensions: High frequency (changes quickly across positions)
- High dimensions: Low frequency (changes slowly across positions)

### Visualization

```
Dimension 0 (high freq):  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
Dimension 2 (med freq):   ∿∿∿∿∿∿∿∿∿
Dimension 4 (low freq):   ∿∿∿∿∿

Position: 0  1  2  3  4  5  6  7  8  9  10
```

Each position gets a unique pattern across all dimensions.

### Implementation

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute frequency bands
        # div_term[i] = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (saved with model but not trained)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]
```

## Why Sinusoidal Works

### 1. Unique Patterns

Each position has a unique encoding across all dimensions.

```
Position 0:  [0.00, 1.00, 0.00, 1.00, 0.00, 1.00, ...]
Position 1:  [0.84, 0.54, 0.01, 1.00, 0.00, 1.00, ...]
Position 2:  [0.91, -0.42, 0.02, 1.00, 0.00, 1.00, ...]
```

No two positions have the same encoding.

### 2. Relative Position Information

The key insight: **Linear combinations of sinusoids can represent relative positions**.

For any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$.

This means the model can learn to attend to relative positions like "the token 3 positions before."

### 3. Extrapolation

Can handle sequences longer than seen during training:

```
Trained on: max_len = 512
Can encode: position 1000 (just compute the sine/cosine)
```

Learned embeddings can't do this—they only know positions 0-511.

### 4. No Parameters

Completely deterministic, no training needed. Saves parameters and computation.

## Properties

### Frequency Spectrum

Different dimensions encode different frequencies:

| Dimension | Wavelength | Interpretation |
|-----------|-----------|----------------|
| 0-1 | 2π | Immediate neighbors |
| 2-3 | ~6.3 | Local context (few tokens) |
| 4-5 | ~20 | Medium context |
| ... | ... | ... |
| d-2, d-1 | ~10000 | Global position |

Low dimensions: fine-grained position  
High dimensions: coarse-grained position

### Dot Product Similarity

Nearby positions have higher dot product:

```python
# Compute similarity
sim = torch.matmul(pe[0], pe[1])  # Positions 0 and 1: high
sim = torch.matmul(pe[0], pe[100])  # Positions 0 and 100: low
```

Model can use this to learn position-aware attention.

## Comparison: Learned vs Sinusoidal

| Aspect | Learned | Sinusoidal |
|--------|---------|------------|
| Parameters | Yes (max_len × d_model) | No |
| Training | Required | None |
| Extrapolation | No (fixed max_len) | Yes |
| Flexibility | Task-specific optimal | Universal |
| Performance | Slightly better (sometimes) | Nearly as good |

**FundamentaLLM choice:** Sinusoidal for simplicity and universality.

## Alternative Approaches

### Rotary Position Embeddings (RoPE)

Used in modern models (GPT-NeoX, LLaMA):
- Rotates attention keys and queries based on position
- Better for long sequences
- More complex implementation

### ALiBi (Attention with Linear Biases)

Add position-dependent bias to attention scores:
- Very simple
- Great extrapolation
- No positional encoding needed

### Relative Positional Encoding

Encode relative distances between tokens:
- Used in Transformer-XL
- Better for recurrence-style models

## Practical Considerations

### When to Add Position

```python
# Standard approach (FundamentaLLM)
x = token_embedding(input_ids)
x = positional_encoding(x)
x = transformer(x)
```

Position is added once at the input.

### Scaling

Token embeddings are often scaled before adding position:

```python
x = token_embedding(input_ids) * math.sqrt(d_model)
x = x + positional_encoding
```

**Why?** Keeps embedding magnitude consistent with positional encoding.

### Position Dropout

Can apply dropout after adding position:

```python
x = token_embedding(input_ids)
x = positional_encoding(x)
x = dropout(x)  # Regularization
```

## Visualization Example

For d_model=4, first few positions:

```
Position 0: [ 0.000,  1.000,  0.000,  1.000]
Position 1: [ 0.841,  0.540,  0.010,  1.000]
Position 2: [ 0.909, -0.416,  0.020,  1.000]
Position 3: [ 0.141, -0.990,  0.030,  1.000]
Position 4: [-0.757, -0.653,  0.040,  1.000]
Position 5: [-0.959,  0.284,  0.050,  0.999]
```

Note how:
- First two dimensions oscillate quickly (high frequency)
- Last two dimensions change slowly (low frequency)

## Implementation Tips

### Register as Buffer

```python
self.register_buffer('pe', pe)
```

**Why?** 
- Saved with model state
- Moved to GPU with model
- Not treated as trainable parameter

### Precompute

Compute all encodings once in `__init__`, not in `forward`:

```python
# Good: Precompute
pe = torch.zeros(max_len, d_model)
# ... compute pe ...
self.register_buffer('pe', pe)

# Bad: Compute every forward pass
def forward(self, x):
    pe = self._compute_pe(x.size(1))  # Slow!
```

### Slice Efficiently

Only use needed positions:

```python
def forward(self, x):
    seq_len = x.size(1)
    return x + self.pe[:seq_len, :]  # Only first seq_len positions
```

## Common Mistakes

### 1. Wrong Dimension Order

```python
# Wrong
pe[:, 0::2] = torch.sin(position / div_term)  # Missing multiply

# Right
pe[:, 0::2] = torch.sin(position * div_term)
```

### 2. Not Broadcasting

```python
# Shape mismatch
x: (batch, seq_len, d_model)
pe: (seq_len, d_model)  # Need to broadcast

# Fix: PyTorch broadcasts automatically
x + pe[:seq_len, :]  # Works!
```

### 3. Forgetting Buffer

```python
# Wrong: Normal tensor (not saved with model)
self.pe = pe

# Right: Register as buffer
self.register_buffer('pe', pe)
```

## Further Reading

- Original Transformer paper: "Attention is All You Need"
- [RoPE paper](https://arxiv.org/abs/2104.09864) - Rotary embeddings
- [ALiBi paper](https://arxiv.org/abs/2108.12409) - Attention biases
- [Relative Position](https://arxiv.org/abs/1803.02155) - Transformer-XL

## Next Steps

- [Transformer Architecture](./transformers.md) - How position fits in
- [Embeddings](./embeddings.md) - Token embeddings
- [Attention Mechanism](./attention.md) - Using position information
- [Models Module](../modules/models.md) - Implementation details
