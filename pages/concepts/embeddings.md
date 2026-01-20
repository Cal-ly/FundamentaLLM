# Embeddings

Learn how tokens are converted to continuous vector representations that neural networks can process.

## What are Embeddings?

**Embeddings** map discrete tokens (words, characters, subwords) to continuous vectors in a high-dimensional space.

```
Token ID:  72  →  Embedding:  [0.2, -0.5, 0.3, 0.1, ...]
          (discrete)              (continuous vector)
```

**Why?** Neural networks need continuous inputs to compute gradients and learn.

## Token Embeddings

### The Basic Concept

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Embedding lookup table
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) of token IDs
        # output: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
```

**What happens:**
1. Token ID → Look up row in embedding table
2. Return that row's vector
3. Scale by √d_model (convention from transformer paper)

### Example

```python
vocab_size = 256  # Character-level
d_model = 128     # Embedding dimension

emb = TokenEmbedding(vocab_size, d_model)

# Encode "Hi" as [72, 105]
tokens = torch.tensor([[72, 105]])

# Get embeddings
output = emb(tokens)
# Shape: (1, 2, 128)
# Two vectors, each 128-dimensional
```

## How Embeddings are Learned

Embeddings are **learned parameters**, updated during training through backpropagation.

### Initialization

```python
# Random initialization (standard practice)
self.embedding = nn.Embedding(vocab_size, d_model)
# Each vector initialized ~ N(0, 1)
```

### Learning Process

```
1. Forward pass: Look up embeddings
2. Process through model
3. Compute loss
4. Backpropagate gradients to embeddings
5. Update embedding vectors
6. Repeat for many steps
```

**What's learned:** Tokens with similar meanings/uses get similar vectors.

### Geometric Properties

After training, embedding space has structure:

```
Semantically similar tokens cluster together:
  "the" ≈ "a" ≈ "an"  (articles)
  "cat" ≈ "dog" ≈ "pet"  (animals)
  "," ≈ "." ≈ "!"  (punctuation)
```

## Embedding Dimension

**Trade-off:** Higher dimension = more capacity, but more parameters.

### Typical Sizes

| Vocab Size | d_model | Parameters |
|------------|---------|------------|
| 256 (char) | 128 | 32,768 |
| 256 (char) | 256 | 65,536 |
| 50,000 (word) | 300 | 15,000,000 |
| 50,000 (word) | 768 | 38,400,000 |

**FundamentaLLM:** Small vocab (256) allows larger d_model without parameter explosion.

### Choosing Dimension

**Rules of thumb:**
- Character-level: 128-512 works well
- Subword-level (BPE): 256-1024
- Word-level: 300-768

Should match model's hidden dimension for consistency.

## Scaling Factor

### Why Multiply by √d_model?

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

**Reason:** Balance embedding magnitude with positional encoding magnitude.

**Without scaling:**
```
Token embedding:    [0.1, 0.2, ..., 0.3]  (magnitude ~1)
Positional encoding: [0.8, 0.5, ..., 0.9]  (magnitude ~1)
Sum:                 [0.9, 0.7, ..., 1.2]  (positional dominates)
```

**With scaling (√128 ≈ 11.3):**
```
Token embedding:     [1.1, 2.3, ..., 3.4]  (magnitude ~11)
Positional encoding: [0.8, 0.5, ..., 0.9]  (magnitude ~1)
Sum:                 [1.9, 2.8, ..., 4.3]  (balanced)
```

## Embedding as First Layer

Embeddings are the first layer of the model:

```
Input IDs: [72, 101, 108, 108, 111]
    ↓
Token Embedding: [[0.2, 0.5, ...], [0.1, -0.3, ...], ...]
    ↓
+ Positional Encoding
    ↓
Transformer Layers
```

**Key point:** Gradients from the entire model flow back to update embeddings.

## Weight Tying

**Technique:** Share embedding weights with output layer.

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Weight tying
        self.output_layer.weight = self.embedding.weight
```

**Benefits:**
- ✅ Fewer parameters (vocab_size × d_model saved)
- ✅ Forces consistency between input and output spaces
- ✅ Often improves performance

**Used by:** Many language models including GPT

**FundamentaLLM:** Can optionally use weight tying.

## Pre-trained Embeddings

### Concept

Use embeddings trained on large corpus:
- Word2Vec
- GloVe  
- FastText

**Advantages:**
- ✅ Better starting point
- ✅ Useful for small datasets

**Disadvantages:**
- ❌ May not fit your domain
- ❌ Fixed vocabulary
- ❌ Less flexible

**FundamentaLLM approach:** Train from scratch (simpler, more educational).

## Embedding Visualization

Embeddings are high-dimensional (128+), but we can visualize with dimensionality reduction:

### t-SNE Example

```python
from sklearn.manifold import TSNE

# Get embedding weights
embeddings = model.embedding.weight.detach().numpy()

# Reduce to 2D
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
```

**What you'll see:** Clusters of similar tokens.

## Contextual vs Static Embeddings

### Static Embeddings (FundamentaLLM)

One vector per token, regardless of context:

```
"bank" → [0.2, 0.5, 0.3, ...]  (always the same)
```

**Problem:** "bank" (river) vs "bank" (money) get same embedding.

### Contextual Embeddings (Transformers)

Embedding changes based on context:

```
"river bank" → [0.2, 0.5, 0.3, ...]  (geographical)
"money bank" → [0.1, 0.8, -0.2, ...]  (financial)
```

**How:** Transformer layers create context-dependent representations.

**Key insight:** 
- Input embedding: Static (lookup)
- Hidden representations: Contextual (after transformer layers)

## Character Embeddings

FundamentaLLM uses character-level, which has unique properties:

### Small Vocabulary

```
256 characters vs 50,000 words
→ 256 embedding vectors vs 50,000 vectors
```

Fewer parameters in embedding layer.

### Compositional Learning

Model must learn to compose characters → words:

```
Embeddings for:  'c', 'a', 't'
Model learns:    → represents "cat" (animal)
```

More challenging but more general.

### Language Agnostic

Same embedding layer handles all languages:

```
English: "Hello"  →  [72, 101, 108, 108, 111]
French:  "Bonjour" →  [66, 111, 110, 106, 111, 117, 114]
Emoji:   "😀"      →  [240, 159, 152, 128]
```

All just bytes!

## Embedding Layer Size

### Memory Usage

```python
vocab_size = 256
d_model = 512
parameters = vocab_size * d_model = 131,072
memory = parameters * 4 bytes = 524 KB
```

For character-level, this is tiny!

### Comparison to Other Layers

```
Embedding layer:     131K parameters  (0.5 MB)
Single transformer:  ~2M parameters   (8 MB)
6-layer transformer: ~12M parameters  (48 MB)
```

Embeddings are small fraction of total model.

## Best Practices

### 1. Match Model Dimension

```python
# Good: Consistent dimensions
d_model = 256
embedding = nn.Embedding(vocab_size, d_model)
transformer = Transformer(d_model, ...)

# Bad: Mismatch requires projection
embedding = nn.Embedding(vocab_size, 128)
transformer = Transformer(256, ...)  # Need projection layer
```

### 2. Initialize Properly

```python
# Default initialization is usually good
embedding = nn.Embedding(vocab_size, d_model)

# Optional: Xavier initialization
nn.init.xavier_uniform_(embedding.weight)
```

### 3. Consider Weight Tying

```python
# Save parameters
self.output_layer.weight = self.embedding.weight
```

### 4. Scale for Transformers

```python
# Standard practice for transformers
return self.embedding(x) * math.sqrt(self.d_model)
```

## Common Issues

### Embedding Collapse

**Problem:** All embeddings become too similar.

**Cause:** Too much regularization or bad initialization.

**Fix:**
- Check initialization
- Reduce dropout on embeddings
- Verify gradients are flowing

### Out of Memory

**Problem:** Embedding layer too large.

**Cause:** Large vocabulary × large dimension.

**Fix:**
- Reduce d_model
- Use smaller vocabulary (e.g., character-level)
- Weight tying to share parameters

### Poor Embedding Quality

**Problem:** Embeddings don't capture meaning well.

**Cause:** Not enough training data or capacity.

**Fix:**
- Train longer
- Use larger d_model
- More diverse training data

## Advanced Topics

### Adaptive Embeddings

Different dimensions for different frequency tokens:
- Frequent tokens: Large embeddings
- Rare tokens: Small embeddings

Saves parameters while maintaining quality.

### Factorized Embeddings

Decompose embedding matrix:
```
vocab_size × d_model  →  vocab_size × d_low × d_model
```

Reduces parameters, used in ALBERT.

### Learned Position Embeddings

Instead of sinusoidal, learn position embeddings:

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)
```

More parameters, but task-specific.

## Further Reading

- Word2Vec paper (Mikolov et al., 2013)
- GloVe paper (Pennington et al., 2014)
- "Attention is All You Need" (embedding scaling)
- FastText (character n-gram embeddings)

## Next Steps

- [Positional Encoding](./positional-encoding.md) - Adding position information
- [Transformer Architecture](./transformers.md) - How embeddings fit in
- [Models Module](../modules/models.md) - Implementation details
