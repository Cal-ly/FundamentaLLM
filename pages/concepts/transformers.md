# Transformer Architecture

The transformer is the core innovation enabling large language models. Let's understand it from first principles.

## The Big Idea

**Problem:** How do we process sequences of data (like text) in parallel while allowing each element to "understand" all other elements?

**Solution:** Attention mechanism - let each token attend to (focus on) all other tokens simultaneously.

## The Basic Flow

```
Input Tokens: [The, cat, sat]
    ↓
Embed (convert to vectors)
    ↓
Add Positional Information (where in sequence?)
    ↓
Transformer Block (repeat N times):
  ├─ Self-Attention (what should I focus on?)
  ├─ Add & Normalize
  ├─ Feed-Forward Network
  └─ Add & Normalize
    ↓
Output Layer (predict next token)
    ↓
Output: Probability for next token
```

## The Transformer Block

Each transformer layer consists of two main components:

### 1. Multi-Head Self-Attention

**Intuition:** For each token, figure out which other tokens are important.

**How it works:**
```
For token "cat":
  - What about "The"? → 0.2 (article, less important)
  - What about "cat"? → 0.5 (self-reference)
  - What about "sat"? → 0.3 (verb describing cat)
  
Result: "cat" focuses 50% on itself, 20% on "The", 30% on "sat"
```

**Multi-head:** Do this multiple times in parallel with different "focuses"
- Head 1: Focus on grammatical relationships
- Head 2: Focus on semantic meaning
- Head 3: Focus on word positions
- Head 4: ... etc

Then combine all the attention patterns.

### 2. Feed-Forward Network

**Intuition:** After attention, process the information through a fully-connected network.

**Structure:**
```
Dense Layer (expand)
  ↓
Activation (ReLU or GELU)
  ↓
Dense Layer (project back)
```

## Position Encoding

**Problem:** Transformers have no inherent sense of position. "The cat sat" and "sat cat The" look identical to the transformer.

**Solution:** Add positional encoding - inject position information into embeddings.

**FundamentaLLM uses sinusoidal positional encoding:**

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:
- `pos` = position in sequence
- `i` = dimension index
- `d` = total dimensions

**Result:** Each position has a unique encoding pattern.

## Layer Normalization

**Problem:** Training deep networks is unstable. Activations shift unpredictably.

**Solution:** Normalize layer outputs to mean=0, std=1.

```
Unnormalized:  [10, 20, 15, 8]  ← varies wildly
Normalized:    [0.8, 1.2, 0.9, -0.3]  ← stable range
```

FundamentaLLM uses **pre-norm architecture** (normalize before processing):

```
Attention In → LayerNorm → Attention → Add Residual
```

This is more stable than post-norm.

## The Full Model

```
┌─────────────────────────────────────┐
│ Input Tokens (IDs)                  │
├─────────────────────────────────────┤
│ Token Embedding                     │
│ (IDs → vectors)                     │
├─────────────────────────────────────┤
│ + Positional Encoding               │
│ (add position information)          │
├─────────────────────────────────────┤
│ Transformer Block 1                 │
│ ├─ LayerNorm                        │
│ ├─ Multi-Head Attention (8 heads)   │
│ ├─ Residual Connection (+)          │
│ ├─ LayerNorm                        │
│ ├─ Feed-Forward (2-layer MLP)       │
│ └─ Residual Connection (+)          │
├─────────────────────────────────────┤
│ Transformer Block 2                 │
│ ... (same structure)                │
├─────────────────────────────────────┤
│ ... repeat 6-12 times ...           │
├─────────────────────────────────────┤
│ Output Layer                        │
│ (linear layer to vocab size)        │
├─────────────────────────────────────┤
│ Softmax                             │
│ (convert to probabilities)          │
├─────────────────────────────────────┤
│ Output Distribution                 │
│ P(next_token | context)             │
└─────────────────────────────────────┘
```

## Key Parameters

### Configurable in FundamentaLLM

- **`model_dim`** (hidden_dim) - Vector dimension (default: 128)
  - Larger = more capacity but slower
  
- **`num_heads`** - Attention heads (default: 2)
  - Must divide `model_dim` evenly
  - More heads = more parallel attention patterns
  
- **`num_layers`** - Transformer blocks (default: 6)
  - More layers = more sequential processing = slower but better
  
- **`ff_expansion`** - FFN expansion ratio (default: 4)
  - FFN inner dimension = `model_dim * ff_expansion`

## What Each Component Does

| Component | Purpose | Why |
|-----------|---------|-----|
| Embedding | Token → Vector | Computers need numerical representations |
| Positional Encoding | Add position info | Model needs to know sequence order |
| Self-Attention | Connect all tokens | Each token should consider all context |
| Feed-Forward | Non-linear processing | Enables complex pattern learning |
| Layer Normalization | Stabilize training | Prevents activation explosion/vanishing |
| Residual Connections | Preserve information | Allows deep networks to train |

## Causal Masking

For language modeling, we can't look into the future:

```
Position:  0   1   2   3
Token:     The cat sat on
Target:        cat sat on ?

Position 0 can only see position 0
Position 1 can see positions 0-1
Position 2 can see positions 0-2
Position 3 can see positions 0-3
(can't see position 4 - it's in the future!)
```

This is enforced with **causal masking** in attention - set future attention weights to -∞ (becomes 0 after softmax).

## Training Objective

```
For each token in sequence:
  1. Feed sequence up to that token into model
  2. Model predicts probability distribution for next token
  3. Compare prediction to actual next token
  4. Compute cross-entropy loss
  5. Backpropagate and update weights
```

Repeat across many sequences and epochs.

## Comparison to RNNs

| Aspect | Transformer | RNN |
|--------|-------------|-----|
| Processing | Parallel (fast) | Sequential (slow) |
| Long-range | Excellent | Poor (vanishing gradients) |
| Training | Fast to train | Slow to train |
| Long sequences | Efficient | Memory issues |
| Per-token cost | O(n²) attention | O(n) but slower in practice |

Transformers revolutionized NLP by making parallelization possible.

## Extensions (Not in FundamentaLLM, but good to know)

- **Multi-query attention:** Share key/value across heads
- **Rotary embeddings:** Better positional encoding
- **Flash attention:** Faster attention computation
- **KV caching:** Speed up generation
- **Attention patterns:** Sparse, local, strided attention

## Implementation in FundamentaLLM

See:
- [Models Module](../modules/models.md) - Code details
- [Attention Mechanism](./attention.md) - Deep dive into attention
- [Training Guide](../guide/training.md) - How we optimize it

## Further Reading

- "Attention is All You Need" (Vaswani et al, 2017) - Original paper
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [3Blue1Brown - Transformers](https://www.youtube.com/watch?v=wjZofJX0j4M)

## Next Steps

- **[Attention Mechanism](./attention.md)** - Deeper dive
- **[Training Models](../guide/training.md)** - How to train transformers
- **[Language Modeling](./language-modeling.md)** - The learning objective
