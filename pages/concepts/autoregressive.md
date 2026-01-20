# Autoregressive Language Modeling

Understand the autoregressive training objective and how language models generate text.

## What is Autoregressive?

**Autoregressive:** Predict next item based on all previous items.

```
Given:    "The cat sat on the"
Predict:  "mat"

Given:    "The cat sat on the mat"
Predict:  "."
```

Each prediction depends **only on what came before**, never what comes after.

## The Core Idea

### Sequential Prediction

Language modeling as sequential prediction:

```
Input:     The  cat  sat  on   the  mat
Target:    cat  sat  on   the  mat  .
           ↑    ↑    ↑    ↑    ↑    ↑
           Predict each token from all previous tokens
```

**Key insight:** Training and generation use the same process!

### Autoregressive Property

```
P(sentence) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁...xₙ₋₁)
```

Decompose full sequence probability into chain of conditional probabilities.

## Training Objective

### Maximum Likelihood

**Goal:** Maximize probability of training data.

$$\max_{\theta} \sum_{i=1}^{N} \log P(x_i | x_{<i}; \theta)$$

**In words:** Make the model assign high probability to the actual next token.

### Cross-Entropy Loss

**Equivalent formulation:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$

**Implementation:**

```python
def compute_loss(model, input_ids, target_ids):
    """
    Args:
        input_ids: (batch, seq_len)
        target_ids: (batch, seq_len) - shifted by 1
    """
    # Get logits
    logits = model(input_ids)  # (batch, seq_len, vocab_size)
    
    # Compute cross-entropy
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1)
    )
    
    return loss
```

### Teacher Forcing

**During training:** Use ground truth as context, not model's predictions.

```
Input:     [The, cat, sat]
Target:    [cat, sat, on]
           ↑
           Use ground truth "cat", not prediction
```

**Why?**
- ✅ Faster training (parallel computation)
- ✅ Stable gradients
- ✅ Efficient use of data

**Trade-off:**
- ❌ Exposure bias (model never sees own mistakes during training)

## Causal Attention

### Masking Future Tokens

**Critical:** Model must not see future tokens!

```
                Future (masked)
                  ↓↓↓↓↓
Attention:   T  c  a  t  .
           T ✓  ✗  ✗  ✗  ✗
           c ✓  ✓  ✗  ✗  ✗
           a ✓  ✓  ✓  ✗  ✗
           t ✓  ✓  ✓  ✓  ✗
           . ✓  ✓  ✓  ✓  ✓
```

Token at position $i$ can only attend to positions $≤ i$.

### Causal Mask Implementation

```python
def create_causal_mask(seq_len):
    """
    Creates lower triangular mask.
    
    Returns:
        mask: (seq_len, seq_len) where mask[i,j] = False if i < j
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.bool()

# Usage in attention
scores = torch.matmul(Q, K.transpose(-2, -1))  # (seq_len, seq_len)
mask = create_causal_mask(seq_len)
scores = scores.masked_fill(~mask, float('-inf'))  # Block future
attn = F.softmax(scores, dim=-1)
```

### Why Causal?

**Matches generation:** During generation, we only have past tokens anyway.

```
Generation time:
"The cat" → predict "sat" (can't see "on the mat" yet!)
```

Training with causal attention ensures model learns in same conditions it will use.

## Training vs Generation

### Training: Parallel

Process entire sequence at once:

```python
# All positions predicted in parallel
input_seq = [The, cat, sat, on, the, mat]
targets = [cat, sat, on, the, mat, .]

logits = model(input_seq)  # (seq_len, vocab_size)
loss = cross_entropy(logits, targets)
```

**Efficient:** Single forward pass for entire sequence.

### Generation: Sequential

One token at a time:

```python
# Start with prompt
generated = [The]

# Generate step by step
for _ in range(max_length):
    logits = model(generated)  # (len(generated), vocab_size)
    next_token = sample(logits[-1])  # Only use last position
    generated.append(next_token)
    
    if next_token == END_TOKEN:
        break
```

**Slow:** Requires $n$ forward passes for $n$ tokens.

## Sequence Probability

### Computing P(sequence)

```python
def sequence_probability(model, tokens):
    """
    Compute probability of a sequence.
    """
    log_prob = 0.0
    
    for i in range(1, len(tokens)):
        # Context: all tokens before position i
        context = tokens[:i]
        
        # Get probability distribution over next token
        logits = model(context)
        probs = F.softmax(logits[-1], dim=-1)
        
        # Probability of actual next token
        next_token = tokens[i]
        log_prob += torch.log(probs[next_token])
    
    return torch.exp(log_prob)
```

### Perplexity

**Perplexity** is exponential of average negative log-likelihood:

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

**Interpretation:** Weighted average branching factor.
- Perplexity of 10 → "model effectively choosing among 10 options"
- Lower is better

## Sampling Strategies

### Greedy Decoding

Always pick most likely token:

```python
next_token = torch.argmax(logits)
```

**Pros:** Deterministic, fast  
**Cons:** Repetitive, no exploration

### Temperature Sampling

Scale logits before sampling:

```python
def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Temperature effects:**
- Low (0.5): More confident, focused on top choices
- Medium (1.0): Balanced sampling
- High (1.5): More random, explores tail of distribution

### Top-k Sampling

Sample from top $k$ most likely:

```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[idx]
```

Prevents sampling from very unlikely tokens.

### Nucleus (Top-p) Sampling

Sample from smallest set whose cumulative probability exceeds $p$:

```python
def nucleus_sampling(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= p
    mask[0] = True  # Always include top token
    
    filtered_probs = sorted_probs[mask]
    filtered_indices = sorted_indices[mask]
    
    # Sample
    idx = torch.multinomial(filtered_probs, num_samples=1)
    return filtered_indices[idx]
```

Adaptive: cutoff changes based on distribution confidence.

## Exposure Bias

### The Problem

**Training:** Always see ground truth context  
**Generation:** See own predictions (which may be wrong)

```
Training:   [The, cat, sat] → predict "on" (correct context)
Generation: [The, cat, ate] → predict "on" (wrong context!)
                    ↑
                    Model predicted "ate" instead of "sat"
```

Model never learns to recover from mistakes during training.

### Mitigation Strategies

#### 1. Scheduled Sampling

Occasionally use model's predictions during training:

```python
if random.random() < scheduled_sampling_rate:
    context = model_prediction  # Use prediction
else:
    context = ground_truth  # Use ground truth
```

Start low, increase over training.

#### 2. More Training

More diverse training data helps model be robust.

#### 3. Beam Search

During generation, explore multiple candidates and pick best overall sequence.

## Context Length

### Fixed Context Window

Most models have maximum context length:

```python
max_seq_len = 256  # Can only attend to 256 previous tokens
```

**Trade-off:**
- Longer: More context, better coherence
- Shorter: Less memory, faster

### Handling Long Sequences

**During training:**
- Split into chunks
- Sliding window

**During generation:**
- Keep only recent context
- Summarization
- Memory mechanisms (advanced)

## Bidirectional vs Autoregressive

### Autoregressive (GPT-style)

✅ Generate text naturally  
✅ Simple training objective  
❌ Only sees past context

### Bidirectional (BERT-style)

✅ Sees both past and future  
✅ Better for understanding tasks  
❌ Can't generate (no left-to-right order)

**FundamentaLLM uses autoregressive** for text generation.

## Conditioning

### Unconditional Generation

No prompt, start from scratch:

```python
generated = [START_TOKEN]
# Generate from here
```

Explores model's learned distribution.

### Conditional Generation

Start with prompt:

```python
prompt = "Once upon a time"
generated = tokenize(prompt)
# Continue from prompt
```

**All generation is conditional** if you provide a prompt!

## Practical Considerations

### 1. Start Token

Some models use explicit start token:

```
<|start|> The cat sat on the mat.
```

Signals beginning of sequence.

### 2. End Token

Signal when generation should stop:

```
The cat sat on the mat. <|end|>
```

Model learns to predict this when sequence is complete.

### 3. Padding

For batching, pad sequences to same length:

```
Sequence 1: [The, cat, <pad>, <pad>]
Sequence 2: [Hello, world, how, are]
```

Mask padding in loss computation:

```python
loss = F.cross_entropy(logits, targets, ignore_index=PAD_ID)
```

## Advanced Topics

### Cached Generation

**Problem:** Recompute all previous tokens at each step.

**Solution:** Cache key and value matrices:

```python
# First step: compute from scratch
kv_cache = compute_kv(prompt)
logits = model(prompt, kv_cache=kv_cache)

# Subsequent steps: reuse cache
for step in range(max_length):
    # Only compute for new token
    logits = model(new_token, kv_cache=kv_cache)
    new_token = sample(logits)
    kv_cache = update_cache(kv_cache, new_token)
```

Much faster for long sequences!

### Parallel Decoding

Speculative decoding, token speculation, and other techniques to generate multiple tokens in parallel (advanced, not in FundamentaLLM baseline).

## Mathematical Properties

### Chain Rule of Probability

Autoregressive factorization follows chain rule:

$$P(x_1, x_2, ..., x_n) = P(x_1) \prod_{i=2}^{n} P(x_i | x_1, ..., x_{i-1})$$

This is exact, not an approximation!

### Information Theory

Each prediction provides information:

$$I(x_i | x_{<i}) = -\log_2 P(x_i | x_{<i})$$

Measured in bits. Good predictions → low surprisal.

### Entropy

Model's uncertainty about next token:

$$H = -\sum_{x} P(x | x_{<i}) \log P(x | x_{<i})$$

High entropy → model is uncertain  
Low entropy → model is confident

## Comparison to Other Objectives

### Masked Language Modeling (BERT)

Randomly mask tokens, predict masked:

```
Input:  The [MASK] sat on the mat
Predict: cat
```

Not autoregressive (can see both sides).

### Sequence-to-Sequence

Encoder-decoder architecture:

```
Encoder: Process full input (bidirectional)
Decoder: Generate output (autoregressive)
```

Used for translation, summarization.

### Denoising Objectives

Add noise, learn to denoise:

```
Input:  Teh cat sat no teh mat
Target: The cat sat on the mat
```

Different from pure language modeling.

**FundamentaLLM:** Pure autoregressive language modeling.

## Further Reading

- "Attention is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)
- "Generating Sequences With Recurrent Neural Networks" (Graves, 2013)

## Next Steps

- [Transformer Architecture](./transformers.md) - Model structure
- [Attention Mechanism](./attention.md) - How causal attention works
- [Language Modeling](./language-modeling.md) - Training objectives
- [Generation Module](../modules/generation.md) - Implementation
