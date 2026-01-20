# Loss Functions

Understand the loss functions used to train language models and optimize their performance.

## Overview

**Loss functions** measure how wrong the model's predictions are. Training minimizes this loss.

For language modeling, we primarily use **cross-entropy loss**.

## Cross-Entropy Loss

### The Fundamental Loss

**Definition:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P_{\theta}(x_i | x_{<i})$$

**In words:** Average negative log-probability of the correct next token.

**Why this loss?**
- Directly corresponds to maximum likelihood estimation
- Measures "surprise" at seeing the actual data
- Theoretically optimal for probability modeling

### Implementation

```python
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    """
    Args:
        logits: (batch, seq_len, vocab_size) - raw model outputs
        targets: (batch, seq_len) - target token IDs
    
    Returns:
        loss: scalar
    """
    # Reshape for cross_entropy
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Compute loss
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    return loss
```

### Example

```python
# Model predicts probabilities
probs = [0.1, 0.6, 0.2, 0.1]  # 4-token vocab
# Actual next token is index 1

# Cross-entropy
loss = -log(probs[1])
     = -log(0.6)
     = 0.51

# If model was confident and correct
probs_good = [0.01, 0.95, 0.02, 0.02]
loss_good = -log(0.95) = 0.05  # Low loss!

# If model was wrong
probs_bad = [0.6, 0.1, 0.2, 0.1]
loss_bad = -log(0.1) = 2.30  # High loss!
```

## Loss Components

### Logits to Probabilities

**Logits:** Raw model outputs (unbounded)

```python
logits = model(input)  # [-3.2, 1.5, 0.8, -0.5, ...]
```

**Probabilities:** Apply softmax

$$P(x_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

```python
probs = F.softmax(logits, dim=-1)
# [0.01, 0.47, 0.23, 0.06, ...]  # Sum to 1.0
```

### Numerical Stability

**Problem:** Softmax can overflow/underflow.

**Solution:** `cross_entropy` combines softmax + log + loss numerically stable:

```python
# Numerically stable (use this!)
loss = F.cross_entropy(logits, targets)

# Numerically unstable (don't do this)
probs = F.softmax(logits, dim=-1)
log_probs = torch.log(probs)  # Can be -inf!
loss = -log_probs[targets].mean()
```

PyTorch's `cross_entropy` handles this internally using log-sum-exp trick.

## Reduction Methods

### Mean Reduction (Default)

```python
loss = F.cross_entropy(logits, targets, reduction='mean')
```

**Average** loss over all tokens.

**Use when:** Standard training

### Sum Reduction

```python
loss = F.cross_entropy(logits, targets, reduction='sum')
```

**Total** loss across all tokens.

**Use when:** You want to manually control averaging (e.g., different sequence lengths)

### None Reduction

```python
losses = F.cross_entropy(logits, targets, reduction='none')
# Shape: (batch, seq_len) - per-token losses
```

**Individual** loss for each token.

**Use when:** 
- Token-level weighting
- Analysis of which tokens are hard
- Custom loss computation

## Special Tokens

### Ignoring Padding

```python
# Padding tokens should not contribute to loss
loss = F.cross_entropy(
    logits, 
    targets, 
    ignore_index=PAD_TOKEN_ID
)
```

**Why?** Padding is artificial, not real data.

### Example

```
Sequence 1: [The, cat, sat, <pad>]
Sequence 2: [Hello, world, <pad>, <pad>]

# Loss computed only for:
Sequence 1: [The, cat, sat] (3 tokens)
Sequence 2: [Hello, world]  (2 tokens)
```

## Perplexity

### Relationship to Loss

**Perplexity** is exp(loss):

$$\text{Perplexity} = \exp(\mathcal{L})$$

```python
loss = 2.0
perplexity = torch.exp(loss)  # 7.39
```

### Interpretation

**Perplexity** ≈ "effective vocabulary size" for next token prediction.

- Perplexity of 256 → Random guess over 256 chars
- Perplexity of 10 → Effectively choosing among ~10 options
- Perplexity of 2 → Very confident (binary choice)

**Lower is better!**

### Why Use It?

Perplexity is more interpretable than raw loss:

```
Loss: 1.5 vs 2.0  → Hard to judge improvement
Perplexity: 4.5 vs 7.4  → Clearer: model is more confident
```

## Label Smoothing

### Motivation

Hard targets can be overconfident:

```
Target: [0, 1, 0, 0]  # 100% certain class 1

Model learns to output very large logits:
Logits: [-10, 100, -10, -10]  → probs ≈ [0, 1, 0, 0]
```

**Problem:** Overconfident, poor calibration, may hurt generalization.

### Label Smoothing Formula

Smooth hard targets toward uniform distribution:

$$q_i = \begin{cases}
(1 - \epsilon) + \epsilon/K & \text{if } i = y \\
\epsilon/K & \text{otherwise}
\end{cases}$$

Where:
- $\epsilon$ = smoothing amount (e.g., 0.1)
- $K$ = vocabulary size
- $y$ = true label

### Implementation

```python
def label_smoothing_loss(logits, targets, smoothing=0.1):
    """
    Cross-entropy with label smoothing.
    """
    vocab_size = logits.size(-1)
    confidence = 1.0 - smoothing
    
    # Create smoothed targets
    smooth_targets = torch.zeros_like(logits)
    smooth_targets.fill_(smoothing / (vocab_size - 1))
    smooth_targets.scatter_(-1, targets.unsqueeze(-1), confidence)
    
    # KL divergence loss
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(smooth_targets * log_probs, dim=-1)
    
    return loss.mean()
```

### Effect

```
Without smoothing:
Target:  [0.0, 1.0, 0.0, 0.0]

With smoothing (ε=0.1, K=4):
Target:  [0.033, 0.933, 0.033, 0.033]
         └─────┬──────┘
               └── 0.1 distributed
```

**Benefits:**
- ✅ Better calibration
- ✅ May improve generalization
- ✅ Prevents overconfidence

**FundamentaLLM:** Optional label smoothing parameter.

## Gradient Flow

### Loss → Gradients

```python
# Forward
logits = model(input)
loss = F.cross_entropy(logits, targets)

# Backward
loss.backward()

# Gradients now computed for all parameters
```

### Softmax Gradient

For cross-entropy + softmax, gradient is elegant:

$$\frac{\partial \mathcal{L}}{\partial z_i} = P_i - \mathbb{1}_{y=i}$$

**In words:** 
- Predicted probability minus 1 if correct class
- Predicted probability minus 0 otherwise

**Example:**
```
Target: class 1
Probs: [0.1, 0.6, 0.2, 0.1]

Gradients: [-0.1, -0.4, -0.2, -0.1]
            └────┬─────┘
                 └── Sum to 0
```

### Why This Matters

- Simple, stable gradient
- Large gradient when wrong (fast learning)
- Small gradient when right (converged)

## Loss Monitoring

### Training Loop

```python
for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch in train_loader:
        # Forward
        logits = model(batch['input'])
        loss = F.cross_entropy(logits, batch['target'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.3f}")
```

### What to Watch

**Good training:**
```
Epoch 1: Loss = 4.234
Epoch 2: Loss = 3.567
Epoch 3: Loss = 2.891
Epoch 4: Loss = 2.456
...
Epoch 20: Loss = 1.234
```

Steady decrease, eventually plateaus.

**Bad training:**
```
Epoch 1: Loss = 4.234
Epoch 2: Loss = nan
```

Explosion → learning rate too high or numerical issues.

```
Epoch 1: Loss = 4.234
...
Epoch 20: Loss = 4.189
```

Not learning → learning rate too low, model too small, or data issues.

## Alternative Losses

### Focal Loss

Focuses on hard examples:

$$\mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log p_t$$

Where $p_t$ is probability of correct class.

**Effect:** Down-weights easy examples (high $p_t$), up-weights hard ones (low $p_t$).

**Use case:** Imbalanced classification

### Ranking Losses

For contrastive learning:

$$\mathcal{L} = \max(0, \text{margin} - s_+ + s_-)$$

Where $s_+$ is score for positive, $s_-$ for negative.

**Not used in FundamentaLLM** (standard LM training).

### Auxiliary Losses

Sometimes add extra objectives:

```python
# Main loss
lm_loss = cross_entropy(logits, targets)

# Auxiliary task
aux_loss = some_auxiliary_objective()

# Combined
total_loss = lm_loss + 0.1 * aux_loss
```

Examples: next sentence prediction, token type prediction.

## Loss Analysis

### Per-Token Loss

```python
# Get loss for each token
losses = F.cross_entropy(logits, targets, reduction='none')

# Analyze
print(f"Mean: {losses.mean():.3f}")
print(f"Max: {losses.max():.3f}")
print(f"Min: {losses.min():.3f}")

# Find hardest tokens
hard_indices = losses.topk(10).indices
print(f"Hardest tokens: {targets[hard_indices]}")
```

### Per-Position Loss

```python
# Loss at each sequence position
position_losses = losses.mean(dim=0)  # Average over batch

# Plot
plt.plot(position_losses)
plt.xlabel('Position')
plt.ylabel('Loss')
plt.title('Loss by Position')
```

**Typical pattern:** Higher loss at beginning (less context).

## Practical Tips

### 1. Watch Initial Loss

For random model, expected loss ≈ log(vocab_size):

```python
# Character-level (vocab=256)
random_loss = math.log(256)  # ≈ 5.55

# If initial loss is very different, something's wrong!
```

### 2. Compare Train vs Val Loss

```
Epoch 10:
  Train Loss: 1.234
  Val Loss:   1.287  ← Small gap: good!

Epoch 20:
  Train Loss: 0.456
  Val Loss:   2.345  ← Large gap: overfitting!
```

### 3. Loss Scaling for Mixed Precision

When using float16:

```python
# Loss might be small, scale up to prevent underflow
scaled_loss = loss * loss_scale
scaled_loss.backward()

# Unscale gradients before optimizer step
```

### 4. Gradient Accumulation

For large effective batch size:

```python
for i, batch in enumerate(data_loader):
    loss = compute_loss(model, batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Common Issues

### Loss is NaN

**Causes:**
- Learning rate too high
- Numerical overflow in softmax
- Invalid input data

**Fixes:**
- Reduce learning rate
- Use gradient clipping
- Check for NaN in data

### Loss Not Decreasing

**Causes:**
- Learning rate too low
- Model too small
- Bad data

**Fixes:**
- Increase learning rate
- Larger model
- Check data quality

### Loss Oscillating

**Causes:**
- Learning rate too high
- Batch size too small

**Fixes:**
- Reduce learning rate
- Use learning rate schedule
- Increase batch size

## Further Reading

- "Pattern Recognition and Machine Learning" (Bishop) - Chapter on loss functions
- "Deep Learning" (Goodfellow et al.) - Chapter 6
- [Label Smoothing paper](https://arxiv.org/abs/1512.00567)
- [Focal Loss paper](https://arxiv.org/abs/1708.02002)

## Next Steps

- [Optimization](./optimization.md) - How loss is minimized
- [Training Guide](../guide/training.md) - Practical training
- [Metrics](../guide/evaluation.md) - Measuring performance
- [Troubleshooting](../guide/troubleshooting.md) - Fixing loss issues
