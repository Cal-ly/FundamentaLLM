# Optimization

Learn how optimizers minimize loss and train neural networks effectively.

## Overview

**Optimization** is the process of adjusting model parameters to minimize the loss function.

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)$$

We use **gradient-based optimization** with backpropagation.

## Gradient Descent

### The Basic Idea

Move parameters in opposite direction of gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)$$

Where:
- $\theta_t$ = parameters at step $t$
- $\eta$ = learning rate
- $\nabla_{\theta} \mathcal{L}$ = gradient of loss w.r.t. parameters

**Intuition:** Gradient points uphill, so negative gradient points downhill.

### Example (1D)

```
Loss:  /\
      /  \
     /    \
    /      \
   /        \

Start at: x = 3 (right side)
Gradient: +2 (positive, uphill to right)
Update: x ← x - η × 2 = 3 - 0.1 × 2 = 2.8 (move left!)

Repeat until converged at minimum
```

## Stochastic Gradient Descent (SGD)

### Batch vs Stochastic

**Full batch:** Use entire dataset to compute gradient
```python
loss = compute_loss(model, entire_dataset)
loss.backward()
```

**Stochastic:** Use single sample
```python
loss = compute_loss(model, single_sample)
loss.backward()
```

**Mini-batch (standard):** Use batch of samples
```python
loss = compute_loss(model, batch)
loss.backward()
```

### Why Mini-batches?

**Trade-offs:**

| Batch Size | Gradient Quality | Speed | Memory |
|------------|------------------|-------|--------|
| 1 (SGD) | Noisy | Slow | Low |
| 32 (mini) | Good | Fast | Medium |
| All (full) | Perfect | Slow | High |

**Sweet spot:** 16-64 for most cases.

### Implementation

```python
import torch.optim as optim

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for batch in data_loader:
    # Forward pass
    logits = model(batch['input'])
    loss = F.cross_entropy(logits, batch['target'])
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update parameters
```

## Adam Optimizer

**Most popular optimizer for deep learning.**

### Key Ideas

1. **Momentum:** Use moving average of gradients
2. **Adaptive learning rates:** Different rate for each parameter
3. **Bias correction:** Account for initialization

### Algorithm

```python
# Hyperparameters
beta1 = 0.9   # Momentum decay
beta2 = 0.999 # Variance decay
eps = 1e-8    # Numerical stability

# Initialize
m = 0  # First moment (mean)
v = 0  # Second moment (variance)

# Update step
for t in range(1, num_steps + 1):
    g = compute_gradient()
    
    # Update biased moments
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    
    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # Update parameters
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

### Why Adam?

**Advantages:**
- ✅ Adaptive learning rates per parameter
- ✅ Robust to hyperparameters
- ✅ Works well out-of-the-box
- ✅ Handles sparse gradients

**Disadvantages:**
- ❌ More memory (stores m and v)
- ❌ May generalize slightly worse than SGD (debated)

### Usage

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,           # Learning rate
    betas=(0.9, 0.999), # Beta1, beta2
    eps=1e-8,
    weight_decay=0.01   # L2 regularization
)
```

**FundamentaLLM default:** Adam with lr=1e-3.

## AdamW

**Improved Adam with better weight decay.**

### The Difference

**Adam:** Weight decay applied to gradient
```python
g = gradient + weight_decay * weights
# Then use g in Adam update
```

**AdamW:** Weight decay applied to weights directly
```python
# Adam update on gradient
# Then separate weight decay
weights = weights * (1 - lr * weight_decay)
```

### Why AdamW?

**Better for:**
- Transformer models
- Transfer learning
- Better generalization

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.1  # Decoupled weight decay
)
```

**Modern best practice:** Use AdamW instead of Adam.

## Learning Rate

**Most important hyperparameter!**

### Finding Good LR

**Too high:**
```
Step 1: loss = 4.23
Step 2: loss = 3.89
Step 3: loss = nan  ← Exploded!
```

**Too low:**
```
Step 1: loss = 4.23
Step 100: loss = 4.19  ← Barely moving
Step 1000: loss = 4.12
```

**Just right:**
```
Step 1: loss = 4.23
Step 100: loss = 3.45
Step 1000: loss = 2.12
Step 5000: loss = 1.56
```

### Learning Rate Range Test

```python
# Try range of learning rates
lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

for lr in lrs:
    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train for a few steps
    losses = train_for_n_steps(model, optimizer, n=100)
    
    print(f"LR {lr}: Final loss = {losses[-1]:.3f}")
```

Pick LR with fastest initial decrease.

### Rule of Thumb

**Starting points:**
- Adam: 1e-3 to 3e-4
- SGD: 0.01 to 0.1
- AdamW: 1e-4 to 1e-3

**For large models:** Often need smaller LR (1e-4 or 3e-5).

## Learning Rate Schedules

### Constant (Baseline)

```python
# Same LR throughout training
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

Simple, works for quick experiments.

### Step Decay

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(
    optimizer,
    step_size=10,  # Every 10 epochs
    gamma=0.1      # Multiply LR by 0.1
)

# In training loop
for epoch in range(num_epochs):
    train(model, optimizer)
    scheduler.step()  # Update LR
```

**Pattern:**
```
Epochs 0-9:   LR = 1e-3
Epochs 10-19: LR = 1e-4
Epochs 20-29: LR = 1e-5
```

### Cosine Annealing

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,  # Period
    eta_min=1e-6       # Minimum LR
)
```

**Pattern:** Smooth decrease following cosine curve.

```
    LR
    ^
1e-3|╲
    | ╲
    |  ╲___
    |      ╲___
1e-6|__________╲
    └────────────→ epochs
```

**Popular for transformers.**

### Warmup

Start with low LR, gradually increase:

```python
def warmup_schedule(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr
```

**Why?** Prevents instability in early training.

**Typical:** 1000-5000 warmup steps.

### Warmup + Cosine (Best Practice)

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedule(step):
    warmup_steps = 1000
    total_steps = 100000
    
    if step < warmup_steps:
        # Linear warmup
        return step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
```

**Used by:** GPT, BERT, most modern LMs.

## Gradient Clipping

### The Problem

Gradients can explode:

```
Step 100: gradient norm = 0.5
Step 101: gradient norm = 1.2
Step 102: gradient norm = 348.7  ← Exploded!
Step 103: parameters = nan
```

### Solution: Clip Gradients

```python
import torch.nn.utils as nn_utils

# Before optimizer.step()
nn_utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Clip to max norm of 1.0
)
optimizer.step()
```

**What it does:**

$$g' = \begin{cases}
g & \text{if } \|g\| \leq \text{max\_norm} \\
\frac{g \cdot \text{max\_norm}}{\|g\|} & \text{otherwise}
\end{cases}$$

Scales down large gradients while preserving direction.

### When to Use

- ✅ RNNs (prone to exploding gradients)
- ✅ Very deep networks
- ✅ High learning rates
- ✅ Early training (before stable)

**Typical value:** 0.5 to 5.0

**FundamentaLLM:** Optional, default 1.0.

## Gradient Accumulation

### The Problem

Large batches don't fit in memory:

```python
batch_size = 128  # Too big! Out of memory
```

### Solution: Accumulate Gradients

```python
accumulation_steps = 4
effective_batch_size = 32 * 4  # = 128

optimizer.zero_grad()

for i, batch in enumerate(data_loader):
    # Forward pass (batch_size = 32)
    loss = compute_loss(model, batch)
    
    # Scale loss
    loss = loss / accumulation_steps
    
    # Backward pass (accumulate gradients)
    loss.backward()
    
    # Update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effect:** Simulate larger batch size with small memory footprint.

## Weight Decay

**L2 regularization** to prevent overfitting.

### Formula

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_i \theta_i^2$$

Penalizes large weights.

### Implementation

```python
# Optimizer applies weight decay automatically
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.1  # Lambda
)
```

**Effect:** Pushes weights toward zero (unless gradient says otherwise).

### When to Use

- ✅ Overfitting (train loss << val loss)
- ✅ Large models
- ✅ Small datasets

**Typical values:** 0.01 to 0.1

**Warning:** Too high can cause underfitting.

## Comparison of Optimizers

| Optimizer | Pros | Cons | Use When |
|-----------|------|------|----------|
| **SGD** | Simple, well-understood | Needs tuning, slow | Research, baselines |
| **SGD + Momentum** | Faster than SGD | Still needs tuning | Computer vision |
| **Adam** | Robust, adaptive | May overfit | Quick experiments |
| **AdamW** | Best for transformers | More hyperparameters | Production LMs |
| **RMSprop** | Good for RNNs | Less popular now | Legacy code |
| **Adagrad** | Per-feature rates | LR decay too aggressive | Sparse data |

**FundamentaLLM:** Primarily uses **AdamW**.

## Training Dynamics

### Loss Curve

**Healthy training:**
```
    Loss
    ^
4.0 |╲
    | ╲___
2.0 |     ╲___
    |         ╲___
1.0 |_____________╲___
    └────────────────────→ steps
```

Smooth, monotonic decrease.

### Gradient Norms

**Monitor gradient statistics:**

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm: {total_norm:.3f}")
```

**Healthy range:** 0.1 to 10.0

**Too small:** Learning may be slow  
**Too large:** Risk of instability

### Parameter Updates

**Monitor update magnitude:**

```python
# Ratio of update to parameter magnitude
for name, param in model.named_parameters():
    if param.grad is not None:
        update_ratio = (lr * param.grad).norm() / param.norm()
        print(f"{name}: {update_ratio:.4f}")
```

**Healthy range:** 0.001 to 0.01

## Advanced Techniques

### Second-Order Methods

Use curvature information (Hessian):
- Newton's method
- L-BFGS

**Pros:** Faster convergence  
**Cons:** Very expensive for deep learning

**Not used in FundamentaLLM** (too expensive).

### Per-Layer Learning Rates

```python
# Different LR for different layers
optimizer = optim.Adam([
    {'params': model.embedding.parameters(), 'lr': 1e-3},
    {'params': model.encoder.parameters(), 'lr': 5e-4},
    {'params': model.output.parameters(), 'lr': 1e-4},
])
```

Can help with deep networks or transfer learning.

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in data_loader:
    optimizer.zero_grad()
    
    # Forward in float16
    with autocast():
        logits = model(batch['input'])
        loss = F.cross_entropy(logits, batch['target'])
    
    # Backward with scaling
    scaler.scale(loss).backward()
    
    # Unscale and step
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2x faster training
- 50% less memory
- Maintains accuracy

**FundamentaLLM:** Optional mixed precision support.

## Practical Tips

### 1. Start with Defaults

```python
# Good starting point for most models
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.1
)
```

### 2. Tune LR First

Learning rate has biggest impact. Try 1e-2, 1e-3, 1e-4.

### 3. Use Warmup for Large Models

```python
# Prevents early instability
warmup_steps = 1000
```

### 4. Monitor Gradients

```python
# Add to training loop
if step % 100 == 0:
    grad_norm = compute_grad_norm(model)
    print(f"Step {step}: grad_norm = {grad_norm:.3f}")
```

### 5. Save Optimizer State

```python
# Save
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, 'checkpoint.pt')

# Resume training with same optimizer state
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
```

## Troubleshooting

### Loss Not Decreasing

1. **Increase learning rate** (try 10x)
2. **Check gradients** are flowing
3. **Verify data** is correct

### Loss Exploding (NaN)

1. **Decrease learning rate** (try 0.1x)
2. **Add gradient clipping** (max_norm=1.0)
3. **Check for bad data** (NaN, Inf)

### Slow Convergence

1. **Increase learning rate**
2. **Add learning rate schedule** (warmup + cosine)
3. **Increase batch size**
4. **Try different optimizer** (SGD → Adam)

## Further Reading

- "An overview of gradient descent optimization algorithms" (Ruder, 2016)
- [Adam paper](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014)
- [AdamW paper](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017)
- "Deep Learning" (Goodfellow et al.) - Chapter 8

## Next Steps

- [Learning Rate Scheduling](./scheduling.md) - Advanced schedules
- [Gradient Computation](./gradients.md) - How backprop works
- [Training Guide](../guide/training.md) - Practical training
- [Hyperparameters](../guide/hyperparameters.md) - Tuning optimization
