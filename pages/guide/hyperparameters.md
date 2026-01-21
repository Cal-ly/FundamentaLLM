# Hyperparameter Tuning

Master the art of hyperparameter optimization for better model performance.

## What are Hyperparameters?

**Hyperparameters** are settings you choose before training that control:
- Model architecture (size, depth)
- Training process (learning rate, batch size)
- Regularization (dropout, weight decay)

Unlike model parameters (weights), hyperparameters are not learned from data.

## Quick Reference

| Category | Parameter | Typical Range | Default |
|----------|-----------|---------------|---------|
| **Model** | `--model-dim` | 64-1024 | 128 |
| | `--num-layers` | 2-24 | 6 |
| | `--num-heads` | 2-16 | 8 |
| | `--dropout` | 0.0-0.3 | 0.1 |
| **Training** | `--learning-rate` | 1e-5 to 1e-2 | 1e-3 |
| | `--batch-size` | 8-128 | 32 |
| | `--epochs` | 5-100 | 20 |
| **Sequence** | `--max-seq-len` | 128-2048 | 256 |

## Model Architecture

### Hidden Dimension (`--model-dim`)

**What it controls:** Size of embedding and hidden layers.

**Effect:**
- Larger → More capacity, better performance, slower training
- Smaller → Faster, less memory, may underfit

**How to tune:**
```bash
# Start with default
--model-dim 128

# If underfitting (high train and val loss)
--model-dim 256

# If overfitting (low train loss, high val loss)
--model-dim 64

# For production
--model-dim 512
```

**Rule of thumb:** Double until you hit memory limits or overfitting.

### Number of Layers (`--num-layers`)

**What it controls:** Depth of the transformer.

**Effect:**
- More layers → Better at complex patterns, slower
- Fewer layers → Faster, may miss complex patterns

**How to tune:**
```bash
# Quick experiments
--num-layers 2-4

# Standard models
--num-layers 6-8

# Large models
--num-layers 12-24
```

**Warning:** Very deep models (>12 layers) need careful initialization and learning rates.

### Attention Heads (`--num-heads`)

**What it controls:** Number of parallel attention mechanisms.

**Effect:**
- More heads → Capture different patterns
- Each head has smaller dimension (d_model / num_heads)

**Constraint:** `num_heads` must divide `model_dim` evenly.

**How to tune:**
```bash
# model_dim=128: use 2, 4, 8
# model_dim=256: use 4, 8, 16
# model_dim=512: use 8, 16

# General rule: num_heads = model_dim / 64
```

**Sweet spot:** 8 heads for most models.

### Dropout (`--dropout`)

**What it controls:** Regularization strength.

**Effect:**
- Higher → Less overfitting, may underfit
- Lower → May overfit on small datasets

**How to tune:**
```bash
# No regularization
--dropout 0.0

# Light (default)
--dropout 0.1

# Medium
--dropout 0.2

# Heavy (small dataset)
--dropout 0.3
```

**When to increase:**
- Small dataset
- Signs of overfitting
- Large model

## Training Configuration

### Learning Rate (`--learning-rate`)

**Most important hyperparameter!**

**Effect:**
- Too high → Loss explodes or NaN
- Too low → Training is slow or stuck
- Just right → Smooth, steady decrease

**How to tune:**
```bash
# Start high, reduce if unstable
--learning-rate 0.01  → Try first

# If loss explodes
--learning-rate 0.001  → Standard

# If training is slow
--learning-rate 0.0001 → Conservative

# If still slow
--learning-rate 0.00001 → Very conservative
```

**Strategy:** Start at 1e-3, multiply by 10 or divide by 10 based on results.

**Learning rate scheduling:**
```bash
# Warmup then decay
--lr-schedule cosine --warmup-steps 1000

# Step decay
--lr-schedule step --lr-decay 0.1 --lr-steps 10000,20000
```

### Batch Size (`--batch-size`)

**What it controls:** Number of samples processed together.

**Effect:**
- Larger → More stable gradients, faster on GPU, more memory
- Smaller → Noisier gradients, less memory

**How to tune:**
```bash
# Memory constrained
--batch-size 8-16

# Balanced
--batch-size 32

# High-end GPU
--batch-size 64-128
```

**Rule:** Increase until you hit memory limit, then back off 20-30%.

**Trade-off:**
```
Small batch (8):  Noisy but more updates per epoch
Large batch (64): Stable but fewer updates per epoch
```

### Number of Epochs (`--epochs`)

**What it controls:** Complete passes through dataset.

**Effect:**
- More epochs → Better training, risk overfitting
- Fewer epochs → Faster, may not converge

**How to tune:**
```bash
# Quick test
--epochs 5

# Standard
--epochs 20-30

# Large dataset
--epochs 50-100
```

**Use early stopping** to prevent overfitting:
```bash
--early-stopping --patience 5
```

## Learning Rate Schedules

### Constant (Default)

```bash
# No schedule
--learning-rate 0.001
```

Simple, works well for small experiments.

### Warmup + Cosine Decay

```bash
--lr-schedule cosine \
--warmup-steps 1000 \
--learning-rate 0.001
```

**Why:** Gradual warmup prevents instability, cosine decay helps convergence.

**Best for:** Medium to large models.

### Step Decay

```bash
--lr-schedule step \
--lr-decay 0.1 \
--lr-steps 10000,20000
```

Reduces LR at specific steps. Simple but effective.

### Exponential Decay

```bash
--lr-schedule exponential \
--lr-decay 0.96
```

Continuous smooth decay.

## Sequence Configuration

### Max Sequence Length (`--max-seq-len`)

**What it controls:** Maximum tokens per training sequence.

**Effect:**
- Longer → More context, much more memory (O(n²))
- Shorter → Less memory, less context

**How to tune:**
```bash
# Short sequences (fast)
--max-seq-len 128

# Standard
--max-seq-len 256

# Long context
--max-seq-len 512

# Very long (requires lots of memory)
--max-seq-len 1024
```

**Memory usage:** Quadratic in sequence length!
- 256 → 512: 4× more memory
- 256 → 1024: 16× more memory

## Tuning Strategy

### 1. Start with Defaults

```bash
fundamentallm train data.txt \ \
    --output-dir baseline \ \
    --model-dim 128 \ \
    --num-layers 6 \ \
    --num-heads 8 \ \
    --learning-rate 0.001 \ \
    --batch-size 32 \ \
    --epochs 20
```

### 2. Grid Search (Systematic)

```bash
for lr in 0.0001 0.001 0.01; do
    for dim in 128 256 512; do
        fundamentallm train data.txt \ \
    --output-dir exp_lr${lr}_dim${dim} \ \
    --learning-rate $lr \ \
    --model-dim $dim
    done
done
```

### 3. Random Search (Efficient)

```python
import random

configs = []
for _ in range(10):
    config = {
        'lr': random.choice([1e-4, 5e-4, 1e-3, 5e-3]),
        'dim': random.choice([128, 256, 512]),
        'layers': random.choice([4, 6, 8]),
        'dropout': random.uniform(0.0, 0.3)
    }
    configs.append(config)

# Train each config
```

**Why random?** Often finds better configs faster than grid search.

### 4. Progressive Tuning

**Phase 1:** Coarse search (3-5 epochs)
```bash
--epochs 5  # Quick feedback
```

**Phase 2:** Fine-tune best configs (20+ epochs)
```bash
--epochs 30  # Full training
```

## What to Optimize For

### Training Loss

Measures fit to training data. Should decrease smoothly.

**Good:**
```
Epoch 1: 4.234
Epoch 5: 2.456
Epoch 10: 1.892
Epoch 20: 1.234
```

**Bad (underfit):**
```
Epoch 1: 4.234
...
Epoch 20: 3.891  ← Still high
```

**Bad (overfit):**
```
Train: 0.234
Val:   2.891   ← Huge gap
```

### Validation Loss

Most important metric! Measures generalization.

**Target:** Minimize validation loss while keeping train/val gap reasonable.

### Perplexity

Lower is better. Intuitive measure of model quality.

**Character-level targets:**
- Random: ~256
- Weak model: 30-100
- Decent model: 10-30
- Good model: 3-10
- Excellent model: 1.5-3

### Generation Quality

**Qualitative assessment:** Does generated text look reasonable?

```bash
# Test after each experiment
fundamentallm generate model.pt \ \
    --prompt "Test prompt" \ \
    --temperature 0.8
```

## Common Patterns

### Small Dataset

```bash
--model-dim 128 \
--num-layers 4 \
--dropout 0.2 \
--learning-rate 0.001 \
--epochs 30
```

More regularization, smaller model, train longer.

### Large Dataset

```bash
--model-dim 512 \
--num-layers 12 \
--dropout 0.1 \
--learning-rate 0.0003 \
--batch-size 64 \
--epochs 20
```

Larger model, less regularization, fewer epochs needed.

### Limited Compute

```bash
--model-dim 128 \
--num-layers 4 \
--batch-size 16 \
--max-seq-len 128
```

Small and fast.

### Best Quality (Unlimited Compute)

```bash
--model-dim 768 \
--num-layers 16 \
--num-heads 12 \
--batch-size 64 \
--max-seq-len 512 \
--mixed-precision \
--lr-schedule cosine \
--warmup-steps 2000
```

## Debugging Tips

### Loss is NaN

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Fixes:**
```bash
--learning-rate 0.0001  # Reduce LR
--gradient-clip 1.0     # Clip gradients
--mixed-precision false # Disable if using
```

### Loss Not Decreasing

**Causes:**
- Learning rate too low
- Model too small
- Data issues

**Fixes:**
```bash
--learning-rate 0.01    # Increase LR
--model-dim 512         # Bigger model
--num-layers 8          # Deeper model
```

### Overfitting

**Signs:**
- Train loss << Val loss
- Val loss increasing

**Fixes:**
```bash
--dropout 0.3          # More regularization
--model-dim 256        # Smaller model
--epochs 15            # Train less
--early-stopping       # Stop automatically
```

### Out of Memory

**Fixes:**
```bash
--batch-size 8         # Reduce batch
--max-seq-len 128      # Shorter sequences
--model-dim 256        # Smaller model
--gradient-accumulation 4  # Simulate larger batch
```

## Automated Hyperparameter Search

### Using Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dim = trial.suggest_categorical('dim', [128, 256, 512])
    layers = trial.suggest_int('layers', 4, 12)
    
    # Train model
    val_loss = train_model(lr=lr, dim=dim, layers=layers)
    
    return val_loss

# Optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best params: {study.best_params}")
```

### Using WandB Sweeps

```yaml
# sweep.yaml
program: train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  model_dim:
    values: [128, 256, 512]
  num_layers:
    values: [4, 6, 8, 12]
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

## Practical Workflow

### Day 1: Baseline

```bash
# Single run with defaults
fundamentallm train data.txt --output-dir baseline
```

### Day 2: Learning Rate

```bash
# Try 5 learning rates
for lr in 0.0001 0.0005 0.001 0.005 0.01; do
    fundamentallm train data.txt \ \
    --output-dir exp_lr_$lr \ \
    --learning-rate $lr \ \
    --epochs 10
done
```

### Day 3: Model Size

```bash
# Best LR from day 2, try model sizes
best_lr=0.001
for dim in 128 256 512; do
    fundamentallm train data.txt \ \
    --output-dir exp_dim_$dim \ \
    --learning-rate $best_lr \ \
    --model-dim $dim \ \
    --epochs 20
done
```

### Day 4: Fine-tune

```bash
# Best config, longer training
fundamentallm train data.txt \ \
    --output-dir final \ \
    --learning-rate 0.001 \ \
    --model-dim 256 \ \
    --num-layers 8 \ \
    --epochs 50 \ \
    --early-stopping
```

## Further Reading

- [Training Guide](./training.md) - Complete training documentation
- [Troubleshooting](./troubleshooting.md) - Common issues
- "Practical recommendations for gradient-based training" (Bengio, 2012)
- [Google's Tuning Playbook](https://github.com/google-research/tuning_playbook)

## Next Steps

- [Training Module](../modules/training.md) - Implementation details
- [Data Preparation](./data-prep.md) - Better data = better models
- [Evaluation](./evaluation.md) - Measure improvements
