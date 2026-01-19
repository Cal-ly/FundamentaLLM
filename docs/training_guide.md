# Training Guide

## Overview

This guide covers best practices for training transformer language models with FundamentaLLM. Whether you're training a small character-level model or scaling to larger datasets, understanding configuration, hyperparameters, and monitoring is essential.

---

## Configuration Best Practices

### Model Size Presets

**Small (fast training, limited capacity)**
```yaml
model:
  d_model: 128
  num_heads: 4
  num_layers: 2
  d_ff: 512
  dropout: 0.1
```
- **Use for:** Rapid prototyping, limited compute, proof-of-concept
- **Training time:** ~5-10 minutes on GPU (1M chars, 5 epochs)
- **Parameters:** ~500K

**Medium (balanced - default)**
```yaml
model:
  d_model: 512
  num_heads: 8
  num_layers: 6
  d_ff: 2048
  dropout: 0.1
```
- **Use for:** Most production use cases
- **Training time:** ~1-2 hours on GPU (1M chars, 10 epochs)
- **Parameters:** ~40M

**Large (high capacity, resource-intensive)**
```yaml
model:
  d_model: 768
  num_heads: 12
  num_layers: 12
  d_ff: 3072
  dropout: 0.1
```
- **Use for:** Large datasets, final production models
- **Training time:** ~4-8 hours on GPU (10M chars, 20 epochs)
- **Parameters:** ~150M

### Hyperparameter Tuning

#### Learning Rate

The most critical hyperparameter for training stability and convergence.

**Guidelines:**
- **Default starting point:** `3e-4` (0.0003)
- **Smaller models:** Can use higher LR (up to `1e-3`)
- **Larger models:** Use lower LR (down to `1e-4`)
- **Fine-tuning:** Use 10x lower than pre-training

**Learning rate finder:**
```bash
# Try different values on small subset
for lr in 1e-4 3e-4 1e-3; do
  fundamentallm train data.txt --learning-rate $lr --epochs 1 -o test_lr_${lr}
done
```

Watch for:
- Too high: Loss explodes or becomes NaN
- Too low: Loss decreases very slowly
- Just right: Steady decrease with occasional plateaus

#### Batch Size

Balances memory usage, training speed, and gradient quality.

**Guidelines:**
- **Start with 32**, then adjust based on GPU memory
- **Larger batch sizes:** Faster training, smoother gradients, more memory
- **Smaller batch sizes:** Less memory, noisier gradients, may need lower LR

**Memory-constrained training:**
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 32
```

**Rule of thumb:** Fill ~80% of GPU memory for best performance.

#### Warmup Steps

Linear warmup stabilizes early training by gradually increasing learning rate.

**Guidelines:**
- **Default:** 100 steps (first 5-10% of training)
- **Longer warmup for larger models:** 500-1000 steps
- **No warmup:** Can work for small models/datasets

```yaml
training:
  warmup_steps: 100  # Steps to ramp up to full LR
```

#### Learning Rate Schedule

Controls how LR changes during training.

**Cosine Annealing (recommended):**
```yaml
training:
  scheduler: "cosine"
  min_lr_ratio: 0.1  # Final LR = 10% of initial
```
- Smoothly decreases LR following cosine curve
- Good for most use cases

**Linear Decay:**
```yaml
training:
  scheduler: "linear"
  min_lr_ratio: 0.01  # Final LR = 1% of initial
```
- Linear decrease from max to min
- Alternative to cosine

**Step Decay:**
```yaml
training:
  scheduler: "step"
  step_size: 3  # Drop every 3 epochs
  gamma: 0.5    # Multiply LR by 0.5
```
- Discrete drops at fixed intervals
- Can cause training instability

---

## Data Preparation

### Text Encoding

**Best practices:**
- Always use **UTF-8** encoding
- Remove or handle control characters appropriately
- Mixed case is fine (no need for lowercasing)
- Preserve punctuation and whitespace

**Example preprocessing:**
```python
# Clean text data
with open('raw_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Optional: remove excessive whitespace
text = ' '.join(text.split())

# Save cleaned data
with open('clean_data.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

### Dataset Size Guidelines

| Size Category | Character Count | Use Case | Expected Quality |
|---------------|-----------------|----------|------------------|
| **Tiny** | 10K-50K | Proof of concept, testing | Coherent short sequences |
| **Small** | 100K-1M | Quick experiments | Reasonable quality |
| **Medium** | 1M-10M | Production prototypes | Good quality |
| **Large** | 10M-100M | Production models | High quality |
| **Very Large** | 100M+ | Research-grade | State-of-the-art |

**Minimum viable:** 10K characters will train, but expect limited quality.

### Train/Validation Split

**Default split:** 90% train / 10% validation

```yaml
training:
  validation_split: 0.1  # 10% for validation
```

**Important considerations:**
- Validation data must be representative of training data
- Token-level splitting avoids data leakage
- Validation loss guides early stopping and hyperparameter tuning

---

## Training Monitoring

### Key Metrics

#### Training Loss
- Should decrease monotonically (with minor fluctuations)
- Rapid initial decrease, then slower convergence
- If plateaus early, increase LR or check data

#### Validation Loss
- Should decrease but may plateau before training loss
- Gap between train/val indicates overfitting
- Use for early stopping

#### Perplexity
- Interpretable measure: `perplexity = exp(loss)`
- Lower is better
- Rule of thumb: Perplexity < 50 for decent models

**Example output:**
```
Epoch 1/10: 100%|██████████| 156/156 [00:45<00:00,  3.42it/s]
  train_loss=4.123, val_loss=3.987, perplexity=53.45
  
Epoch 5/10: 100%|██████████| 156/156 [00:43<00:00,  3.58it/s]
  train_loss=2.145, val_loss=2.234, perplexity=9.34
```

### Early Stopping

Automatically stops training when validation loss stops improving.

```yaml
training:
  early_stopping_patience: 5  # Stop if no improvement for 5 epochs
```

**When to use:**
- Long training runs (to save compute)
- Risk of overfitting
- Hyperparameter search

**When to skip:**
- Short training (<10 epochs)
- Small datasets (more noise in validation)

### Warning Signs

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Loss becomes NaN** | LR too high, gradient explosion | Reduce LR, enable gradient clipping |
| **Loss doesn't decrease** | LR too low, bad initialization | Increase LR, check data loading |
| **Val loss increases while train decreases** | Overfitting | Increase dropout, add regularization |
| **Loss spikes randomly** | Bad batch, corrupt data | Check data quality, reduce LR |
| **Slow convergence** | Model too small, LR too low | Increase capacity or LR |

---

## Hardware Considerations

### GPU Training (Recommended)

**Requirements:**
- CUDA-capable GPU with 4GB+ VRAM
- CUDA toolkit and PyTorch with GPU support

**Enable:**
```bash
fundamentallm train data.txt --device cuda
```

**Benefits:**
- 10-50x faster than CPU
- Enables mixed precision training
- Supports larger batch sizes

**Check GPU usage:**
```bash
nvidia-smi  # Monitor GPU memory and utilization
```

### CPU Training

**When to use:**
- No GPU available
- Debugging (CPU errors are clearer)
- Very small models

**Optimize for CPU:**
```bash
fundamentallm train data.txt \
    --device cpu \
    --config configs/small.yaml \
    --batch-size 16
```

**Tips:**
- Use smaller models (small.yaml)
- Reduce batch size
- Expect 10-50x slower training

### Mixed Precision (AMP)

Automatic mixed precision training uses float16 for speed, float32 for stability.

**Benefits:**
- ~2x faster training
- ~50% less memory
- Minimal accuracy loss

**Enabled by default on GPU. Disable if issues:**
```yaml
training:
  use_mixed_precision: false
```

---

## Advanced Techniques

### Gradient Accumulation

Simulate larger batch sizes without extra memory.

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
```

**How it works:**
1. Forward pass on small batch (size=8)
2. Accumulate gradients (don't update yet)
3. Repeat 4 times
4. Update weights with accumulated gradients

**Use when:**
- GPU memory limited
- Want large effective batch size
- Training with small GPUs

### Gradient Clipping

Prevents gradient explosion by capping gradient norm.

```yaml
training:
  gradient_clip_norm: 1.0  # Clip to max norm of 1.0
```

**When to use:**
- Training becomes unstable
- Loss occasionally spikes to NaN
- Working with RNNs or very deep models

### Checkpointing Strategy

Save model at different points for recovery and analysis.

**Options:**
```yaml
training:
  save_every_n_epochs: 5     # Save every 5 epochs
  keep_last_n: 3             # Keep only last 3 checkpoints
  save_best: true            # Save best validation loss
```

**Example output:**
```
checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── best_model.pt
└── final_model.pt
```

### Learning Rate Finder

Automatically find optimal learning rate before training.

**Manual method:**
```bash
# Test multiple learning rates
for lr in 1e-5 1e-4 3e-4 1e-3 3e-3; do
  fundamentallm train data.txt \
    --learning-rate $lr \
    --epochs 2 \
    -o test_lr_${lr}
done

# Compare validation loss, choose best LR
```

**Plot results:**
```python
import matplotlib.pyplot as plt

lrs = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
losses = [3.2, 2.1, 1.8, 1.9, 5.2]  # From training logs

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Loss')
plt.title('Learning Rate Finder')
plt.show()
```

---

## Common Training Issues

### Issue: OOM (Out of Memory)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**Solutions (in order of impact):**
1. **Reduce batch size:** `--batch-size 16` → `--batch-size 8`
2. **Use smaller model:** `--config configs/small.yaml`
3. **Reduce sequence length:** `--sequence-length 128`
4. **Enable gradient accumulation:** See above
5. **Clear cache:** Add `torch.cuda.empty_cache()` in training loop

### Issue: Training Too Slow

**Symptoms:**
- <1 iteration/second on GPU
- GPU utilization <50%

**Solutions:**
1. **Check GPU usage:** `nvidia-smi` (should be >80%)
2. **Increase batch size:** Fill GPU memory
3. **Reduce num_workers:** Try `num_workers=0` to debug
4. **Check data loading:** Profile with `torch.utils.bottleneck`
5. **Verify CUDA installation:** `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Poor Generation Quality

**Symptoms:**
- Repetitive text
- Nonsensical output
- Only outputs common phrases

**Solutions:**
1. **Train longer:** More epochs or data
2. **Check training loss:** Should be <2.0 for decent quality
3. **Verify data quality:** Check for corrupt/repetitive data
4. **Tune generation params:**
   - Increase temperature: `--temperature 1.0`
   - Enable nucleus sampling: `--top-p 0.95`
   - Reduce top-k: `--top-k 40`

### Issue: Loss Not Decreasing

**Symptoms:**
- Loss plateaus at high value (>4.0)
- No improvement after many epochs

**Solutions:**
1. **Increase learning rate:** Try 10x higher
2. **Check data loading:** Verify batches are different
3. **Verify gradients:** Check for all-zero gradients
4. **Reduce model complexity:** Try smaller model first
5. **Check data quality:** Ensure sufficient diversity

---

## Reproducibility

For reproducible results across runs:

```yaml
training:
  seed: 42
  deterministic: true
```

**Also ensure:**
- Same dataset (no randomization between runs)
- Same hardware (different GPUs can vary slightly)
- Same software versions (PyTorch version matters)
- Same environment (CUDA, cuDNN versions)

**Note:** Full determinism on GPU is difficult. Expect small variations (<1% relative difference) even with fixed seeds.

---

## Performance Benchmarks

Example training times on NVIDIA A100 GPU:

| Config | Data Size | Batch Size | Epochs | Time | Final Loss | Perplexity |
|--------|-----------|------------|--------|------|-----------|-----------|
| small | 100K chars | 32 | 5 | 1 min | 1.45 | 4.26 |
| small | 1M chars | 32 | 5 | 2 min | 1.23 | 3.42 |
| medium | 1M chars | 64 | 10 | 15 min | 0.95 | 2.59 |
| medium | 10M chars | 64 | 20 | 2 hr | 0.87 | 2.39 |
| large | 10M chars | 32 | 20 | 8 hr | 0.65 | 1.92 |

**Your results may vary based on:**
- GPU model
- Data characteristics
- Exact hyperparameters
- PyTorch version

---

## Tips & Tricks

### Quick Iteration Cycle

```bash
# 1. Start with tiny model and data subset
head -c 10000 data.txt > data_tiny.txt
fundamentallm train data_tiny.txt --config configs/small.yaml --epochs 2

# 2. Verify pipeline works end-to-end
fundamentallm generate checkpoints/final_model.pt --prompt "Test"

# 3. Scale up gradually
fundamentallm train data.txt --config configs/medium.yaml --epochs 10
```

### Monitor Training Remotely

```bash
# Run in tmux/screen for long training
tmux new -s training
fundamentallm train data.txt --epochs 50 | tee training.log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training

# Monitor logs from another terminal
tail -f training.log
```

### Hyperparameter Search

```bash
#!/bin/bash
# Simple grid search
for lr in 1e-4 3e-4 1e-3; do
  for bs in 16 32 64; do
    fundamentallm train data.txt \
      --learning-rate $lr \
      --batch-size $bs \
      -o results/lr${lr}_bs${bs}
  done
done

# Compare results
ls -lh results/*/final_model.pt
```

---

## Resources

### Papers
- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original Transformer paper
- **"Language Models are Unsupervised Multitask Learners"** (GPT-2 paper)
- **"Deep Learning"** (Goodfellow et al., 2016) - Comprehensive textbook

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)

### Tools
- [Weights & Biases](https://wandb.ai/) - Experiment tracking (future integration)
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) - Visualization
- [Netron](https://github.com/lutzroeder/netron) - Model visualization

---

## Next Steps

- **[Architecture Guide](architecture.md)** - Understand the implementation
- **[Getting Started](getting_started.md)** - Quick start tutorial
- **[Example Notebooks](notebooks/)** - Interactive examples
- **[API Reference](api_reference.md)** - Complete API docs

---

**Happy Training! 🚀**
