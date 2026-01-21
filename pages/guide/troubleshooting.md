# Troubleshooting

Solutions to common problems when training and using language models.

## Parameter Validation Reference

FundamentaLLM validates parameters before training to catch issues early. Here are the limits and constraints:

### Hard Limits (Will Block Training)

| Parameter | Constraint | Example |
|-----------|-----------|---------|
| `--model-dim` | Must be ≥ 64 | `--model-dim 128` ✅ |
| `--num-layers` | Must be < 48 | `--num-layers 24` ✅ |
| `--num-heads` | Must divide `model_dim` | `--num-heads 8` with `model-dim 256` ✅ |
| `--num-heads` | `model_dim / num_heads` ≥ 8 | `model_dim=512, num_heads=8` (64 per head) ✅ |
| `--dropout` | Must be in [0.0, 1.0] | `--dropout 0.1` ✅ |
| `--val-split` | Must be in (0, 1) | `--val-split 0.2` ✅ |

### Warning Thresholds (Will Log But Allow)

| Parameter | Warn If | Recommended |
|-----------|---------|-------------|
| `--learning-rate` | > 0.1 or < 1e-6 | 1e-4 to 1e-3 |
| `--batch-size` | > 2048 | 8-128 |
| `--max-seq-len` | > 8192 | 128-2048 |
| `--epochs` | > 10000 | 5-100 |
| `--gradient-clip` | > 10 | 1.0-5.0 |
| `--model-dim` | < 64 | ≥ 64 |

### Auto-Fix Behavior

With `--auto-fix-config` (enabled by default), FundamentaLLM automatically fixes validation conflicts:

**Example: num_heads too high**
```bash
# You request:
--model-dim 512 --num-heads 16

# System detects: 512/16 = 32 per head (too small, need ≥8 for safety)
# Auto-fixes to: num_heads = 8 (512/8 = 64 per head)
# Logs: "WARNING: num_heads 16 → 8 (head_dim = 64)"
```

**To disable auto-fix:**
```bash
--auto-fix-config false
```

## Quick Diagnostics

```bash
# Check installation
fundamentallm --version

# Verify data
head -n 10 my_data.txt
file -i my_data.txt  # Check encoding

# Test small model
fundamentallm train small_sample.txt --epochs 2 --model-dim 64
```

## Training Issues

### Loss is NaN

**Symptoms:**
```
Epoch 1, Step 50: loss=3.245
Epoch 1, Step 100: loss=2.891
Epoch 1, Step 150: loss=nan
```

**Causes & Fixes:**

#### 1. Learning Rate Too High

```bash
# Problem
--learning-rate 0.1  # Too high!

# Fix: Reduce by 10x
--learning-rate 0.01
# Or even more conservative
--learning-rate 0.001
```

#### 2. Gradient Explosion

```bash
# Add gradient clipping
--gradient-clip 1.0
```

#### 3. Numerical Instability

```bash
# If using mixed precision, try disabling
--mixed-precision false

# Or adjust scaling
--mixed-precision true --loss-scale 128
```

#### 4. Bad Initialization

```python
# Check initialization isn't broken
# Weights should be small, non-zero
print(model.embedding.weight.mean())  # Should be ~0
print(model.embedding.weight.std())   # Should be ~0.01-0.1
```

**Quick fix pipeline:**
```bash
# Start here
--learning-rate 0.0001 --gradient-clip 1.0

# If still NaN, reduce further
--learning-rate 0.00001 --gradient-clip 0.5

# If still issues, check data for corrupted values
```

### Loss Not Decreasing

**Symptoms:**
```
Epoch 1: loss=4.234
Epoch 5: loss=4.189
Epoch 10: loss=4.156
```

**Causes & Fixes:**

#### 1. Learning Rate Too Low

```bash
# Problem
--learning-rate 0.00001  # Too conservative

# Fix: Increase
--learning-rate 0.001
# Or try schedule
--lr-schedule cosine --warmup-steps 1000
```

#### 2. Model Too Small

```bash
# Problem
--model-dim 32 --num-layers 2  # Too small

# Fix: Increase capacity
--model-dim 256 --num-layers 6
```

#### 3. Data Issues

```bash
# Check data quality
wc -l train.txt  # Enough data?
file -i train.txt  # Correct encoding?
head train.txt  # Makes sense?

# Need at least ~100KB for meaningful learning
du -h train.txt
```

#### 4. Wrong Tokenization

```python
# Verify tokenizer works
from fundamentallm.data.tokenizers import CharTokenizer

tokenizer = CharTokenizer()
tokens = tokenizer.encode("Hello")
text = tokenizer.decode(tokens)
print(text)  # Should print "Hello"
```

### Training is Slow

**Symptoms:** Minutes per epoch on small data.

**Causes & Fixes:**

#### 1. Large Sequence Length

```bash
# Problem
--max-seq-len 2048  # Memory intensive

# Fix: Reduce
--max-seq-len 256  # 16x faster!
```

#### 2. Small Batch Size

```bash
# Problem
--batch-size 4  # Too many iterations

# Fix: Increase (if memory allows)
--batch-size 32

# Or use gradient accumulation
--batch-size 8 --gradient-accumulation 4  # Effective batch: 32
```

#### 3. No GPU Acceleration

```bash
# Check GPU
nvidia-smi  # Should show GPU

# If available but not used
--device cuda

# If error, check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Inefficient Data Loading

```bash
# Use more workers
--num-workers 4

# Reduce logging
--log-interval 1000  # Log less frequently
```

### Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.35 GiB
```

**Fixes (in order of preference):**

#### 1. Reduce Batch Size

```bash
--batch-size 16  # Instead of 32
```

#### 2. Reduce Sequence Length

```bash
--max-seq-len 128  # Instead of 512
```

#### 3. Use Gradient Accumulation

```bash
--batch-size 8 --gradient-accumulation 4
# Effective batch size: 32, but memory of 8
```

#### 4. Smaller Model

```bash
--model-dim 256  # Instead of 512
--num-layers 4   # Instead of 8
```

#### 5. Mixed Precision

```bash
--mixed-precision true
# Uses float16 instead of float32
# ~50% memory savings
```

#### 6. Reduce Number of Workers

```bash
--num-workers 2  # Instead of 4
```

**Memory calculator:**
```python
# Estimate memory usage (rough)
batch_size = 32
seq_len = 256
model_dim = 512
num_layers = 6

# Attention: O(batch * seq^2 * heads)
attention_mem = batch_size * seq_len * seq_len * 8 / 1e9
print(f"Attention: ~{attention_mem:.2f} GB")

# Activations: O(batch * seq * dim * layers)
activation_mem = batch_size * seq_len * model_dim * num_layers * 4 / 1e9
print(f"Activations: ~{activation_mem:.2f} GB")
```

### Overfitting

**Symptoms:**
```
Train loss: 0.234
Val loss:   2.891  # Huge gap!
```

**Fixes:**

#### 1. Add Regularization

```bash
# Increase dropout
--dropout 0.2  # Or 0.3 for aggressive

# Add weight decay
--weight-decay 0.1
```

#### 2. More Data

```bash
# Get more training data
cat file1.txt file2.txt file3.txt > all_data.txt
fundamentallm train all_data.txt
```

#### 3. Smaller Model

```bash
--model-dim 128  # Instead of 512
--num-layers 4   # Instead of 8
```

#### 4. Early Stopping

```bash
--early-stopping --patience 5
# Stops when val loss doesn't improve for 5 epochs
```

#### 5. Data Augmentation

```python
# Add noise to input
# Mix different sources
# Use different splits
```

### Underfitting

**Symptoms:**
```
Train loss: 3.234
Val loss:   3.289  # Both high, small gap
```

**Fixes:**

#### 1. Larger Model

```bash
--model-dim 512   # Instead of 128
--num-layers 12   # Instead of 4
```

#### 2. Train Longer

```bash
--epochs 50  # Instead of 20
```

#### 3. Higher Learning Rate

```bash
--learning-rate 0.003  # Instead of 0.001
```

#### 4. Remove Regularization

```bash
--dropout 0.0  # No dropout
# Remove weight decay
```

#### 5. Better Data

```bash
# Clean data
# More diverse data
# Relevant to task
```

## Generation Issues

### Repetitive Output

**Symptoms:**
```
"The the the the the..."
"Hello hello hello..."
```

**Fixes:**

#### 1. Adjust Temperature

```bash
# Increase randomness
--temperature 1.2  # Instead of 0.8
```

#### 2. Use Top-k Sampling

```bash
--top-k 50
# Only sample from top 50 tokens
```

#### 3. Use Top-p (Nucleus) Sampling

```bash
--top-p 0.9
# Sample from top tokens that sum to 90%
```

#### 4. Repetition Penalty

```bash
--repetition-penalty 1.2
# Penalizes recently used tokens
```

### Nonsensical Output

**Symptoms:**
```
"asdjkfh aslkdfj lkjsdf..."
"The cat ran into the quantum entanglement..."
```

**Fixes:**

#### 1. Lower Temperature

```bash
--temperature 0.5  # More conservative
```

#### 2. Use Greedy Decoding

```bash
--temperature 0.0
# Always pick most likely token
```

#### 3. Constrain Sampling

```bash
--top-k 10  # Only top 10 choices
--top-p 0.8  # Or nucleus with 80%
```

#### 4. Train Longer

Model may not be converged yet.

### Too Conservative

**Symptoms:**
```
# Always generates same thing
"The the the" → Always continues with same text
```

**Fixes:**

#### 1. Increase Temperature

```bash
--temperature 1.0  # Or higher
```

#### 2. Remove Top-k/Top-p

```bash
# Don't constrain sampling
# Remove --top-k and --top-p flags
```

#### 3. Use Different Seeds

```bash
--seed 42   # Try different seeds
--seed 123
--seed 999
```

## Installation Issues

### Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'fundamentallm'
```

**Fixes:**

#### 1. Install in Development Mode

```bash
cd FundamentaLLM
pip install -e .
```

#### 2. Check Python Environment

```bash
# Verify correct environment
which python
pip list | grep fundamentallm
```

#### 3. Reinstall

```bash
pip uninstall fundamentallm
pip install -e .
```

### CUDA Errors

**Symptom:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Fixes:**

#### 1. Check CUDA Installation

```bash
nvidia-smi
nvcc --version
```

#### 2. Reinstall PyTorch

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Use CPU

```bash
--device cpu
```

### Version Conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Fixes:**

#### 1. Create Fresh Environment

```bash
conda create -n fundamentallm python=3.9
conda activate fundamentallm
pip install -e .
```

#### 2. Update Dependencies

```bash
pip install --upgrade pip
pip install -e . --upgrade
```

## Data Issues

### Encoding Errors

**Symptom:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**Fixes:**

#### 1. Convert to UTF-8

```bash
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

#### 2. Ignore Errors

```python
# In code
with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
```

#### 3. Check File

```bash
file -i my_data.txt
# Should show: charset=utf-8
```

### Empty Data

**Symptom:**
```
ValueError: Dataset is empty
```

**Fixes:**

#### 1. Check File

```bash
wc -l my_data.txt  # Should show lines
du -h my_data.txt  # Should show size > 0
```

#### 2. Check Path

```bash
ls -lh my_data.txt  # File exists?
cat my_data.txt | head  # Has content?
```

### Corrupted Data

**Symptom:** Loss is weird or training behaves strangely.

**Fixes:**

#### 1. Analyze Data

```python
with open('my_data.txt', 'rb') as f:
    data = f.read()
    
# Check for null bytes
if b'\x00' in data:
    print("Found null bytes!")
    
# Check for control characters
weird = sum(1 for b in data if b < 32 and b not in (9, 10, 13))
print(f"Control characters: {weird}")
```

#### 2. Clean Data

```python
# Remove null bytes and weird characters
with open('input.txt', 'rb') as f:
    data = f.read()

clean = data.replace(b'\x00', b'')
clean = bytes(b for b in clean if b >= 32 or b in (9, 10, 13))

with open('output.txt', 'wb') as f:
    f.write(clean)
```

## Performance Issues

### Slow Generation

**Symptom:** Takes many seconds to generate 100 tokens.

**Fixes:**

#### 1. Use GPU

```bash
--device cuda
```

#### 2. Batch Generation

```python
# Generate multiple at once
model.generate(prompts=['prompt1', 'prompt2', 'prompt3'])
```

#### 3. Cache Key-Value

If supported, use KV caching for autoregressive generation.

#### 4. Smaller Model

Use smaller model for inference:
```bash
--model-dim 256  # Instead of 1024
```

### High Memory Usage

**Symptom:** System runs out of RAM during training.

**Fixes:**

#### 1. Reduce Batch Size

```bash
--batch-size 16
```

#### 2. Streaming Data

For very large datasets, use streaming mode (automatic in FundamentaLLM).

#### 3. Clear Cache

```python
import torch
torch.cuda.empty_cache()
```

## Common Error Messages

### "RuntimeError: Expected all tensors to be on the same device"

**Fix:**
```bash
# Ensure consistent device
--device cuda  # Or --device cpu for all
```

### "ValueError: num_heads must divide d_model"

**What it means:** The number of attention heads must divide evenly into the model dimension.

**Why it matters:** Each attention head processes a portion of `d_model`. If they don't divide evenly, heads get different dimensions, breaking the architecture.

**Example problems:**
```bash
# ❌ Bad: 200 / 8 = 25 (uneven, causes error)
--model-dim 200 --num-heads 8

# ✅ Good: 256 / 8 = 32 (each head gets 32 dimensions)
--model-dim 256 --num-heads 8

# ✅ Good: 512 / 4 = 128 (each head gets 128 dimensions)
--model-dim 512 --num-heads 4
```

**Auto-fix behavior:** With `--auto-fix-config` (default), FundamentaLLM automatically fixes this:
```bash
# This gets auto-corrected to num_heads=8
fundamentallm train data.txt \ \
    --model-dim 512 \ \
    --num-heads 16  # Too high! Would cause head_dim < 8

# Logs:
# WARNING: num_heads (16) too high relative to d_model (512)
# Auto-fixing: num_heads 16 → 8 (head_dim = 512/8 = 64)
```

**Valid divisors by model dimension:**
```bash
model-dim 128:  num_heads can be: 1, 2, 4, 8
model-dim 256:  num_heads can be: 1, 2, 4, 8, 16
model-dim 512:  num_heads can be: 1, 2, 4, 8, 16
model-dim 1024: num_heads can be: 1, 2, 4, 8, 16
```

**Rule of thumb:** Head dimension should be ≥ 8
```
head_dimension = model_dim / num_heads

Good:  512 / 8 = 64   ✅
Good:  256 / 4 = 64   ✅
Bad:   128 / 32 = 4   ❌ (head_dim too small)
Bad:   200 / 8 = 25   ❌ (doesn't divide evenly)
```

### "RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)"

**Cause:** Shape mismatch, often due to wrong sequence length.

**Fix:**
```bash
# Check max_seq_len matches training
--max-seq-len 256  # Use same as training
```

## Debugging Tips

### 1. Start Small

```bash
# Tiny model, tiny data, few epochs
fundamentallm train small.txt \ \
    --model-dim 64 \ \
    --num-layers 2 \ \
    --batch-size 8 \ \
    --epochs 2

# If this works, scale up incrementally
```

### 2. Check Each Component

```python
# Test tokenizer
from fundamentallm.data.tokenizers import CharTokenizer
tok = CharTokenizer()
assert tok.decode(tok.encode("test")) == "test"

# Test data loading
from fundamentallm.data import get_dataloader
loader = get_dataloader('data.txt', batch_size=4)
batch = next(iter(loader))
print(batch.shape)

# Test model forward pass
model = create_model()
output = model(batch)
print(output.shape)
```

### 3. Enable Debugging

```bash
# Verbose logging
--log-level DEBUG

# Detect anomalies (slower but catches NaN early)
--detect-anomaly true
```

### 4. Use Checkpoints

```bash
# Save frequently
--save-interval 1000

# If crash, resume from checkpoint
fundamentallm train data.txt --resume checkpoints/step_5000.pt
```

## Getting Help

### 1. Check Logs

```bash
# Training logs
cat logs/train.log

# System info
fundamentallm --version
python --version
pip list | grep torch
```

### 2. Minimal Reproducible Example

```bash
# Create minimal script that reproduces issue
fundamentallm train tiny.txt --model-dim 64 --epochs 2
```

### 3. Include Information

When asking for help:
- ✅ Command you ran
- ✅ Full error message
- ✅ Python/PyTorch versions
- ✅ GPU info (if relevant)
- ✅ Data statistics (size, format)

### 4. Check GitHub Issues

Search existing issues for similar problems.

## Further Reading

- [Training Guide](./training.md) - Proper training setup
- [Hyperparameters](./hyperparameters.md) - Tuning guide
- [Evaluation](./evaluation.md) - Measuring performance
- PyTorch Troubleshooting Guide
- CUDA Documentation

## Next Steps

- [Quick Start](./quick-start.md) - If starting fresh
- [Installation](../tutorials/installation.md) - Reinstall properly
- [Training Deep-Dive](../tutorials/training-deep-dive.md) - Advanced training
