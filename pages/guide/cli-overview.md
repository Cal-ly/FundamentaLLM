# CLI Overview

Learn how to use FundamentaLLM from the command line.

## Main Commands

FundamentaLLM has three primary commands: `train`, `generate`, and `evaluate`.

### Training a Model

```bash
fundamentallm train <data_path> [OPTIONS]
```

**Basic usage:**
```bash
fundamentallm train data/samples/sample_data.txt --output-dir my_model
```

**Common options:**
```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir shakespeare_model \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --model-dim 256 \
    --num-heads 4 \
    --num-layers 6
```

**All options:**
- `--output-dir` - Output directory for checkpoints and final model. Relative paths auto-prefix with `models/` (e.g., `my_model` → `models/my_model/`). Default: `models/checkpoints/`
- `--epochs` - Training epochs (default: 10; max: 10000)
- `--batch-size` - Batch size (default: 16; range: 1-2048)
- `--learning-rate` - Learning rate (default: 1e-3; recommended: 1e-4 to 1e-3)
- `--model-dim` - Model hidden dimension (default: 128; minimum: 64)
- `--num-heads` - Number of attention heads (default: 2; must divide `--model-dim`)
- `--num-layers` - Number of transformer layers (default: 6; max: 48)
- `--dropout` - Dropout rate (default: 0.1; range: 0.0-1.0)
- `--max-seq-len` - Maximum sequence length (default: 256; warn if >8192)
- `--val-split` - Validation split ratio (default: 0.2; range: 0.0-1.0)
- `--seed` - Random seed (default: 42)
- `--device` - CPU or CUDA (default: auto-detect)
- `--mixed-precision` - Use mixed precision training (default: false)
- `--auto-fix-config` - Automatically fix parameter conflicts (default: true)
- `--gradient-clip` - Gradient clipping norm to prevent exploding gradients (default: 1.0; typical: 0.5-2.0)

### Generating Text

```bash
fundamentallm generate <model_path> [OPTIONS]
```

**Basic usage:**
```bash
fundamentallm generate my_model/final_model.pt --prompt "The "
```

**Common options:**
```bash
fundamentallm generate my_model/final_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --interactive
```

**All options:**
- `--prompt` - Starting text (default: empty)
- `--max-tokens` - Maximum tokens to generate (default: 100)
- `--temperature` - Sampling temperature (default: 1.0)
  - < 1.0: More focused, less random
  - = 1.0: Standard probability
  - > 1.0: More random, more diverse
- `--top-k` - Only sample from top-k tokens (default: None, no limit)
- `--top-p` - Nucleus sampling parameter (default: 1.0)
- `--num-samples` - Generate multiple outputs (default: 1)
- `--interactive` - Interactive mode
- `--device` - CPU or CUDA

**Understanding temperature:**
```
Temperature 0.1 (greedy):
"The morning light shone on the beautiful day"
(Very predictable, follows most likely path)

Temperature 1.0 (balanced):
"The morning sun crept slowly through my window"
(Natural, balanced randomness)

Temperature 2.0 (very random):
"The morning elephant danced with purple thoughts"
(Creative, less coherent)
```

### Evaluating Models

```bash
fundamentallm evaluate <model_path> <data_path>
```

**Usage:**
```bash
fundamentallm evaluate my_model/final_model.pt data/samples/sample_data.txt
```

**Output:**
```
Model Evaluation Results
========================
Test Loss:    3.234
Perplexity:   25.34
Accuracy:     0.42
```

## Output Directory Organization

FundamentaLLM automatically organizes model outputs under a `models/` directory to keep your project clean.

### How Output Directory Works

**Relative paths auto-prefix with `models/`:**
```bash
# These are equivalent:
fundamentallm train data.txt --output-dir my_model
fundamentallm train data.txt --output-dir models/my_model

# Both create: models/my_model/checkpoints, models/my_model/best.pt, etc.
```

**Absolute paths are unchanged:**
```bash
fundamentallm train data.txt --output-dir /absolute/path/my_model
# Creates: /absolute/path/my_model/
```

**No flag uses default:**
```bash
fundamentallm train data.txt
# Creates: models/checkpoints/ (default)
```

### Directory Structure

```
your_project/
├── models/                          ← All models here
│   ├── checkpoints/                 ← Default output
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── best.pt
│   │   └── final_model.pt
│   ├── shakespeare_model/           ← From: --output-dir shakespeare_model
│   │   └── final_model.pt
│   └── custom_config/               ← From: --output-dir custom_config
│       └── best.pt
├── data/
├── configs/
└── src/
```

### Why This Organization?

- **Clean root:** Your project root stays uncluttered
- **Consistency:** All models in one place
- **Easy discovery:** Know exactly where to find model files
- **Backup-friendly:** Single directory to backup/version control

## Parameter Validation & Auto-Fix

FundamentaLLM validates all parameters before training to catch configuration issues early.

### What Gets Validated

**Critical validation** (will block training if violated):
- `num_heads` must divide `model_dim` evenly (e.g., 256/8 = 32 ✓, but 256/7 = ✗)
- `num_heads` must not exceed `model_dim / 8` (keeps head dimension ≥ 8)
- `model_dim` must be ≥ 64

**Warnings** (logged but won't block):
- `model_dim` < 64 (underfitting risk)
- `learning_rate` > 0.1 or < 1e-6 (unstable learning)
- `max_seq_len` > 8192 (out-of-memory risk)
- `batch_size` > 2048 (memory intensive)
- `gradient_clip` > 10 (may impede learning)

### Auto-Fix Behavior

By default (`--auto-fix-config`, enabled by default), FundamentaLLM will automatically fix parameter conflicts:

**Example: Auto-fixing num_heads**

If you request an invalid configuration:
```bash
fundamentallm train data.txt \
    --model-dim 512 \
    --num-heads 16  # Would cause head_dim < 8
```

The system will log:
```
WARNING: num_heads (16) too high relative to d_model (512).
Auto-fixing: num_heads 16 → 8 (head_dim = 512/8 = 64)
```

**To disable auto-fix:**

If you want strict validation without auto-correction:
```bash
fundamentallm train data.txt \
    --model-dim 512 \
    --num-heads 16 \
    --no-auto-fix-config
```

This will report the issue and block training rather than auto-adjusting.

**Why auto-fix is enabled by default:**
- Prevents training from failing on slightly misconfigured parameters
- Logs clear warnings so you know what changed
- Users can still disable it for strict validation if needed

## Parameter Bounds Reference

Use these ranges when setting parameters to avoid warnings and ensure stability:

| Parameter | Min | Max | Recommended | Warn If | Notes |
|-----------|-----|-----|------------|---------|-------|
| `--model-dim` | 64 | 8192 | 128-512 | < 64 | Must divide evenly by `num_heads` |
| `--num-heads` | 1 | d_model/8 | 4-16 | — | Must divide `model_dim`; auto-fix keeps ≥ 8 |
| `--num-layers` | 1 | 48 | 6-24 | > 48 | Hard limit from model schema |
| `--batch-size` | 1 | 2048 | 16-64 | > 2048 | Affects memory usage significantly |
| `--learning-rate` | >0 | 0.1 | 1e-4 to 1e-3 | < 1e-6 or > 0.1 | Most important for stability |
| `--dropout` | 0.0 | 1.0 | 0.1-0.3 | — | Regularization strength |
| `--gradient-clip` | >0 | 10 | 0.5-2.0 | > 10 | Prevents exploding gradients |
| `--epochs` | 1 | 10000 | 10-50 | > 10000 | Controls training duration |
| `--max-seq-len` | 1 | 8192+ | 256-1024 | > 8192 | Longer sequences need more memory |
| `--val-split` | (0, 1) | — | 0.2 | — | 80/20 split recommended |
| `--seed` | — | — | 42 | — | For reproducibility |

**Head dimension rule:** `head_dimension = model_dim / num_heads` must be ≥ 8
- Example: 512 dim ÷ 8 heads = 64 per head ✅
- Example: 256 dim ÷ 16 heads = 16 per head ✅
- Example: 128 dim ÷ 32 heads = 4 per head ❌ (too small, auto-fix reduces heads)

## Interactive Mode

The most fun way to interact with your model:

```bash
fundamentallm generate my_model/final_model.pt --interactive
```

**Features:**
- Generate text from any prompt
- Change settings mid-session
- Multiple generates from same prompt
- Command shortcuts

**Commands:**
- `/help` - Show available commands
- `/set temperature=0.8` - Change a parameter
- `/set top-k=50` - Set top-k sampling
- `/quit` or `/exit` - Exit interactive mode
- `/clear` - Clear screen
- `/status` - Show current settings

**Example session:**
```
> The library was
The library was filled with ancient scrolls and forgotten
 memories of centuries past.

> /set temperature=0.5
Updated: temperature=0.5

> The library was
The library was quiet and peaceful, with sunlight streaming
 through the tall windows.

> /quit
Goodbye!
```

## Configuration Files

For reproducible training, use configuration files:

```bash
fundamentallm train data/samples/sample_data.txt \
    --config configs/small.yaml \
    --output-dir my_model
```

**Config file format (YAML):**
```yaml
# Training configuration
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  device: cuda
  mixed_precision: true

# Model configuration
model:
  hidden_dim: 256
  num_heads: 4
  num_layers: 8
  dropout: 0.1
  max_seq_len: 512

# Data configuration
data:
  val_split: 0.1
  seed: 42
```

See `configs/` directory for examples.

## Tips & Tricks

### Profile Your Training

```bash
# Use PyTorch profiler (with --profile flag)
fundamentallm train data/samples/sample_data.txt \
    --output-dir my_model \
    --profile
```

### Resume Training from Checkpoint

```bash
# Training automatically checkpoints every epoch
# Resume by pointing to checkpoint directory
fundamentallm train data/samples/sample_data.txt \
    --output-dir my_model \
    --resume-from my_model/checkpoint_epoch_10.pt
```

### Use Different Model Sizes

**Small (for CPU/quick testing):**
```bash
fundamentallm train data/samples/sample_data.txt \
    --output-dir small_model \
    --model-dim 64 \
    --num-heads 2 \
    --num-layers 3 \
    --epochs 5
```

**Medium (typical):**
```bash
fundamentallm train data/raw/shakespeare/shakespeare100k.txt \
    --output-dir medium_model \
    --model-dim 256 \
    --num-heads 4 \
    --num-layers 6
```

**Large (for GPU with lots of VRAM):**
```bash
fundamentallm train data/raw/shakespeare/shakespeare1mil.txt \
    --output-dir large_model \
    --model-dim 512 \
    --num-heads 8 \
    --num-layers 12 \
    --batch-size 64
```

### Batch Processing Generations

```bash
# Generate 10 different outputs
fundamentallm generate my_model/final_model.pt \
    --prompt "Once upon a time" \
    --num-samples 10 \
    --output-file generations.txt
```

### Monitor Training with Logs

```bash
# Training automatically logs metrics
# View them:
tail -f my_model/training.log

# Or with rich formatting:
watch -n 1 'tail -20 my_model/training.log'
```

## Environment Variables

Control behavior with environment variables:

```bash
# Set device
export FUNDAMENTALLM_DEVICE=cuda

# Set random seed globally
export FUNDAMENTALLM_SEED=123

# Enable debug output
export FUNDAMENTALLM_DEBUG=1

# Then run commands
fundamentallm train data/samples/sample_data.txt --output-dir my_model
```

## Getting Help

```bash
# Main help
fundamentallm --help

# Command-specific help
fundamentallm train --help
fundamentallm generate --help
fundamentallm evaluate --help

# Version
fundamentallm --version
```

## Common Workflows

### Quick Testing
```bash
# Fast training to test pipeline
fundamentallm train data/samples/sample_data.txt \
    --output-dir test_model \
    --epochs 2 \
    --batch-size 16 \
    --model-dim 64 \
    --num-layers 2

# Quick generate
fundamentallm generate test_model/final_model.pt --prompt "Test"
```

### Full Training
```bash
# Serious training
fundamentallm train data/raw/shakespeare/shakespeare500k.txt \
    --output-dir real_model \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --model-dim 512 \
    --num-heads 8 \
    --num-layers 12 \
    --mixed-precision

# Explore the trained model
fundamentallm generate real_model/final_model.pt --interactive
```

### Hyperparameter Search
```bash
# Try different learning rates
for lr in 0.0001 0.0005 0.001 0.005; do
    echo "Training with LR=$lr"
    fundamentallm train data/samples/sample_data.txt \
    --output-dir model_lr_${lr} \
    --learning-rate $lr \
    --epochs 5
done

# Evaluate each
for model in model_lr_*/; do
    fundamentallm evaluate "$model/final_model.pt" data/samples/sample_data.txt
done
```

## Next Steps

- **[Training Guide](./training.md)** - Deep dive into training strategies
- **[Generation Guide](./generation.md)** - Control generation behavior
- **[Quick Start](./quick-start.md)** - Run your first model now
