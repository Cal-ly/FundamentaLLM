# FundamentaLLM Improvements - Developer Guide

**Purpose:** Guide developers on how to use and extend the improved error handling, logging, and validation features.

---

## Table of Contents

1. [Device Management](#device-management)
2. [Logging System](#logging-system)
3. [Configuration Validation](#configuration-validation)
4. [Error Handling Best Practices](#error-handling-best-practices)
5. [Checkpoint Management](#checkpoint-management)
6. [Training Monitoring](#training-monitoring)

---

## Device Management

### Overview

The device management system now handles device selection intelligently with automatic fallback.

### Usage

```python
from fundamentallm.utils.device import (
    get_device,
    validate_device,
    get_best_device,
    get_available_devices,
    get_device_info,
)

# Option 1: Auto-select best available device
device = get_device("auto")

# Option 2: Validate and fallback if needed
device_str = validate_device("cuda")  # Returns "cpu" if CUDA unavailable
device = get_device(device_str)

# Option 3: Get available devices
available = get_available_devices()  # ["cpu", "cuda", "mps"]

# Option 4: Get device info
info = get_device_info()
# {
#   "cpu_available": True,
#   "cuda_available": True,
#   "cuda_count": 2,
#   "mps_available": False,
# }
```

### In Your Code

```python
from fundamentallm.utils.device import validate_device
import logging

logger = logging.getLogger(__name__)

# In CLI commands or API functions:
device_choice = validate_device(user_input)  # Automatic fallback with warning
model = model.to(device_choice)
```

### Key Features

- ✅ Automatic CUDA → CPU fallback
- ✅ Validates device is actually available
- ✅ Logs all decisions with warnings
- ✅ Supports "auto" for intelligent selection
- ✅ Provides device diagnostics

---

## Logging System

### Setup

```python
from fundamentallm.utils.logging import setup_logging, get_logger, log_metrics
from pathlib import Path

# Basic setup
setup_logging(level="INFO")

# With file logging and JSON format
setup_logging(
    level="DEBUG",
    log_file=Path("training.log"),
    json_format=False,  # Set True for production
)

# Get logger for your module
logger = get_logger(__name__)
```

### Basic Logging

```python
from fundamentallm.utils.logging import get_logger

logger = get_logger(__name__)

logger.debug("Detailed debugging info")
logger.info("Important milestone reached")
logger.warning("Potentially problematic situation")
logger.error("Error occurred, recovery possible")
logger.critical("System failure, immediate action needed")
```

### Metrics Logging

```python
from fundamentallm.utils.logging import log_metrics

metrics = {
    "loss": 5.1234,
    "accuracy": 0.92,
    "learning_rate": 0.001,
}

log_metrics(logger, metrics, step=100)
# Output: "Step 100: loss=5.1234 | accuracy=0.9200 | learning_rate=0.0010"
```

### Log Levels Guide

| Level | When to Use | Example |
|-------|------------|---------|
| DEBUG | Detailed info for developers | "Loaded config from file.yaml" |
| INFO | Important milestones | "Epoch 1 complete, val_loss=4.5" |
| WARNING | Non-critical issues | "Device fallback: CUDA→CPU" |
| ERROR | Recoverable errors | "Failed to load checkpoint, using fresh model" |
| CRITICAL | System failures | "Out of memory, training halted" |

### Best Practices

```python
# ✅ Good: Specific, helpful context
logger.warning(f"Device fallback: requested '{device}' but using 'cpu'")

# ✅ Good: Include relevant values
logger.info(f"Training epoch {epoch}/{num_epochs}: loss={loss:.4f}")

# ❌ Avoid: Vague messages
logger.info("Error")

# ❌ Avoid: Too much verbosity in INFO
logger.info(f"Processing item {i} of {total}, value={value}")  # Use DEBUG instead
```

---

## Configuration Validation

### Overview

Configuration validation catches errors before training starts.

### Usage

```python
from fundamentallm.config.validation import (
    validate_training_config,
    validate_model_config,
    warn_on_issues,
)
from fundamentallm.config import TransformerConfig
from fundamentallm.config.training import TrainingConfig

# Validate training config
train_config = TrainingConfig(batch_size=1, num_epochs=10000)
issues = validate_training_config(train_config)
warn_on_issues(issues, "TrainingConfig")

# Validate model config
model_config = TransformerConfig(
    vocab_size=256,
    d_model=512,
    num_heads=8,
)
issues = validate_model_config(model_config)
warn_on_issues(issues, "TransformerConfig")
```

### What Gets Validated

**Training Config:**
- ✅ num_epochs: 1-10000 range
- ✅ batch_size: 1-2048, OOM warnings
- ✅ learning_rate: bounds and reasonableness
- ✅ max_grad_norm: validity checks
- ✅ accumulation_steps: vs batch_size
- ✅ warmup_steps: reasonable bounds

**Model Config:**
- ✅ vocab_size: minimum 2
- ✅ d_model: 64-8192 range
- ✅ num_heads: divisible into d_model
- ✅ num_layers: 1-128 range
- ✅ sequence_length: reasonable bounds

### Example Issues Caught

```python
# Issue: d_model not divisible by num_heads
config = TransformerConfig(d_model=512, num_heads=3)  # 512 % 3 != 0
issues = validate_model_config(config)
# Returns: ["d_model (512) must be divisible by num_heads (3)"]

# Issue: learning rate too high
config = TrainingConfig(learning_rate=0.5)
issues = validate_training_config(config)
# Returns: ["learning_rate seems high (0.5), consider 1e-4 to 1e-3"]

# Issue: accumulation > batch
config = TrainingConfig(batch_size=4, accumulation_steps=8)
issues = validate_training_config(config)
# Returns: ["accumulation_steps (8) > batch_size (4) is inefficient"]
```

---

## Error Handling Best Practices

### File I/O

**Before (generic):**
```python
try:
    text = path.read_text()
except Exception as exc:
    raise click.ClickException(f"Failed: {exc}")
```

**After (specific):**
```python
try:
    text = path.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise click.ClickException(f"Data file not found at {path}")
except UnicodeDecodeError as exc:
    logger.error(f"Encoding error: {exc}")
    raise click.ClickException(
        f"File encoding error. Ensure file is UTF-8 encoded."
    )
except Exception as exc:
    logger.exception(f"Unexpected error: {exc}")
    raise click.ClickException(f"Failed to read data: {exc}")
```

### Model Creation

```python
from fundamentallm.models.transformer import Transformer

try:
    model = Transformer(config)
    logger.info(f"Model created with {model.count_parameters():,} parameters")
except Exception as exc:
    logger.error(f"Model creation failed: {exc}")
    raise click.ClickException(f"Failed to create model: {exc}")
```

### Device Fallback

```python
from fundamentallm.utils.device import validate_device

device = validate_device(user_device)  # Auto-fallback if needed
logger.info(f"Using device: {device}")
```

---

## Checkpoint Management

### Loading with Diagnostics

The improved checkpoint loading provides clear diagnostics:

```python
from fundamentallm.generation.generator import TextGenerator

try:
    generator = TextGenerator.from_checkpoint(
        checkpoint_path="model.pt",
        device="cuda",
    )
except FileNotFoundError as exc:
    logger.error(f"Checkpoint not found: {exc}")
    raise
except ValueError as exc:
    logger.error(f"Config/tokenizer issue: {exc}")
    raise
except RuntimeError as exc:
    logger.error(f"Model loading failed: {exc}")
    raise
```

### Log Output

When loading a checkpoint, you'll see:
```
DEBUG: Loading checkpoint from /path/to/model.pt
DEBUG: Loaded config from checkpoint['config']
DEBUG: Model loaded with 42,123,456 parameters
DEBUG: Loaded tokenizer from /path/to/tokenizer.json
INFO: Successfully loaded generator from /path/to/model.pt
```

If things go wrong:
```
ERROR: Could not load config: TransformerConfig not found. Searched in:
  - /path/to/model.yaml
  - /path/to/model.yml
  - /path/to/config.yaml
  - /path/to/model.yaml
Solutions:
  1. Include config in checkpoint payload under 'config' key
  2. Place config.yaml next to checkpoint file
  3. Pass config explicitly to from_checkpoint(config=...)
```

---

## Training Monitoring

### Interpreting Training Logs

```
INFO     Trainer initialized on device: cuda
INFO     Starting training for 10 epochs
INFO     Starting epoch 1/10
DEBUG    Epoch 1 | Batch 50 | Loss: 5.1234 | EMA Loss: 5.2145 | LR: 1.00e-03
DEBUG    Epoch 1 | Batch 100 | Loss: 4.9856 | EMA Loss: 5.1623 | LR: 1.00e-03
INFO     Validation at step 100: val_loss=4.5123 | perplexity=91.34
DEBUG    Saved checkpoint: checkpoints/epoch_0.pt
INFO     Epoch 1/10 completed | train_loss=5.0123 | val_loss=4.5123 | lr=1.00e-03 | throughput=1500 tokens/sec
INFO     New best model saved with val_loss=4.5123
...
INFO     Training completed. Total steps: 1500
```

### Common Issues & Their Meanings

**Issue: NaN Loss**
```
ERROR: Invalid loss detected at step 250: loss=NaN
This usually indicates unstable training.
Try reducing learning_rate or increasing max_grad_norm.
```
**Solution:** Reduce learning_rate by 10x or increase max_grad_norm

**Issue: Very Slow Throughput**
```
INFO: Epoch 1/10 completed | throughput=50 tokens/sec
```
**Solution:** Check GPU utilization, increase batch_size, or use distributed training

**Issue: Early Stopping**
```
INFO: New best model saved with val_loss=4.5123
INFO: Early stopping triggered after epoch 5
```
**Meaning:** Validation loss stopped improving, training halted

---

## Testing Your Improvements

### Unit Tests

```python
# Test device fallback
from fundamentallm.utils.device import validate_device

assert validate_device("cpu") == "cpu"  # Always available
# Test is CUDA-aware:
device = validate_device("cuda")
assert device in ["cuda", "cpu"]

# Test config validation
from fundamentallm.config.validation import validate_training_config

issues = validate_training_config({"batch_size": 0})
assert any("batch_size" in issue for issue in issues)

issues = validate_training_config({"batch_size": 32})
assert all("batch_size" not in issue for issue in issues)
```

### Integration Tests

```python
# Test full training pipeline with new error handling
from fundamentallm.cli.commands import train
import click.testing

runner = click.testing.CliRunner()

# Should handle missing file gracefully
result = runner.invoke(train, ["nonexistent.txt"])
assert result.exit_code != 0
assert "Data file not found" in result.output

# Should handle device fallback
result = runner.invoke(train, ["data.txt", "--device", "nonexistent"])
assert result.exit_code == 0  # Should fallback and continue
```

---

## Extending the System

### Adding New Validators

```python
# In config/validation.py

def validate_custom_config(config):
    """Validate custom configuration."""
    issues = []
    
    value = config.get("custom_param", 0)
    if value < 0:
        issues.append("custom_param must be >= 0")
    if value > 1000:
        issues.append("custom_param seems very high (1000+)")
    
    return issues

# Usage:
issues = validate_custom_config(config)
warn_on_issues(issues, "CustomConfig")
```

### Adding New Log Formatters

```python
# In utils/logging.py

class CSVFormatter(logging.Formatter):
    """Log as CSV for spreadsheet analysis."""
    
    def format(self, record):
        # Format as CSV row
        return f'{record.created},{record.levelname},"{record.getMessage()}"'

# Usage:
formatter = CSVFormatter()
handler = logging.FileHandler("training.csv")
handler.setFormatter(formatter)
logger.addHandler(handler)
```

---

## Quick Reference

### Most Common Operations

```python
# Setup logging
from fundamentallm.utils.logging import setup_logging
setup_logging(level="INFO")

# Validate device
from fundamentallm.utils.device import validate_device
device = validate_device(user_input)

# Validate config
from fundamentallm.config.validation import validate_training_config, warn_on_issues
issues = validate_training_config(config)
warn_on_issues(issues)

# Get logger
from fundamentallm.utils.logging import get_logger
logger = get_logger(__name__)

# Log metrics
from fundamentallm.utils.logging import log_metrics
log_metrics(logger, {"loss": 5.1}, step=10)
```

---

## Summary

The improved FundamentaLLM provides:

1. **Robust Device Handling** - Works everywhere with intelligent fallback
2. **Clear Error Messages** - Users know exactly what went wrong and how to fix it
3. **Comprehensive Logging** - Full visibility into training process
4. **Config Validation** - Catches errors before training starts
5. **Better Diagnostics** - Helpful suggestions for common issues

Use these tools to build more reliable training pipelines and better user experiences!
