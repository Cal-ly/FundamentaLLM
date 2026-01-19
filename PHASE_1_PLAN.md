# Phase 1: Core Infrastructure Setup

**Objective:** Establish the foundational project structure, configuration system, and base abstractions that all other phases depend on.

**Status:** Planning

**Dependencies:** None (starting phase)

**Estimated Timeline:** 2-3 days

---

## Overview

Phase 1 sets up the scaffolding for the entire FundamentaLLM project. This includes:
- Project directory structure matching the design spec
- Python packaging and dependency management (pyproject.toml, setup.py, requirements files)
- Module initialization files
- Abstract base classes for extension points
- Configuration system with Pydantic models
- Configuration file templates (YAML)
- Configuration loaders and validators

All subsequent phases depend on this foundation.

---

## Files to Create

### Project Root Level

```
fundamentallm/
├── pyproject.toml                  # Python package metadata and tool configs
├── setup.py                        # Setup script for editable install
├── requirements.txt                # Pinned production dependencies
├── requirements-dev.txt            # Development dependencies
├── .gitignore                      # Git ignore patterns
├── .env.example                    # Environment variables template
├── README.md                       # Updated with basic info
├── LICENSE                         # MIT License
└── CONTRIBUTING.md                 # Contribution guidelines
```

### Source Code Structure

```
src/fundamentallm/
├── __init__.py                     # Package initialization with version
├── __main__.py                     # Allow `python -m fundamentallm`
├── version.py                      # Version information
│
├── config/
│   ├── __init__.py
│   ├── base.py                     # BaseConfig with YAML I/O
│   ├── model.py                    # TransformerConfig
│   ├── training.py                 # TrainingConfig
│   └── validation.py               # Config validators (if needed)
│
├── models/
│   ├── __init__.py
│   └── base.py                     # BaseModel abstract class
│
├── data/
│   ├── __init__.py
│   ├── tokenizers/
│   │   ├── __init__.py
│   │   └── base.py                 # BaseTokenizer abstract class
│   └── preprocessing.py            # Preprocessing utilities stub
│
├── training/
│   ├── __init__.py
│   └── callbacks.py                # BaseCallback abstract class
│
├── generation/
│   └── __init__.py                 # Placeholder
│
├── evaluation/
│   └── __init__.py                 # Placeholder
│
└── utils/
    ├── __init__.py
    ├── logging.py                  # Logging setup (stub)
    ├── device.py                   # Device management (stub)
    ├── random.py                   # Seed/reproducibility (stub)
    └── paths.py                    # Path utilities (stub)
```

### Configuration Files

```
configs/
├── default.yaml                    # Default training configuration
├── small.yaml                      # Fast training for testing
└── large.yaml                      # Full-scale training
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                     # Pytest configuration and fixtures
└── fixtures/
    └── sample_data.txt             # Small sample text for testing
```

---

## Detailed Tasks

### Task 1.1: Create Project Directory Structure

**Objective:** Set up all required directories

**Subtasks:**
1. Create `src/fundamentallm/` directory tree
2. Create `configs/` directory
3. Create `tests/fixtures/` directory
4. Create `docs/` subdirectories if not present
5. Create `scripts/`, `experiments/`, `data/` directories (can be empty)

**Files to Create:**
- All `__init__.py` files (empty initially)

**Success Criteria:**
- All directories exist
- All `__init__.py` files present
- Can run `ls -R` and see complete tree

---

### Task 1.2: Python Packaging Configuration

**Objective:** Set up `pyproject.toml` and `setup.py` for package management

**Files to Create:**
1. `pyproject.toml` - Complete with:
   - Project metadata (name, version, description, authors, license)
   - Dependencies: torch, pydantic, click, pyyaml, tqdm, rich
   - Dev dependencies: pytest, pytest-cov, black, isort, mypy, pylint, pre-commit
   - Experiment dependencies: wandb, tensorboard, jupyter
   - Tool configurations (black, isort, mypy, pytest, coverage)
   - Console scripts entry point: `fundamentallm = fundamentallm.cli.commands:cli`

2. `setup.py` - Minimal setup for editable installs
   - Uses `setup()` from setuptools
   - Points to pyproject.toml for metadata
   - Enables `pip install -e .`

3. `requirements.txt` - Production dependencies (pinned versions)
   - torch>=2.0.0
   - pydantic>=2.0.0
   - click>=8.0.0
   - pyyaml>=6.0
   - tqdm>=4.65.0
   - rich>=13.0.0

4. `requirements-dev.txt` - Development dependencies (pinned versions)
   - Include everything from requirements.txt
   - Add: pytest, pytest-cov, black, isort, mypy, pylint, pre-commit

**Success Criteria:**
- `pip install -e .` works from project root
- `pip install -e ".[dev]"` works
- `fundamentallm --help` works (once CLI is created)
- Tool configurations in pyproject.toml are valid TOML

---

### Task 1.3: Base Abstractions

**Objective:** Create abstract base classes for extensibility

**File:** `src/fundamentallm/models/base.py`

**Contents:**
```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """Load model from disk."""
        pass
```

**File:** `src/fundamentallm/data/tokenizers/base.py`

**Contents:**
```python
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""
    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """Train tokenizer on corpus."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize tokenizer."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTokenizer":
        """Load tokenizer from disk."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass
```

**File:** `src/fundamentallm/training/callbacks.py`

**Contents:**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at training start."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at training end."""
        pass
    
    def on_epoch_begin(self, trainer: Any) -> None:
        """Called at epoch start."""
        pass
    
    def on_epoch_end(self, trainer: Any) -> None:
        """Called at epoch end."""
        pass
    
    def on_step(self, trainer: Any, loss: float) -> None:
        """Called after each training step."""
        pass

class CallbackList:
    """Manages list of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)
    
    def on_epoch_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)
    
    def on_step(self, trainer: Any, loss: float) -> None:
        for callback in self.callbacks:
            callback.on_step(trainer, loss)
```

**Success Criteria:**
- All abstract classes are properly defined
- Can import all base classes without errors
- Type hints are complete
- Abstract methods cannot be instantiated directly

---

### Task 1.4: Configuration System with Pydantic

**Objective:** Create flexible, validated configuration management

**File:** `src/fundamentallm/config/base.py`

**Contents:**
- `BaseConfig` class with YAML serialization
- `save()` method to write YAML
- `from_yaml()` classmethod to load YAML
- Config validation with `model_validate()`
- Support for environment variable overrides

**File:** `src/fundamentallm/config/model.py`

**Contents:**
- `TransformerConfig` with fields:
  - vocab_size, d_model, num_heads, num_layers, sequence_length
  - dropout, ffn_expansion, pos_encoding
  - head_dim property (computed)
  - Validators for divisibility (d_model % num_heads == 0)

**File:** `src/fundamentallm/config/training.py`

**Contents:**
- `TrainingConfig` with fields:
  - Data: data_path, train_split, sequence_length, batch_size, num_workers
  - Training: max_epochs, max_steps, gradient_accumulation_steps, gradient_clip_norm
  - Optimization: optimizer, learning_rate, weight_decay, adam_beta1/2, adam_epsilon
  - LR Schedule: lr_scheduler, warmup_steps, min_lr_ratio
  - Regularization: use_mixed_precision, dropout
  - Checkpointing: checkpoint_dir, save_every_n_steps, save_every_n_epochs, keep_last_n_checkpoints
  - Early Stopping: early_stopping_patience, early_stopping_metric, early_stopping_mode
  - Logging: log_every_n_steps, eval_every_n_steps, eval_every_n_epochs
  - Reproducibility: seed, deterministic
  - Device: device (cpu, cuda, mps)
  - Path validators for checkpoint_dir and data_path

**Success Criteria:**
- Pydantic validation works correctly
- YAML loading/saving works
- All field validators pass
- Type hints are complete
- Can instantiate configs without errors

---

### Task 1.5: Configuration YAML Files

**Objective:** Create template configuration files for different scenarios

**File:** `configs/default.yaml`

**Contents:**
```yaml
model:
  vocab_size: null  # Set at runtime
  d_model: 512
  num_heads: 8
  num_layers: 6
  sequence_length: 256
  dropout: 0.1
  ffn_expansion: 4
  pos_encoding: learned

training:
  data_path: data/processed/text.txt
  train_split: 0.9
  sequence_length: 256
  batch_size: 32
  num_workers: 4
  
  max_epochs: 10
  max_steps: null
  gradient_accumulation_steps: 1
  gradient_clip_norm: 1.0
  
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  
  lr_scheduler: cosine
  warmup_steps: 100
  min_lr_ratio: 0.1
  
  use_mixed_precision: true
  dropout: 0.1
  
  checkpoint_dir: checkpoints/
  save_every_n_epochs: 1
  keep_last_n_checkpoints: 3
  
  early_stopping_patience: 5
  early_stopping_metric: val_loss
  early_stopping_mode: min
  
  log_every_n_steps: 100
  eval_every_n_epochs: 1
  
  seed: 42
  deterministic: true
  device: cuda
```

**File:** `configs/small.yaml`

**Contents:** (Override default for fast testing)
```yaml
model:
  d_model: 128
  num_heads: 4
  num_layers: 2
  sequence_length: 64

training:
  batch_size: 8
  max_epochs: 2
  learning_rate: 1.0e-3
  log_every_n_steps: 10
```

**File:** `configs/large.yaml`

**Contents:** (Override default for larger model)
```yaml
model:
  d_model: 768
  num_heads: 12
  num_layers: 12
  sequence_length: 512

training:
  batch_size: 64
  max_epochs: 20
  learning_rate: 1.0e-4
```

**Success Criteria:**
- All YAML files are valid YAML syntax
- Can load with PyYAML
- Can instantiate configs from YAML files
- Defaults are sensible for different scenarios

---

### Task 1.6: Version and Package Initialization

**Objective:** Set up versioning and package initialization

**File:** `src/fundamentallm/version.py`

**Contents:**
```python
"""Version information for FundamentaLLM."""

__version__ = "0.1.0"
__author__ = "Your Name"
__author_email__ = "your.email@example.com"
__license__ = "MIT"
```

**File:** `src/fundamentallm/__init__.py`

**Contents:**
```python
"""FundamentaLLM - Minimal educational language model framework."""

from .version import __version__

__all__ = ["__version__"]
```

**File:** `src/fundamentallm/__main__.py`

**Contents:**
```python
"""Allow running fundamentallm as: python -m fundamentallm"""

if __name__ == "__main__":
    # Placeholder - will be replaced with CLI in Phase 6
    print("FundamentaLLM CLI coming soon!")
```

**Success Criteria:**
- Can import fundamentallm module
- `fundamentallm.__version__` accessible
- Can run `python -m fundamentallm`

---

### Task 1.7: Utility Module Stubs

**Objective:** Create placeholder utility modules (implementations in Phase 4+)

**File:** `src/fundamentallm/utils/logging.py`

**Contents:**
```python
"""Logging utilities."""

def get_logger(name: str):
    """Get logger for module."""
    import logging
    return logging.getLogger(name)

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    import logging
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
```

**File:** `src/fundamentallm/utils/device.py`

**Contents:**
```python
"""Device management utilities."""

import torch

def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    return torch.device(device_str)

def get_available_devices() -> list:
    """Get list of available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices
```

**File:** `src/fundamentallm/utils/random.py`

**Contents:**
```python
"""Random seed and reproducibility utilities."""

import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**File:** `src/fundamentallm/utils/paths.py`

**Contents:**
```python
"""Path utilities."""

from pathlib import Path

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

**Success Criteria:**
- All stub modules are importable
- Functions have proper type hints
- No errors on import

---

### Task 1.8: Root Level Documentation

**Objective:** Create root-level documentation files

**File:** `.gitignore`

**Contents:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.dmypy.json
dmypy.json

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Experiments
experiments/runs/
wandb/

# Data
data/raw/
data/processed/
*.csv
*.txt~

# Checkpoints
checkpoints/
*.pt
*.pth

# Logs
*.log

# Environment
.env
.env.local
```

**File:** `.env.example`

**Contents:**
```
# Device configuration
DEVICE=cuda

# Logging
LOG_LEVEL=INFO

# Weights & Biases (optional)
WANDB_PROJECT=fundamentallm
WANDB_DISABLED=true

# Reproducibility
RANDOM_SEED=42
```

**File:** `README.md`

**Contents:**
```markdown
# FundamentaLLM

A minimal, educational character-level transformer language model framework.

## Quick Start

```bash
# Install
pip install -e .

# Train
fundamentallm train data/text.txt --config configs/small.yaml

# Generate
fundamentallm generate checkpoint.pt --prompt "Hello"
```

## Documentation

- [Design System](docs/instruct/DESIGN_SYSTEM.md)
- [Getting Started](docs/getting_started.md) (coming in Phase 7)
- [Training Guide](docs/training_guide.md) (coming in Phase 7)
- [API Reference](docs/api_reference.md) (coming in Phase 7)

## Project Structure

See [DESIGN_SYSTEM.md](docs/instruct/DESIGN_SYSTEM.md) for detailed architecture.

## License

MIT License - see LICENSE file for details.
```

**File:** `CONTRIBUTING.md`

**Contents:**
```markdown
# Contributing to FundamentaLLM

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
make test          # Run all tests
make test-cov      # Run with coverage
```

## Code Quality

```bash
make lint          # Run linters
make format        # Format code
```

## Phases

This project is implemented incrementally. See PHASE_*.md files for current status.
```

**Success Criteria:**
- All files are properly formatted
- README is informative but concise
- .gitignore covers all common cases

---

### Task 1.9: Test Infrastructure

**Objective:** Set up pytest configuration and basic fixtures

**File:** `tests/conftest.py`

**Contents:**
```python
"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def tmp_dir():
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_text():
    """Provide sample text for tokenization."""
    return "hello world\nthis is a test\nfundamentallm framework"

@pytest.fixture
def sample_texts():
    """Provide multiple sample texts."""
    return [
        "hello world",
        "this is a test",
        "fundamentallm",
        "transformer model"
    ]
```

**File:** `tests/fixtures/sample_data.txt`

**Contents:**
```
It was the best of times, it was the worst of times, it was the age of wisdom, 
it was the age of foolishness, it was the epoch of belief, it was the epoch of 
incredulity, it was the season of Light, it was the season of Darkness, it was 
the spring of hope, it was the winter of despair, we had everything before us, 
we had nothing before us, we were all going direct to Heaven, we were all going 
direct the other way – in short, the period was so far like the present period, 
that some of its noisiest authorities insisted on its being received, for good 
or for evil, in the superlative degree of comparison only.
```

**Success Criteria:**
- pytest discovers tests correctly
- Fixtures are accessible in test files
- Sample data file exists and is readable

---

## Implementation Checklist

- [ ] Create directory structure (Task 1.1)
- [ ] Create pyproject.toml (Task 1.2)
- [ ] Create setup.py (Task 1.2)
- [ ] Create requirements.txt and requirements-dev.txt (Task 1.2)
- [ ] Create BaseModel abstract class (Task 1.3)
- [ ] Create BaseTokenizer abstract class (Task 1.3)
- [ ] Create Callback and CallbackList classes (Task 1.3)
- [ ] Create BaseConfig with YAML I/O (Task 1.4)
- [ ] Create TransformerConfig (Task 1.4)
- [ ] Create TrainingConfig (Task 1.4)
- [ ] Create default.yaml configuration (Task 1.5)
- [ ] Create small.yaml configuration (Task 1.5)
- [ ] Create large.yaml configuration (Task 1.5)
- [ ] Create version.py (Task 1.6)
- [ ] Create __init__.py files (Task 1.6)
- [ ] Create __main__.py (Task 1.6)
- [ ] Create logging.py utility (Task 1.7)
- [ ] Create device.py utility (Task 1.7)
- [ ] Create random.py utility (Task 1.7)
- [ ] Create paths.py utility (Task 1.7)
- [ ] Create .gitignore (Task 1.8)
- [ ] Create .env.example (Task 1.8)
- [ ] Create/update README.md (Task 1.8)
- [ ] Create CONTRIBUTING.md (Task 1.8)
- [ ] Create conftest.py (Task 1.9)
- [ ] Create sample_data.txt (Task 1.9)

---

## Success Criteria for Phase 1

1. **Project Structure**
   - ✅ All directories exist and are organized per design spec
   - ✅ All `__init__.py` files present

2. **Packaging**
   - ✅ `pip install -e .` works
   - ✅ `pip install -e ".[dev]"` works
   - ✅ Can import `fundamentallm` module
   - ✅ All dependencies installable

3. **Base Abstractions**
   - ✅ BaseModel can be imported
   - ✅ BaseTokenizer can be imported
   - ✅ Callback can be imported
   - ✅ Abstract methods cannot be instantiated

4. **Configuration System**
   - ✅ Can create TransformerConfig instance
   - ✅ Can create TrainingConfig instance
   - ✅ Can load config from YAML
   - ✅ Can save config to YAML
   - ✅ Validation works for invalid configs
   - ✅ YAML files are complete and valid

5. **Documentation**
   - ✅ README is informative
   - ✅ CONTRIBUTING guide exists
   - ✅ .gitignore is comprehensive

6. **Testing**
   - ✅ pytest discovers tests
   - ✅ Fixtures are available
   - ✅ Sample data exists

---

## Next Phase Dependency

Phase 1 must be complete before starting Phase 2 (Data Pipeline).

All subsequent code depends on:
- Configuration system for loading training parameters
- Base abstractions for implementing concrete classes
- Project structure for organizing code

---

## Notes

- All type hints must be complete (checked with mypy in Phase 7)
- Use Pydantic v2+ syntax for validators
- Configuration files should be minimal but complete
- Utility stubs can be basic implementations (full features in later phases)
- All code should follow PEP 8 conventions (enforced with black/isort in Phase 7)
