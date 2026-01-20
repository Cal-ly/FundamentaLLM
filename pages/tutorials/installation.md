# Tutorial: Installation and Setup

This step-by-step tutorial walks you through installing FundamentaLLM, verifying the installation, and understanding the project structure.

## Tutorial Goals

By the end, you'll have:
- ✅ FundamentaLLM installed and verified
- ✅ Understanding of the project structure
- ✅ All dependencies working correctly
- ✅ Ready to train your first model

## Prerequisites

- Python 3.9 or newer (`python --version`)
- Git (`git --version`)
- 4GB RAM minimum
- ~2GB disk space

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/fundamentallm.git
cd fundamentallm
```

**What happened:**
- Downloaded source code
- Changed to project directory
- You'll see all project files

### 2. Create Virtual Environment

```bash
# Create
python -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Why?** Virtual environments isolate dependencies. Each Python project can have different versions without conflicts.

**Check it worked:**
```bash
which python  # macOS/Linux - should show /path/to/venv/bin/python
# or on Windows:
where python  # should show ...\venv\Scripts\python.exe
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

**Why?** pip is the package installer. Newer versions are more reliable.

### 4. Install FundamentaLLM (Development Mode)

```bash
pip install -e ".[dev]"
```

**What this does:**
- `pip install` - Install package
- `-e` - Editable mode (changes to code are reflected immediately)
- `".[dev]"` - Install package + dev dependencies (testing, linting, etc.)

**What gets installed:**
- **Core:** PyTorch, PyYAML, Rich
- **Dev:** pytest, black, isort, flake8, mypy, coverage

This takes 2-5 minutes depending on your internet.

### 5. Verify Installation

```bash
# Check CLI works
fundamentallm --help

# You should see:
# Usage: fundamentallm [OPTIONS] COMMAND [ARGS]...
# 
#   FundamentaLLM - Train transformer models from scratch
# 
# Options:
#   --version   Show version
#   --help      Show this message
# 
# Commands:
#   train       Train a new model
#   generate    Generate text from model
#   evaluate    Evaluate model performance
```

### 6. Verify Dependencies

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check GPU availability (optional)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Check other dependencies
python -c "import yaml, rich; print('All core dependencies installed!')"
```

### 7. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v         # Unit tests
pytest tests/integration/ -v  # Integration tests

# With coverage report
pytest tests/ --cov=src/fundamentallm --cov-report=term
```

**Expected output:**
- 178+ tests passing (green checkmarks)
- Coverage >85%

## Understanding the Installation

### What Got Installed Where?

```
Your Computer
└─ fundamentallm/                  (Project root)
   ├─ venv/                        (Virtual environment)
   │  └─ lib/python3.x/site-packages/
   │     ├─ torch/                 (PyTorch)
   │     ├─ yaml/                  (PyYAML)
   │     ├─ rich/                  (Rich for formatting)
   │     ├─ pytest/                (Testing framework)
   │     └─ ... other packages
   │
   ├─ src/fundamentallm/           (Your code - editable)
   ├─ tests/                       (Tests)
   ├─ data/                        (Sample data)
   ├─ docs/                        (Documentation)
   └─ setup.py                     (Installation config)
```

### Editable Installation Explained

The `-e` flag means:

```
Normal install: pip copies code to venv/lib/
Editable install: Python looks at your project folder

Advantage: Change code → immediately available
Disadvantage: Project folder must stay in place
```

## Project Structure Overview

```
fundamentallm/
├─ README.md              ← Start here for overview
├─ setup.py               ← Installation config
├─ requirements.txt       ← Core dependencies
├─ requirements-dev.txt   ← Dev dependencies
│
├─ src/fundamentallm/     ← Main code
│  ├─ __main__.py         ← CLI entry point
│  ├─ models/             ← Neural networks
│  ├─ data/               ← Data loading
│  ├─ training/           ← Training loop
│  ├─ generation/         ← Text generation
│  ├─ config/             ← Configuration
│  └─ cli/                ← Command-line interface
│
├─ tests/                 ← Test suite
│  ├─ unit/               ← Component tests
│  └─ integration/        ← End-to-end tests
│
├─ data/                  ← Datasets
│  ├─ raw/shakespeare/    ← Raw text files
│  ├─ processed/          ← Processed data
│  └─ samples/            ← Small sample files
│
├─ docs/                  ← Documentation
└─ configs/               ← Training configs
```

## First-Time Checks

Run these to ensure everything is working:

```bash
# 1. CLI is accessible
fundamentallm --version

# 2. Can import modules
python -c "from fundamentallm.models import Transformer; print('✓ Models work')"

# 3. Can load data
python -c "from fundamentallm.data import CharacterTokenizer; print('✓ Data loading works')"

# 4. Tests pass
pytest tests/unit/test_cli.py -v
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want to use it:

```bash
# 1. Check current PyTorch
pip show torch | grep Location

# 2. Uninstall CPU version
pip uninstall torch -y

# 3. Install CUDA version (choose your CUDA version)
# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verify
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

For Apple Silicon (M1/M2/M3), no special setup needed - Metal acceleration is automatic.

## Updating FundamentaLLM

To get the latest version:

```bash
cd /path/to/fundamentallm
git pull origin main
pip install -e ".[dev]"  # Reinstall in case requirements changed
```

## Troubleshooting

### Problem: `fundamentallm: command not found`

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall
pip install -e .
```

### Problem: `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch
```

### Problem: Tests fail with import errors

**Solution:**
```bash
pip install -e ".[dev]"
```

### Problem: Very slow training (on CPU only)

**Solution:** Install PyTorch with GPU support (see GPU Setup above)

### Problem: `pip install -e ".[dev]"` fails on Windows

**Solution:** Try without quotes:
```bash
pip install -e .[dev]
```

## Next Steps

Great! You're installed. Now:

1. **[Your First Model](./first-model.md)** - Train and generate immediately
2. **[Quick Start](../guide/quick-start.md)** - 5-minute overview
3. **[Concepts](../concepts/overview.md)** - Understand the theory
4. **[Training Deep Dive](./training-deep-dive.md)** - Advanced training

Or jump to what interests you most!

## Getting Help

```bash
# General help
fundamentallm --help

# Specific command help
fundamentallm train --help
fundamentallm generate --help

# Check documentation
# See docs/ folder or this website
```

## Verifying Your Setup

Run this complete verification:

```bash
#!/bin/bash
echo "FundamentaLLM Setup Verification"
echo "================================"

echo "✓ Python version:"
python --version

echo "✓ Project location:"
pwd

echo "✓ Virtual environment:"
which python

echo "✓ CLI available:"
fundamentallm --version

echo "✓ PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  GPU: {torch.cuda.is_available()}')"

echo "✓ Running basic tests:"
pytest tests/unit/test_tokenizers.py -q

echo ""
echo "Setup complete! Ready to train models."
```

Copy this into a file, save as `verify_setup.sh`, make it executable, and run:

```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

---

**You're ready!** Proceed to [Your First Model](./first-model.md) to start training.
