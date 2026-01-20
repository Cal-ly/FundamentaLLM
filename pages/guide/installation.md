# Installation Guide

Get FundamentaLLM up and running in 5 minutes.

## System Requirements

- **Python:** 3.9, 3.10, or 3.11
- **OS:** Linux, macOS, or Windows
- **RAM:** 4GB minimum (8GB+ recommended)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA support)

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/fundamentallm.git
cd fundamentallm
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Why virtual environments?** They isolate dependencies per project, preventing conflicts with other Python projects.

## Step 3: Install FundamentaLLM

### Development Installation (Recommended)

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- **Core dependencies**: PyTorch, PyYAML, Rich
- **Dev dependencies**: pytest, coverage, black, isort, flake8, mypy

### Production Installation

```bash
# Install only core dependencies
pip install -e .
```

## Step 4: Verify Installation

```bash
# Check CLI works
fundamentallm --help

# Run tests (if dev installed)
pytest tests/ -v

# Check type hints (if dev installed)
mypy src/fundamentallm --ignore-missing-imports
```

You should see:
- CLI help output showing `train`, `generate`, `evaluate` commands
- All tests passing (178+ tests)
- No type errors

## GPU Setup (Optional)

### NVIDIA GPUs

If you have an NVIDIA GPU and want to use CUDA:

```bash
# Remove the default CPU-only PyTorch
pip uninstall torch

# Install CUDA-enabled PyTorch (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check: `python -c "import torch; print(torch.cuda.is_available())"`

### Apple Silicon (M1/M2/M3)

PyTorch includes Metal acceleration support automatically:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Troubleshooting

### Issue: `pip install -e ".[dev]"` fails on Windows

**Solution:** Try with single quotes:
```bash
pip install -e .[dev]
```

### Issue: `fundamentallm` command not found

**Solution:** Make sure virtual environment is activated and reinstall:
```bash
source venv/bin/activate
pip install -e .
```

### Issue: `ImportError: No module named 'torch'`

**Solution:** Install PyTorch:
```bash
pip install torch
```

### Issue: Tests fail with import errors

**Solution:** Install dev dependencies:
```bash
pip install -e ".[dev]"
```

### Issue: CUDA not available on GPU system

**Solution:** Reinstall PyTorch with CUDA support (see GPU Setup above)

## What's Next?

1. **[Quick Start](./quick-start.md)** - Train your first model
2. **[CLI Overview](./cli-overview.md)** - Learn the commands
3. **[Concepts](../concepts/overview.md)** - Understand the theory
4. **[Tutorials](../tutorials/first-model.md)** - Step-by-step guides

## Updating FundamentaLLM

To get the latest version:

```bash
cd fundamentallm
git pull origin main
pip install -e ".[dev]"  # Reinstall in case dependencies changed
```

## Uninstalling

```bash
pip uninstall fundamentallm
```

Or just deactivate/delete the virtual environment:

```bash
deactivate
rm -rf venv  # Linux/macOS
rmdir venv   # Windows
```
