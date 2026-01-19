# Contributing to FundamentaLLM

Thank you for your interest in contributing to FundamentaLLM! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Development Setup

1. **Fork and clone the repository:**

```bash
git clone https://github.com/your-username/fundamentallm.git
cd fundamentallm
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -e ".[dev]"
```

4. **Set up pre-commit hooks:**

```bash
pre-commit install
```

5. **Verify installation:**

```bash
make test
```

## 💻 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Your Changes

Follow the [Code Standards](#code-standards) below and ensure:
- Code is well-documented
- Tests are added/updated
- Changes are focused and atomic

### 3. Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_attention.py -v
```

### 4. Format and Lint

```bash
# Format code
make format

# Check formatting
make format-check

# Type checking
make type-check

# Linting
make lint

# Run all pre-commit hooks
make pre-commit
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add top-p sampling to generator

- Implement nucleus sampling in generate() method
- Add top_p parameter to CLI generate command
- Update tests to cover new sampling strategy
- Add documentation for top-p sampling
"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📝 Code Standards

### Style Guidelines

- **Line length:** 100 characters maximum
- **Formatting:** Use Black (enforced by pre-commit)
- **Import sorting:** Use isort (enforced by pre-commit)
- **Type hints:** Use type annotations where practical
- **Docstrings:** Google-style docstrings for public APIs

### Example Code Style

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fundamentallm.config.model import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.
    
    Args:
        config: Model configuration object containing hyperparameters.
        
    Example:
        >>> config = ModelConfig(d_model=512, num_heads=8)
        >>> block = TransformerBlock(config)
        >>> x = torch.randn(2, 10, 512)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask of shape (seq_len, seq_len).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
```

### Documentation Standards

**Module docstrings:**
```python
"""Transformer model components.

This module implements the core transformer architecture including:
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
"""
```

**Function/method docstrings:**
```python
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train model for one epoch.
    
    Args:
        model: The model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer for updating model parameters.
        
    Returns:
        Average loss over all batches.
        
    Raises:
        RuntimeError: If CUDA out of memory during training.
        
    Example:
        >>> model = TransformerLM(config)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>> loss = train_epoch(model, loader, optimizer)
        >>> print(f"Epoch loss: {loss:.4f}")
    """
```

## 🧪 Testing Guidelines

### Test Structure

Tests are organized into:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full pipelines
- `tests/fixtures/` - Shared test fixtures and data

### Writing Tests

```python
import pytest
import torch

from fundamentallm.models.components.attention import MultiHeadAttention
from fundamentallm.config.model import ModelConfig


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention module."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ModelConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
        )
    
    @pytest.fixture
    def attention(self, config):
        """Create attention module."""
        return MultiHeadAttention(config)
    
    def test_forward_shape(self, attention):
        """Test output shape is correct."""
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_with_mask(self, attention):
        """Test attention with causal mask."""
        x = torch.randn(2, 10, 64)
        mask = torch.tril(torch.ones(10, 10))
        
        output = attention(x, mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
```

### Test Coverage

- Aim for >85% coverage on core modules
- 100% coverage on critical paths (training, generation)
- Test edge cases and error conditions
- Use parametrize for testing multiple inputs

```python
@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
def test_temperature_sampling(temperature):
    """Test sampling with different temperatures."""
    logits = torch.randn(100)
    probs = sample_with_temperature(logits, temperature)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
```

## 🔄 Pull Request Process

### Before Submitting

1. ✅ All tests pass: `make test`
2. ✅ Code is formatted: `make format`
3. ✅ No linting errors: `make lint`
4. ✅ Type checking passes: `make type-check`
5. ✅ Documentation is updated
6. ✅ CHANGELOG.md is updated (if applicable)

### PR Title and Description

**Title format:**
```
<type>: <short description>
```

**Description template:**
```markdown
## Description
Brief description of changes and motivation.

## Changes
- Added X feature
- Fixed Y bug
- Refactored Z component

## Testing
- Added unit tests for X
- Verified Y works on CPU and CUDA
- Tested edge cases for Z

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks:** CI must pass
2. **Code review:** At least one approving review required
3. **Discussion:** Address reviewer comments
4. **Merge:** Squash and merge after approval

### After Merge

- Delete your feature branch
- Pull latest main: `git pull origin main`
- Celebrate! 🎉

## 📁 Project Structure

```
fundamentallm/
├── src/fundamentallm/          # Main package
│   ├── config/                 # Configuration classes
│   ├── data/                   # Data loading and tokenization
│   ├── models/                 # Model architectures
│   ├── training/               # Training infrastructure
│   ├── generation/             # Text generation
│   ├── evaluation/             # Model evaluation
│   └── cli/                    # CLI interface
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
├── docs/                       # Documentation
│   ├── getting_started.md      # Quick start guide
│   ├── training_guide.md       # Training best practices
│   ├── architecture.md         # Architecture deep dive
│   └── notebooks/              # Jupyter notebooks
├── configs/                    # Config presets
├── data/                       # Sample data
└── scripts/                    # Utility scripts
```

## 🗺️ Roadmap

See [Phase Plans](PLAN_INDEX.md) for detailed roadmap.

**Current priorities:**
- Example Jupyter notebooks
- Additional sampling strategies
- Performance optimizations
- Expanded documentation

**Future enhancements:**
- Subword tokenization (BPE, WordPiece)
- Fine-tuning support
- Multi-GPU training
- Model quantization
- Beam search decoding

## 📚 Resources

### For Contributors

- **[Architecture Guide](docs/architecture.md)** - Understand the codebase
- **[Training Guide](docs/training_guide.md)** - Learn training internals
- **[Phase Plans](PLAN_INDEX.md)** - See project history and roadmap
- **[Lessons Learned](docs/instruct/LL_LI.md)** - Design decisions and insights

### External Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## ❓ Questions?

- **General questions:** [GitHub Discussions](https://github.com/your-org/fundamentallm/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/your-org/fundamentallm/issues)
- **Feature requests:** [GitHub Issues](https://github.com/your-org/fundamentallm/issues)

## 🙏 Thank You!

Every contribution, no matter how small, helps make FundamentaLLM better for everyone. We appreciate your time and effort!

---

**Happy Contributing! 🚀**
