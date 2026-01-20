# Tech Stack Overview

FundamentaLLM is built on proven, production-ready technologies. Understanding this stack helps you understand the design decisions throughout the project.

## Core Dependencies

### PyTorch
**What:** Deep learning framework  
**Why:** Industry standard for ML research and production. Provides automatic differentiation (gradients), GPU acceleration, and intuitive tensor operations.  
**How we use it:**
- Tensor operations for model forward/backward passes
- Autograd for gradient computation
- Mixed precision training with `torch.cuda.amp`
- Checkpoint serialization

### Python 3.9+
**What:** Programming language  
**Why:** Dominant in ML/AI with rich ecosystem. Type hints support makes code self-documenting.  
**How we use it:**
- Type hints throughout for code clarity
- Dataclasses for configuration
- F-strings for readable code

## Development & Testing

### pytest
**What:** Testing framework  
**Why:** Flexible, pythonic test discovery and execution  
**How we use it:**
- Unit tests for individual components
- Integration tests for pipelines
- Fixtures for shared test data

### Coverage
**What:** Code coverage measurement  
**Why:** Ensures tests cover actual code paths. Targets >85% coverage.  
**How we use it:**
- Identify untested code paths
- Maintain code quality standards

### Type Checking: mypy
**What:** Static type checker  
**Why:** Catches type errors before runtime. Makes code intent explicit.  
**How we use it:**
- Validates type hints across codebase
- Part of CI/CD pipeline

### Code Quality: black, isort, flake8
**What:** Code formatting and linting  
**Why:** Consistency across team/project. Catches common issues.  
**How we use it:**
- `black`: Enforces style consistency
- `isort`: Organizes imports
- `flake8`: Catches bugs and style violations

## Configuration & CLI

### Click (via PyYAML)
**What:** YAML configuration files + CLI framework  
**Why:** Reproducible experiments with version control friendly configs  
**How we use it:**
- YAML config files for training hyperparameters
- Loaded with validation before training
- Makes experiments reproducible

### Rich
**What:** Rich terminal output library  
**Why:** Makes CLI tools visually clear and user-friendly  
**How we use it:**
- Formatted tables for model metrics
- Progress bars for training
- Interactive mode with rich formatting

## Why This Stack?

| Goal | Technology | Alternative | Why Chosen |
|------|-----------|-------------|-----------|
| Deep Learning | PyTorch | TensorFlow | Better for research/learning, more intuitive |
| Testing | pytest | unittest | More pythonic, better discovery |
| Type Safety | mypy | No checking | Catches errors early |
| Config | YAML | .json/.ini | Human readable, version control friendly |
| Code Quality | black+isort+flake8 | No tools | Consistency, catches issues early |

## Architecture Stack

The framework is organized in layers:

```
┌─────────────────────────────────────┐
│ CLI (Click + Rich)                  │
├─────────────────────────────────────┤
│ Configuration (YAML + Dataclasses)  │
├─────────────────────────────────────┤
│ Training Pipeline (PyTorch + Metrics)
├─────────────────────────────────────┤
│ Model (Transformer + Attention)     │
├─────────────────────────────────────┤
│ Data (PyTorch Datasets/Dataloaders) │
├─────────────────────────────────────┤
│ Tokenizer (Character-level)         │
└─────────────────────────────────────┘
       Built on PyTorch
```

## Why Not...?

### TensorFlow?
- Steeper learning curve
- Less intuitive for research
- Overkill for educational project

### JAX?
- Excellent for research
- Not production-ready enough yet
- Steeper onboarding

### Hugging Face Transformers?
- Black-box abstractions
- Defeats educational goal
- We implement things from scratch for learning

### Other config formats?
- JSON: Too verbose
- .ini: Less nested structure support
- CLI only: Non-reproducible

## Dependency Justification

Every dependency is chosen because:
1. **Industry standard** - You'll see it in real projects
2. **Well maintained** - Active development and bug fixes
3. **Minimal** - We avoid dependency bloat
4. **Educational** - Teaches best practices

No unnecessary dependencies = cleaner, easier to understand codebase.

## Next Steps

- [Installation](./installation.md) - Get it running
- [Quick Start](./quick-start.md) - Train your first model
- [Transformer Concepts](../concepts/transformers.md) - Understand the architecture
