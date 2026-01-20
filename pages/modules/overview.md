# Modules Overview

FundamentaLLM is organized into logical modules. Each handles a specific part of the ML pipeline.

## Module Structure

```
src/fundamentallm/
├── config/          ← Configuration management
├── data/            ← Data loading and preprocessing
├── models/          ← Neural network architectures
├── training/        ← Training loop and utilities
├── generation/      ← Text generation strategies
├── evaluation/      ← Model evaluation metrics
├── cli/             ← Command-line interface
└── utils/           ← Shared utilities
```

## Core Modules

### Data Pipeline (`data/`)

**Purpose:** Load, tokenize, and prepare data for training

**Key components:**
- `tokenizers/` - Character tokenization
- `dataset.py` - PyTorch Dataset wrapper
- `loaders.py` - DataLoader creation
- `preprocessing.py` - Text cleaning and formatting

**Why it matters:** Good data handling is 80% of ML success. This module shows best practices.

[Deep dive → Data Module](./data.md)

### Models (`models/`)

**Purpose:** Implement the transformer architecture

**Key components:**
- `base.py` - Base model class
- `transformer.py` - Full transformer architecture
- `components/` - Attention, feedforward, normalization layers

**Why it matters:** See how transformers are built from scratch

[Deep dive → Models Module](./models.md)

### Training (`training/`)

**Purpose:** Orchestrate the training loop

**Key components:**
- `callbacks.py` - Training hooks (logging, checkpointing)
- `checkpoint.py` - Save/load model state
- `metrics.py` - Training metrics
- `losses.py` - Loss computation
- `early_stopping.py` - Stop training when validation plateaus

**Why it matters:** Production training involves much more than just gradients

[Deep dive → Training Module](./training.md)

### Generation (`generation/`)

**Purpose:** Generate text from trained models

**Key components:**
- `generator.py` - Main generation orchestrator
- `sampling.py` - Sampling strategies (temperature, top-k)
- `constraints.py` - Generation constraints

**Why it matters:** Generation strategy (greedy vs sampling) affects output quality

[Deep dive → Generation Module](./generation.md)

### Evaluation (`evaluation/`)

**Purpose:** Measure model quality

**Key components:**
- `evaluator.py` - Main evaluation logic
- `benchmarks.py` - Standard benchmarks

**Why it matters:** You can't improve what you don't measure

[Deep dive → Evaluation Module](./evaluation.md)

### Configuration (`config/`)

**Purpose:** Manage all hyperparameters

**Key components:**
- `base.py` - Base configuration
- `model.py` - Model hyperparameters
- `training.py` - Training hyperparameters
- `validation.py` - Config validation

**Why it matters:** Reproducible science requires versioned configs

[Deep dive → Config Module](./config.md)

### CLI (`cli/`)

**Purpose:** Command-line interface for users

**Key components:**
- `commands.py` - CLI command definitions
- `interactive.py` - Interactive generation mode

**Why it matters:** Good UX makes ML accessible

[Deep dive → CLI Module](./cli.md)

## Data Flow

```
User Input (CLI or Python API)
    ↓
Config Validation (config/)
    ↓
Data Loading (data/)
    ↓
Model Creation (models/)
    ↓
Training Loop (training/)
    ├─ Forward pass (models/)
    ├─ Loss calculation (training/losses.py)
    ├─ Backward pass (PyTorch)
    ├─ Weight update (training/)
    └─ Checkpointing (training/checkpoint.py)
    ↓
Model Saved
    ↓
Generation (generation/)
    ├─ Load checkpoint (training/checkpoint.py)
    ├─ Generate tokens (generation/generator.py)
    └─ Sample outputs (generation/sampling.py)
    ↓
Output to User
```

## Why This Organization?

1. **Separation of concerns** - Each module has a single responsibility
2. **Reusability** - Components can be used independently
3. **Testability** - Each module is tested separately
4. **Learning** - Clear where to look for specific concepts

## Module Dependencies

```
cli → config, training, generation
training → models, data, config
generation → models
evaluation → models, data
```

Notice:
- `models` is low-level (no dependencies except PyTorch)
- `training` and `generation` depend on `models`
- CLI depends on many things (it's the orchestrator)

## Next Steps

- **[Project Structure](./structure.md)** - File-by-file walkthrough
- **[Models Module](./models.md)** - Transformer implementation
- **[Data Module](./data.md)** - Data handling
- **[Training Module](./training.md)** - Training loop
- **[Concepts](../concepts/overview.md)** - Theory behind modules

Or dive into a specific module page for deep details.
