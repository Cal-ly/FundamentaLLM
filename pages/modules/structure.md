# Project Structure

Understand the organization and architecture of the FundamentaLLM codebase.

## Directory Overview

```
FundamentaLLM/
├── src/fundamentallm/          # Main package
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point
│   ├── version.py
│   ├── cli/                    # Command-line interface
│   ├── config/                 # Configuration management
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training logic
│   ├── generation/             # Text generation
│   ├── evaluation/             # Evaluation and metrics
│   └── utils/                  # Utilities
├── configs/                    # Configuration files
├── data/                       # Training data
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── tests/                      # Unit and integration tests
├── experiments/                # Experiment tracking
├── pyproject.toml              # Project metadata
├── setup.py                    # Installation
└── requirements.txt            # Dependencies
```

## Module Details

### CLI (`cli/`)

**Purpose:** Command-line interface for training, generation, and evaluation.

```
cli/
├── __init__.py
├── commands.py        # Command definitions
└── interactive.py     # Interactive mode
```

**Key files:**
- `commands.py`: Defines all CLI commands (train, generate, evaluate, interactive)
- `interactive.py`: REPL-style interactive generation

**Entry point:**
```bash
fundamentallm train data.txt  # Calls cli.commands.train()
```

### Configuration (`config/`)

**Purpose:** Configuration management and validation.

```
config/
├── __init__.py
├── base.py           # Base configuration
├── model.py          # Model config (dim, layers, heads)
├── training.py       # Training config (lr, batch size)
└── validation.py     # Config validation
```

**Key classes:**
- `ModelConfig`: Model architecture parameters
- `TrainingConfig`: Training hyperparameters
- `Config`: Combined configuration

**Usage:**
```python
from fundamentallm.config import Config

config = Config.from_file('configs/default.yaml')
model = create_model(config.model)
```

### Data (`data/`)

**Purpose:** Data loading, preprocessing, and tokenization.

```
data/
├── __init__.py
├── dataset.py           # PyTorch Dataset
├── loaders.py           # DataLoader utilities
├── preprocessing.py     # Text preprocessing
└── tokenizers/          # Tokenization
    ├── __init__.py
    ├── base.py          # Base tokenizer
    └── char.py          # Character tokenizer
```

**Key components:**
- `CharTokenizer`: Character-level tokenization (256-token vocab)
- `TextDataset`: PyTorch Dataset for text
- `get_dataloader()`: Create train/val dataloaders

**Data flow:**
```
Raw text → Preprocessing → Tokenization → Dataset → DataLoader → Batches
```

### Models (`models/`)

**Purpose:** Neural network architectures.

```
models/
├── __init__.py
├── base.py              # Base model class
├── transformer.py       # Transformer LM
└── components/          # Model components
    ├── __init__.py
    ├── attention.py     # Multi-head attention
    ├── embeddings.py    # Token + positional embeddings
    ├── feedforward.py   # Feed-forward network
    └── normalization.py # Layer normalization
```

**Key classes:**
- `TransformerLM`: Full transformer language model
- `MultiHeadAttention`: Attention mechanism
- `FeedForward`: FFN layers
- `TokenEmbedding`: Token embeddings
- `PositionalEncoding`: Position information

**Model architecture:**
```
Input IDs
    ↓
TokenEmbedding
    ↓
PositionalEncoding
    ↓
[TransformerBlock] × N
    ├─ MultiHeadAttention
    ├─ LayerNorm
    ├─ FeedForward
    └─ LayerNorm
    ↓
Output projection (vocab)
```

### Training (`training/`)

**Purpose:** Training loop, optimization, and checkpointing.

```
training/
├── __init__.py
├── trainer.py           # Main training loop
├── optimizers.py        # Optimizer setup
├── schedulers.py        # Learning rate schedules
├── losses.py            # Loss functions
├── metrics.py           # Training metrics
├── callbacks.py         # Training callbacks
├── checkpoint.py        # Model checkpointing
└── early_stopping.py    # Early stopping
```

**Key components:**
- `Trainer`: Main training orchestration
- `train_epoch()`: Single epoch training
- `validate()`: Validation loop
- `CheckpointManager`: Save/load checkpoints
- `EarlyStopping`: Stop when val loss plateaus

**Training flow:**
```
for epoch in epochs:
    train_epoch()
        → forward pass
        → compute loss
        → backward pass
        → optimizer step
    validate()
    save_checkpoint()
    check_early_stopping()
```

### Generation (`generation/`)

**Purpose:** Text generation and sampling strategies.

```
generation/
├── __init__.py
├── generator.py         # Generation logic
├── sampling.py          # Sampling strategies
└── constraints.py       # Generation constraints
```

**Key components:**
- `Generator`: Main generation class
- `sample_token()`: Token sampling
- Temperature, top-k, top-p sampling
- Repetition penalties

**Generation loop:**
```
Start with prompt
while not done:
    model(current_tokens) → logits
    sample(logits) → next_token
    append next_token
    check stopping criteria
```

### Evaluation (`evaluation/`)

**Purpose:** Model evaluation and benchmarks.

```
evaluation/
├── __init__.py
├── evaluator.py        # Evaluation logic
└── benchmarks.py       # Standard benchmarks
```

**Key metrics:**
- Perplexity
- Cross-entropy loss
- Bits per character
- Token accuracy

### Utils (`utils/`)

**Purpose:** Shared utilities and helpers.

```
utils/
├── __init__.py
├── logging.py          # Logging setup
├── device.py           # Device management (CPU/GPU)
├── random.py           # Random seed setting
└── io.py               # File I/O utilities
```

**Common utilities:**
- `setup_logging()`: Configure logging
- `get_device()`: Auto-detect CPU/GPU
- `set_seed()`: Reproducibility
- `save_model()`, `load_model()`

## Configuration Files (`configs/`)

**Purpose:** Pre-defined configuration templates.

```
configs/
├── default.yaml        # Default configuration
├── small.yaml          # Small model (quick experiments)
└── large.yaml          # Large model (production)
```

**Example config structure:**
```yaml
model:
  d_model: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 20
  max_seq_len: 256
```

## Data Directory (`data/`)

**Purpose:** Training and evaluation data.

```
data/
├── raw/                # Raw text data
│   └── shakespeare/
│       ├── shakespeare_complete.txt
│       ├── shakespeare25k.txt
│       └── ...
├── processed/          # Preprocessed data
└── samples/            # Sample outputs
```

## Tests (`tests/`)

**Purpose:** Unit and integration tests.

```
tests/
├── __init__.py
├── conftest.py         # Pytest configuration
├── fixtures/           # Test data
├── unit/               # Unit tests
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_tokenizers.py
│   └── ...
└── integration/        # Integration tests
    ├── test_training_pipeline.py
    ├── test_generation_pipeline.py
    └── ...
```

**Run tests:**
```bash
pytest tests/
pytest tests/unit/test_attention.py  # Single test
pytest -v                             # Verbose
pytest --cov                          # Coverage
```

## Experiments (`experiments/`)

**Purpose:** Experiment tracking and notebooks.

```
experiments/
├── notebooks/          # Jupyter notebooks
│   └── exploration.ipynb
└── runs/              # Training runs
    ├── run_001/
    │   ├── config.yaml
    │   ├── model.pt
    │   └── logs/
    └── run_002/
```

## Scripts (`scripts/`)

**Purpose:** Utility scripts for data preparation, analysis, etc.

```
scripts/
├── download_data.py    # Download datasets
├── preprocess.py       # Data preprocessing
└── analyze.py          # Data analysis
```

## Package Installation

### Development Mode

```bash
# Install in development mode
pip install -e .

# Editable install with dev dependencies
pip install -e ".[dev]"
```

**Effect:** Changes to source code are immediately available without reinstall.

### Dependencies

Defined in multiple places:

**`requirements.txt`:** Core dependencies
```
torch>=2.0.0
numpy>=1.21.0
pyyaml>=6.0
tqdm>=4.62.0
```

**`requirements-dev.txt`:** Development dependencies
```
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
```

**`pyproject.toml`:** Modern packaging
```toml
[project]
name = "fundamentallm"
version = "0.1.0"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    ...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    ...
]
```

## Import Structure

### Package Root

```python
from fundamentallm import (
    Config,
    TransformerLM,
    Trainer,
    Generator,
    CharTokenizer,
)
```

### Submodules

```python
# Explicit imports
from fundamentallm.models import TransformerLM
from fundamentallm.training import Trainer
from fundamentallm.data import CharTokenizer
from fundamentallm.generation import Generator

# Component-level
from fundamentallm.models.components import MultiHeadAttention
from fundamentallm.training.callbacks import CheckpointCallback
```

## Design Patterns

### Separation of Concerns

Each module has single responsibility:
- `data/`: Data handling only
- `models/`: Architecture only
- `training/`: Training logic only

### Configuration-Driven

All settings in config files:
```python
config = Config.from_file('config.yaml')
model = create_model(config.model)
trainer = Trainer(config.training)
```

### Composability

Components can be mixed and matched:
```python
# Custom combination
model = TransformerLM(
    vocab_size=256,
    d_model=512,
    num_layers=8,
    num_heads=8,
)

optimizer = create_optimizer(model, config)
scheduler = create_scheduler(optimizer, config)
```

### Extensibility

Easy to add new components:
```python
# New attention mechanism
class MyAttention(nn.Module):
    ...

# Use in model
model = TransformerLM(
    ...,
    attention_class=MyAttention
)
```

## Data Flow

### Training Pipeline

```
1. Data Loading
   Raw text → Tokenizer → Dataset → DataLoader

2. Model Forward
   Tokens → Embedding → Transformer → Logits

3. Loss Computation
   Logits + Targets → Cross-entropy → Loss

4. Optimization
   Loss → Backward → Gradients → Optimizer Step

5. Checkpointing
   Model state → Save to disk
```

### Generation Pipeline

```
1. Load Model
   Checkpoint → TransformerLM

2. Tokenize Prompt
   Text → Tokenizer → Token IDs

3. Generate Loop
   for each step:
     Model(tokens) → Logits
     Sample(logits) → Next token
     Append token

4. Decode
   Token IDs → Tokenizer → Text
```

## Code Style

### Conventions

- **Python:** PEP 8
- **Docstrings:** Google style
- **Type hints:** Throughout
- **Line length:** 88 characters (Black formatter)

### Example

```python
def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: The model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for parameter updates.
        device: Device to train on (cpu or cuda).
    
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch in data_loader:
        # Training logic...
        pass
    
    return total_loss / len(data_loader)
```

## Documentation

### In-Code Documentation

- Docstrings for all public functions/classes
- Inline comments for complex logic
- Type hints for clarity

### External Documentation

- `docs/`: Markdown documentation
- `README.md`: Project overview
- `CONTRIBUTING.md`: Contribution guide
- `DEVELOPER_GUIDE.md`: Developer setup

### VitePress Site

This documentation site:
```
pages/
├── guide/          # User guides
├── concepts/       # Theory
├── modules/        # Implementation details
└── tutorials/      # Step-by-step
```

## Further Reading

- [Installation](../tutorials/installation.md) - Setup guide
- [Developer Guide](../../docs/DEVELOPER_GUIDE.md) - Development setup
- [Contributing](../../CONTRIBUTING.md) - How to contribute
- [Models Module](./models.md) - Model architecture details

## Next Steps

- [Models Module](./models.md) - Dive into model implementation
- [Data Module](./data.md) - Data pipeline details
- [Training Module](./training.md) - Training implementation
- [Generation Module](./generation.md) - Generation details
