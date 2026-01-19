# FundamentaLLM Design Guide

**A comprehensive architectural blueprint for building an educational language model framework from scratch**

## Overview

FundamentaLLM is a minimal character-level transformer language model designed to demonstrate the fundamental principles of modern large language models in an accessible, educational format. It aims to provide a clear, production-quality implementation that balances pedagogical clarity with software engineering best practices, enabling learners to understand how transformers work by training small models on modest hardware. The project achieves this through a modular PyTorch-based architecture featuring well-documented transformer blocks with multi-head attention, a robust training pipeline with early stopping and gradient clipping, and comprehensive configuration management—all while maintaining code quality through type safety, testing, and modern Python tooling.

---

## Table of Contents

1. [Philosophy & Goals](#philosophy--goals)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Core Components](#core-components)
5. [Configuration System](#configuration-system)
6. [Data Pipeline](#data-pipeline)
7. [Model Architecture](#model-architecture)
8. [Training System](#training-system)
9. [Inference & Generation](#inference--generation)
10. [CLI & User Interface](#cli--user-interface)
11. [Logging & Monitoring](#logging--monitoring)
12. [Testing Strategy](#testing-strategy)
13. [Development Workflow](#development-workflow)
14. [Deployment & Distribution](#deployment--distribution)
15. [Extension Points](#extension-points)

---

## Philosophy & Goals

### Primary Goals
1. **Educational First**: Code should be readable, well-documented, and demonstrate ML concepts clearly
2. **Production Ready**: Despite being minimal, follow professional software engineering practices
3. **Extensible**: Easy to add new tokenizers, architectures, or training strategies
4. **Reproducible**: Full control over randomness, versioned dependencies, deterministic training
5. **Type Safe**: Comprehensive type hints with mypy validation
6. **Well Tested**: High test coverage for core functionality

### Non-Goals
- Distributed training (keep it simple)
- Multiple model formats (focus on PyTorch)
- Production deployment at scale (educational use case)

---

## Project Structure

```
fundamentallm/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Run tests, linters on PR
│       └── release.yml               # Package & publish releases
├── configs/
│   ├── default.yaml                  # Default training config
│   ├── small.yaml                    # Fast training for testing
│   └── large.yaml                    # Full-scale training
├── data/
│   ├── raw/                          # Original unprocessed data
│   ├── processed/                    # Tokenized/preprocessed data
│   └── samples/                      # Small datasets for testing
├── docs/
│   ├── architecture.md               # Architecture deep dive
│   ├── getting_started.md            # Quick start guide
│   ├── training_guide.md             # Training best practices
│   └── api_reference.md              # Generated API docs
├── experiments/
│   ├── runs/                         # Experiment tracking (MLflow, W&B)
│   └── notebooks/                    # Jupyter notebooks for analysis
├── scripts/
│   ├── train.py                      # Main training entry point
│   ├── generate.py                   # Text generation CLI
│   ├── evaluate.py                   # Model evaluation
│   ├── preprocess.py                 # Data preprocessing
│   └── export_model.py               # Model export utilities
├── src/
│   └── fundamentallm/
│       ├── __init__.py
│       ├── __main__.py               # Allow `python -m fundamentallm`
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── commands.py           # Click command groups
│       │   └── interactive.py        # REPL interface
│       ├── config/
│       │   ├── __init__.py
│       │   ├── base.py               # Base config classes
│       │   ├── model.py              # Model configs
│       │   ├── training.py           # Training configs
│       │   └── validation.py         # Config validators
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py            # PyTorch Dataset classes
│       │   ├── loaders.py            # DataLoader builders
│       │   ├── preprocessing.py      # Text preprocessing
│       │   └── tokenizers/
│       │       ├── __init__.py
│       │       ├── base.py           # Abstract tokenizer
│       │       ├── character.py      # Character tokenizer
│       │       └── bpe.py            # Future: BPE tokenizer
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py               # Base model interface
│       │   ├── transformer.py        # Transformer implementation
│       │   ├── components/
│       │   │   ├── __init__.py
│       │   │   ├── attention.py      # Attention mechanisms
│       │   │   ├── embeddings.py     # Embedding layers
│       │   │   ├── feedforward.py    # FFN layers
│       │   │   └── normalization.py  # LayerNorm, RMSNorm
│       │   └── registry.py           # Model factory/registry
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py            # Main Trainer class
│       │   ├── callbacks.py          # Training callbacks
│       │   ├── metrics.py            # Metric computation
│       │   ├── optimizers.py         # Optimizer builders
│       │   ├── schedulers.py         # LR schedulers
│       │   └── early_stopping.py     # Early stopping logic
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── generator.py          # Text generation
│       │   ├── sampling.py           # Sampling strategies
│       │   └── constraints.py        # Generation constraints
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py          # Model evaluation
│       │   └── benchmarks.py         # Standard benchmarks
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── checkpoint.py         # Save/load checkpoints
│       │   ├── device.py             # Device management
│       │   ├── logging.py            # Logging setup
│       │   ├── random.py             # Seed control
│       │   ├── paths.py              # Path utilities
│       │   └── visualization.py      # Training plots
│       └── version.py                # Version info
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── test_tokenizers.py
│   │   ├── test_models.py
│   │   ├── test_data.py
│   │   └── test_training.py
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   └── test_training_loop.py
│   └── fixtures/
│       ├── sample_data.txt
│       └── test_configs.yaml
├── .env.example                      # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml           # Pre-commit hooks
├── pyproject.toml                    # Project metadata, deps, tool configs
├── setup.py                          # Editable install support
├── requirements.txt                  # Pinned dependencies
├── requirements-dev.txt              # Development dependencies
├── Makefile                          # Development shortcuts
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## Architecture Overview

### Design Principles

#### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- **Data**: Loading, preprocessing, tokenization
- **Models**: Architecture definitions only
- **Training**: Training loop, optimization, callbacks
- **Generation**: Inference and text generation
- **Utils**: Cross-cutting concerns (logging, checkpointing)

#### 2. **Dependency Inversion**
- High-level modules depend on abstractions (interfaces)
- Use abstract base classes for extensibility
- Example: `BaseTokenizer`, `BaseModel`, `BaseCallback`

#### 3. **Configuration Over Code**
- All hyperparameters in config files (YAML)
- No magic numbers in code
- Config validation at startup
- Environment variable overrides supported

#### 4. **Testability First**
- Pure functions where possible
- Dependency injection for testing
- Mock-friendly interfaces
- Fixtures for common test scenarios

### System Layers

```
┌─────────────────────────────────────────┐
│         CLI / User Interface            │
│  (Click commands, REPL, Jupyter)        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Training Orchestration           │
│  (Trainer, Callbacks, Metrics)          │
└─────┬─────────────────────────┬─────────┘
      │                         │
┌─────▼─────────┐     ┌─────────▼─────────┐
│  Model Layer  │     │   Data Pipeline   │
│  (Transformer)│     │  (Loaders, Token) │
└───────────────┘     └───────────────────┘
      │                         │
┌─────▼─────────────────────────▼─────────┐
│           Core Utilities                │
│  (Logging, Checkpointing, Device)       │
└─────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration System

**Location**: `src/fundamentallm/config/`

#### Design Pattern: Pydantic Models

```python
# config/base.py
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Optional

class BaseConfig(BaseModel):
    """Base configuration with validation."""
    
    class Config:
        validate_assignment = True
        extra = "forbid"  # Fail on unknown fields
        
    def save(self, path: Path) -> None:
        """Save config to YAML."""
        pass
    
    @classmethod
    def from_yaml(cls, path: Path) -> "BaseConfig":
        """Load config from YAML."""
        pass

# config/model.py
class TransformerConfig(BaseConfig):
    """Transformer architecture configuration."""
    
    vocab_size: int = Field(..., gt=0)
    d_model: int = Field(512, gt=0)
    num_heads: int = Field(8, gt=0)
    num_layers: int = Field(6, gt=0, le=24)
    sequence_length: int = Field(256, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    ffn_expansion: int = Field(4, gt=0)
    
    # Positional encoding type
    pos_encoding: str = Field("learned", regex="^(learned|sinusoidal|rope)$")
    
    @validator("d_model")
    def d_model_divisible_by_heads(cls, v, values):
        num_heads = values.get("num_heads", 8)
        if v % num_heads != 0:
            raise ValueError(f"d_model ({v}) must be divisible by num_heads ({num_heads})")
        return v
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

# config/training.py
class TrainingConfig(BaseConfig):
    """Training hyperparameters."""
    
    # Data
    data_path: Path = Field(...)
    train_split: float = Field(0.9, gt=0.0, lt=1.0)
    sequence_length: int = Field(256, gt=0)
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(4, ge=0)
    
    # Training
    max_epochs: int = Field(10, gt=0)
    max_steps: Optional[int] = Field(None, gt=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    gradient_clip_norm: Optional[float] = Field(1.0, gt=0)
    
    # Optimization
    optimizer: str = Field("adamw", regex="^(adam|adamw|sgd)$")
    learning_rate: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    adam_beta1: float = Field(0.9, ge=0, le=1)
    adam_beta2: float = Field(0.999, ge=0, le=1)
    adam_epsilon: float = Field(1e-8, gt=0)
    
    # Learning rate schedule
    lr_scheduler: str = Field("cosine", regex="^(constant|linear|cosine|cosine_with_restarts)$")
    warmup_steps: int = Field(100, ge=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    
    # Regularization
    use_mixed_precision: bool = Field(True)
    dropout: float = Field(0.1, ge=0, le=1)
    
    # Checkpointing
    checkpoint_dir: Path = Field(Path("checkpoints"))
    save_every_n_steps: Optional[int] = Field(None, gt=0)
    save_every_n_epochs: int = Field(1, gt=0)
    keep_last_n_checkpoints: int = Field(3, gt=0)
    
    # Early stopping
    early_stopping_patience: int = Field(5, gt=0)
    early_stopping_metric: str = Field("val_loss")
    early_stopping_mode: str = Field("min", regex="^(min|max)$")
    
    # Logging
    log_every_n_steps: int = Field(100, gt=0)
    eval_every_n_steps: Optional[int] = Field(None, gt=0)
    eval_every_n_epochs: int = Field(1, gt=0)
    
    # Reproducibility
    seed: int = Field(42, ge=0)
    deterministic: bool = Field(True)
    
    # Device
    device: str = Field("cuda", regex="^(cpu|cuda|mps)$")
    
    @validator("checkpoint_dir", "data_path")
    def resolve_paths(cls, v):
        return Path(v).resolve()
```

#### Configuration Hierarchy

```yaml
# configs/default.yaml
model:
  vocab_size: null  # Set at runtime
  d_model: 512
  num_heads: 8
  num_layers: 6
  sequence_length: 256
  dropout: 0.1
  pos_encoding: learned

training:
  data_path: data/processed/shakespeare.txt
  batch_size: 32
  max_epochs: 10
  learning_rate: 3e-4
  seed: 42
  
  # Early stopping
  early_stopping_patience: 5
  early_stopping_metric: val_loss
  
  # Checkpointing
  checkpoint_dir: checkpoints/
  save_every_n_epochs: 1
  keep_last_n_checkpoints: 3

generation:
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  max_tokens: 200
  
logging:
  level: INFO
  log_to_file: true
  log_to_wandb: false
  wandb_project: fundamentallm
```

---

### 2. Data Pipeline

**Location**: `src/fundamentallm/data/`

#### Tokenizer Architecture

```python
# data/tokenizers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
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
        """Serialize tokenizer to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTokenizer":
        """Load tokenizer from disk."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def batch_decode(self, token_ids: List[List[int]]) -> List[str]:
        """Decode multiple token sequences."""
        return [self.decode(tokens) for tokens in token_ids]

# data/tokenizers/character.py
import json
from collections import Counter
from typing import Dict, List

class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer with special tokens."""
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self, min_frequency: int = 1):
        self.min_frequency = min_frequency
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self._vocab_size = 0
        self._is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)
        
        # Build vocabulary
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        vocab.extend([
            char for char, count in char_counts.most_common()
            if count >= self.min_frequency
        ])
        
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self._vocab_size = len(vocab)
        self._is_trained = True
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.char_to_id[self.BOS_TOKEN])
        
        unk_id = self.char_to_id[self.UNK_TOKEN]
        tokens.extend([self.char_to_id.get(char, unk_id) for char in text])
        
        if add_special_tokens:
            tokens.append(self.char_to_id[self.EOS_TOKEN])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        special_ids = {
            self.char_to_id[self.PAD_TOKEN],
            self.char_to_id[self.BOS_TOKEN],
            self.char_to_id[self.EOS_TOKEN]
        } if skip_special_tokens else set()
        
        chars = [
            self.id_to_char.get(token, self.UNK_TOKEN)
            for token in tokens
            if token not in special_ids
        ]
        return "".join(chars)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def pad_token_id(self) -> int:
        return self.char_to_id[self.PAD_TOKEN]
    
    def save(self, path: Path) -> None:
        """Save tokenizer state."""
        state = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "vocab_size": self._vocab_size,
            "min_frequency": self.min_frequency,
            "version": "1.0"
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        """Load tokenizer from disk."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        tokenizer = cls(min_frequency=state["min_frequency"])
        tokenizer.char_to_id = state["char_to_id"]
        tokenizer.id_to_char = {int(k): v for k, v in state["id_to_char"].items()}
        tokenizer._vocab_size = state["vocab_size"]
        tokenizer._is_trained = True
        return tokenizer
```

#### Dataset Implementation

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path

class LanguageModelDataset(Dataset):
    """Dataset for autoregressive language modeling."""
    
    def __init__(
        self,
        token_ids: torch.Tensor,
        sequence_length: int,
        stride: Optional[int] = None
    ):
        """
        Args:
            token_ids: Tensor of token IDs
            sequence_length: Length of each sequence
            stride: Stride for overlapping sequences (default: sequence_length)
        """
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        
        # Calculate number of sequences
        self.num_sequences = max(
            0,
            (len(token_ids) - sequence_length - 1) // self.stride + 1
        )
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) pair."""
        start = idx * self.stride
        end = start + self.sequence_length + 1
        
        sequence = self.token_ids[start:end]
        
        # Input: all but last token
        # Target: all but first token (next token prediction)
        return sequence[:-1], sequence[1:]

# data/loaders.py
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import torch

def create_dataloaders(
    text: str,
    tokenizer: BaseTokenizer,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Tokenize entire text
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split at token level (not character level!)
    train_size = int(len(token_ids) * config.train_split)
    train_tokens = token_ids[:train_size]
    val_tokens = token_ids[train_size:]
    
    # Create datasets
    train_dataset = LanguageModelDataset(
        train_tokens,
        sequence_length=config.sequence_length
    )
    val_dataset = LanguageModelDataset(
        val_tokens,
        sequence_length=config.sequence_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader
```

---

### 3. Model Architecture

**Location**: `src/fundamentallm/models/`

#### Model Registry Pattern

```python
# models/registry.py
from typing import Dict, Type, Callable
from .base import BaseModel

class ModelRegistry:
    """Registry for model architectures."""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model class."""
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: "ModelConfig") -> BaseModel:
        """Create model instance from registry."""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name](config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List registered models."""
        return list(cls._registry.keys())

# Usage in transformer.py
@ModelRegistry.register("transformer")
class Transformer(BaseModel):
    ...
```

#### Improved Transformer Implementation

```python
# models/transformer.py
import torch
import torch.nn as nn
from typing import Optional
from .base import BaseModel
from .components.attention import MultiHeadAttention
from .components.embeddings import PositionalEncoding
from .components.normalization import RMSNorm
from ..config.model import TransformerConfig

class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.ffn_expansion),
            nn.GELU(),
            nn.Linear(config.d_model * config.ffn_expansion, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Pre-normalization (better training stability)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention + residual
        x = x + self.attention(self.norm1(x), mask)
        # Pre-norm FFN + residual
        x = x + self.feed_forward(self.norm2(x))
        return x

@ModelRegistry.register("transformer")
class Transformer(BaseModel):
    """Transformer language model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = self._create_position_encoding(config)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_norm = RMSNorm(config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.output_norm = RMSNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_position_encoding(self, config: TransformerConfig):
        """Create positional encoding based on config."""
        if config.pos_encoding == "learned":
            return nn.Embedding(config.sequence_length, config.d_model)
        elif config.pos_encoding == "sinusoidal":
            return PositionalEncoding(config.d_model, config.sequence_length)
        elif config.pos_encoding == "rope":
            # TODO: Implement RoPE
            raise NotImplementedError("RoPE not yet implemented")
        else:
            raise ValueError(f"Unknown positional encoding: {config.pos_encoding}")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Add positional encoding
        if isinstance(self.position_encoding, nn.Embedding):
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.position_encoding(pos_ids)
            x = token_emb + pos_emb
        else:
            x = self.position_encoding(token_emb)
        
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    @torch.no_grad()
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

### 4. Training System

**Location**: `src/fundamentallm/training/`

#### Trainer Class

```python
# training/trainer.py
from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.base import BaseModel
from ..config.training import TrainingConfig
from .callbacks import CallbackList, Callback
from .metrics import MetricTracker
from .early_stopping import EarlyStopping
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import get_logger
from ..utils.random import set_seed

logger = get_logger(__name__)

class Trainer:
    """Main training orchestrator."""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device setup
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimization
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None
        
        # Gradient clipping
        self.gradient_clip_norm = config.gradient_clip_norm
        
        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        # Metrics
        self.metrics = MetricTracker()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            metric=config.early_stopping_metric,
            mode=config.early_stopping_mode
        )
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=config.keep_last_n_checkpoints
        )
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        
        # Reproducibility
        set_seed(config.seed)
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
        elif self.config.lr_scheduler == "constant":
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        self.callbacks.on_train_begin(self)
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self)
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = self._validate_epoch()
            
            # Update metrics
            self.metrics.update({
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
            
            # Early stopping check
            val_loss = val_metrics['loss']
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save best model
            if self.early_stopping.is_best:
                self.save_checkpoint("best")
                self.best_metric = val_loss
            
            self.callbacks.on_epoch_end(self)
        
        self.callbacks.on_train_end(self)
        
        return self.metrics.get_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.config.max_epochs}",
            disable=not torch.cuda.is_available()
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.scaler:
                with autocast('cuda'):
                    logits = self.model(inputs)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            else:
                logits = self.model(inputs)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Step callback
            if self.global_step % self.config.log_every_n_steps == 0:
                self.callbacks.on_step(self, loss.item())
        
        avg_loss = epoch_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.scaler:
                with autocast('cuda'):
                    logits = self.model(inputs)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
            else:
                logits = self.model(inputs)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for the epoch."""
        logger.info(
            f"Epoch {self.current_epoch+1}/{self.config.max_epochs} - "
            f"train_loss: {train_metrics['loss']:.4f}, "
            f"train_ppl: {train_metrics['perplexity']:.2f}, "
            f"val_loss: {val_metrics['loss']:.4f}, "
            f"val_ppl: {val_metrics['perplexity']:.2f}, "
            f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"
        )
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "metrics": self.metrics.get_history()
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        self.checkpoint_manager.save(checkpoint, name)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")
```

---

### 5. CLI Interface

**Location**: `src/fundamentallm/cli/`

#### Click-based CLI

```python
# cli/commands.py
import click
from pathlib import Path
from ..config.training import TrainingConfig
from ..config.model import TransformerConfig
from ..utils.logging import setup_logging

@click.group()
@click.version_option()
def cli():
    """TinyPythonLLM - Minimal educational language model."""
    pass

@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True), help="Config YAML file")
@click.option("--output-dir", type=click.Path(), default="checkpoints", help="Output directory")
@click.option("--epochs", type=int, help="Number of epochs")
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--seed", type=int, help="Random seed")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), help="Device")
def train(data_path, config, output_dir, **overrides):
    """Train a language model."""
    setup_logging()
    
    # Load config
    if config:
        train_config = TrainingConfig.from_yaml(Path(config))
    else:
        train_config = TrainingConfig(data_path=Path(data_path))
    
    # Apply CLI overrides
    for key, value in overrides.items():
        if value is not None:
            setattr(train_config, key, value)
    
    # Run training
    from ..training.trainer import Trainer
    # ... training logic
    
@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--prompt", help="Initial prompt")
@click.option("--max-tokens", type=int, default=200)
@click.option("--temperature", type=float, default=0.8)
@click.option("--interactive", is_flag=True, help="Launch interactive mode")
def generate(model_path, prompt, max_tokens, temperature, interactive):
    """Generate text from a trained model."""
    from ..generation.generator import TextGenerator
    
    generator = TextGenerator.from_checkpoint(model_path)
    
    if interactive:
        from .interactive import InteractiveREPL
        repl = InteractiveREPL(generator)
        repl.run()
    else:
        if not prompt:
            click.echo("Error: --prompt required in non-interactive mode")
            return
        
        text = generator.generate(prompt, max_tokens, temperature)
        click.echo(text)

@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Output JSON file")
def evaluate(model_path, data_path, output):
    """Evaluate a trained model."""
    from ..evaluation.evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator.from_checkpoint(model_path)
    results = evaluator.evaluate(data_path)
    
    click.echo(f"Loss: {results['loss']:.4f}")
    click.echo(f"Perplexity: {results['perplexity']:.2f}")
    
    if output:
        import json
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    cli()
```

---

### 6. Testing Strategy

**Location**: `tests/`

#### Test Structure

```python
# tests/unit/test_tokenizers.py
import pytest
from fundamentallm.data.tokenizers.character import CharacterTokenizer

class TestCharacterTokenizer:
    """Test suite for character tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance."""
        tokenizer = CharacterTokenizer()
        tokenizer.train(["hello world", "test data"])
        return tokenizer
    
    def test_vocab_size(self, tokenizer):
        """Test vocabulary size includes special tokens."""
        # 4 special tokens + unique chars
        assert tokenizer.vocab_size > 4
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encoding and decoding produces original text."""
        text = "hello"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_unknown_character_handling(self, tokenizer):
        """Test that unknown characters are handled."""
        # Character not in training data
        tokens = tokenizer.encode("★")
        assert tokenizer.char_to_id[tokenizer.UNK_TOKEN] in tokens
    
    def test_special_tokens(self, tokenizer):
        """Test special token encoding."""
        text = "test"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert tokens[0] == tokenizer.char_to_id[tokenizer.BOS_TOKEN]
        assert tokens[-1] == tokenizer.char_to_id[tokenizer.EOS_TOKEN]
    
    def test_save_load(self, tokenizer, tmp_path):
        """Test serialization and deserialization."""
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(save_path)
        
        loaded = CharacterTokenizer.load(save_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.char_to_id == tokenizer.char_to_id

# tests/integration/test_training_loop.py
import pytest
import torch
from fundamentallm.training.trainer import Trainer
from fundamentallm.models.transformer import Transformer
from fundamentallm.config.model import TransformerConfig
from fundamentallm.config.training import TrainingConfig

class TestTrainingLoop:
    """Integration test for training pipeline."""
    
    @pytest.fixture
    def small_config(self):
        """Small model for fast testing."""
        return TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=2,
            num_layers=2,
            sequence_length=32
        )
    
    @pytest.fixture
    def train_config(self, tmp_path):
        """Training config for tests."""
        return TrainingConfig(
            data_path=tmp_path / "data.txt",
            batch_size=4,
            max_epochs=2,
            sequence_length=32,
            checkpoint_dir=tmp_path / "checkpoints"
        )
    
    def test_training_runs(self, small_config, train_config, sample_dataloader):
        """Test that training completes without errors."""
        model = Transformer(small_config)
        train_loader, val_loader = sample_dataloader
        
        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        metrics = trainer.train()
        
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert len(metrics["train_loss"]) == train_config.max_epochs
    
    def test_checkpoint_saving(self, small_config, train_config, sample_dataloader):
        """Test checkpoint saving and loading."""
        model = Transformer(small_config)
        train_loader, val_loader = sample_dataloader
        
        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        trainer.save_checkpoint("test")
        
        # Check checkpoint exists
        checkpoint_path = train_config.checkpoint_dir / "test.pt"
        assert checkpoint_path.exists()
        
        # Load and verify
        trainer.load_checkpoint(checkpoint_path)
        assert trainer.global_step >= 0
```

---

### 7. Development Workflow

#### Setup Instructions

```bash
# pyproject.toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fundamentallm"
version = "2.0.0"
description = "Minimal educational language model framework"
authors = [{name = "Your Name", email = "you@example.com"}]
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Education",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "torch>=2.0.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pylint>=2.17.0",
    "pre-commit>=3.3.0",
]

experiment = [
    "wandb>=0.15.0",
    "tensorboard>=2.13.0",
    "jupyter>=1.0.0",
]

[project.scripts]
fundamentallm = "fundamentallm.cli.commands:cli"

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --cov=src/fundamentallm --cov-report=html --cov-report=term"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*.py"]
```

#### Makefile

```makefile
.PHONY: help install install-dev test lint format clean

help:
	@echo "TinyPythonLLM v2 Development Commands"
	@echo ""
	@echo "  install        - Install package"
	@echo "  install-dev    - Install with dev dependencies"
	@echo "  test           - Run test suite"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  lint           - Run linters (mypy, pylint)"
	@echo "  format         - Format code (black, isort)"
	@echo "  format-check   - Check formatting without changes"
	@echo "  clean          - Remove build artifacts"
	@echo "  docs           - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,experiment]"
	pre-commit install

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src/fundamentallm --cov-report=html

lint:
	mypy src/fundamentallm
	pylint src/fundamentallm

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html
```

#### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, types-PyYAML]
```

---

## Summary of Improvements Over Legacy Codebase

### 1. **Configuration Management**
- ✅ Pydantic models with validation
- ✅ YAML-based configuration
- ✅ Environment variable overrides
- ✅ Configuration hierarchy

### 2. **Data Pipeline**
- ✅ Token-level train/val split (fixes data leakage)
- ✅ Abstract tokenizer interface
- ✅ Special token support
- ✅ Unknown character handling

### 3. **Model Architecture**
- ✅ Pre-normalization (better stability)
- ✅ RMSNorm option (better performance)
- ✅ Weight tying (reduced parameters)
- ✅ Multiple positional encoding options
- ✅ Model registry pattern

### 4. **Training**
- ✅ Callback system for extensibility
- ✅ Early stopping with validation monitoring
- ✅ Gradient clipping
- ✅ Proper checkpoint management
- ✅ Perplexity tracking
- ✅ Reproducible training (seeds)

### 5. **Code Quality**
- ✅ Type hints throughout
- ✅ Comprehensive testing
- ✅ Linting and formatting
- ✅ Pre-commit hooks
- ✅ CI/CD pipeline

### 6. **Developer Experience**
- ✅ Virtual environment support
- ✅ Editable install with `pip install -e .`
- ✅ Click-based CLI
- ✅ Rich progress bars and logging
- ✅ Development Makefile

### 7. **Extensibility**
- ✅ Plugin architecture for tokenizers
- ✅ Model registry
- ✅ Callback system
- ✅ Easy to add new components

---

## Next Steps for Implementation

1. **Phase 1: Core Infrastructure**
   - Set up project structure
   - Implement configuration system
   - Create base abstractions

2. **Phase 2: Data Pipeline**
   - Implement tokenizers
   - Build dataset and dataloader
   - Add preprocessing utilities

3. **Phase 3: Model**
   - Implement transformer components
   - Build complete model
   - Add model registry

4. **Phase 4: Training**
   - Implement trainer class
   - Add callbacks and metrics
   - Implement early stopping

5. **Phase 5: Generation & Evaluation**
   - Text generation utilities
   - Sampling strategies
   - Evaluation metrics

6. **Phase 6: CLI & Testing**
   - Click commands
   - Interactive REPL
   - Comprehensive test suite

7. **Phase 7: Documentation & Polish**
   - API documentation
   - User guides
   - Example notebooks

---

**This design guide provides a production-ready foundation while maintaining educational clarity. Follow it to build a maintainable, extensible, and professional language model framework.**
