"""Training configuration definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator

from .base import BaseConfig


class TrainingConfig(BaseConfig):
    """Training hyperparameters and runtime settings."""

    # Data
    data_path: Optional[Path] = Field(None)
    train_split: float = Field(0.8, gt=0.0, lt=1.0)
    sequence_length: int = Field(256, gt=0)
    batch_size: int = Field(16, gt=0)
    num_workers: int = Field(4, ge=0)

    # Optimization
    learning_rate: float = Field(1e-3, gt=0)
    optimizer: str = Field("adamw", pattern="^(adamw|adam|sgd|rmsprop)$")
    optimizer_eps: float = Field(1e-8, gt=0)
    optimizer_weight_decay: float = Field(0.01, ge=0)
    adam_beta1: float = Field(0.9, ge=0, le=1)
    adam_beta2: float = Field(0.999, ge=0, le=1)

    # Scheduler
    scheduler: str = Field("linear_warmup", pattern="^(constant|linear_warmup|cosine|exponential)$")
    warmup_steps: int = Field(500, ge=0)
    total_steps: Optional[int] = Field(None, gt=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    train_steps_per_epoch: Optional[int] = Field(None, gt=0)

    # Training mechanics
    max_grad_norm: float = Field(1.0, gt=0)
    accumulation_steps: int = Field(1, gt=0)
    gradient_accumulation_steps: Optional[int] = Field(None, gt=0)
    gradient_clip_norm: Optional[float] = Field(None, gt=0)
    gradient_checkpointing: bool = Field(False)
    mixed_precision: bool = Field(False)

    num_epochs: int = Field(10, gt=0)
    max_epochs: Optional[int] = Field(None, gt=0)
    max_steps: Optional[int] = Field(None, gt=0)
    eval_steps: int = Field(100, ge=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)

    # Checkpointing
    checkpoint_dir: Path = Field(Path("./models/checkpoints"))
    checkpoint_keep_last: int = Field(3, gt=0)
    save_every_n_steps: Optional[int] = Field(None, gt=0)
    save_every_n_epochs: int = Field(1, gt=0)

    # Early stopping
    early_stopping_patience: int = Field(0, ge=0)
    early_stopping_metric: str = Field("val_loss")
    early_stopping_mode: str = Field("min", pattern="^(min|max)$")
    early_stopping_min_delta: float = Field(0.0, ge=0.0)

    # Logging / reproducibility
    seed: int = Field(42, ge=0)
    deterministic: bool = Field(True)

    # Device
    device: str = Field("auto")

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data

        # YAML/CLI doc aliases
        if "epochs" in data and "num_epochs" not in data:
            data["num_epochs"] = data.pop("epochs")
        if "max_seq_len" in data and "sequence_length" not in data:
            data["sequence_length"] = data.pop("max_seq_len")
        if "val_split" in data and "train_split" not in data:
            try:
                val = float(data.pop("val_split"))
                data["train_split"] = 1.0 - val
            except Exception:
                data["train_split"] = data.get("train_split")
        return data

    @field_validator("checkpoint_dir", "data_path", mode="before")
    @classmethod
    def resolve_paths(cls, value: Path | None) -> Optional[Path]:
        if value is None:
            return None
        return Path(value).expanduser().resolve()

    @field_validator("device")
    @classmethod
    def normalize_device(cls, value: str) -> str:
        if not value:
            raise ValueError("device must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _synchronize_fields(self) -> "TrainingConfig":
        if self.gradient_accumulation_steps is not None:
            object.__setattr__(self, "accumulation_steps", self.gradient_accumulation_steps)
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "max_grad_norm", self.gradient_clip_norm)
        if self.max_epochs is not None:
            object.__setattr__(self, "num_epochs", self.max_epochs)
        if self.total_steps is None:
            if self.max_steps is not None:
                object.__setattr__(self, "total_steps", self.max_steps)
            elif self.train_steps_per_epoch is not None:
                object.__setattr__(
                    self, "total_steps", self.train_steps_per_epoch * self.num_epochs
                )
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be > 0")
        return self
