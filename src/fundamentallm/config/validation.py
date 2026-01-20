"""Configuration validation helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_training_config(config: dict[str, Any] | Any) -> list[str]:
    """Validate training configuration and return list of issues.

    Args:
        config: TrainingConfig dictionary or Pydantic object.

    Returns:
        List of validation issues (empty if all valid).
    """
    issues = []
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config

    # Extract values with defaults
    num_epochs = config_dict.get("num_epochs", 1)
    batch_size = config_dict.get("batch_size", 32)
    accumulation_steps = config_dict.get("accumulation_steps", 1)
    eval_steps = config_dict.get("eval_steps", 100)
    learning_rate = config_dict.get("learning_rate", 1e-3)
    max_grad_norm = config_dict.get("max_grad_norm", 1.0)
    warmup_steps = config_dict.get("warmup_steps", 0)
    sequence_length = config_dict.get("sequence_length", 512)

    # Validate num_epochs
    if num_epochs < 1:
        issues.append(f"num_epochs must be >= 1, got {num_epochs}")
    if num_epochs > 10000:
        issues.append(f"num_epochs seems too high ({num_epochs}), consider lower values")

    # Validate batch_size
    if batch_size < 1:
        issues.append(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > 2048:
        issues.append(f"batch_size is very large ({batch_size}), may cause OOM")

    # Validate accumulation_steps
    if accumulation_steps < 1:
        issues.append(f"accumulation_steps must be >= 1, got {accumulation_steps}")
    if accumulation_steps > batch_size:
        issues.append(
            f"accumulation_steps ({accumulation_steps}) > batch_size ({batch_size}) "
            "is inefficient"
        )

    # Validate eval_steps
    if eval_steps < 0:
        issues.append(f"eval_steps must be >= 0, got {eval_steps}")
    if eval_steps == 0:
        logger.debug("eval_steps=0: validation will be skipped during training")

    # Validate learning_rate
    if learning_rate <= 0:
        issues.append(f"learning_rate must be > 0, got {learning_rate}")
    if learning_rate > 0.1:
        issues.append(f"learning_rate seems high ({learning_rate}), consider 1e-4 to 1e-3")
    if learning_rate < 1e-6:
        issues.append(f"learning_rate seems very low ({learning_rate}), may not learn")

    # Validate max_grad_norm
    if max_grad_norm <= 0:
        issues.append(f"max_grad_norm must be > 0, got {max_grad_norm}")
    if max_grad_norm > 10:
        issues.append(f"max_grad_norm is high ({max_grad_norm}), may limit gradient clipping")

    # Validate warmup_steps
    if warmup_steps < 0:
        issues.append(f"warmup_steps must be >= 0, got {warmup_steps}")
    if warmup_steps > 10000:
        issues.append(f"warmup_steps seems high ({warmup_steps}), consider lower value")

    # Validate sequence_length
    if sequence_length < 1:
        issues.append(f"sequence_length must be >= 1, got {sequence_length}")
    if sequence_length > 8192:
        issues.append(f"sequence_length is very long ({sequence_length}), may cause OOM")

    return issues


def validate_model_config(config: dict[str, Any] | Any) -> list[str]:
    """Validate model configuration.

    Args:
        config: TransformerConfig dictionary or Pydantic object.

    Returns:
        List of validation issues (empty if all valid).
    """
    issues = []
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config

    vocab_size = config_dict.get("vocab_size", 256)
    d_model = config_dict.get("d_model", 512)
    num_heads = config_dict.get("num_heads", 8)
    num_layers = config_dict.get("num_layers", 6)
    sequence_length = config_dict.get("sequence_length", 512)

    # Validate vocab_size
    if vocab_size < 2:
        issues.append(f"vocab_size must be >= 2, got {vocab_size}")

    # Validate d_model
    if d_model < 64:
        issues.append(f"d_model seems too small ({d_model}), consider >= 64")
    if d_model > 8192:
        issues.append(f"d_model is very large ({d_model}), may cause OOM")

    # Validate num_heads
    if num_heads < 1:
        issues.append(f"num_heads must be >= 1, got {num_heads}")
    if d_model % num_heads != 0:
        issues.append(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if num_heads > d_model // 64:
        issues.append(
            f"num_heads ({num_heads}) too high relative to d_model ({d_model}), "
            f"d_model//num_heads would be < 64"
        )

    # Validate num_layers
    if num_layers < 1:
        issues.append(f"num_layers must be >= 1, got {num_layers}")
    if num_layers > 128:
        issues.append(f"num_layers is very large ({num_layers}), may cause OOM/slowdown")

    # Validate sequence_length
    if sequence_length < 1:
        issues.append(f"sequence_length must be >= 1, got {sequence_length}")

    return issues


def warn_on_issues(issues: list[str], config_name: str = "config") -> None:
    """Log warnings for configuration issues.

    Args:
        issues: List of issue strings from validation.
        config_name: Name of config for logging context.
    """
    if not issues:
        logger.debug(f"{config_name} validation passed")
        return

    logger.warning(f"{config_name} validation found {len(issues)} issue(s):")
    for issue in issues:
        logger.warning(f"  - {issue}")
