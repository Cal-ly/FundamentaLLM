"""Click-based command line interface for FundamentaLLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
import torch
import yaml

from fundamentallm.config import TransformerConfig
from fundamentallm.config.training import TrainingConfig
from fundamentallm.config.validation import (
    validate_model_config,
    validate_training_config,
    warn_on_issues,
)
from fundamentallm.data.loaders import create_dataloaders
from fundamentallm.data.tokenizers.character import CharacterTokenizer
from fundamentallm.evaluation.evaluator import ModelEvaluator
from fundamentallm.generation.generator import TextGenerator
from fundamentallm.training.losses import LanguageModelingLoss
from fundamentallm.training.optimizers import OptimizerBuilder
from fundamentallm.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    LearningRateScheduler,
    LinearWarmup,
)
from fundamentallm.training.trainer import Trainer
from fundamentallm.utils.device import validate_device
from fundamentallm.utils.logging import get_logger, setup_logging
from fundamentallm.utils.paths import ensure_dir
from fundamentallm.utils.random import set_seed
from fundamentallm.version import __version__

logger = get_logger(__name__)


def _load_configs(
    config_path: Optional[Path],
    data_path: Path,
    vocab_size: int,
) -> Tuple[TrainingConfig, TransformerConfig]:
    if config_path is None:
        train_cfg = TrainingConfig(data_path=data_path)
        model_cfg = TransformerConfig(
            vocab_size=vocab_size,
            sequence_length=train_cfg.sequence_length,
        )
        return train_cfg, model_cfg

    raw = yaml.safe_load(config_path.read_text()) or {}
    if isinstance(raw, dict) and ("model" in raw or "training" in raw):
        train_payload = raw.get("training", {}) or {}
        train_payload.setdefault("data_path", str(data_path))
        training_config = TrainingConfig.model_validate(train_payload)

        model_payload = raw.get("model", {}) or {}
        model_payload["vocab_size"] = model_payload.get("vocab_size") or vocab_size
        if "sequence_length" not in model_payload and training_config.sequence_length:
            model_payload["sequence_length"] = training_config.sequence_length
        model_config = TransformerConfig.model_validate(model_payload)
        return training_config, model_config

    training_config = TrainingConfig.from_yaml(config_path)
    training_config.data_path = data_path
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        sequence_length=training_config.sequence_length,
    )
    return training_config, model_config


def _apply_overrides(
    training_config: TrainingConfig,
    *,
    output_dir: Optional[Path],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    device: Optional[str],
    seed: Optional[int],
    val_split: Optional[float],
    sequence_length: Optional[int],
    dropout: Optional[float],
    mixed_precision: Optional[bool],
    gradient_clip: Optional[float],
) -> None:
    if output_dir is not None:
        output_path = Path(output_dir).expanduser()
        # If relative path and doesn't already start with 'models/', prepend it
        if not output_path.is_absolute() and not str(output_path).startswith("models"):
            output_path = Path("models") / output_path
        training_config.checkpoint_dir = output_path
    if epochs is not None:
        training_config.num_epochs = epochs
    if batch_size is not None:
        training_config.batch_size = batch_size
    if learning_rate is not None:
        training_config.learning_rate = learning_rate
    if device is not None:
        training_config.device = device
    if seed is not None:
        training_config.seed = seed
    if val_split is not None:
        training_config.train_split = 1.0 - val_split
    if sequence_length is not None:
        training_config.sequence_length = sequence_length
    if dropout is not None:
        training_config.dropout = dropout
    if mixed_precision is not None:
        training_config.mixed_precision = mixed_precision
    if gradient_clip is not None:
        training_config.gradient_clip_norm = gradient_clip


def _apply_model_overrides(
    model_config: TransformerConfig,
    *,
    model_dim: Optional[int],
    num_heads: Optional[int],
    num_layers: Optional[int],
    dropout: Optional[float],
    sequence_length: Optional[int],
) -> None:
    if model_dim is not None:
        model_config.d_model = model_dim
    if num_heads is not None:
        model_config.num_heads = num_heads
    if num_layers is not None:
        model_config.num_layers = num_layers
    if dropout is not None:
        model_config.dropout = dropout
    if sequence_length is not None:
        model_config.sequence_length = sequence_length


HEAD_DIM_FLOOR = 8


def _adjust_num_heads(model_config: TransformerConfig) -> Optional[Tuple[int, int, int]]:
    """Auto-fix num_heads to satisfy divisibility and head-dim floor (>=HEAD_DIM_FLOOR).

    Returns old/new heads and d_model when an adjustment occurs, otherwise None.
    """

    d_model = model_config.d_model
    adjusted = _safe_num_heads(d_model, model_config.num_heads)

    if adjusted == model_config.num_heads:
        return None

    old = model_config.num_heads
    model_config.num_heads = adjusted
    return old, adjusted, d_model


def _safe_num_heads(d_model: int, requested_heads: int) -> int:
    """Compute a safe num_heads without mutating configs."""

    max_heads_by_dim = max(d_model // HEAD_DIM_FLOOR, 1)
    target = min(requested_heads, max_heads_by_dim)

    adjusted = target
    while adjusted > 1 and d_model % adjusted != 0:
        adjusted -= 1
    if adjusted < 1:
        adjusted = 1
    return adjusted


def _enforce_model_config(model_config: TransformerConfig, auto_fix: bool) -> None:
    """Convert critical model config issues into errors, with optional auto-fix."""

    issues = validate_model_config(model_config)
    if not issues:
        return

    # Identify critical head-related issues.
    critical = [issue for issue in issues if "num_heads" in issue or "divisible" in issue]

    if auto_fix:
        fix = _adjust_num_heads(model_config)
        if fix:
            old, new, d_model = fix
            click.echo(
                f"Warning: auto-fix adjusted num_heads from {old} to {new} (d_model={d_model}, head_dim={d_model // new})"
            )
        issues = validate_model_config(model_config)
        warn_on_issues(issues, "TransformerConfig")
        remaining_critical = [
            issue for issue in issues if "num_heads" in issue or "divisible" in issue
        ]
        if remaining_critical:
            raise click.ClickException(
                "TransformerConfig still invalid after auto-fix: " + "; ".join(remaining_critical)
            )
        return

    if critical:
        d_model = model_config.d_model
        rec_heads = _safe_num_heads(d_model, model_config.num_heads)
        recommendation = (
            f" Try num_heads={rec_heads} (head_dim={d_model // rec_heads}) or enable --auto-fix-config."
            if rec_heads != model_config.num_heads
            else ""
        )
        raise click.ClickException(
            "Invalid TransformerConfig: " + "; ".join(critical) + recommendation
        )

    warn_on_issues(issues, "TransformerConfig")


def _print_eval_results(results: dict) -> None:
    click.echo("Evaluation results (validation split)")
    click.echo("-" * 80)
    for key in ("loss", "perplexity", "accuracy"):
        value = results.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            click.echo(f"{key:12s}: {value:.4f}")
        else:
            click.echo(f"{key:12s}: {value}")
    click.echo("-" * 80)


def _save_validation_split(text: str, tokenizer: CharacterTokenizer, cfg: TrainingConfig) -> None:
    tokens = tokenizer.encode(text)
    train_size = max(int(len(tokens) * cfg.train_split), 1)
    val_tokens = tokens[train_size:]
    if not val_tokens:
        logger.info("Validation split empty; skipping validation export")
        return

    val_text = tokenizer.decode(val_tokens)
    out_path = Path(cfg.checkpoint_dir) / "validation.txt"
    out_path.write_text(val_text, encoding="utf-8")
    logger.info(f"Validation split written to {out_path}")


def _cleanup_intermediate_checkpoints(checkpoint_dir: Path) -> None:
    """Remove intermediate epoch and step checkpoints, keeping only final and best."""
    checkpoint_dir = Path(checkpoint_dir)
    final_path = checkpoint_dir / "final_model.pt"
    best_path = checkpoint_dir / "best.pt"

    if not final_path.exists():
        logger.debug("No final_model.pt found; skipping cleanup")
        return

    deleted = []
    # Remove epoch checkpoints
    for ckpt in checkpoint_dir.glob("epoch_*.pt"):
        try:
            ckpt.unlink()
            deleted.append(ckpt.name)
        except OSError as exc:
            logger.warning(f"Failed to delete {ckpt.name}: {exc}")

    # Remove step checkpoints
    for ckpt in checkpoint_dir.glob("step_*.pt"):
        try:
            ckpt.unlink()
            deleted.append(ckpt.name)
        except OSError as exc:
            logger.warning(f"Failed to delete {ckpt.name}: {exc}")

    if deleted:
        click.echo(f"Cleaned up {len(deleted)} intermediate checkpoints")


def _build_scheduler(
    training_config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: Optional[int],
) -> Optional[LearningRateScheduler]:
    name = getattr(training_config, "scheduler", None)
    if not name:
        return None

    total_steps = training_config.total_steps or training_config.max_steps
    if total_steps is None and steps_per_epoch is not None:
        total_steps = steps_per_epoch * training_config.num_epochs

    min_lr = training_config.learning_rate * getattr(training_config, "min_lr_ratio", 0.0)

    name = name.lower()
    if name == "constant":
        return ConstantLRScheduler(optimizer, lr=training_config.learning_rate)
    if name == "linear_warmup":
        return LinearWarmup(
            optimizer,
            warmup_steps=training_config.warmup_steps,
            target_lr=training_config.learning_rate,
        )
    if name == "cosine" and total_steps:
        return CosineAnnealingScheduler(optimizer, total_steps=total_steps, min_lr=min_lr)
    if name == "exponential":
        return ExponentialDecayScheduler(optimizer, decay_rate=0.99, min_lr=min_lr)
    return None


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """FundamentaLLM - Minimal educational language model framework."""


@cli.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML config file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default="checkpoints",
    help="Directory for checkpoints",
)
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--batch-size", type=int, help="Batch size override")
@click.option("--learning-rate", type=float, help="Learning rate override")
@click.option("--model-dim", type=int, help="Model hidden dimension (d_model)")
@click.option("--num-heads", type=int, help="Number of attention heads")
@click.option("--num-layers", type=int, help="Number of transformer layers")
@click.option("--dropout", type=float, help="Dropout rate")
@click.option("--max-seq-len", type=int, help="Maximum sequence length")
@click.option("--val-split", type=float, help="Validation split ratio")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    help="Device override (auto selects best available)",
)
@click.option("--mixed-precision/--no-mixed-precision", default=None, help="Use mixed precision")
@click.option("--seed", type=int, help="Random seed override")
@click.option("--gradient-clip", type=float, help="Gradient clipping norm")
@click.option(
    "--resume-from",
    type=click.Path(exists=True, path_type=Path),
    help="Resume training from a checkpoint",
)
@click.option("--profile/--no-profile", default=False, help="Enable lightweight profiler")
@click.option("--quiet", is_flag=True, help="Reduce logging verbosity")
@click.option(
    "--auto-fix-config/--no-auto-fix-config",
    default=True,
    help="Auto-adjust invalid config (e.g., num_heads vs d_model)",
)
def train(
    data_path: Path,
    config_path: Optional[Path],
    output_dir: Path,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    model_dim: Optional[int],
    num_heads: Optional[int],
    num_layers: Optional[int],
    dropout: Optional[float],
    max_seq_len: Optional[int],
    val_split: Optional[float],
    device: Optional[str],
    mixed_precision: Optional[bool],
    seed: Optional[int],
    gradient_clip: Optional[float],
    resume_from: Optional[Path],
    profile: bool,
    quiet: bool,
    auto_fix_config: bool,
) -> None:
    """Train a language model on text data."""

    setup_logging(level="WARNING" if quiet else "INFO")
    logger.info("Starting training run")

    # Load and validate data
    try:
        text = data_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        logger.error(f"Data file not found: {data_path}")
        raise click.ClickException(f"Data file not found: {data_path}") from exc
    except UnicodeDecodeError as exc:
        logger.error(f"Encoding error reading {data_path}: {exc}")
        raise click.ClickException(
            f"File encoding error at {data_path}. Data must be UTF-8 encoded."
        ) from exc
    except Exception as exc:
        logger.exception(f"Unexpected error reading data from {data_path}")
        raise click.ClickException(f"Failed to read data: {exc}") from exc

    if not text or len(text.strip()) == 0:
        raise click.ClickException("Data file is empty; provide non-empty text data.")

    tokenizer = CharacterTokenizer()
    tokenizer.train([text])
    click.echo(f"Vocab size: {tokenizer.vocab_size}")

    training_config, model_config = _load_configs(config_path, data_path, tokenizer.vocab_size)
    _apply_overrides(
        training_config,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        val_split=val_split,
        sequence_length=max_seq_len,
        dropout=dropout,
        mixed_precision=mixed_precision,
        gradient_clip=gradient_clip,
    )

    # Validate device and apply fallback if needed
    validated_device = validate_device(training_config.device)
    if validated_device != training_config.device:
        training_config.device = validated_device
        click.echo(f"Device updated to: {validated_device}")

    # Validate configurations
    training_issues = validate_training_config(training_config)
    warn_on_issues(training_issues, "TrainingConfig")

    model_config.vocab_size = tokenizer.vocab_size
    if model_config.sequence_length is None:
        model_config.sequence_length = training_config.sequence_length

    _apply_model_overrides(
        model_config,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=max_seq_len,
    )

    _enforce_model_config(model_config, auto_fix_config)

    if training_config.seed is not None:
        set_seed(training_config.seed)

    ensure_dir(training_config.checkpoint_dir)

    click.echo("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(text, tokenizer, training_config)

    # Persist validation split for manual re-use
    try:
        _save_validation_split(text, tokenizer, training_config)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to export validation split: %s", exc)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise click.ClickException(
            "Training dataset is empty. Adjust sequence_length or provide more data. "
            f"(got {len(text)} characters, sequence_length={training_config.sequence_length})"
        )

    from fundamentallm.models.transformer import Transformer

    try:
        model = Transformer(model_config)
        logger.info(f"Model created with {model.count_parameters():,} parameters")
    except Exception as exc:
        logger.error(f"Failed to create model with config: {exc}")
        raise click.ClickException(f"Model creation failed: {exc}") from exc

    loss_fn = LanguageModelingLoss()
    builder = OptimizerBuilder(
        weight_decay=training_config.optimizer_weight_decay,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        epsilon=training_config.optimizer_eps,
    )
    optimizer = builder.build(training_config.optimizer, model, lr=training_config.learning_rate)
    scheduler = _build_scheduler(training_config, optimizer, steps_per_epoch)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=training_config.device,
        config=training_config,
    )

    start_epoch = 0
    if resume_from is not None:
        try:
            _, _, _, _, restored_epoch, restored_step = trainer.checkpoint_manager.load(
                resume_from,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
            )
            trainer.global_step = restored_step
            start_epoch = restored_epoch + 1
            click.echo(
                f"Resumed from {resume_from} at epoch {restored_epoch + 1}, step {restored_step}"
            )
        except Exception as exc:
            raise click.ClickException(f"Failed to resume from {resume_from}: {exc}") from exc

    click.echo("Training...")
    target_epochs = training_config.num_epochs
    epochs_to_run = max(target_epochs - start_epoch, 0)
    if epochs_to_run == 0:
        click.echo("No epochs to run (already at or past target). Skipping training loop.")
        history = []
    else:
        if profile:
            profile_dir = Path(training_config.checkpoint_dir) / "profile"
            ensure_dir(profile_dir)
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False,
                ) as prof:
                    history = trainer.train(
                        num_epochs=epochs_to_run,
                        checkpoint_dir=training_config.checkpoint_dir,
                        start_epoch=start_epoch,
                        total_epochs=target_epochs,
                    )
                trace_path = profile_dir / "profile.json"
                prof.export_chrome_trace(str(trace_path))
                click.echo(f"Profiler trace saved to {trace_path}")
            except Exception as exc:  # pragma: no cover - defensive
                raise click.ClickException(f"Profiling failed: {exc}") from exc
        else:
            history = trainer.train(
                num_epochs=epochs_to_run,
                checkpoint_dir=training_config.checkpoint_dir,
                start_epoch=start_epoch,
                total_epochs=target_epochs,
            )

    eval_results = None
    if val_loader is not None:
        try:
            if len(val_loader) == 0:
                click.echo("Validation set empty; skipping evaluation.")
            else:
                click.echo("Evaluating model on validation split...")
                evaluator = ModelEvaluator(model, tokenizer, device=training_config.device)
                eval_results = evaluator.evaluate(val_loader)
                _print_eval_results(eval_results)
        except Exception as exc:
            logger.warning("Validation evaluation failed: %s", exc)

    final_metrics = history[-1] if history else {}
    if eval_results:
        for key, value in eval_results.items():
            if key not in final_metrics:
                final_metrics[key] = float(value) if isinstance(value, float) else value
    final_path = Path(training_config.checkpoint_dir) / "final_model.pt"
    manager = trainer.checkpoint_manager
    epoch_value = final_metrics.get("epoch", training_config.num_epochs - 1)
    state = manager._build_state(
        trainer.model,
        trainer.optimizer,
        trainer.scheduler,
        final_metrics,
        int(epoch_value),
        trainer.global_step,
    )
    state["model_config"] = model_config.model_dump()
    training_payload = training_config.model_dump()
    for key in ("data_path", "checkpoint_dir"):
        if key in training_payload and training_payload[key] is not None:
            training_payload[key] = str(training_payload[key])
    state["training_config"] = training_payload
    torch.save(state, final_path)

    model_config.save(Path(training_config.checkpoint_dir) / "model.yaml")
    training_config.save(Path(training_config.checkpoint_dir) / "training.yaml")
    tokenizer_path = Path(training_config.checkpoint_dir) / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    click.echo(f"Saved model to {final_path}")
    click.echo(f"Saved tokenizer to {tokenizer_path}")

    # Clean up intermediate checkpoints
    _cleanup_intermediate_checkpoints(training_config.checkpoint_dir)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", help="Initial prompt for generation")
@click.option(
    "--max-tokens", type=int, default=100, show_default=True, help="Maximum tokens to generate"
)
@click.option(
    "--temperature", type=float, default=1.0, show_default=True, help="Sampling temperature"
)
@click.option("--top-k", type=int, help="Top-k sampling")
@click.option(
    "--top-p", type=float, default=1.0, show_default=True, help="Top-p (nucleus) sampling"
)
@click.option(
    "--num-samples", type=int, default=1, show_default=True, help="Number of samples to generate"
)
@click.option("--output-file", type=click.Path(path_type=Path), help="Write generations to file")
@click.option("--interactive", is_flag=True, help="Launch interactive REPL")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
    show_default=True,
    help="Device for inference",
)
def generate(
    model_path: Path,
    prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    num_samples: int,
    output_file: Optional[Path],
    interactive: bool,
    device: str,
) -> None:
    """Generate text from a trained model."""

    setup_logging()

    validated_device = validate_device(device)
    if validated_device != device:
        device = validated_device
        click.echo(f"Device updated to: {validated_device}")

    generator = TextGenerator.from_checkpoint(model_path, device=device)

    if interactive:
        from fundamentallm.cli.interactive import InteractiveREPL

        repl = InteractiveREPL(
            generator,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        repl.run()
        return

    if not prompt:
        prompt = click.prompt("Enter prompt")

    click.echo(f"Generating with T={temperature}, max_tokens={max_tokens}...")
    click.echo("-" * 80)

    outputs = []
    for idx in range(num_samples):
        text = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        outputs.append(text)
        label = f"Sample {idx + 1}" if num_samples > 1 else "Output"
        click.echo(f"{label}:")
        click.echo(text)
        if num_samples > 1 and idx != num_samples - 1:
            click.echo("-" * 40)

    if output_file is not None:
        ensure_dir(output_file.parent)
        output_file.write_text("\n\n".join(outputs), encoding="utf-8")
        click.echo(f"Generations written to {output_file}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), help="Save results to JSON")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
    show_default=True,
    help="Device for evaluation",
)
def evaluate(
    model_path: Path,
    data_path: Path,
    output: Optional[Path],
    device: str,
) -> None:
    """Evaluate a trained model on test data."""

    setup_logging()

    validated_device = validate_device(device)
    if validated_device != device:
        device = validated_device
        click.echo(f"Device updated to: {validated_device}")

    evaluator = ModelEvaluator.from_checkpoint(model_path, device=device)
    text = data_path.read_text(encoding="utf-8")

    config = TrainingConfig(batch_size=4, sequence_length=32, data_path=data_path)
    _, val_loader = create_dataloaders(text, evaluator.tokenizer, config)

    if len(val_loader) == 0:
        raise click.ClickException(
            "Evaluation dataset is empty; provide more data or reduce sequence_length."
        )

    results = evaluator.evaluate(val_loader)

    click.echo("-" * 80)
    for key, value in results.items():
        if isinstance(value, float):
            click.echo(f"{key:15s}: {value:.4f}")
    click.echo("-" * 80)

    if output:
        output_dict = {k: float(v) if isinstance(v, float) else str(v) for k, v in results.items()}
        ensure_dir(output.parent)
        output.write_text(json.dumps(output_dict, indent=2), encoding="utf-8")
        click.echo(f"Results saved to {output}")
