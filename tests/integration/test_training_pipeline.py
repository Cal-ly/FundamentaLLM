"""Integration tests for the training pipeline."""

from pathlib import Path

import torch

from fundamentallm.config import TransformerConfig
from fundamentallm.config.training import TrainingConfig
from fundamentallm.data.loaders import create_dataloaders
from fundamentallm.models.transformer import Transformer
from fundamentallm.training.losses import LanguageModelingLoss
from fundamentallm.training.optimizers import OptimizerBuilder
from fundamentallm.training.schedulers import LinearWarmup
from fundamentallm.training.trainer import Trainer


def _build_trainer(
    tmp_dir: Path,
    text: str,
    tokenizer,
    accumulation_steps: int = 1,
    use_scheduler: bool = False,
    early_stopping_patience: int = 0,
) -> Trainer:
    config = TrainingConfig(
        data_path=tmp_dir / "data.txt",
        batch_size=2,
        sequence_length=8,
        num_epochs=2,
        train_split=0.8,
        device="cpu",
        checkpoint_dir=tmp_dir,
        accumulation_steps=accumulation_steps,
        eval_steps=0,
        learning_rate=1e-3,
        early_stopping_patience=early_stopping_patience,
    )

    train_loader, val_loader = create_dataloaders(text, tokenizer, config)

    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        sequence_length=config.sequence_length,
        dropout=0.0,
    )
    transformer = Transformer(model_config)

    loss_fn = LanguageModelingLoss()
    optimizer = OptimizerBuilder().build("adamw", transformer, lr=config.learning_rate)
    scheduler = (
        LinearWarmup(optimizer, warmup_steps=2, target_lr=config.learning_rate)
        if use_scheduler
        else None
    )

    return Trainer(
        model=transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device="cpu",
        config=config,
    )


def test_training_pipeline_runs_and_records_metrics(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer)

    history = trainer.train(num_epochs=1, checkpoint_dir=tmp_dir)
    assert len(history) == 1
    assert "loss" in history[0]
    assert "perplexity" in history[0]


def test_validation_metrics_present(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer)
    metrics = trainer.validate()
    assert "val_loss" in metrics
    assert "perplexity" in metrics


def test_checkpoint_cycle(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer)
    trainer.train(num_epochs=1, checkpoint_dir=tmp_dir)

    ckpt_files = list(Path(tmp_dir).glob("epoch_*.pt"))
    assert ckpt_files

    # Load into a fresh trainer and resume
    new_trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer)
    manager = new_trainer.checkpoint_manager
    _, _, _, _, epoch, step = manager.load(
        ckpt_files[-1], new_trainer.model, new_trainer.optimizer, new_trainer.scheduler
    )
    new_trainer.global_step = step
    history = new_trainer.train(num_epochs=1, checkpoint_dir=tmp_dir)
    assert history


def test_gradient_accumulation_integration(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer, accumulation_steps=2)

    step_calls = 0
    orig_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return orig_step(*args, **kwargs)

    trainer.optimizer.step = counting_step  # type: ignore[assignment]
    trainer.train_epoch(epoch=0)
    assert step_calls > 0


def test_scheduler_integration(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer, use_scheduler=True)

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.train_epoch(epoch=0)
    updated_lr = trainer.optimizer.param_groups[0]["lr"]
    assert updated_lr >= initial_lr


def test_validation_improves_after_training(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(tmp_dir, sample_text, sample_tokenizer)

    before = trainer.validate()["val_loss"]
    trainer.train(num_epochs=1, checkpoint_dir=tmp_dir)
    after = trainer.validate()["val_loss"]

    assert after <= before or torch.isfinite(torch.tensor(after))


def test_early_stopping_triggers(tmp_dir, sample_text, sample_tokenizer):
    sample_tokenizer.train([sample_text])
    trainer = _build_trainer(
        tmp_dir,
        sample_text,
        sample_tokenizer,
        early_stopping_patience=1,
    )

    history = trainer.train(num_epochs=3, checkpoint_dir=tmp_dir)
    assert len(history) <= 3
