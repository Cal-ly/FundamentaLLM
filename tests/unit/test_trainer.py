"""Unit tests for Trainer."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from fundamentallm.config import TransformerConfig
from fundamentallm.config.training import TrainingConfig
from fundamentallm.models.transformer import Transformer
from fundamentallm.training.losses import LanguageModelingLoss
from fundamentallm.training.optimizers import OptimizerBuilder
from fundamentallm.training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        self.inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def _build_trainer(tmp_dir: str, accumulation_steps: int = 1):
    seq_len = 8
    vocab_size = 32
    dataset = TinyDataset(num_samples=16, seq_len=seq_len, vocab_size=vocab_size)
    train_loader = DataLoader(dataset, batch_size=4, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=4, drop_last=True)

    model = Transformer(
        TransformerConfig(
            vocab_size=vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            sequence_length=seq_len,
            dropout=0.0,
        )
    )

    loss_fn = LanguageModelingLoss()
    optimizer = OptimizerBuilder().build("adamw", model, lr=1e-3)
    config = TrainingConfig(
        sequence_length=seq_len,
        batch_size=4,
        num_epochs=1,
        accumulation_steps=accumulation_steps,
        checkpoint_dir=tmp_dir,
        eval_steps=0,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device="cpu",
        config=config,
    )
    return trainer


def test_train_epoch_updates_global_step(tmp_dir):
    trainer = _build_trainer(tmp_dir, accumulation_steps=1)
    start_step = trainer.global_step
    metrics = trainer.train_epoch(epoch=0)

    assert trainer.global_step > start_step
    assert "loss" in metrics
    assert metrics["loss"] > 0


def test_gradient_accumulation_reduces_optimizer_steps(tmp_dir):
    trainer = _build_trainer(tmp_dir, accumulation_steps=2)

    step_calls = 0
    orig_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return orig_step(*args, **kwargs)

    trainer.optimizer.step = counting_step  # type: ignore[assignment]

    trainer.train_epoch(epoch=0)
    # DataLoader has 4 batches, accumulation_steps=2 => 2 optimizer steps
    assert step_calls == 2


def test_train_saves_checkpoint(tmp_dir):
    trainer = _build_trainer(tmp_dir, accumulation_steps=1)
    history = trainer.train(num_epochs=1, checkpoint_dir=tmp_dir)
    assert len(history) == 1
    saved = list(Path(tmp_dir).glob("epoch_*.pt"))
    assert saved
