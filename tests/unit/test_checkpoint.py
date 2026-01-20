"""Tests for checkpoint manager."""

from pathlib import Path

import torch
import torch.nn as nn

from fundamentallm.training.checkpoint import CheckpointManager


def test_save_and_load_cycle(tmp_dir):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager()

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt_path = tmp_dir / "checkpoint.pt"
    manager.save(
        ckpt_path, model, optimizer, scheduler=None, metrics={"val_loss": 1.0}, epoch=1, step=10
    )

    # Modify weights then restore
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    model, optimizer, _, metrics, epoch, step = manager.load(ckpt_path, model, optimizer)
    assert epoch == 1 and step == 10
    assert metrics["val_loss"] == 1.0
    # After load, parameters should match saved state
    for name, param in model.state_dict().items():
        assert torch.allclose(param, initial_state[name])


def test_save_best(tmp_dir):
    model = nn.Linear(1, 1)
    manager = CheckpointManager()
    path = tmp_dir / "best.pt"

    first = manager.save_best(path, model, {"val_loss": 0.5})
    second = manager.save_best(path, model, {"val_loss": 0.6})
    third = manager.save_best(path, model, {"val_loss": 0.4})

    assert first is not None
    assert second is None  # worse
    assert third is not None  # improved
    assert path.exists()


def test_keep_last(tmp_dir):
    model = nn.Linear(1, 1)
    manager = CheckpointManager(keep_last=2)

    paths = [tmp_dir / f"ckpt_{i}.pt" for i in range(3)]
    for i, path in enumerate(paths):
        manager.save_last(path, model, metrics={"val_loss": float(i)}, epoch=i, step=i)

    existing = list(Path(tmp_dir).glob("ckpt_*.pt"))
    assert len(existing) <= 2
