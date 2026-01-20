"""Tests for learning rate schedulers."""

import pytest
import torch

from fundamentallm.training.schedulers import (
    ConstantLRScheduler,
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    LinearWarmup,
)


def _make_optimizer(lr: float = 0.1) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.ones(1))
    return torch.optim.SGD([param], lr=lr)


def test_constant_scheduler_keeps_lr():
    opt = _make_optimizer(lr=0.05)
    sched = ConstantLRScheduler(opt)
    for _ in range(3):
        lr = sched.step()
        assert lr == 0.05
        assert opt.param_groups[0]["lr"] == 0.05


def test_linear_warmup_progression():
    opt = _make_optimizer(lr=0.2)
    sched = LinearWarmup(opt, warmup_steps=4, target_lr=0.2)
    lrs = [sched.step() for _ in range(4)]
    assert lrs[0] < lrs[-1]
    assert pytest.approx(lrs[-1], rel=1e-4) == 0.2


def test_cosine_scheduler_decays():
    opt = _make_optimizer(lr=0.1)
    sched = CosineAnnealingScheduler(opt, total_steps=4, min_lr=0.01)
    lrs = [sched.step() for _ in range(4)]
    assert lrs[0] >= lrs[-1]
    assert lrs[-1] >= 0.01


def test_exponential_decay_scheduler():
    opt = _make_optimizer(lr=0.1)
    sched = ExponentialDecayScheduler(opt, decay_rate=0.5, min_lr=0.01)
    lrs = [sched.step() for _ in range(3)]
    assert lrs[0] > lrs[1] > lrs[2]
    assert lrs[-1] >= 0.01
