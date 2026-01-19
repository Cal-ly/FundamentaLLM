"""Learning rate schedulers for FundamentaLLM."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from torch.optim import Optimizer


class LearningRateScheduler:
    """Base scheduler that updates optimizer learning rates."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    @property
    def last_lr(self) -> float:
        return float(self._last_lr[0]) if self._last_lr else 0.0

    def _update_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - simple container
        return {"step_count": self.step_count, "last_lr": self._last_lr}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step_count = int(state_dict.get("step_count", 0))
        last_lr = state_dict.get("last_lr")
        if last_lr:
            self._last_lr = list(last_lr)
            self._update_lr(float(self._last_lr[0]))

    def step(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class ConstantLRScheduler(LearningRateScheduler):
    """Keep a fixed learning rate."""

    def __init__(self, optimizer: Optimizer, lr: Optional[float] = None) -> None:
        super().__init__(optimizer)
        self.initial_lr = float(lr if lr is not None else self.last_lr)
        self._update_lr(self.initial_lr)

    def step(self) -> float:
        self.step_count += 1
        self._update_lr(self.initial_lr)
        return self.initial_lr


class LinearWarmup(LearningRateScheduler):
    """Linear warmup from 0 to target LR over ``warmup_steps``."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, target_lr: Optional[float] = None) -> None:
        super().__init__(optimizer)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self.target_lr = float(target_lr if target_lr is not None else self.last_lr)
        self._update_lr(0.0 if self.warmup_steps > 0 else self.target_lr)

    def step(self) -> float:
        self.step_count += 1
        if self.warmup_steps == 0:
            lr = self.target_lr
        else:
            progress = min(self.step_count / self.warmup_steps, 1.0)
            lr = self.target_lr * progress
        self._update_lr(lr)
        return lr


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine decay from initial LR down to ``min_lr`` over ``total_steps``."""

    def __init__(self, optimizer: Optimizer, total_steps: int, min_lr: float = 0.0) -> None:
        super().__init__(optimizer)
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.initial_lr = self.last_lr

    def step(self) -> float:
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self._update_lr(lr)
        return lr


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay scheduler: lr = initial_lr * decay_rate^step."""

    def __init__(self, optimizer: Optimizer, decay_rate: float = 0.99, min_lr: float = 0.0) -> None:
        super().__init__(optimizer)
        if decay_rate <= 0:
            raise ValueError("decay_rate must be > 0")
        self.decay_rate = float(decay_rate)
        self.min_lr = float(min_lr)
        self.initial_lr = self.last_lr

    def step(self) -> float:
        self.step_count += 1
        lr = max(self.min_lr, self.initial_lr * (self.decay_rate ** self.step_count))
        self._update_lr(lr)
        return lr
