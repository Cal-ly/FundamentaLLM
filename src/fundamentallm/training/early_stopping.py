"""Early stopping utility."""

from __future__ import annotations

from typing import Callable


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(
        self,
        patience: int,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if patience < 0:
            raise ValueError("patience must be >= 0")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        if mode == "min":
            self.best_value = float("inf")
            self._is_better: Callable[[float, float], bool] = lambda current, best: current < best - self.min_delta
        else:
            self.best_value = float("-inf")
            self._is_better = lambda current, best: current > best + self.min_delta

        self.counter = 0
        self.is_best = False

    def step(self, current_value: float) -> bool:
        """Update state with new metric value.

        Returns True if training should stop.
        """
        if self._is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False

        return self.counter >= self.patience if self.patience > 0 else False

    def reset(self) -> None:
        """Reset tracked state."""
        self.counter = 0
        self.is_best = False
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
