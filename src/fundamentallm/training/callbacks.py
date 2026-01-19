"""Training callback abstractions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any) -> None:
        pass

    def on_epoch_end(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, loss: float) -> None:
        pass

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        pass


class CallbackList:
    """Manage a list of callbacks."""

    def __init__(self, callbacks: Optional[Iterable[Callback]] = None):
        self.callbacks: List[Callback] = list(callbacks) if callbacks else []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)

    def on_epoch_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_step_end(self, trainer: Any, loss: float) -> None:
        for callback in self.callbacks:
            callback.on_step_end(trainer, loss)

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)
