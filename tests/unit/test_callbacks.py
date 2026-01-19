"""Tests for callback hooks."""

from fundamentallm.training.callbacks import Callback, CallbackList


class Recorder(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_begin(self, trainer):
        self.events.append("train_begin")

    def on_train_end(self, trainer):
        self.events.append("train_end")

    def on_epoch_begin(self, trainer):
        self.events.append("epoch_begin")

    def on_epoch_end(self, trainer):
        self.events.append("epoch_end")

    def on_step_end(self, trainer, loss: float):
        self.events.append(f"step:{loss}")

    def on_validation_end(self, trainer, metrics):
        self.events.append(f"val:{metrics}")


def test_callbacks_invoke_hooks():
    rec = Recorder()
    clist = CallbackList([rec])
    dummy = object()

    clist.on_train_begin(dummy)
    clist.on_epoch_begin(dummy)
    clist.on_step_end(dummy, 1.23)
    clist.on_validation_end(dummy, {"val_loss": 0.5})
    clist.on_epoch_end(dummy)
    clist.on_train_end(dummy)

    assert rec.events[0] == "train_begin"
    assert "step:1.23" in rec.events
    assert "val:{'val_loss': 0.5}" in rec.events
    assert rec.events[-1] == "train_end"
