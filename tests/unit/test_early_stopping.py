"""Tests for early stopping utility."""

from fundamentallm.training.early_stopping import EarlyStopping


def test_early_stopping_min_mode():
    stopper = EarlyStopping(patience=2, metric="val_loss", mode="min", min_delta=0.0)
    should_stop = []
    for value in [1.0, 1.0, 1.0, 1.1]:
        should_stop.append(stopper.step(value))
    # No stop until patience exceeded
    assert should_stop[-1] is True
    assert stopper.best_value == 1.0


def test_early_stopping_max_mode():
    stopper = EarlyStopping(patience=1, metric="acc", mode="max", min_delta=0.0)
    assert stopper.step(0.5) is False  # best
    assert stopper.is_best is True
    assert stopper.step(0.4) is True  # no improvement triggers stop


def test_early_stopping_reset():
    stopper = EarlyStopping(patience=1)
    stopper.step(1.0)
    stopper.step(1.1)
    stopper.reset()
    assert stopper.counter == 0
    assert stopper.is_best is False
