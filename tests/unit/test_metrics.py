"""Tests for metric tracker."""

from fundamentallm.training.metrics import MetricTracker


def test_metric_tracker_updates_and_queries():
    tracker = MetricTracker()
    tracker.update({"loss": 1.0, "acc": 0.5})
    tracker.update({"loss": 0.8, "acc": 0.6})

    history = tracker.get_history()
    assert history["loss"] == [1.0, 0.8]
    assert tracker.get_latest("acc") == 0.6
    assert tracker.get_best("loss", mode="min") == 0.8
    assert tracker.get_best("acc", mode="max") == 0.6


def test_metric_tracker_reset():
    tracker = MetricTracker()
    tracker.update({"x": 1.0})
    tracker.reset()
    assert tracker.get_history() == {}
