"""Metric tracking utilities."""

from __future__ import annotations

from typing import Dict, List, Optional


class MetricTracker:
    """Track scalar metrics over time."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}

    def update(self, metrics_dict: Dict[str, float]) -> None:
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))

    def get_history(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self.metrics.items()}

    def get_latest(self, metric_name: str) -> Optional[float]:
        series = self.metrics.get(metric_name)
        if not series:
            return None
        return series[-1]

    def get_best(self, metric_name: str, mode: str = "min") -> Optional[float]:
        series = self.metrics.get(metric_name)
        if not series:
            return None
        if mode == "min":
            return min(series)
        if mode == "max":
            return max(series)
        raise ValueError("mode must be 'min' or 'max'")

    def reset(self) -> None:
        self.metrics.clear()
