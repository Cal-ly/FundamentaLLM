"""Benchmark placeholders for future evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""

    name: str
    metrics: Dict[str, float]


def aggregate_results(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Compute average metrics across benchmarks."""
    if not results:
        return {}
    aggregate: Dict[str, float] = {}
    for result in results:
        for key, value in result.metrics.items():
            aggregate.setdefault(key, 0.0)
            aggregate[key] += value
    for key in aggregate:
        aggregate[key] /= len(results)
    return aggregate
