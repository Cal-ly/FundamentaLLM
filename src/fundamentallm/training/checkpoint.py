"""Checkpoint management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from fundamentallm.utils.paths import ensure_dir


class CheckpointManager:
    """Save and restore training state for resuming experiments."""

    def __init__(
        self,
        keep_last: int = 3,
        best_metric_key: str = "val_loss",
        minimize_metric: bool = True,
    ) -> None:
        self.keep_last = keep_last
        self.best_metric_key = best_metric_key
        self.minimize_metric = minimize_metric
        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None
        self.last_checkpoints: list[Path] = []

    def _build_state(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Optional[Dict[str, float]],
        epoch: int,
        step: int,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            state["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state"] = scheduler.state_dict()
        return state

    def save(
        self,
        path: Path | str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        epoch: int = 0,
        step: int = 0,
    ) -> Path:
        """Persist a full training checkpoint."""
        path = Path(path)
        ensure_dir(path.parent)
        state = self._build_state(model, optimizer, scheduler, metrics, epoch, step)
        try:
            torch.save(state, path)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to save checkpoint to {path}: {exc}") from exc
        return path

    def load(
        self,
        path: Path | str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[
        torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], Dict[str, float], int, int
    ]:
        """Restore model/optimizer/scheduler state from ``path``.

        Returns a tuple ``(model, optimizer, scheduler, metrics, epoch, step)``.
        """
        path = Path(path)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Checkpoint at {path} appears corrupted: {exc}") from exc

        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise RuntimeError(f"Invalid checkpoint format at {path}")

        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (
            scheduler is not None
            and "scheduler_state" in checkpoint
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        metrics = checkpoint.get("metrics", {})
        epoch = int(checkpoint.get("epoch", 0))
        step = int(checkpoint.get("step", 0))
        return model, optimizer, scheduler, metrics, epoch, step

    def save_best(
        self,
        path: Path | str,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
    ) -> Optional[Path]:
        """Save if ``metrics[best_metric_key]`` improves over previous best."""
        if self.best_metric_key not in metrics:
            return None

        metric_value = float(metrics[self.best_metric_key])
        is_better = (
            self.best_metric is None
            or (self.minimize_metric and metric_value < self.best_metric)
            or (not self.minimize_metric and metric_value > self.best_metric)
        )
        if not is_better:
            return None

        self.best_metric = metric_value
        self.best_path = Path(path)
        return self.save(path, model, optimizer, scheduler, metrics, epoch, step)

    def save_last(
        self,
        path: Path | str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        epoch: int = 0,
        step: int = 0,
    ) -> Path:
        """Keep a rolling window of the last ``keep_last`` checkpoints."""
        saved_path = self.save(path, model, optimizer, scheduler, metrics, epoch, step)
        self.last_checkpoints.append(Path(saved_path))
        if len(self.last_checkpoints) > self.keep_last:
            oldest = self.last_checkpoints.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except OSError:
                pass  # pragma: no cover - best effort cleanup
        return saved_path
