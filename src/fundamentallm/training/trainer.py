"""Trainer orchestration for FundamentaLLM."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import amp

from fundamentallm.training.callbacks import Callback, CallbackList
from fundamentallm.training.checkpoint import CheckpointManager
from fundamentallm.training.early_stopping import EarlyStopping
from fundamentallm.training.metrics import MetricTracker
from fundamentallm.config.training import TrainingConfig


class Trainer:
    """End-to-end training loop manager."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: Any,
        device: str | torch.device,
        config: TrainingConfig,
        callbacks: Optional[Iterable[Callback]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.callbacks = CallbackList(callbacks)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            keep_last=getattr(config, "checkpoint_keep_last", 3)
        )

        patience = getattr(config, "early_stopping_patience", 0)
        self.early_stopping = EarlyStopping(
            patience=patience,
            metric=getattr(config, "early_stopping_metric", "val_loss"),
            mode=getattr(config, "early_stopping_mode", "min"),
            min_delta=getattr(config, "early_stopping_min_delta", 0.0),
        ) if patience > 0 else None

        self.metric_tracker = MetricTracker()

        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)
        self.accumulation_steps = max(1, getattr(config, "accumulation_steps", 1))
        self.eval_steps = getattr(config, "eval_steps", 0)
        self.global_step = 0
        self.ema_loss: Optional[float] = None

        mixed_precision = bool(getattr(config, "mixed_precision", False)) and self.device.type == "cuda"
        self.scaler = amp.GradScaler(enabled=mixed_precision)

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch

    def _compute_tokens(self, targets: torch.Tensor) -> int:
        mask = (targets != -100) & (targets != -1)
        return int(mask.sum().item())

    def _train_step(self, batch: Any) -> tuple[float, int]:
        self.model.train()
        inputs, targets = batch
        inputs = self._to_device(inputs)
        targets = self._to_device(targets)

        with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets, reduction="mean")
        loss = loss / self.accumulation_steps

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        step_tokens = self._compute_tokens(targets)

        if (self.global_step + 1) % self.accumulation_steps == 0:
            self._apply_optimizer_step()

        total_loss = float(loss.detach().item() * self.accumulation_steps)
        self.global_step += 1

        if self.ema_loss is None:
            self.ema_loss = total_loss
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * total_loss

        return total_loss, step_tokens

    def _apply_optimizer_step(self) -> None:
        if self.max_grad_norm:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._to_device(batch[0]), self._to_device(batch[1])
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets, reduction="mean")
                tokens = self._compute_tokens(targets)
                total_loss += float(loss.item()) * max(tokens, 1)
                total_tokens += max(tokens, 1)

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
        return {"val_loss": avg_loss, "perplexity": perplexity}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.callbacks.on_epoch_begin(self)
        start_time = time.time()

        running_loss = 0.0
        running_tokens = 0
        num_batches = 0

        for batch in self.train_loader:
            loss_value, tokens = self._train_step(batch)
            if not math.isfinite(loss_value):
                raise RuntimeError("Detected non-finite loss during training")

            num_batches += 1
            running_loss += loss_value
            running_tokens += tokens

            self.callbacks.on_step_end(self, loss_value)

            if self.eval_steps and self.val_loader is not None and self.global_step % self.eval_steps == 0:
                val_metrics = self.validate()
                if self.checkpoint_manager and val_metrics:
                    ckpt_path = Path(self.config.checkpoint_dir) / f"step_{self.global_step}.pt"
                    self.checkpoint_manager.save_best(
                        ckpt_path,
                        self.model,
                        val_metrics,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.global_step,
                    )

        if num_batches and self.global_step % self.accumulation_steps != 0:
            # Flush remaining gradients when dataset size is not divisible by accumulation steps
            self._apply_optimizer_step()

        avg_loss = running_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        throughput = running_tokens / max(elapsed, 1e-8)

        metrics = {
            "loss": avg_loss,
            "ema_loss": self.ema_loss if self.ema_loss is not None else avg_loss,
            "throughput_tokens_per_sec": throughput,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        self.callbacks.on_epoch_end(self)
        return metrics

    def train(self, num_epochs: Optional[int] = None, checkpoint_dir: Optional[Path | str] = None) -> list[Dict[str, float]]:
        history: list[Dict[str, float]] = []
        checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        self.callbacks.on_train_begin(self)

        epochs = num_epochs if num_epochs is not None else getattr(self.config, "num_epochs", 1)
        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            combined = {"epoch": epoch, **epoch_metrics, **val_metrics}
            history.append(combined)

            self.metric_tracker.update({f"train_{k}": v for k, v in epoch_metrics.items()})
            if val_metrics:
                self.metric_tracker.update({f"val_{k}": v for k, v in val_metrics.items()})
                self.callbacks.on_validation_end(self, val_metrics)

            if self.checkpoint_manager is not None:
                # Save rolling checkpoints
                last_path = checkpoint_dir / f"epoch_{epoch}.pt"
                self.checkpoint_manager.save_last(
                    last_path,
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=combined,
                    epoch=epoch,
                    step=self.global_step,
                )
                if val_metrics:
                    best_path = checkpoint_dir / "best.pt"
                    self.checkpoint_manager.save_best(
                        best_path,
                        self.model,
                        val_metrics,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.global_step,
                    )

            if self.early_stopping is not None and val_metrics:
                monitored = val_metrics.get(self.early_stopping.metric)
                if monitored is not None:
                    should_stop = self.early_stopping.step(monitored)
                    if self.early_stopping.is_best and self.checkpoint_manager is not None:
                        best_path = checkpoint_dir / "best.pt"
                        self.checkpoint_manager.save_best(
                            best_path,
                            self.model,
                            val_metrics,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=self.global_step,
                        )
                    if should_stop:
                        break

        self.callbacks.on_train_end(self)
        return history
