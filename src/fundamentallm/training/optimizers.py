"""Optimizer factory for FundamentaLLM training."""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from fundamentallm.models.components.normalization import LayerNorm, RMSNorm


class OptimizerBuilder:
    """Factory for constructing optimizers with sensible defaults.

    Weight decay is applied only to Linear layer weights. Biases and
    normalization layers are excluded to avoid over-regularization.
    """

    def __init__(
        self,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        momentum: float = 0.9,
    ) -> None:
        self.weight_decay = weight_decay
        self.betas = betas
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, optimizer_name: str, model: nn.Module, lr: float, **kwargs: Any) -> Optimizer:
        """Create an optimizer for ``model``.

        Args:
            optimizer_name: One of ``adamw``, ``adam``, ``sgd``, ``rmsprop``.
            model: Model with parameters to optimize.
            lr: Learning rate.
            **kwargs: Optional overrides (``weight_decay``, ``betas``, ``eps``,
                ``momentum``).
        """
        name = optimizer_name.lower()
        weight_decay = float(kwargs.pop("weight_decay", self.weight_decay))
        betas = kwargs.pop("betas", self.betas)
        eps = float(kwargs.pop("eps", self.epsilon))
        momentum = float(kwargs.pop("momentum", self.momentum))

        decay_params, no_decay_params = self._separate_parameters(model)

        param_groups: List[dict] = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        if name == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
        if name == "adam":
            return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
        if name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        if name == "rmsprop":
            return torch.optim.RMSprop(
                param_groups, lr=lr, alpha=kwargs.pop("alpha", 0.99), eps=eps, momentum=momentum
            )

        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @staticmethod
    def _separate_parameters(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Split parameters into decay and no-decay groups."""
        decay: List[nn.Parameter] = []
        no_decay: List[nn.Parameter] = []
        seen: set[int] = set()

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))

                if param_name.endswith("bias"):
                    no_decay.append(param)
                elif isinstance(module, (nn.LayerNorm, LayerNorm, RMSNorm)):
                    no_decay.append(param)
                elif isinstance(module, nn.Linear):
                    decay.append(param)
                else:
                    # Default to no weight decay for non-linear layers (e.g., embeddings)
                    no_decay.append(param)

        # Catch any parameters not captured above
        for param in model.parameters():
            if param.requires_grad and id(param) not in seen:
                no_decay.append(param)

        return decay, no_decay
