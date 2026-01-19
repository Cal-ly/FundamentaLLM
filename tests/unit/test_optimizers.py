"""Tests for optimizer builder."""

import torch
import torch.nn as nn

from fundamentallm.training.optimizers import OptimizerBuilder


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not used
        return self.norm(self.linear(x))


def test_build_supported_optimizers():
    model = TinyModel()
    builder = OptimizerBuilder()

    for name, cls in [
        ("adamw", torch.optim.AdamW),
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ]:
        opt = builder.build(name, model, lr=1e-3)
        assert isinstance(opt, cls)
        assert opt.param_groups[0]["lr"] == 1e-3


def test_weight_decay_excludes_norm_and_bias():
    model = TinyModel()
    builder = OptimizerBuilder(weight_decay=0.1)
    opt = builder.build("adamw", model, lr=1e-3)

    # Bias and norm parameters should reside in a zero weight decay group
    bias_decay = None
    norm_decay = None
    for group in opt.param_groups:
        if any(p is model.linear.bias for p in group["params"]):
            bias_decay = group["weight_decay"]
        if any(p is model.norm.weight for p in group["params"]):
            norm_decay = group["weight_decay"]

    assert bias_decay == 0.0
    assert norm_decay == 0.0
