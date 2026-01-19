"""Tests for ModelEvaluator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from fundamentallm.data.tokenizers.base import BaseTokenizer
from fundamentallm.evaluation.evaluator import ModelEvaluator
from fundamentallm.models.base import BaseModel


class SimpleTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.tokens = ["<PAD>", "<UNK>", "a", "b", "c"]
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self._trained = True

    def train(self, texts):
        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self.token_to_id.get(ch, self.unk_token_id) for ch in text]

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        pieces = []
        for tok in tokens:
            if skip_special_tokens and tok == self.pad_token_id:
                continue
            pieces.append(self.id_to_token.get(int(tok), "<UNK>"))
        return "".join(pieces)

    def save(self, path: Path) -> None:
        Path(path).write_text("tokenizer", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":  # pragma: no cover - unused
        return cls()

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def bos_token_id(self) -> int:
        return self.unk_token_id

    @property
    def eos_token_id(self) -> int:
        return self.unk_token_id


class CopyModel(BaseModel):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(sequence_length=8)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits.scatter_(2, input_ids.unsqueeze(-1), 6.0)
        return logits

    def save(self, path: Path) -> None:
        Path(path).write_text("copy", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CopyModel":  # pragma: no cover - unused
        raise NotImplementedError


def _build_dataloader(tokenizer: SimpleTokenizer) -> DataLoader:
    inputs = torch.tensor([[2, 3, 4], [3, 2, 4]], dtype=torch.long)  # tokens for a, b, c
    targets = inputs.clone()
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)


def test_evaluate_returns_metrics():
    tokenizer = SimpleTokenizer()
    model = CopyModel(vocab_size=tokenizer.vocab_size)
    evaluator = ModelEvaluator(model, tokenizer, device="cpu")

    data_loader = _build_dataloader(tokenizer)
    results = evaluator.evaluate(data_loader)

    assert 0 <= results["loss"] < 0.5
    assert 0 <= results["accuracy"] <= 1
    assert results["perplexity"] > 0


def test_evaluate_can_return_predictions():
    tokenizer = SimpleTokenizer()
    model = CopyModel(vocab_size=tokenizer.vocab_size)
    evaluator = ModelEvaluator(model, tokenizer, device="cpu")

    data_loader = _build_dataloader(tokenizer)
    results = evaluator.evaluate(data_loader, return_predictions=True)

    assert "predictions" in results and "targets" in results
    assert results["predictions"].shape == results["targets"].shape
    assert torch.equal(results["predictions"], results["targets"])
