"""Tests for TextGenerator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from fundamentallm.data.tokenizers.base import BaseTokenizer
from fundamentallm.generation.generator import TextGenerator
from fundamentallm.generation.sampling import GreedySampler
from fundamentallm.models.base import BaseModel


class FakeTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "a", "b", "c", "d"]
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self._trained = True

    def train(self, texts):
        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = [self.token_to_id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        pieces = []
        for tok in tokens:
            if skip_special_tokens and tok < 4:
                continue
            pieces.append(self.id_to_token.get(int(tok), "<UNK>"))
        return "".join(pieces)

    def save(self, path: Path) -> None:
        Path(path).write_text("fake", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FakeTokenizer":
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
        return self.token_to_id["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<EOS>"]


class DummyModel(BaseModel):
    def __init__(self, vocab_size: int, sequence_length: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(sequence_length=sequence_length)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        next_tokens = (input_ids[:, -1] + 1) % self.vocab_size
        indices = next_tokens.unsqueeze(-1).repeat(1, seq_len, 1)
        logits.scatter_(2, indices, 5.0)
        return logits

    def save(self, path: Path) -> None:
        Path(path).write_text("dummy", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DummyModel":  # pragma: no cover - unused
        raise NotImplementedError


def test_generate_appends_tokens():
    tokenizer = FakeTokenizer()
    model = DummyModel(vocab_size=tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer, device="cpu", sampler=GreedySampler())

    output = generator.generate("a", max_tokens=3)
    assert output.startswith("a")
    assert len(output) == len("a") + 3


def test_generate_honors_stop_sequences():
    tokenizer = FakeTokenizer()
    model = DummyModel(vocab_size=tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer, device="cpu")

    output = generator.generate("a", max_tokens=5, stop_sequences=["c"])
    assert output.endswith("c")
    assert len(output) <= len("a") + 2  # stop at first occurrence of "c"


def test_batch_generate_multiple_prompts():
    tokenizer = FakeTokenizer()
    model = DummyModel(vocab_size=tokenizer.vocab_size)
    generator = TextGenerator(model, tokenizer, device="cpu")

    outputs = generator.batch_generate(["a", "b"], max_tokens=2)
    assert len(outputs) == 2
    assert all(isinstance(text, str) for text in outputs)


def test_generation_respects_max_sequence_length():
    tokenizer = FakeTokenizer()
    model = DummyModel(vocab_size=tokenizer.vocab_size, sequence_length=3)
    generator = TextGenerator(model, tokenizer, device="cpu")

    output = generator.generate("ab", max_tokens=5)
    assert len(output) == len("ab") + 1  # limited by model sequence length
