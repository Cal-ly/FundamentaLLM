"""Pytest configuration and fixtures."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def tmp_dir() -> Path:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for tokenization tests."""
    return "hello world\nthis is a test\nfundamentallm framework"


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide a list of short sample texts."""
    return [
        "hello world",
        "this is a test",
        "fundamentallm",
        "transformer model",
    ]


@pytest.fixture
def sample_tokenizer(sample_text):
    """Create and train a character tokenizer for tests."""
    from fundamentallm.data.tokenizers.character import CharacterTokenizer

    tokenizer = CharacterTokenizer()
    tokenizer.train([sample_text, "hello world", "test data", "fundamentallm"])
    return tokenizer


@pytest.fixture
def sample_tokens(sample_tokenizer, sample_text):
    """Tokenize sample text into a torch tensor."""
    import torch

    tokens = sample_tokenizer.encode(sample_text)
    return torch.tensor(tokens, dtype=torch.long)


@pytest.fixture
def train_config(tmp_path):
    """Minimal TrainingConfig for data loader tests."""
    from fundamentallm.config.training import TrainingConfig

    return TrainingConfig(
        data_path=tmp_path / "data.txt",
        batch_size=4,
        max_epochs=1,
        sequence_length=8,
        checkpoint_dir=tmp_path / "checkpoints",
    )
