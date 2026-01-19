"""Integration test for data pipeline."""

from __future__ import annotations

import torch

from fundamentallm.config.training import TrainingConfig
from fundamentallm.data.loaders import create_dataloaders
from fundamentallm.data.tokenizers.character import CharacterTokenizer


def test_end_to_end_data_pipeline(tmp_path):
    raw_text = "hello world\nthis is a test\nfundamentallm"
    tokenizer = CharacterTokenizer()
    tokenizer.train([raw_text])

    config = TrainingConfig(
        data_path=tmp_path / "data.txt",
        batch_size=2,
        max_epochs=1,
        sequence_length=6,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    train_loader, val_loader = create_dataloaders(raw_text, tokenizer, config)

    for inputs, targets in train_loader:
        assert inputs.shape[0] == config.batch_size
        assert inputs.shape[1] == config.sequence_length
        assert targets.shape == inputs.shape
        assert torch.all(inputs >= 0)
        assert torch.all(inputs < tokenizer.vocab_size)
        break

    sample_ids = tokenizer.encode("hi", add_special_tokens=True)
    decoded = tokenizer.decode(sample_ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
