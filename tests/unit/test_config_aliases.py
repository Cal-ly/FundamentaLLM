"""Tests for config alias handling."""

from fundamentallm.config.model import TransformerConfig
from fundamentallm.config.training import TrainingConfig


def test_training_config_accepts_doc_aliases():
    cfg = TrainingConfig.model_validate(
        {
            "epochs": 2,
            "val_split": 0.2,
            "max_seq_len": 128,
            "batch_size": 8,
        }
    )

    assert cfg.num_epochs == 2
    assert cfg.train_split == 0.8
    assert cfg.sequence_length == 128
    assert cfg.batch_size == 8


def test_transformer_config_accepts_hidden_dim_alias():
    cfg = TransformerConfig.model_validate(
        {
            "vocab_size": 10,
            "hidden_dim": 64,
            "num_heads": 4,
            "max_seq_len": 50,
        }
    )

    assert cfg.d_model == 64
    assert cfg.num_heads == 4
    assert cfg.sequence_length == 50
