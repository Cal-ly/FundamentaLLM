"""Integration test for CLI pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from fundamentallm.cli.commands import cli


def _write_config(path: Path, checkpoint_dir: Path) -> None:
    config = {
        "model": {
            "vocab_size": None,
            "d_model": 32,
            "num_heads": 2,
            "num_layers": 1,
            "sequence_length": 32,
            "dropout": 0.0,
            "ffn_expansion": 2,
            "pos_encoding": "learned",
        },
        "training": {
            "train_split": 0.8,
            "sequence_length": 32,
            "batch_size": 4,
            "num_workers": 0,
            "num_epochs": 1,
            "optimizer": "adamw",
            "optimizer_weight_decay": 0.0,
            "learning_rate": 1e-3,
            "scheduler": "constant",
            "warmup_steps": 0,
            "min_lr_ratio": 0.0,
            "checkpoint_dir": str(checkpoint_dir),
            "eval_steps": 0,
            "accumulation_steps": 1,
            "device": "cpu",
            "dropout": 0.0,
            "early_stopping_patience": 0,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_cli_train_generate_evaluate(tmp_path):
    runner = CliRunner()

    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world " * 100, encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, checkpoint_dir)

    train_result = runner.invoke(
        cli,
        [
            "train",
            str(data_file),
            "--config",
            str(config_file),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--output-dir",
            str(checkpoint_dir),
            "--quiet",
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert "Evaluating model on validation split" in train_result.output
    val_file = checkpoint_dir / "validation.txt"
    assert val_file.exists()
    assert val_file.read_text(encoding="utf-8").strip()

    model_file = checkpoint_dir / "final_model.pt"
    assert model_file.exists()

    generate_result = runner.invoke(
        cli,
        [
            "generate",
            str(model_file),
            "--prompt",
            "hello",
            "--max-tokens",
            "5",
            "--device",
            "cpu",
        ],
    )
    assert generate_result.exit_code == 0, generate_result.output
    assert "hello" in generate_result.output

    evaluate_result = runner.invoke(
        cli,
        [
            "evaluate",
            str(model_file),
            str(data_file),
            "--device",
            "cpu",
        ],
    )
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert "loss" in evaluate_result.output.lower()


def test_cli_resume_and_multi_generate(tmp_path):
    runner = CliRunner()

    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world " * 50, encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    config_file = tmp_path / "config.yaml"
    _write_config(config_file, checkpoint_dir)

    # Initial 1-epoch run to create a checkpoint
    train_result = runner.invoke(
        cli,
        [
            "train",
            str(data_file),
            "--config",
            str(config_file),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--output-dir",
            str(checkpoint_dir),
            "--quiet",
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    val_file = checkpoint_dir / "validation.txt"
    assert val_file.exists()
    assert val_file.read_text(encoding="utf-8").strip()

    # Get the model before cleanup removes epoch files
    model_file = checkpoint_dir / "final_model.pt"
    assert model_file.exists()

    # Now verify intermediate checkpoints were cleaned up
    epoch_checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    assert len(epoch_checkpoints) == 0, "Intermediate epoch checkpoints should be cleaned up"

    # Resume test: create a new run with more epochs to test resume
    checkpoint_dir_2 = tmp_path / "checkpoints_resume"
    config_file_2 = tmp_path / "config_resume.yaml"
    _write_config(config_file_2, checkpoint_dir_2)

    # First, train for 1 epoch to generate an intermediate checkpoint (but it will be cleaned up)
    train_result_2 = runner.invoke(
        cli,
        [
            "train",
            str(data_file),
            "--config",
            str(config_file_2),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--output-dir",
            str(checkpoint_dir_2),
            "--quiet",
        ],
    )
    assert train_result_2.exit_code == 0, train_result_2.output

    # Verify first training cleaned up epochs
    first_epochs = list(checkpoint_dir_2.glob("epoch_*.pt"))
    assert len(first_epochs) == 0, "First training should clean up epochs"

    out_file = checkpoint_dir / "generations.txt"
    generate_result = runner.invoke(
        cli,
        [
            "generate",
            str(model_file),
            "--prompt",
            "hello",
            "--max-tokens",
            "5",
            "--device",
            "cpu",
            "--num-samples",
            "2",
            "--output-file",
            str(out_file),
        ],
    )
