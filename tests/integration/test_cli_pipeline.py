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

    epoch_ckpt = checkpoint_dir / "epoch_0.pt"
    assert epoch_ckpt.exists()

    # Resume to a 2-epoch target, starting from epoch_0
    resume_result = runner.invoke(
        cli,
        [
            "train",
            str(data_file),
            "--config",
            str(config_file),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--output-dir",
            str(checkpoint_dir),
            "--resume-from",
            str(epoch_ckpt),
            "--quiet",
        ],
    )
    assert resume_result.exit_code == 0, resume_result.output
    assert "Resumed from" in resume_result.output

    model_file = checkpoint_dir / "final_model.pt"
    assert model_file.exists()

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
    assert generate_result.exit_code == 0, generate_result.output
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "hello" in content.lower()
    # We expect two samples separated by blank lines
    assert content.count("\n\n") >= 1
