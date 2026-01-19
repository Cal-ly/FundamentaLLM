"""CLI help surface tests."""

from __future__ import annotations

from click.testing import CliRunner

from fundamentallm.cli.commands import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "FundamentaLLM" in result.output


def test_train_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "DATA_PATH" in result.output


def test_generate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "MODEL_PATH" in result.output


def test_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "MODEL_PATH" in result.output
