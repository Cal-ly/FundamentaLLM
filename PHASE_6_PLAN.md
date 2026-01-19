# Phase 6: CLI & Interactive Interface

**Objective:** Create command-line interface with Click, implement interactive REPL, and add comprehensive test suite.

**Status:** Planning

**Dependencies:** Phases 1-5 ✅

**Estimated Timeline:** 2-3 days

---

## Overview

Phase 6 brings FundamentaLLM to users:
- Click-based CLI with train/generate/evaluate commands
- Interactive REPL for conversation-like interaction
- Argument parsing and validation
- Help messages and documentation
- Comprehensive unit and integration tests
- pytest fixtures and test utilities

This phase makes the framework accessible and usable.

---

## Files to Create

### CLI

```
src/fundamentallm/cli/
├── __init__.py                     # CLI module exports
├── commands.py                     # Click command groups
└── interactive.py                  # Interactive REPL
```

### Testing

```
tests/
├── unit/
│   ├── test_cli.py                 # CLI command tests
│   └── test_interactive.py         # REPL tests
├── integration/
│   ├── test_end_to_end.py          # Full pipeline tests
│   └── test_cli_pipeline.py        # CLI end-to-end tests
└── conftest.py                     # (update with CLI fixtures)
```

---

## Detailed Tasks

### Task 6.1: Click-Based CLI - Train Command

**Objective:** Implement `fundamentallm train` command

**File:** `src/fundamentallm/cli/commands.py`

```python
import click
from pathlib import Path
from ..config.training import TrainingConfig
from ..config.model import TransformerConfig
from ..utils.logging import setup_logging, get_logger
from ..utils.random import set_seed

logger = get_logger(__name__)

@click.group()
@click.version_option()
def cli():
    """FundamentaLLM - Minimal educational language model framework.
    
    Train character-level transformer models on any text corpus.
    """
    pass

@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to YAML config file (overrides defaults)"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="checkpoints",
    help="Directory for checkpoints [default: checkpoints]"
)
@click.option(
    "--epochs",
    type=int,
    help="Number of training epochs (overrides config)"
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size (overrides config)"
)
@click.option(
    "--learning-rate",
    type=float,
    help="Learning rate (overrides config)"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Device (overrides config)"
)
@click.option(
    "--seed",
    type=int,
    help="Random seed (overrides config)"
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress detailed logging"
)
def train(
    data_path,
    config,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    quiet
):
    """Train a language model on text data.
    
    Example:
        fundamentallm train data/text.txt --config configs/small.yaml --epochs 10
    """
    setup_logging(level="WARNING" if quiet else "INFO")
    
    try:
        # Load config
        if config:
            click.echo(f"Loading config from {config}")
            train_config = TrainingConfig.from_yaml(Path(config))
        else:
            click.echo("Using default config")
            train_config = TrainingConfig(data_path=Path(data_path))
        
        # Apply CLI overrides
        overrides = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "seed": seed,
            "output_dir": output_dir
        }
        
        for key, value in overrides.items():
            if value is not None:
                # Handle nested config attributes
                if key == "output_dir":
                    train_config.checkpoint_dir = Path(value)
                elif key == "epochs":
                    train_config.max_epochs = value
                else:
                    setattr(train_config, key, value)
        
        # Set seed for reproducibility
        if train_config.seed is not None:
            set_seed(train_config.seed)
        
        # Load data
        click.echo(f"\nLoading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        click.echo(f"Data size: {len(text):,} characters")
        
        # Create tokenizer
        from ..data.tokenizers.character import CharacterTokenizer
        click.echo("\nTraining tokenizer...")
        tokenizer = CharacterTokenizer()
        tokenizer.train([text])
        click.echo(f"Vocab size: {tokenizer.vocab_size}")
        
        # Create dataloaders
        from ..data.loaders import create_dataloaders
        click.echo("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(text, tokenizer, train_config)
        
        # Create model
        from ..models.transformer import Transformer
        model_config = TransformerConfig(vocab_size=tokenizer.vocab_size)
        model = Transformer(model_config)
        click.echo(f"Model parameters: {model.count_parameters():,}")
        
        # Create trainer
        from ..training.trainer import Trainer
        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Train
        click.echo("\nStarting training...")
        click.echo("=" * 80)
        
        metrics = trainer.train()
        
        click.echo("=" * 80)
        click.echo("\nTraining completed!")
        
        # Save final model
        final_path = Path(train_config.checkpoint_dir) / "final_model.pt"
        trainer.save_checkpoint("final_model")
        click.echo(f"Model saved to {final_path}")
        
        # Save tokenizer
        tokenizer_path = Path(train_config.checkpoint_dir) / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        click.echo(f"Tokenizer saved to {tokenizer_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if not quiet:
            logger.exception("Training failed")
        raise click.Abort()
```

**Success Criteria:**
- ✅ Command accepts data path
- ✅ Can override config with CLI args
- ✅ Displays progress
- ✅ Saves model and tokenizer
- ✅ Handles errors gracefully

---

### Task 6.2: Click-Based CLI - Generate Command

**Objective:** Implement `fundamentallm generate` command

```python
@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--prompt",
    help="Initial prompt for generation"
)
@click.option(
    "--max-tokens",
    type=int,
    default=200,
    help="Maximum tokens to generate [default: 200]"
)
@click.option(
    "--temperature",
    type=float,
    default=0.8,
    help="Sampling temperature [default: 0.8]"
)
@click.option(
    "--top-k",
    type=int,
    help="Top-k sampling (if set)"
)
@click.option(
    "--top-p",
    type=float,
    help="Top-p (nucleus) sampling (if set)"
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Launch interactive mode for multi-turn generation"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    help="Device for inference [default: cuda]"
)
def generate(
    model_path,
    prompt,
    max_tokens,
    temperature,
    top_k,
    top_p,
    interactive,
    device
):
    """Generate text from a trained model.
    
    Example:
        fundamentallm generate checkpoints/best.pt --prompt "Once upon a time"
        
        fundamentallm generate checkpoints/best.pt --interactive
    """
    setup_logging()
    
    try:
        from ..generation.generator import TextGenerator
        
        # Load model
        click.echo(f"Loading model from {model_path}")
        generator = TextGenerator.from_checkpoint(model_path, device=device)
        
        if interactive:
            # Launch REPL
            from .interactive import InteractiveREPL
            repl = InteractiveREPL(generator, max_tokens, temperature, top_k, top_p)
            repl.run()
        else:
            # Single generation
            if not prompt:
                prompt = click.prompt("Enter prompt")
            
            click.echo(f"\nGenerating (T={temperature}, max_tokens={max_tokens})...")
            click.echo("-" * 80)
            
            text = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                verbose=True
            )
            
            click.echo("-" * 80)
            click.echo(text)
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
```

**Success Criteria:**
- ✅ Can load model from checkpoint
- ✅ Generates text with prompt
- ✅ Supports all sampling modes
- ✅ Interactive mode available
- ✅ Displays output clearly

---

### Task 6.3: Click-Based CLI - Evaluate Command

**Objective:** Implement `fundamentallm evaluate` command

```python
@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--output",
    type=click.Path(),
    help="Save results to JSON file"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cuda",
    help="Device for evaluation [default: cuda]"
)
def evaluate(model_path, data_path, output, device):
    """Evaluate a trained model on test data.
    
    Example:
        fundamentallm evaluate checkpoints/best.pt data/test.txt
    """
    setup_logging()
    
    try:
        from ..evaluation.evaluator import ModelEvaluator
        from ..data.loaders import create_dataloaders
        from ..config.training import TrainingConfig
        import json
        
        # Load model
        click.echo(f"Loading model from {model_path}")
        evaluator = ModelEvaluator.from_checkpoint(model_path, device=device)
        
        # Load data
        click.echo(f"Loading test data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create dataloader
        config = TrainingConfig()
        train_loader, val_loader = create_dataloaders(text, evaluator.tokenizer, config)
        
        # Evaluate
        click.echo("\nEvaluating...")
        results = evaluator.evaluate(val_loader)
        
        # Display results
        click.echo("-" * 80)
        for key, value in results.items():
            if isinstance(value, float):
                click.echo(f"{key:20s}: {value:.4f}")
        click.echo("-" * 80)
        
        # Save if requested
        if output:
            output_dict = {k: float(v) if isinstance(v, float) else str(v) 
                          for k, v in results.items()}
            with open(output, 'w') as f:
                json.dump(output_dict, f, indent=2)
            click.echo(f"\nResults saved to {output}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
```

**Success Criteria:**
- ✅ Can evaluate on test data
- ✅ Displays metrics clearly
- ✅ Can save results to JSON
- ✅ Handles various data formats

---

### Task 6.4: Interactive REPL

**Objective:** Implement interactive conversation interface

**File:** `src/fundamentallm/cli/interactive.py`

```python
import cmd
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

class InteractiveREPL:
    """Interactive REPL for text generation."""
    
    def __init__(
        self,
        generator,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        self.generator = generator
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.console = Console()
        self.history = []
    
    def run(self) -> None:
        """Run interactive loop."""
        self.console.print(Panel.fit(
            "[bold cyan]FundamentaLLM Interactive Mode[/bold cyan]\n"
            "Type 'help' for commands, 'quit' to exit",
            border_style="cyan"
        ))
        
        while True:
            try:
                prompt = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                if prompt.lower() in ["quit", "exit", "q"]:
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if prompt.lower() == "help":
                    self._show_help()
                    continue
                
                if prompt.lower() == "history":
                    self._show_history()
                    continue
                
                if prompt.startswith("/set "):
                    self._handle_settings(prompt)
                    continue
                
                # Generate response
                self._generate_response(prompt)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def _generate_response(self, prompt: str) -> None:
        """Generate and display response."""
        self.console.print("[yellow]Generating...[/yellow]", end="")
        
        response = self.generator.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        
        # Extract just the generated part (after prompt)
        generated = response[len(prompt):]
        
        self.console.print(f"\r[bold green]Model[/bold green]: {generated}")
        
        self.history.append((prompt, generated))
    
    def _show_help(self) -> None:
        """Display help message."""
        help_text = """
[bold]Commands:[/bold]
  quit, exit, q    - Exit interactive mode
  help             - Show this help
  history          - Show conversation history
  /set param=val   - Set parameters (temperature, max_tokens, top_k, top_p)

[bold]Examples:[/bold]
  /set temperature=0.5  - Lower temperature (more deterministic)
  /set max_tokens=500   - Generate longer responses
        """
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def _show_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            self.console.print("[yellow]No history yet[/yellow]")
            return
        
        history_text = ""
        for i, (prompt, response) in enumerate(self.history, 1):
            history_text += f"\n[bold cyan]Turn {i}[/bold cyan]\n"
            history_text += f"[cyan]You:[/cyan] {prompt}\n"
            history_text += f"[green]Model:[/green] {response}\n"
        
        self.console.print(Panel(history_text, title="History", border_style="cyan"))
    
    def _handle_settings(self, command: str) -> None:
        """Handle /set command."""
        parts = command[5:].split("=")
        if len(parts) != 2:
            self.console.print("[red]Invalid format. Use: /set param=value[/red]")
            return
        
        param, value = parts[0].strip(), parts[1].strip()
        
        try:
            if param == "temperature":
                self.temperature = float(value)
                self.console.print(f"[green]temperature set to {self.temperature}[/green]")
            elif param == "max_tokens":
                self.max_tokens = int(value)
                self.console.print(f"[green]max_tokens set to {self.max_tokens}[/green]")
            elif param == "top_k":
                self.top_k = int(value) if value.lower() != "none" else None
                self.console.print(f"[green]top_k set to {self.top_k}[/green]")
            elif param == "top_p":
                self.top_p = float(value) if value.lower() != "none" else None
                self.console.print(f"[green]top_p set to {self.top_p}[/green]")
            else:
                self.console.print(f"[red]Unknown parameter: {param}[/red]")
        except ValueError as e:
            self.console.print(f"[red]Invalid value: {e}[/red]")
```

**Success Criteria:**
- ✅ Can accept user input
- ✅ Generates responses
- ✅ Shows conversation history
- ✅ Allow parameter adjustment
- ✅ Handle errors gracefully

---

### Task 6.5: CLI Entry Point and Configuration

**Objective:** Complete CLI setup

**Update:** `src/fundamentallm/__main__.py`

```python
"""Allow running as: python -m fundamentallm"""

if __name__ == "__main__":
    from .cli.commands import cli
    cli()
```

**Verify:** `pyproject.toml` has entry point:
```toml
[project.scripts]
fundamentallm = "fundamentallm.cli.commands:cli"
```

**Success Criteria:**
- ✅ Can run `python -m fundamentallm`
- ✅ Can run `fundamentallm` after install
- ✅ Help works for all commands
- ✅ Tab completion works (if using Click-extras)

---

### Task 6.6: CLI Tests

**Objective:** Test CLI commands

**File:** `tests/unit/test_cli.py`

```python
import pytest
from click.testing import CliRunner
from fundamentallm.cli.commands import cli

@pytest.fixture
def runner():
    return CliRunner()

class TestCLI:
    def test_cli_help(self, runner):
        """Test main help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FundamentaLLM" in result.output
    
    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "DATA_PATH" in result.output
    
    def test_generate_help(self, runner):
        """Test generate command help."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
    
    def test_evaluate_help(self, runner):
        """Test evaluate command help."""
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
```

**Integration Tests:**

```python
# tests/integration/test_cli_pipeline.py

def test_full_training_pipeline(tmp_path, runner):
    """Test complete train → generate → evaluate pipeline."""
    # Create sample data
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world " * 100)
    
    # Train
    result = runner.invoke(cli, [
        "train",
        str(data_file),
        "--output-dir", str(tmp_path),
        "--epochs", "1",
        "--quiet"
    ])
    
    assert result.exit_code == 0
    model_file = tmp_path / "final_model.pt"
    assert model_file.exists()
    
    # Generate
    result = runner.invoke(cli, [
        "generate",
        str(model_file),
        "--prompt", "hello",
        "--max-tokens", "20"
    ])
    
    assert result.exit_code == 0
    assert "hello" in result.output
```

**Success Criteria:**
- ✅ All commands have help
- ✅ Train command works end-to-end
- ✅ Generate command works
- ✅ Evaluate command works
- ✅ Errors handled gracefully

---

### Task 6.7: Integration Tests

**Objective:** End-to-end testing

**File:** `tests/integration/test_end_to_end.py`

Tests for complete pipeline:
- ✅ Load data → Train tokenizer → Train model → Generate text
- ✅ Different configurations work
- ✅ Checkpoints save/load correctly
- ✅ Multiple training runs don't interfere

**Success Criteria:**
- ✅ Full pipeline works
- ✅ Results reproducible
- ✅ All edge cases handled

---

## Implementation Checklist

- [ ] Create train command (Task 6.1)
- [ ] Create generate command (Task 6.2)
- [ ] Create evaluate command (Task 6.3)
- [ ] Create interactive REPL (Task 6.4)
- [ ] Setup CLI entry point (Task 6.5)
- [ ] Create CLI tests (Task 6.6)
- [ ] Create integration tests (Task 6.7)
- [ ] Test coverage > 85%

---

## Success Criteria for Phase 6

1. **CLI Commands**
   - ✅ Train: `fundamentallm train data.txt`
   - ✅ Generate: `fundamentallm generate model.pt --prompt "text"`
   - ✅ Evaluate: `fundamentallm evaluate model.pt test.txt`
   - ✅ All commands have help and clear output

2. **Interactive Mode**
   - ✅ Can launch with `--interactive`
   - ✅ Accept user input and generate responses
   - ✅ Show conversation history
   - ✅ Allow parameter adjustment

3. **Testing**
   - ✅ Unit tests for CLI commands
   - ✅ Integration tests for full pipeline
   - ✅ Coverage > 85%
   - ✅ All tests pass

4. **User Experience**
   - ✅ Clear error messages
   - ✅ Progress indicators
   - ✅ Helpful output formatting
   - ✅ Good documentation

---

## Notes

- Use Click for argument handling (handles validation, help, etc.)
- Use Rich for pretty terminal output
- Test with CliRunner for non-interactive mode
- Use temporary directories in tests for isolation
- Document common use cases in help text
