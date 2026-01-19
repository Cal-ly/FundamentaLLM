# FundamentaLLM

[![CI](https://github.com/your-org/fundamentallm/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/fundamentallm/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An educational, character-level transformer language model framework built in PyTorch. FundamentaLLM makes core LLM concepts approachable while maintaining production-quality engineering practices: type safety, comprehensive tests, and configuration-first design.

## Features

- **Educational Focus**: Clean, well-documented code designed for learning
- **Complete Pipeline**: Tokenization → Training → Generation → Evaluation
- **Interactive CLI**: User-friendly commands with Rich-powered REPL
- **Modern PyTorch**: Mixed precision training, gradient clipping, LR scheduling
- **Comprehensive Testing**: 178 tests with >85% coverage
- **Extensive Docs**: Getting started guide, training best practices, architecture deep dive
- **Checkpoint Management**: Save/load/resume training with full state preservation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/fundamentallm.git
cd fundamentallm

# Install in development mode
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

### Train Your First Model

```bash
# Train on sample data
fundamentallm train data/samples/sample_data.txt \
    --output-dir my_model \
    --epochs 5

# Generate text
fundamentallm generate my_model/final_model.pt \
    --prompt "Once upon a time" \
    --temperature 0.8
```

### Interactive Mode

```bash
fundamentallm generate my_model/final_model.pt --interactive
```

**Output:**
```
╭──────────────────────────────────────────────────╮
│ FundamentaLLM Interactive Mode                   │
│ Type /help for commands, /quit to exit          │
╰──────────────────────────────────────────────────╯

> Hello!
Hello! How can I help you today?

> /set temperature=1.0
Updated setting: temperature=1.0
```

## Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Installation, first model, troubleshooting
- **[Training Guide](docs/training_guide.md)** - Hyperparameter tuning, best practices, performance tips
- **[Architecture Guide](docs/architecture.md)** - Deep dive into transformer implementation
- **[Example Notebooks](docs/notebooks/)** - Interactive tutorials (coming soon)
- **[API Reference](docs/api_reference.md)** - Complete API documentation (coming soon)

## Architecture

FundamentaLLM implements a decoder-only transformer (GPT-style) with:

- **Character-level tokenization**: No vocabulary limits, handles any UTF-8 text
- **Multi-head attention**: Causal masking for autoregressive generation
- **Position-wise FFN**: GELU activation, configurable expansion ratio
- **Layer normalization**: Pre-norm architecture for training stability
- **Sinusoidal positional encoding**: No learned position embeddings

```
Input Text → Tokenizer → Dataset → Model → Training → Checkpoints
                                      ↓
                                  Generation → Output Text
```

See [Architecture Guide](docs/architecture.md) for detailed explanation.

## CLI Commands

### Train

```bash
fundamentallm train <data_path> [OPTIONS]

Options:
  --config PATH           Config file (YAML)
  --output-dir PATH       Output directory [default: checkpoints]
  --epochs INT           Training epochs [default: 10]
  --batch-size INT       Batch size [default: 32]
  --learning-rate FLOAT  Learning rate [default: 3e-4]
  --device CHOICE        Device (cpu/cuda/mps) [default: auto]
```

### Generate

```bash
fundamentallm generate <checkpoint_path> [OPTIONS]

Options:
  --prompt TEXT          Generation prompt
  --interactive          Interactive mode (REPL)
  --max-tokens INT       Max tokens [default: 100]
  --temperature FLOAT    Sampling temperature [default: 0.8]
  --top-k INT           Top-k sampling [default: 50]
  --top-p FLOAT         Nucleus sampling [default: 0.95]
```

### Evaluate

```bash
fundamentallm evaluate <checkpoint_path> --data <test_data> [OPTIONS]

Options:
  --batch-size INT       Batch size [default: 32]
  --json                 Output as JSON
```

## Configuration

Use YAML files for reproducible training:

```yaml
# config.yaml
model:
  d_model: 512
  num_heads: 8
  num_layers: 6
  d_ff: 2048
  dropout: 0.1

training:
  max_epochs: 10
  learning_rate: 3e-4
  batch_size: 32
  warmup_steps: 100
  scheduler: "cosine"
  gradient_clip_norm: 1.0
```

**Preset configs:**
- `configs/small.yaml` - Fast training, limited capacity (~500K params)
- `configs/default.yaml` - Balanced performance (~40M params)
- `configs/large.yaml` - High capacity, resource-intensive (~150M params)

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test
pytest tests/unit/test_attention.py -v
```

### Code Quality

```bash
# Format code
make format

# Check formatting
make format-check

# Type checking
make type-check

# Linting
make lint

# All pre-commit hooks
make pre-commit
```

### Available Make Commands

```bash
make help  # Show all commands
```

## Project Status

**Current Version:** 0.1.0 (Release Candidate)

**Test Coverage:** 178 passing tests, >85% coverage

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Contribution Workflow:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "Add amazing feature"`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Learning Resources

**Recommended Reading:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)

**Related Projects:**
- [Karpathy's minGPT](https://github.com/karpathy/minGPT) - Minimal GPT implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Production library
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimalist character-level GPT

## Roadmap

**Future Enhancements:**
- [ ] Subword tokenization (BPE, WordPiece)
- [ ] Fine-tuning support
- [ ] Multi-GPU/distributed training
- [ ] Beam search decoding
- [ ] Model quantization for inference
- [ ] Weights & Biases integration
- [ ] More example notebooks
- [ ] Gradio/Streamlit web interface

## License

This project is licensed under the AGPL-3.0 License - see [LICENSE](LICENSE) for details.
