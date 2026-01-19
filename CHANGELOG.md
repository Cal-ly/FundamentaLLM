# Changelog

All notable changes to FundamentaLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete documentation suite (getting started, training guide, architecture)
- Makefile for development workflows
- Pre-commit hooks configuration
- GitHub Actions CI/CD pipelines
- Example Jupyter notebooks

## [0.1.0] - 2026-01-20

### Added
- Character-level tokenizer with special tokens support
- Dataset and DataLoader implementations
- Complete transformer architecture:
  - Multi-head attention with causal masking
  - Position-wise feed-forward networks
  - Layer normalization
  - Token and positional embeddings
- Training infrastructure:
  - Trainer class with train/validation loops
  - Checkpoint management (save/load/resume)
  - Early stopping support
  - Mixed precision training (AMP)
  - Gradient clipping
  - Learning rate scheduling (cosine, linear, step)
- Text generation:
  - Multiple sampling strategies (greedy, temperature, top-k, top-p)
  - Batch generation support
  - Interactive REPL mode
- Model evaluation:
  - Loss and perplexity metrics
  - Bits per character calculation
  - JSON output support
- CLI interface:
  - `train` command with extensive configuration options
  - `generate` command with interactive mode
  - `evaluate` command with metrics reporting
- Configuration system:
  - YAML-based configuration files
  - Preset configs (small, default, large)
  - CLI argument overrides
  - Type-safe config classes
- Comprehensive test suite:
  - 178 passing tests
  - Unit tests for all components
  - Integration tests for full pipelines
  - >85% code coverage
- Documentation:
  - Phase-by-phase implementation plans
  - Lessons learned document
  - API documentation in docstrings
  - Example usage in README

### Fixed
- Path serialization in YAML config files
- Checkpoint format for self-contained model loading
- Evaluation sequence length handling for small datasets

## [0.0.1] - 2026-01-15

### Added
- Initial project structure
- Package scaffolding with setuptools
- Basic README and LICENSE
- Development requirements

---

## Release Notes

### Version 0.1.0 - First Release

FundamentaLLM's first release provides a complete, educational implementation of a transformer language model. Key features:

**Core Capabilities:**
- Train character-level language models from scratch
- Generate text with multiple sampling strategies
- Evaluate model performance with standard metrics
- User-friendly CLI and interactive REPL

**Educational Focus:**
- Clean, well-documented code
- Comprehensive test coverage
- Detailed architecture documentation
- Step-by-step training guides

**Production-Ready Infrastructure:**
- Checkpoint management with resume support
- Mixed precision training for efficiency
- Extensive configuration options
- CI/CD pipelines for automated testing

**Known Limitations:**
- Character-level tokenization only (no BPE/WordPiece)
- Limited to decoder-only architecture
- No fine-tuning support (from-scratch training only)
- Single-GPU training only (no distributed training)

**Future Roadmap:**
- Subword tokenization (BPE, WordPiece)
- Fine-tuning capabilities
- Multi-GPU/distributed training
- Model quantization for inference
- Beam search decoding
- Additional evaluation metrics

---

[Unreleased]: https://github.com/your-org/fundamentallm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/fundamentallm/releases/tag/v0.1.0
[0.0.1]: https://github.com/your-org/fundamentallm/releases/tag/v0.0.1
