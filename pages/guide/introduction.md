# Introduction to FundamentaLLM

Welcome to FundamentaLLM, an educational framework designed to make large language models (LLMs) and transformer architectures understandable and approachable.

## What is FundamentaLLM?

FundamentaLLM is a PyTorch-based framework that implements a complete language modeling pipeline. It's specifically designed for **learning**, combining:

- **Clear, educational code** - Every function is documented with "why" not just "what"
- **Complete implementations** - Not toy examples, but working production-quality code
- **Theory + Practice** - Learn mathematical foundations alongside practical implementations
- **Best practices** - Type safety, comprehensive testing, proper configuration management

## Who Should Use This?

- **Students** learning about transformers and NLP
- **Practitioners** wanting to understand LLM internals before using APIs
- **Developers** building from scratch to appreciate the engineering
- **Researchers** exploring LLM architectures

## The Educational Philosophy

Rather than treating LLMs as black boxes, FundamentaLLM breaks them down:

```
┌─────────────────────────────────────────┐
│ What: Character-Level Transformer       │
├─────────────────────────────────────────┤
│ Why: Handles any UTF-8 text without     │
│      vocabulary management overhead     │
├─────────────────────────────────────────┤
│ How: Multi-head attention + FFN layers  │
│      with layer normalization           │
└─────────────────────────────────────────┘
```

This documentation follows the same approach: explaining **what** each component does, **why** it's designed that way, and **how** it connects to theory.

## What You'll Learn

### By Using FundamentaLLM, you'll understand:

1. **How transformers work** - From positional encoding to multi-head attention
2. **The complete ML pipeline** - Data → Model → Training → Generation
3. **Practical engineering** - Configuration, checkpoints, mixed precision training
4. **How to evaluate models** - Beyond just accuracy
5. **Why design decisions matter** - And how to make them

## Getting Started

1. **[Tech Stack Overview](./tech-stack.md)** - What this project uses and why
2. **[Installation](./installation.md)** - Get up and running in 5 minutes
3. **[Quick Start](./quick-start.md)** - Train your first model
4. **[Concept Deep Dives](../concepts/overview.md)** - Understand the theory

## Structure of This Documentation

- **Guide**: Practical how-to documentation
- **Concepts**: Theoretical foundations and explanations
- **Modules**: Deep dives into each Python module
- **Tutorials**: Step-by-step walkthroughs

Each section includes both the "what works" and the "why it works that way."

## Key Features

- **Character-level tokenization** - No vocabulary complexity
- **Complete transformer** - Attention, FFN, LayerNorm, positional encoding
- **Advanced training** - Mixed precision, gradient clipping, LR scheduling
- **Checkpoint management** - Save, load, resume training
- **Interactive generation** - Chat-like interface for exploring models
- **Comprehensive testing** - See ML best practices in action

## Next Steps

Ready to dive in? Start with the [Tech Stack Overview](./tech-stack.md), then head to [Installation](./installation.md).

Or jump straight to [Concepts](../concepts/overview.md) if you want to understand the theory first.
