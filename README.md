# FundamentaLLM

An educational, character-level transformer language model framework built in PyTorch. The goal is to make core LLM ideas approachable while keeping production-quality engineering practices (type safety, tests, configuration-first design).

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Train (CLI arrives in Phase 6):

```bash
python -m fundamentallm  # Placeholder until CLI lands
```

## Docs

- Architecture & design: [docs/instruct/DESIGN_SYSTEM.md](docs/instruct/DESIGN_SYSTEM.md)
- Phase plans: [PLAN_INDEX.md](PLAN_INDEX.md)

## Project Structure

See [docs/instruct/DESIGN_SYSTEM.md](docs/instruct/DESIGN_SYSTEM.md) for the full layout and design rationale.

## License

This project is currently published under AGPL-3.0. See [LICENSE](LICENSE).
