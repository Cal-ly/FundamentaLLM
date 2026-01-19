# Contributing to FundamentaLLM

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest -q
pytest --cov=fundamentallm --cov-report=term-missing
```

## Code Quality

```bash
black .
isort .
mypy .
pylint src/fundamentallm
```

## Phased Delivery

This project is implemented incrementally. See the phase plan documents (PHASE_*.md) and [PLAN_INDEX.md](PLAN_INDEX.md) for current focus.
