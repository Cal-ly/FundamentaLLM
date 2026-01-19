.PHONY: help install install-dev test test-cov lint format format-check type-check clean docs pre-commit

help:
	@echo "FundamentaLLM Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linters (flake8)"
	@echo "  make type-check    - Run type checker (mypy)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make format-check  - Check format without changes"
	@echo "  make pre-commit    - Run all pre-commit hooks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts"
	@echo ""
	@echo "Docs:"
	@echo "  make docs          - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install || echo "pre-commit not installed, skipping hook setup"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/fundamentallm --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503 || echo "flake8 not installed"

type-check:
	@echo "Running mypy..."
	mypy src/fundamentallm --ignore-missing-imports || echo "mypy not installed or found issues"

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/

format-check:
	@echo "Checking format with black..."
	black --check src/ tests/
	@echo "Checking imports with isort..."
	isort --check src/ tests/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

docs:
	@echo "Building documentation..."
	cd docs && make html 2>/dev/null || echo "Sphinx not installed. Run: pip install sphinx sphinx-rtd-theme"

pre-commit:
	pre-commit run --all-files || echo "pre-commit not installed. Run: pip install pre-commit"
