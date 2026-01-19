"""Logging utilities."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a simple format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
