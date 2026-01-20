"""Preprocessing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def load_text(path: Path, encoding: str = "utf-8") -> str:
    """Load text from disk, falling back gracefully on decoding errors."""
    path = Path(path)
    return path.read_text(encoding=encoding, errors="replace").strip()


def clean_text(text: str, normalizer: Callable[[str], str] | None = None) -> str:
    """Lightweight cleaning: drop control chars and normalize whitespace."""
    if normalizer is not None:
        text = normalizer(text)
    # Remove control characters except newlines and tabs
    text = "".join(ch for ch in text if ch.isprintable() or ch in {"\n", "\t"})
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_training_data(raw_text_path: Path, output_path: Path, clean: bool = True) -> str:
    """Load raw text, optionally clean, and persist processed text."""
    text = load_text(raw_text_path)
    if clean:
        text = clean_text(text)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
