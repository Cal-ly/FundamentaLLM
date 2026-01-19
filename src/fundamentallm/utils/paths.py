"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """Create the directory if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
