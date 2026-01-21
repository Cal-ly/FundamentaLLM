"""Configuration base class with YAML I/O and env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")


def _apply_env_overrides(data: Dict[str, Any], prefixes: list[str]) -> None:
    """Apply environment variable overrides using PREFIX__SECTION__KEY naming."""

    for env_key, raw_value in os.environ.items():
        matched_prefix = next((p for p in prefixes if env_key.startswith(p)), None)
        if matched_prefix is None:
            continue

        key_path = env_key[len(matched_prefix) :].lstrip("_")
        if not key_path:
            continue

        keys = key_path.lower().split("__")
        cursor: Dict[str, Any] = data
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = yaml.safe_load(raw_value)


class BaseConfig(BaseModel):
    """Shared config behavior for FundamentaLLM."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dumped = self.model_dump()
        # Convert Path objects to strings for YAML serialization
        for key, value in list(dumped.items()):
            if isinstance(value, Path):
                dumped[key] = str(value)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dumped, handle, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[T], path: Path, env_prefix: str | list[str] = "FLLM") -> T:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Configuration YAML must define a mapping at the root")

        prefixes = [env_prefix] if isinstance(env_prefix, str) else list(env_prefix)
        # Support both legacy (FLLM__) and documented (FUNDAMENTALLM__) prefixes
        if "FUNDAMENTALLM" not in prefixes:
            prefixes.append("FUNDAMENTALLM")
        _apply_env_overrides(data, prefixes)
        return cls.model_validate(data)
