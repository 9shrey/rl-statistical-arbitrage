"""YAML config loader with pydantic validation."""
from __future__ import annotations

from pathlib import Path

import yaml

from arb.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig(**raw)
