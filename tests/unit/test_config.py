from __future__ import annotations

from pathlib import Path

import yaml

from arb.config import load_config


def test_load_default_config(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs" / "default.yaml")
    assert cfg.seed == 42
    assert len(cfg.data.symbols) >= 2


def test_load_smoke_config(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs" / "smoke.yaml")
    assert cfg.data.source == "fixture"
    assert cfg.train.algo in {"zscore_baseline", "grid_baseline", "buy_hold", "ppo", "sac"}


def test_invalid_config_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"seed": 1, "data": {"start": "2020-01-01", "end": "2020-12-31", "symbols": ["X"]}}))
    try:
        load_config(p)
    except Exception:
        return
    raise AssertionError("expected validation error for single-symbol config")
