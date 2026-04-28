"""End-to-end smoke test: full pipeline on the synthetic fixture pair.

This is the test that gates the whole project: data -> features -> walk-forward
backtest, with at least one fold and finite metrics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arb.config import load_config
from arb.cli import _pipeline


@pytest.mark.integration
def test_smoke_pipeline_end_to_end(repo_root: Path) -> None:
    cfg_path = repo_root / "configs" / "smoke.yaml"
    out = _pipeline(str(cfg_path))
    assert out["n_folds"] >= 1
    agg = out["aggregate"]
    assert "mean_sharpe" in agg
    assert agg["folds"] == out["n_folds"]


@pytest.mark.integration
def test_smoke_pipeline_deterministic(repo_root: Path) -> None:
    cfg_path = repo_root / "configs" / "smoke.yaml"
    a = _pipeline(str(cfg_path))
    b = _pipeline(str(cfg_path))
    assert a["aggregate"] == b["aggregate"]


@pytest.mark.integration
def test_smoke_config_loads_strict(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs" / "smoke.yaml")
    assert cfg.data.source == "fixture"
