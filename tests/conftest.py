from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arb.data.ingest import align_pair, load_fixture_pair
from arb.signals.features import build_features


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    d = repo_root / "tests" / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def synthetic_pair(fixtures_dir: Path) -> pd.DataFrame:
    bars = load_fixture_pair(
        cache_dir=fixtures_dir,
        symbols=["SYNA", "SYNB"],
        start="2020-01-01",
        end="2022-12-31",
        seed=42,
    )
    return align_pair(bars)


@pytest.fixture(scope="session")
def feature_df(synthetic_pair: pd.DataFrame) -> pd.DataFrame:
    return build_features(synthetic_pair, hedge_window=30, zscore_window=10)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
