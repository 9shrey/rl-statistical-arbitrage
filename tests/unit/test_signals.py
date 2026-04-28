from __future__ import annotations

import numpy as np
import pandas as pd

from arb.signals.spread import make_spread, rolling_hedge_ratio
from arb.signals.zscore import rolling_zscore


def test_rolling_hedge_ratio_recovers_known_beta(rng: np.random.Generator) -> None:
    n = 300
    x = pd.Series(np.cumsum(rng.normal(0, 1.0, size=n)) + 50.0)
    noise = pd.Series(rng.normal(0, 0.05, size=n))
    true_beta = 1.7
    y = true_beta * x + noise + 10.0
    beta_hat = rolling_hedge_ratio(y, x, window=60).dropna()
    assert abs(beta_hat.iloc[-1] - true_beta) < 0.05


def test_rolling_zscore_no_lookahead_first_window_is_nan(rng: np.random.Generator) -> None:
    n = 50
    s = pd.Series(rng.normal(0, 1.0, size=n))
    z = rolling_zscore(s, window=10)
    # Need 10 prior obs (after shift(1)), so first 10 values must be NaN.
    assert z.iloc[:10].isna().all()
    assert not z.iloc[15:].isna().all()


def test_make_spread_matches_formula() -> None:
    a = pd.Series([10.0, 11.0, 12.0])
    b = pd.Series([5.0, 5.5, 6.0])
    beta = pd.Series([2.0, 2.0, 2.0])
    s = make_spread(a, b, beta)
    assert list(s.values) == [0.0, 0.0, 0.0]
