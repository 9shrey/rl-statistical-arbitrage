"""Property tests for look-ahead-free signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from arb.signals.spread import rolling_hedge_ratio
from arb.signals.zscore import rolling_zscore


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n=st.integers(min_value=80, max_value=200),
    window=st.integers(min_value=5, max_value=20),
)
def test_rolling_zscore_no_lookahead_shift_invariance(seed: int, n: int, window: int) -> None:
    """Z-score at index t may depend only on data at indices <= t-1.

    Construct two series identical up to index t* and arbitrary after; the
    z-score values up to and including t* must match.
    """
    rng = np.random.default_rng(seed)
    base = pd.Series(rng.normal(0, 1, size=n))
    perturb = base.copy()
    t_star = n // 2
    perturb.iloc[t_star + 1 :] += rng.normal(0, 5, size=n - t_star - 1)

    z_base = rolling_zscore(base, window=window)
    z_pert = rolling_zscore(perturb, window=window)

    a = z_base.iloc[: t_star + 1].dropna()
    b = z_pert.iloc[: t_star + 1].dropna()
    common = a.index.intersection(b.index)
    if len(common):
        np.testing.assert_allclose(a.loc[common].values, b.loc[common].values, rtol=1e-9, atol=1e-9)


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n=st.integers(min_value=120, max_value=200),
    window=st.integers(min_value=10, max_value=30),
)
def test_rolling_hedge_ratio_no_lookahead(seed: int, n: int, window: int) -> None:
    rng = np.random.default_rng(seed)
    a = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
    b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50)
    a_pert = a.copy()
    b_pert = b.copy()
    t_star = n // 2
    a_pert.iloc[t_star + 1 :] += rng.normal(0, 5, size=n - t_star - 1)
    b_pert.iloc[t_star + 1 :] += rng.normal(0, 5, size=n - t_star - 1)

    h1 = rolling_hedge_ratio(a, b, window=window)
    h2 = rolling_hedge_ratio(a_pert, b_pert, window=window)

    cmp = h1.iloc[: t_star + 1].dropna()
    cmp2 = h2.iloc[: t_star + 1].dropna()
    common = cmp.index.intersection(cmp2.index)
    if len(common):
        np.testing.assert_allclose(cmp.loc[common].values, cmp2.loc[common].values, rtol=1e-8, atol=1e-8)
