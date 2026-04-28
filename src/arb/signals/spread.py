"""Rolling hedge ratio + spread construction.

Strict no-look-ahead: every value at index t uses information from indices
``[..., t-1]`` only, then is shifted by one bar before being applied at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_hedge_ratio(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """OLS slope of price_a ~ price_b over a trailing window, shifted by 1 bar."""
    if window < 2:
        raise ValueError("window must be >= 2")

    a = price_a.to_numpy(dtype=float)
    b = price_b.to_numpy(dtype=float)
    n = len(a)
    out = np.full(n, np.nan)

    # Cumulative sums for O(n) rolling OLS.
    csum_a = np.concatenate([[0.0], np.cumsum(a)])
    csum_b = np.concatenate([[0.0], np.cumsum(b)])
    csum_aa = np.concatenate([[0.0], np.cumsum(a * a)])
    csum_bb = np.concatenate([[0.0], np.cumsum(b * b)])
    csum_ab = np.concatenate([[0.0], np.cumsum(a * b)])

    for t in range(window, n + 1):
        s, e = t - window, t
        sum_a = csum_a[e] - csum_a[s]
        sum_b = csum_b[e] - csum_b[s]
        sum_bb = csum_bb[e] - csum_bb[s]
        sum_ab = csum_ab[e] - csum_ab[s]
        mean_a = sum_a / window
        mean_b = sum_b / window
        var_b = sum_bb / window - mean_b * mean_b
        cov_ab = sum_ab / window - mean_a * mean_b
        if var_b > 1e-12:
            beta = cov_ab / var_b
        else:
            beta = np.nan
        out[t - 1] = beta

    s = pd.Series(out, index=price_a.index, name="hedge_ratio")
    # Shift by 1 so beta used at t is computed strictly from data <= t-1.
    return s.shift(1)


def make_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: pd.Series) -> pd.Series:
    return (price_a - hedge_ratio * price_b).rename("spread")
