"""Rolling z-score, look-ahead-free."""
from __future__ import annotations

import pandas as pd


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Z = (s - rolling_mean) / rolling_std, computed strictly from past bars.

    Uses ``shift(1)`` so the value at t depends only on bars <= t-1.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    mean = spread.shift(1).rolling(window=window, min_periods=window).mean()
    std = spread.shift(1).rolling(window=window, min_periods=window).std(ddof=0)
    z = (spread - mean) / std.replace(0.0, float("nan"))
    return z.rename("zscore")
