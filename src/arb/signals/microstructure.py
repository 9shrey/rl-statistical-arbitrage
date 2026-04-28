"""Microstructure proxy features from OHLCV bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling std of returns, look-ahead free."""
    return returns.shift(1).rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(252)


def bid_ask_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Proxy: trailing mean of (high - low) / close. No look-ahead."""
    rng = ((high - low) / close.replace(0.0, float("nan"))).shift(1)
    return rng.rolling(window=window, min_periods=window).mean()


def return_autocorr(returns: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    """Rolling autocorrelation of returns at given lag. No look-ahead."""
    r = returns.shift(1)
    return r.rolling(window=window, min_periods=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )
