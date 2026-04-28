"""Engle-Granger 2-step cointegration test.

Uses statsmodels if available; otherwise a numpy fallback (OLS + ADF p-value
approximation via MacKinnon-style critical values).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CointegrationResult:
    pvalue: float
    hedge_ratio: float
    intercept: float
    half_life_days: float


def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Return (intercept, beta) from y = a + b*x + e via OLS."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion via AR(1) on Δspread = phi * spread_{t-1}."""
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return float("inf")
    ds = np.diff(s)
    s_lag = s[:-1]
    # OLS without intercept: ds = phi * s_lag
    denom = float(np.dot(s_lag, s_lag))
    if denom <= 0:
        return float("inf")
    phi = float(np.dot(s_lag, ds) / denom)
    if phi >= 0:
        return float("inf")  # not mean-reverting
    return float(-np.log(2.0) / np.log(1.0 + phi))


def engle_granger(price_a: pd.Series | np.ndarray, price_b: pd.Series | np.ndarray) -> CointegrationResult:
    """Two-step Engle-Granger: regress a on b, ADF on residuals."""
    a = np.asarray(price_a, dtype=float)
    b = np.asarray(price_b, dtype=float)
    intercept, beta = _ols_hedge_ratio(a, b)
    resid = a - (intercept + beta * b)
    pvalue = _adf_pvalue(resid)
    hl = _half_life(resid)
    return CointegrationResult(pvalue=pvalue, hedge_ratio=beta, intercept=intercept, half_life_days=hl)


def _adf_pvalue(resid: np.ndarray) -> float:
    """ADF p-value on residuals. Tries statsmodels; falls back to a coarse approx."""
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        out = adfuller(resid, regression="c", autolag="AIC")
        return float(out[1])
    except Exception:  # pragma: no cover - fallback path
        # Crude fallback: AR(1) coef test using a normal approximation.
        s = np.asarray(resid, dtype=float)
        ds = np.diff(s)
        s_lag = s[:-1]
        n = len(s_lag)
        if n < 5:
            return 1.0
        denom = float(np.dot(s_lag, s_lag))
        if denom <= 0:
            return 1.0
        phi = float(np.dot(s_lag, ds) / denom)
        # SE approximation
        resid_ar = ds - phi * s_lag
        sigma2 = float(np.dot(resid_ar, resid_ar) / max(n - 1, 1))
        se = float(np.sqrt(sigma2 / denom)) if denom > 0 else 1.0
        t_stat = phi / se if se > 0 else 0.0
        # Use approximate critical values: t < -2.86 ~ 5%, t < -3.43 ~ 1%
        # Map t to a heuristic p-value in (0, 1).
        if t_stat <= -3.43:
            return 0.01
        if t_stat <= -2.86:
            return 0.05
        if t_stat <= -2.57:
            return 0.10
        return 0.50
