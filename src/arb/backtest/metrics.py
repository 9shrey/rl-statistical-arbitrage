"""Performance metrics."""
from __future__ import annotations

import numpy as np

ANN = 252.0


def _to_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def sharpe(returns, periods_per_year: float = ANN) -> float:
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * mu / sd)


def sortino(returns, periods_per_year: float = ANN) -> float:
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=0) if downside.size else 0.0
    if dd <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dd)


def max_drawdown(equity_curve) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.where(peaks > 0, peaks, 1.0)
    return float(dd.max())


def calmar(equity_curve, periods_per_year: float = ANN) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size < 2:
        return 0.0
    n_periods = eq.size - 1
    total_return = eq[-1] / eq[0] - 1.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if (1.0 + total_return) > 0 else -1.0
    mdd = max_drawdown(eq)
    if mdd <= 1e-12:
        return 0.0
    return float(cagr / mdd)


def hit_rate(returns) -> float:
    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    return float((r > 0).mean())


def turnover(positions) -> float:
    p = np.asarray(positions, dtype=float)
    if p.size < 2:
        return 0.0
    return float(np.abs(np.diff(p)).sum() / max(p.size - 1, 1))


def summary(equity_curve, positions) -> dict[str, float]:
    eq = np.asarray(equity_curve, dtype=float)
    rets = (eq[1:] / eq[:-1]) - 1.0 if eq.size >= 2 else np.array([])
    return {
        "total_return": float(eq[-1] / eq[0] - 1.0) if eq.size >= 2 else 0.0,
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "calmar": calmar(eq),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": hit_rate(rets),
        "turnover": turnover(positions),
        "n_bars": int(eq.size),
    }
