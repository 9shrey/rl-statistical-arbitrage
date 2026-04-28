from __future__ import annotations

import numpy as np

from arb.backtest.metrics import calmar, hit_rate, max_drawdown, sharpe, sortino, summary, turnover


def test_sharpe_positive_for_steady_returns() -> None:
    rng = np.random.default_rng(0)
    r = 0.001 + rng.normal(0, 0.0005, size=100)  # positive mean, small noise
    assert sharpe(r) > 0


def test_sharpe_zero_for_zero_std() -> None:
    assert sharpe(np.zeros(50)) == 0.0


def test_max_drawdown_known() -> None:
    eq = np.array([1.0, 1.2, 0.6, 0.9])
    assert abs(max_drawdown(eq) - 0.5) < 1e-9


def test_turnover_known() -> None:
    pos = np.array([0, 1, -1, 0, 0])
    # diffs: 1, 2, 1, 0 -> mean = 1.0
    assert abs(turnover(pos) - 1.0) < 1e-9


def test_summary_contract() -> None:
    eq = np.array([1.0, 1.01, 1.02, 1.0])
    pos = np.array([0, 1, 1, 0])
    s = summary(eq, pos)
    for k in ["total_return", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "turnover", "n_bars"]:
        assert k in s


def test_sortino_handles_no_downside() -> None:
    assert sortino(np.full(20, 0.001)) == 0.0


def test_calmar_zero_when_no_drawdown() -> None:
    assert calmar(np.linspace(1.0, 2.0, 50)) == 0.0


def test_hit_rate_known() -> None:
    assert hit_rate(np.array([0.1, -0.1, 0.2])) == 2 / 3
