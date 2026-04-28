from __future__ import annotations

from arb.env.reward import compute_reward


def test_reward_is_clipped() -> None:
    r = compute_reward(pnl_pct=10.0, delta_position=0, cost_bps=0, drawdown=0, clip=0.05)
    assert r == 0.05
    r = compute_reward(pnl_pct=-10.0, delta_position=0, cost_bps=0, drawdown=0, clip=0.05)
    assert r == -0.05


def test_reward_penalizes_churn() -> None:
    # Same pnl, but with turnover -> lower reward
    r_no_trade = compute_reward(pnl_pct=0.001, delta_position=0, cost_bps=0, drawdown=0)
    r_trade = compute_reward(pnl_pct=0.001, delta_position=2, cost_bps=10.0, drawdown=0)
    assert r_trade < r_no_trade


def test_reward_penalizes_drawdown_excess() -> None:
    base = compute_reward(pnl_pct=0.001, delta_position=0, cost_bps=0, drawdown=0.05, dd_threshold=0.10)
    deep = compute_reward(pnl_pct=0.001, delta_position=0, cost_bps=0, drawdown=0.30, dd_threshold=0.10)
    assert deep < base
