"""Reward computation: risk-adjusted PnL minus costs and penalties."""
from __future__ import annotations


def compute_reward(
    pnl_pct: float,
    delta_position: int,
    cost_bps: float,
    drawdown: float,
    *,
    lambda_turnover: float = 0.0005,
    dd_threshold: float = 0.10,
    dd_penalty: float = 0.5,
    clip: float = 0.05,
) -> float:
    """Per-step reward in PnL-equivalent fractional units.

    Bounded by ``[-clip, +clip]`` to stabilize training.
    """
    cost_pct = (cost_bps / 1e4) * abs(delta_position)
    turnover_penalty = lambda_turnover * abs(delta_position)
    dd_excess = max(0.0, drawdown - dd_threshold)
    dd_pen = dd_penalty * dd_excess
    raw = pnl_pct - cost_pct - turnover_penalty - dd_pen
    if raw > clip:
        return clip
    if raw < -clip:
        return -clip
    return raw
