"""Property tests for reward bounds."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from arb.env.reward import compute_reward


@settings(max_examples=200, deadline=None)
@given(
    pnl=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    delta=st.integers(min_value=-2, max_value=2),
    cost_bps=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    drawdown=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_reward_always_within_clip(pnl: float, delta: int, cost_bps: float, drawdown: float) -> None:
    r = compute_reward(pnl_pct=pnl, delta_position=delta, cost_bps=cost_bps, drawdown=drawdown, clip=0.05)
    assert -0.05 <= r <= 0.05
