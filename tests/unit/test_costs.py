from __future__ import annotations

from arb.env.costs import CostModel


def test_cost_model_basic() -> None:
    cm = CostModel(commission_bps_per_side=0.5, slippage_half_spread_bps=1.0, impact_bps_per_pct_adv=10.0)
    # 1% of ADV trade
    bps = cm.trade_cost_bps(0.01)
    assert abs(bps - (0.5 + 1.0 + 10.0 * 0.01)) < 1e-9


def test_cost_model_no_negative_impact() -> None:
    cm = CostModel()
    assert cm.trade_cost_bps(-1.0) == cm.trade_cost_bps(0.0)
