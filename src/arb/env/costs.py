"""Cost model: commission, slippage, and linear market impact."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps_per_side: float = 0.5
    slippage_half_spread_bps: float = 1.0
    impact_bps_per_pct_adv: float = 10.0

    def trade_cost_bps(self, traded_notional_pct_adv: float) -> float:
        """Total cost in basis points for a single side of a trade."""
        return (
            self.commission_bps_per_side
            + self.slippage_half_spread_bps
            + self.impact_bps_per_pct_adv * max(traded_notional_pct_adv, 0.0)
        )
