"""Portfolio bookkeeping for a single spread position.

Position semantics: position in {-1, 0, +1} (units of "spread"):
  +1  =>  long  price_a, short hedge_ratio * price_b
  -1  =>  short price_a, long  hedge_ratio * price_b
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Portfolio:
    cash: float = 1.0  # normalized equity = 1.0 at start
    position: int = 0  # in {-1, 0, +1}
    equity: float = 1.0
    peak_equity: float = 1.0

    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def update_equity(self, new_equity: float) -> None:
        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
