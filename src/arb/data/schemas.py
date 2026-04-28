"""Data row schemas (pydantic) and dataframe column constants."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

BAR_COLS = ["ts", "symbol", "open", "high", "low", "close", "volume"]
SPREAD_COLS = [
    "ts",
    "sym_a",
    "sym_b",
    "price_a",
    "price_b",
    "hedge_ratio",
    "spread",
    "zscore",
]


class Bar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class SpreadRow(BaseModel):
    ts: datetime
    sym_a: str
    sym_b: str
    price_a: float
    price_b: float
    hedge_ratio: float
    spread: float
    zscore: float
    regime: int = 0
