"""Pair ranking by composite score."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairScore:
    sym_a: str
    sym_b: str
    pvalue: float
    half_life_days: float
    score: float


def composite_score(pvalue: float, half_life_days: float) -> float:
    """Higher is better. Penalize p-value and very long half-lives."""
    pv = max(0.0, min(1.0, pvalue))
    hl = max(1.0, half_life_days) if half_life_days != float("inf") else 1e6
    return 0.6 * (1.0 - pv) + 0.4 * (1.0 / hl)
