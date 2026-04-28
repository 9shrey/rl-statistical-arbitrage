"""Risk sizing helpers (vol-targeting; v1 returns scaling factor only)."""
from __future__ import annotations


def vol_target_scale(realized_vol_annual: float, target_annual: float) -> float:
    """Position scaling factor to hit target annualized vol; clipped to [0.1, 5.0]."""
    if realized_vol_annual <= 1e-8:
        return 1.0
    s = target_annual / realized_vol_annual
    return max(0.1, min(5.0, s))
