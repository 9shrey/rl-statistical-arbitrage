"""Point-in-time universe stub. v1: trivial pass-through over configured symbols."""
from __future__ import annotations


def universe_at(_date: str, symbols: list[str]) -> list[str]:
    """Return the configured universe. Real PIT logic would look up snapshots."""
    return list(symbols)
