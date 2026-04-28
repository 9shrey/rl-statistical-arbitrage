"""Trading-day calendar utilities (NYSE-like)."""
from __future__ import annotations

import pandas as pd


def business_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Return tz-naive business days between start and end (inclusive)."""
    return pd.bdate_range(start=start, end=end)
