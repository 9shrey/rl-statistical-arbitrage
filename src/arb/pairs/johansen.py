"""Johansen test wrapper (statsmodels, optional)."""
from __future__ import annotations

import numpy as np


def johansen_trace_stat(prices: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> np.ndarray:
    """Return Johansen trace statistics; raises if statsmodels not installed."""
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("statsmodels required for Johansen test") from e
    res = coint_johansen(prices, det_order=det_order, k_ar_diff=k_ar_diff)
    return np.asarray(res.lr1)
