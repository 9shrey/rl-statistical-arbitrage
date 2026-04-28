"""Assemble env feature matrix from a price-aligned pair frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from arb.signals.microstructure import bid_ask_proxy, realized_vol
from arb.signals.spread import make_spread, rolling_hedge_ratio
from arb.signals.zscore import rolling_zscore


def build_features(
    pair_df: pd.DataFrame,
    hedge_window: int,
    zscore_window: int,
) -> pd.DataFrame:
    """Build the full per-bar feature frame used by env / regime / backtest.

    Required input columns: ts, price_a, price_b, high_a, low_a, high_b, low_b.
    Output columns include: hedge_ratio, spread, zscore, realized_vol, bid_ask_proxy,
    abs_zscore, ret_a, ret_b.
    """
    df = pair_df.copy().reset_index(drop=True)
    df["hedge_ratio"] = rolling_hedge_ratio(df["price_a"], df["price_b"], hedge_window).values
    df["spread"] = make_spread(df["price_a"], df["price_b"], df["hedge_ratio"]).values
    df["zscore"] = rolling_zscore(df["spread"], zscore_window).values
    df["abs_zscore"] = df["zscore"].abs()
    df["ret_a"] = df["price_a"].pct_change()
    df["ret_b"] = df["price_b"].pct_change()
    spread_ret = df["spread"].diff()
    df["realized_vol"] = realized_vol(spread_ret.fillna(0.0), window=zscore_window).values
    bap_a = bid_ask_proxy(df["high_a"], df["low_a"], df["price_a"], window=zscore_window)
    bap_b = bid_ask_proxy(df["high_b"], df["low_b"], df["price_b"], window=zscore_window)
    df["bid_ask_proxy"] = ((bap_a + bap_b) / 2.0).values
    return df


def feature_matrix(df: pd.DataFrame, regime_onehot: np.ndarray | None = None) -> np.ndarray:
    """Return float32 (N, F) observation matrix in the canonical column order."""
    cols = ["zscore", "realized_vol", "bid_ask_proxy"]
    base = df[cols].to_numpy(dtype=np.float32)
    if regime_onehot is not None:
        base = np.concatenate([base, regime_onehot.astype(np.float32)], axis=1)
    return base
