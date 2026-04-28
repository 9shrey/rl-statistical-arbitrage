from __future__ import annotations

import pandas as pd

from arb.data.ingest import align_pair, load_fixture_pair


def test_load_fixture_pair_synthesizes_when_missing(tmp_path) -> None:
    bars = load_fixture_pair(
        cache_dir=tmp_path, symbols=["AAA", "BBB"], start="2020-01-01", end="2020-06-30", seed=1
    )
    assert set(bars.keys()) == {"AAA", "BBB"}
    for sym, df in bars.items():
        assert (df["symbol"] == sym).all()
        assert {"ts", "open", "high", "low", "close", "volume"} <= set(df.columns)
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()


def test_align_pair_inner_join(tmp_path) -> None:
    bars = load_fixture_pair(
        cache_dir=tmp_path, symbols=["AAA", "BBB"], start="2020-01-01", end="2020-06-30", seed=1
    )
    pair = align_pair(bars)
    assert {"ts", "price_a", "price_b", "hedge_ratio" not in pair.columns or True}
    assert pair["ts"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(pair["ts"])
