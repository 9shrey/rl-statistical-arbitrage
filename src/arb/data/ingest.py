"""Data ingestion: fixture / csv / yfinance.

Tests must use ``source="fixture"`` (or ``"csv"`` against committed files).
``yfinance`` is opt-in for the CLI ``arb data fetch`` command only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from arb.data.calendar import business_days
from arb.data.schemas import BAR_COLS


def _read_csv_bars(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df["symbol"] = symbol
    return df[BAR_COLS]


def load_fixture_pair(
    cache_dir: str | Path,
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load a pair from CSV fixtures; if missing, synthesize a cointegrated pair.

    Synthesis uses a common stochastic factor + idiosyncratic noise so the pair
    is genuinely cointegrated. Reproducible given ``seed``.
    """
    cache = Path(cache_dir)
    out: dict[str, pd.DataFrame] = {}
    missing = [s for s in symbols if not (cache / f"{s}.csv").exists()]
    if not missing:
        for sym in symbols:
            out[sym] = _read_csv_bars(cache / f"{sym}.csv", sym)
        return out

    # Synthesize.
    rng = np.random.default_rng(seed)
    idx = business_days(start, end)
    n = len(idx)
    # Common factor (random walk)
    factor = np.cumsum(rng.normal(0, 1.0, size=n))
    # Cointegrated levels: y_a = a + b*factor + e_a; y_b = c + d*factor + e_b
    # Choose so spread = price_a - beta * price_b is stationary AR(1).
    base_a = 100.0
    base_b = 50.0
    e_a = rng.normal(0, 0.5, size=n)
    e_b = rng.normal(0, 0.5, size=n)
    # Stationary spread component
    ar_phi = 0.85
    ar_eps = rng.normal(0, 0.6, size=n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = ar_phi * spread[t - 1] + ar_eps[t]
    price_b = base_b + 0.5 * factor + e_b
    price_a = base_a + 1.0 * factor + spread + e_a

    for sym, prices in zip(symbols, [price_a, price_b], strict=True):
        # Build OHLCV with small intra-bar range and synthetic volume.
        rng_local = np.random.default_rng(seed + hash(sym) % 1000)
        noise = rng_local.normal(0, 0.05, size=(n, 2))
        high = prices + np.abs(noise[:, 0])
        low = prices - np.abs(noise[:, 1])
        # Open ~ previous close (first open = first close).
        open_ = np.concatenate([[prices[0]], prices[:-1]])
        volume = rng_local.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame(
            {
                "ts": idx,
                "symbol": sym,
                "open": open_,
                "high": high,
                "low": low,
                "close": prices,
                "volume": volume,
            }
        )
        out[sym] = df[BAR_COLS]

    cache.mkdir(parents=True, exist_ok=True)
    for sym, df in out.items():
        df.to_csv(cache / f"{sym}.csv", index=False)
    return out


def load_pair(
    source: str,
    cache_dir: str | Path,
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if source in ("fixture", "csv"):
        return load_fixture_pair(cache_dir, symbols, start, end, seed=seed)
    if source == "yfinance":  # pragma: no cover - network
        try:
            import yfinance as yf  # type: ignore
        except ImportError as e:
            raise RuntimeError("yfinance not installed; pip install '.[data]'") from e
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = yf.download(sym, start=str(start), end=str(end), progress=False, auto_adjust=True)
            df = df.reset_index().rename(
                columns={
                    "Date": "ts",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["symbol"] = sym
            out[sym] = df[BAR_COLS]
        return out
    raise ValueError(f"Unknown data source: {source}")


def align_pair(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join two symbols on ts; return columns: ts, sym_a, price_a, price_b, ...."""
    syms = list(bars.keys())
    if len(syms) != 2:
        raise ValueError("align_pair expects exactly 2 symbols")
    a, b = syms
    da = bars[a].set_index("ts")
    db = bars[b].set_index("ts")
    joined = da.join(db, lsuffix="_a", rsuffix="_b", how="inner").reset_index()
    out = pd.DataFrame(
        {
            "ts": joined["ts"],
            "sym_a": a,
            "sym_b": b,
            "price_a": joined["close_a"],
            "price_b": joined["close_b"],
            "high_a": joined["high_a"],
            "low_a": joined["low_a"],
            "high_b": joined["high_b"],
            "low_b": joined["low_b"],
            "volume_a": joined["volume_a"],
            "volume_b": joined["volume_b"],
        }
    )
    return out.sort_values("ts").reset_index(drop=True)
