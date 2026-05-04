from __future__ import annotations

import numpy as np
import pandas as pd

from arb.env import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    CostModel,
    PairsTradingEnv,
    make_env_spec,
)


def _toy_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    spread = np.cumsum(rng.normal(0, 0.1, n))
    price_a = 100 + spread
    price_b = 50 + 0.0 * spread
    return pd.DataFrame(
        {
            "ts": pd.bdate_range("2020-01-01", periods=n),
            "price_a": price_a,
            "price_b": price_b,
            "hedge_ratio": np.full(n, 1.0),
            "spread": price_a - 1.0 * price_b,
            "zscore": (spread - spread.mean()) / (spread.std() + 1e-9),
            "realized_vol": np.full(n, 0.2),
            "bid_ask_proxy": np.full(n, 0.001),
        }
    )


def test_env_step_terminates_at_end() -> None:
    df = _toy_df(40)
    spec = make_env_spec(df, ["zscore", "realized_vol", "bid_ask_proxy"], CostModel())
    env = PairsTradingEnv(spec, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == spec.features.shape[1] + 1
    steps = 0
    done = False
    while not done:
        _, _r, term, trunc, _ = env.step(ACTION_FLAT)
        done = term or trunc
        steps += 1
    assert steps == len(spec.features)


def test_env_action_changes_position() -> None:
    df = _toy_df(20)
    spec = make_env_spec(df, ["zscore", "realized_vol", "bid_ask_proxy"], CostModel())
    env = PairsTradingEnv(spec, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(ACTION_LONG)
    assert info["position"] == 1
    _, _, _, _, info = env.step(ACTION_SHORT)
    assert info["position"] == -1


def test_env_obs_finite() -> None:
    df = _toy_df(30)
    spec = make_env_spec(df, ["zscore", "realized_vol", "bid_ask_proxy"], CostModel())
    env = PairsTradingEnv(spec, seed=0)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
