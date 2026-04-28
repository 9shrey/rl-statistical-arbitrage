"""Pairs trading environment.

Implements Gymnasium's API when available; otherwise exposes the same
``reset/step`` surface as a plain class so the project remains usable
without the heavy RL stack installed.

Observation:  [zscore, realized_vol, bid_ask_proxy, regime_onehot..., position]
Action space: Discrete(3) -> {0: FLAT, 1: LONG_SPREAD, 2: SHORT_SPREAD}
Reward:       see :func:`arb.env.reward.compute_reward`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from arb.env.costs import CostModel
from arb.env.portfolio import Portfolio
from arb.env.reward import compute_reward

try:  # pragma: no cover - optional
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False


ACTION_FLAT = 0
ACTION_LONG = 1
ACTION_SHORT = 2
_ACTION_TO_POS = {ACTION_FLAT: 0, ACTION_LONG: 1, ACTION_SHORT: -1}


@dataclass
class EnvSpec:
    features: np.ndarray  # (T, F) finite, no NaN
    spread_returns: np.ndarray  # (T,) bar-over-bar spread % return for a +1 position
    bar_volume_pct_adv: np.ndarray  # (T,) traded notional fraction proxy used for impact
    cost: CostModel = field(default_factory=CostModel)
    lambda_turnover: float = 0.0005
    dd_threshold: float = 0.10
    dd_penalty: float = 0.5
    clip: float = 0.05
    hard_dd_stop: float = 0.5


def make_env_spec(
    df: pd.DataFrame,
    feature_cols: list[str],
    cost: CostModel,
    *,
    lambda_turnover: float = 0.0005,
    dd_threshold: float = 0.10,
    dd_penalty: float = 0.5,
    clip: float = 0.05,
    hard_dd_stop: float = 0.5,
) -> EnvSpec:
    """Materialize an EnvSpec from a feature DataFrame.

    The DataFrame must already include feature columns and a 'spread' column.
    Rows with NaN in any required column are dropped to ensure finiteness.
    """
    needed = list(feature_cols) + ["spread", "price_a", "price_b", "hedge_ratio"]
    work = df.dropna(subset=needed).reset_index(drop=True).copy()
    # Per-bar % return on a +1 spread position: change in spread / |gross_notional|
    # Approximate gross notional with |price_a| + |hedge_ratio * price_b|
    gross = (work["price_a"].abs() + (work["hedge_ratio"].abs() * work["price_b"].abs())).replace(0.0, np.nan)
    spread_ret = work["spread"].diff().fillna(0.0) / gross
    spread_ret = spread_ret.fillna(0.0).to_numpy(dtype=np.float64)
    # Volume as fraction of ADV proxy: assume each trade is a fixed small fraction.
    adv = np.full(len(work), 0.001, dtype=np.float64)  # 0.1% of ADV per trade
    feats = work[feature_cols].to_numpy(dtype=np.float32)
    return EnvSpec(
        features=feats,
        spread_returns=spread_ret,
        bar_volume_pct_adv=adv,
        cost=cost,
        lambda_turnover=lambda_turnover,
        dd_threshold=dd_threshold,
        dd_penalty=dd_penalty,
        clip=clip,
        hard_dd_stop=hard_dd_stop,
    )


class PairsTradingEnv:
    """Plain Python environment with a Gymnasium-compatible surface."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec = spec
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self.portfolio = Portfolio()
        self._n = len(spec.features)
        self._n_features = spec.features.shape[1] + 1  # +1 for current position

        if _HAS_GYM:  # pragma: no cover
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._n_features,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.portfolio = Portfolio()
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t >= self._n:
            raise RuntimeError("step called past end of episode")

        new_pos = _ACTION_TO_POS[int(action)]
        old_pos = self.portfolio.position
        delta_position = new_pos - old_pos

        # PnL on existing position over this bar.
        bar_ret = float(self.spec.spread_returns[self._t]) * old_pos
        # Equity multiplicative update.
        new_equity = self.portfolio.equity * (1.0 + bar_ret)

        # Trading cost for the size change (in bps of notional traded).
        adv_pct = float(self.spec.bar_volume_pct_adv[self._t])
        cost_bps = self.spec.cost.trade_cost_bps(adv_pct) if delta_position != 0 else 0.0
        cost_pct = (cost_bps / 1e4) * abs(delta_position)
        new_equity *= 1.0 - cost_pct
        self.portfolio.update_equity(new_equity)
        self.portfolio.position = new_pos

        reward = compute_reward(
            pnl_pct=bar_ret,
            delta_position=delta_position,
            cost_bps=cost_bps,
            drawdown=self.portfolio.drawdown(),
            lambda_turnover=self.spec.lambda_turnover,
            dd_threshold=self.spec.dd_threshold,
            dd_penalty=self.spec.dd_penalty,
            clip=self.spec.clip,
        )

        self._t += 1
        terminated = self._t >= self._n
        truncated = self.portfolio.drawdown() >= self.spec.hard_dd_stop
        info = {
            "equity": self.portfolio.equity,
            "drawdown": self.portfolio.drawdown(),
            "position": self.portfolio.position,
            "bar_ret": bar_ret,
            "cost_bps": cost_bps,
        }
        if terminated or truncated:
            return self._obs_terminal(), reward, terminated, truncated, info
        return self._obs(), reward, terminated, truncated, info

    # ---- helpers ----
    def _obs(self) -> np.ndarray:
        feats = self.spec.features[self._t]
        return np.concatenate([feats, np.array([self.portfolio.position], dtype=np.float32)]).astype(np.float32)

    def _obs_terminal(self) -> np.ndarray:
        # Use last valid features as terminal observation.
        idx = min(self._t, self._n - 1)
        feats = self.spec.features[idx]
        return np.concatenate([feats, np.array([self.portfolio.position], dtype=np.float32)]).astype(np.float32)
