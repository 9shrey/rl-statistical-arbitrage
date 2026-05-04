"""Walk-forward backtest engine.

For each fold:
  1. Slice the feature dataframe into train / test by integer index.
  2. Fit the regime model on train rows; predict regimes for test rows.
  3. Build EnvSpec for train and test (cost model from config).
  4. Construct or train the policy on train (baselines: fit thresholds; PPO: train).
  5. Roll the policy on test, collect equity / positions / actions, compute metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from arb.backtest.metrics import summary
from arb.backtest.runner import run_policy_full
from arb.backtest.splitter import walk_forward_splits
from arb.config.schema import AppConfig
from arb.env.costs import CostModel
from arb.env.gym_env import make_env_spec
from arb.policy.base import Policy
from arb.policy.baseline_grid import fit_grid_zscore
from arb.policy.baseline_zscore import ZScoreThresholdPolicy
from arb.policy.buy_hold import BuyHoldSpreadPolicy
from arb.regime.hmm import RegimeModel


@dataclass
class FoldResult:
    fold_id: int
    train_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    equity_curve: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    metrics: dict[str, float]


def _make_policy(name: str, cfg: AppConfig, train_spec) -> Policy:  # type: ignore[no-untyped-def]
    if name == "zscore_baseline":
        return ZScoreThresholdPolicy(entry=2.0, exit_z=0.5, stop=4.0)
    if name == "grid_baseline":
        return fit_grid_zscore(train_spec)
    if name == "buy_hold":
        return BuyHoldSpreadPolicy()
    if name == "ppo":
        from arb.policy.ppo import train_ppo

        return train_ppo(
            train_spec,
            total_timesteps=cfg.train.total_timesteps,
            seed=cfg.seed,
            learning_rate=cfg.train.ppo.learning_rate,
            n_steps=cfg.train.ppo.n_steps,
            batch_size=cfg.train.ppo.batch_size,
            gamma=cfg.train.ppo.gamma,
            gae_lambda=cfg.train.ppo.gae_lambda,
            clip_range=cfg.train.ppo.clip_range,
            ent_coef=cfg.train.ppo.ent_coef,
        )
    raise ValueError(f"Unknown policy/algo: {name}")


def _select_regime_features(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = [c for c in names if c in df.columns]
    if not cols:
        cols = ["abs_zscore"] if "abs_zscore" in df.columns else ["zscore"]
    return df[cols].fillna(0.0).to_numpy(dtype=float)


def run_walk_forward(features_df: pd.DataFrame, cfg: AppConfig) -> list[FoldResult]:
    """Run walk-forward backtest. ``features_df`` already has features + spread."""
    df = features_df.dropna().reset_index(drop=True)
    n = len(df)
    folds = walk_forward_splits(
        n=n,
        train_window=cfg.backtest.train_window_days,
        test_window=cfg.backtest.test_window_days,
        step=cfg.backtest.step_days,
    )

    feature_cols = ["zscore", "realized_vol", "bid_ask_proxy"]
    n_regime = cfg.regime.n_states
    feature_cols_with_regime = feature_cols + [f"regime_{i}" for i in range(n_regime)]
    cost = CostModel(
        commission_bps_per_side=cfg.costs.commission_bps_per_side,
        slippage_half_spread_bps=cfg.costs.slippage_half_spread_bps,
        impact_bps_per_pct_adv=cfg.costs.impact_bps_per_pct_adv,
    )

    results: list[FoldResult] = []
    for fold in folds:
        train_df = df.iloc[fold.train_start : fold.train_end].reset_index(drop=True)
        test_df = df.iloc[fold.test_start : fold.test_end].reset_index(drop=True)

        # Regime fit on train, predict on test (causal).
        regime_X_train = _select_regime_features(train_df, cfg.regime.features)
        regime_X_test = _select_regime_features(test_df, cfg.regime.features)
        rm = RegimeModel(n_states=n_regime, seed=cfg.seed).fit(regime_X_train)
        train_states = rm.predict_causal(regime_X_train)
        test_states = rm.predict_causal(regime_X_test)
        train_oh = rm.onehot(train_states)
        test_oh = rm.onehot(test_states)
        for i in range(n_regime):
            train_df[f"regime_{i}"] = train_oh[:, i]
            test_df[f"regime_{i}"] = test_oh[:, i]

        train_spec = make_env_spec(
            train_df,
            feature_cols_with_regime,
            cost,
            lambda_turnover=cfg.env.reward.lambda_turnover,
            dd_threshold=cfg.env.reward.dd_threshold,
            dd_penalty=cfg.env.reward.dd_penalty,
            clip=cfg.env.reward.clip,
            hard_dd_stop=cfg.risk.hard_dd_stop,
        )
        test_spec = make_env_spec(
            test_df,
            feature_cols_with_regime,
            cost,
            lambda_turnover=cfg.env.reward.lambda_turnover,
            dd_threshold=cfg.env.reward.dd_threshold,
            dd_penalty=cfg.env.reward.dd_penalty,
            clip=cfg.env.reward.clip,
            hard_dd_stop=cfg.risk.hard_dd_stop,
        )
        if test_spec.features.shape[0] < 2:
            continue

        policy = _make_policy(cfg.train.algo, cfg, train_spec)
        equity, positions, actions = run_policy_full(test_spec, policy, seed=cfg.seed)
        m = summary(equity, positions)
        results.append(
            FoldResult(
                fold_id=fold.fold_id,
                train_range=(_safe_ts(train_df, 0), _safe_ts(train_df, -1)),
                test_range=(_safe_ts(test_df, 0), _safe_ts(test_df, -1)),
                equity_curve=equity,
                positions=positions,
                actions=actions,
                metrics=m,
            )
        )
    return results


def _safe_ts(df: pd.DataFrame, idx: int) -> datetime:
    if "ts" in df.columns and len(df) > 0:
        ts = df["ts"].iloc[idx]
        return pd.Timestamp(ts).to_pydatetime()
    return datetime(1970, 1, 1)


def aggregate(results: list[FoldResult]) -> dict[str, float]:
    """Aggregate per-fold metrics (mean) and summary across folds."""
    if not results:
        return {"folds": 0}
    keys = results[0].metrics.keys()
    agg = {f"mean_{k}": float(np.mean([r.metrics[k] for r in results])) for k in keys}
    agg["folds"] = len(results)
    return agg
