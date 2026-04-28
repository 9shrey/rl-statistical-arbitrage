"""Pydantic config schemas (typed, validated)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    source: Literal["yfinance", "csv", "fixture"] = "fixture"
    bar_interval: str = "1d"
    start: date
    end: date
    symbols: list[str]
    cache_dir: str = "data/raw"

    @field_validator("symbols")
    @classmethod
    def _two_or_more(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Need at least two symbols.")
        return [s.upper() for s in v]


class PairsConfig(BaseModel):
    corr_threshold: float = 0.8
    corr_window_days: int = 252
    pvalue_max: float = 0.05
    half_life_min_days: int = 1
    half_life_max_days: int = 30
    top_k: int = 10


class SpreadConfig(BaseModel):
    hedge_window_days: int = Field(60, ge=5)
    zscore_window_days: int = Field(20, ge=2)


class RegimeConfig(BaseModel):
    n_states: int = Field(2, ge=2, le=4)
    features: list[str] = ["realized_vol", "abs_zscore"]
    refit_each_fold: bool = True


class RewardConfig(BaseModel):
    lambda_turnover: float = 0.0005
    dd_threshold: float = 0.10
    dd_penalty: float = 0.5
    clip: float = 0.05


class EnvConfig(BaseModel):
    action_space: Literal["discrete", "continuous"] = "discrete"
    features: list[str] = ["zscore", "realized_vol", "bid_ask_proxy", "regime_onehot", "position"]
    reward: RewardConfig = Field(default_factory=RewardConfig)


class CostsConfig(BaseModel):
    commission_bps_per_side: float = 0.5
    slippage_half_spread_bps: float = 1.0
    impact_bps_per_pct_adv: float = 10.0


class RiskConfig(BaseModel):
    vol_target_annual: float = 0.10
    max_position_notional_pct_equity: float = 0.50
    hard_dd_stop: float = 0.20


class PPOHyperparams(BaseModel):
    learning_rate: float = 3.0e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0


class TrainConfig(BaseModel):
    algo: Literal["zscore_baseline", "grid_baseline", "buy_hold", "ppo", "sac"] = "zscore_baseline"
    total_timesteps: int = 200_000
    n_envs: int = 1
    ppo: PPOHyperparams = Field(default_factory=PPOHyperparams)


class BacktestConfig(BaseModel):
    scheme: Literal["walk_forward"] = "walk_forward"
    train_window_days: int = 504
    test_window_days: int = 126
    step_days: int = 126
    warm_start: bool = False


class MlflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment: str = "rl_stat_arb"


class AppConfig(BaseModel):
    seed: int = 42
    data: DataConfig
    pairs: PairsConfig = Field(default_factory=PairsConfig)
    spread: SpreadConfig = Field(default_factory=SpreadConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    artifacts_dir: str = "artifacts"
