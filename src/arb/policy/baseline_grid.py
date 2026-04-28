"""Grid search over z-score policy thresholds. Fit on train fold; freeze for test."""
from __future__ import annotations

from itertools import product

from arb.env.gym_env import EnvSpec
from arb.policy.baseline_zscore import ZScoreThresholdPolicy


def fit_grid_zscore(
    train_spec: EnvSpec,
    entry_grid: tuple[float, ...] = (1.5, 2.0, 2.5),
    exit_grid: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> ZScoreThresholdPolicy:
    # Local imports to break circular dependency (backtest -> policy -> backtest).
    from arb.backtest.metrics import sharpe
    from arb.backtest.runner import run_policy

    best: tuple[float, ZScoreThresholdPolicy] | None = None
    for e, x in product(entry_grid, exit_grid):
        if x >= e:
            continue
        pol = ZScoreThresholdPolicy(entry=e, exit_z=x)
        eq = run_policy(train_spec, pol)
        rets = (eq[1:] / eq[:-1]) - 1.0
        s = sharpe(rets)
        if best is None or s > best[0]:
            best = (s, pol)
    if best is None:
        return ZScoreThresholdPolicy()
    return best[1]
