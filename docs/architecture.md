# Architecture

```
              ┌──────────────┐
              │   configs/   │  pydantic-validated YAML
              └──────┬───────┘
                     │
   ┌─────────────────▼──────────────────┐
   │  arb.data.ingest  (fixture/csv/yf) │  point-in-time, calendar-aware
   └─────────────────┬──────────────────┘
                     ▼
       arb.signals.features
        ├─ rolling_hedge_ratio  (no look-ahead)
        ├─ make_spread
        ├─ rolling_zscore       (no look-ahead)
        └─ realized_vol, bid_ask_proxy
                     │
                     ▼
       arb.regime.RegimeModel   (HMM / GMM fallback)
        - fit on TRAIN fold only
        - predict_causal on TEST
                     │
                     ▼
       arb.env.PairsTradingEnv  (Gymnasium-compatible)
        - obs: [zscore, vol, spread_proxy, regime_oh..., position]
        - act: {FLAT, LONG_SPREAD, SHORT_SPREAD}
        - reward: clipped PnL − costs − turnover − dd_excess
                     │
        ┌────────────┴────────────────────┐
        │                                 │
        ▼                                 ▼
 arb.policy.baseline_*           arb.policy.ppo (SB3)
                     │
                     ▼
       arb.backtest.engine.run_walk_forward
        - expanding train, fixed test
        - per-fold metrics + aggregation
                     │
                     ▼
              arb.report.* (leaderboard, plots)
```

## Module map

| Module | Responsibility |
|---|---|
| `arb.config` | YAML loader + pydantic schemas |
| `arb.data` | Bar ingestion, alignment, calendar, PIT universe |
| `arb.pairs` | Engle-Granger, Johansen, half-life, ranking |
| `arb.signals` | Hedge ratio, spread, z-score, microstructure features |
| `arb.regime` | Gaussian HMM (or GaussianMixture fallback) |
| `arb.env` | Custom Gymnasium env, cost model, reward, portfolio |
| `arb.policy` | Z-score baseline, grid baseline, buy-and-hold, PPO adapter |
| `arb.backtest` | Walk-forward splitter, runner, metrics, engine |
| `arb.risk` | Position sizing helpers |
| `arb.report` | Leaderboards and plots |
| `arb.cli` | Typer entrypoint (`arb`) |

## Key invariants

- **No look-ahead.** Every rolling stat uses `shift(1)` before being applied.
- **Train-only preprocessing.** Hedge ratio, z-score windows, regime model, normalization stats — all fit on the train fold.
- **Determinism.** Single seed propagated through numpy / random / torch / SB3.
- **Bounded reward.** Clipped to `[-clip, +clip]` for training stability.
- **No network in tests.** Synthetic cointegrated fixtures live under `tests/fixtures/`.
