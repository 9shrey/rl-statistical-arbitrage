# RL Statistical Arbitrage

A regime-aware **reinforcement learning** agent for **statistical arbitrage** on cointegrated equity pairs. Replaces fixed-threshold mean-reversion rules with a learned policy that adapts entry/exit timing using market microstructure features and an HMM-estimated regime label. Backtested with strict **walk-forward validation**, point-in-time universes, and slippage-adjusted PnL.

> This project sits at the intersection of **quantitative finance**, **deep RL**, and **rigorous backtesting discipline** — built to be reviewed by quants and ML engineers alike.

## Highlights

- **Cointegration-driven pair selection** — Engle-Granger and Johansen tests, half-life filtering.
- **Rolling, look-ahead-free spreads & z-scores** — every statistic uses strictly trailing windows.
- **HMM market-regime feature** — Gaussian Hidden Markov Model labels states as trending vs mean-reverting.
- **Custom Gymnasium environment** — discrete action space `{LONG_SPREAD, SHORT_SPREAD, FLAT}`, realistic cost model (commission + half-spread slippage + linear market impact), drawdown-penalized reward.
- **PPO policy** (Stable-Baselines3) trained walk-forward; SAC variant for continuous actions.
- **Baselines** — static z-score thresholds, grid-search-optimal thresholds, buy-and-hold.
- **Bias controls** — point-in-time universe, no look-ahead, no survivorship bias, deterministic seeding, no network calls in tests.
- **Reproducibility** — MLflow-tracked runs, hashed equity curves, single-command smoke pipeline.

## Quick Start

```bash
make setup            # install package + dev deps
make test             # unit + property tests
make smoke            # end-to-end smoke pipeline on fixtures (~30s)
```

For a fuller research run with downloaded data:

```bash
make data             # fetch sample bars (yfinance)
make train   CONFIG=configs/fast.yaml
make backtest CONFIG=configs/fast.yaml
make report  CONFIG=configs/fast.yaml
```

## Architecture

| Component         | Stack                                   |
| ----------------- | --------------------------------------- |
| RL algorithm      | PPO, SAC (Stable-Baselines3)            |
| Pair selection    | Engle-Granger, Johansen (statsmodels)   |
| Regime model      | Gaussian HMM (hmmlearn)                 |
| Environment       | Custom Gymnasium env                    |
| Risk metrics      | Sharpe, Sortino, Calmar, Max Drawdown   |
| Validation        | Walk-forward, expanding window, no leak |
| Tracking          | MLflow (local file backend)             |

See [`docs/architecture.md`](docs/architecture.md), [`docs/methodology.md`](docs/methodology.md), and the [ADRs](docs/decisions/).

## Bias & Correctness

This project takes backtest correctness seriously. See [`BIAS_CHECKLIST.md`](BIAS_CHECKLIST.md):

- **Look-ahead bias** — property-tested via shift invariance.
- **Survivorship bias** — point-in-time universe snapshots.
- **Train/test leakage** — all preprocessing (hedge ratio, z-score stats, HMM, observation normalization) fit on train fold only.
- **Determinism** — every randomness source is seeded; CI hashes the equity curve.
- **No network in tests** — fixtures committed under `tests/fixtures/`.

## Repository Layout

See [`MASTER_PROMPT.md`](MASTER_PROMPT.md) for the canonical project spec, locked tech stack, public interfaces, and phased execution plan.

## ATS Keywords

Statistical Arbitrage · Pairs Trading · Cointegration · Engle-Granger · Johansen Test · Reinforcement Learning · PPO · SAC · Stable Baselines3 · Gymnasium · OpenAI Gym · Mean Reversion · Market Microstructure · HMM · Hidden Markov Model · Market Regime · Backtesting · Walk-Forward Validation · Sharpe Ratio · Sortino Ratio · Calmar Ratio · Drawdown · PnL · Transaction Costs · Slippage · Market Impact · Alpha Generation · High Frequency · Algorithmic Trading · Quantitative Finance · Point-in-Time Universe · Survivorship Bias · Look-Ahead Bias

## License

MIT — see [LICENSE](LICENSE).
