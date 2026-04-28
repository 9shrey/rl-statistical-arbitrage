# Runbook

## First-time setup

```powershell
cd 13-rl-statistical-arbitrage
python -m pip install -e ".[dev]"
```

If you also want yfinance for live data:
```powershell
python -m pip install -e ".[dev,data]"
```

## Common workflows

| Task | Command |
|---|---|
| Run all tests | `pytest` |
| Run with coverage | `pytest --cov=arb --cov-report=term-missing` |
| Lint | `python -m ruff check src tests` |
| Smoke run on fixtures | `python -m arb.cli smoke --config configs/smoke.yaml` |
| Backtest with default config | `python -m arb.cli backtest --config configs/default.yaml` |
| Generate leaderboard | `python -m arb.cli report --config configs/default.yaml` |

## Troubleshooting

- **`hmmlearn` unavailable** — the regime model silently falls back to `sklearn.mixture.GaussianMixture`; results may differ slightly.
- **`stable-baselines3` / `torch` not installed** — only the PPO algo path requires them; baselines (`zscore_baseline`, `grid_baseline`, `buy_hold`) work without.
- **Slow tests** — use `pytest -m "not slow and not integration"` for the inner loop; CI runs the full suite.
