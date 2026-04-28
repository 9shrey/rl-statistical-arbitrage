# Bias & Correctness Checklist

Every item below is verified by a test or pinned to a code reference. PR reviewers MUST confirm.

| # | Bias / Pitfall | Control | Evidence |
|---|---|---|---|
| 1 | **Look-ahead bias** | All rolling stats use strictly trailing windows (`shift(1)` after `rolling().mean()/std()`). | [`tests/property/test_no_lookahead.py`](tests/property/test_no_lookahead.py) — Hypothesis shift-invariance test. |
| 2 | **Survivorship bias** | Universe is loaded as point-in-time snapshots; delisted symbols included where data permits. | [`src/arb/data/universe.py`](src/arb/data/universe.py); fixtures include a delisted ticker. |
| 3 | **Data snooping** | Hyperparameters frozen in `configs/*.yaml` before final walk-forward. Smoke config NEVER influences final reported metrics. | `configs/full.yaml` is committed and immutable in `mlruns/<id>/artifacts/config.yaml`. |
| 4 | **Train/test leakage (preprocessing)** | Hedge ratio, z-score stats, HMM, and `VecNormalize` are fit on the train fold only and applied to test. | [`src/arb/backtest/engine.py`](src/arb/backtest/engine.py) `run_walk_forward`. |
| 5 | **Regime-label leakage** | HMM uses forward filtering on test; no Viterbi smoothing across the train/test boundary. | [`src/arb/regime/hmm.py`](src/arb/regime/hmm.py) `predict_causal`. |
| 6 | **Cost realism** | Commission, half-spread slippage, and linear market impact applied on every trade. Defaults documented. | [`src/arb/env/costs.py`](src/arb/env/costs.py); ADR-0005. |
| 7 | **Reward hacking** | Reward is bounded (`clip`), turnover-penalized, drawdown-penalized; pure churn yields negative reward in unit test. | [`tests/unit/test_reward.py`](tests/unit/test_reward.py). |
| 8 | **Determinism** | numpy, torch, SB3, env, splitter all seeded. CI asserts equity curve hash is stable across two runs. | [`src/arb/utils/seeding.py`](src/arb/utils/seeding.py); [`tests/integration/test_smoke.py`](tests/integration/test_smoke.py). |
| 9 | **No network in tests** | All tests use committed fixtures under `tests/fixtures/`. yfinance only used by `arb data fetch`, never by tests. | `pyproject.toml` test config; CI runs offline. |
| 10 | **Time zones** | All timestamps are tz-aware UTC; conversion to `America/New_York` only at display layer. | [`src/arb/data/calendar.py`](src/arb/data/calendar.py); schema enforces `datetime` with tz. |

## Sign-off

- [ ] Reviewer 1: ___
- [ ] Reviewer 2: ___
- [ ] Date: ___
