# MASTER PROMPT — HF Statistical Arbitrage via Reinforcement Learning

> Use this document as the single source of truth for an autonomous coding agent (or a human contributor) building this project end-to-end. It encodes scope, architecture, interfaces, acceptance criteria, and a phased execution plan. Do not deviate from the contracts in §6 without an ADR.

---

## 1. Project Identity

- **Name:** `rl-statistical-arbitrage`
- **One-liner:** A regime-aware reinforcement learning agent that trades cointegrated equity pairs, replacing fixed-threshold mean-reversion rules with a learned, microstructure-conditioned policy — backtested with strict walk-forward validation and slippage-adjusted PnL.
- **Domain:** Quantitative Finance · Deep Reinforcement Learning · Market Microstructure
- **Primary persona:** Quant researcher / ML engineer interviewing for HF, prop, or systematic trading roles.
- **Why it exists:** Demonstrate the intersection of (a) classical econometrics (cointegration, HMM regimes), (b) modern deep RL (PPO/SAC), and (c) production-grade backtesting discipline (no look-ahead, no survivorship bias, realistic costs).

---

## 2. Outcomes & Success Criteria

A reviewer cloning the repo must be able to:

1. `make setup && make data && make train && make backtest && make report` — end-to-end in one command chain on a laptop in < 30 minutes for the smoke config.
2. Reproduce a leaderboard comparing the **RL policy** vs three baselines:
   - Static z-score threshold (entry ±2σ, exit 0).
   - Optimal-threshold grid search (in-sample fit, out-of-sample test).
   - Buy-and-hold of each leg.
3. See per-fold walk-forward metrics: **Sharpe, Sortino, Calmar, Max Drawdown, Hit Rate, Avg Holding Period, Turnover, Slippage-Adjusted Return**.
4. Inspect a regime-conditioned action heatmap (z-score × regime → action probability).
5. Run a single CLI to evaluate a trained policy on any user-supplied pair: `arb eval --pair AAPL,MSFT --start 2022-01-01 --end 2023-12-31`.

**Definition of Done (hard gates):**
- ≥ 90% unit test coverage on `env/`, `data/`, `signals/`, `risk/`.
- Property tests: no look-ahead leakage, no negative position sizing where disallowed, reward function is bounded for bounded inputs.
- Determinism: same seed → bit-identical backtest output.
- A `BIAS_CHECKLIST.md` audited and signed off.
- ADRs for every irreversible design choice (see §10).

---

## 3. Non-Goals

- Live trading, broker integration, or order routing.
- Tick-by-tick LOB reconstruction (start with 1-minute bars; design for tick-level extension).
- Options, futures, crypto. Equities only in v1.
- Multi-asset portfolio optimization across pairs (single-pair agent in v1; portfolio wrapper is v2).
- Replacing PPO with custom RL research — use Stable-Baselines3 as the trusted implementation.

---

## 4. Tech Stack (locked)

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | SB3 + scientific stack maturity |
| Package manager | `uv` | Fast, reproducible, lockfile-first |
| RL framework | Stable-Baselines3 (PPO primary, SAC secondary) | Battle-tested, well-documented |
| Env API | Gymnasium (modern Gym fork) | SB3-compatible, actively maintained |
| Econometrics | `statsmodels` (Engle-Granger, Johansen, ADF) | Standard reference implementations |
| Regime model | `hmmlearn` (Gaussian HMM) | Simple, interpretable, well-tested |
| Data handling | `polars` (primary), `pandas` (interop) | Speed + correctness on large bar data |
| Numerics | `numpy`, `scipy` | — |
| Backtest engine | Custom (thin, in-process, vectorized where possible) | Full control over fill model, costs, no-lookahead guarantees |
| Experiment tracking | `mlflow` (local file backend) | Zero-infra, portable artifacts |
| Config | `pydantic` + YAML | Typed, validated, diffable |
| CLI | `typer` | Ergonomic, auto-help |
| Plotting | `matplotlib` + `plotly` (HTML reports) | Static + interactive |
| Testing | `pytest`, `hypothesis`, `pytest-cov` | Unit + property + coverage |
| Lint/format | `ruff`, `black`, `mypy --strict` on `src/` | Keep agents from drifting |
| CI | GitHub Actions | Standard |
| Container | Dockerfile + `docker-compose` for optional jupyter | Reproducible eval |

**Forbidden:** TensorFlow, Keras, custom RL loops, pandas-only data pipeline, hardcoded paths, network calls during tests.

---

## 5. Repository Layout

```
13-rl-statistical-arbitrage/
├── README.md
├── MASTER_PROMPT.md              # this file
├── LICENSE                       # MIT
├── Makefile                      # setup | data | train | backtest | report | test | lint
├── pyproject.toml                # uv-managed, pinned
├── uv.lock
├── .python-version               # 3.11
├── .gitignore
├── BIAS_CHECKLIST.md             # see §11
│
├── configs/
│   ├── default.yaml              # full config, all knobs
│   ├── smoke.yaml                # 1 pair, 1 fold, ~30s train (CI gate)
│   ├── fast.yaml                 # 5 pairs, 3 folds, laptop-runnable
│   └── full.yaml                 # research-grade, GPU-friendly
│
├── data/
│   ├── raw/                      # immutable downloaded bars (gitignored)
│   ├── interim/                  # cleaned, aligned, PIT-filtered
│   ├── processed/                # spread series, features, regime labels
│   ├── universe/                 # point-in-time universe snapshots
│   └── README.md                 # provenance, schema, refresh cadence
│
├── src/arb/
│   ├── __init__.py
│   ├── cli.py                    # typer entrypoint: `arb ...`
│   ├── config/
│   │   ├── schema.py             # pydantic models
│   │   └── loader.py
│   ├── data/
│   │   ├── ingest.py             # adapters: yfinance | csv | parquet | (stub: polygon, alpaca)
│   │   ├── universe.py           # point-in-time universe construction
│   │   ├── calendar.py           # NYSE trading calendar
│   │   └── schemas.py            # OHLCV, Spread, Feature row schemas
│   ├── pairs/
│   │   ├── selection.py          # correlation pre-filter
│   │   ├── engle_granger.py      # 2-step EG cointegration test
│   │   ├── johansen.py           # multivariate cointegration
│   │   └── ranking.py            # half-life, hedge ratio, score
│   ├── signals/
│   │   ├── spread.py             # rolling hedge ratio, residual spread
│   │   ├── zscore.py             # rolling z-score (no look-ahead)
│   │   ├── microstructure.py     # bid-ask proxy, realized vol, volume features
│   │   └── features.py           # feature assembly into env state
│   ├── regime/
│   │   ├── hmm.py                # Gaussian HMM fit + predict
│   │   └── labels.py             # decode states → trending | mean-reverting
│   ├── env/
│   │   ├── gym_env.py            # PairsTradingEnv (Gymnasium API)
│   │   ├── action_space.py       # discrete: {LONG_SPREAD, SHORT_SPREAD, FLAT}
│   │   ├── reward.py             # risk-adjusted PnL minus costs
│   │   ├── costs.py              # commission, half-spread slippage, impact
│   │   └── portfolio.py          # position state, cash, equity curve
│   ├── policy/
│   │   ├── ppo.py                # PPO trainer (SB3 wrapper)
│   │   ├── sac.py                # SAC trainer (continuous-action variant)
│   │   ├── baseline_zscore.py    # static threshold policy
│   │   └── baseline_grid.py      # grid-search optimal thresholds
│   ├── backtest/
│   │   ├── engine.py             # walk-forward orchestrator
│   │   ├── splitter.py           # expanding / rolling window splits
│   │   ├── runner.py             # executes a policy over a fold
│   │   └── metrics.py            # Sharpe, Sortino, Calmar, MDD, etc.
│   ├── risk/
│   │   ├── sizing.py             # vol-targeted position sizing
│   │   └── limits.py             # max position, max DD circuit-breaker
│   ├── report/
│   │   ├── leaderboard.py        # markdown + html
│   │   ├── plots.py              # equity curve, regime overlay, action heatmap
│   │   └── html.py               # standalone HTML report
│   └── utils/
│       ├── seeding.py            # deterministic seeds across np/torch/sb3/env
│       ├── logging.py            # structlog
│       └── io.py
│
├── scripts/
│   ├── download_sample.py        # fetch a small public sample (yfinance)
│   ├── make_universe.py
│   └── smoke.sh                  # end-to-end smoke (used in CI)
│
├── notebooks/
│   ├── 01_cointegration_eda.ipynb
│   ├── 02_regime_inspection.ipynb
│   └── 03_policy_diagnostics.ipynb
│
├── tests/
│   ├── unit/                     # one file per src module
│   ├── property/                 # hypothesis: no-lookahead, reward bounds, etc.
│   ├── integration/              # end-to-end: smoke config completes
│   └── fixtures/                 # tiny deterministic CSVs
│
├── docs/
│   ├── architecture.md
│   ├── runbook.md
│   ├── methodology.md            # cointegration → regime → RL pipeline, with citations
│   ├── interview_talking_points.md
│   └── decisions/                # ADRs
│       ├── 0001-rl-algorithm-ppo.md
│       ├── 0002-discrete-vs-continuous-actions.md
│       ├── 0003-walk-forward-vs-kfold.md
│       ├── 0004-regime-model-hmm.md
│       ├── 0005-cost-model.md
│       └── README.md
│
└── .github/workflows/
    ├── ci.yml                    # lint + type + unit + smoke
    └── docs.yml
```

---

## 6. Public Interfaces (contracts)

These signatures are **stable**. Implementations may evolve; signatures may not without an ADR.

### 6.1 Data schemas (pydantic / polars)

```python
class Bar(BaseModel):
    symbol: str
    ts: datetime          # tz-aware UTC, bar CLOSE time
    open: float
    high: float
    low: float
    close: float
    volume: int

class SpreadRow(BaseModel):
    ts: datetime
    sym_a: str
    sym_b: str
    price_a: float
    price_b: float
    hedge_ratio: float    # rolling, computed using ONLY data <= ts
    spread: float         # price_a - hedge_ratio * price_b
    zscore: float         # rolling, no look-ahead
    regime: int           # HMM state id, predicted using ONLY data <= ts
```

### 6.2 Environment

```python
class PairsTradingEnv(gymnasium.Env):
    """
    Observation: Box(low=-inf, high=inf, shape=(N_FEATURES,), dtype=float32)
        [zscore, realized_vol, bid_ask_proxy, regime_onehot..., position]
    Action: Discrete(3)  # 0=FLAT, 1=LONG_SPREAD, 2=SHORT_SPREAD
    Reward: float         # delta_equity / equity_0  -  costs  -  lambda * |delta_position|
    Episode: one walk-forward fold (train or eval)
    """
    metadata = {"render_modes": ["human", "none"]}
```

### 6.3 Policies

```python
class Policy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Policy": ...
```

### 6.4 Backtest

```python
@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    equity_curve: pl.DataFrame      # ts, equity, position, action
    metrics: dict[str, float]       # sharpe, sortino, calmar, mdd, hit_rate, turnover, ...

def run_walk_forward(
    pair: tuple[str, str],
    policy_factory: Callable[[], Policy],
    cfg: BacktestConfig,
) -> list[FoldResult]: ...
```

### 6.5 CLI

```
arb data fetch     --config configs/fast.yaml
arb pairs find     --universe sp100 --window 2018-01-01:2020-12-31
arb regime fit     --pair AAPL,MSFT
arb train          --pair AAPL,MSFT --algo ppo --config configs/fast.yaml
arb backtest       --pair AAPL,MSFT --policy artifacts/ppo_aapl_msft.zip
arb eval           --pair AAPL,MSFT --start 2022-01-01 --end 2023-12-31
arb report         --run-id <mlflow_run_id>
```

---

## 7. Algorithmic Specification

### 7.1 Pair selection
1. Pre-filter universe by sector + rolling correlation (|ρ| ≥ 0.8 over 252 days).
2. For each candidate pair, run **Engle-Granger** 2-step on the training window only.
3. Keep pairs with cointegration p-value ≤ 0.05 **and** half-life of mean reversion in [1, 30] days.
4. Optional: **Johansen** test on baskets of 3+.
5. Rank by composite score: `0.4 * (1 - p_value) + 0.3 * stability_of_hedge_ratio + 0.3 * 1/half_life`.

### 7.2 Spread construction (no look-ahead)
- Hedge ratio β_t fit by OLS on a **trailing** window of length W (e.g., 60 days), using bars strictly before t.
- Spread_t = price_a_t − β_t · price_b_t.
- Z-score uses trailing mean/std over window Z (e.g., 20 days), bars strictly before t.

### 7.3 Regime model
- Fit Gaussian HMM with K=2 states on (rolling realized vol, |zscore|, autocorrelation of returns).
- Refit only on training windows; **predict** (forward filter) on test windows — never refit on test data.
- Map states by their unconditional mean of |zscore|: lower → "mean-reverting", higher → "trending".

### 7.4 Reward
```
r_t = (equity_t - equity_{t-1}) / equity_{t-1}
      - commission_per_trade * 1[action_t != action_{t-1}]
      - half_spread * |delta_position_t| * notional_t
      - lambda_turnover * |delta_position_t|
      - penalty_drawdown * max(0, current_dd - dd_threshold)
```
Reward is clipped to `[-R_MAX, +R_MAX]` for training stability; raw PnL is preserved separately for metrics.

### 7.5 Walk-forward validation
- **Expanding** train window, fixed test window (default: 24 months train / 6 months test, step 6 months).
- Train policy from scratch on each fold (default) OR warm-start from previous fold (configurable, ADR required).
- All preprocessing fit on train fold only; transforms applied to test fold.
- Splitter emits `(train_idx, test_idx)` index arrays — never timestamps that could be mutated.

### 7.6 Cost model
- Commission: 0.5 bps per side (configurable).
- Slippage: half of trailing 20-bar bid-ask proxy (high-low range fallback if no quotes).
- Market impact: linear in `|order_notional| / ADV`, coefficient configurable; default 10 bps per 1% of ADV.

---

## 8. Phased Execution Plan

Each phase ends with: tests green, lint green, docs updated, ADRs filed, `make smoke` passes.

### Phase 0 — Scaffolding (foundation)
- `pyproject.toml`, `uv.lock`, `.python-version`, Makefile, ruff/black/mypy config, pre-commit.
- Empty package `src/arb/` importable.
- CI runs lint + `pytest -k "not slow"`.
- Sample dataset committed under `tests/fixtures/` (≤ 200 KB total).

### Phase 1 — Data layer
- `data/ingest.py` with `yfinance` adapter + CSV/Parquet loader.
- `data/calendar.py` (pandas-market-calendars or hand-rolled NYSE).
- Schemas + validation (pydantic + polars dtype enforcement).
- Universe snapshot loader (PIT).
- Tests: schema invariants, calendar correctness, no NaN propagation.

### Phase 2 — Pair selection & spread
- Engle-Granger and Johansen implementations (wrap statsmodels).
- Rolling hedge ratio, rolling z-score (strictly trailing).
- Property test: shifting input by k bars shifts output by exactly k (no look-ahead).
- Notebook `01_cointegration_eda.ipynb`.

### Phase 3 — Regime model
- Gaussian HMM fit/predict.
- Train-only fit, forward-filter predict.
- Diagnostic plots: state probabilities over time vs realized vol.
- Tests: deterministic given seed, state count = K, no test-set leakage.

### Phase 4 — Gym environment
- `PairsTradingEnv` with discrete actions, full observation vector.
- Cost model wired in.
- Tests:
  - `gymnasium.utils.env_checker.check_env` passes.
  - Reward is finite for all valid actions on fixture data.
  - Episode terminates exactly at end of fold.
  - No future bars are accessible from `step()`.

### Phase 5 — Baselines
- Static z-score policy (entry ±2, exit 0, stop ±4).
- Grid-search policy (entry ∈ {1.5, 2, 2.5}, exit ∈ {0, 0.5}; fit on train, frozen on test).
- Buy-and-hold reference.
- Backtest harness produces `FoldResult` for each.

### Phase 6 — RL training
- PPO trainer (SB3) with `VecNormalize` for observations (stats fit on train only).
- Hyperparams in YAML, logged to MLflow.
- Early stopping on validation Sharpe.
- Determinism test: same seed → identical policy weights hash.

### Phase 7 — Walk-forward orchestrator
- Splitter (expanding window).
- Per-fold: fit hedge ratio + zscore stats + HMM + VecNormalize stats + train PPO + evaluate.
- Aggregated metrics across folds with confidence intervals (block bootstrap).

### Phase 8 — Reporting
- Markdown leaderboard (`benchmarks/leaderboard.md`).
- HTML report with: equity curves, drawdown, regime overlay, action heatmap, per-fold metrics table, cost decomposition.
- `arb report --run-id ...` regenerates from MLflow artifacts.

### Phase 9 — SAC variant & continuous actions (stretch)
- Continuous action ∈ [-1, +1] interpreted as target spread position.
- ADR comparing discrete vs continuous results.

### Phase 10 — Hardening
- Property tests for no-lookahead, no-survivorship, deterministic backtest.
- Performance: full backtest on 5 pairs × 8 folds in < 10 min on laptop CPU.
- Docs complete, all ADRs filed, BIAS_CHECKLIST.md signed off.

---

## 9. Configuration Surface (`configs/default.yaml`)

```yaml
seed: 42

data:
  source: yfinance         # yfinance | csv | parquet
  bar_interval: 1d         # 1d | 1h | 15m | 1m
  start: 2015-01-01
  end:   2024-12-31
  universe: sp100_pit      # path or named PIT universe
  cache_dir: data/raw

pairs:
  corr_threshold: 0.80
  corr_window_days: 252
  pvalue_max: 0.05
  half_life_min_days: 1
  half_life_max_days: 30
  top_k: 10

spread:
  hedge_window_days: 60
  zscore_window_days: 20

regime:
  n_states: 2
  features: [realized_vol, abs_zscore, return_autocorr]
  refit_each_fold: true

env:
  action_space: discrete   # discrete | continuous
  features: [zscore, realized_vol, bid_ask_proxy, regime_onehot, position]
  episode_length: full_fold
  reward:
    lambda_turnover: 0.0005
    dd_threshold: 0.10
    dd_penalty: 0.5
    clip: 0.05

costs:
  commission_bps_per_side: 0.5
  slippage_half_spread_bps: 1.0
  impact_bps_per_pct_adv: 10.0

risk:
  vol_target_annual: 0.10
  max_position_notional_pct_equity: 0.50
  hard_dd_stop: 0.20

train:
  algo: ppo               # ppo | sac
  total_timesteps: 200000
  n_envs: 4
  ppo:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.0

backtest:
  scheme: walk_forward
  train_window_months: 24
  test_window_months: 6
  step_months: 6
  warm_start: false

mlflow:
  tracking_uri: file:./mlruns
  experiment: rl_stat_arb
```

`smoke.yaml` overrides: 1 pair, 1 fold, `total_timesteps: 5000`, ~30s wall-clock.

---

## 10. Architecture Decision Records (ADRs)

Required ADRs (each ≤ 1 page, Context / Decision / Consequences / Alternatives):

1. **0001 — RL algorithm = PPO** (vs SAC, DQN, TD3). Discrete action stability + SB3 maturity.
2. **0002 — Discrete action space in v1**. Easier credit assignment; continuous deferred.
3. **0003 — Walk-forward (expanding) over k-fold**. Time-series correctness.
4. **0004 — HMM for regime detection** (vs change-point, threshold rules). Probabilistic, established.
5. **0005 — Cost model parameters and sources**. Defensible defaults with citations.
6. **0006 — Gymnasium over legacy Gym**. Maintained, SB3-supported.
7. **0007 — Polars primary, pandas interop**. Speed + correctness.
8. **0008 — Single-pair agent in v1**. Portfolio composition is a separable concern.
9. **0009 — `uv` as package manager**. Lockfile, speed, reproducibility.

---

## 11. Bias & Correctness Checklist (`BIAS_CHECKLIST.md`)

The agent MUST produce this file and verify each item with a linked test or code reference.

- [ ] **Look-ahead bias:** every rolling statistic uses strictly trailing windows. Property test shifts input and confirms output shifts identically.
- [ ] **Survivorship bias:** universe is point-in-time; delisted symbols included where data permits. Documented gaps.
- [ ] **Data snooping:** hyperparameters fixed before final walk-forward; only the smoke config touches test data during development (and is excluded from final reporting).
- [ ] **Train/test leakage:** `VecNormalize` stats, hedge ratio, z-score windows, HMM parameters all fit on train fold only.
- [ ] **Regime label leakage:** HMM forward-filtered on test (no Viterbi smoothing across the train/test boundary).
- [ ] **Cost realism:** commission, slippage, and impact applied on every trade; documented sources.
- [ ] **Reward hacking:** reward bounded; turnover penalized; pure churn produces negative reward in unit test.
- [ ] **Determinism:** seeded numpy, torch, SB3, env, splitter; CI asserts hash of equity curve is stable.
- [ ] **No network calls in tests:** all tests run with `--disable-network` (use fixtures).
- [ ] **Time zones:** all timestamps tz-aware UTC; conversion to `America/New_York` only at display.

---

## 12. Testing Strategy

- **Unit (`tests/unit/`):** one test file per `src/arb/**/*.py`; ≥ 90% coverage on core modules.
- **Property (`tests/property/`, `hypothesis`):**
  - Rolling z-score has no look-ahead (shift invariance).
  - Reward is finite and within `[-R_MAX, R_MAX]` for any valid `(obs, action)` pair from fixtures.
  - Walk-forward splitter produces non-overlapping train/test in the time dimension.
  - HMM `predict` on prefix equals `predict` on full series restricted to prefix (causal).
- **Integration (`tests/integration/`):**
  - `make smoke` runs end-to-end on fixtures in < 60s.
  - Trained policy's evaluation is deterministic across two runs with same seed.
- **Performance (`tests/perf/`):** mark `@pytest.mark.slow`, run nightly only.

---

## 13. Reporting Artifacts (mandatory)

For each run, persist under `mlruns/<run_id>/artifacts/`:

1. `config.yaml` — exact resolved config.
2. `git_sha.txt`, `uv.lock.snapshot`.
3. `pairs_selected.csv` — symbols, p-values, half-lives.
4. `regime_diagnostics.html` — state probabilities + features overlay.
5. `equity_curve.parquet` per fold + aggregate.
6. `metrics.json` — Sharpe, Sortino, Calmar, MDD, hit rate, avg holding, turnover, slippage cost share.
7. `action_heatmap.png` — z-score (x) × regime (y) → P(action).
8. `report.html` — single-file standalone HTML.
9. `policy.zip` — SB3 saved model.
10. `BIAS_CHECKLIST.md` — checked.

---

## 14. Interview Talking Points (`docs/interview_talking_points.md`)

The agent must populate this with concrete numbers from the latest run, framed for:

- **Why RL over fixed thresholds:** adaptivity to regime; learned exit timing reduces tail losses (cite drawdown delta vs baseline).
- **Why cointegration first:** ensures the spread is stationary in expectation; RL solves the *timing*, not the *existence* of mean reversion.
- **Why HMM regime feature:** measurable lift in Sharpe when regime onehot included (run an ablation; report numbers).
- **Bias controls:** reference §11 line by line.
- **What you'd do next:** continuous actions, multi-pair portfolio, intraday with LOB features, transaction-cost-aware policy gradient.

---

## 15. ATS Keywords (must appear naturally in README + docs)

Statistical Arbitrage · Pairs Trading · Cointegration · Engle-Granger · Johansen Test · Reinforcement Learning · PPO · SAC · Stable Baselines3 · Gymnasium · OpenAI Gym · Mean Reversion · Market Microstructure · HMM · Hidden Markov Model · Market Regime · Backtesting · Walk-Forward Validation · Sharpe Ratio · Sortino Ratio · Calmar Ratio · Drawdown · PnL · Transaction Costs · Slippage · Market Impact · Alpha Generation · High Frequency · Algorithmic Trading · Quantitative Finance · Point-in-Time Universe · Survivorship Bias · Look-Ahead Bias

---

## 16. Working Agreements for the Coding Agent

1. **Read before writing.** Inspect any file before editing it.
2. **One ADR per irreversible decision.** No silent design changes.
3. **No new dependencies without justification** in the relevant ADR or PR description.
4. **Tests first for any module touching data or reward.** Property tests are non-optional for §11 items.
5. **Determinism is a feature, not an aspiration.** Every randomness source seeded.
6. **No network calls in tests. Ever.** Use fixtures committed under `tests/fixtures/`.
7. **Keep the smoke config fast.** If `make smoke` exceeds 60s, fix it before adding features.
8. **Document as you go.** `docs/methodology.md` updated alongside code, not after.
9. **No look-ahead. No survivorship. No data snooping.** Every PR description must affirm these.
10. **Stop and ask** if a requirement here conflicts with reality. Do not silently relax constraints.

---

## 17. Quick-Start (what the README will promise)

```bash
git clone <repo> && cd 13-rl-statistical-arbitrage
make setup                       # uv sync, pre-commit install
make smoke                       # ~30s end-to-end on fixtures
make data                        # fetch sample (yfinance, ~2 min)
make train CONFIG=configs/fast.yaml
make backtest CONFIG=configs/fast.yaml
make report                      # opens report.html
```

End of master prompt.
