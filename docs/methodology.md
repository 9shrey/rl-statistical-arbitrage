# Methodology

## 1. Universe and data

We trade equity pairs. v1 uses synthetic fixtures (for tests) and `yfinance`-fetched daily bars (for the CLI). Universe is loaded as **point-in-time** snapshots (`arb.data.universe`) — no symbol may appear in a fold's training set unless it was tradeable on that date.

Time stamps are **tz-aware UTC**; conversion to `America/New_York` happens only at display time.

## 2. Pair selection

Candidate pairs are first pre-filtered by rolling correlation (`|ρ| ≥ 0.7` over 252 days). For each survivor we run the **Engle-Granger 2-step** test (`arb.pairs.engle_granger`):

1. OLS hedge ratio: `price_a = α + β · price_b + ε`.
2. ADF on residuals; report the p-value via `statsmodels` (or a coarse fallback if unavailable).
3. Half-life of mean reversion via AR(1) on `Δspread`.

Pairs with `pvalue ≤ p_max` and `1 ≤ half_life_days ≤ HL_max` are scored:

$$ \text{score} = 0.6 \cdot (1 - \text{pvalue}) + 0.4 \cdot 1 / \text{half\_life} $$

`Johansen` is provided for k≥2 baskets but is not used in v1.

## 3. Spread and z-score (look-ahead-free)

For each bar t:

- Hedge ratio $\beta_t$ is fit by OLS on `(price_a, price_b)` over a trailing window of length $W$ ending at $t-1$.
- Spread $s_t = p^A_t - \beta_t \cdot p^B_t$.
- Z-score uses trailing mean/std over $Z$ bars, **shifted by one bar**, so $z_t$ depends only on data $\le t-1$.

Both quantities are property-tested (`tests/property/test_no_lookahead.py`) for shift invariance under future-only perturbations.

## 4. Regime model

A Gaussian HMM (`hmmlearn`, with a `GaussianMixture` fallback) is fit on training-fold features `(realized_vol, |zscore|, ...)`. State labels are remapped so that index 0 is the lowest-mean (mean-reverting-friendly) regime. Decoding on the test fold is done with the **fitted-only** model (`predict_causal`); no Viterbi smoothing crosses the train/test boundary.

## 5. Environment and cost model

- **Action space**: discrete `{FLAT, LONG_SPREAD, SHORT_SPREAD}` (continuous variant deferred).
- **Observation**: `[zscore, realized_vol, bid_ask_proxy, regime_onehot..., position]`, all finite.
- **Cost** per side (bps): commission + half-spread slippage + linear market impact (`bps_per_pct_adv · pct_adv_traded`).
- **Reward** (`arb.env.reward.compute_reward`): clipped PnL − cost − turnover penalty − drawdown excess penalty.

## 6. Policies

- **Z-score threshold baseline**: enter at `|z| ≥ entry`, exit at `|z| ≤ exit_z`, hard stop at `|z| ≥ stop`.
- **Grid baseline**: search `(entry, exit_z)` on the train fold, freeze for test.
- **Buy-and-hold spread**: control.
- **PPO** (SB3, optional): trained on the train fold's environment, evaluated on the test fold.

## 7. Walk-forward validation

Expanding-train, fixed-test splits over the integer index (`arb.backtest.splitter`). For each fold:

1. Fit hedge ratio + z-score windows are already computed on the full series, but they only see strictly-past data due to `shift(1)`. Regime, normalization stats, and policy training are fit only on the train slice.
2. Roll the policy through the test slice; collect equity, positions, actions.
3. Compute Sharpe, Sortino, Calmar, Max Drawdown, Hit Rate, Turnover.

Aggregated across folds, metrics are reported with mean (and, in v2, block-bootstrap CIs).

## 8. References

- Engle, R., & Granger, C. (1987). *Co-integration and Error Correction.*
- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.*
- Schulman et al. (2017). *Proximal Policy Optimization.*
- Hamilton, J. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series.* (HMM regimes.)
