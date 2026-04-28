# 0003 — Walk-forward (expanding) over k-fold

## Status
Accepted

## Context
Financial series exhibit non-stationarity, autocorrelation, and regime shifts. Standard k-fold cross-validation shuffles in time, which is invalid: future information leaks into past folds.

## Decision
Use **expanding-window walk-forward** validation. Each fold trains on `[0, t)` and tests on `[t, t + W_test)`. The train window grows; the test window is fixed.

## Consequences
- Honors the arrow of time; no future data in training.
- Reflects how the strategy would actually be retrained in production.
- Per-fold and aggregated metrics are interpretable.

## Alternatives considered
- **Rolling-window walk-forward** — also valid; preferable when older data is suspected non-representative. Configurable in v2.
- **k-fold** — rejected (causality violation).
- **Purged k-fold (López de Prado)** — overkill for daily bars; interesting for v2 with intraday data.
