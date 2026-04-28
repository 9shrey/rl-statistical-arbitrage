# 0004 — HMM (with GMM fallback) for regime detection

## Status
Accepted

## Context
The policy benefits from awareness of the market regime (trending vs mean-reverting). We need a probabilistic, train-only model that can label test-fold bars without leaking future information.

## Decision
Fit a **Gaussian HMM** (`hmmlearn`) on training-fold microstructure features. Decode test-fold states using the fitted model only (`predict_causal`). When `hmmlearn` is unavailable, fall back to `sklearn.mixture.GaussianMixture` (lose temporal coupling but keep clustering structure). State indices are canonicalized by mean magnitude so downstream code can reason about "low-vol" vs "high-vol" labels stably.

## Consequences
- Probabilistic, interpretable, and well-supported.
- The fallback ensures the smoke pipeline runs in minimal environments.
- Refit per fold matches walk-forward semantics.

## Alternatives considered
- **Change-point detection** (e.g., BOCPD) — heavier, less interpretable for a feature input.
- **Threshold rules on realized vol** — too brittle.
