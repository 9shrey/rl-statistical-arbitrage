# 0002 — Discrete action space in v1

## Status
Accepted

## Context
The agent must choose between holding the spread long, short, or flat. Continuous sizing (e.g., `[-1, +1]`) is appealing for risk control but complicates credit assignment and inflates the cost model (every micro-resize incurs slippage).

## Decision
v1 uses `Discrete(3)`: `{FLAT, LONG_SPREAD, SHORT_SPREAD}`. Continuous sizing is deferred to v2.

## Consequences
- Simpler reward landscape; faster convergence with PPO.
- Realistic transaction-cost accounting (one trade per regime change).
- Comparable to z-score baselines (which are also discrete).

## Alternatives considered
- **Continuous `Box([-1, 1])`** — saved for SAC variant; needs a smarter cost model and turnover penalty tuning.
