# 0005 — Cost model parameters and sources

## Status
Accepted

## Context
A backtest with unrealistic costs is meaningless. We need defensible defaults for commission, slippage, and market impact.

## Decision
Per-side cost in basis points:

```
cost_bps = commission_bps + slippage_half_spread_bps + impact_bps_per_pct_adv * pct_adv_traded
```

Defaults:
- `commission_bps_per_side = 0.5` (typical retail/prime broker, US large-cap equities).
- `slippage_half_spread_bps = 1.0` (approximate half spread for liquid US large caps).
- `impact_bps_per_pct_adv = 10.0` (linear impact, conservative; Almgren-style square-root models in v2).

These are exposed in `configs/*.yaml` and applied in `arb.env.costs.CostModel`.

## Consequences
- Reproducible, transparent, configurable.
- Slippage is conservative for liquid names; aggressive for small caps (caveat in `BIAS_CHECKLIST`).

## Alternatives considered
- **Square-root impact** (Kyle / Almgren-Chriss) — better for large orders; v2.
- **Tick-level fill simulation** — needs LOB data; out of scope.
