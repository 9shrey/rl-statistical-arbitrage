# 0001 — RL algorithm = PPO

## Status
Accepted

## Context
We need an RL algorithm for a discrete-action, single-asset (spread) trading environment with bounded reward and modest episode lengths (one walk-forward fold ≈ 100–500 bars).

## Decision
Use **Proximal Policy Optimization (PPO)** from Stable-Baselines3 as the primary algorithm.

## Consequences
- Stable, well-documented, on-policy training that handles discrete action spaces natively.
- Easy to reproduce and audit (clip-range bounds policy update size).
- SB3's interface is compatible with our Gymnasium env without custom wrappers.

## Alternatives considered
- **DQN** — sample-inefficient on small datasets; off-policy replay couples poorly with non-stationary financial series.
- **SAC** — primarily continuous-action; saved for v2 (continuous spread sizing).
- **Custom policy gradient** — rejected; reinventing battle-tested SB3 code adds risk.
