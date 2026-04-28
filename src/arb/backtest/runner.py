"""Run a policy through the env and collect equity curve / positions / actions."""
from __future__ import annotations

import numpy as np

from arb.env.gym_env import EnvSpec, PairsTradingEnv
from arb.policy.base import Policy


def run_policy(spec: EnvSpec, policy: Policy, seed: int = 0) -> np.ndarray:
    """Return equity curve as a numpy array of length T+1 (initial 1.0 + T steps)."""
    env = PairsTradingEnv(spec, seed=seed)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    equity = [env.portfolio.equity]
    while True:
        a = policy.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(int(a))
        equity.append(info["equity"])
        if terminated or truncated:
            break
    return np.asarray(equity, dtype=float)


def run_policy_full(
    spec: EnvSpec, policy: Policy, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (equity, positions, actions). All length-T+1 / T as appropriate."""
    env = PairsTradingEnv(spec, seed=seed)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    equity = [env.portfolio.equity]
    positions = [env.portfolio.position]
    actions: list[int] = []
    while True:
        a = int(policy.predict(obs, deterministic=True))
        actions.append(a)
        obs, _r, terminated, truncated, info = env.step(a)
        equity.append(info["equity"])
        positions.append(info["position"])
        if terminated or truncated:
            break
    return (
        np.asarray(equity, dtype=float),
        np.asarray(positions, dtype=float),
        np.asarray(actions, dtype=int),
    )
