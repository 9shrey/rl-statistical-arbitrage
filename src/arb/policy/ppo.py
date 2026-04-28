"""PPO trainer wrapper around Stable-Baselines3 (optional dependency)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arb.env.gym_env import EnvSpec, PairsTradingEnv


class PPOPolicyAdapter:
    """Adapter exposing our Policy protocol over an SB3 model."""

    def __init__(self, model: object):
        self._model = model

    def reset(self) -> None:
        return None

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self._model.predict(obs, deterministic=deterministic)  # type: ignore[attr-defined]
        return int(np.asarray(action).flatten()[0])


def train_ppo(
    spec: EnvSpec,
    *,
    total_timesteps: int = 200_000,
    seed: int = 42,
    learning_rate: float = 3.0e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
) -> PPOPolicyAdapter:
    """Train PPO if SB3 is installed; raise a clear error otherwise."""
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "stable-baselines3 + torch required for PPO training. "
            "Install with: pip install stable-baselines3 torch gymnasium"
        ) from e

    def _make() -> PairsTradingEnv:
        return PairsTradingEnv(spec, seed=seed)

    venv = DummyVecEnv([_make])
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return PPOPolicyAdapter(model)


def save_ppo(adapter: PPOPolicyAdapter, path: str | Path) -> None:
    adapter._model.save(str(path))  # type: ignore[attr-defined]
