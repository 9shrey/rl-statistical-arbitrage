"""Static z-score threshold baseline policy.

Convention: z = (price_a - hedge*price_b - mean) / std.
- Enter LONG_SPREAD when z <= -entry (spread is depressed, expect mean-revert up).
- Enter SHORT_SPREAD when z >=  entry.
- Exit to FLAT when |z| <= exit (and apply hard stop at |z| >= stop).
The previous position is tracked across calls; observations are
``[zscore, ..., position]`` so we can read state from the obs vector.
"""
from __future__ import annotations

import numpy as np

from arb.env.gym_env import ACTION_FLAT, ACTION_LONG, ACTION_SHORT


class ZScoreThresholdPolicy:
    def __init__(self, entry: float = 2.0, exit_z: float = 0.5, stop: float = 4.0):
        self.entry = float(entry)
        self.exit_z = float(exit_z)
        self.stop = float(stop)

    def reset(self) -> None:
        return None

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:  # noqa: ARG002
        z = float(obs[0])
        position = int(round(float(obs[-1])))
        if not np.isfinite(z):
            return ACTION_FLAT if position == 0 else ACTION_FLAT
        # Hard stop
        if abs(z) >= self.stop:
            return ACTION_FLAT
        if position == 0:
            if z <= -self.entry:
                return ACTION_LONG
            if z >= self.entry:
                return ACTION_SHORT
            return ACTION_FLAT
        if position > 0:
            return ACTION_FLAT if z >= -self.exit_z else ACTION_LONG
        # position < 0
        return ACTION_FLAT if z <= self.exit_z else ACTION_SHORT
