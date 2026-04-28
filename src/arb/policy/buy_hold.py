"""Buy-and-hold the spread (always +1) baseline."""
from __future__ import annotations

import numpy as np

from arb.env.gym_env import ACTION_LONG


class BuyHoldSpreadPolicy:
    def reset(self) -> None:
        return None

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:  # noqa: ARG002
        _ = obs
        return ACTION_LONG
