"""Policy interfaces."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:  # pragma: no cover
        ...

    def reset(self) -> None:  # pragma: no cover
        ...
