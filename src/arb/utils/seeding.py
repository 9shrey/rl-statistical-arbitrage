"""Deterministic seeding for numpy, random, and (optionally) torch / SB3."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed all available randomness sources for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:  # pragma: no cover - optional dep
        pass
