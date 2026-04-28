from __future__ import annotations

import numpy as np

from arb.pairs.engle_granger import _half_life, engle_granger
from arb.pairs.ranking import composite_score


def test_engle_granger_detects_cointegration_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 500
    factor = np.cumsum(rng.normal(0, 1.0, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0, 0.5)
    b = 50 + 0.5 * factor + rng.normal(0, 0.2, n)
    a = 100 + 1.0 * factor + spread + rng.normal(0, 0.2, n)
    res = engle_granger(a, b)
    assert res.pvalue <= 0.10
    assert res.half_life_days < 50


def test_half_life_inf_for_random_walk() -> None:
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 1.0, 500))
    hl = _half_life(rw)
    # A random walk should not have a short, finite half-life.
    assert hl > 50 or hl == float("inf")


def test_composite_score_ordering() -> None:
    a = composite_score(0.01, 5)
    b = composite_score(0.20, 30)
    assert a > b
