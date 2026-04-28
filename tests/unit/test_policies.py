from __future__ import annotations

import numpy as np

from arb.env import ACTION_FLAT, ACTION_LONG
from arb.policy.baseline_zscore import ZScoreThresholdPolicy
from arb.policy.buy_hold import BuyHoldSpreadPolicy


def test_zscore_policy_enters_long_when_z_low() -> None:
    p = ZScoreThresholdPolicy(entry=2.0, exit_z=0.5, stop=4.0)
    obs = np.array([-2.5, 0.0, 0.0, 0.0], dtype=np.float32)  # last = position
    assert p.predict(obs) == ACTION_LONG


def test_zscore_policy_holds_flat_when_within_band() -> None:
    p = ZScoreThresholdPolicy(entry=2.0, exit_z=0.5, stop=4.0)
    obs = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    assert p.predict(obs) == ACTION_FLAT


def test_zscore_policy_stops_out_at_extreme() -> None:
    p = ZScoreThresholdPolicy(entry=2.0, exit_z=0.5, stop=4.0)
    obs = np.array([-5.0, 0.0, 0.0, 1.0], dtype=np.float32)  # already long, blow-out
    assert p.predict(obs) == ACTION_FLAT


def test_buy_hold_always_long() -> None:
    p = BuyHoldSpreadPolicy()
    assert p.predict(np.zeros(4, dtype=np.float32)) == ACTION_LONG
