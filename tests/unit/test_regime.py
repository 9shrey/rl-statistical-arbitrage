from __future__ import annotations

import numpy as np

from arb.regime.hmm import RegimeModel


def test_regime_model_fit_predict_shapes() -> None:
    rng = np.random.default_rng(0)
    X_train = np.concatenate(
        [
            rng.normal(0, 0.1, size=(100, 2)),
            rng.normal(2, 0.1, size=(100, 2)),
        ],
        axis=0,
    )
    rng.shuffle(X_train)
    rm = RegimeModel(n_states=2, seed=0).fit(X_train)
    states = rm.predict_causal(X_train)
    assert states.shape == (200,)
    assert set(np.unique(states).tolist()).issubset({0, 1})
    oh = rm.onehot(states)
    assert oh.shape == (200, 2)
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_regime_model_canonical_state_ordering() -> None:
    rng = np.random.default_rng(1)
    low = rng.normal(0.0, 0.1, size=(100, 1))
    high = rng.normal(5.0, 0.1, size=(100, 1))
    X = np.vstack([low, high])
    rm = RegimeModel(n_states=2, seed=0).fit(X)
    states = rm.predict_causal(X)
    # Lower-mean cluster should map to state 0 by convention.
    assert states[:100].mean() < states[100:].mean()
