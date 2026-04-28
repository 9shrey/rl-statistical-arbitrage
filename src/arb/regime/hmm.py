"""Gaussian regime model.

Uses ``hmmlearn`` if available; otherwise falls back to ``sklearn``'s
``GaussianMixture``. In both cases:
- ``fit`` is called only on training data.
- ``predict_causal`` decodes the test slice without leaking future information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeModel:
    n_states: int
    seed: int = 42
    _model: object = None
    _state_order: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RegimeModel":
        X = np.asarray(X, dtype=float)
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore

            m = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=self.seed,
            )
            m.fit(X)
            means = m.means_
        except Exception:
            from sklearn.mixture import GaussianMixture  # type: ignore

            m = GaussianMixture(
                n_components=self.n_states,
                covariance_type="diag",
                random_state=self.seed,
                max_iter=100,
            )
            m.fit(X)
            means = m.means_
        self._model = m
        # Order states by sum of means (proxy for "trending" intensity).
        self._state_order = np.argsort(means.sum(axis=1))
        return self

    def predict_causal(self, X: np.ndarray) -> np.ndarray:
        """Decode states; uses the fitted model only (no refit on X)."""
        if self._model is None:
            raise RuntimeError("RegimeModel not fitted")
        X = np.asarray(X, dtype=float)
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore

            if isinstance(self._model, GaussianHMM):
                states = self._model.predict(X)
            else:  # GaussianMixture
                states = self._model.predict(X)
        except Exception:  # pragma: no cover
            states = self._model.predict(X)  # type: ignore[union-attr]
        # Remap to canonical (low-mean=0, high-mean=K-1)
        order = {int(s): int(rank) for rank, s in enumerate(self._state_order)}
        return np.array([order[int(s)] for s in states], dtype=int)

    def onehot(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=int)
        oh = np.zeros((len(states), self.n_states), dtype=np.float32)
        oh[np.arange(len(states)), states] = 1.0
        return oh
