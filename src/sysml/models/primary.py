"""Primary (directional) model interface and factory.

The primary model answers one question: *is the local trend up (1) or down (0)?* It sees a
window of ``seq_len`` hourly feature vectors and returns ``P(up)``.

Two implementations share the interface:

* ``lstm``   - the stacked LSTM of the coursework (needs TensorFlow, ``pip install .[tf]``).
* ``logreg`` - a regularised logistic regression on the window summary. It is both a sanity
  baseline ("does the LSTM add anything over a linear model?") and what lets the whole
  pipeline run, and be tested, without TensorFlow.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from sysml.config import PrimaryConfig


class PrimaryModel(Protocol):
    history: dict[str, list[float]] | None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_weight: dict[int, float] | None = None,
        epochs: int | None = None,
    ) -> PrimaryModel: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """P(label = 1) for each sequence, shape ``(n,)``."""
        ...


class LogisticPrimary:
    """Logistic regression on ``[last step, window mean]`` of the standardised features."""

    def __init__(self, c: float = 1.0, seed: int = 42):
        self.c, self.seed = c, seed
        self.history = None
        self._clf: LogisticRegression | None = None

    @staticmethod
    def _summarise(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[:, -1, :], x.mean(axis=1)], axis=1)

    def fit(self, x, y, class_weight=None, epochs=None):  # noqa: ARG002 - interface parity
        self._clf = LogisticRegression(
            C=self.c, class_weight=class_weight, max_iter=2000, random_state=self.seed
        ).fit(self._summarise(x), y)
        return self

    def predict_proba(self, x):
        assert self._clf is not None, "call fit() first"
        return self._clf.predict_proba(self._summarise(x))[:, 1]


def make_primary(cfg: PrimaryConfig, n_features: int, seed: int) -> PrimaryModel:
    if cfg.type == "logreg":
        return LogisticPrimary(cfg.logreg_C, seed)
    if cfg.type == "lstm":
        from sysml.models.lstm import LSTMPrimary  # lazy: TensorFlow is optional

        return LSTMPrimary(cfg.lstm, cfg.seq_len, n_features, seed)
    raise ValueError(f"Unknown primary model type {cfg.type!r} (use 'lstm' or 'logreg')")
