"""Meta-model: should we act on the primary model's long signal?

Meta-labeling (López de Prado) splits the problem in two. The primary model decides the *side*;
the meta-model only decides *whether the bet is worth taking* (and how big it should be), given
that the primary model already said long. Its target is "did this long clear the cost hurdle
over the next ``meta_horizon`` hours?".

The meta-model is a random forest with isotonic calibration. Calibration matters because the
output probability is used directly as the position size, so 0.7 has to mean ~70%.

Threshold selection is done on a chronological hold-out that the model has **not** been fitted
on (the coursework version scored the threshold on rows it had been trained on, which produced a
meaningless 100% precision).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from sysml.config import MetaConfig


def select_threshold(
    proba: np.ndarray, y: np.ndarray, cfg: MetaConfig
) -> tuple[float, float, int]:
    """Threshold maximising precision subject to a minimum support. Returns (thr, precision, n)."""
    min_support = max(10, int(cfg.min_support_fraction * len(proba)))
    best_thr, best_prec, best_n = cfg.default_threshold, -1.0, 0
    for thr in np.arange(cfg.threshold_min, cfg.threshold_max, cfg.threshold_step):
        keep = proba > thr
        if keep.sum() < min_support:
            continue
        prec = float((y[keep] == 1).mean())
        if prec > best_prec:  # strict '>' -> lowest threshold wins ties
            best_thr, best_prec, best_n = float(thr), prec, int(keep.sum())
    return best_thr, max(best_prec, 0.0), best_n


@dataclass
class MetaModel:
    cfg: MetaConfig
    seed: int = 42
    threshold_: float = 0.5
    val_precision_: float = float("nan")
    val_support_: int = 0
    val_base_rate_: float = float("nan")
    model_: CalibratedClassifierCV | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, embargo: int = 0) -> MetaModel:
        """Fit on the early part of ``x`` (chronological), pick the threshold on the later part.

        ``embargo`` rows are dropped between the two parts to avoid label overlap.
        """
        n = len(x)
        n_val = int(n * self.cfg.val_fraction)
        n_fit = n - n_val - embargo
        if n_fit < 50 or n_val < 20:
            raise ValueError(f"Not enough meta samples to fit and validate (n={n})")
        if len(np.unique(y[:n_fit])) < 2:
            raise ValueError("Meta training block contains a single class")

        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )
        # TimeSeriesSplit: every calibrator is fitted on data that precedes the data it scores.
        cv = TimeSeriesSplit(n_splits=self.cfg.calibration_splits)
        self.model_ = CalibratedClassifierCV(rf, cv=cv, method="isotonic")
        self.model_.fit(x[:n_fit], y[:n_fit])

        x_val, y_val = x[n - n_val :], y[n - n_val :]
        p_val = self.predict_proba(x_val)
        self.threshold_, self.val_precision_, self.val_support_ = select_threshold(
            p_val, y_val, self.cfg
        )
        self.val_base_rate_ = float(y_val.mean())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "call fit() first"
        return self.model_.predict_proba(x)[:, 1]
