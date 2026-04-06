"""Time-ordered validation: expanding-window out-of-fold predictions with purging."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from sysml.models.primary import PrimaryModel


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """``n / (2 * n_class)`` weights, recomputed on every training block.

    Per-block recomputation matters: 2018 is much more bearish than 2020, so a global weight
    would bake future label balance into early folds.
    """
    classes = np.array([0, 1])
    if len(np.unique(y)) < 2:
        return {0: 1.0, 1: 1.0}
    w = compute_class_weight("balanced", classes=classes, y=y.astype(int))
    return {0: float(w[0]), 1: float(w[1])}


def expanding_window_oof(
    make_model: Callable[[], PrimaryModel],
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    gap: int,
    epochs: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold ``P(up)`` from an expanding-window scheme.

    Fold ``k`` trains on blocks ``1..k`` and predicts block ``k+1``; the first block is never
    predicted (NaN). ``gap`` samples are dropped between the end of the training block and the
    start of the validation block: trend-scanning labels look up to ``max_window`` bars ahead,
    so without the gap the last training labels overlap the validation period (purging, as in
    López de Prado's purged k-fold).

    Early stopping inside ``fit`` uses a tail of the *training* block, never the fold being
    predicted, so the OOF predictions are not tuned on their own targets.
    """
    oof = np.full(len(x), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(x), 1):
        model = make_model()
        model.fit(x[tr], y[tr], class_weight=balanced_class_weights(y[tr]), epochs=epochs)
        oof[va] = model.predict_proba(x[va])
        f1 = f1_score(y[va], (oof[va] > 0.5).astype(int), average="macro", zero_division=0)
        rows.append({"fold": fold, "n_train": len(tr), "n_val": len(va), "f1_macro": f1})
    return oof, pd.DataFrame(rows)
