"""Classification-side evaluation of the directional signal."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics for labels in ``{-1, +1}``. ``precision_long`` is the one that matters for a
    long-only book: of the bars we go long, how many sit inside a genuine up-trend?"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_long": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_long": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "share_predicted_long": float((y_pred == 1).mean()),
        "share_true_long": float((y_true == 1).mean()),
        "n": int(len(y_true)),
    }
