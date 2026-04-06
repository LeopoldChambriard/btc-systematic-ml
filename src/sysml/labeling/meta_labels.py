"""Forward returns and meta-labels.

Timing convention used everywhere in this project
-------------------------------------------------
A decision is taken at the *close of bar d* using information available at that close. The
resulting position earns the return of the *next* bar, ``close[d+1] / close[d] - 1``.

* ``forward_return(close, h)[d] = close[d+h] / close[d] - 1``: what a long opened at the close of
  ``d`` would have earned after ``h`` hours (used to judge meta-labels).
* ``next_bar_return(close)[d] = close[d+1] / close[d] - 1``: the P&L bar of the position decided
  at ``d`` (used by the backtest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def next_bar_return(close: pd.Series) -> pd.Series:
    return close.shift(-1) / close - 1.0


def make_meta_labels(
    primary_is_long: np.ndarray, fwd_return: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Meta-labels for the bars where the primary model says *long*.

    A long is "good" (label 1) when its forward return clears ``threshold`` (the cost hurdle),
    "bad" (label 0) otherwise. Returns ``(mask, labels)``: ``mask`` selects the long bars and
    ``labels`` holds the 0/1 target for those bars only.
    """
    mask = np.asarray(primary_is_long, dtype=bool)
    labels = (np.asarray(fwd_return)[mask] > threshold).astype(int)
    return mask, labels
