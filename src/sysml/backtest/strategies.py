"""Position-construction rules. Each function maps decision-time information to a position.

Every input must be known at the close of the decision bar; see ``pipeline.build_strategies``
for how they are aligned and lagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(n: int) -> np.ndarray:
    return np.ones(n)


def primary_long_only(p_up: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """1 when the primary model says P(up) > threshold, else cash."""
    return (np.asarray(p_up) > threshold).astype(float)


def meta_sized(
    primary_long: np.ndarray, meta_proba: np.ndarray, threshold: float
) -> np.ndarray:
    """Size = calibrated meta-probability when the primary says long *and* the meta-model clears
    its threshold; cash otherwise."""
    p = np.where(np.asarray(primary_long, dtype=bool), meta_proba, 0.0)
    return np.where(p > threshold, p, 0.0)


def vol_scalar(
    vol: np.ndarray, vol_ref: float, clip_min: float, clip_max: float
) -> np.ndarray:
    """``vol_ref / vol`` bounded to ``[clip_min, clip_max]``.

    ``vol_ref`` must come from the *training* period (the coursework used the test-period
    median, which is look-ahead), and ``vol`` must be the trailing volatility known at the
    decision close.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.asarray(vol) > 0, vol_ref / np.asarray(vol), 1.0)
    return np.clip(s, clip_min, clip_max)


def apply_vol_target(signal: np.ndarray, scalar: np.ndarray, max_leverage: float) -> np.ndarray:
    return np.clip(signal * scalar, 0.0, max_leverage)


def apply_bear_cap(
    signal: np.ndarray, p_bear: np.ndarray, threshold: float, cap: float
) -> np.ndarray:
    """Cap exposure at ``cap`` when the (causal) probability of the bear regime exceeds
    ``threshold``. NaN probabilities (warm-up) leave the signal untouched."""
    out = np.asarray(signal, dtype=float).copy()
    bear = np.nan_to_num(np.asarray(p_bear, dtype=float), nan=0.0) > threshold
    out[bear] = np.minimum(out[bear], cap)
    return out


def sma_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Naive trend-following baseline: long when SMA(fast) > SMA(slow) at the decision close."""
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
