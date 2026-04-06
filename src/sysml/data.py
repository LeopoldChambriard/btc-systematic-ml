"""Loading, validating and splitting hourly OHLCV data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sysml.config import SplitConfig

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def end_of_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """Last instant of a calendar day, so date bounds are inclusive."""
    return pd.Timestamp(date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def load_ohlcv(path: str | Path, asset: str | None = None) -> pd.DataFrame:
    """Read a CSV with a datetime first column and open/high/low/close/volume.

    An optional ``coin`` column is used to select ``asset``.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    if asset is not None and "coin" in df.columns:
        df = df[df["coin"] == asset]
        if df.empty:
            raise ValueError(f"No rows for asset {asset!r} in {path}")
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV column(s): {missing}")
    return df[OHLCV].astype(float).sort_index()


def validate_ohlcv(df: pd.DataFrame, max_gap_hours: int = 6) -> dict:
    """Fail loudly on corrupt data, warn on holes. Returns a small report."""
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be strictly increasing with no duplicates")
    if df[OHLCV].isna().any().any():
        raise ValueError("OHLCV contains NaN values")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive price found")
    if (df["high"] < df["low"]).any():
        raise ValueError("Found bars with high < low")

    deltas = df.index.to_series().diff().dropna()
    gaps = deltas[deltas > pd.Timedelta(hours=1)]
    longest = float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 1.0
    report = {
        "rows": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "n_gaps": int(len(gaps)),
        "longest_gap_hours": longest,
    }
    if longest > max_gap_hours:
        log.warning("Longest gap in the series is %.0f hours (> %d)", longest, max_gap_hours)
    return report


def split_masks(index: pd.DatetimeIndex, split: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, test) over ``index`` using inclusive end-of-day bounds."""
    train = np.asarray(index <= end_of_day(split.train_end))
    test = np.asarray(
        (index >= pd.Timestamp(split.test_start)) & (index <= end_of_day(split.test_end))
    )
    return train, test
