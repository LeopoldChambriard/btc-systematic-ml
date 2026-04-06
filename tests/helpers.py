"""Shared helpers for the test-suite (plain functions, no fixtures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sysml.config import Config, load_config
from sysml.synthetic import make_synthetic_ohlcv

ROOT = Path(__file__).resolve().parents[1]


def synthetic(periods: int = 24 * 420, seed: int = 0) -> pd.DataFrame:
    """~14 months of hourly bars from 2018-01-01."""
    return make_synthetic_ohlcv(periods=periods, seed=seed)


def small_cfg(causal: bool = True) -> Config:
    """A configuration small enough to run the whole pipeline in seconds."""
    return load_config(
        None,
        [
            "split.train_end=2018-09-30",
            "split.test_start=2018-10-01",
            "split.test_end=2019-02-20",
            f"regimes.causal={'true' if causal else 'false'}",
            "regimes.hmm_n_iter=30",
            "primary.type=logreg",
            "primary.cv_splits=3",
            "meta.n_estimators=30",
            "importance.n_estimators=30",
            "importance.n_repeats=2",
            "backtest.stress_periods=[]",
        ],
    )
