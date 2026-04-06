"""Vectorised long-only backtest with proportional transaction costs.

Conventions
-----------
* ``position[d]`` in ``[0, max_leverage]`` is decided at the close of bar ``d`` and earns
  ``returns[d]``, which must already be the *next-bar* return (see ``sysml.labeling.meta_labels``).
  The caller is responsible for that alignment; ``tests/test_backtest.py`` pins it down.
* Cost = ``cost_bps / 1e4 * |position[d] - position[d-1]|``, charged on bar ``d``. The book starts
  flat, so the first entry pays too. Cash earns nothing and there is no borrowing/funding cost,
  so positions above 1 are *free leverage*: read results with ``max_leverage > 1`` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sysml.backtest.metrics import performance_metrics


@dataclass
class BacktestResult:
    label: str
    net_returns: pd.Series
    equity: pd.Series
    position: pd.Series
    metrics: dict[str, float]


def run_backtest(
    returns: pd.Series,
    position: np.ndarray | pd.Series,
    label: str,
    cost_bps: float = 10.0,
    periods_per_year: int = 365 * 24,
) -> BacktestResult:
    pos = pd.Series(np.asarray(position, dtype=float), index=returns.index)
    if len(pos) != len(returns):
        raise ValueError("position and returns must have the same length")
    if (pos < -1e-12).any():
        raise ValueError("Long-only engine: negative positions are not supported")

    turnover = pos.diff().abs().fillna(pos.abs())  # first bar: from flat to pos[0]
    costs = turnover * (cost_bps / 1e4)
    gross = pos * returns
    net = gross - costs
    equity = (1.0 + net).cumprod()

    metrics = performance_metrics(
        net, pos, gross, turnover, costs, cost_bps / 1e4, periods_per_year
    )
    return BacktestResult(label, net, equity, pos, metrics)
