"""Synthetic hourly OHLCV from a 3-regime Markov-switching process.

Used by the test-suite and by ``make demo`` so that the whole pipeline can run without any
market data. It is **not** a market simulator: it only needs regimes, volatility clustering and
valid OHLC bars. Do not read performance numbers produced on it as evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    start: str = "2018-01-01",
    periods: int = 24 * 365 * 2,
    seed: int = 0,
    s0: float = 10_000.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # hourly drift / volatility for (bull, bear, chop); regimes last ~ 2-6 weeks on average
    drift = np.array([0.00008, -0.00010, 0.00000])
    vol = np.array([0.0060, 0.0090, 0.0045])
    stay = 1.0 - 1.0 / (24 * np.array([21.0, 14.0, 28.0]))

    state = np.empty(periods, dtype=int)
    state[0] = 2
    for t in range(1, periods):
        if rng.random() < stay[state[t - 1]]:
            state[t] = state[t - 1]
        else:
            state[t] = rng.choice([k for k in range(3) if k != state[t - 1]])

    ret = drift[state] + vol[state] * rng.standard_normal(periods)
    close = s0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[s0], close[:-1]])
    spread = np.abs(rng.standard_normal(periods)) * vol[state] * 0.6
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=2.0, sigma=0.5, size=periods) * (1 + 40 * np.abs(ret) / vol[state])

    index = pd.date_range(start, periods=periods, freq="h", name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
