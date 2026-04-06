"""Stationary technical features from hourly OHLCV.

Everything here is *causal*: the value at row ``t`` only uses rows ``<= t``. All
features are returns, ratios or bounded oscillators, i.e. **scale-free**: multiplying every
price by a constant leaves them unchanged (``tests/test_features.py`` enforces it). A model
trained on 10k-USD Bitcoin is therefore not fooled by a 60k-USD Bitcoin. (The coursework left
the MA crossovers, ATR and MACD in USD, which drifts out of the training range as the price
rises; here they are expressed relative to price.)

The rolling computations that were Python loops in the original notebook
(autocorrelation, CCI mean-absolute-deviation, OLS trend t-value) are vectorised
with ``sliding_window_view``; ``tests/test_features.py`` checks them against
straightforward loop references.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-10

# Features fed to the models, in the order used throughout the project (49 columns).
FEATURE_GROUPS: dict[str, list[str]] = {
    "momentum": ["roc_2", "roc_5", "momentum_3", "momentum_10", "momentum_20", "momentum_accel"],
    "trend": [
        "crossover_5_20", "crossover_20_50", "crossover_50_200",
        "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
        "trend_strength",
    ],
    "volatility": [
        "vol_10", "vol_20", "vol_ratio", "atr_14", "bb_width", "bb_pos", "high_low_ratio",
    ],
    "oscillators": ["rsi", "stoch_rsi", "williams_r", "cci", "macd", "macd_signal", "macd_hist"],
    "volume": ["volume_ratio", "obv_signal"],
    "autocorrelation": ["autocorr_1", "autocorr_3", "autocorr_5", "autocorr_10"],
    "lagged_returns": [
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
    ],
    "calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    # Regime features are appended by sysml.features.regimes (GMM + HMM probabilities).
}
REGIME_FEATURES: list[str] = [
    "gmm_prob_0", "gmm_prob_1", "gmm_prob_2",
    "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_persistence",
]
TECHNICAL_FEATURES: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]


def rolling_autocorr(x: np.ndarray, window: int, lag: int) -> np.ndarray:
    """Pearson autocorrelation at ``lag`` inside each trailing window of ``window`` points."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    w = sliding_window_view(x, window)
    a, b = w[:, : window - lag], w[:, lag:]
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out[window - 1 :] = np.where(den > 0, num / den, np.nan)
    return out


def rolling_ols_tvalue(y: np.ndarray, window: int) -> np.ndarray:
    """t-value of the OLS slope of ``y`` on time inside each trailing window.

    Output ``k`` describes ``y[k : k + window]`` (length ``len(y) - window + 1``).
    Perfect fits (zero residual) return ``+/-1e6``; flat windows return 0.
    """
    n = len(y)
    if n < window:
        return np.empty(0)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    ss_xx = float((xc**2).sum())
    w = sliding_window_view(np.asarray(y, dtype=float), window)
    wc = w - w.mean(axis=1, keepdims=True)
    slope = (wc @ xc) / ss_xx
    sst = (wc**2).sum(axis=1)
    ss_res = np.maximum(sst - slope**2 * ss_xx, 0.0)
    perfect = ss_res <= 1e-12 * np.maximum(sst, 1e-300)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(ss_res / (window - 2) / ss_xx)
        t = np.where(se > 0, slope / se, 0.0)
    t = np.where(perfect, np.sign(slope) * 1e6, t)
    return t


def trend_strength_backward(close: pd.Series, window: int = 20) -> pd.Series:
    """t-value of the OLS slope over the ``window`` closes *before* each row.

    Row ``i`` uses ``close[i-window : i]`` (the current bar is excluded), which is
    the conservative convention of the original notebook.
    """
    prices = close.to_numpy(dtype=float)
    out = np.full(len(prices), np.nan)
    t = rolling_ols_tvalue(prices, window)
    if len(t) > 1:
        # t[k] describes prices[k : k + window] -> available at row k + window.
        # A degenerate (flat / perfect-fit) window reads as "no usable trend": 0.
        out[window:] = np.where(np.abs(t[:-1]) >= 1e6, 0.0, t[:-1])
    return pd.Series(out, index=close.index)


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the base columns, helper columns and all technical features."""
    df = ohlcv[["open", "high", "low", "close", "volume"]].copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # Returns: pct_change for labels/lags, log-returns for volatility (additive).
    df["return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    # Momentum, and its acceleration (a second-derivative proxy for early reversals).
    df["roc_2"] = close.pct_change(2)
    df["roc_5"] = close.pct_change(5)
    df["momentum_3"] = close.pct_change(3)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_20"] = close.pct_change(20)
    df["momentum_accel"] = df["momentum_3"] - df["momentum_10"]

    # Moving averages; crossovers as relative spreads (strength, not just sign).
    ma = {n: close.rolling(n).mean() for n in (5, 20, 50, 200)}
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["crossover_5_20"] = ma[5] / ma[20] - 1
    df["crossover_20_50"] = ma[20] / ma[50] - 1
    df["crossover_50_200"] = ma[50] / ma[200] - 1
    df["price_vs_ma20"] = close / ma[20] - 1
    df["price_vs_ma50"] = close / ma[50] - 1
    df["price_vs_ma200"] = close / ma[200] - 1

    # Volatility family.
    df["vol_10"] = df["log_return"].rolling(10).std()
    df["vol_20"] = df["log_return"].rolling(20).std()
    df["vol_ratio"] = df["vol_10"] / (df["vol_20"] + EPS)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    df["atr_14"] = atr / close  # ATR as a fraction of price
    df["high_low_ratio"] = (high - low) / (atr + EPS)

    bb_std = close.rolling(20).std()
    bb_upper, bb_lower = ma[20] + 2 * bb_std, ma[20] - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / (ma[20] + EPS)
    df["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower + EPS)

    # Oscillators. RSI uses simple (Cutler) averages of gains/losses.
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + EPS)))
    rsi_min, rsi_max = df["rsi"].rolling(14).min(), df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + EPS)
    hh, ll = high.rolling(14).max(), low.rolling(14).min()
    df["williams_r"] = -100 * (hh - close) / (hh - ll + EPS)

    typical = (high + low + close) / 3
    tp_ma = typical.rolling(20).mean()
    tp_mad = np.full(len(df), np.nan)
    if len(df) >= 20:
        w = sliding_window_view(typical.to_numpy(dtype=float), 20)
        tp_mad[19:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    df["cci"] = (typical - tp_ma) / (0.015 * tp_mad + EPS)

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd / close  # all three MACD lines relative to price
    df["macd_signal"] = macd_signal / close
    df["macd_hist"] = (macd - macd_signal) / close

    # Volume: relative volume and a de-trended on-balance volume.
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (vol_ma + EPS)
    direction = np.where(close > prev_close, 1.0, np.where(close < prev_close, -1.0, 0.0))
    obv = pd.Series(direction * volume.to_numpy(), index=df.index).cumsum()
    df["obv_signal"] = (obv - obv.rolling(20).mean()) / (vol_ma + EPS)  # in units of average volume

    # Serial dependence: autocorrelation of returns (momentum > 0, mean-reversion < 0).
    r = df["return"].to_numpy(dtype=float)
    for lag in (1, 3, 5, 10):
        df[f"autocorr_{lag}"] = rolling_autocorr(r, window=30, lag=lag)
    for lag in (1, 2, 3, 5, 10):
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    # Cyclical calendar encoding: 23h and 0h must be neighbours.
    hour, dow = df.index.hour, df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["trend_strength"] = trend_strength_backward(close, window=20)
    return df
