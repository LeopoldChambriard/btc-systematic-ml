"""Unsupervised market-regime features: GMM (hourly) and HMM (daily).

Two things matter here, and both were wrong in the original coursework notebook:

1. **Filtering, not smoothing.** ``hmmlearn``'s ``predict`` / ``predict_proba`` run
   Viterbi / forward-backward over the *whole* series, so the state of a day in 2021
   is informed by the days that follow it. A live system only has the forward pass.
   ``forward_filter`` implements it explicitly from the fitted parameters.
2. **Availability of the daily value.** A daily regime computed from the day's last
   close is only known once that day is over. Forward-filling it onto the hourly grid
   without a lag lets 10:00 see the 23:00 close. In ``causal`` mode the daily value
   becomes available the following day.

``causal=False`` reproduces the legacy behaviour so the size of the bias can be measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sysml.config import RegimeConfig
from sysml.data import end_of_day

HMMFactory = Callable[[RegimeConfig, int], Any]


def forward_filter(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    """Filtered state probabilities ``P(state_t | x_1..x_t)`` of a Gaussian HMM.

    ``x`` is ``(T, d)``; ``covars`` is ``(K, d, d)``. Row ``t`` of the result depends only
    on ``x[: t + 1]``. Computed in log space for numerical robustness.
    """
    x = np.atleast_2d(x)
    n_states = len(startprob)
    log_b = np.column_stack(
        [
            np.atleast_1d(multivariate_normal.logpdf(x, mean=means[k], cov=covars[k]))
            for k in range(n_states)
        ]
    )
    tiny = np.finfo(float).tiny
    log_a = np.log(np.clip(transmat, tiny, None))
    log_pi = np.log(np.clip(startprob, tiny, None))

    out = np.empty_like(log_b)
    alpha = log_pi + log_b[0]
    out[0] = alpha - logsumexp(alpha)
    for t in range(1, len(x)):
        alpha = logsumexp(out[t - 1][:, None] + log_a, axis=0) + log_b[t]
        out[t] = alpha - logsumexp(alpha)
    return np.exp(out)


def _default_hmm_factory(cfg: RegimeConfig, seed: int) -> Any:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover - exercised only without hmmlearn
        raise ImportError("hmmlearn is required: pip install hmmlearn") from exc
    return GaussianHMM(
        n_components=cfg.n_hmm, covariance_type="full", n_iter=cfg.hmm_n_iter, random_state=seed
    )


def daily_frame(close: pd.Series, vol_window: int) -> pd.DataFrame:
    """Daily log-return and trailing volatility, stamped at midnight of the day they describe."""
    daily = close.resample("D").last().to_frame("close")
    daily["ret"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["vol"] = daily["ret"].rolling(vol_window).std()
    return daily[["ret", "vol"]].dropna()


def to_hourly(
    daily_values: pd.DataFrame, hourly_index: pd.DatetimeIndex, causal: bool
) -> pd.DataFrame:
    """Project day-stamped values onto the hourly grid.

    ``causal=True`` makes day ``D``'s value available from ``D + 1`` (once the day is complete).
    """
    if causal:
        daily_values = daily_values.shift(1, freq="D")
    return daily_values.reindex(hourly_index, method="ffill")


@dataclass
class RegimeModels:
    gmm: GaussianMixture
    gmm_scaler: StandardScaler
    hmm: Any
    hmm_scaler: StandardScaler
    cfg: RegimeConfig
    bull_state: int
    bear_state: int
    neutral_state: int

    def _hmm_posteriors(self, x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (state, probabilities) under the configured inference mode."""
        if self.cfg.causal:
            probs = forward_filter(
                x_scaled, self.hmm.startprob_, self.hmm.transmat_, self.hmm.means_, self.hmm.covars_
            )
            return probs.argmax(axis=1), probs
        return self.hmm.predict(x_scaled), self.hmm.predict_proba(x_scaled)  # legacy / leaky

    def transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Regime columns on ``feat``'s hourly index. ``feat`` needs close, return, vol_20."""
        out = pd.DataFrame(index=feat.index, dtype=float)

        # GMM: row-wise, no temporal dependency, hence no look-ahead by construction.
        g = feat[["return", "vol_20"]].dropna()
        proba = self.gmm.predict_proba(self.gmm_scaler.transform(g))
        for k in range(self.cfg.n_gmm):
            out.loc[g.index, f"gmm_prob_{k}"] = proba[:, k]

        daily = daily_frame(feat["close"], self.cfg.vol_window_daily)
        state, probs = self._hmm_posteriors(self.hmm_scaler.transform(daily))
        values = pd.DataFrame(index=daily.index, dtype=float)
        values["hmm_regime"] = state
        for k in range(self.cfg.n_hmm):
            values[f"hmm_prob_{k}"] = probs[:, k]
        # Persistence = P(stay in the current state). Note it is a function of the state only.
        values["hmm_persistence"] = self.hmm.transmat_[state, state]
        hourly = to_hourly(values, feat.index, self.cfg.causal)
        return out.join(hourly)

    @property
    def state_names(self) -> dict[int, str]:
        return {self.bull_state: "Bull", self.bear_state: "Bear", self.neutral_state: "Neutral"}


def fit_regimes(
    feat: pd.DataFrame,
    train_end: str,
    cfg: RegimeConfig,
    seed: int = 42,
    hmm_factory: HMMFactory | None = None,
) -> RegimeModels:
    """Fit GMM and HMM and their scalers on the training period only."""
    cutoff = end_of_day(train_end)

    g = feat[["return", "vol_20"]].dropna()
    g_train = g[g.index <= cutoff]
    gmm_scaler = StandardScaler().fit(g_train)
    gmm = GaussianMixture(
        n_components=cfg.n_gmm, covariance_type="full", n_init=10, random_state=seed
    ).fit(gmm_scaler.transform(g_train))

    daily = daily_frame(feat["close"], cfg.vol_window_daily)
    d_train = daily[daily.index <= cutoff]
    hmm_scaler = StandardScaler().fit(d_train)
    hmm = (hmm_factory or _default_hmm_factory)(cfg, seed)
    hmm.fit(hmm_scaler.transform(d_train))

    # Name the states from training behaviour only: highest / lowest mean daily return.
    train_probs = forward_filter(
        hmm_scaler.transform(d_train), hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_
    )
    states = pd.Series(train_probs.argmax(axis=1), index=d_train.index)
    mean_ret = d_train["ret"].groupby(states).mean().reindex(range(cfg.n_hmm))
    bull, bear = int(mean_ret.idxmax()), int(mean_ret.idxmin())
    rest = [k for k in range(cfg.n_hmm) if k not in (bull, bear)]
    neutral = rest[0] if rest else bull
    return RegimeModels(gmm, gmm_scaler, hmm, hmm_scaler, cfg, bull, bear, neutral)
