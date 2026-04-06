"""Regime features: the filtered-vs-smoothed distinction is the core of the leakage fix."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from helpers import synthetic
from scipy.stats import multivariate_normal

from sysml.config import RegimeConfig
from sysml.features.regimes import fit_regimes, forward_filter, to_hourly
from sysml.features.technical import compute_technical_features

# A small 2-state, 1-d Gaussian HMM with known parameters.
PI = np.array([0.5, 0.5])
A = np.array([[0.95, 0.05], [0.05, 0.95]])
MU = np.array([[-1.0], [1.0]])
COV = np.array([[[0.5]], [[0.5]]])


def _sample(n=300, seed=0):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = rng.choice(2, p=A[state[t - 1]])
    return (MU[state, 0] + np.sqrt(0.5) * rng.standard_normal(n))[:, None]


def _smoothed(x):
    """Reference forward-backward (scaled, no log tricks) for the tiny model above."""
    b = np.column_stack([multivariate_normal.pdf(x, MU[k], COV[k]) for k in range(2)])
    T = len(x)
    alpha, beta = np.zeros((T, 2)), np.ones((T, 2))
    alpha[0] = PI * b[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * b[t]
        alpha[t] /= alpha[t].sum()
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (b[t + 1] * beta[t + 1])
        beta[t] /= beta[t].sum()
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def test_filter_probabilities_are_valid():
    p = forward_filter(_sample(), PI, A, MU, COV)
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_filter_equals_smoother_on_last_step_only():
    x = _sample()
    filt, smooth = forward_filter(x, PI, A, MU, COV), _smoothed(x)
    assert np.allclose(filt[-1], smooth[-1], atol=1e-9)  # nothing lies ahead of the last point
    # ...but earlier on they differ: smoothing borrows from the future. This is the leak.
    assert np.abs(filt[:-1] - smooth[:-1]).max() > 1e-2


def test_filter_is_causal():
    x = _sample()
    full = forward_filter(x, PI, A, MU, COV)
    for t in (1, 17, 120, 299):
        assert np.allclose(forward_filter(x[:t], PI, A, MU, COV), full[:t], atol=1e-12)


def test_filter_survives_extreme_observations():
    x = np.array([[0.0], [1e3], [-1e3], [0.0]])  # likelihood underflows in linear space
    p = forward_filter(x, PI, A, MU, COV)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)


def test_daily_values_become_available_the_next_day():
    daily = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
    hourly = pd.date_range("2021-01-01", "2021-01-03 23:00", freq="h")

    causal = to_hourly(daily, hourly, causal=True)["v"]
    assert causal["2021-01-01"].isna().all()  # day 1 is not known during day 1
    assert (causal["2021-01-02"] == 1.0).all()  # ...it is known during day 2
    assert (causal["2021-01-03"] == 2.0).all()

    legacy = to_hourly(daily, hourly, causal=False)["v"]
    assert (legacy["2021-01-01"] == 1.0).all()  # 00:00 already "knows" the 23:00 close


def _fitted(causal=True):
    pytest.importorskip("hmmlearn")
    tech = compute_technical_features(synthetic(periods=24 * 420, seed=7))
    cfg = RegimeConfig(hmm_n_iter=30, causal=causal)
    return tech, fit_regimes(tech, "2018-09-30", cfg, seed=0)


def test_state_naming_is_a_partition():
    _, models = _fitted()
    assert sorted([models.bull_state, models.bear_state, models.neutral_state]) == [0, 1, 2]


def test_regime_columns_are_probabilities():
    tech, models = _fitted()
    out = models.transform(tech).dropna()
    assert np.allclose(out[[f"gmm_prob_{k}" for k in range(3)]].sum(axis=1), 1.0)
    assert np.allclose(out[[f"hmm_prob_{k}" for k in range(3)]].sum(axis=1), 1.0)
    assert out["hmm_persistence"].between(0, 1).all()


def test_causal_regimes_do_not_see_the_future():
    tech, models = _fitted(causal=True)
    full = models.transform(tech)
    cut = 24 * 300 + 11  # mid-day on purpose, after the training period
    part = models.transform(tech.iloc[:cut])
    pd.testing.assert_frame_equal(part, full.iloc[:cut], check_exact=False, atol=1e-9)


def test_legacy_regimes_do_leak():
    """Documents the bug: the same check fails with smoothing + same-day mapping."""
    tech, models = _fitted(causal=True)
    legacy = dataclasses.replace(models, cfg=dataclasses.replace(models.cfg, causal=False))
    full = legacy.transform(tech)
    part = legacy.transform(tech.iloc[: 24 * 300 + 11])
    cols = [c for c in full.columns if c.startswith("hmm_prob")]
    diff = (part[cols] - full[cols].iloc[: len(part)]).abs().max().max()
    assert diff > 1e-6
