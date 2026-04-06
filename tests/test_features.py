"""Feature contract, equivalence with the original loop implementations, and causality."""

import numpy as np
import pandas as pd
from helpers import synthetic

from sysml.config import RegimeConfig
from sysml.features import FEATURE_GROUPS, TECHNICAL_FEATURES, compute_technical_features
from sysml.features.build import feature_columns, feature_groups
from sysml.features.technical import rolling_autocorr, rolling_ols_tvalue


def test_feature_contract():
    cols = feature_columns(RegimeConfig())
    assert len(cols) == 49 and len(set(cols)) == 49
    grouped = [c for g in feature_groups(RegimeConfig()).values() for c in g]
    assert sorted(grouped) == sorted(cols)  # every feature belongs to exactly one family
    assert [c for g in FEATURE_GROUPS.values() for c in g] == TECHNICAL_FEATURES


def test_technical_features_are_finite_after_warmup():
    feat = compute_technical_features(synthetic(periods=24 * 60))[TECHNICAL_FEATURES]
    warm = feat.iloc[250:]
    assert np.isfinite(warm.to_numpy()).all()


def test_features_are_scale_free():
    """Multiplying every price by a constant must not change any feature (stationarity)."""
    df = synthetic(periods=24 * 60, seed=3)
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 37.0
    a = compute_technical_features(df)[TECHNICAL_FEATURES]
    b = compute_technical_features(scaled)[TECHNICAL_FEATURES]
    bad = [c for c in TECHNICAL_FEATURES
           if not np.allclose(a[c], b[c], rtol=1e-8, atol=1e-10, equal_nan=True)]
    assert bad == []


def test_features_are_causal():
    """Row t must be identical whether or not the future is present."""
    df = synthetic(periods=24 * 90, seed=4)
    full = compute_technical_features(df)
    for cut in (24 * 40 + 7, 24 * 70):
        part = compute_technical_features(df.iloc[:cut])
        pd.testing.assert_frame_equal(
            part, full.iloc[:cut], check_exact=False, rtol=1e-10, atol=1e-12
        )


def test_rolling_autocorr_matches_numpy_loop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    for lag in (1, 3, 10):
        got = rolling_autocorr(x, window=30, lag=lag)
        for i in (29, 77, 150, 299):
            w = x[i - 29 : i + 1]
            want = np.corrcoef(w[: 30 - lag], w[lag:])[0, 1]
            assert abs(got[i] - want) < 1e-12
        assert np.isnan(got[:29]).all()


def test_rolling_ols_tvalue_matches_scipy():
    from scipy.stats import linregress

    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(200)) + 100
    t = rolling_ols_tvalue(y, 25)
    assert len(t) == 200 - 25 + 1
    for k in (0, 40, 100, 175):
        r = linregress(np.arange(25), y[k : k + 25])
        assert abs(t[k] - r.slope / r.stderr) < 1e-8


def test_ols_tvalue_edge_cases():
    flat = rolling_ols_tvalue(np.full(30, 5.0), 10)
    assert (flat == 0).all()  # no slope, no residual: no trend
    line = rolling_ols_tvalue(np.arange(30, dtype=float), 10)
    assert (line == 1e6).all()  # perfect up-trend is flagged with a huge t-value
    assert (rolling_ols_tvalue(-np.arange(30, dtype=float), 10) == -1e6).all()
