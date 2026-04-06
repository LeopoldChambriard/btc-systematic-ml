import numpy as np

from sysml.config import ImportanceConfig
from sysml.importance import (
    cluster_features,
    clustered_pfi,
    fit_surrogate,
    mdi_importance,
    pfi_importance,
)

NAMES = ["a1", "a2", "b1", "b2", "noise1", "noise2"]


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    x = np.column_stack(
        [z + 0.1 * rng.standard_normal(n), z + 0.1 * rng.standard_normal(n),
         w + 0.1 * rng.standard_normal(n), w + 0.1 * rng.standard_normal(n),
         rng.standard_normal(n), rng.standard_normal(n)]
    )
    y = np.where(z + 0.2 * rng.standard_normal(n) > 0, 1, -1)  # only the "a" pair is informative
    return x, y


def test_correlated_features_share_a_cluster():
    x, _ = _data()
    c = cluster_features(x, NAMES, n_clusters=4)
    assert c["a1"] == c["a2"] and c["b1"] == c["b2"] and c["a1"] != c["b1"]


def test_clustered_pfi_credits_the_group_that_carries_the_signal():
    x, y = _data()
    cfg = ImportanceConfig(n_estimators=80, max_depth=6, n_repeats=5)
    model = fit_surrogate(x[:1000], y[:1000], cfg, seed=0)
    clusters = cluster_features(x[:1000], NAMES, n_clusters=4)
    out = clustered_pfi(model, x[1000:], y[1000:], NAMES, clusters, n_repeats=5, seed=0)
    top = out.iloc[0]
    assert "a1" in top["features"] and "a2" in top["features"]
    assert top["importance"] > 0.15
    noise = out[out["features"].str.contains("noise") & ~out["features"].str.contains("a1")]
    assert (noise["importance"] < 0.05).all()


def test_pfi_and_mdi_shapes_and_normalisation():
    x, y = _data()
    cfg = ImportanceConfig(n_estimators=60, max_depth=6)
    model = fit_surrogate(x[:1000], y[:1000], cfg, seed=0)
    mdi = mdi_importance(model, NAMES)
    assert abs(mdi.sum() - 1.0) < 1e-9 and set(mdi.index) == set(NAMES)
    pfi = pfi_importance(model, x[1000:], y[1000:], NAMES, n_repeats=3, seed=0)
    assert list(pfi.columns) == ["importance", "std"] and len(pfi) == len(NAMES)
    assert pfi["importance"].iloc[0] > 0
