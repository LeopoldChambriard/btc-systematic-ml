"""Feature importance: MDI, PFI and clustered PFI.

* **MDI** (mean decrease in impurity): free by-product of a random forest, but measured
  in-sample and biased towards high-variance continuous features.
* **PFI** (permutation importance): drop in out-of-sample F1 when one feature is shuffled.
  Honest about generalisation, but *diluted* under multicollinearity: if RSI and StochRSI carry
  the same information, shuffling one leaves the other intact and both look unimportant.
* **Clustered PFI**: group features by hierarchical clustering of ``1 - |corr|`` and shuffle each
  *group* jointly (same row permutation for all its columns). This breaks the group's link
  to the target without breaking its internal correlation, so shared information is credited
  to the cluster instead of being diluted.

The tree surrogate is a stand-in for the LSTM, which is a black box; importance is therefore
"what a tree finds predictive in these features", not "what the LSTM uses".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from sysml.config import ImportanceConfig


def fit_surrogate(
    x: np.ndarray, y: np.ndarray, cfg: ImportanceConfig, seed: int = 42
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)


def mdi_importance(model: RandomForestClassifier, names: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)


def pfi_importance(
    model, x: np.ndarray, y: np.ndarray, names: list[str], n_repeats: int, seed: int = 42
) -> pd.DataFrame:
    res = permutation_importance(
        model, x, y, n_repeats=n_repeats, scoring="f1_macro", random_state=seed, n_jobs=-1
    )
    return pd.DataFrame(
        {"importance": res.importances_mean, "std": res.importances_std}, index=names
    ).sort_values("importance", ascending=False)


def cluster_features(
    x: np.ndarray, names: list[str], n_clusters: int, method: str = "ward"
) -> pd.Series:
    """Cluster id (1..n_clusters) per feature from hierarchical clustering on ``1 - |corr|``."""
    corr = pd.DataFrame(x, columns=names).corr().abs().fillna(0.0)
    dist = (1.0 - corr).clip(lower=0.0).to_numpy()
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    ids = fcluster(linkage(squareform(dist, checks=False), method=method), n_clusters, "maxclust")
    return pd.Series(ids, index=names, name="cluster")


def clustered_pfi(
    model,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    clusters: pd.Series,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Joint-permutation importance per cluster (drop in macro-F1)."""
    base = f1_score(y, model.predict(x), average="macro", zero_division=0)
    rng = np.random.RandomState(seed)
    position = {n: i for i, n in enumerate(names)}
    rows = []
    for c in sorted(clusters.unique()):
        members = clusters.index[clusters == c].tolist()
        cols = [position[m] for m in members]
        drops = []
        for _ in range(n_repeats):
            xp = x.copy()
            xp[:, cols] = x[rng.permutation(len(x))][:, cols]  # one permutation for the group
            drops.append(base - f1_score(y, model.predict(xp), average="macro", zero_division=0))
        rows.append(
            {
                "cluster": f"C{c}",
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
                "n_features": len(members),
                "features": ", ".join(members),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
