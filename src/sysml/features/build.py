"""Assemble the full feature matrix (technical + regime)."""

from __future__ import annotations

import pandas as pd

from sysml.config import Config, RegimeConfig
from sysml.features.regimes import HMMFactory, RegimeModels, fit_regimes
from sysml.features.technical import FEATURE_GROUPS, TECHNICAL_FEATURES, compute_technical_features


def regime_feature_names(cfg: RegimeConfig) -> list[str]:
    return (
        [f"gmm_prob_{k}" for k in range(cfg.n_gmm)]
        + [f"hmm_prob_{k}" for k in range(cfg.n_hmm)]
        + ["hmm_persistence"]
    )


def feature_columns(cfg: RegimeConfig) -> list[str]:
    return TECHNICAL_FEATURES + regime_feature_names(cfg)


def feature_groups(cfg: RegimeConfig) -> dict[str, list[str]]:
    groups = {k: list(v) for k, v in FEATURE_GROUPS.items()}
    groups["regimes"] = regime_feature_names(cfg)
    return groups


def build_feature_frame(
    ohlcv: pd.DataFrame,
    cfg: Config,
    hmm_factory: HMMFactory | None = None,
) -> tuple[pd.DataFrame, RegimeModels]:
    """Compute all features. Regime models are fitted on the training period only."""
    tech = compute_technical_features(ohlcv)
    regimes = fit_regimes(tech, cfg.split.train_end, cfg.regimes, cfg.seed, hmm_factory)
    frame = tech.join(regimes.transform(tech))
    return frame, regimes
