"""Typed configuration.

The dataclass defaults below are the single source of truth. ``configs/default.yaml``
mirrors them (a test enforces it) and exists so a run can be described, diffed and
overridden from the command line, e.g. ``--set primary.type=logreg``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    path: str = "data/raw/btc_1h.csv"
    asset: str | None = "BTC"  # filter on the optional `coin` column
    max_gap_hours: int = 6  # warn when the hourly series has holes longer than this


@dataclass
class SplitConfig:
    # Dates are inclusive and interpreted as end-of-day.
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2021-12-31"


@dataclass
class RegimeConfig:
    n_gmm: int = 3
    n_hmm: int = 3
    hmm_n_iter: int = 200
    vol_window_daily: int = 20
    # True : HMM posteriors are *filtered* (forward pass only) and lagged by one
    #        completed day. This is what a live system could have computed.
    # False: legacy coursework behaviour (Viterbi / forward-backward over the whole
    #        series + same-day daily value). It leaks future information; keep it
    #        only to quantify the bias (see docs/leakage_audit.md).
    causal: bool = True


@dataclass
class LabelConfig:
    min_window: int = 10  # trend scanning: shortest look-forward window (hours)
    max_window: int = 40  # trend scanning: longest look-forward window (hours)
    meta_horizon: int = 3  # hours over which a long signal is judged
    meta_threshold: float = 0.003  # 30 bps: round-trip cost hurdle for a "good" long


@dataclass
class LSTMConfig:
    units1: int = 128
    units2: int = 64
    dense: int = 32
    dropout: float = 0.3
    lr: float = 5e-4
    batch_size: int = 256
    epochs_cv: int = 50
    epochs_final: int = 60
    patience: int = 6
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    val_split: float = 0.1  # last 10% of the training block, used only for early stopping


@dataclass
class PrimaryConfig:
    type: str = "lstm"  # "lstm" | "logreg"
    seq_len: int = 24
    cv_splits: int = 5
    logreg_C: float = 1.0
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


@dataclass
class MetaConfig:
    n_estimators: int = 300
    max_depth: int = 5
    min_samples_leaf: int = 10
    calibration_splits: int = 3
    val_fraction: float = 0.2  # chronological hold-out used to choose the threshold
    threshold_min: float = 0.30
    threshold_max: float = 0.75
    threshold_step: float = 0.01
    default_threshold: float = 0.5
    min_support_fraction: float = 0.10  # min share of hold-out signals a threshold must keep
    fallback_min_invested: float = 0.10  # below this exposure, fall back to the primary signal


@dataclass
class ImportanceConfig:
    n_estimators: int = 200
    max_depth: int = 8
    n_repeats: int = 20
    n_clusters: int = 8
    linkage: str = "ward"


@dataclass
class StressPeriod:
    name: str = "May-Jul 2021 crash"
    start: str = "2021-05-01"
    end: str = "2021-07-31"


@dataclass
class BacktestConfig:
    cost_bps: float = 10.0  # per unit of position change
    periods_per_year: int = 365 * 24
    vol_window: int = 24
    vol_clip_min: float = 0.2
    vol_clip_max: float = 3.0
    max_leverage: float = 1.5
    bear_prob_threshold: float = 0.9  # P(bear regime) above which exposure is capped
    bear_cap: float = 0.2
    sma_fast: int = 50  # trend-following baseline
    sma_slow: int = 200
    stress_periods: list[StressPeriod] = field(default_factory=lambda: [StressPeriod()])


@dataclass
class Config:
    seed: int = 42
    run_name: str = "default"
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            value = _build(hint, value or {})
        elif get_origin(hint) is list and is_dataclass(get_args(hint)[0]):
            value = [_build(get_args(hint)[0], item) for item in (value or [])]
        kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data = apply_overrides(data, overrides)
    return _build(Config, data)
