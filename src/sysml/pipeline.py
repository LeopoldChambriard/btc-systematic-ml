"""End-to-end research pipeline.

    OHLCV -> features (technical + causal regimes) -> trend-scanning labels
          -> primary model (expanding-window OOF + final fit)
          -> meta-model on the primary's OOF longs -> position sizing
          -> importance analysis -> cost-aware backtest + stress periods

Look-ahead control is concentrated here, so it can be audited in one place:

* every fitted object (scalers, GMM, HMM, vol reference, meta threshold) sees training data only;
* sequence samples *end at* the decision bar; the position earns the next bar's return;
* the last ``max_window`` training bars are purged (their labels look into the test period);
* validation folds are separated from their training block by the same gap;
* overlay inputs (volatility, regime probabilities) are read at the decision bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sysml import plotting
from sysml.backtest import (
    BacktestResult,
    apply_bear_cap,
    apply_vol_target,
    buy_and_hold,
    meta_sized,
    primary_long_only,
    run_backtest,
    sma_crossover,
    vol_scalar,
)
from sysml.config import Config
from sysml.data import load_ohlcv, split_masks, validate_ohlcv
from sysml.evaluation import classification_summary
from sysml.features import build_feature_frame, feature_columns
from sysml.features.regimes import HMMFactory, RegimeModels
from sysml.importance import (
    cluster_features,
    clustered_pfi,
    fit_surrogate,
    mdi_importance,
    pfi_importance,
)
from sysml.labeling import forward_return, make_meta_labels, next_bar_return, trend_scanning_labels
from sysml.models import MetaModel, balanced_class_weights, expanding_window_oof, make_primary
from sysml.models.sequences import make_sequences
from sysml.utils import set_global_seed

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    frame: pd.DataFrame  # one row per usable bar: features, labels, forward returns
    cols: list[str]
    x: np.ndarray  # standardised features, one row per frame row
    scaler: StandardScaler
    train_rows: np.ndarray  # row positions in `frame` (after purge) that are training bars
    test_rows: np.ndarray
    regimes: RegimeModels
    data_report: dict


@dataclass
class PipelineResult:
    config: Config
    dataset: Dataset
    oof_folds: pd.DataFrame
    classification: dict[str, dict[str, float]]
    meta_info: dict[str, Any]
    importance: dict[str, Any]
    backtests: list[BacktestResult]
    stress: dict[str, list[BacktestResult]]
    signals: pd.DataFrame
    history: dict[str, list[float]] | None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame({r.label: r.metrics for r in self.backtests}).T


def prepare_dataset(
    ohlcv: pd.DataFrame, cfg: Config, hmm_factory: HMMFactory | None = None
) -> Dataset:
    report = validate_ohlcv(ohlcv, cfg.data.max_gap_hours)
    cols = feature_columns(cfg.regimes)

    feat, regimes = build_feature_frame(ohlcv, cfg, hmm_factory)
    close = feat["close"]
    feat = feat.join(
        trend_scanning_labels(close, cfg.labels.min_window, cfg.labels.max_window)
    )
    feat["fwd_ret_meta"] = forward_return(close, cfg.labels.meta_horizon)
    feat["ret_next"] = next_bar_return(close)
    feat["vol_decision"] = feat["log_return"].rolling(cfg.backtest.vol_window).std()

    frame = feat.dropna(subset=[*cols, "label", "fwd_ret_meta", "ret_next", "vol_decision"])
    train_mask, test_mask = split_masks(frame.index, cfg.split)
    train_rows = np.flatnonzero(train_mask)
    test_rows = np.flatnonzero(test_mask)
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise ValueError("Empty train or test period: check the split dates against the data range")

    # Purge: trend-scanning labels of the last `max_window` training bars look into the test period.
    embargo = cfg.labels.max_window
    train_rows = train_rows[:-embargo] if embargo else train_rows

    # Context rows needed to form the first sequence of each block.
    seq_len = cfg.primary.seq_len
    train_rows = train_rows[train_rows >= seq_len - 1]
    test_rows = test_rows[test_rows >= seq_len - 1]

    scaler = StandardScaler().fit(frame[cols].to_numpy()[train_rows])
    x = scaler.transform(frame[cols].to_numpy())
    log.info("Dataset: %d usable bars, %d train / %d test samples (after purge)",
             len(frame), len(train_rows), len(test_rows))
    return Dataset(frame, cols, x, scaler, train_rows, test_rows, regimes, report)


def _sample_index(rows: np.ndarray, seq_len: int) -> np.ndarray:
    """Row ``r`` is sample ``r - (seq_len - 1)`` of ``make_sequences(x, seq_len)``."""
    return rows - (seq_len - 1)


def fit_primary_and_oof(ds: Dataset, cfg: Config):
    seq_len = cfg.primary.seq_len
    seq = make_sequences(ds.x, seq_len)
    s_train, s_test = _sample_index(ds.train_rows, seq_len), _sample_index(ds.test_rows, seq_len)
    y_all = (ds.frame["label"].to_numpy() == 1).astype(int)
    x_tr, y_tr = seq[s_train], y_all[ds.train_rows]

    n_features = len(ds.cols)
    is_lstm = cfg.primary.type == "lstm"
    log.info("Primary model '%s': %d-fold expanding-window OOF", cfg.primary.type,
             cfg.primary.cv_splits)
    oof, folds = expanding_window_oof(
        lambda: make_primary(cfg.primary, n_features, cfg.seed),
        x_tr, y_tr, cfg.primary.cv_splits, gap=cfg.labels.max_window,
        epochs=cfg.primary.lstm.epochs_cv if is_lstm else None,
    )
    log.info("OOF macro-F1: %.4f +/- %.4f", folds["f1_macro"].mean(), folds["f1_macro"].std())

    final = make_primary(cfg.primary, n_features, cfg.seed)
    final.fit(x_tr, y_tr, class_weight=balanced_class_weights(y_tr))
    p_test = final.predict_proba(seq[s_test])
    return oof, folds, p_test, final.history


def fit_meta(ds: Dataset, cfg: Config, oof: np.ndarray, n_test: int, p_up_test: np.ndarray):
    """Train the meta-model on the primary's OOF long signals; score the test longs."""
    info: dict[str, Any] = {"status": "ok"}
    fwd = ds.frame["fwd_ret_meta"].to_numpy()
    valid = ~np.isnan(oof)
    rows_oof = ds.train_rows[valid]
    long_oof = oof[valid] > 0.5

    mask, y_meta = make_meta_labels(long_oof, fwd[rows_oof], cfg.labels.meta_threshold)
    info["n_meta_samples"] = int(mask.sum())
    info["meta_base_rate"] = float(y_meta.mean()) if len(y_meta) else float("nan")
    meta_proba_test = np.zeros(n_test)
    try:
        meta = MetaModel(cfg.meta, cfg.seed).fit(
            ds.x[rows_oof[mask]], y_meta, embargo=cfg.labels.meta_horizon
        )
    except ValueError as exc:
        info["status"] = f"skipped ({exc})"
        info["threshold"] = cfg.meta.default_threshold
        return meta_proba_test, info

    long_test = p_up_test > 0.5
    if long_test.any():
        meta_proba_test[long_test] = meta.predict_proba(ds.x[ds.test_rows[long_test]])
    info.update(
        threshold=meta.threshold_,
        holdout_precision=meta.val_precision_,
        holdout_support=meta.val_support_,
        holdout_base_rate=meta.val_base_rate_,
    )
    return meta_proba_test, info


def build_strategies(
    ds: Dataset, cfg: Config, p_up_test: np.ndarray, meta_proba: np.ndarray, meta_info: dict
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """All positions share one index: the test decision bars (information known at the close)."""
    bt = cfg.backtest
    frame = ds.frame
    test = frame.iloc[ds.test_rows]
    primary_long = p_up_test > 0.5

    sig_primary = primary_long_only(p_up_test)
    sig_meta = meta_sized(primary_long, meta_proba, meta_info["threshold"])
    fallback = (sig_meta > 0.01).mean() < cfg.meta.fallback_min_invested
    if fallback:
        log.warning(
            "Meta filter keeps only %.1f%% of bars (< %.0f%%): falling back to the primary signal. "
            "'Meta sizing' is then identical to 'Primary long-only'.",
            100 * (sig_meta > 0.01).mean(), 100 * cfg.meta.fallback_min_invested,
        )
        sig_meta = sig_primary.copy()
    meta_info["fallback_to_primary"] = bool(fallback)

    # Reference volatility from the training period only (never the test period).
    vol_ref = float(np.median(frame["vol_decision"].to_numpy()[ds.train_rows]))
    scalar = vol_scalar(test["vol_decision"].to_numpy(), vol_ref, bt.vol_clip_min, bt.vol_clip_max)
    p_bear = test[f"hmm_prob_{ds.regimes.bear_state}"].to_numpy()

    sig_vt = apply_vol_target(sig_meta, scalar, bt.max_leverage)
    sig_hmm = apply_bear_cap(sig_meta, p_bear, bt.bear_prob_threshold, bt.bear_cap)
    sig_full = apply_bear_cap(sig_vt, p_bear, bt.bear_prob_threshold, bt.bear_cap)
    sma = sma_crossover(frame["close"], bt.sma_fast, bt.sma_slow).reindex(test.index).fillna(0.0)

    positions = {
        "Buy & Hold": buy_and_hold(len(test)),
        f"SMA {bt.sma_fast}/{bt.sma_slow}": sma.to_numpy(),
        "Primary long-only": sig_primary,
        "Meta sizing": sig_meta,
        "Meta + VolTarget": sig_vt,
        "Meta + HMM cap": sig_hmm,
        "Meta + Vol + HMM": sig_full,
    }
    signals = pd.DataFrame(positions, index=test.index)
    signals.insert(0, "p_up", p_up_test)
    signals.insert(1, "meta_proba", meta_proba)
    signals["p_bear"] = p_bear
    signals["ret_next"] = test["ret_next"].to_numpy()
    return signals, positions


def run_backtests(signals: pd.DataFrame, positions: dict[str, np.ndarray], cfg: Config):
    returns = signals["ret_next"]
    bt = cfg.backtest
    results = [
        run_backtest(returns, pos, label, bt.cost_bps, bt.periods_per_year)
        for label, pos in positions.items()
    ]
    stress: dict[str, list[BacktestResult]] = {}
    for period in bt.stress_periods:
        sel = (signals.index >= pd.Timestamp(period.start)) & (
            signals.index <= pd.Timestamp(period.end) + pd.Timedelta(days=1, nanoseconds=-1)
        )
        if sel.sum() < 24:
            log.warning("Stress period %r has no data in the test window; skipped", period.name)
            continue
        stress[period.name] = [
            run_backtest(returns[sel], pos[sel], label, bt.cost_bps, bt.periods_per_year)
            for label, pos in positions.items()
        ]
    return results, stress


def run_importance(ds: Dataset, cfg: Config) -> dict[str, Any]:
    ic = cfg.importance
    y = ds.frame["label"].to_numpy().astype(int)
    x_tr, y_tr = ds.x[ds.train_rows], y[ds.train_rows]
    x_te, y_te = ds.x[ds.test_rows], y[ds.test_rows]
    surrogate = fit_surrogate(x_tr, y_tr, ic, cfg.seed)
    mdi = mdi_importance(surrogate, ds.cols)
    pfi = pfi_importance(surrogate, x_te, y_te, ds.cols, ic.n_repeats, cfg.seed)
    clusters = cluster_features(x_tr, ds.cols, ic.n_clusters, ic.linkage)
    cpfi = clustered_pfi(surrogate, x_te, y_te, ds.cols, clusters, ic.n_repeats, cfg.seed)
    return {"mdi": mdi, "pfi": pfi, "clusters": clusters, "clustered_pfi": cpfi}


def run_pipeline(
    cfg: Config,
    ohlcv: pd.DataFrame | None = None,
    hmm_factory: HMMFactory | None = None,
) -> PipelineResult:
    set_global_seed(cfg.seed)
    if ohlcv is None:
        ohlcv = load_ohlcv(cfg.data.path, cfg.data.asset)
    ds = prepare_dataset(ohlcv, cfg, hmm_factory)

    oof, folds, p_up_test, history = fit_primary_and_oof(ds, cfg)
    y_test = np.where(ds.frame["label"].to_numpy()[ds.test_rows] == 1, 1, -1)
    pred_primary = np.where(p_up_test > 0.5, 1, -1)

    meta_proba, meta_info = fit_meta(ds, cfg, oof, len(ds.test_rows), p_up_test)
    signals, positions = build_strategies(ds, cfg, p_up_test, meta_proba, meta_info)

    pred_final = np.where(signals["Meta sizing"].to_numpy() > 0.01, 1, -1)
    classification = {
        "primary": classification_summary(y_test, pred_primary),
        "after_meta_filter": classification_summary(y_test, pred_final),
    }
    backtests, stress = run_backtests(signals, positions, cfg)
    importance = run_importance(ds, cfg)

    return PipelineResult(
        config=cfg, dataset=ds, oof_folds=folds, classification=classification,
        meta_info=meta_info, importance=importance, backtests=backtests, stress=stress,
        signals=signals, history=history,
        extras={"y_test": y_test, "pred_primary": pred_primary, "pred_final": pred_final},
    )


def make_figures(res: PipelineResult, out_dir) -> list[str]:
    """Write the standard figure set. Returns the file names produced."""
    fig_dir = Path(out_dir) / "figures"
    ds, cfg, imp, ex = res.dataset, res.config, res.importance, res.extras
    train = ds.frame.iloc[ds.train_rows]
    period = f"{res.signals.index[0]:%Y-%m-%d} to {res.signals.index[-1]:%Y-%m-%d}"

    jobs: dict[str, Any] = {
        "correlation.png": lambda p: plotting.correlation_heatmap(train[ds.cols], p),
        "hmm_regimes.png": lambda p: plotting.regime_plot(train, ds.regimes.state_names, p),
        "trend_scanning_labels.png": lambda p: plotting.labels_plot(train, p),
        "convergence.png": lambda p: plotting.convergence_plot(res.history, p),
        "feature_importance.png": lambda p: plotting.importance_plot(
            imp["mdi"], imp["pfi"], imp["clustered_pfi"], p
        ),
        "confusion.png": lambda p: plotting.confusion_plot(
            {
                "Primary model": (ex["y_test"], ex["pred_primary"]),
                "After meta filter": (ex["y_test"], ex["pred_final"]),
            },
            p,
        ),
        "backtest.png": lambda p: plotting.backtest_panels(
            res.backtests, p,
            f"Backtest {period} (costs {cfg.backtest.cost_bps:.0f} bps per unit turnover)",
        ),
    }
    for name, results in res.stress.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        jobs[f"stress_{slug}.png"] = lambda p, n=name, r=results: plotting.stress_plot(r, n, p)

    produced = []
    for fname, draw in jobs.items():
        drew = draw(fig_dir / fname)
        if drew is not False:  # convergence_plot returns False when there is no history
            produced.append(fname)
    return produced


__all__ = ["Dataset", "PipelineResult", "make_figures", "prepare_dataset", "run_pipeline"]
