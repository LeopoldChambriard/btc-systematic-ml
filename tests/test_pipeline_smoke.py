import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import small_cfg, synthetic

from sysml.pipeline import prepare_dataset, run_pipeline
from sysml.reporting import save_run


def test_dataset_purges_labels_that_reach_into_the_test_period():
    pytest.importorskip("hmmlearn")
    cfg = small_cfg()
    ds = prepare_dataset(synthetic(), cfg)
    last_train, first_test = ds.train_rows[-1], ds.test_rows[0]
    assert first_test - last_train > cfg.labels.max_window  # purge gap
    assert ds.frame.index[ds.train_rows].max() < ds.frame.index[ds.test_rows].min()
    # the scaler has only ever seen training rows
    train_x = ds.frame[ds.cols].to_numpy()[ds.train_rows]
    assert np.allclose(ds.scaler.mean_, train_x.mean(axis=0))


def test_pipeline_runs_and_writes_a_complete_report():
    pytest.importorskip("hmmlearn")
    res = run_pipeline(small_cfg(), ohlcv=synthetic())

    table = res.metrics_table
    assert list(table.index) == ["Buy & Hold", "SMA 50/200", "Primary long-only", "Meta sizing",
                                 "Meta + VolTarget", "Meta + HMM cap", "Meta + Vol + HMM"]
    assert table.loc["Buy & Hold", "Exposure"] == 1.0 and table.loc["Buy & Hold", "NbTrades"] == 1
    assert np.isfinite(table[["CAGR", "Vol", "Sharpe", "MaxDD"]].to_numpy()).all()
    assert (table["MaxDD"] <= 0).all()
    if res.meta_info.get("fallback_to_primary"):  # fallback must be explicit, never silent
        assert table.loc["Meta sizing", "Sharpe"] == table.loc["Primary long-only", "Sharpe"]

    with tempfile.TemporaryDirectory() as tmp:
        out = save_run(res, root=tmp)
        names = {p.name for p in Path(out).iterdir()}
        assert {"summary.md", "metrics.csv", "metrics.json", "config.yaml", "signals.csv.gz",
                "oof_folds.csv", "importance_mdi.csv", "importance_pfi.csv",
                "importance_clustered_pfi.csv", "figures"} <= names
        figures = {p.name for p in (Path(out) / "figures").iterdir()}
        assert {"backtest.png", "feature_importance.png", "hmm_regimes.png"} <= figures
        summary = (Path(out) / "summary.md").read_text()
        assert "Backtest" in summary and "Meta-model" in summary
