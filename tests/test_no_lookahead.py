"""End-to-end look-ahead check: the future must not change past decisions.

We corrupt every price after a cut-off date and re-run the whole signal pipeline (features, regimes,
scaling, primary model, meta-model, overlays). Decisions taken at or before the cut-off must be
bit-for-bit what they were. A pipeline that peeks - for instance an HMM smoothed over the whole
series - fails this test, which is why the same test is expected to *fail* in legacy mode.
"""

import numpy as np
import pandas as pd
import pytest
from helpers import small_cfg, synthetic

from sysml.pipeline import build_strategies, fit_meta, fit_primary_and_oof, prepare_dataset

CUTOFF = pd.Timestamp("2018-12-15 00:00")
SIGNAL_COLUMNS = ["p_up", "meta_proba", "p_bear", "Primary long-only", "Meta sizing",
                  "Meta + VolTarget", "Meta + HMM cap", "Meta + Vol + HMM", "SMA 50/200"]


def _corrupt_future(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    future = out.index > CUTOFF
    rng = np.random.default_rng(123)
    factor = rng.uniform(0.5, 2.0, future.sum())
    out.loc[future, ["open", "high", "low", "close"]] *= factor[:, None]  # keeps high >= low
    out.loc[future, "volume"] *= 3.0
    return out


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    cfg = small_cfg(causal=True)
    ds = prepare_dataset(df, cfg)
    oof, _, p_up, _ = fit_primary_and_oof(ds, cfg)
    meta_proba, info = fit_meta(ds, cfg, oof, len(ds.test_rows), p_up)
    signals, _ = build_strategies(ds, cfg, p_up, meta_proba, info)
    return signals


def test_signals_before_the_cutoff_ignore_the_future():
    pytest.importorskip("hmmlearn")
    df = synthetic()
    before = _signals(df)
    after = _signals(_corrupt_future(df))
    past = before.index <= CUTOFF
    assert past.sum() > 1000  # the check is meaningful: lots of test-period decisions precede it
    pd.testing.assert_frame_equal(
        before.loc[past, SIGNAL_COLUMNS], after.loc[past, SIGNAL_COLUMNS],
        check_exact=False, rtol=1e-9, atol=1e-12,
    )


def test_leaky_hmm_inference_fails_the_same_check():
    pytest.importorskip("hmmlearn")
    df = synthetic()
    regime_cols = {}
    for name, data in (("before", df), ("after", _corrupt_future(df))):
        ds = prepare_dataset(data, small_cfg(causal=False))
        cols = [c for c in ds.frame.columns if c.startswith("hmm_prob")] + ["hmm_regime"]
        regime_cols[name] = ds.frame.loc[ds.frame.index <= CUTOFF, cols]
    diff = (regime_cols["before"] - regime_cols["after"]).abs()
    # With same-day mapping, 00:00 of the cut-off day already "knows" that day's 23:00 close.
    assert diff.max().max() > 0.5


def test_causal_hmm_features_pass_the_same_check():
    pytest.importorskip("hmmlearn")
    df = synthetic()
    regime_cols = {}
    for name, data in (("before", df), ("after", _corrupt_future(df))):
        ds = prepare_dataset(data, small_cfg(causal=True))
        cols = [c for c in ds.frame.columns if c.startswith("hmm_prob")] + ["hmm_regime"]
        regime_cols[name] = ds.frame.loc[ds.frame.index <= CUTOFF, cols]
    assert (regime_cols["before"] - regime_cols["after"]).abs().max().max() < 1e-9
