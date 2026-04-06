import numpy as np
import pandas as pd
import pytest
from helpers import synthetic

from sysml.labeling import forward_return, make_meta_labels, next_bar_return, trend_scanning_labels


def _tvalue(y):  # the original notebook implementation, kept as the reference
    n = len(y)
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    ss_xx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / ss_xx
    ss_res = ((y - (slope * (x - xm) + ym)) ** 2).sum()
    if ss_res == 0:
        return float(np.sign(slope)) * 1e6
    se = np.sqrt(ss_res / (n - 2) / ss_xx)
    return slope / se if se > 0 else 0.0


def _reference(prices, mn, mx):
    out = []
    for i in range(len(prices) - mn):
        best_t, best_h = 0.0, mn
        for h in range(mn, min(mx, len(prices) - i) + 1):
            t = _tvalue(prices[i : i + h])
            if abs(t) > abs(best_t):
                best_t, best_h = t, h
        out.append((1 if best_t >= 0 else -1, best_t, best_h))
    return np.array(out)


def test_matches_original_loop_implementation():
    prices = synthetic(periods=500, seed=5)["close"]
    got = trend_scanning_labels(prices, 10, 40)
    ref = _reference(prices.to_numpy(), 10, 40)
    n = len(ref)
    assert (got["label"].to_numpy()[:n] == ref[:, 0]).all()
    assert np.allclose(got["t_value"].to_numpy()[:n], ref[:, 1], atol=1e-8)
    assert (got["best_window"].to_numpy()[:n] == ref[:, 2]).all()


def test_labels_follow_obvious_trends():
    up = pd.Series(np.linspace(100, 130, 80) + np.sin(np.arange(80)) * 0.2)
    down = pd.Series(np.linspace(130, 100, 80) + np.sin(np.arange(80)) * 0.2)
    assert (trend_scanning_labels(up, 10, 30)["label"].dropna() == 1).all()
    assert (trend_scanning_labels(down, 10, 30)["label"].dropna() == -1).all()


def test_last_rows_have_no_label_and_horizon_is_bounded():
    prices = synthetic(periods=300, seed=2)["close"]
    lab = trend_scanning_labels(prices, 10, 40)
    assert lab["label"].iloc[-9:].isna().all() and lab["label"].iloc[:-9].notna().all()
    assert lab["best_window"].dropna().between(10, 40).all()


def test_label_only_uses_future_inside_its_own_window():
    """Changing prices beyond i + max_window must not change the label at i."""
    prices = synthetic(periods=300, seed=2)["close"]
    base = trend_scanning_labels(prices, 10, 40)
    tampered = prices.copy()
    tampered.iloc[150:] *= 3.0
    after = trend_scanning_labels(tampered, 10, 40)
    pd.testing.assert_frame_equal(base.iloc[:110], after.iloc[:110])  # 110 + 40 <= 150


def test_invalid_windows_rejected():
    with pytest.raises(ValueError):
        trend_scanning_labels(pd.Series(np.arange(50.0)), 2, 10)
    with pytest.raises(ValueError):
        trend_scanning_labels(pd.Series(np.arange(50.0)), 20, 10)


def test_forward_and_next_bar_returns():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    assert forward_return(close, 2).iloc[0] == pytest.approx(0.21)
    assert next_bar_return(close).iloc[1] == pytest.approx(0.10)
    assert np.isnan(next_bar_return(close).iloc[-1])


def test_meta_labels_only_on_long_bars():
    long_mask = np.array([True, False, True, True])
    fwd = np.array([0.010, 0.020, 0.001, 0.004])
    mask, labels = make_meta_labels(long_mask, fwd, threshold=0.003)
    assert mask.tolist() == [True, False, True, True]
    assert labels.tolist() == [1, 0, 1]  # 0.010 and 0.004 clear 30 bps; 0.001 does not
