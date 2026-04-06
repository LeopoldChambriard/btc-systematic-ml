import numpy as np
import pandas as pd
import pytest

from sysml.backtest import (
    apply_bear_cap,
    apply_vol_target,
    max_drawdown,
    meta_sized,
    run_backtest,
    sma_crossover,
    vol_scalar,
)
from sysml.backtest.metrics import downside_deviation


def _series(values):
    return pd.Series(values, index=pd.date_range("2021-01-01", periods=len(values), freq="h"))


def test_position_decided_at_d_earns_the_return_of_bar_d():
    # `returns` are already next-bar returns: position[d] meets returns[d], nothing else.
    r = _series([0.10, -0.10, 0.10])
    res = run_backtest(r, [1.0, 0.0, 1.0], "x", cost_bps=0)
    assert res.equity.iloc[-1] == pytest.approx(1.10 * 1.0 * 1.10)


def test_costs_are_charged_on_turnover_including_the_first_entry():
    r = _series([0.0, 0.0, 0.0, 0.0])
    res = run_backtest(r, [1.0, 1.0, 0.5, 0.0], "x", cost_bps=10)
    # turnover: 1 (enter), 0, 0.5, 0.5 -> 2.0 * 10 bps
    assert res.net_returns.sum() == pytest.approx(-0.0020)
    assert res.metrics["TotalCost"] == pytest.approx(0.0020)


def test_flat_book_has_zero_pnl_and_zero_cost():
    res = run_backtest(_series([0.01, -0.02, 0.03]), [0, 0, 0], "cash")
    assert res.equity.iloc[-1] == 1.0 and res.metrics["NbTrades"] == 0


def test_buy_and_hold_cagr_and_drawdown():
    ppy = 365 * 24
    r = _series([0.001] * ppy)  # exactly one year
    res = run_backtest(r, np.ones(ppy), "bh", cost_bps=0, periods_per_year=ppy)
    assert res.metrics["CAGR"] == pytest.approx(1.001**ppy - 1)
    assert res.metrics["MaxDD"] == 0.0


def test_max_drawdown_counts_a_loss_on_the_very_first_bar():
    assert max_drawdown(_series([-0.10, 0.05])) == pytest.approx(-0.10)
    assert max_drawdown(_series([0.10, -0.20, 0.05])) == pytest.approx(-0.20)


def test_sortino_uses_target_downside_deviation():
    net = _series([0.02, -0.01, 0.03, -0.02, 0.01])
    expected = np.sqrt(np.mean(np.minimum(net.to_numpy(), 0) ** 2)) * np.sqrt(8760)
    assert downside_deviation(net, 8760) == pytest.approx(expected)


def test_trade_accounting():
    r = _series([0.01, 0.02, -0.01, 0.05, -0.03, -0.01, 0.0])
    pos = [1, 1, 0, 0, 1, 1, 0]
    m = run_backtest(r, pos, "t", cost_bps=0).metrics
    assert m["NbTrades"] == 2
    assert m["AvgTradeHours"] == pytest.approx(2.0)
    # trade 1: 1.01 * 1.02 - 1 > 0 ; trade 2: 0.97 * 0.99 - 1 < 0
    assert m["TradeWinRate"] == pytest.approx(0.5)
    assert m["Exposure"] == pytest.approx(4 / 7)


def test_negative_positions_are_rejected():
    with pytest.raises(ValueError):
        run_backtest(_series([0.0, 0.0]), [1.0, -0.5], "short")


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        run_backtest(_series([0.0, 0.0]), [1.0], "bad")


def test_meta_sizing_requires_primary_long_and_confident_meta():
    long_ = np.array([True, True, False, True])
    meta = np.array([0.80, 0.30, 0.90, 0.55])
    out = meta_sized(long_, meta, threshold=0.5)
    assert out.tolist() == [0.80, 0.0, 0.0, 0.55]


def test_vol_scalar_is_bounded_and_uses_the_given_reference():
    s = vol_scalar(np.array([0.001, 0.01, 0.1, 0.0]), vol_ref=0.01, clip_min=0.2, clip_max=3.0)
    assert s.tolist() == [3.0, 1.0, 0.2, 1.0]
    sig = apply_vol_target(np.array([1.0, 1.0, 0.0]), np.array([3.0, 0.5, 3.0]), max_leverage=1.5)
    assert sig.tolist() == [1.5, 0.5, 0.0]


def test_bear_cap_leaves_nan_warmup_untouched():
    out = apply_bear_cap(np.array([1.0, 1.0, 1.0, 0.1]), np.array([np.nan, 0.5, 0.95, 0.99]),
                         threshold=0.9, cap=0.2)
    assert out.tolist() == [1.0, 1.0, 0.2, 0.1]


def test_sma_crossover_is_causal():
    close = pd.Series(np.r_[np.linspace(100, 90, 30), np.linspace(90, 130, 30)])
    sig = sma_crossover(close, 3, 10)
    cut = sma_crossover(close.iloc[:40], 3, 10)
    assert (sig.iloc[:40].to_numpy() == cut.to_numpy()).all()
