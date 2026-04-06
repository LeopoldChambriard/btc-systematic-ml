"""Performance metrics for an hourly return series (crypto trades 24/7: 8,760 periods per year)."""

from __future__ import annotations

import numpy as np
import pandas as pd

INVESTED_THRESHOLD = 0.01  # a position above 1% of capital counts as "invested"


def max_drawdown(net: pd.Series) -> float:
    """Worst peak-to-trough fall of the equity curve, which starts at 1 before the first bar."""
    equity = np.concatenate([[1.0], (1.0 + net).cumprod().to_numpy()])
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def downside_deviation(net: pd.Series, periods_per_year: int) -> float:
    """sqrt(E[min(r, 0)^2]) annualised: target-downside deviation with a 0% target.

    (The standard deviation of only the negative returns, which the coursework used, ignores how
    often losses occur and makes Sortino look worse than Sharpe for sparse strategies.)
    """
    return float(np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2)) * np.sqrt(periods_per_year))


def _trade_blocks(invested: np.ndarray) -> np.ndarray:
    """Integer id for each maximal run of consecutive invested bars (0 elsewhere)."""
    start = invested & ~np.concatenate([[False], invested[:-1]])
    return np.where(invested, np.cumsum(start), 0)


def performance_metrics(
    net: pd.Series,
    position: pd.Series,
    gross: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    cost_rate: float,
    periods_per_year: int,
) -> dict[str, float]:
    """``gross`` is ``position * return`` before costs; ``net = gross - costs``."""
    n = len(net)
    years = n / periods_per_year
    equity_end = float((1.0 + net).prod())
    cagr = equity_end ** (1.0 / years) - 1.0 if equity_end > 0 and years > 0 else -1.0
    ann_return = float(net.mean() * periods_per_year)
    vol = float(net.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    dd_dev = downside_deviation(net, periods_per_year)
    mdd = max_drawdown(net)

    invested = (position.to_numpy() > INVESTED_THRESHOLD)
    blocks = _trade_blocks(invested)
    n_trades = int(blocks.max()) if len(blocks) else 0

    trade_returns = []
    for b in range(1, n_trades + 1):
        idx = np.where(blocks == b)[0]
        gross_ret = float((1.0 + gross.iloc[idx]).prod() - 1.0)
        # Entry/rebalance cost inside the block plus the exit back to flat.
        t_cost = (turnover.iloc[idx].sum() + position.iloc[idx[-1]]) * cost_rate
        trade_returns.append(gross_ret - t_cost)
    trade_returns = np.asarray(trade_returns)

    inv_net = net[invested]
    gains = inv_net[inv_net > 0].sum()
    losses = -inv_net[inv_net < 0].sum()

    return {
        "CAGR": cagr,
        "AnnReturn": ann_return,
        "Vol": vol,
        "Sharpe": ann_return / vol if vol > 0 else 0.0,
        "Sortino": ann_return / dd_dev if dd_dev > 0 else 0.0,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else 0.0,
        "Exposure": float(invested.mean()),
        "NbTrades": n_trades,
        "TradeWinRate": float((trade_returns > 0).mean()) if n_trades else 0.0,
        "AvgTradeHours": float(invested.sum() / n_trades) if n_trades else 0.0,
        "HourlyHitRatio": float((inv_net > 0).mean()) if len(inv_net) else 0.0,
        "ProfitFactor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "TurnoverPerYear": float(turnover.sum() / years) if years > 0 else 0.0,
        "TotalCost": float(costs.sum()),
        "FinalEquity": equity_end,
        "Bars": n,
    }
