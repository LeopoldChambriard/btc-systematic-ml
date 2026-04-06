"""Trend-scanning labels (López de Prado: AFML lecture 3/10, SSRN 3257419, and *Machine Learning for
Asset Managers*, 2020).

For every bar ``t`` we regress the *future* prices ``p[t : t+h]`` on time for each horizon
``h`` in ``[min_window, max_window]`` and keep the horizon whose slope has the largest
absolute t-value. The sign of that t-value is the label (+1 / -1). The horizon therefore
adapts to the speed of the market instead of being fixed in advance, and the t-value is a
built-in signal-to-noise filter.

Labels look *forward* by construction. They are targets only; any train/validation split must
purge (or embargo) the last ``max_window`` bars before the boundary. ``sysml.pipeline`` does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sysml.features.technical import rolling_ols_tvalue


def trend_scanning_labels(
    prices: pd.Series, min_window: int = 10, max_window: int = 40
) -> pd.DataFrame:
    """Return ``label`` (+1/-1), ``t_value`` and ``best_window`` indexed like ``prices``.

    Rows that cannot host a full ``min_window`` window at the end of the series are NaN.
    Ties between horizons resolve to the shortest window.
    """
    if min_window < 3 or max_window < min_window:
        raise ValueError("Need 3 <= min_window <= max_window")
    p = prices.to_numpy(dtype=float)
    n = len(p)
    best_abs = np.zeros(n)
    best_t = np.zeros(n)
    best_h = np.full(n, min_window)

    for h in range(min_window, min(max_window, n) + 1):
        t = rolling_ols_tvalue(p, h)  # t[i] describes p[i : i + h]
        m = len(t)
        better = np.abs(t) > best_abs[:m]
        best_abs[:m] = np.where(better, np.abs(t), best_abs[:m])
        best_t[:m] = np.where(better, t, best_t[:m])
        best_h[:m] = np.where(better, h, best_h[:m])

    out = pd.DataFrame(
        {
            "label": np.where(best_t >= 0, 1.0, -1.0),
            "t_value": best_t,
            "best_window": best_h.astype(float),
        },
        index=prices.index,
    )
    out.iloc[max(n - min_window + 1, 0) :] = np.nan  # no complete window available
    return out
