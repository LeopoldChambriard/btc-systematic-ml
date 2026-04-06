"""Sliding-window sequence construction for the sequence models."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_sequences(x: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack trailing windows: sample ``d`` is ``x[d - seq_len + 1 : d + 1]``.

    Returns ``(n - seq_len + 1, seq_len, n_features)``; sample ``j`` corresponds to row
    ``j + seq_len - 1`` of ``x``. The window *ends at* the decision row, so it only contains
    information available at the decision time.
    """
    if len(x) < seq_len:
        raise ValueError("Fewer rows than seq_len")
    windows = sliding_window_view(x, seq_len, axis=0)  # (n - L + 1, F, L)
    return np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)
