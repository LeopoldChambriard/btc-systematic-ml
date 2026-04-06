import numpy as np
import pandas as pd
import pytest
from helpers import synthetic

from sysml.config import SplitConfig
from sysml.data import end_of_day, split_masks, validate_ohlcv


def test_validate_accepts_clean_data():
    report = validate_ohlcv(synthetic(periods=500))
    assert report["rows"] == 500 and report["n_gaps"] == 0


def test_validate_rejects_corrupt_data():
    df = synthetic(periods=100)

    dup = pd.concat([df, df.iloc[[5]]]).sort_index()
    with pytest.raises(ValueError):
        validate_ohlcv(dup)

    nan = df.copy()
    nan.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        validate_ohlcv(nan)

    inverted = df.copy()
    inverted.iloc[4, inverted.columns.get_loc("high")] = inverted.iloc[4]["low"] * 0.5
    with pytest.raises(ValueError):
        validate_ohlcv(inverted)


def test_validate_reports_gaps():
    df = synthetic(periods=100).drop(synthetic(periods=100).index[10:15])
    assert validate_ohlcv(df)["longest_gap_hours"] == 6.0


def test_split_dates_are_inclusive_end_of_day():
    idx = pd.date_range("2020-12-30", "2021-01-02 23:00", freq="h")
    train, test = split_masks(idx, SplitConfig("2020-12-31", "2021-01-01", "2021-01-01"))
    assert idx[train][-1] == pd.Timestamp("2020-12-31 23:00")
    assert idx[test][0] == pd.Timestamp("2021-01-01 00:00")
    assert idx[test][-1] == pd.Timestamp("2021-01-01 23:00")
    assert not (train & test).any()


def test_end_of_day():
    assert end_of_day("2021-03-04") == pd.Timestamp("2021-03-04 23:59:59.999999999")
