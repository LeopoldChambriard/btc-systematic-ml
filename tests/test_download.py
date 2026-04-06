import pandas as pd

from sysml.data import load_ohlcv, validate_ohlcv
from sysml.download import HOUR_MS, fetch_klines, klines_to_frame

T0 = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)


def _kline(i):
    ts = T0 + i * HOUR_MS
    return [ts, "100.0", "101.0", "99.0", "100.5", "12.5", ts + HOUR_MS - 1, "0", 10, "0", "0", "0"]


def _fake_server(total_bars, page_size):
    """Mimics the klines endpoint: returns at most `page_size` bars from startTime."""
    calls = []

    def get_json(url):
        calls.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        first = (start - T0) // HOUR_MS
        return [_kline(i) for i in range(first, min(first + page_size, total_bars))]

    return get_json, calls


def test_pagination_covers_the_whole_range_without_gaps_or_duplicates():
    get_json, calls = _fake_server(total_bars=24 * 5, page_size=50)
    rows = fetch_klines("BTCUSDT", "2021-01-01", "2021-01-05", get_json=get_json, pause=0)
    frame = klines_to_frame(rows)
    assert len(calls) == 3 and len(frame) == 24 * 5
    assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_frame_matches_the_project_csv_schema_and_round_trips():
    import tempfile
    from pathlib import Path

    rows = [_kline(i) for i in range(48)] + [_kline(10)]  # a duplicate row must be dropped
    frame = klines_to_frame(rows, coin="BTC")
    assert list(frame.columns) == ["coin", "open", "high", "low", "close", "volume"]
    assert frame.index.name == "date" and len(frame) == 48
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.csv"
        frame.to_csv(path)
        loaded = load_ohlcv(path, asset="BTC")
    assert validate_ohlcv(loaded)["rows"] == 48
