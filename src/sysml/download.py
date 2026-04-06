"""Download hourly OHLCV candles from Binance's public market-data endpoint (no API key).

``data-api.binance.vision`` serves public market data worldwide; ``api.binance.com`` is blocked in
some jurisdictions. Timestamps are bar *open* times in UTC.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

DEFAULT_BASE_URL = "https://data-api.binance.vision"
HOUR_MS = 3_600_000
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def _get_json(url: str) -> list:
    with urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https endpoint
        return json.loads(resp.read().decode())


def fetch_klines(
    symbol: str,
    start: str,
    end: str,
    base_url: str = DEFAULT_BASE_URL,
    interval: str = "1h",
    limit: int = 1000,
    pause: float = 0.2,
    get_json: Callable[[str], list] = _get_json,
) -> list[list]:
    """Page through the klines endpoint between ``start`` and ``end`` (inclusive days, UTC)."""
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    stop = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    rows: list[list] = []
    while cursor < stop:
        query = urlencode({"symbol": symbol, "interval": interval, "startTime": cursor,
                           "endTime": stop - 1, "limit": limit})
        page = get_json(f"{base_url}/api/v3/klines?{query}")
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + HOUR_MS
        if next_cursor <= cursor:  # defensive: never loop forever on a misbehaving server
            break
        cursor = next_cursor
        time.sleep(pause)
    return rows


def klines_to_frame(rows: list[list], coin: str = "BTC") -> pd.DataFrame:
    """Binance kline rows -> the CSV schema used by the project (date, coin, OHLCV)."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["open_time"].astype("int64"), unit="ms"),
            "coin": coin,
            **{c: df[c].astype(float) for c in ("open", "high", "low", "close", "volume")},
        }
    )
    return out.drop_duplicates("date").sort_values("date").set_index("date")


def download_to_csv(
    symbol: str, coin: str, start: str, end: str, out: str, **kwargs
) -> pd.DataFrame:
    frame = klines_to_frame(fetch_klines(symbol, start, end, **kwargs), coin)
    frame.to_csv(out)
    return frame
