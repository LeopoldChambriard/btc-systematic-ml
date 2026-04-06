# Data

No market data is committed to this repository (exchange data licensing, and a 54k-row CSV adds
nothing to review). Everything under `data/raw/` is git-ignored.

## Expected format

A CSV with a datetime first column and hourly bars, UTC, in the exact schema the code reads:

```
date,coin,open,high,low,close,volume
2018-01-01 01:00:00,BTC,13791.4,13804.9,13680.0,13680.0,5.922596
```

`coin` is optional (used to filter when several assets share a file); `volume` is in base-asset
units. The loader (`sysml.data.load_ohlcv`) and `validate_ohlcv` reject duplicated or unsorted
timestamps, NaNs, non-positive prices and `high < low`, and warn on gaps longer than
`data.max_gap_hours`.

## Getting it

```bash
make data    # = sysml download --symbol BTCUSDT --start 2018-01-01 --end 2023-12-31
```

This pulls hourly klines from Binance's public market-data endpoint (no API key) into
`data/raw/btc_1h.csv`. Timestamps are bar open times.

The original April 2026 coursework used a different hourly BTC/USD file (2018-01-01 to 2024,
exchange not recorded in the notebook). Different venues quote slightly different prices and
volumes, so expect numbers to move a little even with an identical pipeline. To use your own file,
place it at `data/raw/btc_1h.csv` or point `data.path` at it:

```bash
sysml run --set data.path=/path/to/my_btc_1h.csv
```

## Coverage needed

| Experiment | Needs data through |
|---|---|
| `configs/default.yaml` (train 2018-20, test 2021) | 2021-12-31 |
| `configs/eval_2022_bear.yaml` (test 2022) | 2022-12-31 |
