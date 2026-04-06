"""Command-line entry point: ``sysml download | run | demo | compare``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from sysml.config import load_config


def _cmd_run(args: argparse.Namespace) -> None:
    from sysml.pipeline import run_pipeline
    from sysml.reporting import save_run

    cfg = load_config(args.config, args.set)
    out = save_run(run_pipeline(cfg), figures=not args.no_figures)
    print(f"Done. Results in {out}/  (start with summary.md)")


def _cmd_demo(args: argparse.Namespace) -> None:
    """Run the full pipeline on synthetic data: checks the install, says nothing about markets."""
    from sysml.pipeline import run_pipeline
    from sysml.reporting import save_run
    from sysml.synthetic import make_synthetic_ohlcv

    overrides = [
        "run_name=demo_synthetic",
        "primary.type=logreg",
        "split.train_end=2018-12-31",
        "split.test_start=2019-01-01",
        "split.test_end=2019-12-31",
        "backtest.stress_periods=[]",
        "importance.n_repeats=5",
        *(args.set or []),
    ]
    cfg = load_config(args.config, overrides)
    out = save_run(run_pipeline(cfg, ohlcv=make_synthetic_ohlcv()), figures=True)
    print(f"Synthetic demo done. Results in {out}/ (numbers are NOT market evidence)")


def _cmd_download(args: argparse.Namespace) -> None:
    from sysml.download import download_to_csv

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame = download_to_csv(args.symbol, args.coin, args.start, args.end, args.out,
                            base_url=args.base_url)
    print(f"Wrote {len(frame):,} hourly bars ({frame.index[0]} -> {frame.index[-1]}) to {args.out}")


def _cmd_compare(args: argparse.Namespace) -> None:
    rows = {}
    for run in args.runs:
        metrics = json.loads((Path(run) / "metrics.json").read_text())["backtest"]
        for strategy, m in metrics.items():
            rows[(Path(run).name, strategy)] = {
                k: m[k] for k in ("CAGR", "Sharpe", "MaxDD", "Exposure")
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["run", "strategy"]
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sysml", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the pipeline from a config")
    run.add_argument("-c", "--config", default="configs/default.yaml")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override a config value, e.g. --set primary.type=logreg")
    run.add_argument("--no-figures", action="store_true")
    run.set_defaults(func=_cmd_run)

    demo = sub.add_parser("demo", help="run on synthetic data (no download needed)")
    demo.add_argument("-c", "--config", default=None)
    demo.add_argument("--set", action="append", metavar="KEY=VALUE")
    demo.set_defaults(func=_cmd_demo)

    dl = sub.add_parser("download", help="fetch hourly candles from Binance public data")
    dl.add_argument("--symbol", default="BTCUSDT")
    dl.add_argument("--coin", default="BTC", help="value written to the `coin` column")
    dl.add_argument("--start", default="2018-01-01")
    dl.add_argument("--end", default="2023-12-31")
    dl.add_argument("--out", default="data/raw/btc_1h.csv")
    dl.add_argument("--base-url", default="https://data-api.binance.vision")
    dl.set_defaults(func=_cmd_download)

    cmp_ = sub.add_parser("compare", help="side-by-side metrics of several result folders")
    cmp_.add_argument("runs", nargs="+")
    cmp_.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
