"""Persist a run: config snapshot, metric tables, signals, figures and a human-readable summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sysml.pipeline import PipelineResult, make_figures

PCT_COLS = ["CAGR", "AnnReturn", "Vol", "MaxDD", "Exposure", "TradeWinRate", "HourlyHitRatio",
            "TotalCost"]
FMT = {c: "{:.1%}" for c in PCT_COLS} | {
    "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}", "ProfitFactor": "{:.2f}",
    "AvgTradeHours": "{:.1f}", "TurnoverPerYear": "{:.0f}", "FinalEquity": "{:.3f}",
    "NbTrades": "{:.0f}", "Bars": "{:.0f}",
}
SUMMARY_COLS = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "Exposure", "NbTrades",
                "TradeWinRate", "TurnoverPerYear", "TotalCost"]


def to_markdown(df: pd.DataFrame, columns: list[str], index_name: str = "Strategy") -> str:
    """Tiny dependency-free markdown table with per-column number formats."""
    header = "| " + " | ".join([index_name, *columns]) + " |"
    rule = "|" + "|".join(["---"] + ["---:"] * len(columns)) + "|"
    lines = [header, rule]
    for name, row in df.iterrows():
        cells = [FMT.get(c, "{:.3f}").format(row[c]) if np.isfinite(row[c]) else "n/a"
                 for c in columns]
        lines.append("| " + " | ".join([str(name), *cells]) + " |")
    return "\n".join(lines)


def build_summary(res: PipelineResult) -> str:
    cfg, ds, meta = res.config, res.dataset, res.meta_info
    sig = res.signals
    clf, folds = res.classification, res.oof_folds
    mode = (
        "causal (filtered, lagged one day)"
        if cfg.regimes.causal
        else "LEGACY (smoothed, same-day) - leaks"
    )

    out = [
        f"# Run summary: `{cfg.run_name}`",
        "",
        f"- **Data**: {ds.data_report['rows']:,} hourly bars, {ds.data_report['start']} to "
        f"{ds.data_report['end']} ({ds.data_report['n_gaps']} gap(s))",
        f"- **Train**: up to {cfg.split.train_end} ({len(ds.train_rows):,} samples after purge) | "
        f"**Test**: {sig.index[0]:%Y-%m-%d} to {sig.index[-1]:%Y-%m-%d} ({len(sig):,} bars)",
        f"- **Primary model**: `{cfg.primary.type}`, window {cfg.primary.seq_len}h | "
        f"**HMM inference**: {mode}",
        "",
        "## Directional signal",
        "",
        f"- Expanding-window OOF macro-F1 ({cfg.primary.cv_splits} folds, purged): "
        f"**{folds['f1_macro'].mean():.3f} +/- {folds['f1_macro'].std():.3f}** "
        f"(per fold: {', '.join(f'{v:.3f}' for v in folds['f1_macro'])})",
        f"- Test accuracy / macro-F1: **{clf['primary']['accuracy']:.3f} / "
        f"{clf['primary']['f1_macro']:.3f}** | precision of long calls: "
        f"**{clf['primary']['precision_long']:.3f}** "
        f"(base rate of up-trend bars: {clf['primary']['share_true_long']:.3f})",
        "",
        "## Meta-model",
        "",
        f"- Status: `{meta['status']}` | meta-labels: {meta.get('n_meta_samples', 0):,} "
        f"(base rate {meta.get('meta_base_rate', float('nan')):.3f})",
    ]
    if "holdout_precision" in meta:
        out.append(
            f"- Threshold {meta['threshold']:.2f} chosen on a chronological hold-out: precision "
            f"{meta['holdout_precision']:.3f} on {meta['holdout_support']} signals "
            f"(hold-out base rate {meta['holdout_base_rate']:.3f})"
        )
    if meta.get("fallback_to_primary"):
        out += [
            "",
            "> **The meta filter was too restrictive and the pipeline fell back to the primary "
            "signal.** 'Meta sizing' and 'Primary long-only' are therefore identical rows, and any "
            "gain attributed to the meta-model would be spurious.",
        ]
    out += [
        "",
        f"## Backtest (net of {cfg.backtest.cost_bps:.0f} bps per unit turnover, hourly bars)",
        "",
        to_markdown(res.metrics_table, SUMMARY_COLS),
    ]
    for name, results in res.stress.items():
        table = pd.DataFrame({r.label: r.metrics for r in results}).T
        out += ["", f"### Stress period: {name}", "",
                to_markdown(table, ["CAGR", "Sharpe", "MaxDD", "Exposure", "FinalEquity"])]
    out += [
        "",
        "## Notes",
        "",
        "- Positions above 1.0 (volatility targeting) are free leverage: no funding cost.",
        "- Slippage and market impact are not modelled beyond the proportional fee.",
        "- Single train/test split on one asset: every number is one draw, not a distribution.",
    ]
    return "\n".join(out) + "\n"


def save_run(res: PipelineResult, root: str | Path | None = None, figures: bool = True) -> Path:
    cfg = res.config
    out_dir = Path(root or cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    res.metrics_table.to_csv(out_dir / "metrics.csv")
    (out_dir / "metrics.json").write_text(json.dumps({
        "classification": res.classification,
        "meta": {k: (v if not isinstance(v, np.floating) else float(v))
                 for k, v in res.meta_info.items()},
        "oof_f1_mean": float(res.oof_folds["f1_macro"].mean()),
        "oof_f1_std": float(res.oof_folds["f1_macro"].std()),
        "backtest": {r.label: r.metrics for r in res.backtests},
    }, indent=2, default=float))
    for name, results in res.stress.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        pd.DataFrame({r.label: r.metrics for r in results}).T.to_csv(out_dir / f"stress_{slug}.csv")
    res.oof_folds.to_csv(out_dir / "oof_folds.csv", index=False)
    res.signals.to_csv(out_dir / "signals.csv.gz")
    imp = res.importance
    imp["mdi"].rename("mdi").to_csv(out_dir / "importance_mdi.csv")
    imp["pfi"].to_csv(out_dir / "importance_pfi.csv")
    imp["clustered_pfi"].to_csv(out_dir / "importance_clustered_pfi.csv", index=False)
    if figures:
        make_figures(res, out_dir)
    (out_dir / "summary.md").write_text(build_summary(res))
    return out_dir
