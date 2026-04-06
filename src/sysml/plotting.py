"""Figures. Each function draws one figure, saves it to ``path`` and closes it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

BLUE, GREEN, RED, AMBER, GREY = "#378ADD", "#1D9E75", "#E24B4A", "#EF9F27", "#888780"
STRATEGY_COLORS = [GREY, "#444441", BLUE, GREEN, "#9B59B6", "#D85A30", "#E67E22"]


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(x: pd.DataFrame, path) -> None:
    corr = x.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap="RdBu_r", vmin=-1, vmax=1,
        center=0, ax=ax, linewidths=0.1, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation (training period)", fontweight="bold")
    ax.tick_params(labelsize=7)
    _save(fig, path)


def regime_plot(frame: pd.DataFrame, state_names: dict[int, str], path) -> None:
    """Price shaded by the (causal) HMM regime, with the persistence of the current state."""
    sub = frame.dropna(subset=["hmm_regime"])
    palette = {"Bull": GREEN, "Bear": RED, "Neutral": AMBER}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(sub["close"], color=BLUE, lw=1)
    run_id = (sub["hmm_regime"] != sub["hmm_regime"].shift()).cumsum()
    seen = set()
    for _, block in sub.groupby(run_id):
        name = state_names[int(block["hmm_regime"].iloc[0])]
        ax1.axvspan(block.index[0], block.index[-1], color=palette[name], alpha=0.25,
                    label=None if name in seen else name)
        seen.add(name)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.plot(sub["hmm_persistence"], color=GREY, lw=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(stay in state)")
    ax2.grid(alpha=0.3)
    fig.suptitle("HMM regimes (filtered, lagged one day)", fontweight="bold")
    _save(fig, path)


def labels_plot(frame: pd.DataFrame, path, months: int = 6) -> None:
    sub = frame.dropna(subset=["label"])
    sub = sub[sub.index >= sub.index[-1] - pd.DateOffset(months=months)]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(sub["close"], color=BLUE, lw=1.2)
    for lab, color in ((1, GREEN), (-1, RED)):
        pts = sub[sub["label"] == lab]
        ax1.scatter(pts.index, pts["close"], c=color, s=6, alpha=0.7, label=f"{lab:+d}", zorder=3)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel("Price (USD)")
    ax2.plot(sub["t_value"].clip(-60, 60), color=GREY, lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("t-value (clipped)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Trend-scanning labels", fontweight="bold")
    _save(fig, path)


def convergence_plot(history: dict[str, list[float]] | None, path) -> bool:
    if not history or "loss" not in history:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val loss")
    ax.set_title("Primary model convergence", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)
    return True


def importance_plot(
    mdi: pd.Series, pfi: pd.DataFrame, clusters: pd.DataFrame, path, top: int = 15
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.suptitle("Feature importance: three complementary views", fontweight="bold")
    m = mdi.head(top)[::-1]
    axes[0].barh(m.index, m.values, color=BLUE)
    axes[0].set_title("MDI (in-sample, random forest)")
    p = pfi.head(top)[::-1]
    axes[1].barh(p.index, p["importance"], xerr=p["std"], color=BLUE, ecolor="gray", capsize=3)
    axes[1].set_title("PFI (out-of-sample)")
    c = clusters.sort_values("importance")
    colors = [GREEN if v > 0 else RED for v in c["importance"]]
    axes[2].barh(c["cluster"], c["importance"], xerr=c["std"], color=colors, ecolor="gray",
                 capsize=3)
    axes[2].set_title("Clustered PFI (drop in macro-F1)")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.6)
    _save(fig, path)


def confusion_plot(panels: dict[str, tuple[np.ndarray, np.ndarray]], path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    for ax, (title, (y_true, y_pred)) in zip(np.atleast_1d(axes), panels.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(title)
    _save(fig, path)


def backtest_panels(results: list, path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontweight="bold")
    for res, color in zip(results, STRATEGY_COLORS):
        ls = "--" if res.label == "Buy & Hold" else "-"
        axes[0, 0].plot(res.equity.index, res.equity.values, color=color, ls=ls, lw=1.5,
                        label=f"{res.label} (SR {res.metrics['Sharpe']:.2f})")
        peak = np.maximum.accumulate(np.concatenate([[1.0], res.equity.values]))[1:]
        axes[0, 1].plot(res.equity.index, res.equity.values / peak - 1, color=color, lw=1.1,
                        label=res.label)
    axes[0, 0].set(title="Equity (net of costs, base 1)")
    axes[0, 0].axhline(1, color="k", lw=0.5, ls=":")
    axes[0, 1].set(title="Drawdown")
    for ax in axes[0]:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    names = [r.label for r in results]
    sharpes = [r.metrics["Sharpe"] for r in results]
    bars = axes[1, 0].bar(range(len(names)), sharpes, color=STRATEGY_COLORS[: len(names)],
                          width=0.6)
    axes[1, 0].set_title("Sharpe ratio (net of costs)")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    for b, v in zip(bars, sharpes):
        axes[1, 0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")

    x = np.arange(len(names))
    axes[1, 1].bar(x - 0.2, [r.metrics["Calmar"] for r in results], 0.4, label="Calmar", color=BLUE)
    pf = [min(r.metrics["ProfitFactor"], 5.0) for r in results]
    axes[1, 1].bar(x + 0.2, pf, 0.4, label="Profit factor (capped at 5)", color=GREEN)
    axes[1, 1].set_title("Calmar and profit factor")
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xticks(x if ax is axes[1, 1] else range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def stress_plot(period_results: list, name: str, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for res, color in zip(period_results, STRATEGY_COLORS):
        ax.plot(res.equity.index, res.equity.values, color=color, lw=1.8,
                ls="--" if res.label == "Buy & Hold" else "-", label=res.label)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_title(f"Stress period: {name} (net of costs, flat start)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)
