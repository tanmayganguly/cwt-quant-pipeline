"""
Module 4 — Matrix Output Generator

Given the concatenated out-of-sample predictions from the walk-forward
report, produce:

  1. A heatmap of the best-scoring permutation for each
     (hour_of_day x weekday) cell, based on mean PREDICTED pnl —
     this is the actionable "which strategy parameters to run when" output.
  2. A companion heatmap of the actual out-of-sample pnl achieved by
     that recommended choice, so the recommendation can be sanity-checked
     against reality (not just the model's own opinion of itself).
  3. An equity curve comparing:
       - Model-selected portfolio: at each (weekday, hour) cell, take the
         actual OOS pnl of whichever permutation the model recommended
         for that cell, aggregated chronologically across all folds.
       - Baseline: a single fixed default permutation, run consistently,
         same trades/time period.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import SETTINGS, OUTPUTS_DIR
from .walk_forward import WalkForwardReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DEFAULT_BASELINE_PERMUTATION = "P_conservative"


def build_recommendation_matrix(oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_perm_matrix: weekday x hour -> recommended permutation (string)
      best_score_matrix: weekday x hour -> that permutation's mean predicted pnl
    Recommendation is based on mean PREDICTED pnl per (weekday, hour, permutation)
    cell, computed purely from out-of-sample rows (never in-sample fits).
    """
    grouped = (
        oos.groupby(["weekday", "hour_of_day", "permutation"])["predicted_pnl"]
        .mean()
        .reset_index()
    )
    idx = grouped.groupby(["weekday", "hour_of_day"])["predicted_pnl"].idxmax()
    best = grouped.loc[idx].set_index(["weekday", "hour_of_day"])

    hours = sorted(oos["hour_of_day"].unique())
    weekdays = sorted(oos["weekday"].unique())

    perm_matrix = pd.DataFrame(index=weekdays, columns=hours, dtype=object)
    score_matrix = pd.DataFrame(index=weekdays, columns=hours, dtype=float)

    for (wd, hr), row in best.iterrows():
        perm_matrix.loc[wd, hr] = row["permutation"]
        score_matrix.loc[wd, hr] = row["predicted_pnl"]

    perm_matrix.index = [WEEKDAY_NAMES[i] for i in perm_matrix.index]
    score_matrix.index = [WEEKDAY_NAMES[i] for i in score_matrix.index]
    return perm_matrix, score_matrix


def plot_recommendation_heatmap(perm_matrix: pd.DataFrame, score_matrix: pd.DataFrame,
                                 out_path=None) -> str:
    out_path = out_path or (OUTPUTS_DIR / "recommendation_matrix.png")

    perms = sorted(pd.unique(perm_matrix.values.ravel()))
    perms = [p for p in perms if p is not None and not (isinstance(p, float) and np.isnan(p))]
    perm_to_int = {p: i for i, p in enumerate(perms)}
    numeric = perm_matrix.map(lambda p: perm_to_int.get(p, np.nan))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = sns.color_palette("Set2", n_colors=max(len(perms), 1))
    sns.heatmap(
        numeric.astype(float), annot=perm_matrix.values, fmt="", cmap=cmap,
        cbar=False, linewidths=0.6, linecolor="white", ax=ax,
        annot_kws={"fontsize": 8, "rotation": 0},
    )
    ax.set_title("Recommended Strategy Permutation by Hour-of-Day x Weekday\n"
                 "(highest mean predicted P&L, out-of-sample)", fontsize=11)
    ax.set_xlabel("Hour of day (exchange-local)")
    ax.set_ylabel("Weekday")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved recommendation heatmap to %s", out_path)
    return str(out_path)


def build_equity_curves(oos: pd.DataFrame,
                         perm_matrix: pd.DataFrame,
                         baseline_permutation: str = DEFAULT_BASELINE_PERMUTATION) -> pd.DataFrame:
    """
    Model-selected curve: for each OOS trade, check what the model would
    have recommended for that trade's (weekday, hour) cell; if the trade's
    actual permutation matches the recommendation, its actual pnl counts
    toward the model-selected portfolio for that slot.
    Baseline curve: actual pnl of trades that used `baseline_permutation`,
    over the same period, for a fair like-for-like comparison.
    """
    wd_lookup = {name: i for i, name in enumerate(WEEKDAY_NAMES)}
    rec_lookup = {}
    for wd_name in perm_matrix.index:
        for hr in perm_matrix.columns:
            val = perm_matrix.loc[wd_name, hr]
            if isinstance(val, str):
                rec_lookup[(wd_lookup[wd_name], hr)] = val

    oos = oos.copy()
    oos["recommended_permutation"] = oos.apply(
        lambda r: rec_lookup.get((r["weekday"], r["hour_of_day"])), axis=1
    )
    oos["timestamp"] = pd.to_datetime(oos["timestamp"])

    model_trades = oos[oos["permutation"] == oos["recommended_permutation"]].sort_values("timestamp")
    baseline_trades = oos[oos["permutation"] == baseline_permutation].sort_values("timestamp")

    # Multiple trades can share an identical timestamp value, so aggregate
    # pnl per timestamp before cumulative-summing (a plain set_index would
    # otherwise leave duplicate index labels, which breaks the later
    # DataFrame join of the two curves).
    model_curve = model_trades.groupby("timestamp")["pnl"].sum().sort_index().cumsum()
    baseline_curve = baseline_trades.groupby("timestamp")["pnl"].sum().sort_index().cumsum()

    curves = pd.DataFrame({
        "model_selected_equity": model_curve,
        "baseline_equity": baseline_curve,
    }).sort_index().ffill().fillna(0)

    logger.info(
        "Equity curves built: model-selected uses %d trades (final=%.1f), "
        "baseline ('%s') uses %d trades (final=%.1f)",
        len(model_trades), model_curve.iloc[-1] if len(model_curve) else 0,
        baseline_permutation, len(baseline_trades),
        baseline_curve.iloc[-1] if len(baseline_curve) else 0,
    )
    return curves


def plot_equity_curves(curves: pd.DataFrame, out_path=None) -> str:
    out_path = out_path or (OUTPUTS_DIR / "equity_curve.png")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curves.index, curves["model_selected_equity"], label="Model-selected portfolio", linewidth=2)
    ax.plot(curves.index, curves["baseline_equity"], label="Baseline (single default permutation)",
            linewidth=2, linestyle="--")
    ax.set_title("Out-of-Sample Equity Curve: Model-Selected vs Baseline Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved equity curve chart to %s", out_path)
    return str(out_path)


def performance_ratios(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Sharpe, Sortino, Max Drawdown on a P&L series (per-trade, resampled to daily)."""
    if returns.empty:
        return {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": float("nan")}

    daily = returns.resample("1D").sum().fillna(0)
    mean, std = daily.mean(), daily.std(ddof=0)
    downside_std = daily[daily < 0].std(ddof=0)

    sharpe = (mean / std) * np.sqrt(periods_per_year) if std > 0 else float("nan")
    sortino = (mean / downside_std) * np.sqrt(periods_per_year) if downside_std and downside_std > 0 else float("nan")

    cum = daily.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_drawdown = drawdown.min()

    return {"sharpe": float(sharpe), "sortino": float(sortino), "max_drawdown": float(max_drawdown)}


def generate_all_outputs(report: WalkForwardReport) -> dict:
    oos = report.oos_predictions
    perm_matrix, score_matrix = build_recommendation_matrix(oos)
    heatmap_path = plot_recommendation_heatmap(perm_matrix, score_matrix)

    curves = build_equity_curves(oos, perm_matrix)
    equity_path = plot_equity_curves(curves)

    model_returns = curves["model_selected_equity"].diff().fillna(curves["model_selected_equity"])
    baseline_returns = curves["baseline_equity"].diff().fillna(curves["baseline_equity"])

    return {
        "perm_matrix": perm_matrix,
        "score_matrix": score_matrix,
        "heatmap_path": heatmap_path,
        "equity_curve_path": equity_path,
        "curves": curves,
        "model_ratios": performance_ratios(model_returns),
        "baseline_ratios": performance_ratios(baseline_returns),
    }
