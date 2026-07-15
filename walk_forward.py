"""
Module 3 — Machine Learning Model & Walk-Forward Validation

Core design principle (the "crucial rule" of the assignment): NO fold's
test period may ever be used, directly or indirectly, to influence a
model trained on data that precedes it. Concretely:

  - Folds are strictly chronological: train on [t, t+30d), test on
    [t+30d, t+30d+7d), then slide forward by 7 days and repeat.
  - The model is refit from scratch every fold — no warm-starting across
    folds and no leakage of test-fold statistics into training
    (e.g. no global mean/quantile encoding computed on the full dataset).
  - Categorical target (permutation) is label-encoded using only the
    categories observed in the training fold up to that point.
  - Aggregation (hour-of-day x weekday matrix, equity curve) is only ever
    built from the concatenation of each fold's OUT-OF-SAMPLE predictions,
    never from in-sample fitted values.

This directly follows the walk-forward pattern described in
qoppac.blogspot.com's "three kinds of overfitting" and the walk-forward
CV articles referenced in the assignment brief: expanding/rolling train
window, sliding test window, never shuffled.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "hour_of_day",
    "weekday",
    "is_month_end_week",
    "minutes_since_last_macro_event",
    "last_macro_surprise_score",
    "within_30min_of_macro",
    "quantity",
    "permutation_enc",
]
TARGET_COLUMN = "pnl"


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    predictions: pd.DataFrame  # out-of-sample predictions for this fold


@dataclass
class WalkForwardReport:
    folds: list[FoldResult] = field(default_factory=list)
    oos_predictions: pd.DataFrame = None  # concatenation across all folds


def load_trades(db_path=None) -> pd.DataFrame:
    db_path = db_path or SETTINGS.db_path
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM trades", conn)
    finally:
        conn.close()
    # Stored timestamps carry mixed UTC offsets (EST/EDT across DST
    # transitions in the exchange-local zone), so parse with utc=True to
    # get a common tz-aware series, then keep it in the exchange zone for
    # readability of hour-of-day features downstream.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(SETTINGS.exchange_tz)
    return df


def _make_folds(min_ts: pd.Timestamp, max_ts: pd.Timestamp,
                 train_days: int, test_days: int, step_days: int):
    folds = []
    train_start = min_ts
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > max_ts:
            break
        folds.append((train_start, train_end, test_end))
        train_start = train_start + pd.Timedelta(days=step_days)
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    train_days: int | None = None,
    test_days: int | None = None,
    step_days: int | None = None,
) -> WalkForwardReport:
    train_days = train_days or SETTINGS.train_window_days
    test_days = test_days or SETTINGS.test_window_days
    step_days = step_days or SETTINGS.step_days

    df = df.sort_values("timestamp").reset_index(drop=True)
    min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
    fold_bounds = _make_folds(min_ts, max_ts, train_days, test_days, step_days)

    if not fold_bounds:
        raise ValueError(
            f"Not enough data for even one fold: need >= {train_days + test_days} days, "
            f"have {(max_ts - min_ts).days} days. Generate more synthetic history or "
            f"reduce train_window_days/test_window_days in config.py."
        )

    logger.info("Constructed %d walk-forward folds (train=%dd, test=%dd, step=%dd)",
                len(fold_bounds), train_days, test_days, step_days)

    report = WalkForwardReport(folds=[])
    all_oos = []

    for i, (tr_start, tr_end, te_end) in enumerate(fold_bounds):
        train_mask = (df["timestamp"] >= tr_start) & (df["timestamp"] < tr_end)
        test_mask = (df["timestamp"] >= tr_end) & (df["timestamp"] < te_end)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(train_df) < 50 or len(test_df) == 0:
            logger.info("Fold %d skipped: insufficient rows (train=%d, test=%d)",
                        i, len(train_df), len(test_df))
            continue

        # Label-encode permutation using ONLY categories seen in the
        # training fold — unseen test-time categories get a sentinel value
        # rather than leaking information about the full-dataset category set.
        le = LabelEncoder()
        train_df["permutation_enc"] = le.fit_transform(train_df["permutation"])
        known = set(le.classes_)
        test_df["permutation_enc"] = test_df["permutation"].apply(
            lambda p: le.transform([p])[0] if p in known else -1
        )

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SETTINGS.random_seed,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        fold_preds = test_df[
            ["timestamp", "account_id", "symbol", "permutation", "hour_of_day", "weekday", "pnl"]
        ].copy()
        fold_preds["predicted_pnl"] = preds
        fold_preds["fold_id"] = i

        report.folds.append(FoldResult(
            fold_id=i, train_start=tr_start, train_end=tr_end,
            test_start=tr_end, test_end=te_end,
            n_train=len(train_df), n_test=len(test_df),
            predictions=fold_preds,
        ))
        all_oos.append(fold_preds)

        mae = np.abs(fold_preds["pnl"] - fold_preds["predicted_pnl"]).mean()
        logger.info(
            "Fold %2d | train %s->%s (n=%4d) | test %s->%s (n=%4d) | OOS MAE=%.3f",
            i, tr_start.date(), tr_end.date(), len(train_df),
            tr_end.date(), te_end.date(), len(test_df), mae,
        )

    report.oos_predictions = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    return report


def summarize_oos_performance(report: WalkForwardReport) -> dict:
    """Aggregate out-of-sample accuracy metrics across all folds combined."""
    oos = report.oos_predictions
    if oos.empty:
        return {}

    errors = oos["pnl"] - oos["predicted_pnl"]
    ss_res = (errors ** 2).sum()
    ss_tot = ((oos["pnl"] - oos["pnl"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Directional accuracy: did the model correctly call profit vs loss?
    directional_hit_rate = (np.sign(oos["pnl"]) == np.sign(oos["predicted_pnl"])).mean()

    return {
        "n_folds": len(report.folds),
        "n_oos_trades": len(oos),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "r2": float(r2),
        "directional_hit_rate": float(directional_hit_rate),
    }


if __name__ == "__main__":
    trades = load_trades()
    report = run_walk_forward(trades)
    metrics = summarize_oos_performance(report)
    print("\nOut-of-sample performance across all folds:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
