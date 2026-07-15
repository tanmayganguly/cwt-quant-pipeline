"""
Module 2 — Quantitative Feature Engineering & Data Preparation

Responsibilities:
  1. Clean raw trade logs: dedupe near-simultaneous fills, filter by account.
  2. Build time-based features (hour, weekday).
  3. Join against macro_events to build proximity-to-macro-event features
     (time since last high-impact release, and the surprise magnitude of
     that release).
  4. Persist the cleaned+featurized trades to the `trades` table.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from .config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.1 Cleaning
# ---------------------------------------------------------------------------

def deduplicate_trades(df: pd.DataFrame, window_ms: int | None = None) -> pd.DataFrame:
    """
    Collapse near-simultaneous fills into a single logical trade.

    Real execution logs frequently split one intended order into several
    partial fills a few hundred milliseconds apart. For strategy-level
    P&L modeling we want one row per *decision*, not per fill, so we
    group consecutive trades on the same (account, symbol, direction,
    permutation) that land within `window_ms` of each other and merge
    them: quantities sum, price becomes the quantity-weighted average,
    pnl sums.
    """
    window_ms = window_ms or SETTINGS.dedup_window_ms
    df = df.sort_values(["account_id", "symbol", "direction", "permutation", "timestamp"]).copy()

    group_keys = ["account_id", "symbol", "direction", "permutation"]
    df["_gap_ms"] = (
        df.groupby(group_keys)["timestamp"]
        .diff()
        .dt.total_seconds()
        .mul(1000)
    )
    # A new "cluster" starts whenever the gap exceeds the window (or is NaN,
    # i.e. first row in the group).
    df["_new_cluster"] = (df["_gap_ms"].isna()) | (df["_gap_ms"] > window_ms)
    df["_cluster_id"] = df.groupby(group_keys)["_new_cluster"].cumsum()

    agg = (
        df.groupby(group_keys + ["_cluster_id"], as_index=False)
        .apply(lambda g: pd.Series({
            "timestamp": g["timestamp"].iloc[0],  # first fill's timestamp
            "quantity": g["quantity"].sum(),
            "price": np.average(g["price"], weights=g["quantity"]),
            "pnl": g["pnl"].sum(),
        }), include_groups=False)
    )
    # re-attach group keys (lost with include_groups=False in some pandas versions)
    agg = agg.reset_index(drop=True)
    keys_df = df.groupby(group_keys + ["_cluster_id"], as_index=False).size()[group_keys + ["_cluster_id"]]
    merged = pd.concat([keys_df.reset_index(drop=True), agg], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    n_before, n_after = len(df), len(merged)
    logger.info(
        "Deduplication (%dms window): %d fills -> %d logical trades (%.1f%% collapsed)",
        window_ms, n_before, n_after, 100 * (1 - n_after / n_before),
    )
    return merged.sort_values("timestamp").reset_index(drop=True)


def filter_accounts(df: pd.DataFrame, allowed_accounts: list[str] | None = None,
                     min_trades: int = 30) -> pd.DataFrame:
    """
    Account-level filtering: drop accounts with too few trades to be
    statistically meaningful, and optionally restrict to an explicit
    allow-list (e.g. excluding demo/test accounts).
    """
    if allowed_accounts:
        before = len(df)
        df = df[df["account_id"].isin(allowed_accounts)]
        logger.info("Account allow-list filter: %d -> %d rows", before, len(df))

    counts = df["account_id"].value_counts()
    keep = counts[counts >= min_trades].index
    dropped = set(counts.index) - set(keep)
    if dropped:
        logger.info("Dropping low-activity accounts (<%d trades): %s", min_trades, sorted(dropped))
    return df[df["account_id"].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.2 Time-based features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(SETTINGS.exchange_tz)
    df["timestamp"] = ts
    df["hour_of_day"] = ts.dt.hour
    df["weekday"] = ts.dt.dayofweek  # 0=Monday
    df["weekday_name"] = ts.dt.day_name()
    df["is_month_end_week"] = ts.dt.is_month_end | (ts.dt.day >= 25)
    return df


# ---------------------------------------------------------------------------
# 2.3 Macro proximity features
# ---------------------------------------------------------------------------

def load_macro_events(db_path=None) -> pd.DataFrame:
    db_path = db_path or SETTINGS.db_path
    conn = sqlite3.connect(db_path)
    try:
        macro = pd.read_sql("SELECT * FROM macro_events", conn)
    finally:
        conn.close()
    macro["event_time_utc"] = pd.to_datetime(macro["event_time_utc"], utc=True)
    return macro


def add_macro_proximity_features(trades: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade, find the most recent high-impact macro event that
    occurred *before* the trade (no look-ahead), and compute:
      - minutes_since_last_macro_event
      - last_macro_surprise_score (actual vs forecast, standardized)
      - within_30min_of_macro (binary flag — high-vol window)
    Uses a merge_asof for an efficient, leakage-safe backward join.
    """
    trades = trades.copy()
    trades_utc = trades["timestamp"].dt.tz_convert("UTC")

    high_impact = macro[macro["is_high_impact"] == 1].sort_values("event_time_utc").copy()
    if high_impact.empty:
        logger.warning("No high-impact macro events found; proximity features will be null.")
        trades["minutes_since_last_macro_event"] = np.nan
        trades["last_macro_surprise_score"] = 0.0
        trades["within_30min_of_macro"] = 0
        return trades

    left = pd.DataFrame({"timestamp_utc": trades_utc}).sort_values("timestamp_utc")
    merged = pd.merge_asof(
        left,
        high_impact[["event_time_utc", "surprise_score", "event_name"]],
        left_on="timestamp_utc",
        right_on="event_time_utc",
        direction="backward",  # only look at events that already happened
    )
    merged = merged.reindex(left.index)  # merge_asof reorders; restore original order
    merged = merged.sort_index()

    # Recover original (unsorted) trade order alignment
    order = trades_utc.sort_values().index
    merged.index = order
    merged = merged.sort_index()

    trades["minutes_since_last_macro_event"] = (
        (trades_utc.values - merged["event_time_utc"].values) / np.timedelta64(1, "m")
    )
    trades["last_macro_surprise_score"] = merged["surprise_score"].fillna(0.0).values
    trades["within_30min_of_macro"] = (
        trades["minutes_since_last_macro_event"].le(30)
        & trades["minutes_since_last_macro_event"].ge(0)
    ).astype(int)

    logger.info(
        "Macro proximity features added. %.1f%% of trades fall within 30min of a high-impact release.",
        100 * trades["within_30min_of_macro"].mean(),
    )
    return trades


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_feature_table(raw_trades: pd.DataFrame, db_path=None) -> pd.DataFrame:
    db_path = db_path or SETTINGS.db_path

    cleaned = deduplicate_trades(raw_trades)
    cleaned = filter_accounts(cleaned)
    cleaned = add_time_features(cleaned)

    macro = load_macro_events(db_path)
    featurized = add_macro_proximity_features(cleaned, macro)

    conn = sqlite3.connect(db_path)
    try:
        to_write = featurized.copy()
        to_write["timestamp"] = to_write["timestamp"].astype(str)
        to_write.to_sql("trades", conn, if_exists="replace", index=False)
        conn.commit()
        logger.info("Wrote %d feature rows to trades table at %s", len(to_write), db_path)
    finally:
        conn.close()

    return featurized


if __name__ == "__main__":
    from .synthetic_trades import generate_synthetic_trades

    raw = generate_synthetic_trades()
    feats = build_feature_table(raw)
    print(feats.head())
    print(f"\nColumns: {list(feats.columns)}")
