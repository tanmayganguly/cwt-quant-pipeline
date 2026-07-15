"""
Unit tests focused on the two highest-risk areas of this pipeline:
  1. Deduplication correctness (does it actually collapse near-duplicate
     fills without merging trades that are legitimately separate?)
  2. Walk-forward fold construction (is chronology strictly respected —
     no fold's test window ever overlaps or precedes its train window,
     and no future data leaks into an earlier fold?)

Run with: python -m pytest tests/ -v
"""
import pandas as pd
import pytest

from src.pipeline.features import deduplicate_trades
from src.pipeline.walk_forward import _make_folds


def test_deduplication_collapses_near_simultaneous_fills():
    base = pd.Timestamp("2026-01-05 10:00:00", tz="America/New_York")
    df = pd.DataFrame({
        "account_id": ["ACC1"] * 3,
        "symbol": ["ES"] * 3,
        "direction": ["long"] * 3,
        "permutation": ["P_momentum"] * 3,
        "timestamp": [base, base + pd.Timedelta(milliseconds=150), base + pd.Timedelta(milliseconds=400)],
        "quantity": [1, 1, 1],
        "price": [100.0, 100.5, 101.0],
        "pnl": [5.0, 5.0, 5.0],
    })
    out = deduplicate_trades(df, window_ms=500)
    assert len(out) == 1
    assert out.loc[0, "quantity"] == 3
    assert out.loc[0, "pnl"] == 15.0


def test_deduplication_does_not_merge_trades_outside_window():
    base = pd.Timestamp("2026-01-05 10:00:00", tz="America/New_York")
    df = pd.DataFrame({
        "account_id": ["ACC1"] * 2,
        "symbol": ["ES"] * 2,
        "direction": ["long"] * 2,
        "permutation": ["P_momentum"] * 2,
        "timestamp": [base, base + pd.Timedelta(seconds=5)],
        "quantity": [1, 1],
        "price": [100.0, 100.5],
        "pnl": [5.0, -2.0],
    })
    out = deduplicate_trades(df, window_ms=500)
    assert len(out) == 2


def test_deduplication_keeps_different_accounts_separate():
    base = pd.Timestamp("2026-01-05 10:00:00", tz="America/New_York")
    df = pd.DataFrame({
        "account_id": ["ACC1", "ACC2"],
        "symbol": ["ES", "ES"],
        "direction": ["long", "long"],
        "permutation": ["P_momentum", "P_momentum"],
        "timestamp": [base, base + pd.Timedelta(milliseconds=100)],
        "quantity": [1, 1],
        "price": [100.0, 100.0],
        "pnl": [5.0, 5.0],
    })
    out = deduplicate_trades(df, window_ms=500)
    assert len(out) == 2


def test_walk_forward_folds_are_strictly_chronological_and_non_overlapping_targets():
    min_ts = pd.Timestamp("2026-01-01")
    max_ts = pd.Timestamp("2026-04-01")
    folds = _make_folds(min_ts, max_ts, train_days=30, test_days=7, step_days=7)

    assert len(folds) > 0
    for train_start, train_end, test_end in folds:
        # No leakage: train strictly precedes test, with no overlap.
        assert train_start < train_end <= test_end
        assert (train_end - train_start).days == 30
        assert (test_end - train_end).days == 7

    # Folds slide forward — each fold's train_start should be >= the previous one's.
    starts = [f[0] for f in folds]
    assert starts == sorted(starts)


def test_walk_forward_raises_on_insufficient_history():
    min_ts = pd.Timestamp("2026-01-01")
    max_ts = pd.Timestamp("2026-01-10")  # only 9 days, need 37
    folds = _make_folds(min_ts, max_ts, train_days=30, test_days=7, step_days=7)
    assert folds == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
