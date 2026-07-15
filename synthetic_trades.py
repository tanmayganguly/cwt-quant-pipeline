"""
Synthetic historical trading log generator.

*** THIS IS A STAND-IN FOR REAL BROKER/SIMULATION LOGS. ***
Replace `generate_synthetic_trades()` with a loader for your real export
(CSV/DB) that produces a DataFrame with the same column schema:

    timestamp   (tz-aware, exchange-local)
    account_id
    symbol
    direction        ('long' / 'short')
    quantity
    price
    pnl
    permutation      (the strategy/parameter-set identifier being tested)

The rest of the pipeline (cleaning, features, walk-forward model) only
depends on this schema, not on how the rows were produced — so swapping
in real logs is a one-function change in main.py.

We intentionally bias the synthetic P&L generator so that certain
permutations perform better around certain hours and near high-impact
macro surprises, so the walk-forward model has real, learnable signal
to recover (useful for sanity-checking the pipeline end to end).
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import SETTINGS

logger = logging.getLogger(__name__)

PERMUTATIONS = ["P_conservative", "P_momentum", "P_meanrevert", "P_breakout", "P_scalp"]
SYMBOLS = ["ES", "NQ", "CL", "GC"]


def generate_synthetic_trades(
    n_days: int = 200,
    trades_per_day: int = 60,
    n_accounts: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    seed = seed if seed is not None else SETTINGS.random_seed
    rng = np.random.default_rng(seed)

    end = pd.Timestamp.now(tz=SETTINGS.exchange_tz).normalize()
    start = end - timedelta(days=n_days)
    trading_days = pd.bdate_range(start, end)  # weekdays only, like real markets

    # Each permutation has a hidden hour-of-day "edge" profile — this is
    # the signal the walk-forward model is meant to (partially) recover.
    perm_hour_edge = {
        "P_conservative": {h: 0.15 for h in range(9, 16)},
        "P_momentum": {h: (0.6 if h in (9, 10, 15) else 0.05) for h in range(9, 16)},
        "P_meanrevert": {h: (0.5 if h in (12, 13) else 0.0) for h in range(9, 16)},
        "P_breakout": {h: (0.7 if h == 9 else -0.1) for h in range(9, 16)},
        "P_scalp": {h: 0.2 - 0.02 * abs(h - 12) for h in range(9, 16)},
    }

    rows = []
    for day in trading_days:
        day_trade_count = rng.poisson(trades_per_day)
        for _ in range(day_trade_count):
            hour = int(rng.integers(9, 16))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            ts = day + timedelta(hours=hour, minutes=minute, seconds=second)

            account = f"ACC{rng.integers(1, n_accounts + 1)}"
            symbol = rng.choice(SYMBOLS)
            permutation = rng.choice(PERMUTATIONS)
            direction = rng.choice(["long", "short"])
            quantity = int(rng.integers(1, 6))
            price = round(rng.uniform(100, 5000), 2)

            edge = perm_hour_edge[permutation].get(hour, 0.0)
            noise = rng.normal(0, 1.0)
            pnl = round((edge + noise) * quantity * rng.uniform(8, 20), 2)

            rows.append((ts, account, symbol, direction, quantity, price, pnl, permutation))

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "account_id", "symbol", "direction",
                 "quantity", "price", "pnl", "permutation"],
    )

    # Inject some near-duplicate fills (same account/symbol/direction within
    # a few hundred ms) to exercise the deduplication logic downstream —
    # this mimics partial-fill duplication seen in real broker logs.
    dup_idx = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
    dup_rows = df.loc[dup_idx].copy()
    dup_rows["timestamp"] = dup_rows["timestamp"] + pd.to_timedelta(
        rng.integers(50, 400, size=len(dup_rows)), unit="ms"
    )
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "Generated %d synthetic trade rows (%d days, incl. ~%.0f%% duplicate fills)",
        len(df), n_days, 100 * len(dup_rows) / len(df),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trades = generate_synthetic_trades()
    print(trades.head())
    print(f"\nTotal rows: {len(trades)}")
