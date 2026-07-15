"""
Central configuration for the pipeline.

All tunables live here so the rest of the codebase never hardcodes
paths, windows, or thresholds inline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # --- Apify ---
    apify_api_token: str = os.getenv("APIFY_API_TOKEN", "")
    apify_actor_id: str = os.getenv(
        "APIFY_ACTOR_ID", "pintostudio/economic-calendar-data-investing-com"
    )
    macro_lookback_days: int = int(os.getenv("MACRO_LOOKBACK_DAYS", "180"))

    # --- Database ---
    db_path: Path = DATA_DIR / "pipeline.db"

    # --- Data cleaning ---
    dedup_window_ms: int = 500  # trades within this window on the same
    # account/symbol/direction are treated as one fill and collapsed

    # --- Walk-forward validation ---
    train_window_days: int = 30
    test_window_days: int = 7
    step_days: int = 7  # how far the window slides each fold

    # --- Timezone ---
    exchange_tz: str = "America/New_York"  # trading log timestamps are
    # assumed to be exchange-local (EST/EDT); macro calendar is normalized
    # to UTC then converted to the same zone before joining

    random_seed: int = 42


SETTINGS = Settings()
