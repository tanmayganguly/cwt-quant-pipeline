"""
End-to-end pipeline entry point.

    python -m src.main                 # offline demo (synthetic trades + synthetic macro fixture)
    python -m src.main --live-macro    # real Apify scrape for macro events, synthetic trades
    python -m src.main --live-macro --days 90

To use REAL trading logs instead of synthetic ones, replace the
`generate_synthetic_trades()` call below with your own loader that
returns a DataFrame matching the schema documented in
src/pipeline/synthetic_trades.py — nothing else in the pipeline needs
to change.
"""
from __future__ import annotations

import argparse
import logging

from .pipeline import macro_scraper, features, walk_forward, matrix_output
from .pipeline.synthetic_trades import generate_synthetic_trades
from .pipeline.config import SETTINGS, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(live_macro: bool = False, lookback_days: int | None = None):
    logger.info("=" * 70)
    logger.info("STEP 1/4 — Macro event ingestion")
    logger.info("=" * 70)
    macro_scraper.run(offline=not live_macro, lookback_days=lookback_days)

    logger.info("=" * 70)
    logger.info("STEP 2/4 — Trade log ingestion, cleaning & feature engineering")
    logger.info("=" * 70)
    # SWAP POINT: replace this line with your real trade log loader.
    raw_trades = generate_synthetic_trades(n_days=200)
    feature_table = features.build_feature_table(raw_trades)

    logger.info("=" * 70)
    logger.info("STEP 3/4 — Walk-forward model training & validation")
    logger.info("=" * 70)
    report = walk_forward.run_walk_forward(feature_table)
    metrics = walk_forward.summarize_oos_performance(report)
    logger.info("Aggregate OOS metrics: %s", metrics)

    logger.info("=" * 70)
    logger.info("STEP 4/4 — Matrix output & equity curve generation")
    logger.info("=" * 70)
    outputs = matrix_output.generate_all_outputs(report)

    logger.info("Model portfolio ratios: %s", outputs["model_ratios"])
    logger.info("Baseline portfolio ratios: %s", outputs["baseline_ratios"])
    logger.info("Heatmap saved to: %s", outputs["heatmap_path"])
    logger.info("Equity curve saved to: %s", outputs["equity_curve_path"])
    logger.info("All outputs available in: %s", OUTPUTS_DIR)

    return {"report": report, "metrics": metrics, "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full quant strategy-selection pipeline.")
    parser.add_argument("--live-macro", action="store_true",
                         help="Use a live Apify scrape for macro events instead of the offline fixture.")
    parser.add_argument("--days", type=int, default=None, help="Macro lookback window in days.")
    args = parser.parse_args()
    main(live_macro=args.live_macro, lookback_days=args.days)
