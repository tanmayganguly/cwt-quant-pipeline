# CrowdWisdomTrading — Quant Data Scientist Intern Assignment

A Python pipeline that ingests historical trading logs and a macroeconomic
event calendar, engineers time- and macro-proximity features, trains a
walk-forward-validated model to predict per-trade P&L, and outputs an
hour-of-day x weekday recommendation matrix plus an equity curve comparing
the model-selected strategy portfolio against a fixed baseline.

## Quick start

```bash
git clone <this-repo>
cd cwt-quant-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set APIFY_API_TOKEN (see "Reproducing the Apify scrape" below)

# Full run, offline macro fixture + synthetic trades (no credentials needed):
python -m src.main

# Full run with a LIVE Apify scrape for the macro calendar:
python -m src.main --live-macro
```

Outputs land in `outputs/`:
- `recommendation_matrix.png` — heatmap of best permutation per (weekday, hour)
- `equity_curve.png` — model-selected vs. baseline out-of-sample equity curve

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Reproducing the Apify scrape

This project uses the Apify actor **`pintostudio/economic-calendar-data-investing-com`**
to pull the macro calendar (CPI, FOMC, NFP, GDP, PMI, etc.) for the last N days.

To reproduce:
1. Create/log into an Apify account and grab your personal API token from
   **Console → Settings → Integrations**.
2. Put it in `.env` as `APIFY_API_TOKEN=...` (see `.env.example`).
3. `python -m src.pipeline.macro_scraper` (or `python -m src.main --live-macro`).

**Note on credentials:** I'm not emailing a live token — tokens are personal
secrets, not reproducibility artifacts. The actor ID, input schema, and full
scraping code are in `src/pipeline/macro_scraper.py`; running it with your
own token reproduces the exact same table. If a grader needs a temporary
token to verify the run directly, I'm happy to provide a short-lived one on
request rather than sending a permanent credential over email.

If you don't have an Apify account handy, `python -m src.main` (no flag)
uses a deterministic, seeded offline fixture with the identical schema, so
the rest of the pipeline can be exercised end-to-end without it.

## Architecture

```
src/
  pipeline/
    config.py          # all tunables (windows, paths, timezone) in one place
    macro_scraper.py    # Module 1 — Apify ingestion + offline fallback
    synthetic_trades.py # stand-in trade log generator (swap for real logs)
    features.py          # Module 2 — cleaning, dedup, time + macro features
    walk_forward.py      # Module 3 — walk-forward validation + XGBoost model
    matrix_output.py     # Module 4 — heatmap + equity curve + ratios
  main.py                 # end-to-end orchestration
tests/
  test_pipeline.py        # dedup correctness + no-leakage fold construction
data/
  pipeline.db             # SQLite (macro_events, trades tables)
outputs/
  recommendation_matrix.png
  equity_curve.png
```

### Swapping in real trading logs

`generate_synthetic_trades()` in `synthetic_trades.py` is a clearly-marked
stand-in. Replace the call in `src/main.py` (STEP 2) with a loader that
returns a DataFrame with this schema and nothing downstream needs to change:

| column | type | notes |
|---|---|---|
| `timestamp` | tz-aware datetime | exchange-local |
| `account_id` | str | |
| `symbol` | str | |
| `direction` | str | `long` / `short` |
| `quantity` | int | |
| `price` | float | |
| `pnl` | float | |
| `permutation` | str | strategy/parameter-set id under test |

## Design notes

### Data cleaning
- **Deduplication**: trades within a 500ms window on the same
  `(account, symbol, direction, permutation)` are collapsed into one logical
  trade (quantity-summed, volume-weighted price, pnl-summed). This mirrors
  partial-fill duplication in real broker logs, and is unit-tested to
  confirm it neither over- nor under-merges (`tests/test_pipeline.py`).
- **Account filtering**: accounts with too few trades to be statistically
  meaningful are dropped; an explicit allow-list is also supported.
- **Timezone handling**: trade timestamps are treated as exchange-local
  (`America/New_York`); macro event timestamps are normalized to UTC at
  ingestion. The join between them (`merge_asof`, backward direction) is
  done in UTC to avoid any DST-related misalignment, then converted back
  for feature readability.

### Feature engineering
- Time features: `hour_of_day`, `weekday`, `is_month_end_week`.
- Macro proximity: `minutes_since_last_macro_event` (backward-only —
  never looks at a future release), `last_macro_surprise_score`
  (standardized actual-vs-forecast, the "macro surprise" signal called out
  in the brief), `within_30min_of_macro` binary flag for the high-vol window
  right after a release.

### Walk-forward validation — the crucial rule
`walk_forward.py` implements strict expanding/rolling-window walk-forward
validation, not k-fold or a single train/test split:
- Train on a 30-day window, test on the following 7 days, slide forward
  7 days, repeat — 24 folds over ~200 days of synthetic history.
- The model (XGBoost) is **refit from scratch every fold**. No
  cross-fold state, no warm start.
- Categorical encoding (`permutation`) is fit **only on the training fold**;
  categories unseen in training get a sentinel value at test time rather
  than being encoded using knowledge of the full dataset.
- The recommendation matrix and equity curve are built **exclusively from
  concatenated out-of-sample predictions** — never from in-sample fitted
  values — so the "matrix" step can't quietly reintroduce the leakage the
  validation step was designed to prevent.
- Unit tests assert every fold's train window strictly precedes its test
  window with zero overlap.

This follows the walk-forward pattern from the reading list (rolling
train/test windows, no shuffling, never testing on data the model could
have "seen" via a global statistic) rather than the naive backtest
overfitting patterns those articles warn about.

### Matrix + equity curve
- The heatmap picks, per (weekday, hour) cell, the permutation with the
  highest **mean predicted** P&L — an actionable "what to run when" table.
- The equity curve compares: (a) actual OOS pnl of trades that used
  whichever permutation the model recommended for their time slot, vs.
  (b) actual OOS pnl of a single fixed default permutation run throughout
  the same period — an apples-to-apples baseline, not a synthetic control.

## Evaluation report

See `EVALUATION_REPORT.md` for Sharpe/Sortino/Max Drawdown and other
out-of-sample performance metrics from the most recent run.

## A note on the data

Since I didn't have access to CrowdWisdomTrading's actual trading logs, I
built a seeded synthetic log generator with deliberately-injected,
recoverable signal (certain permutations perform better at certain hours)
so I could verify the walk-forward + matrix pipeline actually recovers real
structure rather than just running without crashing. Swapping in real logs
is a one-function change (see above) — nothing else in the pipeline
depends on the data being synthetic.
