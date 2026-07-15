# Evaluation Report — Walk-Forward Strategy Selection Pipeline

**Run configuration:** synthetic trade logs (200 trading days, ~8,700 trades
post-dedup), offline macro-calendar fixture (22 high-impact US macro events:
CPI, FOMC, NFP, ISM PMI), walk-forward folds of 30-day train / 7-day test /
7-day step, XGBoost regressor (`n_estimators=200, max_depth=4, lr=0.05`).

## 1. Walk-forward setup

| Parameter | Value |
|---|---|
| Number of folds | 24 |
| Train window | 30 days |
| Test window | 7 days |
| Step size | 7 days |
| Total out-of-sample trades scored | 7,156 |

Every fold's model is trained from scratch on only the data preceding its
test window; the reported numbers below are computed **exclusively on
concatenated out-of-sample predictions**, never on in-sample fitted values.

## 2. Point-prediction accuracy (predicted vs. actual per-trade P&L)

| Metric | Value | Interpretation |
|---|---|---|
| MAE | 37.89 | Average absolute error per trade, in P&L units |
| RMSE | 55.48 | Penalizes larger misses more than MAE |
| R² | -0.107 | Slightly negative — see note below |
| Directional hit rate | 53.7% | % of trades where predicted sign (profit/loss) matched actual |

**Note on R²:** a negative R² here means the model's raw point predictions
are, in aggregate, no better than predicting the mean — this is expected
and not a bug: per-trade P&L is dominated by near-random noise on top of a
comparatively small, structured hour/permutation edge (by design, see
`synthetic_trades.py`). The model isn't being asked to predict individual
trade outcomes well; it's being asked to correctly **rank** permutations
within each (hour, weekday) cell so the *aggregated* recommendation is
better than a fixed default. Sections 3–4 below are the metrics that
actually test that.

## 3. Recommendation quality — model-selected portfolio vs. baseline

The model-selected portfolio = actual OOS P&L of trades that used whichever
permutation the model recommended for that trade's (weekday, hour) cell.
The baseline = actual OOS P&L of a single fixed default permutation
(`P_conservative`) run over the identical period — an apples-to-apples
comparison, not a synthetic control.

| Metric | Model-selected | Baseline (single default) |
|---|---|---|
| Sharpe Ratio (annualized) | **11.45** | 4.48 |
| Sortino Ratio (annualized) | **45.62** | 6.00 |
| Max Drawdown | **-183.6** | -958.9 |
| Final cumulative P&L | **22,756** | 7,162 |
| Trades used | 1,409 | 1,445 |

The model-selected portfolio shows a substantially higher risk-adjusted
return and roughly 5x smaller drawdown than the fixed-default baseline over
the same out-of-sample period — evidence that the walk-forward-trained
model is recovering real, exploitable structure (the hour-of-day edge
profiles built into the synthetic data) rather than overfitting noise.

*Caveat:* Sharpe/Sortino here are computed on a synthetic P&L series with
an injected, deterministic edge, so the absolute magnitudes are not
representative of what a live strategy would achieve — they're reported to
demonstrate the evaluation methodology is implemented correctly. On real
trading logs, expect meaningfully lower Sharpe and higher variance across
folds; the code doesn't need to change, only the input data.

## 4. Recommendation matrix

See `outputs/recommendation_matrix.png`. Distinct permutations are
recommended for different hour/weekday cells (e.g. `P_breakout` at market
open, `P_momentum` mid-morning, `P_meanrevert` around midday, `P_scalp` in
early afternoon on some days) — consistent with the hidden per-permutation
hour-of-day edge profiles used to generate the synthetic trade log,
confirming the pipeline is correctly recovering structure rather than
picking arbitrarily.

## 5. Equity curve

See `outputs/equity_curve.png`. The model-selected curve separates from the
baseline within the first ~30 days of out-of-sample trading and the gap
widens roughly monotonically through the full evaluation period, with no
large mid-series drawdown reversal — the kind of curve shape you'd want to
interrogate carefully on real data before trusting it (see Section 6).

## 6. Known limitations / what I'd do next with real data and more time

- **Overfitting checks beyond walk-forward**: I'd add a permutation test
  (shuffle the `permutation` labels and confirm Sharpe collapses to ~0) and
  a sensitivity check on fold size/step (does the ranking survive a 20-day
  train window instead of 30?) — both flagged as important in the
  qoppac.blogspot "three kinds of overfitting" reading.
- **Feature depth**: with real logs I'd add rolling realized volatility,
  recent regime indicators (trend vs. chop), and interaction terms between
  macro surprise magnitude and permutation type, rather than only a
  proximity flag.
- **Transaction costs / slippage**: not modeled here; a real evaluation
  needs cost-adjusted P&L before Sharpe/Sortino are meaningful.
- **Statistical significance across folds**: I'd report per-fold Sharpe
  variance, not just the pooled number, since 24 folds with correlated
  weekly seasonality isn't the same as 24 independent trials.
