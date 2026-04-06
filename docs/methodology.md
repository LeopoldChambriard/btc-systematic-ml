# Methodology

This document explains what each stage of the pipeline does, why it is built that way, and where
it lives in the code. The thread running through all of it is **look-ahead control**: every
fitted object sees training data only, and every decision uses information available at the close
of the bar it is taken on. `docs/leakage_audit.md` lists the places where the original coursework
fell short of that and what changed.

```
OHLCV (1h) ─► features ─► trend-scanning labels ─► primary model ─► meta-model ─► positions ─► backtest
              │  technical (42, scale-free)         (side: up/down)   (size: is the long      │  costs, stress periods
              │  GMM + causal HMM (7)                expanding-window   worth taking?)         │
              └─► clustered feature importance (surrogate RF)                                  └─► metrics
```

## 1. Framing

Hourly crypto returns are very noisy. Following López de Prado (*Advances in Financial Machine
Learning*, 2018), the problem is split in two:

* a **primary model** that only decides the *side* (is the local trend up or down?), and
* a **meta-model** that, given a long signal, decides whether the bet clears the cost hurdle and
  how large it should be.

The book is **long-only**: it holds BTC or cash. Shorting is excluded by design (squeezes make
drawdown control hard), so the primary model's "down" calls simply mean "stay in cash".

## 2. Timing convention

Everything in the code follows one rule, and `tests/test_no_lookahead.py` enforces it end to end:

```
bar d-1          bar d                       bar d+1
   |---------------|---------------------------|
                 close[d] known ──► decide position[d]
                                   └──────── earns close[d+1]/close[d] - 1 ────────►
```

* Features at row `d` use data up to and including `close[d]`.
* A sequence sample *ends at* row `d` (`sysml.models.sequences.make_sequences`).
* The position decided at `d` earns the **next** bar's return (`next_bar_return`), and pays costs on
  the change in position at `d`.
* Anything computed at a coarser frequency (the daily HMM) becomes available only after that
  period is complete, i.e. it is lagged by one day (`sysml.features.regimes.to_hourly`).

Execution is assumed at the close of bar `d`, with a flat proportional fee. That is optimistic for
a real desk and is listed under limitations.

## 3. Data and split

`sysml.data` loads a CSV of hourly bars and validates it (monotonic unique index, no NaN,
positive prices, `high >= low`, gap report). The split is chronological and configured by dates
(`configs/default.yaml`): train through 2020-12-31, test on 2021. The last
`labels.max_window` training bars are **purged**, because their trend-scanning labels look up to 40
hours into the test period.

## 4. Features (`sysml.features`)

49 inputs in 9 families. All are returns, ratios or bounded oscillators, i.e. **scale-free**
(`test_features_are_scale_free` multiplies prices by a constant and checks nothing moves), so a
model fitted on 5k-USD Bitcoin is not out of distribution at 60k.

| Family | Columns | Idea |
|---|---|---|
| momentum | `roc_2`, `roc_5`, `momentum_{3,10,20}`, `momentum_accel` | multi-horizon returns and their acceleration |
| trend | `crossover_*` (3), `price_vs_ma*` (3), `trend_strength` | MA spreads, distance to MAs, OLS-slope t-value of the last 20 closes |
| volatility | `vol_10`, `vol_20`, `vol_ratio`, `atr_14`, `bb_width`, `bb_pos`, `high_low_ratio` | realised vol, range, Bollinger geometry |
| oscillators | `rsi`, `stoch_rsi`, `williams_r`, `cci`, `macd`, `macd_signal`, `macd_hist` | bounded momentum gauges; MACD relative to price |
| volume | `volume_ratio`, `obv_signal` | relative volume; OBV minus its 20-bar mean, in units of average volume |
| autocorrelation | `autocorr_{1,3,5,10}` | rolling (30-bar) return autocorrelation: momentum vs mean reversion |
| lagged returns | `return_lag_{1,2,3,5,10}` | explicit short memory |
| calendar | `hour_sin/cos`, `dow_sin/cos` | cyclical encoding, so 23h and 0h are neighbours |
| regimes | `gmm_prob_*` (3), `hmm_prob_*` (3), `hmm_persistence` | soft market-regime memberships (below) |

Rolling statistics that were Python loops in the coursework (autocorrelation, CCI mean absolute
deviation, OLS t-values) are vectorised with `sliding_window_view` and tested against the original
loops to ~1e-13.

### Regimes (`features/regimes.py`)

* **GMM** (3 components, full covariance) on hourly `(return, vol_20)`. Row-wise, so it cannot look
  ahead by construction. Probabilities (not hard labels) are kept: 60/40 differs from 95/5.
* **HMM** (3 states, full covariance) on **daily** `(log-return, 20-day vol)`. Hourly data makes
  the state flip on microstructure noise; daily data captures persistent cycles (transition
  diagonals ≈ 0.96-0.98 in the coursework).
* Both are fitted, with their scalers, on the training period only.
* The HMM state probabilities are **filtered**, `P(state_t | x_1..x_t)`, using an explicit forward
  pass over the fitted parameters (`forward_filter`). `hmmlearn.predict/predict_proba` smooth over the
  whole series and would let 2021 states depend on later 2021 data. The result is then lagged one
  day before being forward-filled onto the hourly grid.
* States are named from training behaviour only (highest / lowest mean daily return = bull / bear).

## 5. Labels (`sysml.labeling`)

**Trend scanning** (López de Prado, lecture slides *Advances in Financial Machine Learning*, lecture 3/10,
and *Machine Learning for Asset Managers*, 2020). For each bar `t`
and every horizon `h ∈ [10, 40]` hours, regress the future prices `p[t:t+h]` on time and compute the
slope's t-value

```
t(h) = β̂ / se(β̂),     se(β̂) = sqrt( SSR / (h-2) / Σ(x - x̄)² )
```

The label is `sign(t)` at the horizon with the largest `|t|`. The horizon adapts to the market's
speed and `|t|` acts as a built-in signal-to-noise filter. The implementation is vectorised per
horizon and checked against the original double loop.

**Meta-labels.** For bars where the primary model says long, label 1 if the forward 3-hour return
exceeds 30 bps (the cost hurdle), else 0.

## 6. Primary model (`sysml.models`)

* Input: a window of the last `seq_len = 24` standardised feature vectors (one trading day).
* **LSTM** (`models/lstm.py`): LSTM(128, sequences) → BatchNorm → Dropout(0.3) → LSTM(64) →
  Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid); binary cross-entropy, Adam (5e-4), early
  stopping and LR halving on plateau.
* **Logistic regression** (`models/primary.py`) on `[last step, window mean]` as the baseline
  and as the TensorFlow-free path. Both implement the same `fit / predict_proba` interface.
* **Class imbalance:** `class_weight='balanced'`, recomputed on every training block. SMOTE is
  deliberately avoided: interpolating inside a 24-hour window creates sequences that never existed.
* **Validation:** expanding-window out-of-fold predictions (`models/validation.py`), 5 folds with
  train = 1/6, 2/6 ... 5/6 of the sample, validated on the next sixth, **purged** by `max_window`
  bars between train and validation. Early stopping uses a tail of the training block, never the
  fold being predicted.

The OOF predictions feed the meta-model, so each meta-label comes from a primary model that had not
seen that bar.

## 7. Meta-model (`models/meta.py`)

Random forest (300 trees, depth 5, balanced classes) wrapped in isotonic calibration with
time-ordered folds, because the probability *is* the position size. The decision threshold is
chosen by maximising precision on a **chronological hold-out the model was not fitted on**, subject
to a minimum number of surviving signals (otherwise the best "precision" is a handful of lucky
calls). If the resulting filter invests less than 10% of the time, the pipeline falls back to the
primary signal **and says so** (`fallback_to_primary` in the summary); the meta rows of the backtest
are then identical to the primary row and carry no information.

## 8. Strategy variants (`backtest/strategies.py`)

| Name | Position at bar `d` |
|---|---|
| Buy & Hold | 1 |
| SMA 50/200 | 1 if SMA(50) > SMA(200) at the close, else 0 (naive trend baseline) |
| Primary long-only | 1 if P(up) > 0.5 |
| Meta sizing | meta probability if primary long and probability > threshold, else 0 |
| Meta + VolTarget | meta signal × clip(σ_ref/σ_24h, 0.2, 3), capped at 1.5×. σ_ref = training-period median, σ_24h = trailing vol at the decision close |
| Meta + HMM cap | meta signal capped at 20% when the filtered P(bear) > 0.9 |
| Meta + Vol + HMM | both overlays |

Positions above 1 are free leverage: no funding cost is modelled.

## 9. Backtest (`sysml.backtest`)

Vectorised, hourly, 8,760 periods per year. Cost = `cost_bps × |Δposition|` (default 10 bps, the
book starts flat so the first entry pays). Metrics: CAGR, annualised volatility, Sharpe
(`mean/std`, rf = 0), Sortino (target downside deviation, 0% target), max drawdown on the equity
curve including the initial 1.0, Calmar, exposure, number of trades, per-trade win rate, average
trade length, hourly hit ratio, profit factor, turnover and total cost. Stress periods re-run
every strategy on a sub-window from a flat start, **with costs**, using the same engine.

## 10. Feature importance (`sysml.importance`)

The LSTM is a black box, so importance is measured on a **surrogate** random forest trained to
predict the same labels. Three views are compared:

* **MDI**: free, in-sample, biased towards high-variance continuous features.
* **PFI**: drop in out-of-sample macro-F1 when one feature is shuffled; diluted by collinearity.
* **Clustered PFI**: features grouped by Ward clustering on `1 − |ρ|` (8 clusters) and shuffled
  *jointly* with one permutation, which credits shared information to the group.

These describe what a tree finds predictive, not what the LSTM uses.

## 11. Limitations and next steps

* One asset, one training window, **one test year per run**. Several strategy variants are compared
  on the same year, so the best-looking one is selected with hindsight; no deflated Sharpe or
  multiple-testing correction yet.
* No walk-forward retraining; the model is frozen at the end of 2020.
* Execution at the bar close with a flat fee: no slippage, no market impact, no latency, no funding.
* The scaler is fitted once on the whole training block, not per CV fold (minor, conservative to
  fix).
* OHLCV only. On-chain flows and sentiment are natural orthogonal inputs.
* Next experiments, already supported by configs: test on 2022 (`configs/eval_2022_bear.yaml`),
  LSTM vs logistic baseline (`configs/baseline_logreg.yaml`), size of the HMM look-ahead bias
  (`configs/ablation_leaky_regimes.yaml`).

## References

* M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018: meta-labeling, purged
  and embargoed cross-validation, MDI / MDA feature importance.
* M. López de Prado, *Machine Learning for Asset Managers*, Cambridge University Press, 2020:
  trend scanning, clustered feature importance.
* M. López de Prado, *Advances in Financial Machine Learning, Lecture 3/10* (SSRN 3257419):
  where trend scanning was first presented.
* L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech
  recognition", *Proceedings of the IEEE* 77(2), 1989: forward (filtering) vs forward-backward
  (smoothing) recursions.
