# Audit of the April 2026 coursework notebook

This repository started as a five-person coursework notebook
([`notebooks/archive/coursework_2026-04_original.ipynb`](../notebooks/archive/coursework_2026-04_original.ipynb)).
Before turning it into a research codebase, the code was re-read line by line against one
question: **could a live system have computed this number at that time?** The answer was "no" in
several places. This page lists each finding, its likely direction of bias, the fix, and the test
that now guards it. It exists because knowing how a backtest can fool you is most of the job.

> **Consequence:** the performance figures in the original notebook (reproduced at the bottom for
> the record) must not be cited. The numbers in this repository come from the corrected pipeline.

## Findings

Severity: **A** = look-ahead that can inflate results, **B** = methodology / reporting error,
**C** = claim not supported by the code. "Cell" refers to the archived notebook.

| # | Sev. | Finding (cell) | Likely effect | Fix in this repo | Guard |
|---|---|---|---|---|---|
| 1 | A | **HMM states smoothed over the whole series** (5). `hmm.predict` / `predict_proba` use Viterbi / forward-backward on 2018-2021, so a 2021 regime is informed by later 2021 data. | Inflates regime features | `forward_filter`: forward pass only, from the fitted parameters | `test_filter_equals_smoother_on_last_step_only`, `test_filter_is_causal` |
| 2 | A | **Daily regime mapped onto earlier hours** (5). `resample('D').last()` stamps day D at 00:00 with the 23:00 close, then forward-fills onto D's own hours: at 10:00 the feature already knows the 23:00 close. | Inflates regime features | Daily values become available the next day (`to_hourly(causal=True)`) | `test_daily_values_become_available_the_next_day`, `test_leaky_hmm_inference_fails_the_same_check` |
| 3 | A | **Early stopping on the fold being predicted** (11). `validation_data=(X_val_f, y_val_f)` + `restore_best_weights` picks the epoch that is best *on the OOF fold itself*; those OOF predictions then build the meta-labels. | Optimistic OOF | Early stopping on a tail of the training block only | `expanding_window_oof` docstring + `test_expanding_window_oof_is_purged_and_forward_only` |
| 4 | A | **Meta threshold scored on training rows** (12). The calibrated forest is fitted on `X_train_meta`, then the "validation" threshold search runs on `X_train_meta[-20%:]`, rows it was trained on. Result: "precision val 1.000". | Meaningless threshold | Chronological hold-out the model has not seen, plus a minimum-support rule | `test_meta_model_learns_and_is_scored_out_of_sample`, `test_threshold_ignores_tiny_support` |
| 5 | A | **Volatility targeting reference taken on the test set** (17): `vol_median = nanmedian(test_vol_24h)`. | Look-ahead (sign unclear) | Median of the *training* period | `vol_scalar` contract, `test_vol_scalar_is_bounded_and_uses_the_given_reference` |
| 6 | A | **Trailing vol includes the return being earned** (17). `rolling(24).std()` at row *i* contains return *i*, which the position at *i* earns. | Look-ahead (sign unclear) | One timing convention: decide at close *d*, earn bar *d+1* | `test_signals_before_the_cutoff_ignore_the_future` |
| 7 | A/B | **Meta-model features not aligned with the decision** (12). In training the meta-model reads `X[row i]`, which includes `close[i]`, while the position earns `return[i]` (one bar ahead). At test time it reads `X_test[:n]`, i.e. row *j* instead of row *j + 24*: features 24 hours stale. Neither reached the reported numbers (see #10) but they would have. | Latent look-ahead / train-test mismatch | Single convention: features at the decision row, same row at train and test (#6) | `test_sequences_end_at_the_decision_row`, `test_signals_before_the_cutoff_ignore_the_future` |
| 8 | A | **No purge between overlapping labels.** Trend-scanning labels look up to 40 h ahead, yet train/validation and train/test boundaries are abutting. | Mild optimism | Purge `max_window` bars at every boundary (`TimeSeriesSplit(gap=...)`, train/test embargo) | `test_expanding_window_oof_is_purged_and_forward_only`, `test_dataset_purges_labels_that_reach_into_the_test_period` |
| 9 | B | **"Stationary" features that are not** (5). `crossover_*`, `atr_14`, `macd*` were in USD, and `obv_signal` in coins. Their 2021 values sit far outside the 2018-2020 range the scaler saw, yet they are among the top MDI/PFI features. | Out-of-distribution inputs | Expressed relative to price / average volume | `test_features_are_scale_free` |
| 10 | C | **The meta-model never trades** (12, 17). The final cell prints `meta trop restrictif -> fallback LSTM long-only`. "Meta sizing" is therefore identical to "LSTM long-only" in every column, but the write-up credits "le filtrage strict du méta-modèle". | Misattribution | Fallback is explicit (`fallback_to_primary` flag and a banner in `summary.md`) | `test_pipeline_runs_and_writes_a_complete_report` |
| 11 | B | **Vacuous persistence filter** (17). `hmm_persistence` is `A[s,s]` of the *current state*, and every diagonal is ≥ 0.957, so `persistence > 0.90` is always true: the "confirmed bear" cap was just "state == bear". | Misleading description | Cap on the filtered probability P(bear) > 0.9, which does measure confidence | `test_bear_cap_leaves_nan_warmup_untouched` |
| 12 | B | **Sortino computed from the std of negative returns only** (17), which gave Sortino < Sharpe for sparse strategies. | Wrong ratio | Target downside deviation | `test_sortino_uses_target_downside_deviation` |
| 13 | B | **Stress test without costs and with a drawdown that ignores the initial capital** (17). | Flattering stress numbers | Same engine, same costs, drawdown measured from 1.0 | `test_max_drawdown_counts_a_loss_on_the_very_first_bar` |
| 14 | B | "Win rate" was an hourly hit ratio; "AvgHold" was the mean length of runs of *identical* positions (2.2 h for the vol-targeted variant). | Misleading labels | Per-trade win rate and trade length; hourly hit ratio kept under its real name | `test_trade_accounting` |
| 15 | C | Text says 50 features (there are 49), 8 families (9), folds trained on "20% → 80%" (they train on 1/6 → 5/6). | Documentation | Corrected | `test_feature_contract` |
| 16 | C | **Conclusions stronger than the evidence.** OOF macro-F1 was 0.477 ± 0.035 (below 0.5 in 4 of 5 folds, while the final model scored 0.548 on 2021), yet the write-up concludes the model "a réussi à extraire des patterns généralisables". Six strategy variants were compared on one bull year and the best Sharpe (1.63, "Meta + HMM") was highlighted; with the meta-model inactive that variant is really "LSTM long-only with a bear cap". And the crash-test text says the strategy sat in cash "plus de la moitié du temps" while the table shows 53% invested. | Over-claiming | Neutral, data-generated `summary.md`; limitations documented | `docs/methodology.md` §11 |

The HMM probabilities were the top features by both MDI and PFI in the coursework (`hmm_prob_1`
first by MDI; `hmm_prob_2` and `hmm_prob_0` in the PFI top three). That is consistent with the
leak in #1-#2 but is not proof of it, which is why the ablation below exists.

## What was kept

Design choices from the coursework that hold up and are unchanged: separating side from size
(meta-labeling), trend-scanning labels, daily-frequency HMM, probabilities instead of hard regime
labels, cyclical time encoding, class weights instead of SMOTE, expanding-window validation, the
long-only constraint, costs in every backtest, and the MDI vs PFI vs clustered-PFI comparison.

## Measuring the bias instead of asserting it

The leakage in #1-#2 is not removed by deletion: the legacy behaviour is a config switch, so its
effect can be measured with everything else held fixed.

```bash
make run ablation compare      # default (causal) vs configs/ablation_leaky_regimes.yaml
```

## Original coursework results (pre-audit, for the record only)

Test year 2021, 10 bps per change of position, as printed by the archived notebook:

| | Buy & Hold | LSTM long-only | "Meta sizing" | + VolTarget | + HMM | + Vol + HMM |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 62.0% | 96.0% | 96.0% | 48.8% | 78.4% | 52.1% |
| Sharpe | 0.99 | 1.28 | 1.28 | 1.06 | 1.63 | 1.30 |
| Max drawdown | -54.8% | -41.8% | -41.8% | -38.1% | -21.6% | -21.3% |
| Invested | 100% | 48% | 48% | 48% | 48% | 48% |

Classification on 2021: accuracy = macro-F1 = 0.548, precision of long calls 0.573. Expanding-window
OOF macro-F1: 0.477 ± 0.035. Items #1-#8 mean these figures are not a reliable estimate of live
performance; the direction of the bias is known for #1-#4 (optimistic) but its size is not, and
only a re-run of the corrected pipeline on the same data can say how much survives.
