# search_history/ — the 3-round robust hyperparameter search lineage

This directory keeps the 3 successive rounds of `kfold_robust_search{,2,3}.py`
as separate, legible files rather than collapsing them into one script. The
ridge-grid widening, case-subset narrowing, and the round-3 scoring change
are each part of the scientific record of *why* the final hyperparameters
were chosen, not implementation debt to clean away.

It also keeps `offline_replay.py` / `offline_replay_ext.py` and
`offline_cv_search.py` (both superseded by later work) and
`parametric_study.py` (still live — feeds `report/`'s sensitivity tables).

## Round-to-round deltas (read from the actual source files, not secondhand)

All three rounds share the same protocol: for each case, search
`(tolerance, k, order, ridge_alpha)` to minimize the **worst of
`N_FOLDS_SEARCH=10` folds**, on an independent split
(`SEARCH_SPLIT_SEED=555`, different from `kfold_cv.py`'s own evaluation
split `CV_SPLIT_SEED=2026`) — selecting and reporting on the same split
would be circular/optimistic. Selection between candidate `(score, k)` pairs
always uses `IdentifiablePRS._is_better` (prefers materially lower worst-fold
score; among near-ties, prefers lower `k` — simpler model).

### Round 1 — `kfold_robust_search.py`
- **Trigger**: `kfold_cv.py`'s baseline 10-fold sweep found only 3/8 cases
  (2, 6, 7) kept every fold ≤2% MRE at the single-split-selected
  hyperparameters — e.g. Case 4's single-split hold-out was 1.45% but its
  10-fold worst fold was 20.3% at the SAME hyperparameters.
- **Grid**: `TOLERANCES=[1.0,1.1,1.25,1.5,2.0,5.0,10.0,inf]`,
  `K_GRID=[8,10,12,14,16,20]` (capped at 20 — established finding),
  `ORDER_GRID=[2,3]`, **`RIDGE_GRID=[0.0, 1e-7, 1e-5, 1e-3, 1e-2]`**
  (`config.RIDGE_GRID_R1`).
- **Cases searched**: all 8.
- **Outcome**: 5/8 cases (1, 2, 5, 6, and initially 4/7 with different
  values) converged; cases 0, 3, 4, 5, 7 still had marginal (2.1–3.1%)
  worst-fold violations once merged against round 2's improvements for those
  cases (see `chosen_hyperparams.json` vs. `chosen_hyperparams_merged.json`
  in `report_10fold_validation/results/02_robust_search_round1/`).

### Round 2 — `kfold_robust_search2.py`
- **Trigger**: round 1's ridge grid topped out at `1e-2`, not enough
  regularization for cases 0, 3, 4, 5, 7.
- **Grid delta**: same tolerances/k/order; **ridge widened substantially, up
  to 0.5**: `RIDGE_GRID=[0.0, 1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1]`
  (`config.RIDGE_GRID_R2`) — trading bias for variance reduction, the direct
  lever for a worst-fold criterion.
- **Cases searched**: restricted to `CASES_TO_SEARCH=[0, 3, 4, 5, 7]`
  (`config.CASES_TO_SEARCH_R2`) — the cases still failing after round 1.
- **Outcome**: merging round 1 ⊕ round 2 (overlay wins for the 5 searched
  cases — see `validation/merge_hyperparams.py`) produced
  `chosen_hyperparams_merged.json`. Cases 0, 3, 5 landed on the SAME
  hyperparameters round 1 already found (round 2 didn't improve them
  further); cases 4 and 7 improved (case 4: ridge 0.0→0.5, worst-fold
  2.275→2.121; case 7: k 14→20, ridge 0.0→0.3, worst-fold 1.262→1.185). Only
  cases 3 and 4 remained above 2% after this merge.

### Round 3 — `kfold_robust_search3.py`
- **Trigger**: `diagnose_worst_rows.py` (2026-07-30) found every remaining
  worst-fold failure (cases 0, 3, 4, 5) traced to a single band-invalid test
  row — violation ratio 7×–30× the sampler's own physical uncertainty band
  (Case 3 row 609: ratio 26.99; Case 0 row 1702: ratio 27.55; both Class B at
  ~1e-9 distance). That is a row the sampler's own class-B `m=3`
  rejection test should have rejected — an unreliable ground-truth target,
  not a model deficiency to search harder against.
- **Scoring change**: same ridge grid as round 2
  (`config.RIDGE_GRID_R3 = config.RIDGE_GRID_R2`), but each fold's
  VALIDATION rows are now filtered to **band-valid only**
  (`BAND_TOLERANCE_EVAL=1.0`, `IdentifiablePRS.violation_detail` /
  `band_valid_mask`) before scoring — training stays on the full,
  unfiltered pool; only which points COUNT toward the worst-fold metric
  changes.
- **Cases searched**: restricted to `CASES_TO_SEARCH=[3, 4]`
  (`config.CASES_TO_SEARCH_R3`) — the only 2 cases still failing.
- **Outcome**: case 3 → tol=10.0, k=20, ridge=0.0, worst-fold(band-valid)
  1.733%; case 4 → tol=1.0, k=20, ridge=0.5, worst-fold(band-valid) 1.274%.
  Merging this into round1⊕round2's output (overlay wins for cases 3, 4)
  produced `kfold_final_hyperparams.json` — the FINAL production
  hyperparameters, all 8 cases passing (see
  `report_10fold_validation/results/07_final_10fold/`).

**Round 3's band-valid idea is now in production**, not just a one-off
search-time trick: `IdentifiablePRS.fit_prs_with_cv_search` (and
`fit_prs_with_search`) take a `band_tolerance_eval` parameter that applies
the exact same filter — confirmed by grep:

```
$ grep -n "band_tolerance_eval\|fit_prs_with_cv_search" IdentifiablePRS.py
390:                         band_tolerance_eval=None):
402:	band_tolerance_eval: if set (e.g. 1.0), Xval/yval are filtered to
404:	see fit_prs_with_cv_search's docstring for why. Default None preserves
420:	if band_tolerance_eval is not None:
421:		vmask = band_valid_mask(Xval, blocks, band_tolerance_eval)
460:def fit_prs_with_cv_search(ResponseSurfaceClass, X_pool, y_pool, blocks, n_comp,
470:                            band_tolerance_eval=None):
502:	band_tolerance_eval: if set (e.g. 1.0), each fold's VALIDATION rows are
531:	eval_valid_mask = (band_valid_mask(X_pool, blocks, band_tolerance_eval)
532:	                    if band_tolerance_eval is not None else None)
```

## Merging the 3 rounds — `validation/merge_hyperparams.py`

Historically `chosen_hyperparams_merged.json` (round1⊕round2) and
`kfold_final_hyperparams.json` (⊕round3) were produced by hand, with no
producing script — `merge_hyperparams.py` makes that reproducible. Merge
rule: OVERLAY's entry wins outright for every case in `--cases` (or every
case key present in OVERLAY if `--cases` is omitted); other cases keep
BASE's entry. See `merge_hyperparams.py`'s own docstring and the two
self-checks recorded in `validation/README.md`.

## Superseded scripts kept here

- `offline_replay.py` / `offline_replay_ext.py` (byte-identical pair) —
  superseded by the 10-fold CV work in `validation/kfold_cv.py`; kept for
  historical reproducibility of the numbers `report/main.tex`'s Task #3
  section cites.
- `offline_cv_search.py` — superseded by production
  `IdentifiablePRS.fit_prs_with_cv_search` (the same function this script's
  logic was ported into).
- `parametric_study.py` — NOT superseded; still feeds `report/`'s
  sensitivity tables (`sensitivity_k.tex`, `sensitivity_order.tex`,
  `sensitivity_tolerance.tex`, `parametric_best.tex`). Output goes to
  `report/results/parametric_study/`, not `report_10fold_validation/`.
