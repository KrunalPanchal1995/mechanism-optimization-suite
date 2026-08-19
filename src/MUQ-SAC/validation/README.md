# validation/ — reorganized 10-fold CV / search / offline-replay code

This directory is the modular replacement for `test_build/`'s 19 flat
investigation scripts (`kfold_cv.py`, `kfold_robust_search{,2,3}.py`,
`offline_replay.py`, etc.). The 19 original files are left in place,
byte-unchanged, as the provenance archive — nothing here deletes or
modifies them.

Built and validated here FIRST, per this project's "build → validate →
port" convention; only after every check below passes does the identical
tree get mirrored to `src/MUQ-SAC/validation/` (a separate, later task —
**not done by this pass**, which deliberately does not touch
`src/MUQ-SAC/` at all).

## Layout

```
validation/
  _bootstrap.py       sys.path shim -> the MUQ-SAC root (test_build/ today,
                       src/MUQ-SAC/ once mirrored), resolved from __file__.
                       Also makes src/MUQ-SAC/ importable even while
                       MUQSAC_ROOT is test_build/, READ-ONLY, because
                       VisualAid.py currently exists only there (see
                       _bootstrap.py's own docstring for why).
  config.py            REPO_ROOT / DATA_DIR / results_dir() / figures_dir() /
                       tables_dir() / scratch_dir(), seeds, hyperparameter
                       grids, BLAS thread-pinning helper. Single source of
                       truth for every path and constant used below.
  case_data.py         The ONE shared data-loading module: load_case,
                       load_outlier_pool, load_sigma_and_rxns, load_temps,
                       make_layout, load_final_hyperparams, build_case_pool,
                       make_folds. Replaces 4 duplicated loader blocks and
                       ~7 duplicated inline fold-split recipes.
  merge_hyperparams.py CLI that makes the round1(+)round2(+)round3 hand-merge
                       (previously done ad hoc) reproducible.
  kfold_cv.py           Baseline 10-fold CV of the FINAL production config.
  kfold_cv_bandvalid.py 10-fold CV scored on band-valid test points only.
  kfold_cv_verify.py    Verifies searched hyperparameters on kfold_cv.py's
                       own split (independent of the search split).
  diagnose_worst_rows.py  Row-level diagnosis of worst-fold failures.
  search_history/       The 3-round hyperparameter search lineage + the
                       superseded offline_replay/offline_cv_search scripts +
                       the still-live parametric_study.py. See its own
                       README.md for the round-to-round deltas.
  report_figures/       Figure-drawing scripts (parity plots, band-violation
                       samples, row-generator diagnostics). See its own
                       README.md for the figure <-> script <-> results map.
```

## Data flow / provenance

Every script here reads from the ALREADY-FROZEN result snapshots under
`report_10fold_validation/results/` and `report/results/` (see those
directories' own READMEs for how they were snapshotted from `/tmp` — that
step is out of scope for this task and was completed earlier). Nothing here
recomputes those snapshots by default; running these scripts is how you
INDEPENDENTLY REPRODUCE the numbers already published there, not how you
regenerate the "official" copy (use `--output-dir`/`--results-dir` flags to
point at a scratch location instead of overwriting the frozen snapshot).

## Verification (run against `test_build/`, before any mirror to `src/`)

All 8 gates below were run and passed during this task. Exact commands and
observed numbers:

### 1. Baseline reproduction (`kfold_cv.py`)
```
python3.9 validation/kfold_cv.py \
    --hp-csv report_10fold_validation/results/inputs/Prediction_errors_detailed.csv \
    --results-dir <staging dir>
```
Worst-fold-per-case output: **2.747, 6.715, 1.777, 3.156, 20.265, 2.668,
1.393, 1.519** — exact match to `report_10fold_validation/tables/baseline_10fold.tex`.
JSON diff against `report_10fold_validation/results/01_baseline_10fold/kfold_cv_results.json`
(rel-tol 1e-9, every numeric field): **0 differences**.

### 2. Final reproduction (`kfold_cv_bandvalid.py`)
```
python3.9 validation/kfold_cv_bandvalid.py \
    --hp-json report_10fold_validation/results/07_final_10fold/kfold_final_hyperparams.json \
    --output-dir <staging dir>
```
Worst-fold-valid-per-case: **1.687, 1.607, 1.544, 1.846, 1.289, 1.748, 1.376,
1.118** (max 1.846) — exact match to `report_10fold_validation/tables/final_10fold.tex`.
JSON diff against `report_10fold_validation/results/07_final_10fold/kfold_cv_bandvalid_results.json`
(rel-tol 1e-9): **0 differences**.

### 3. `kfold_cv_verify.py` (both historical runs)
Re-ran with `chosen_hyperparams.json` (round 1) and
`chosen_hyperparams_merged.json` (round1⊕round2) as `--hp-json`. JSON diff
against `report_10fold_validation/results/05_verify_independent_split/{kfold_cv_verify_results.json,kfold_cv_verify2_results.json}`
(rel-tol 1e-9): **0 differences** for both.

### 4. `diagnose_worst_rows.py`
Reproduced overall-worst rows for cases 0, 3, 4, 5. Confirmed the two golden
row/ratio facts exactly: **Case 3 row 609, ratio 26.99**; **Case 0 row 1702,
ratio 27.55** (both from the SAME script; the figure scripts independently
re-confirm these below).

### 5. `merge_hyperparams.py` self-checks
- `merge(round1, round2, --cases 0,3,4,5,7)` vs.
  `chosen_hyperparams_merged.json` (ignoring `_provenance` and
  `search_worst_fold_pct`): **match**.
- `merge(<above output>, round3, --cases 3,4)` vs.
  `kfold_final_hyperparams.json` (same exclusions): **match**.

### 6. `report_figures/*.py` (staged to `/tmp/report_figures_staging/`, NOT
   written into `report_10fold_validation/figures/` — a later pass pixel-diffs
   and promotes them)
- `plot_parity_bad_fold.py`: Case 4 fold 5 before all-points=**20.27%**,
  valid=**1.10%**; after all-points=**64.12%**, valid=**0.91%**. Case 3 fold
  7 before=**3.16%**/**1.15%**, after=**4.43%**/**1.05%**. All 8 numbers
  exact matches to the goldens in `report_10fold_validation/main.tex`.
- `plot_band_violation_samples.py`: Case 3 row 609 ratio=**26.99**; Case 0
  row 1702 ratio=**27.55**. Exact matches.
- `plot_row_generator_diagnostics.py`: both rows matched generator
  **Class B** at distance **0.0** (i.e. exact match, consistent with the
  golden "~1e-9 distance, i.e. exactly" claim).

### 7. Import hygiene (`cd /` first, so cwd never masks a path bug)
```
cd / && python3.9 -c "
import sys; sys.path.insert(0, '.../test_build/validation')
import _bootstrap
import ResponseSurface, IdentifiablePRS, VisualAid, config, case_data, kfold_cv
print(ResponseSurface.__file__)   # -> test_build/ResponseSurface.py
print(IdentifiablePRS.__file__)   # -> test_build/IdentifiablePRS.py
print(VisualAid.__file__)         # -> src/MUQ-SAC/VisualAid.py (see _bootstrap.py docstring)
print(config.__file__)            # -> test_build/validation/config.py
print(kfold_cv.__file__)          # -> test_build/validation/kfold_cv.py (NOT test_build/kfold_cv.py)
"
```
Confirms the reorganized modules resolve to `validation/`, production
modules resolve under the tree being tested (`test_build/`), and the one
exception (`VisualAid`) resolves to its documented, deliberate location.
This is exactly the check that caught (and let us fix) an early bug in this
task: `_bootstrap.add_muqsac_to_path()` originally PREPENDED
`test_build/` to `sys.path`, which shadowed `validation/kfold_cv.py` with
the OLD flat `test_build/kfold_cv.py` (same filename, different tree). Now
it appends instead, so a script's own directory always wins for a
same-named module.

### 8. Syntax / tests
- `ast.parse()` on every new `.py` file: all OK.
- `python3.9 tests/test_doptimal.py` run from `cd /` (foreign cwd): **ALL
  TESTS PASSED** (7/7 test functions; `pytest tests/ -v` also reports
  `7 passed`).

## Known deviations from the plan (see also the task's final report)

- `_bootstrap.py` also makes `src/MUQ-SAC/` importable (read-only) because
  `VisualAid.py` is not currently mirrored into `test_build/` — the
  original `plot_band_violation_samples.py` had two hardcoded
  `sys.path.insert` calls for exactly this reason. This task does not WRITE
  anything under `src/MUQ-SAC/`; it only reads `VisualAid.py` from there,
  same as the original script did.
- `VERIFY_SPLIT_SEED` in `config.py` is an alias for `CV_SPLIT_SEED`
  (`= 2026`), not a distinct third seed — `kfold_cv_verify.py` was found
  (by reading the actual source, per instructions) to deliberately reuse
  `kfold_cv.py`'s own split, not a third one; only the SEARCH split (555)
  differs. See `config.py`'s comment for the full explanation.
