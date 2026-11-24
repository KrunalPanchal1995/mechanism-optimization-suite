# Merge log — 2026-07-03 — src/MUQ-SAC three-way merge

Merged two external working copies into `src/MUQ-SAC/` (no git, per KP's
instruction — full pre-merge copies of every variant are preserved in the
sibling folders here).

## Sources

| Source | Location | Vintage | Contents |
|--------|----------|---------|----------|
| Current working dir | `src/MUQ-SAC/` | Jul-02 | Outlier-retraining work (first version) |
| pPRS outlier training | `/data/pPRS_OUTLIER_TRAINING/` | Jul-03 17:05 | 7 flat MUQ-SAC files; evolved outlier retraining (`resample_global`, DM-faithful gap filling, extra `target.opt` knobs) |
| SA correlation test | `/data/SA_CORRELATION_TEST/src/` | Jul-03 16:52 | 3 flat files; SA-data-only correlation pipeline with Pearson/Spearman support |

## Lineage findings (what made this merge safe)

* **pPRS files are strict evolutions of current** — every "current-only" diff
  line was an older version of a block that pPRS rewrote/extended. Nothing lost
  by taking them wholesale.
* **SA's `run_script.py` was based on the OLD Jul-01 backup**
  (`src/MUQ-SAC/bkp/run_script.py` + only 5 lines of its own). Copying it
  wholesale would have destroyed the Jul-02/03 outlier-retraining work — its
  small delta was ported instead.
* **API break resolved:** SA's `correlation_analysis.py` removed the
  `within_reaction_rescue=` kwarg from `run_combined_3p_pipeline()` and added
  `method="pearson"|"spearman"`. The merged `run_script.py` call site was
  updated to the new convention.

## Resolution per file (all in `src/MUQ-SAC/`)

| File | Taken from | Extra step |
|------|-----------|-----------|
| `AdaptiveResampler.py` | pPRS (Jul-03) | — (adds `resample_global()`, `uns_cap`/`shuffle_factor`/`method` params) |
| `run_script.py` | pPRS (Jul-03) | **+ SA delta ported**: 3 comment lines + `within_reaction_rescue=True` → `method="pearson"` at the `run_combined_3p_pipeline()` call |
| `correlation_analysis.py` | SA (Jul-03) | — (Pearson/Spearman `method=` support; same function inventory as current) |
| `partial_PRS_system.py` | SA (Jul-03) | — (docstring-only change: correlation-aware rescue now documented as wired) |
| `DesignMatrix.py`, `ResponseSurface.py`, `simulation_manager2_0.py`, `Uncertainty.py` | unchanged | identical in pPRS and current |

## Preserved variants (this folder)

```
current_before_merge/    the 4 affected files exactly as they were in src/MUQ-SAC before the merge
pPRS_OUTLIER_TRAINING/   the 2 differing files from /data/pPRS_OUTLIER_TRAINING
SA_CORRELATION_TEST/     the 3 differing files from /data/SA_CORRELATION_TEST/src
```

To roll back: `cp version_history/2026-07-03_muqsac_merge/current_before_merge/* src/MUQ-SAC/`

## Verification

* `python -m py_compile` passes on all 4 merged files.
* `grep within_reaction_rescue src/ --include='*.py'` → no live callers of the
  removed kwarg remain (only `bkp/` and this snapshot folder).
* Merged `run_script.py` diff vs the pPRS variant = exactly the 4-line SA delta.
* `src/Sensitivity_Analysis/correlation_analysis.py` is a separate,
  self-contained copy — deliberately untouched.
