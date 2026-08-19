# report_figures/ — figure ↔ script ↔ results-file map

All figures below currently live in `report_10fold_validation/figures/`.
This README maps each one to the script that produces it and the results/
file(s) it reads. Figures produced by scripts in THIS task (Part 1) are
written to a staging location during validation, not directly into
`report_10fold_validation/figures/` — see `validation/README.md`'s
verification section for the promotion step.

| Figure file(s) | Produced by | Reads |
|---|---|---|
| `parity_scaled_case_4_case4_fold5_before.{pdf,png}`<br>`parity_scaled_case_4_case4_fold5_after.{pdf,png}` | `plot_parity_bad_fold.py` | `case_data` loaders (design matrices / Sigma / outlier pool directly, not a results/ JSON) |
| `parity_scaled_case_3_case3_fold7_before.{pdf,png}`<br>`parity_scaled_case_3_case3_fold7_after.{pdf,png}` | `plot_parity_bad_fold.py` | same |
| `violation_nc7h16_+_ho2_=>_c7h15-4_+_h2o2_case3.{pdf,png}` | `plot_band_violation_samples.py` | same (case 3 row 609 target) |
| `violation_nc7h16_+_ho2_=>_c7h15-2_+_h2o2_case0.{pdf,png}` | `plot_band_violation_samples.py` | same (case 0 row 1702 target) |
| `violation_row609_case3_generators.{pdf,png}` | `plot_row_generator_diagnostics.py` | same, plus `DM_FOR_PARTIAL_PRS/<case>/DesignMatrix.csv` for generator-class attribution |
| `violation_row1702_case0_generators.{pdf,png}` | `plot_row_generator_diagnostics.py` | same |
| `fig_10fold_before.{pdf,png}`, `fig_10fold_after.{pdf,png}`, `fig_10fold_summary.{pdf,png}` | `plot_10fold_evolution.py` | **NOT part of this task** — see Part 2 of the plan; this script is intentionally left out of `report_figures/` for now (neither ported nor copied) and stays only at `test_build/plot_10fold_evolution.py` until a later pass redesigns it CSV-driven per Part 2. |

## New JSON metrics artifacts (this task's ask #2, "persist study artifacts")

Each `report_figures/*.py` script here now ALSO writes a small JSON summary
of the numbers it prints, alongside the figures, so a study's headline
numbers are independently checkable without re-parsing stdout or
re-rendering a plot:

- `plot_parity_bad_fold.py` → `results_dir("tenfold")/parity_bad_fold_metrics.json`
- `plot_band_violation_samples.py` → `results_dir("tenfold")/band_violation_samples.json`
- `plot_row_generator_diagnostics.py` → `results_dir("tenfold")/row_generator_diagnostics.json`

## `plot_before_after_paar.py`

Not built in this task — see Part 3 of the plan.
