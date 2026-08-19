# docs/artifacts/ — provenance

Both PDFs below were copied VERBATIM (binary-identical, md5-confirmed) from
`test_build/` on 2026-07-31, as part of the Part 1 reorganization. Neither
file's content was regenerated or modified in any way.

| File | Source | md5 | Status |
|---|---|---|---|
| `only generating samples in outlier local region.pdf` | `test_build/only generating samples in outlier local region.pdf` | `5e38a0ae1991a5a1686d21156a5f99e4` | Current — no known superseding figure. |
| `Parity_plot_case_3_TESTING.pdf` | `test_build/Parity_plot_case_3_TESTING.pdf` | `b5300b9e3f291b756493eedf241059fe` | **Superseded** by `report_10fold_validation/figures/parity_scaled_case_3_case3_fold7_{before,after}.pdf` (produced by `validation/report_figures/plot_parity_bad_fold.py`), which show the same case-3 parity comparison against the current, validated 10-fold CV hyperparameters rather than this earlier ad hoc test run. Kept here for historical reference, not as the current source of truth. |

Verify at any time with:

```
md5sum "docs/artifacts/only generating samples in outlier local region.pdf" \
       "docs/artifacts/Parity_plot_case_3_TESTING.pdf"
```
