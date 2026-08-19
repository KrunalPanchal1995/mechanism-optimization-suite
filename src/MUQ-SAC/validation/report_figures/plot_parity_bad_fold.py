"""
Parity (predicted vs actual) plots for the worst 10-fold-CV fits, to show
*why* the baseline (last-production) hyperparameters failed the 10-fold
robustness check: the outlier points that blow up the error are exactly the
band-invalid rows (violation ratio > 1, i.e. a row the sampler's own
physical-uncertainty test would have rejected).

Plotting itself is delegated to ResponseSurface.plot_parity_scaled (the
production method -- see ResponseSurface.py), which draws BOTH a zoomed-out
panel at the case's PROPER (global) PRS scale and a zoomed-in panel
restricted to band-valid points -- this script only assembles the fold's fit
and the data needed to call it.

Reproduces, with zero new simulations (same cached design matrices used by
the 2026-07-30 10-fold CV audit, report_10fold_validation/main.tex):
  Case 4, fold 5, BASELINE hp (tol=10.0, k=20, ridge=0.0)
      -> the report's headline number: worst-fold MRE 20.3%
  Case 3, fold 7, BASELINE hp (tol=inf, k=20, ridge=0.0)
      -> worst-fold MRE 3.16%
For each, also refits the SAME fold split with the FINAL hyperparameters
(Case 4: tol=1.0,k=20,ridge=0.5; Case 3: tol=10.0,k=20,ridge=0.0) so the
before/after parity comparison is apples-to-apples on one partition.

CASE_CONFIGS below (the hyperparameter values, case/fold pairs) are kept
VERBATIM from test_build/plot_parity_bad_fold.py -- do not change them, they
reproduce published figures. The only changes here are plumbing: the
hardcoded absolute `sys.path.insert(0, ".../test_build")` is replaced with
`_bootstrap`, loaders come from case_data, and D/OUT come from config.
Additionally (ask #2, "persist study artifacts"): writes
results_dir("tenfold")/parity_bad_fold_metrics.json with the before/after
all-points and valid-only MRE numbers this script previously only printed.

Run: python3.9 plot_parity_bad_fold.py [--output-dir DIR] [--results-dir DIR]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import build_layout_blocks, fit_feature_map, screen_rows, violation_detail
import case_data

CASE_CONFIGS = [
    dict(case=4, fold=5, before=dict(tolerance=10.0, k=20, order=2, ridge_alpha=0.0),
         after=dict(tolerance=1.0, k=20, order=2, ridge_alpha=0.5),
         before_tag="baseline hp (tol=10, k=20, ridge=0) -- report worst-fold 20.3%",
         after_tag="final hp (tol=1, k=20, ridge=0.5), same fold split"),
    dict(case=3, fold=7, before=dict(tolerance=float("inf"), k=20, order=2, ridge_alpha=0.0),
         after=dict(tolerance=10.0, k=20, order=2, ridge_alpha=0.0),
         before_tag="baseline hp (tol=$\\infty$, k=20, ridge=0) -- report worst-fold 3.16%",
         after_tag="final hp (tol=10, k=20, ridge=0), same fold split"),
]


def prep_case(case, rxns, sigma_of_rxn, Temp, data_dir=None):
    pool = case_data.build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
    X, y, sel, blocks = pool["X"], pool["y"], pool["sel"], pool["blocks"]
    fold_of = case_data.make_folds(len(y), case, seed=config.CV_SPLIT_SEED)
    ratio, worst_rxn = violation_detail(X, blocks)
    return X, y, sel, blocks, fold_of, ratio, worst_rxn


def fit_fold(case, X, y, sel, blocks, fold_of, fold, hp, scratch):
    tr_idx_all = np.where(fold_of != fold)[0]
    te_idx = np.where(fold_of == fold)[0]
    keep_tr, _ = screen_rows(X[tr_idx_all], blocks, hp["tolerance"])
    tr_idx = tr_idx_all[keep_tr]
    fmap = fit_feature_map(X[tr_idx], y[tr_idx], blocks, X.shape[1], k=hp["k"])

    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)
    coef_path = os.path.join(scratch, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
    if os.path.isfile(coef_path):
        os.remove(coef_path)
    cwd0 = os.getcwd(); os.chdir(scratch)
    try:
        rs = PRSmod.ResponseSurface(X[tr_idx], y[tr_idx], case, case, responseOrder=hp["order"],
                                     prs_type="Partial", selected_params=sel, criteria=2.0,
                                     feature_map=fmap, ridge_alpha=hp["ridge_alpha"])
        rs.create_response_surface()
        rs.test(X[te_idx], y[te_idx])
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
    return rs, te_idx


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Figure output dir (default: config.figures_dir('tenfold'))")
    ap.add_argument("--results-dir", default=None,
                     help="Metrics JSON output dir (default: config.results_dir('tenfold'))")
    ap.add_argument("--scratch", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    out_figs = args.output_dir or config.figures_dir("tenfold")
    out_results = args.results_dir or config.results_dir("tenfold")
    scratch = args.scratch or config.scratch_dir("parity_plot")
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)

    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)

    metrics = []
    for cfg in CASE_CONFIGS:
        case, fold = cfg["case"], cfg["fold"]
        X, y, sel, blocks, fold_of, ratio, worst_rxn = prep_case(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
        y_global = (float(y.min()), float(y.max()))  # PROPER (case-global) PRS scale

        case_metrics = {"case": case, "fold": fold}
        for which in ("before", "after"):
            hp = cfg[which]
            rs, te_idx = fit_fold(case, X, y, sel, blocks, fold_of, fold, hp,
                                   os.path.join(scratch, f"case-{case}-{which}"))
            out_path = rs.plot_parity_scaled(
                ratio=ratio[te_idx], y_global=y_global, location=out_figs,
                tag=f"_case{case}_fold{fold}_{which}")
            valid_mask = ratio[te_idx] <= config.BAND_TOLERANCE_EVAL
            rel = np.asarray(rs.error_testing_relative, dtype=float)
            valid_only_max = float(np.max(np.abs(rel[valid_mask]))) if valid_mask.any() else None
            print(f"Case {case} fold {fold} [{which}] ({cfg[which + '_tag']}): "
                  f"train max={rs.MaxError:.2f}%  test max (all)={rs.ytestMaxError:.2f}%  "
                  f"test max (valid-only)={valid_only_max}  -> {out_path}")
            case_metrics[which] = {
                "hp": hp, "train_max_pct": float(rs.MaxError),
                "test_max_all_pct": float(rs.ytestMaxError),
                "test_max_valid_only_pct": valid_only_max,
                "figure_path": out_path,
            }
        metrics.append(case_metrics)

    out_path = os.path.join(out_results, "parity_bad_fold_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"cases": metrics}, f, indent=2)
    print(f"\nmetrics written to {out_path}")
    return metrics


if __name__ == "__main__":
    main()
