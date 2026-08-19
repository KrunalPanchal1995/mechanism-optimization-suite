"""
Reproduces the ORIGINAL (pre-2026-08-19-fix) PRS fitting behaviour for all 68
cases of THREE-PARAM-MSO-LLNL-N-HEPTANE_159_CORRELATION_AWARE, standalone and
read-only, for the "before" side of the before/after comparison plots and
report table. Companion to case_band_valid_fit.py (the "after"/fixed run) --
see that file's module docstring for the shared row-loading/layout helpers,
reused here via import.

What "original" means, precisely (matches target.opt / run_script.py before
this investigation's F1/F4 changes):
  - band-validity scored with the REDUCED-subspace f_prior (fprior_mode=
    "reduced", IdentifiablePRS.reaction_W_fprior's pre-existing default) --
    the metric bug (root cause 2: for m<3 reactions this denominator can be
    >100x smaller than the sampler's true acceptance-test band).
  - training-row screening searched over the FULL original tolerance grid
    [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, inf], not pinned to 1.0.
  - the HEADLINE number is the ALL-POINTS hold-out MRE (what
    Prediction_errors.csv's "Max_Testing_Error" column reported), not the
    band-valid-only number the pipeline was already (silently) computing
    for its internal PASS/FAIL gate.

This reproduces the class-B m=3 sampler bug's effect too, since it reads the
SAME on-disk design-matrix rows (DM_FOR_PARTIAL_PRS/<case>/pDesignMatrix.csv)
that bug produced -- nothing here re-simulates anything.

Usage: OMP_NUM_THREADS=1 python3.9 case_before_fit.py <study_dir> <out_csv> [--cases 0,1,2] [--workers N]
"""
import os
import sys
import csv
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IdentifiablePRS as IPRS
import ResponseSurface as PRS
from case_band_valid_fit import load_rxns_sigma, load_temps, make_layout, load_case_xy

ORIGINAL_TOL_GRID = [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")]
K_GRID = [8, 10, 12, 14, 16, 20]
ORDER_GRID = [2, 3]
RIDGE_GRID = [0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5]


def run_one_case(args):
    D, case, MRE_criteria, cache_root = args
    warnings.filterwarnings("ignore")
    rxns, sigma = load_rxns_sigma(D)
    T = load_temps(D)
    X, y, sel = load_case_xy(D, case)
    layout = make_layout(sel, rxns)
    Tc = T[case]
    T_local = np.array([Tc * 0.75, Tc, Tc * 1.25])
    blocks = IPRS.build_layout_blocks(layout, lambda r: sigma[r], T_local, svd_tol=1e-6,
                                       fprior_mode="reduced")   # the metric bug, unfixed
    n_comp = X.shape[1]
    cache_dir = os.path.join(cache_root, f"case-{case}")
    os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)

    itr, ival, iho = IPRS.three_way_split(len(y), seed_outer=104, seed_inner=7,
                                            train_frac=0.70, val_frac=0.15)
    xTrain, yTrain = X[itr], y[itr]
    xVal, yVal = X[ival], y[ival]
    xHold, yHold = X[iho], y[iho]
    sel_params = np.ones(n_comp, dtype=int)

    Xp = np.vstack([xTrain, xVal]); yp = np.concatenate([yTrain, yVal])
    search = IPRS.fit_prs_with_cv_search(
        PRS.ResponseSurface, Xp, yp, blocks, n_comp, case, case, "Partial", sel_params,
        MRE_criteria, cache_dir=cache_dir, tol_grid=ORIGINAL_TOL_GRID, k_grid=K_GRID,
        order_grid=ORDER_GRID, ridge_grid=RIDGE_GRID, cv_folds=10, cv_seed=13,
        band_tolerance_eval=1.0)
    keep_mask, _ = IPRS.screen_rows(xTrain, blocks, search["tolerance"])
    xTrainK, yTrainK = xTrain[keep_mask], yTrain[keep_mask]
    fmap = IPRS.fit_feature_map(xTrainK, yTrainK, blocks, n_comp, k=search["k"])

    coef_path = os.path.join(cache_dir, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
    if os.path.isfile(coef_path):
        os.remove(coef_path)
    cwd0 = os.getcwd()
    os.chdir(cache_dir)
    try:
        resp = PRS.ResponseSurface(xTrainK, yTrainK, case, case, responseOrder=search["order"],
                                    prs_type="Partial", selected_params=sel_params, criteria=MRE_criteria,
                                    feature_map=fmap, ridge_alpha=search["ridge_alpha"])
        resp.create_response_surface()
        mtr, mva, _ = resp.test(xVal, yVal)                     # ALL-POINTS (the historical headline)
        mtrH, mvaH, _ = resp.test(xHold, yHold)                 # ALL-POINTS hold-out
        hmask = IPRS.band_valid_mask(xHold, blocks, 1.0)
        rel = np.asarray(resp.error_testing_relative, dtype=float)
        hold_bv = float(np.max(np.abs(rel[hmask]))) if hmask.any() else mvaH
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)

    return dict(case=case, n_rows=len(X), tolerance=search["tolerance"], k=search["k"],
                order=search["order"], ridge_alpha=search["ridge_alpha"], n_kept=search["n_kept"],
                train=mtrH, val_allpoints=mva, hold_allpoints=mvaH, hold_bandvalid=hold_bv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    D = args.study_dir.rstrip("/") + "/"
    if args.cases:
        cases = [int(c) for c in args.cases.split(",")]
    else:
        cases = sorted(int(d) for d in os.listdir(D + "DM_FOR_PARTIAL_PRS")
                        if os.path.isdir(D + "DM_FOR_PARTIAL_PRS/" + d) and d.isdigit())

    cache_root = "/tmp/claude-1000/case_before_fit_cache"
    os.makedirs(cache_root, exist_ok=True)
    tasks = [(D, c, 2.0, cache_root) for c in cases]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_case, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[case {c}] train={r['train']:.3f}% hold(all)={r['hold_allpoints']:.3f}% "
                      f"hold(bandvalid)={r['hold_bandvalid']:.3f}%", flush=True)
            except Exception as exc:
                print(f"[case {c}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                results.append(dict(case=c, error=str(exc)))

    results.sort(key=lambda r: r["case"])
    fieldnames = sorted({k for r in results for k in r.keys()}, key=lambda k: (k != "case", k))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nwrote {args.out_csv} ({len(results)} cases)")


if __name__ == "__main__":
    main()
