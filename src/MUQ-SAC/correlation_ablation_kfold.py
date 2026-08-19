"""
Worst-of-10-fold correlation-aware ablation, all 68 cases -- the ROBUST
version of case_band_valid_fit.py's single-split aware/unaware comparison.

Why this exists: the single-split ablation (case_band_valid_fit.py) gave a
noisy, inconsistent signal -- mean delta across the 49 rescue-affected cases
was slightly NEGATIVE (ablation looked marginally BETTER on average) and only
18/49 cases showed the expected direction. That is exactly the failure mode
this project's own IdentifiablePRS.fit_prs_with_cv_search already diagnosed
and fixed for HYPERPARAMETER selection (see its docstring: "a single split
can prefer a combo whose validation score is only marginally better... while
generalizing very differently to an independent hold-out"). The same logic
applies here: whether omitting a handful of correlated columns hurts a fit
is a question about GENERALIZATION, and one 15%-of-rows hold-out split
(~50-110 rows for most cases) is not enough evidence either way.

Fix, mirroring validation/kfold_cv_bandvalid.py: for every case with
n_rescued>0, 10-fold split the case's band-valid rows (same fold recipe as
kfold_cv.py: rng=default_rng(seed+case); perm; fold_of), and for EACH fold
fit both the aware (all columns) and unaware (rescued columns zeroed) PRS on
the training rows, score on that fold's held-out rows. Report the WORST
fold's MRE for each -- the same robustness-oriented statistic
fit_prs_with_cv_search uses. Uses the hyperparameters (k, order, ridge)
ALREADY CHOSEN for each case by case_band_valid_fit.py's aware fit (read
from its output CSV) for BOTH aware and unaware -- deliberately not
re-searching per fold (would be 10x68x(grid size)x2 fits, far more expensive
for a check whose only purpose is robustness, not hyperparameter tuning) --
this keeps the comparison apples-to-apples: same subspace dimension/order/
regularization, the only difference being which raw columns are visible.

Usage: OMP_NUM_THREADS=1 python3.9 correlation_ablation_kfold.py <study_dir> <aware_csv> <out_csv> [--workers N]
"""
import os
import sys
import csv
import argparse
import warnings
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IdentifiablePRS as IPRS
import ResponseSurface as PRS
from case_band_valid_fit import (load_rxns_sigma, load_temps, make_layout, load_case_xy,
                                   rescued_columns)


def make_folds(n, case, n_folds=10, seed=2026):
    rng = np.random.default_rng(seed + case)
    perm = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % n_folds
    return fold_of


def fold_worst_mre(X, y, blocks, n_comp, case, k, order, ridge_alpha, cache_dir, n_folds=10):
    fold_of = make_folds(len(y), case, n_folds=n_folds)
    sel_params = np.ones(n_comp, dtype=int)
    coef_path = os.path.join(cache_dir, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
    worst = 0.0
    cwd0 = os.getcwd()
    os.chdir(cache_dir)
    try:
        for f in range(n_folds):
            tr = np.where(fold_of != f)[0]
            te = np.where(fold_of == f)[0]
            if len(tr) < max(20, n_comp) or len(te) < 3:
                continue
            try:
                fmap = IPRS.fit_feature_map(X[tr], y[tr], blocks, n_comp, k=k)
            except Exception:
                continue
            if os.path.isfile(coef_path):
                os.remove(coef_path)
            resp = PRS.ResponseSurface(X[tr], y[tr], case, case, responseOrder=order,
                                        prs_type="Partial", selected_params=sel_params,
                                        criteria=2.0, feature_map=fmap, ridge_alpha=ridge_alpha)
            resp.create_response_surface()
            mtr, mte, _ = resp.test(X[te], y[te])
            worst = max(worst, mtr, mte)
            if os.path.isfile(coef_path):
                os.remove(coef_path)
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
    return worst


def run_one_case(args):
    D, case, k, order, ridge_alpha, cache_root = args
    warnings.filterwarnings("ignore")
    rxns, sigma = load_rxns_sigma(D)
    T = load_temps(D)
    X, y, sel = load_case_xy(D, case)
    layout = make_layout(sel, rxns)
    Tc = T[case]
    T_local = np.array([Tc * 0.75, Tc, Tc * 1.25])
    blocks = IPRS.build_layout_blocks(layout, lambda r: sigma[r], T_local, svd_tol=1e-6, fprior_mode="full")
    n_comp = X.shape[1]
    valid_mask = IPRS.band_valid_mask(X, blocks, 1.0)
    Xv, yv = X[valid_mask], y[valid_mask]

    with open(D + "unsrt.pkl", "rb") as fh:
        unsrt = pickle.load(fh)
    active_param_map = {r: unsrt[r].activeParameters for r in rxns if r in unsrt}
    rcols = rescued_columns(D, case, rxns, sel, active_param_map)

    cache_dir = os.path.join(cache_root, f"case-{case}")
    os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)

    aware_worst = fold_worst_mre(Xv, yv, blocks, n_comp, case, k, order, ridge_alpha, cache_dir)
    if len(rcols) > 0:
        Xu = Xv.copy(); Xu[:, rcols] = 0.0
        unaware_worst = fold_worst_mre(Xu, yv, blocks, n_comp, case, k, order, ridge_alpha, cache_dir)
    else:
        unaware_worst = aware_worst

    return dict(case=case, n_valid=len(yv), n_rescued_cols=int(len(rcols)),
                aware_worst_fold=aware_worst, unaware_worst_fold=unaware_worst,
                delta=unaware_worst - aware_worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("aware_csv", help="case_band_valid_fit.py output, for chosen (k, order, ridge_alpha) per case")
    ap.add_argument("out_csv")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    D = args.study_dir.rstrip("/") + "/"
    hp = {}
    with open(args.aware_csv) as f:
        for r in csv.DictReader(f):
            hp[int(r["case"])] = (int(r["aware_k"]), int(r["aware_order"]), float(r["aware_ridge_alpha"]))

    cache_root = "/tmp/claude-1000/correlation_ablation_kfold_cache"
    os.makedirs(cache_root, exist_ok=True)
    tasks = [(D, c, k, order, ridge, cache_root) for c, (k, order, ridge) in sorted(hp.items())]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_case, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[case {c}] aware_worst={r['aware_worst_fold']:.3f}% "
                      f"unaware_worst={r['unaware_worst_fold']:.3f}% delta={r['delta']:+.3f} "
                      f"n_rescued={r['n_rescued_cols']}", flush=True)
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
