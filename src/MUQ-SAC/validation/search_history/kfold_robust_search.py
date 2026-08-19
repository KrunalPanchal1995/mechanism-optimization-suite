"""
ROUND 1 of 3 -- see search_history/README.md for the full round-to-round
lineage (ridge grid widening, case subsets, round-3's band-valid scoring
change).

Robust (worst-of-10-fold) hyperparameter re-selection, triggered by
kfold_cv.py finding that the single-split-selected hyperparameters do NOT
keep every fold <=2% MRE for most cases (only 3/8: Case 2, 6, 7) -- a single
70/15/15 split massively underestimated true out-of-sample risk for e.g.
Case 4 (single-split hold-out 1.45%, but 10-fold mean test error 8.1%, worst
fold 20.3%, at the SAME hyperparameters).

For each case, searches (tolerance, k, order, ridge) to MINIMIZE the worst of
N_FOLDS_SEARCH folds (a different, independent split from kfold_cv.py's
N_FOLDS=10 evaluation split -- selecting and reporting on the same exact
split would be circular/optimistic). The result is then re-verified by
kfold_cv_verify.py against a THIRD, independently-seeded 10-fold split to
confirm the chosen combo is genuinely robust, not just fit to the search
split.

Zero new simulations -- reuses the same cached data kfold_cv.py loads.
Parallelized over (case, tolerance) pairs (up to 64 workers).

Search/selection logic is UNCHANGED from test_build/kfold_robust_search.py --
only loaders (-> case_data) and paths (-> config) were refactored.

Run: python3.9 kfold_robust_search.py
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import json
import time
from math import comb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import fit_feature_map, screen_rows, _fit_once, _is_better
import case_data

N_FOLDS_SEARCH = 10
SEARCH_SPLIT_SEED = config.SEARCH_SPLIT_SEED

TOLERANCES = config.TOLERANCES
K_GRID = config.K_GRID
ORDER_GRID = config.ORDER_GRID
RIDGE_GRID = config.RIDGE_GRID_R1
MAX_TERMS_FRAC = config.MAX_TERMS_FRAC


def best_for_case_tol(case, tol, sel, X, y, blocks, scratch_root):
    scratch = os.path.join(scratch_root, f"case-{case}-tol-{tol}")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)
    n_comp = X.shape[1]
    n = len(y)
    fold_of = case_data.make_folds(n, case, n_folds=N_FOLDS_SEARCH, seed=SEARCH_SPLIT_SEED)

    best = None
    for k in K_GRID:
        for order in ORDER_GRID:
            for ridge_alpha in RIDGE_GRID:
                fold_scores = []
                ok = True
                for f in range(N_FOLDS_SEARCH):
                    tr_idx_all = np.where(fold_of != f)[0]
                    va_idx = np.where(fold_of == f)[0]
                    keep_tr, _ = screen_rows(X[tr_idx_all], blocks, tol)
                    tr_idx = tr_idx_all[keep_tr]
                    if len(tr_idx) < max(50, 2 * n_comp) or len(va_idx) < 5:
                        ok = False; break
                    Xa, ya = X[tr_idx], y[tr_idx]
                    Xb, yb = X[va_idx], y[va_idx]
                    try:
                        fmap = fit_feature_map(Xa, ya, blocks, n_comp, k=k)
                    except Exception:
                        ok = False; break
                    n_terms_est = sum(comb(fmap.k + d - 1, d) for d in range(order + 1))
                    if n_terms_est > MAX_TERMS_FRAC * len(Xa):
                        ok = False; break
                    mtr, mva = _fit_once(PRSmod.ResponseSurface, scratch, Xa, ya, Xb, yb,
                                          case, case, order, fmap, ridge_alpha,
                                          "Partial", sel, 2.0)
                    fold_scores.append(max(mtr, mva))
                if not ok or not fold_scores:
                    continue
                worst = max(fold_scores)
                if best is None or _is_better(worst, k, best["score"], best["k"]):
                    best = {"score": worst, "tolerance": tol, "k": k, "order": order,
                            "ridge_alpha": ridge_alpha, "fold_scores": fold_scores}
    return best


def _worker(args):
    case, tol, data_dir, scratch_root = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    try:
        pool = case_data.build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
        best = best_for_case_tol(case, tol, pool["sel"], pool["X"], pool["y"], pool["blocks"], scratch_root)
        if best is None:
            return {"case": case, "tolerance": tol, "status": "NO_MODEL"}
        return {"case": case, "tolerance": tol, "status": "OK", **best}
    except Exception as e:
        import traceback
        return {"case": case, "tolerance": tol, "status": "ERROR", "error": str(e),
                "tb": traceback.format_exc()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('tenfold')/02_robust_search_round1/)")
    ap.add_argument("--scratch", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch_root = args.scratch or config.scratch_dir("kfold_robust_search")
    out_dir = args.output_dir or os.path.join(config.results_dir("tenfold"), "02_robust_search_round1")

    os.makedirs(scratch_root, exist_ok=True)
    tasks = [(c, t, data_dir, scratch_root) for c in range(8) for t in TOLERANCES]
    print(f"Robust worst-of-{N_FOLDS_SEARCH}-fold search: {len(tasks)} (case,tol) tasks...")
    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(processes=64) as pool:
        raw = pool.map(_worker, tasks)
    print(f"sweep done in {time.time()-t0:.0f}s")

    for r in raw:
        if r["status"] == "ERROR":
            print(f"case {r['case']} tol {r['tolerance']}: ERROR {r['error']}")
            print(r["tb"])

    by_case = {c: [] for c in range(8)}
    for r in raw:
        if r["status"] == "OK":
            by_case[r["case"]].append(r)

    chosen = {}
    for c in range(8):
        cands = by_case[c]
        if not cands:
            print(f"case {c}: NO VALID MODEL FOUND")
            continue
        best = None
        for r in cands:
            if best is None or _is_better(r["score"], r["k"], best["score"], best["k"]):
                best = r
        chosen[c] = {"tolerance": best["tolerance"], "k": best["k"],
                     "order": best["order"], "ridge_alpha": best["ridge_alpha"],
                     "search_worst_fold_pct": best["score"]}
        tol_s = "inf" if best["tolerance"] == float("inf") else f"{best['tolerance']:g}"
        print(f"case {c}: tol={tol_s} k={best['k']} order={best['order']} "
              f"ridge={best['ridge_alpha']:.0e}  search_worst_fold={best['score']:.3f}%")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chosen_hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"chosen hyperparameters written to {out_path}")
    return chosen


if __name__ == "__main__":
    main()
