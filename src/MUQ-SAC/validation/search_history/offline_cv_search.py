"""
SUPERSEDED by production IdentifiablePRS.fit_prs_with_cv_search.

Offline, zero-new-simulation search for whether a stricter <=1.2% (or <=1.0%)
max-residual-error criterion is achievable on TEST_CORRELATIONS_CORRECTED,
using k-fold cross-validated hyperparameter search
(IdentifiablePRS.fit_prs_with_cv_search's logic) instead of offline_replay.py's
single train/validation split.

Motivation (see report/main.tex, Section "Tuning Case 1"): a single split's
validation score cannot reliably rank near-tied (tolerance, k, order, ridge)
combinations -- two combos 0.03 points apart on one split produced hold-out
scores of 1.89% and 3.30%. cv_search scores each combo by its WORST k-fold
score, a strictly harder, more robust bar, implemented but not needed for the
2% goal. Pushing to 1.2%/1.0% is exactly the regime where that should matter.

Parallelized over (case, tolerance) pairs (64 tasks) rather than just case
(8 tasks) -- this machine has 160 cores and a single case's full
(k, order, ridge) x 5-fold sweep at one tolerance is independent work.

Search/scoring logic is UNCHANGED from test_build/offline_cv_search.py --
only the loader block (-> case_data) and output path
(-> report/results/offline_cv_search/) were refactored.

Run: python3.9 offline_cv_search.py
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
from IdentifiablePRS import (build_layout_blocks, fit_feature_map, screen_rows,
                              three_way_split, _fit_once, _is_better)
import case_data

TOLERANCES = [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")]
K_GRID     = [8, 10, 12, 14, 16, 18, 20]
ORDER_GRID = [2, 3]
RIDGE_GRID = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
CV_FOLDS   = 5
CV_SEED    = 13
MAX_TERMS_FRAC = 0.4


def _case_context(case, rxns, sigma_of_rxn, Temp, data_dir=None):
    X, y, sel = case_data.load_case(case, data_dir=data_dir)
    layout = case_data.make_layout(sel, rxns)
    Tc = Temp[case]
    T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
    blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)
    itr, ival, iho = three_way_split(len(y))
    itrva = np.concatenate([itr, ival])
    return X, y, sel, blocks, itrva, iho


def best_for_case_tol(case, tol, sel, X, y, blocks, itrva, scratch_root):
    """Full (k, order, ridge) x CV_FOLDS sweep at one fixed tolerance.

    Critical: screening is applied ONLY to each fold's training rows, never
    to its validation rows (mirrors fit_prs_with_search, which screens Xtr
    but scores against an untouched, unscreened Xval). Screening BOTH sides
    of a fold (an earlier version of this script did that) lets aggressive
    screening look artificially good on held-out data drawn from the same
    screened pool, hiding exactly the over-screening generalization failure
    documented in report/main.tex's parametric study (median 150-360% error
    at tight tolerances vs 2.10% unscreened) -- it does not protect against
    it, so a real (unscreened) hold-out is required to see the failure.

    Returns best dict for this (case, tol), or None if nothing fittable."""
    scratch = os.path.join(scratch_root, f"case-{case}-tol-{tol}")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)

    n_comp = X.shape[1]
    rng = np.random.default_rng(CV_SEED)
    perm = rng.permutation(len(itrva))
    fold_of = np.empty(len(itrva), dtype=int)
    fold_of[perm] = np.arange(len(itrva)) % CV_FOLDS

    best = None
    for k in K_GRID:
        for order in ORDER_GRID:
            for ridge_alpha in RIDGE_GRID:
                fold_scores = []
                ok = True
                for f in range(CV_FOLDS):
                    tr_idx_all = itrva[fold_of != f]
                    va_idx = itrva[fold_of == f]          # UNSCREENED validation fold
                    keep_tr, _ = screen_rows(X[tr_idx_all], blocks, tol)
                    tr_idx = tr_idx_all[keep_tr]
                    if len(tr_idx) < max(50, 2 * n_comp) or len(va_idx) < 5:
                        ok = False
                        break
                    Xa, ya = X[tr_idx], y[tr_idx]
                    Xb, yb = X[va_idx], y[va_idx]
                    try:
                        fmap = fit_feature_map(Xa, ya, blocks, n_comp, k=k)
                    except Exception:
                        ok = False
                        break
                    n_terms_est = sum(comb(fmap.k + d - 1, d) for d in range(order + 1))
                    if n_terms_est > MAX_TERMS_FRAC * len(Xa):
                        ok = False
                        break
                    mtr, mva = _fit_once(PRSmod.ResponseSurface, scratch, Xa, ya, Xb, yb,
                                          case, case, order, fmap, ridge_alpha,
                                          "Partial", sel, 1.2)
                    fold_scores.append(max(mtr, mva))
                if not ok or not fold_scores:
                    continue
                worst = max(fold_scores)
                if best is None or _is_better(worst, k, best["score"], best["k"]):
                    keep_full, _ = screen_rows(X[itrva], blocks, tol)
                    best = {"score": worst, "tolerance": tol, "k": k, "order": order,
                            "ridge_alpha": ridge_alpha, "fold_scores": fold_scores,
                            "keep": keep_full, "n_kept": int(keep_full.sum())}
    return best


def _worker(args):
    case, tol, data_dir, scratch_root = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    try:
        X, y, sel, blocks, itrva, iho = _case_context(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
        best = best_for_case_tol(case, tol, sel, X, y, blocks, itrva, scratch_root)
        if best is None:
            return {"case": case, "tolerance": tol, "status": "NO_MODEL"}
        return {"case": case, "tolerance": tol, "status": "OK",
                "score": best["score"], "k": best["k"], "order": best["order"],
                "ridge_alpha": best["ridge_alpha"], "fold_scores": best["fold_scores"],
                "n_kept": best["n_kept"]}
    except Exception as e:
        import traceback
        return {"case": case, "tolerance": tol, "status": "ERROR", "error": str(e),
                "tb": traceback.format_exc()}


def finalize_case(case, tol_results, rxns, sigma_of_rxn, Temp, data_dir, scratch_root):
    """Pick the best tolerance for this case (by _is_better on worst-fold
    score), then do the ONE real refit-on-full-pool + untouched-holdout score
    that matters for reporting."""
    ok_results = [r for r in tol_results if r["status"] == "OK"]
    if not ok_results:
        return {"case": case, "status": "NO_MODEL"}

    best = None
    for r in ok_results:
        if best is None or _is_better(r["score"], r["k"], best["score"], best["k"]):
            best = r

    X, y, sel, blocks, itrva, iho = _case_context(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
    keep, _ = screen_rows(X[itrva], blocks, best["tolerance"])
    idx_kept = itrva[keep]
    fmap_final = fit_feature_map(X[idx_kept], y[idx_kept], blocks, X.shape[1], k=best["k"])

    scratch = os.path.join(scratch_root, f"case-{case}-final")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)
    mtr_final, mho_final = _fit_once(
        PRSmod.ResponseSurface, scratch, X[idx_kept], y[idx_kept], X[iho], y[iho],
        case, case, best["order"], fmap_final, best["ridge_alpha"],
        "Partial", sel, 1.2)

    return {
        "case": case, "status": "OK",
        "tolerance": best["tolerance"], "k": best["k"], "order": best["order"],
        "ridge_alpha": best["ridge_alpha"], "cv_worst_fold_pct": best["score"],
        "fold_scores": best["fold_scores"], "n_kept": best["n_kept"],
        "final_train_pct": mtr_final, "final_holdout_pct": mho_final,
        "pass_1_2pct": bool(max(mtr_final, mho_final) <= 1.2),
        "pass_1_0pct": bool(max(mtr_final, mho_final) <= 1.0),
        "all_tolerances_tried": [(r["tolerance"], r["score"]) for r in ok_results],
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('report')/offline_cv_search/)")
    ap.add_argument("--scratch", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch_root = args.scratch or config.scratch_dir("offline_cv_search")
    out_dir = args.output_dir or os.path.join(config.results_dir("report"), "offline_cv_search")

    os.makedirs(scratch_root, exist_ok=True)
    tasks = [(c, t, data_dir, scratch_root) for c in range(8) for t in TOLERANCES]
    print(f"Launching {len(tasks)} (case, tolerance) tasks across up to 64 workers...")

    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(processes=64) as pool:
        raw_results = pool.map(_worker, tasks)
    print(f"(case,tol) sweep done in {time.time()-t0:.0f}s")

    for r in raw_results:
        if r["status"] == "ERROR":
            print(f"case {r['case']} tol {r['tolerance']}: ERROR {r['error']}")
            print(r["tb"])

    by_case = {c: [] for c in range(8)}
    for r in raw_results:
        by_case[r["case"]].append(r)

    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)

    final = []
    for c in range(8):
        final.append(finalize_case(c, by_case[c], rxns, sigma_of_rxn, Temp, data_dir, scratch_root))

    for r in final:
        if r["status"] == "OK":
            print(f"case {r['case']}: tol={r['tolerance']:<4} k={r['k']:<3} order={r['order']} "
                  f"ridge={r['ridge_alpha']:.0e}  cv_worst_fold={r['cv_worst_fold_pct']:.3f}%  "
                  f"TRAIN={r['final_train_pct']:.3f}%  HOLDOUT={r['final_holdout_pct']:.3f}%  "
                  f"{'PASS<=1.2' if r['pass_1_2pct'] else 'fail1.2'}  "
                  f"{'PASS<=1.0' if r['pass_1_0pct'] else 'fail1.0'}")
        else:
            print(f"case {r['case']}: {r['status']}")

    n12 = sum(1 for r in final if r.get("pass_1_2pct"))
    n10 = sum(1 for r in final if r.get("pass_1_0pct"))
    print(f"\n==> {n12}/8 cases meet <=1.2% on BOTH train and untouched hold-out")
    print(f"==> {n10}/8 cases meet <=1.0% on BOTH train and untouched hold-out")
    print(f"(total wall time {time.time()-t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "offline_cv_search_results.json")
    with open(out_path, "w") as f:
        json.dump({"per_case_tol": raw_results, "final": final}, f, indent=2)
    print(f"results written to {out_path}")
    return final


if __name__ == "__main__":
    main()
