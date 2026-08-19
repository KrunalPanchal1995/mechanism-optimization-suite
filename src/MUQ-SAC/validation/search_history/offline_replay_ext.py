"""
SUPERSEDED by the 10-fold CV work in validation/kfold_cv.py -- kept for
historical reproducibility of the numbers report/main.tex's Task #3 section
cites.

Offline replay validation (Task #3 of the PAAR plan).

Exercises the REAL ResponseSurface class (not throwaway numpy) end-to-end on
all 8 cached cases in TEST_CORRELATIONS_CORRECTED, using an honest 3-way
split (70% train / 15% validation / 15% untouched hold-out). Hyperparameters
(band-validity tolerance c, subspace dim k, polynomial order, ridge alpha)
are swept and selected purely on the validation split; the hold-out is
touched exactly once per case, at the very end, to report the number that
matters. This mirrors -- and gates -- what run_script.py's real fitting loop
will do once wired (Task #7).

Fitting logic is UNCHANGED from test_build/offline_replay.py -- only the
loader block (-> case_data) and output path (-> report/results/offline_replay/,
since this feeds report/main.tex, not report_10fold_validation/) were
refactored.

NOTE ON THE offline_replay.py / offline_replay_ext.py PAIR: the original
test_build/offline_replay.py and test_build/offline_replay_ext.py were
byte-identical duplicates (both md5 a63831b480a375f4694946009d08ad4d). That
duplication is preserved here deliberately -- this file and its sibling
validation/search_history/offline_replay_ext.py are kept byte-identical to
EACH OTHER after the SAME refactor was applied to both, so the historical
duplication stays visible and traceable rather than being silently
collapsed into a single file.

Run: python3.9 offline_replay.py
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

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import fit_feature_map, screen_rows, three_way_split
import case_data

TOLERANCES = [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")]
K_GRID     = [8, 10, 12, 14, 16, 20]
ORDER_GRID = [2, 3]
RIDGE_GRID = [0.0, 1e-9, 1e-7, 1e-5]

RNG_SPLIT = 104          # matches run_script.py's existing train_test_split seed
VAL_SEED  = 7


def fit_and_score(Xtr, ytr, Xte, yte, sel, case, order, feature_map, ridge_alpha, scratch):
    """One real ResponseSurface fit; returns (train_max_pct, test_max_pct)."""
    coef_path = os.path.join(scratch, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
    if os.path.isfile(coef_path):
        os.remove(coef_path)
    cwd0 = os.getcwd()
    os.chdir(scratch)
    try:
        rs = PRSmod.ResponseSurface(Xtr, ytr, case, case, responseOrder=order,
                                     prs_type="Partial", selected_params=sel, criteria=2,
                                     feature_map=feature_map, ridge_alpha=ridge_alpha)
        rs.create_response_surface()
        mtr, mte, _ = rs.test(Xte, yte)
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
    return mtr, mte


def run_case(case, rxns, sigma_of_rxn, Temp, log_dir, data_dir=None, scratch=None):
    scratch = os.path.join(scratch, f"case-{case}")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)

    X, y, sel = case_data.load_case(case, data_dir=data_dir)
    layout = case_data.make_layout(sel, rxns)
    from IdentifiablePRS import build_layout_blocks
    Tc = Temp[case]
    T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
    blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)

    itr, ival, iho = three_way_split(len(y))

    best = None   # (score, tol, k, order, ridge, tr, va, ho, n_kept)
    for tol in TOLERANCES:
        keep, ratio = screen_rows(X[itr], blocks, tol)
        itr_kept = itr[keep]
        if len(itr_kept) < 500:
            continue
        for k in K_GRID:
            try:
                fmap = fit_feature_map(X[itr_kept], y[itr_kept], blocks, X.shape[1], k=k)
            except Exception:
                continue
            for order in ORDER_GRID:
                n_terms_est = 1
                kk = fmap.k
                if order == 2:
                    n_terms_est = 1 + kk + kk * (kk + 1) // 2
                else:
                    from math import comb
                    n_terms_est = sum(comb(kk + d - 1, d) for d in range(order + 1))
                if n_terms_est > 0.4 * len(itr_kept):
                    continue
                for ridge_alpha in RIDGE_GRID:
                    mtr, mva = fit_and_score(X[itr_kept], y[itr_kept], X[ival], y[ival],
                                              sel, case, order, fmap, ridge_alpha, scratch)
                    score = max(mtr, mva)
                    if best is None or score < best[0]:
                        best = (score, tol, k, order, ridge_alpha, mtr, mva, fmap, keep, ratio)

    if best is None:
        return {"case": case, "status": "NO_VALID_MODEL"}

    score, tol, k, order, ridge_alpha, mtr, mva, fmap, keep, ratio = best
    # Refit on train+val (still never touching hold-out) with the chosen
    # hyperparameters, exactly mirroring what a real pipeline run would do
    # once hyperparameters are fixed, then score ONCE on the untouched hold-out.
    itrva = np.concatenate([itr, ival])
    keep_full, ratio_full = screen_rows(X[itrva], blocks, tol,
                                         log_path=os.path.join(log_dir, f"screened_rows_case-{case}.csv"),
                                         row_ids=itrva)
    itrva_kept = itrva[keep_full]
    fmap_final = fit_feature_map(X[itrva_kept], y[itrva_kept], blocks, X.shape[1], k=k)
    mtr_final, mho_final = fit_and_score(X[itrva_kept], y[itrva_kept], X[iho], y[iho],
                                          sel, case, order, fmap_final, ridge_alpha, scratch)

    return {
        "case": case, "status": "OK",
        "n_total": len(y), "n_train": len(itr), "n_val": len(ival), "n_holdout": len(iho),
        "tolerance": tol, "rows_kept_frac": float(keep_full.mean()),
        "k": k, "order": order, "ridge_alpha": ridge_alpha,
        "val_selection_train_pct": mtr, "val_selection_val_pct": mva,
        "final_train_pct": mtr_final, "final_holdout_pct": mho_final,
        "n_coef": fmap_final.k if order == 2 else None,
        "pass_2pct": bool(max(mtr_final, mho_final) <= 2.0),
    }


def _run_case_worker(args):
    case, log_dir, data_dir, scratch = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    return run_case(case, rxns, sigma_of_rxn, Temp, log_dir, data_dir=data_dir, scratch=scratch)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('report')/offline_replay/)")
    ap.add_argument("--scratch", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch = args.scratch or config.scratch_dir("offline_replay")
    out_dir = args.output_dir or os.path.join(config.results_dir("report"), "offline_replay")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)

    log_dir = os.path.join(scratch, "screening_logs")
    os.makedirs(log_dir, exist_ok=True)

    import multiprocessing as mp
    import time
    t0 = time.time()
    # 160 cores available; single-threaded BLAS (forced above) + one process
    # per case avoids the thread-oversubscription that made the first attempt
    # at this sweep take >13 min to not even finish case 0.
    with mp.Pool(processes=8) as pool:
        results = pool.map(_run_case_worker, [(c, log_dir, data_dir, scratch) for c in range(8)])
    results = sorted(results, key=lambda r: r["case"])

    for r in results:
        if r["status"] == "OK":
            print(f"case {r['case']}: tol={r['tolerance']:<4} k={r['k']:<3} order={r['order']} "
                  f"ridge={r['ridge_alpha']:.0e}  rows_kept={100*r['rows_kept_frac']:.0f}%  "
                  f"TRAIN={r['final_train_pct']:.3f}%  HOLDOUT={r['final_holdout_pct']:.3f}%  "
                  f"{'PASS' if r['pass_2pct'] else 'FAIL'}")
        else:
            print(f"case {r['case']}: {r['status']}")

    npass = sum(1 for r in results if r.get("pass_2pct"))
    print(f"\n==> {npass}/8 cases meet <=2% on BOTH train and untouched hold-out"
          f"  (wall time {time.time()-t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "offline_replay_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out_path}")
    return results


if __name__ == "__main__":
    main()
