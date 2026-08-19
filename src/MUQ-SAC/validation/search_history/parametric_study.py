"""
Parametric hyperparameter study (Task #11 of the PAAR plan).

Full-grid sweep of Stage 1+2 (band-validity tolerance c, identifiable-
subspace dimension k, polynomial order, ridge alpha), fitting the REAL
ResponseSurface class for every combination on every case, and logging
EVERY result (not just the best per case, unlike offline_replay.py, which
this module historically reused for data loading / splitting / layout
construction -- now both share validation/case_data.py instead).
Zero additional simulations -- entirely off the cached 55,045-row dataset.

Output: parametric_study_full.csv (one row per case x combo) plus a
per-case "best" summary (parametric_study_best.csv), both consumed by the
LaTeX report generator. Feeds report/main.tex's sensitivity tables (NOT
report_10fold_validation/) -- output goes to report/results/parametric_study/.

Fitting/sweep logic is UNCHANGED from test_build/parametric_study.py -- only
the loader block (-> case_data) and output path were refactored.

Run: python3.9 parametric_study.py
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import build_layout_blocks, fit_feature_map, screen_rows, three_way_split
import case_data

TOLERANCES = [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")]
K_GRID     = [8, 10, 12, 14, 16, 20]
ORDER_GRID = [2, 3]
RIDGE_GRID = [0.0, 1e-9, 1e-7, 1e-5]


def fit_and_score(Xtr, ytr, Xte, yte, sel, case, order, feature_map, ridge_alpha, scratch):
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


def run_case(case, rxns, sigma_of_rxn, Temp, data_dir=None, scratch_root=None):
    scratch = os.path.join(scratch_root, f"case-{case}")
    os.makedirs(os.path.join(scratch, "Data", "ResponseSurface"), exist_ok=True)

    X, y, sel = case_data.load_case(case, data_dir=data_dir)
    layout = case_data.make_layout(sel, rxns)
    Tc = Temp[case]
    T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
    blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)

    itr, ival, iho = three_way_split(len(y))
    rows = []
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
                from math import comb
                n_terms = sum(comb(fmap.k + d - 1, d) for d in range(order + 1))
                if n_terms > 0.4 * len(itr_kept):
                    continue
                for ridge_alpha in RIDGE_GRID:
                    mtr, mva = fit_and_score(X[itr_kept], y[itr_kept], X[ival], y[ival],
                                              sel, case, order, fmap, ridge_alpha, scratch)
                    rows.append({
                        "case": case, "tolerance": tol, "k": k, "order": order,
                        "ridge_alpha": ridge_alpha, "rows_kept": int(len(itr_kept)),
                        "rows_available": int(len(itr)), "n_coef": int(n_terms),
                        "train_pct": mtr, "val_pct": mva,
                        "pass_2pct_trainval": bool(max(mtr, mva) <= 2.0),
                    })
    # Best combo by val-selection criterion -> also report its HOLD-OUT number
    # (touched only here, once, for this one designated "winner" per case).
    if rows:
        # NOTE: explicit dict() copy -- `best` must NOT alias an entry already
        # in `rows` (which the caller extends into the full-grid CSV below),
        # or annotating it with holdout-only keys corrupts that row too.
        best = dict(min(rows, key=lambda r: max(r["train_pct"], r["val_pct"])))
        itrva = np.concatenate([itr, ival])
        keep_full, _ = screen_rows(X[itrva], blocks, best["tolerance"])
        itrva_kept = itrva[keep_full]
        fmap_final = fit_feature_map(X[itrva_kept], y[itrva_kept], blocks, X.shape[1], k=best["k"])
        mtr_f, mho_f = fit_and_score(X[itrva_kept], y[itrva_kept], X[iho], y[iho],
                                      sel, case, best["order"], fmap_final, best["ridge_alpha"], scratch)
        best["holdout_pct"] = mho_f
        best["final_train_pct"] = mtr_f
        best["pass_2pct_final"] = bool(max(mtr_f, mho_f) <= 2.0)
    return rows, (best if rows else None)


def _worker(args):
    case, data_dir, scratch_root = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    return run_case(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir, scratch_root=scratch_root)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('report')/parametric_study/)")
    ap.add_argument("--scratch", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch_root = args.scratch or config.scratch_dir("parametric_study")
    out_dir = args.output_dir or os.path.join(config.results_dir("report"), "parametric_study")

    import multiprocessing as mp
    import time
    t0 = time.time()
    with mp.Pool(processes=8) as pool:
        results = pool.map(_worker, [(c, data_dir, scratch_root) for c in range(8)])

    all_rows = []
    best_rows = []
    for rows, best in results:
        all_rows.extend(rows)
        if best is not None:
            best_rows.append(best)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "parametric_study_full.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"full grid ({len(all_rows)} rows) -> {out_csv}")

    best_csv = os.path.join(out_dir, "parametric_study_best.csv")
    with open(best_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(best_rows, key=lambda r: r["case"]))
    print(f"per-case best ({len(best_rows)} rows) -> {best_csv}")

    npass = sum(1 for r in best_rows if r.get("pass_2pct_final"))
    print(f"\n==> {npass}/8 cases <=2% (train & hold-out) at their best Stage1+2 combo, "
          f"wall time {time.time()-t0:.0f}s")
    for r in sorted(best_rows, key=lambda r: r["case"]):
        print(f"  case {r['case']}: tol={r['tolerance']} k={r['k']} order={r['order']} "
              f"ridge={r['ridge_alpha']:.0e} train={r['final_train_pct']:.3f}% "
              f"holdout={r['holdout_pct']:.3f}%  {'PASS' if r['pass_2pct_final'] else 'FAIL'}")


if __name__ == "__main__":
    main()
