"""
For cases 0, 3, 4, 5 (still failing the 10-fold <=2% check), identify the
SPECIFIC rows responsible for each fold's worst error and check whether they
are band-invalid (violation_detail ratio > 1.0, i.e. outside the sampler's
own physical uncertainty band) -- if the worst offenders are band-invalid,
the "instability" is arguably unreliable ground-truth data, not a model
deficiency, and the fix is a full-pool band-validity filter, not more
regularization/capacity search.

Refactor note: the original test_build/diagnose_worst_rows.py loaded its
hyperparameters at MODULE scope (`HP = json.load(open("/tmp/..."))`), which
meant merely IMPORTING the module (e.g. from a test collector) would crash
if that /tmp path didn't exist. That load now happens inside main(), and the
default path points at the pinned results/ snapshot instead of /tmp.

Additionally (ask #2, "persist study artifacts"): besides printing, this
now writes results_dir("tenfold")/worst_rows_diagnosis.json with per-case
per-fold worst row id, error%, violation ratio, and worst reaction name.

Run: python3.9 diagnose_worst_rows.py [--hp-json PATH] [--cases 0,3,4,5]
"""
import argparse
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import fit_feature_map, screen_rows, violation_detail, build_layout_blocks
import case_data

DEFAULT_CASES = [0, 3, 4, 5]


def diagnose_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=None, scratch=None,
                   n_folds=None, seed=None):
    tol, k, order, ridge = float(hp["tolerance"]), int(hp["k"]), int(hp["order"]), float(hp["ridge_alpha"])

    pool = case_data.build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
    X, y, sel, blocks = pool["X"], pool["y"], pool["sel"], pool["blocks"]

    ratio, worst_rxn = violation_detail(X, blocks)

    n_folds = n_folds if n_folds is not None else config.N_FOLDS
    fold_of = case_data.make_folds(len(y), case, n_folds=n_folds, seed=seed)

    case_scratch = os.path.join(scratch, f"case-{case}")
    os.makedirs(os.path.join(case_scratch, "Data", "ResponseSurface"), exist_ok=True)

    print(f"\n=== Case {case} (tol={tol}, k={k}, order={order}, ridge={ridge:.0e}) ===")
    per_fold = []
    worst_overall = (-1.0, None, None, None)
    for f in range(n_folds):
        tr_idx_all = np.where(fold_of != f)[0]
        te_idx = np.where(fold_of == f)[0]
        keep_tr, _ = screen_rows(X[tr_idx_all], blocks, tol)
        tr_idx = tr_idx_all[keep_tr]
        fmap = fit_feature_map(X[tr_idx], y[tr_idx], blocks, X.shape[1], k=k)

        coef_path = os.path.join(case_scratch, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        cwd0 = os.getcwd(); os.chdir(case_scratch)
        try:
            rs = PRSmod.ResponseSurface(X[tr_idx], y[tr_idx], case, case, responseOrder=order,
                                         prs_type="Partial", selected_params=sel, criteria=2.0,
                                         feature_map=fmap, ridge_alpha=ridge)
            rs.create_response_surface()
            mtr, mte, _ = rs.test(X[te_idx], y[te_idx])
            rel = np.asarray(rs.error_testing_relative, dtype=float)
        finally:
            os.chdir(cwd0)
            if os.path.isfile(coef_path):
                os.remove(coef_path)

        worst_local_i = int(np.argmax(np.abs(rel)))
        worst_row_global = int(te_idx[worst_local_i])
        worst_err = float(np.abs(rel[worst_local_i]))
        worst_viol = float(ratio[worst_row_global])
        worst_rxn_name = worst_rxn[worst_row_global]
        if worst_err > worst_overall[0]:
            worst_overall = (worst_err, worst_row_global, worst_viol, worst_rxn_name)
        flag = "BAND-INVALID" if worst_viol > 1.0 else "band-valid"
        print(f"  fold {f}: test_max={mte:.3f}%  worst_row={worst_row_global}  "
              f"violation_ratio={worst_viol:.2f} ({flag})  worst_rxn={worst_rxn_name}")
        per_fold.append({
            "fold": f, "test_max_pct": float(mte), "worst_row": worst_row_global,
            "worst_error_pct": worst_err, "violation_ratio": worst_viol,
            "band_invalid": bool(worst_viol > 1.0), "worst_rxn": worst_rxn_name,
        })

    print(f"  -> overall worst: row {worst_overall[1]}, err {worst_overall[0]:.2f}%, "
          f"violation_ratio {worst_overall[2]:.2f}")

    return {
        "case": case, "hp": hp, "per_fold": per_fold,
        "overall_worst": {
            "row": worst_overall[1], "error_pct": worst_overall[0],
            "violation_ratio": worst_overall[2], "worst_rxn": worst_overall[3],
        },
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('tenfold')/)")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cases", default=",".join(str(c) for c in DEFAULT_CASES),
                     help=f"Comma-separated case ids (default: {DEFAULT_CASES})")
    ap.add_argument("--hp-json", default=None,
                     help="Hyperparameter JSON (default: "
                          "report_10fold_validation/results/02_robust_search_round1/chosen_hyperparams_merged.json)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch = args.scratch or config.scratch_dir("diagnose_worst_rows")
    n_folds = args.n_folds if args.n_folds is not None else config.N_FOLDS
    seed = args.seed if args.seed is not None else config.CV_SPLIT_SEED
    out_dir = args.output_dir or config.results_dir("tenfold")
    cases = [int(c) for c in args.cases.split(",") if c.strip() != ""]
    hp_path = args.hp_json or os.path.join(config.results_dir("tenfold"), "02_robust_search_round1",
                                            "chosen_hyperparams_merged.json")

    with open(hp_path) as f:
        HP = json.load(f)

    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)

    per_case = []
    for case in cases:
        hp = HP[str(case)]
        per_case.append(diagnose_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=data_dir,
                                       scratch=scratch, n_folds=n_folds, seed=seed))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "worst_rows_diagnosis.json")
    with open(out_path, "w") as f:
        json.dump({"cases": per_case}, f, indent=2)
    print(f"\nresults written to {out_path}")
    return per_case


if __name__ == "__main__":
    main()
