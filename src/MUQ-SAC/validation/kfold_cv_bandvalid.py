"""
10-fold CV, evaluating MRE only over BAND-VALID test-fold points.

Motivation (diagnose_worst_rows.py, 2026-07-30): every worst-fold failure in
kfold_cv.py's initial sweep was traced to a single test-fold row with a
band-validity violation ratio of 7x-30x the sampler's own physical
uncertainty band -- i.e. a row the sampler should never have produced (the
same class-B m=3 rejection-test bug documented in report/main.tex Section
"Recommendations: fixing the sampler at the source", item 1). Training stays
on the FULL pool unchanged (screening training data was already shown not to
help in aggregate, report/, parametric study) -- only the EVALUATION metric
changes: a case's MRE is the worst error over band-VALID test points only,
since a band-invalid row's "ground truth" is not a meaningful target for a
surrogate whose entire purpose is supporting optimization over physically
plausible mechanisms.

Refactor of plumbing only (loaders -> case_data, SCRATCH/CV_SPLIT_SEED ->
config + argparse); run_case/summarize/main's fitting logic is unchanged.
Default HP path is the FINAL production hyperparameters
(report_10fold_validation/results/07_final_10fold/kfold_final_hyperparams.json);
override with --hp-json to re-verify a different hyperparameter set (e.g.
during the search rounds this script also validated intermediate results).

Run: python3.9 kfold_cv_bandvalid.py [--hp-json PATH] [--output-dir DIR]
"""
import argparse
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import os
import json
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import fit_feature_map, screen_rows, violation_detail, _fit_once
import case_data


def run_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=None, scratch=None,
             n_folds=None, seed=None, band_tolerance_eval=None):
    band_tolerance_eval = band_tolerance_eval if band_tolerance_eval is not None else config.BAND_TOLERANCE_EVAL
    pool = case_data.build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
    X, y, sel, blocks = pool["X"], pool["y"], pool["sel"], pool["blocks"]

    ratio, _ = violation_detail(X, blocks)
    band_valid_mask = ratio <= band_tolerance_eval

    n_folds = n_folds if n_folds is not None else config.N_FOLDS
    fold_of = case_data.make_folds(len(y), case, n_folds=n_folds, seed=seed)

    tol, k, order, ridge_alpha = hp["tolerance"], hp["k"], hp["order"], hp["ridge_alpha"]
    case_scratch = os.path.join(scratch, f"case-{case}")
    os.makedirs(os.path.join(case_scratch, "Data", "ResponseSurface"), exist_ok=True)

    fold_results = []
    for f in range(n_folds):
        tr_idx_all = np.where(fold_of != f)[0]
        te_idx_all = np.where(fold_of == f)[0]
        te_idx_valid = te_idx_all[band_valid_mask[te_idx_all]]
        keep_tr, _ = screen_rows(X[tr_idx_all], blocks, tol)
        tr_idx = tr_idx_all[keep_tr]
        Xa, ya = X[tr_idx], y[tr_idx]
        fmap = fit_feature_map(Xa, ya, blocks, X.shape[1], k=k)

        # Score against ALL test points (for comparison) and band-valid-only
        Xb_all, yb_all = X[te_idx_all], y[te_idx_all]
        Xb_valid, yb_valid = X[te_idx_valid], y[te_idx_valid]
        mtr, mte_all = _fit_once(PRSmod.ResponseSurface, case_scratch, Xa, ya, Xb_all, yb_all,
                                  case, case, order, fmap, ridge_alpha, "Partial", sel, 2.0)
        _, mte_valid = _fit_once(PRSmod.ResponseSurface, case_scratch, Xa, ya, Xb_valid, yb_valid,
                                  case, case, order, fmap, ridge_alpha, "Partial", sel, 2.0)
        fold_results.append({
            "fold": f, "n_train": len(tr_idx),
            "n_test_all": len(te_idx_all), "n_test_valid": len(te_idx_valid),
            "n_test_excluded": len(te_idx_all) - len(te_idx_valid),
            "train_pct": mtr, "test_pct_all": mte_all, "test_pct_valid": mte_valid,
            "worst_pct_all": max(mtr, mte_all), "worst_pct_valid": max(mtr, mte_valid),
        })
    return {"case": case, "n_total": len(y), "n_band_invalid_total": int((~band_valid_mask).sum()),
            "hp": hp, "folds": fold_results}


def summarize(result, criterion=None):
    if criterion is None:
        criterion = config.CRITERION
    folds = result["folds"]
    train = np.array([f["train_pct"] for f in folds])
    test_all = np.array([f["test_pct_all"] for f in folds])
    test_valid = np.array([f["test_pct_valid"] for f in folds])
    worst_all = np.array([f["worst_pct_all"] for f in folds])
    worst_valid = np.array([f["worst_pct_valid"] for f in folds])
    n_excluded = sum(f["n_test_excluded"] for f in folds)
    n_test_total = sum(f["n_test_all"] for f in folds)
    return {
        "case": result["case"], "n_total": result["n_total"],
        "n_band_invalid_total": result["n_band_invalid_total"],
        "pct_test_excluded": 100.0 * n_excluded / n_test_total,
        "hp": result["hp"],
        "train_mean": float(train.mean()), "train_max": float(train.max()),
        "test_all_mean": float(test_all.mean()), "test_all_max": float(test_all.max()),
        "test_valid_mean": float(test_valid.mean()), "test_valid_max": float(test_valid.max()),
        "worst_fold_all": float(worst_all.max()),
        "worst_fold_valid": float(worst_valid.max()),
        "pass_all_folds_valid_only": bool(worst_valid.max() <= criterion),
    }


def _worker(args):
    case, hp, data_dir, scratch, n_folds, seed, band_tol = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    return run_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=data_dir, scratch=scratch,
                     n_folds=n_folds, seed=seed, band_tolerance_eval=band_tol)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('tenfold')/07_final_10fold/)")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Default: config.CV_SPLIT_SEED (SAME as kfold_cv.py)")
    ap.add_argument("--band-tolerance-eval", type=float, default=None)
    ap.add_argument("--hp-json", default=None,
                     help="Hyperparameter JSON (default: "
                          "report_10fold_validation/results/07_final_10fold/kfold_final_hyperparams.json)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch = args.scratch or config.scratch_dir("kfold_cv_bandvalid")
    n_folds = args.n_folds if args.n_folds is not None else config.N_FOLDS
    seed = args.seed if args.seed is not None else config.CV_SPLIT_SEED
    band_tol = args.band_tolerance_eval if args.band_tolerance_eval is not None else config.BAND_TOLERANCE_EVAL
    out_dir = args.output_dir or os.path.join(config.results_dir("tenfold"), "07_final_10fold")
    hp_path = args.hp_json or os.path.join(config.results_dir("tenfold"), "07_final_10fold",
                                            "kfold_final_hyperparams.json")

    with open(hp_path) as f:
        raw_hp = json.load(f)
    hp_by_case = {int(k): dict(tolerance=float(v["tolerance"]), k=int(v["k"]),
                                order=int(v["order"]), ridge_alpha=float(v["ridge_alpha"]))
                  for k, v in raw_hp.items() if k != "_provenance"}

    os.makedirs(scratch, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    tasks = [(c, hp_by_case[c], data_dir, scratch, n_folds, seed, band_tol) for c in range(8)]
    with mp.Pool(processes=8) as pool:
        results = pool.map(_worker, tasks)
    results = sorted(results, key=lambda r: r["case"])
    summaries = [summarize(r) for r in results]

    print(f"{'Case':<5}{'#Invalid':<10}{'%TestExcl':<11}{'WorstAll':<10}{'WorstValid':<12}{'Pass(valid-only)'}")
    for s in summaries:
        print(f"{s['case']:<5}{s['n_band_invalid_total']:<10}{s['pct_test_excluded']:<11.1f}"
              f"{s['worst_fold_all']:<10.3f}{s['worst_fold_valid']:<12.3f}"
              f"{'YES' if s['pass_all_folds_valid_only'] else 'NO'}")

    n_ok = sum(1 for s in summaries if s["pass_all_folds_valid_only"])
    print(f"\n==> {n_ok}/8 cases keep EVERY fold <= 2% MRE when scored on BAND-VALID "
          f"test points only (wall time {time.time()-t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kfold_cv_bandvalid_results.json")
    with open(out_path, "w") as f:
        json.dump({"raw": results, "summary": summaries}, f, indent=2)
    print(f"results written to {out_path}")
    return results, summaries


if __name__ == "__main__":
    main()
