"""
10-fold cross-validation of the FINAL production configuration (design matrix
and simulations already done -- zero new simulations here).

For each of the 8 cases in TEST_CORRELATIONS_CORRECTED, this reconstructs the
exact FINAL training pool the production pipeline ended up using:
  - the original main design matrix (cached, DM_FOR_PARTIAL_PRS / Opt/Data/Simulations)
  - PLUS, for Case 3 only, the 1,000 accepted D-optimal outlier-resample rows
    (Opt/Data/Outlier_Simulations/Outlier_DM_case-3_pass-{1,2}.csv +
    sim_data_case-3_pass-{1,2}.lst) -- the only case whose retraining pass was
    accepted in the final run.
and the FINAL chosen hyperparameters (tolerance, k, order, ridge) per case,
read from Prediction_errors_detailed.csv (written by the last production run;
default here is the pinned inputs/ snapshot -- see --hp-csv).

10-fold CV protocol (mirrors fit_prs_with_search's leakage-safe design):
for each fold, tolerance-screening is applied ONLY to the 9 training folds;
the held-out fold is scored unscreened, exactly as production scores its
untouched hold-out. The feature map (PLS rotation) is refit fresh on each
fold's own screened training data -- never on data that includes the held-out
fold -- so no fold's evaluation is contaminated by its own label information.

This is a refactor of PLUMBING ONLY (loaders -> case_data, D/SCRATCH/N_FOLDS/
CV_SPLIT_SEED -> config defaults + argparse overrides, output path ->
config.results_dir("tenfold")/01_baseline_10fold/); run_case/summarize/main's
actual fitting logic is unchanged from test_build/kfold_cv.py.

Run: python3.9 kfold_cv.py [--hp-csv PATH] [--data-dir DIR] [--results-dir DIR]
                            [--scratch DIR] [--n-folds N] [--seed N]
"""
import argparse
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import os
import sys
import json
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import _bootstrap
import ResponseSurface as PRSmod
from IdentifiablePRS import build_layout_blocks, fit_feature_map, screen_rows, _fit_once
import case_data


def run_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=None, scratch=None,
             n_folds=None, seed=None):
    pool = case_data.build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
    X, y, sel, blocks = pool["X"], pool["y"], pool["sel"], pool["blocks"]

    n_folds = n_folds if n_folds is not None else config.N_FOLDS
    fold_of = case_data.make_folds(len(y), case, n_folds=n_folds, seed=seed)

    tol, k, order, ridge_alpha = hp["tolerance"], hp["k"], hp["order"], hp["ridge_alpha"]
    case_scratch = os.path.join(scratch, f"case-{case}")
    os.makedirs(os.path.join(case_scratch, "Data", "ResponseSurface"), exist_ok=True)

    fold_results = []
    for f in range(n_folds):
        tr_idx_all = np.where(fold_of != f)[0]
        te_idx = np.where(fold_of == f)[0]
        keep_tr, _ = screen_rows(X[tr_idx_all], blocks, tol)
        tr_idx = tr_idx_all[keep_tr]
        Xa, ya = X[tr_idx], y[tr_idx]
        Xb, yb = X[te_idx], y[te_idx]
        fmap = fit_feature_map(Xa, ya, blocks, X.shape[1], k=k)
        mtr, mte = _fit_once(PRSmod.ResponseSurface, case_scratch, Xa, ya, Xb, yb,
                              case, case, order, fmap, ridge_alpha, "Partial", sel, 2.0)
        fold_results.append({"fold": f, "n_train": len(tr_idx), "n_test": len(te_idx),
                              "train_pct": mtr, "test_pct": mte,
                              "worst_pct": max(mtr, mte)})
    return {"case": case, "n_total": len(y), "n_outlier_rows": pool["n_outlier_rows"],
            "hp": hp, "folds": fold_results}


def _worker(args):
    case, hp, data_dir, scratch, n_folds, seed = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    return run_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=data_dir,
                     scratch=scratch, n_folds=n_folds, seed=seed)


def summarize(result, criterion=None):
    if criterion is None:
        criterion = config.CRITERION
    folds = result["folds"]
    train = np.array([f["train_pct"] for f in folds])
    test = np.array([f["test_pct"] for f in folds])
    worst = np.array([f["worst_pct"] for f in folds])
    return {
        "case": result["case"], "n_total": result["n_total"],
        "n_outlier_rows": result["n_outlier_rows"], "hp": result["hp"],
        "train_mean": float(train.mean()), "train_std": float(train.std()),
        "train_min": float(train.min()), "train_max": float(train.max()),
        "test_mean": float(test.mean()), "test_std": float(test.std()),
        "test_min": float(test.min()), "test_max": float(test.max()),
        "worst_fold_value": float(worst.max()), "worst_fold_index": int(np.argmax(worst)),
        "n_folds_exceeding": int((worst > criterion).sum()),
        "pass_all_folds": bool(worst.max() <= criterion),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="Override config.DATA_DIR")
    ap.add_argument("--results-dir", default=None,
                     help="Override output dir (default: config.results_dir('tenfold')/01_baseline_10fold/)")
    ap.add_argument("--scratch", default=None, help="Override config.scratch_dir('kfold_cv')")
    ap.add_argument("--n-folds", type=int, default=None, help="Override config.N_FOLDS")
    ap.add_argument("--seed", type=int, default=None, help="Override config.CV_SPLIT_SEED")
    ap.add_argument("--hp-csv", default=None,
                     help="Prediction_errors_detailed.csv path (default: pinned "
                          "report_10fold_validation/results/inputs/ snapshot)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch = args.scratch or config.scratch_dir("kfold_cv")
    n_folds = args.n_folds if args.n_folds is not None else config.N_FOLDS
    seed = args.seed if args.seed is not None else config.CV_SPLIT_SEED
    out_dir = args.results_dir or os.path.join(config.results_dir("tenfold"), "01_baseline_10fold")

    hp_by_case = case_data.load_final_hyperparams(path=args.hp_csv)

    t0 = time.time()
    import multiprocessing as mp
    tasks = [(c, hp_by_case[c], data_dir, scratch, n_folds, seed) for c in range(8)]
    with mp.Pool(processes=8) as pool:
        results = pool.map(_worker, tasks)
    results = sorted(results, key=lambda r: r["case"])

    summaries = [summarize(r) for r in results]
    print(f"{'Case':<5}{'Tol':<7}{'k':<4}{'TrainMean':<11}{'TrainMax':<10}"
          f"{'TestMean':<10}{'TestMax':<10}{'WorstFold':<11}{'#>2%':<6}{'AllFolds<=2%'}")
    for s in summaries:
        tol = s["hp"]["tolerance"]
        tol_s = "inf" if tol == float("inf") else f"{tol:g}"
        print(f"{s['case']:<5}{tol_s:<7}{s['hp']['k']:<4}{s['train_mean']:<11.3f}"
              f"{s['train_max']:<10.3f}{s['test_mean']:<10.3f}{s['test_max']:<10.3f}"
              f"{s['worst_fold_value']:<11.3f}{s['n_folds_exceeding']:<6}"
              f"{'YES' if s['pass_all_folds'] else 'NO'}")

    n_ok = sum(1 for s in summaries if s["pass_all_folds"])
    print(f"\n==> {n_ok}/8 cases keep EVERY fold <= 2% MRE (wall time {time.time()-t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kfold_cv_results.json")
    with open(out_path, "w") as f:
        json.dump({"raw": results, "summary": summaries}, f, indent=2)
    print(f"results written to {out_path}")
    return results, summaries


if __name__ == "__main__":
    main()
