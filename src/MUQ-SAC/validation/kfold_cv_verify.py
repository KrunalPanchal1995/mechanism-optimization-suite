"""
Verify the robustly-selected hyperparameters (kfold_robust_search.py's
output, chosen by minimizing worst-of-10-fold error on an INDEPENDENT split,
seed=config.SEARCH_SPLIT_SEED=555) against kfold_cv.py's own 10-fold split
(seed=config.CV_SPLIT_SEED=2026) -- i.e. a different, held-out fold
assignment than the one used for selection, so this is a genuine
out-of-selection check, not circular.

Refactor note: the original test_build/kfold_cv_verify.py mutated
`kfold_cv.SCRATCH` on the imported module from inside its multiprocessing
worker (`import kfold_cv as kc; kc.SCRATCH = SCRATCH`) to redirect
run_case's scratch directory without changing kfold_cv.py's signature. Now
that validation/kfold_cv.py's run_case takes `scratch` as a real parameter,
that module-mutation hack is no longer needed and has been dropped.

Run: python3.9 kfold_cv_verify.py [--hp-json PATH] [--output-dir DIR]
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
import case_data
from kfold_cv import run_case, summarize


def _worker(args):
    case, hp, data_dir, scratch, n_folds, seed = args
    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    return run_case(case, hp, rxns, sigma_of_rxn, Temp, data_dir=data_dir,
                     scratch=scratch, n_folds=n_folds, seed=seed)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None,
                     help="Override output dir (default: config.results_dir('tenfold')/05_verify_independent_split/)")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                     help="Default: config.VERIFY_SPLIT_SEED (== config.CV_SPLIT_SEED=2026 -- "
                          "SAME split as kfold_cv.py's own baseline evaluation, deliberately "
                          "DIFFERENT from the search's SEARCH_SPLIT_SEED=555)")
    ap.add_argument("--hp-json", default=None,
                     help="Hyperparameter JSON to verify (default: "
                          "report_10fold_validation/results/02_robust_search_round1/chosen_hyperparams.json)")
    ap.add_argument("--out-name", default="kfold_cv_verify_results.json",
                     help="Output filename (use kfold_cv_verify2_results.json for a second run)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    scratch = args.scratch or config.scratch_dir("kfold_cv_verify")
    n_folds = args.n_folds if args.n_folds is not None else config.N_FOLDS
    seed = args.seed if args.seed is not None else config.VERIFY_SPLIT_SEED
    out_dir = args.output_dir or os.path.join(config.results_dir("tenfold"), "05_verify_independent_split")
    hp_path = args.hp_json or os.path.join(config.results_dir("tenfold"), "02_robust_search_round1",
                                            "chosen_hyperparams.json")

    with open(hp_path) as f:
        raw_hp = json.load(f)
    hp_by_case = {}
    for k, v in raw_hp.items():
        if k == "_provenance":
            continue
        hp_by_case[int(k)] = dict(tolerance=float(v["tolerance"]), k=int(v["k"]),
                                   order=int(v["order"]), ridge_alpha=float(v["ridge_alpha"]))

    os.makedirs(scratch, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    tasks = [(c, hp_by_case[c], data_dir, scratch, n_folds, seed) for c in range(8)]
    with mp.Pool(processes=8) as pool:
        results = pool.map(_worker, tasks)
    results = sorted(results, key=lambda r: r["case"])

    summaries = [summarize(r) for r in results]
    print(f"{'Case':<5}{'Tol':<7}{'k':<4}{'ridge':<9}{'TrainMean':<11}{'TrainMax':<10}"
          f"{'TestMean':<10}{'TestMax':<10}{'WorstFold':<11}{'#>2%':<6}{'AllFolds<=2%'}")
    for s in summaries:
        tol = s["hp"]["tolerance"]
        tol_s = "inf" if tol == float("inf") else f"{tol:g}"
        print(f"{s['case']:<5}{tol_s:<7}{s['hp']['k']:<4}{s['hp']['ridge_alpha']:<9.0e}"
              f"{s['train_mean']:<11.3f}{s['train_max']:<10.3f}{s['test_mean']:<10.3f}"
              f"{s['test_max']:<10.3f}{s['worst_fold_value']:<11.3f}{s['n_folds_exceeding']:<6}"
              f"{'YES' if s['pass_all_folds'] else 'NO'}")

    n_ok = sum(1 for s in summaries if s["pass_all_folds"])
    print(f"\n==> {n_ok}/8 cases keep EVERY fold <= 2% MRE on this INDEPENDENT verification "
          f"split (wall time {time.time()-t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump({"raw": results, "summary": summaries}, f, indent=2)
    print(f"results written to {out_path}")
    return results, summaries


if __name__ == "__main__":
    main()
