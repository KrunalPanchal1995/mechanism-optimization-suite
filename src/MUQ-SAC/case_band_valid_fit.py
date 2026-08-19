"""
Standalone, read-only verification + correlation-aware ablation for the
THREE-PARAM-MSO-LLNL-N-HEPTANE_159_CORRELATION_AWARE study, run 2026-08-19
while diagnosing the 0.42% train / 1107% test MRE blowup.

Two questions, answered from data ALREADY ON DISK -- no new simulations, no
mutation of DM_FOR_PARTIAL_PRS/ or Opt/Data/Simulations/:

1. "aware" fit -- does IdentifiablePRS.py's F1 fix (band-validity scored
   against the TRUE full 3-parameter physical band, not the reduced/
   selected-subspace one -- see reaction_W_fprior's docstring) plus
   screening training rows to band-valid-only (tolerance 1.0) recover
   train/val/hold-out MRE < 2% for all 68 cases? Mirrors exactly what
   run_script.py's per-case loop does with prs_band_fprior: full,
   prs_screening_tolerances: [1.0], prs_band_tolerance_eval: 1.0 (this
   study's target.opt, updated the same day) -- same three_way_split,
   same fit_prs_with_cv_search, same criteria -- just standalone so it can
   run in minutes instead of going through sensitivity analysis /
   correlation analysis / simulation-cache bookkeeping.

2. "aware vs unaware" ablation -- of the parameters correlation-aware
   selection added (sa_pipeline_v2_out/final_active_set_combined.csv,
   in_rescued == True), does the fit get measurably worse if the regressor
   is blinded to exactly those columns? Ablation method: zero the rescued
   columns of X_comp (both when fitting the feature map/PRS AND when
   scoring) -- y is untouched, so this asks "how much of this case's
   ignition-delay variance depends on the parameters correlation-aware
   selection rescued", not a different sampling or fit procedure.

Usage: OMP_NUM_THREADS=1 python3.9 case_band_valid_fit.py <study_dir> <out_csv> [--cases 0,1,2] [--workers N]
"""
import os
import sys
import csv
import json
import pickle
import argparse
import warnings
from math import comb
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IdentifiablePRS as IPRS
import ResponseSurface as PRS


def load_rxns_sigma(D):
    rxns = [l.split("\t")[0].strip() for l in open(D + "ACTIVE_RXN_INDEX.csv")]
    sigma = {r: np.loadtxt(D + f"Uncertainty_Matrices/{r}/Sigma.csv", delimiter=",") for r in rxns}
    return rxns, sigma


def load_temps(D):
    T = {}
    for r in csv.DictReader(open(D + "targets.csv")):
        T[int(r["sr_no"]) - 1] = float(r["Temperature_K"])
    return T


def make_layout(sel, rxns):
    layout = []
    col = 0
    for j, r in enumerate(rxns):
        idx = [p for p in range(3) if sel[3 * j + p] == 1]
        if not idx:
            continue
        layout.append((r, slice(col, col + len(idx)), idx))
        col += len(idx)
    return layout


def load_case_xy(D, case):
    X = np.asarray([np.fromstring(l.strip().rstrip(","), sep=",")
                     for l in open(f"{D}DM_FOR_PARTIAL_PRS/{case}/pDesignMatrix.csv") if l.strip()])
    y = np.asarray([float(l.split()[1])
                     for l in open(f"{D}Opt/Data/Simulations/sim_data_case-{case}.lst") if len(l.split()) >= 2])
    sel = np.fromstring(open(f"{D}DM_FOR_PARTIAL_PRS/{case}/pSelectionMatrix.csv").readline().strip().rstrip(","),
                         sep=",").astype(int)
    n = min(len(X), len(y))
    return X[:n], y[:n], sel


def rescued_columns(D, case, rxns, sel, active_param_map):
    """Compressed-column indices for this case's rescued parameters
    (final_active_set_combined.csv, in_rescued==True), via
    unsrt_data[rxn].activeParameters[0/1/2] -> "R<id>_<A|n|Ea>" strings,
    the SAME strings correlation_analysis.py writes."""
    path = D + "sa_pipeline_v2_out/final_active_set_combined.csv"
    if not os.path.isfile(path):
        return np.array([], dtype=int)
    rescued_names = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["case"] == str(case) and r["in_rescued"] == "True":
                rescued_names.add(r["feature"])
    if not rescued_names:
        return np.array([], dtype=int)
    cols = []
    col = 0
    for j, rxn in enumerate(rxns):
        idx = [p for p in range(3) if sel[3 * j + p] == 1]
        if not idx:
            continue
        names = active_param_map[rxn]  # [A_name, n_name, Ea_name], local index order 0,1,2
        for local_pos, p in enumerate(idx):
            if names[p] in rescued_names:
                cols.append(col + local_pos)
        col += len(idx)
    return np.array(sorted(cols), dtype=int)


def fit_and_score(X, y, blocks, n_comp, case, case_index, MRE_criteria,
                   tol_grid, k_grid, order_grid, ridge_grid, cache_dir,
                   split_seed=104, split_seed_inner=7, val_frac=0.15, holdout_frac=0.15,
                   cv_folds=10, cv_seed=13, band_tolerance_eval=1.0, search_method="cv_kfold",
                   valid_mask=None):
    """
    valid_mask: optional pre-computed boolean mask (len(y),) of physically
    band-valid rows, computed from the case's ORIGINAL (un-ablated) X. When
    given, the row split/screening/eval-filtering below all key off this
    FIXED mask instead of re-deriving band-validity from whatever X is
    passed in here -- essential for the correlation-aware ablation
    (case_band_valid_fit.rescued_columns): zeroing a handful of correlated
    columns can shift a row's OWN violation-ratio a lot (near-cancelling
    Arrhenius directions, see IdentifiablePRS.py's module docstring on
    corr(n, Ea/R) ~ -0.996), so re-screening on the ablated X answers a
    different, confounded question ("how does blinding + re-selecting rows
    change the fit") instead of the intended one ("holding the row set
    fixed, does blinding the regressor to these columns hurt"). tol_grid is
    forced to [inf] and band_tolerance_eval to None internally when
    valid_mask is given, since the filtering already happened.
    """
    warnings.filterwarnings("ignore")
    if valid_mask is not None:
        X, y = X[valid_mask], y[valid_mask]
        tol_grid = [float("inf")]
        band_tolerance_eval = None

    train_frac = 1.0 - val_frac - holdout_frac
    itr, ival, iho = IPRS.three_way_split(len(y), seed_outer=split_seed, seed_inner=split_seed_inner,
                                            train_frac=train_frac, val_frac=val_frac)
    xTrain, yTrain = X[itr], y[itr]
    xVal, yVal = X[ival], y[ival]
    xHold, yHold = X[iho], y[iho]

    sel_params = np.ones(n_comp, dtype=int)  # feature_map path never reads
                                              # this beyond len()/indices, see
                                              # ResponseSurface.__init__
    if search_method == "cv_kfold":
        Xp = np.vstack([xTrain, xVal]); yp = np.concatenate([yTrain, yVal])
        search = IPRS.fit_prs_with_cv_search(
            PRS.ResponseSurface, Xp, yp, blocks, n_comp, case, case_index, "Partial",
            sel_params, MRE_criteria, cache_dir=cache_dir, tol_grid=tol_grid, k_grid=k_grid,
            order_grid=order_grid, ridge_grid=ridge_grid, cv_folds=cv_folds, cv_seed=cv_seed,
            band_tolerance_eval=band_tolerance_eval)
        keep_mask, _ = IPRS.screen_rows(xTrain, blocks, search["tolerance"])
    else:
        search = IPRS.fit_prs_with_search(
            PRS.ResponseSurface, xTrain, yTrain, xVal, yVal, blocks, n_comp, case, case_index,
            "Partial", sel_params, MRE_criteria, cache_dir=cache_dir, tol_grid=tol_grid, k_grid=k_grid,
            order_grid=order_grid, ridge_grid=ridge_grid, band_tolerance_eval=band_tolerance_eval)
        keep_mask = search["keep_mask"]

    xTrainK, yTrainK = xTrain[keep_mask], yTrain[keep_mask]
    fmap = IPRS.fit_feature_map(xTrainK, yTrainK, blocks, n_comp, k=search["k"])
    coef_path = os.path.join(cache_dir, "Data", "ResponseSurface", f"responsecoef_case-{case_index}.csv")
    if os.path.isfile(coef_path):
        os.remove(coef_path)
    # ResponseSurface.create_response_surface() writes/reads paths built from
    # os.getcwd() (Data/ResponseSurface/...) -- must chdir into cache_dir for
    # this fit, same as IdentifiablePRS._fit_once does for the search's own
    # trial fits, or it silently writes into whatever the caller's cwd is.
    cwd0 = os.getcwd()
    os.chdir(cache_dir)
    try:
        resp = PRS.ResponseSurface(xTrainK, yTrainK, case, case_index, responseOrder=search["order"],
                                    prs_type="Partial", selected_params=sel_params, criteria=MRE_criteria,
                                    feature_map=fmap, ridge_alpha=search["ridge_alpha"])
        resp.create_response_surface()
        mtr, mva, _ = resp.test(xVal, yVal)
        vmask = IPRS.band_valid_mask(xVal, blocks, band_tolerance_eval) if band_tolerance_eval is not None else None
        val_bv = float(np.max(np.abs(np.asarray(resp.error_testing_relative)[vmask]))) if (vmask is not None and vmask.any()) else mva
        mtrH, mvaH, _ = resp.test(xHold, yHold)
        hmask = IPRS.band_valid_mask(xHold, blocks, band_tolerance_eval) if band_tolerance_eval is not None else None
        hold_bv = float(np.max(np.abs(np.asarray(resp.error_testing_relative)[hmask]))) if (hmask is not None and hmask.any()) else mvaH
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
    return dict(tolerance=search["tolerance"], k=search["k"], order=search["order"],
                ridge_alpha=search["ridge_alpha"], n_kept=search["n_kept"], n_train=len(xTrain),
                train=mtrH, val=mva, val_bandvalid=val_bv, hold=mvaH, hold_bandvalid=hold_bv)


def run_one_case(args):
    (D, case, tol_grid, k_grid, order_grid, ridge_grid, MRE_criteria, cache_root,
     compute_ablation) = args
    warnings.filterwarnings("ignore")
    rxns, sigma = load_rxns_sigma(D)
    T = load_temps(D)
    X, y, sel = load_case_xy(D, case)
    layout = make_layout(sel, rxns)
    Tc = T[case]
    T_local = np.array([Tc * 0.75, Tc, Tc * 1.25])
    blocks = IPRS.build_layout_blocks(layout, lambda r: sigma[r], T_local, svd_tol=1e-6, fprior_mode="full")
    n_comp = X.shape[1]
    cache_dir = os.path.join(cache_root, f"case-{case}")
    os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)

    # Physical-validity mask computed ONCE from the ORIGINAL (un-ablated) X --
    # both the aware and (if requested) unaware fits below are trained/scored
    # on exactly this row set, so any MRE difference between them reflects
    # only which COLUMNS the regressor was shown, never a different row
    # selection (see fit_and_score's valid_mask docstring for why re-deriving
    # validity from ablated data would confound the comparison).
    valid_mask = IPRS.band_valid_mask(X, blocks, 1.0)
    out_n_valid = int(valid_mask.sum())

    aware = fit_and_score(X, y, blocks, n_comp, case, case, MRE_criteria,
                           tol_grid, k_grid, order_grid, ridge_grid, cache_dir,
                           valid_mask=valid_mask)
    out = dict(case=case, n_rows=len(X), n_band_valid=out_n_valid,
               **{f"aware_{k}": v for k, v in aware.items()})

    if compute_ablation:
        import pickle as _pkl
        with open(D + "unsrt.pkl", "rb") as fh:
            unsrt = _pkl.load(fh)
        active_param_map = {r: unsrt[r].activeParameters for r in rxns if r in unsrt}
        rcols = rescued_columns(D, case, rxns, sel, active_param_map)
        out["n_rescued_cols"] = int(len(rcols))
        if len(rcols) > 0:
            X_unaware = X.copy()
            X_unaware[:, rcols] = 0.0
            unaware = fit_and_score(X_unaware, y, blocks, n_comp, case, case, MRE_criteria,
                                     tol_grid, k_grid, order_grid, ridge_grid, cache_dir,
                                     valid_mask=valid_mask)
            out.update({f"unaware_{k}": v for k, v in unaware.items()})
        else:
            out.update({f"unaware_{k}": out.get(f"aware_{k}") for k in aware})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--cases", default=None, help="comma-separated case indices; default all found")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    D = args.study_dir.rstrip("/") + "/"
    if args.cases:
        cases = [int(c) for c in args.cases.split(",")]
    else:
        cases = sorted(int(d) for d in os.listdir(D + "DM_FOR_PARTIAL_PRS")
                        if os.path.isdir(D + "DM_FOR_PARTIAL_PRS/" + d) and d.isdigit())

    tol_grid = [1.0]
    k_grid = [8, 10, 12, 14, 16, 20]
    order_grid = [2, 3]
    ridge_grid = [0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5]
    MRE_criteria = 2.0
    cache_root = "/tmp/claude-1000/case_band_valid_fit_cache"
    os.makedirs(cache_root, exist_ok=True)

    tasks = [(D, c, tol_grid, k_grid, order_grid, ridge_grid, MRE_criteria, cache_root,
              not args.no_ablation) for c in cases]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_case, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[case {c}] train={r['aware_train']:.3f}% val_bv={r['aware_val_bandvalid']:.3f}% "
                      f"hold_bv={r['aware_hold_bandvalid']:.3f}% n_rescued={r.get('n_rescued_cols','-')}", flush=True)
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
