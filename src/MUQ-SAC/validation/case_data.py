"""
The ONE shared data-loading module for the 10-fold CV / search / offline-
replay / parametric-study scripts.

Replaces 4 duplicated copies of {load_case, load_outlier_pool,
load_sigma_and_rxns, load_temps, make_layout} (previously pasted verbatim
into kfold_cv.py, offline_replay.py / offline_replay_ext.py,
offline_cv_search.py, and parametric_study.py) and ~7 duplicated inline
copies of the fold-split recipe (previously in kfold_cv.py,
kfold_cv_bandvalid.py, kfold_cv_verify.py (via kfold_cv.run_case),
diagnose_worst_rows.py, plot_parity_bad_fold.py, kfold_robust_search.py,
kfold_robust_search2.py, kfold_robust_search3.py).

`make_folds` in particular is lifted VERBATIM from kfold_cv.py's original
run_case (byte-for-byte the same rng/perm/modulo recipe) -- reordering or
"simplifying" it would silently reassign fold membership and change every
published number downstream. See validation/README.md verification section
for the exact byte-identical-JSON diff this was checked against.

Every loader is parameterized on data_dir=None, defaulting to
config.DATA_DIR, so callers can point at a different dataset without editing
this file.
"""
import os
import csv

import numpy as np

import config


def load_case(case, data_dir=None):
    """Original main design matrix + simulation outputs + selection mask for
    `case`, from the cached DM_FOR_PARTIAL_PRS / Opt/Data/Simulations trees.
    Returns (X, y, sel)."""
    D = data_dir if data_dir is not None else config.DATA_DIR
    X = np.asarray([np.fromstring(l.strip().rstrip(","), sep=",")
                     for l in open(f"{D}DM_FOR_PARTIAL_PRS/{case}/pDesignMatrix.csv") if l.strip()])
    y = np.asarray([float(l.split()[1])
                     for l in open(f"{D}Opt/Data/Simulations/sim_data_case-{case}.lst") if len(l.split()) >= 2])
    sel = np.fromstring(open(f"{D}DM_FOR_PARTIAL_PRS/{case}/pSelectionMatrix.csv").readline().strip().rstrip(","),
                         sep=",").astype(int)
    return X, y, sel


def load_outlier_pool(case, data_dir=None):
    """Accepted D-optimal outlier-resample rows for `case`, if any (only
    Case 3 in the final production run). Returns (X_extra, y_extra), each
    possibly empty."""
    D = data_dir if data_dir is not None else config.DATA_DIR
    sim_dir = os.path.join(D, "Opt", "Data", "Outlier_Simulations")
    Xs, ys = [], []
    pass_i = 1
    while True:
        dm_path = os.path.join(sim_dir, f"Outlier_DM_case-{case}_pass-{pass_i}.csv")
        sim_path = os.path.join(sim_dir, f"sim_data_case-{case}_pass-{pass_i}.lst")
        if not (os.path.isfile(dm_path) and os.path.isfile(sim_path)):
            break
        Xp = np.asarray([np.fromstring(l.strip().rstrip(","), sep=",")
                          for l in open(dm_path) if l.strip()])
        yp = np.asarray([float(l.split()[1]) for l in open(sim_path) if len(l.split()) >= 2])
        n = min(len(Xp), len(yp))
        Xs.append(Xp[:n]); ys.append(yp[:n])
        pass_i += 1
    if not Xs:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack(Xs), np.concatenate(ys)


def load_sigma_and_rxns(data_dir=None):
    D = data_dir if data_dir is not None else config.DATA_DIR
    rxns = [l.split("\t")[0].strip() for l in open(D + "ACTIVE_RXN_INDEX.csv")]
    sigma_of_rxn = {}
    for rxn in rxns:
        path = f'{D}Uncertainty_Matrices/{rxn}/Sigma.csv'
        if os.path.isfile(path):
            sigma_of_rxn[rxn] = np.loadtxt(path, delimiter=",")
    return rxns, sigma_of_rxn


def load_temps(data_dir=None):
    D = data_dir if data_dir is not None else config.DATA_DIR
    Temp = {}
    with open(D + "targets.csv") as f:
        for r in csv.DictReader(f):
            Temp[int(r["sr_no"]) - 1] = float(r["Temperature_K"])
    return Temp


def make_layout(sel, rxns):
    layout = []
    col = 0
    for j, rxn in enumerate(rxns):
        idx = [p for p in range(3) if sel[3 * j + p] == 1]
        if not idx:
            continue
        layout.append((rxn, slice(col, col + len(idx)), idx))
        col += len(idx)
    return layout


def load_final_hyperparams(path=None):
    """Per-case (tolerance, k, order, ridge) chosen by the last production
    run, read from Prediction_errors_detailed.csv. Default path is the
    pinned snapshot under report_10fold_validation/results/inputs/ (the live
    production copy under DATA_DIR is mutable -- the next production run
    silently overwrites it, changing "baseline" forever)."""
    if path is None:
        path = os.path.join(config.results_dir("tenfold"), "inputs", "Prediction_errors_detailed.csv")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            case = int(r["Case_ID"])
            tol = float(r["Tolerance"])
            out[case] = dict(tolerance=tol, k=int(r["K"]), order=int(r["Order"]),
                              ridge_alpha=float(r["Ridge_Alpha"]))
    return out


def build_case_pool(case, rxns, sigma_of_rxn, Temp, data_dir=None):
    """Assemble the full (original + accepted outlier-resample) training
    pool for `case`: X, y, sel, blocks, layout, T_arr. This is the ~5-7x
    duplicated assembly block lifted into one place.

    `blocks` requires IdentifiablePRS.build_layout_blocks -- imported lazily
    inside this function so case_data.py itself has no hard dependency on
    the production tree being on sys.path yet (callers are expected to have
    already run `import _bootstrap` / `_bootstrap.add_muqsac_to_path()`
    before calling this)."""
    from IdentifiablePRS import build_layout_blocks

    X0, y0, sel = load_case(case, data_dir=data_dir)
    Xo, yo = load_outlier_pool(case, data_dir=data_dir)
    if len(yo) > 0:
        X = np.vstack([X0, Xo])
        y = np.concatenate([y0, yo])
    else:
        X, y = X0, y0

    layout = make_layout(sel, rxns)
    Tc = Temp[case]
    T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
    blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)

    return dict(X=X, y=y, sel=sel, blocks=blocks, layout=layout, T_arr=T_arr,
                n_outlier_rows=len(yo))


def make_folds(n, case, n_folds=None, seed=None):
    """The exact fold-split recipe, lifted VERBATIM from kfold_cv.py's
    original run_case:

        rng = np.random.default_rng(seed + case)
        perm = rng.permutation(n)
        fold_of[perm] = np.arange(n) % n_folds

    THE SINGLE MOST IMPORTANT CORRECTNESS CONSTRAINT IN THIS MODULE: if this
    doesn't reproduce byte-identical fold membership to the original
    scripts, every downstream published number (worst-fold errors,
    hyperparameter search results, verification results) silently changes.
    Do not "simplify" or reorder these three lines.

    n_folds defaults to config.N_FOLDS, seed defaults to config.CV_SPLIT_SEED
    (callers doing a search or verification pass with a different seed, e.g.
    config.SEARCH_SPLIT_SEED, must pass it explicitly -- there is no implicit
    per-caller default beyond the baseline CV_SPLIT_SEED).

    Returns fold_of: an (n,)-length int array, fold_of[i] = which fold row i
    belongs to.
    """
    if n_folds is None:
        n_folds = config.N_FOLDS
    if seed is None:
        seed = config.CV_SPLIT_SEED
    rng = np.random.default_rng(seed + case)
    perm = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % n_folds
    return fold_of
