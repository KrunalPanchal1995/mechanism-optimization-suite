"""
Dump raw (y_simulated, y_predicted) pairs for ONE case, in three fit
variants -- "before" (reduced f_prior metric, full tolerance search, scored
all-points -- the original buggy configuration), "after"/aware (full f_prior
metric, band-valid rows only -- this investigation's fix), and "unaware"
(same as "after" but with this case's correlation-rescued columns zeroed) --
for the journal-style parity plots in VisualAid.py's
ParityBeforeAfterPlotter / CorrelationAwareParityPlotter. Companion to
case_band_valid_fit.py / case_before_fit.py, which report only summary MRE
numbers; this script exists purely to get the point clouds those figures
need, for a small number of representative cases (not all 68 -- the report
figures only need 2-4 cases to make the point).

Usage: OMP_NUM_THREADS=1 python3.9 case_parity_dump.py <study_dir> <case> <out_json>
"""
import os
import sys
import csv
import json
import pickle
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import IdentifiablePRS as IPRS
import ResponseSurface as PRS
from case_band_valid_fit import load_rxns_sigma, load_temps, make_layout, load_case_xy, rescued_columns

warnings.filterwarnings("ignore")


def fit_variant(X, y, blocks, n_comp, case, k, order, ridge_alpha, cache_dir, seed_outer=104, seed_inner=7):
    itr, ival, iho = IPRS.three_way_split(len(y), seed_outer=seed_outer, seed_inner=seed_inner,
                                            train_frac=0.70, val_frac=0.15)
    sel_params = np.ones(n_comp, dtype=int)
    fmap = IPRS.fit_feature_map(X[itr], y[itr], blocks, n_comp, k=k)
    coef_path = os.path.join(cache_dir, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
    os.makedirs(os.path.dirname(coef_path), exist_ok=True)
    if os.path.isfile(coef_path):
        os.remove(coef_path)
    cwd0 = os.getcwd()
    os.chdir(cache_dir)
    try:
        resp = PRS.ResponseSurface(X[itr], y[itr], case, case, responseOrder=order, prs_type="Partial",
                                    selected_params=sel_params, criteria=2.0, feature_map=fmap,
                                    ridge_alpha=ridge_alpha)
        resp.create_response_surface()
        y_pred_train = np.asarray(resp.resFramWrk, dtype=float)
        _, _, _ = resp.test(X[ival], y[ival])
        y_pred_val = np.asarray(resp.y_Test_Predict, dtype=float)
        _, _, _ = resp.test(X[iho], y[iho])
        y_pred_hold = np.asarray(resp.y_Test_Predict, dtype=float)
    finally:
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
    return dict(y_train=y[itr].tolist(), yhat_train=y_pred_train.tolist(),
                y_val=y[ival].tolist(), yhat_val=y_pred_val.tolist(),
                y_hold=y[iho].tolist(), yhat_hold=y_pred_hold.tolist())


def main():
    D = sys.argv[1].rstrip("/") + "/"
    case = int(sys.argv[2])
    out_path = sys.argv[3]

    rxns, sigma = load_rxns_sigma(D)
    T = load_temps(D)
    X, y, sel = load_case_xy(D, case)
    layout = make_layout(sel, rxns)
    Tc = T[case]
    T_local = np.array([Tc * 0.75, Tc, Tc * 1.25])
    cache_root = "/tmp/claude-1000/case_parity_dump_cache"

    out = {"case": case, "n_rows": len(X)}

    # ── "after" / aware: fixed metric, band-valid rows only ──────────────
    blocks_full = IPRS.build_layout_blocks(layout, lambda r: sigma[r], T_local, svd_tol=1e-6, fprior_mode="full")
    n_comp = X.shape[1]
    valid_mask = IPRS.band_valid_mask(X, blocks_full, 1.0)
    Xv, yv = X[valid_mask], y[valid_mask]

    # search once (aware) to get chosen hyperparameters, reused for unaware too
    cache_dir = os.path.join(cache_root, f"case-{case}-search")
    os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)
    sel_params = np.ones(n_comp, dtype=int)
    itr, ival, iho = IPRS.three_way_split(len(yv), seed_outer=104, seed_inner=7, train_frac=0.70, val_frac=0.15)
    Xp = np.vstack([Xv[itr], Xv[ival]]); yp = np.concatenate([yv[itr], yv[ival]])
    search = IPRS.fit_prs_with_cv_search(
        PRS.ResponseSurface, Xp, yp, blocks_full, n_comp, case, case, "Partial", sel_params, 2.0,
        cache_dir=cache_dir, tol_grid=[float("inf")], k_grid=[8, 10, 12, 14, 16, 20], order_grid=[2, 3],
        ridge_grid=[0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5], cv_folds=10, cv_seed=13,
        band_tolerance_eval=None)
    k, order, ridge = search["k"], search["order"], search["ridge_alpha"]
    out["chosen_hyperparams"] = dict(k=k, order=order, ridge_alpha=ridge)

    out["after"] = fit_variant(Xv, yv, blocks_full, n_comp, case, k, order, ridge,
                                os.path.join(cache_root, f"case-{case}-after"))

    with open(D + "unsrt.pkl", "rb") as fh:
        unsrt = pickle.load(fh)
    active_param_map = {r: unsrt[r].activeParameters for r in rxns if r in unsrt}
    rcols = rescued_columns(D, case, rxns, sel, active_param_map)
    out["n_rescued_cols"] = int(len(rcols))
    if len(rcols) > 0:
        Xu = Xv.copy(); Xu[:, rcols] = 0.0
        out["unaware"] = fit_variant(Xu, yv, blocks_full, n_comp, case, k, order, ridge,
                                      os.path.join(cache_root, f"case-{case}-unaware"))

    # ── "before": reduced metric, full tolerance search, ALL-POINTS scoring ──
    blocks_red = IPRS.build_layout_blocks(layout, lambda r: sigma[r], T_local, svd_tol=1e-6, fprior_mode="reduced")
    cache_dir_b = os.path.join(cache_root, f"case-{case}-before-search")
    os.makedirs(os.path.join(cache_dir_b, "Data", "ResponseSurface"), exist_ok=True)
    itrB, ivalB, ihoB = IPRS.three_way_split(len(y), seed_outer=104, seed_inner=7, train_frac=0.70, val_frac=0.15)
    XpB = np.vstack([X[itrB], X[ivalB]]); ypB = np.concatenate([y[itrB], y[ivalB]])
    try:
        searchB = IPRS.fit_prs_with_cv_search(
            PRS.ResponseSurface, XpB, ypB, blocks_red, n_comp, case, case, "Partial", sel_params, 2.0,
            cache_dir=cache_dir_b, tol_grid=[1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")],
            k_grid=[8, 10, 12, 14, 16, 20], order_grid=[2, 3],
            ridge_grid=[0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5], cv_folds=10, cv_seed=13,
            band_tolerance_eval=1.0)
        keep_mask, _ = IPRS.screen_rows(X[itrB], blocks_red, searchB["tolerance"])
        Xk, yk = X[itrB][keep_mask], y[itrB][keep_mask]
        fmapB = IPRS.fit_feature_map(Xk, yk, blocks_red, n_comp, k=searchB["k"])
        cwd0 = os.getcwd(); os.chdir(cache_dir_b)
        coef_path = os.path.join(cache_dir_b, "Data", "ResponseSurface", f"responsecoef_case-{case}.csv")
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        respB = PRS.ResponseSurface(Xk, yk, case, case, responseOrder=searchB["order"], prs_type="Partial",
                                     selected_params=sel_params, criteria=2.0, feature_map=fmapB,
                                     ridge_alpha=searchB["ridge_alpha"])
        respB.create_response_surface()
        y_pred_train = np.asarray(respB.resFramWrk, dtype=float)
        _, _, _ = respB.test(X[ivalB], y[ivalB])
        y_pred_val = np.asarray(respB.y_Test_Predict, dtype=float)
        _, _, _ = respB.test(X[ihoB], y[ihoB])
        y_pred_hold = np.asarray(respB.y_Test_Predict, dtype=float)
        os.chdir(cwd0)
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        out["before"] = dict(y_train=yk.tolist(), yhat_train=y_pred_train.tolist(),
                              y_val=y[ivalB].tolist(), yhat_val=y_pred_val.tolist(),
                              y_hold=y[ihoB].tolist(), yhat_hold=y_pred_hold.tolist(),
                              chosen_hyperparams=dict(tolerance=searchB["tolerance"], k=searchB["k"],
                                                       order=searchB["order"], ridge_alpha=searchB["ridge_alpha"]))
    except Exception as exc:
        out["before_error"] = f"{type(exc).__name__}: {exc}"

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
