"""
HYBRID_PRS_v2_GP.py  —  Ridge Polynomial PRS + Gaussian Process Residual Correction
======================================================================================

Why this approach is better than v1
-------------------------------------
v1 shares ONE weight vector between the polynomial part and the tanh correction.
This couples the two terms, so the polynomial part cannot independently capture
the global trend while the nonlinear part handles local deviations.

v2 separates them into two additive stages (the standard "trend + deviation"
decomposition from kriging / design-of-experiments literature):

    ŷ(x) = PRS(x)  +  GP(x; residuals)

    Stage 1 — Ridge Polynomial PRS
        • Degree-2 polynomial expansion (300 features for 23 inputs).
        • Ridge (L2) regularisation avoids overfitting the high-dimensional
          polynomial space; alpha is selected by 5-fold cross-validation.
        • Replaces the ill-conditioned OLS in the original.

    Stage 2 — Gaussian Process correction on residuals
        • A GP with an anisotropic ARD (Automatic Relevance Determination)
          Matérn-5/2 kernel + white noise is fitted to the OLS residuals.
        • ARD: each input dimension gets its own length-scale, so the GP
          automatically down-weights irrelevant/insensitive parameters.
          This is physically meaningful in combustion kinetics: only a
          subset of reactions has significant sensitivity.
        • The GP provides a principled, probabilistic correction that is
          consistent with the data noise level.
        • Training cost: O(n³) for n training points; with n~800 and no
          hyperparameter scan it runs in a few seconds.

    Combined:
        • PRS captures the smooth polynomial global trend.
        • GP captures structured residuals the polynomial misses.
        • Together this is equivalent to a Kriging model with a polynomial
          mean function — well-established in aerospace/combustion DOE.

HYBRID_PRS_v2_GP_fixed.py  —  Ridge Polynomial PRS + GP Residual Correction (fixed)
======================================================================================

WHY v2 GAVE TRAIN=0.00%, TEST=3.05% (perfect train, flat test)
---------------------------------------------------------------
The parity plot showed two very different behaviours:
  - Training (blue circles): perfect diagonal  ← GP MEMORISED training data
  - Testing  (black dots):   flat horizontal   ← GP reverted to mean(residuals)

Three compounding causes:

CAUSE 1 (CODE BUG): StandardScaler zeroes the bias column.
    Same as v3. PolynomialFeatures(include_bias=True) creates an all-ones
    column in X_poly[:,0]. StandardScaler subtracts mean(1)=1, making it
    all zeros. With fit_intercept=False in Ridge, the model has no intercept
    at all. Stage 1 therefore predicted values near 0 instead of near 5.53,
    and passed FULL y values (not small residuals) to Stage 2 as "residuals".
    Fix: include_bias=False + fit_intercept=True.

CAUSE 2 (CODE BUG): GP was given the raw y values as residuals, not
    actual residuals from a working Stage 1. Because Stage 1 predicted ~0
    everywhere (Cause 1), the "residuals" fed to Stage 2 were simply y itself
    (~5.53 for all points). The GP then had to explain a signal that had
    nothing to do with the input features — pure noise from its perspective.

CAUSE 3 (GP OVER-PARAMETERISATION): ARD with 21 free length-scales +
    near-zero WhiteKernel lower bound = degenerate optimiser solution.
    The marginal likelihood optimiser found a local minimum where several
    length-scales collapsed to 0.08–0.17 (extremely sharp kernels) while
    others hit the 10.0 upper bound. A sharp kernel in even ONE dimension
    allows the GP to memorise the ordering of training points along that
    axis. Combined with normalize_y=True absorbing the ~5.53 offset, the GP
    could perfectly reconstruct all 809 training targets while reverting to
    mean(y) for every test point (which was far from any training point
    in the sharp-kernel dimensions).
    Fix: length_scale lower bound ≥ 0.5 (in normalised space) prevents
    kernel widths below the typical inter-point spacing.

WHAT HAPPENS IN HIGH DIMENSIONS (why v2 is harder to use than v3)
    With d=21 normalised features, median pairwise distance ≈ 6.45.
    An isotropic RBF-type kernel with length-scale=1.0 has
    k(xi, xj) = exp(−6.45²/2) ≈ 10⁻⁹ — essentially zero for all i≠j.
    This means every test point is effectively "out-of-distribution" for the
    GP regardless of how many training points exist. The GP can only avoid
    flat test predictions by using length-scales large enough to create
    non-negligible kernel values across the typical inter-point distance.
    The minimum useful length-scale is ≈ median_dist / 2 ≈ 3.2.
    The fix constrains length_scale_bounds=(0.5, 10.0) and sets a minimum
    noise floor so the optimiser is guided away from the degenerate solution.

    Practical guidance: when Stage-1 R²(train) < 0.1, skip Stage 2 entirely
    (a GP on near-random residuals will always overfit in high dimensions).

Usage
-----
    python HYBRID_PRS_v2_GP_fixed.py \\
        --response     sim_data_case-7.lst \\
        --design_matrix pDesignMatrix.csv

    # Skip GP Stage 2 (pure Ridge PRS, fastest):
    python HYBRID_PRS_v2_GP_fixed.py --response ... --design_matrix ... --skip_gp
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------------------------
# Signal diagnostic  (shared with v3_fixed)
# ---------------------------------------------------------------------------

def signal_check(X, y):
    max_r, max_term = 0.0, ""
    for i in range(X.shape[1]):
        for term, vals in [(f"x{i}", X[:,i]), (f"x{i}^2", X[:,i]**2)]:
            r = abs(np.corrcoef(vals, y)[0,1])
            if r > max_r:
                max_r, max_term = r, term
        for j in range(i+1, X.shape[1]):
            r = abs(np.corrcoef(X[:,i]*X[:,j], y)[0,1])
            if r > max_r:
                max_r, max_term = r, f"x{i}*x{j}"
    return max_r, max_term


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def hybrid_prs_fit(X, y, degree=2, cv_folds=5, skip_gp=False):
    """
    Stage 1: Ridge polynomial PRS  (bias-column bug fixed).
    Stage 2: GP residual correction (length-scale floor fixed).

    Parameters
    ----------
    X         : ndarray (n, d)
    y         : ndarray (n,)  — log-space eta (already transformed)
    degree    : int
    cv_folds  : int
    skip_gp   : bool  — if True, return Stage 1 only (useful when signal is absent)

    Returns
    -------
    dict with all fitted objects
    """
    # ------------------------------------------------------------------ #
    # Stage 1: Ridge Polynomial PRS  (FIXED)                             #
    # ------------------------------------------------------------------ #
    X_sc = StandardScaler()
    X_norm = X_sc.fit_transform(X)

    # FIX: include_bias=False so StandardScaler never touches a bias col.
    # FIX: fit_intercept=True so Ridge fits its own (unpenalised) intercept.
    poly   = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)

    poly_sc   = StandardScaler()
    X_poly_s  = poly_sc.fit_transform(X_poly)

    ridge = RidgeCV(alphas=np.logspace(-6, 6, 60), cv=cv_folds,
                    fit_intercept=True)
    ridge.fit(X_poly_s, y)
    y_prs = ridge.predict(X_poly_s)

    r2_s1_tr  = r2_score(y, y_prs)
    rmse_s1   = np.sqrt(np.mean((y - y_prs)**2))
    print(f"  [Stage 1 Ridge] alpha = {ridge.alpha_:.4g}")
    print(f"  [Stage 1] R²(train) = {r2_s1_tr:.6f}   RMSE = {rmse_s1:.6f}")

    residuals  = y - y_prs
    res_var    = residuals.var()
    res_std    = residuals.std()

    result = {
        'X_sc': X_sc, 'poly': poly, 'poly_sc': poly_sc, 'ridge': ridge,
        'r2_stage1_train': r2_s1_tr,
        'gp': None,           # filled below unless skip_gp
    }

    if skip_gp:
        print("  [Stage 2] Skipped (--skip_gp flag set).")
        return result

    # ------------------------------------------------------------------ #
    # Stage 2: GP on residuals  (overfitting fixes applied)               #
    # ------------------------------------------------------------------ #
    n, d = X_norm.shape

    # Safety check: if Stage 1 has near-zero R², the "residuals" are just y
    # minus the mean.  A GP on this in d=21 will overfit train and revert
    # to mean(residuals) on test.  Warn and offer to skip.
    if r2_s1_tr < 0.05:
        print(
            "\n  [Stage 2 WARNING] Stage-1 R²(train) < 0.05 — residuals "
            "carry no polynomial signal.\n"
            "  A GP fitted on near-random residuals in high dimensions WILL "
            "memorise\n"
            "  training data (train≈perfect) while generalising to the mean "
            "on test\n"
            "  (test≈flat). This is NOT a model bug — it is the curse of "
            "dimensionality:\n"
            "  median inter-point distance in d=21 normalised space is "
            f"~{_median_dist(X_norm):.2f}, so\n"
            "  kernel values between any train and test point are "
            "near zero.\n"
            "  Proceeding anyway; use --skip_gp to return Stage-1 only.\n"
        )

    # Median-heuristic gamma / length-scale guidance
    med_dist = _median_dist(X_norm)
    # Minimum useful length-scale ≈ med_dist/2 so kernel values are non-trivial
    ls_min  = max(med_dist / 4.0, 0.5)
    ls_init = max(med_dist / 2.0, 1.0)
    ls_max  = max(med_dist * 2.0, 10.0)
    print(f"\n  [Stage 2] median inter-point dist = {med_dist:.3f}")
    print(f"  [Stage 2] length_scale bounds     = ({ls_min:.2f}, {ls_max:.2f})")

    # FIX: length_scale lower bound set from data geometry (not 0.01).
    # FIX: WhiteKernel lower bound ≥ 1e-3 (not 1e-6) to prevent near-zero noise.
    # FIX: normalize_y=False — Stage 1 now has a proper intercept, so residuals
    #      already have near-zero mean; we don't need the GP to correct for it.
    # FIX: alpha set to 0.5 * res_std (explicit noise floor).
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=ls_init * np.ones(d),
            length_scale_bounds=(ls_min, ls_max),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=res_var,
            noise_level_bounds=(max(1e-3, 0.1 * res_var), 10.0 * res_var),
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,    # reduced from 5 (fewer restarts → fewer degenerate sols)
        normalize_y=False,         # Stage 1 already handles the mean
        alpha=0.5 * max(res_var, 1e-6),   # explicit noise floor, prevents exact interpolation
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_norm, residuals)

    gp_corr_tr  = gp.predict(X_norm)
    r2_s2_tr    = r2_score(residuals, gp_corr_tr)
    print(f"  [Stage 2 GP] fitted kernel: {gp.kernel_}")
    print(f"  [Stage 2] R²(residuals, train) = {r2_s2_tr:.6f}")

    result['gp']              = gp
    result['r2_stage2_train'] = r2_s2_tr
    return result


def _median_dist(X_norm, n_sub=250, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_norm), min(n_sub, len(X_norm)), replace=False)
    D   = euclidean_distances(X_norm[idx])
    return float(np.median(D[np.triu_indices(len(idx), k=1)]))


def hybrid_prs_predict(X, model_params):
    """ŷ = PRS(X) + GP_correction(X)  (GP correction = 0 if skip_gp)."""
    X      = np.asarray(X)
    X_norm = model_params['X_sc'].transform(X)
    X_poly = model_params['poly'].transform(X_norm)
    Xp_s   = model_params['poly_sc'].transform(X_poly)
    y_prs  = model_params['ridge'].predict(Xp_s)

    if model_params['gp'] is None:
        return y_prs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_corr = model_params['gp'].predict(X_norm)
    return y_prs + gp_corr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label):
    rel_err = np.abs(y_true - y_pred) / np.abs(y_pred) * 100
    rmse    = np.sqrt(np.mean((y_true - y_pred)**2))
    r2      = r2_score(y_true, y_pred)
    print(f"  {label:10s} | R²={r2:+.4f}  RMSE={rmse:.6f}"
          f"  max_rel_err={rel_err.max():.3f}%  mean_rel_err={rel_err.mean():.3f}%")
    return rel_err.max(), r2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid PRS v2 (fixed): Ridge Poly + GP residual correction"
    )
    p.add_argument("--response",      required=True,
                   help="Tab-separated .lst; column 1 = log-space eta target")
    p.add_argument("--design_matrix", required=True,
                   help="Design matrix CSV (no header, comma-separated)")
    p.add_argument("--degree",     type=int,   default=2)
    p.add_argument("--test_size",  type=float, default=0.2)
    p.add_argument("--cv_folds",   type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip_gp",    action='store_true',
                   help="Return Stage-1 Ridge PRS only (fast; use when signal is absent)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- load data (trailing-comma-robust) --------------------------------
    X_df = pd.read_csv(args.design_matrix, header=None)
    phantom  = X_df.columns[X_df.isna().all()].tolist()
    zero_var = X_df.columns[X_df.std() == 0].tolist()
    if phantom:
        print(f"[INFO] Dropping {len(phantom)} all-NaN cols (trailing comma): {phantom}")
        X_df = X_df.dropna(axis=1, how='all')
    if zero_var:
        print(f"[INFO] Dropping {len(zero_var)} zero-variance cols: {zero_var}")
        X_df = X_df.drop(columns=zero_var)
    X = X_df.values.astype(float)
    print(f"Design matrix loaded : {X.shape}")

    y_df = pd.read_csv(args.response, sep='\t', header=None)
    y    = y_df.iloc[:, 1].values.astype(float)
    print(f"Response loaded      : {y.shape}  "
          f"range [{y.min():.4f}, {y.max():.4f}]  std={y.std():.6f}")

    if len(X) != len(y):
        raise ValueError(f"Row mismatch: X={len(X)}, y={len(y)}")

    # --- signal check -----------------------------------------------------
    print("\n=== Signal Check ===")
    max_r, best_term = signal_check(X, y)
    print(f"  Max |Pearson r| across all degree-2 terms: {max_r:.5f}  "
          f"(term: {best_term})")
    if max_r < 0.10:
        print(
            "\n  *** WARNING: No polynomial signal detected (max|r| < 0.10). ***\n"
            "  Stage-1 R² will be near zero.  Stage-2 GP will overfit training\n"
            "  data and generalise to mean(y) on test — train≈perfect, test≈flat.\n"
            "  This is the curse of dimensionality in d=21, NOT a code bug.\n"
            "  Try a different case number with more sensitive reactions.\n"
            "  You can add --skip_gp to skip Stage 2 and avoid the overfitting.\n"
        )
    elif max_r < 0.20:
        print("  [CAUTION] Weak signal. GP Stage 2 may still overfit.")
    else:
        print("  [OK] Signal detected.")

    # --- split ------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

    # --- fit --------------------------------------------------------------
    print(f"\nFitting Hybrid PRS v2 (fixed) ...")
    model = hybrid_prs_fit(X_tr, y_tr, degree=args.degree,
                           cv_folds=args.cv_folds, skip_gp=args.skip_gp)

    # --- predict ----------------------------------------------------------
    y_pred_tr = hybrid_prs_predict(X_tr, model)
    y_pred_te = hybrid_prs_predict(X_te, model)

    # --- metrics ----------------------------------------------------------
    print("\n=== Residual metrics ===")
    max_err_tr, r2_tr = _metrics(y_tr, y_pred_tr, "Train")
    max_err_te, r2_te = _metrics(y_te, y_pred_te, "Test")
    train_test_gap = r2_tr - r2_te
    print(f"\n  R² gap (train-test) = {train_test_gap:.4f}")
    if train_test_gap > 0.5:
        print("  *** Large gap detected — GP is overfitting. ***")
        print("  Cause: no real signal in residuals → GP memorises training data.")
        print("  Re-run with --skip_gp to return Stage-1 only.")

    # --- parity + residual histogram --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.scatter(y_te, y_pred_te, c='black',        alpha=0.7, s=16, label='Test')
    ax.scatter(y_tr, y_pred_tr, facecolors='none',
               edgecolors='royalblue', alpha=0.4, s=16, label='Train')
    lims = [min(y.min(), y_pred_tr.min(), y_pred_te.min()),
            max(y.max(), y_pred_tr.max(), y_pred_te.max())]
    ax.plot(lims, lims, 'r--', lw=1.2, label='Perfect fit')
    ax.set_xlabel('Actual  η');  ax.set_ylabel('Predicted  η')
    ax.set_title(
        f"Hybrid PRS v2 (fixed) — Ridge Poly (deg {args.degree}) + GP\n"
        f"R²  Train: {r2_tr:+.4f}   Test: {r2_te:+.4f}"
    )
    ax.legend(fontsize=9)

    ax2 = axes[1]
    res_tr = y_tr - y_pred_tr;  res_te = y_te - y_pred_te
    bins = np.linspace(min(res_tr.min(), res_te.min()),
                       max(res_tr.max(), res_te.max()), 30)
    ax2.hist(res_tr, bins=bins, alpha=0.6, color='royalblue', label='Train')
    ax2.hist(res_te, bins=bins, alpha=0.6, color='black',     label='Test')
    ax2.axvline(0, color='r', ls='--')
    ax2.set_xlabel('Residual  (actual − predicted)')
    ax2.set_ylabel('Count')
    ax2.set_title(f"Residual distribution\n"
                  f"Train std={res_tr.std():.5f}   Test std={res_te.std():.5f}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("parity_v2_fixed.png", dpi=150)
    plt.show()
    print("Figure saved → parity_v2_fixed.png")


if __name__ == "__main__":
    main()
