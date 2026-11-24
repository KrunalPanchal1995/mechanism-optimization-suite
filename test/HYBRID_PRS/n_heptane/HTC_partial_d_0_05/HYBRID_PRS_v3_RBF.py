"""
HYBRID_PRS_v3_RBF.py  —  Ridge Polynomial PRS + Kernel Ridge (RBF) Residual Correction
=========================================================================================

Why this approach is better than v1 (and different from v2)
-------------------------------------------------------------
v1 tanh correction:  shares one weight vector between polynomial and activation
                     terms → coupled, hard to initialise, BFGS can stall.
v2 GP correction  :  probabilistic, ARD, but O(n³) in training; slow for n>2000.

v3 uses:

    ŷ(x) = PRS(x)  +  KRR_rbf(x; residuals)

    Stage 1 — Ridge Polynomial PRS  (same as v2)
        • Cross-validated Ridge alpha for robust OLS.

    Stage 2 — Kernel Ridge Regression with RBF kernel on residuals
        • KRR solves a closed-form linear system: α = (K + λI)⁻¹ r
          where K_ij = exp(-γ ||x_i - x_j||²) (Gaussian / RBF kernel).
        • No iterative optimisation (no BFGS) → deterministic, fast.
        • gamma and lambda are selected by 5-fold cross-validation over
          a 2-D log-spaced grid.
        • RBF kernel is universal (can approximate any continuous function),
          so the residual correction can capture highly nonlinear patterns.
        • Unlike GP (v2), KRR does NOT produce predictive uncertainty
          estimates, but it is faster and scales better for larger n.

Why KRR-RBF for combustion PRS residuals?
        The residuals of a degree-2 polynomial from a 23-dimensional Sobol
        or Latin hypercube design are typically structured (non-random):
        higher-order interaction terms, Arrhenius curvature, NTC artefacts.
        A universal kernel captures these without any assumption about their
        functional form.

Training cost: O(n³) one-time (same as GP), but KRR has a simpler
kernel matrix (no marginal-likelihood optimisation) so it is 5-20x faster
in practice than a GP with hyperparameter restarts.


HYBRID_PRS_v3_fixed.py  —  Ridge Polynomial PRS + KRR-RBF (fully corrected)
=============================================================================

WHY THE PARITY PLOT WAS FLAT — THREE COMPOUNDING CAUSES
---------------------------------------------------------

CAUSE 1 (CODE BUG): StandardScaler destroys the bias column.
    PolynomialFeatures(include_bias=True) places a column of all-ones in
    X_poly[:,0].  StandardScaler(with_mean=True) then subtracts mean(1)=1,
    turning that column into all-zeros.  With fit_intercept=False in Ridge,
    the model has no intercept at all.  Since y ≈ 5.53 everywhere, a model
    with no intercept must force all that constant into the polynomial
    coefficients, completely distorting the Ridge solution.
    Fix: PolynomialFeatures(include_bias=False) + Ridge(fit_intercept=True).

CAUSE 2 (CODE BUG): KRR-RBF gamma range is not data-informed.
    The old grid [0.01/d, 10/d] = [0.00048, 0.476] was derived from
    dimensionality alone.  The actual median pairwise distance in the
    normalised feature space is ~6.45, giving a median-heuristic gamma of
    0.012.  This IS inside the old grid, but only barely, and the grid
    density around the optimal value was too sparse for reliable CV.
    Fix: gamma grid centred on median-heuristic value with ±2 decades.

CAUSE 3 (DATA): case-7 has no polynomial signal.
    Max |Pearson r| between any degree-2 polynomial term and y = 0.089.
    This is essentially zero.  When there is no signal, the best any model
    can do is predict mean(y) — which is exactly what the flat parity plot
    shows.  This is CORRECT model behaviour, not a bug.
    Consequence: fixing Causes 1 & 2 will not produce a good parity plot
    for case-7.  You need to test on a case where the 21 reactions matter
    (one where max|r| > ~0.2 across the degree-2 terms).

    What to try: run this script on sim_data_case-<N>.lst for other cases
    and check the printed "Signal check" section.  A case with
    Stage-1 R²(test) > 0.3 will show a clear diagonal parity plot.

NEW FEATURES IN THIS FILE
--------------------------
* Signal check printed before fitting: max |r| across all degree-2 terms,
  with a WARNING if the signal is too weak for polynomial PRS.
* Per-stage R² (train and test) so you can see each stage's contribution.
* R² displayed in the parity plot title (relative error alone is misleading
  when all y-values cluster near 5.53: even a flat model gets ~3% rel err).
* ElasticNet option for Stage 1 (flag --stage1 elasticnet) — handles sparse
  signals better than Ridge when only a few reactions dominate.

Usage
-----
    python HYBRID_PRS_v3_fixed.py \\
        --response     sim_data_case-7.lst \\
        --design_matrix pDesignMatrix.csv \\
        --degree 2 \\
        --test_size 0.2

    # Use ElasticNet for Stage 1 (better for sparse signals):
    python HYBRID_PRS_v3_fixed.py \\
        --response     sim_data_case-7.lst \\
        --design_matrix pDesignMatrix.csv \\
        --stage1 elasticnet
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------------------------
# Signal diagnostic
# ---------------------------------------------------------------------------

def signal_check(X, y):
    """
    Compute max |Pearson r| across all degree-1 and degree-2 polynomial
    terms.  Returns (max_r, warning_str).
    """
    max_r = 0.0
    max_term = ""
    for i in range(X.shape[1]):
        r = abs(np.corrcoef(X[:, i],    y)[0, 1]);  max_r, max_term = (r, f"x{i}")   if r > max_r else (max_r, max_term)
        r = abs(np.corrcoef(X[:, i]**2, y)[0, 1]);  max_r, max_term = (r, f"x{i}^2") if r > max_r else (max_r, max_term)
        for j in range(i + 1, X.shape[1]):
            r = abs(np.corrcoef(X[:, i] * X[:, j], y)[0, 1])
            if r > max_r:
                max_r, max_term = r, f"x{i}*x{j}"
    return max_r, max_term


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def hybrid_prs_fit(X, y, degree=2, cv_folds=5, stage1='ridge'):
    """
    Stage 1: Ridge (or ElasticNet) polynomial PRS.
    Stage 2: KRR-RBF on residuals with data-informed gamma range.

    Parameters
    ----------
    X        : ndarray (n, d)
    y        : ndarray (n,)   — log-space eta, already transformed
    degree   : int
    cv_folds : int
    stage1   : 'ridge' | 'elasticnet'

    Returns
    -------
    dict with all fitted objects and per-stage R² on training data.
    """
    # ------------------------------------------------------------------ #
    # Stage 1: Polynomial PRS                                             #
    # NOTE: include_bias=False + fit_intercept=True avoids the           #
    # StandardScaler-zeroing-bias-column bug.                             #
    # ------------------------------------------------------------------ #
    X_sc = StandardScaler()
    X_norm = X_sc.fit_transform(X)                        # normalise raw X

    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)                   # (n, p)

    poly_sc = StandardScaler()
    X_poly_s = poly_sc.fit_transform(X_poly)              # scale poly features

    if stage1 == 'elasticnet':
        # ElasticNet: L1+L2, does variable selection — better for sparse signals
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        stage1_model = ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=cv_folds,
            fit_intercept=True,
            max_iter=5000,
            n_jobs=-1
        )
    else:
        # Ridge: pure L2 shrinkage, more stable on dense signals
        stage1_model = RidgeCV(
            alphas=np.logspace(-6, 6, 60),
            cv=cv_folds,
            fit_intercept=True
        )

    stage1_model.fit(X_poly_s, y)
    y_prs = stage1_model.predict(X_poly_s)
    r2_stage1_train = r2_score(y, y_prs)

    if stage1 == 'elasticnet':
        print(f"  [Stage 1 ElasticNet] alpha={stage1_model.alpha_:.4g}  "
              f"l1_ratio={stage1_model.l1_ratio_:.2f}  "
              f"non-zero coefs={np.sum(stage1_model.coef_ != 0)}/{len(stage1_model.coef_)}")
    else:
        print(f"  [Stage 1 Ridge] CV-selected alpha = {stage1_model.alpha_:.4g}")
    print(f"  [Stage 1] R²(train) = {r2_stage1_train:.6f}  "
          f"RMSE(train) = {np.sqrt(np.mean((y - y_prs)**2)):.6f}")

    residuals = y - y_prs

    # ------------------------------------------------------------------ #
    # Stage 2: KRR-RBF on residuals                                       #
    # Gamma grid centred on the median-distance heuristic.                #
    # ------------------------------------------------------------------ #
    # Compute median pairwise distance for gamma initialisation
    n_sub = min(300, len(X_norm))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_norm), n_sub, replace=False)
    D = euclidean_distances(X_norm[idx])
    upper = D[np.triu_indices(n_sub, k=1)]
    median_dist = np.median(upper)
    gamma_med = 1.0 / (2.0 * median_dist ** 2)
    gamma_grid = np.logspace(
        np.log10(gamma_med / 100),
        np.log10(gamma_med * 100),
        25
    )
    alpha_grid = np.logspace(-5, 2, 20)
    print(f"  [Stage 2] median dist={median_dist:.4f}  "
          f"gamma_heuristic={gamma_med:.5f}  "
          f"gamma_grid=[{gamma_grid[0]:.2e}, {gamma_grid[-1]:.2e}]")

    gs = GridSearchCV(
        KernelRidge(kernel='rbf'),
        {'alpha': alpha_grid, 'gamma': gamma_grid},
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    gs.fit(X_norm, residuals)
    krr = gs.best_estimator_
    krr_corr = krr.predict(X_norm)
    r2_stage2_train = r2_score(residuals, krr_corr)
    print(f"  [Stage 2 KRR-RBF] alpha={gs.best_params_['alpha']:.3g}  "
          f"gamma={gs.best_params_['gamma']:.5f}")
    print(f"  [Stage 2] R²(residuals, train) = {r2_stage2_train:.6f}")

    return {
        'X_sc': X_sc,
        'poly': poly,
        'poly_sc': poly_sc,
        'stage1': stage1_model,
        'krr': krr,
        'r2_stage1_train': r2_stage1_train,
        'r2_stage2_train': r2_stage2_train,
    }


def hybrid_prs_predict(X, model_params):
    """
    ŷ = PRS(X) + KRR_rbf_correction(X)

    Parameters
    ----------
    X            : array-like (n, d)
    model_params : dict from hybrid_prs_fit

    Returns
    -------
    y_pred : ndarray (n,)
    """
    X = np.asarray(X)
    X_norm  = model_params['X_sc'].transform(X)
    X_poly  = model_params['poly'].transform(X_norm)
    X_poly_s = model_params['poly_sc'].transform(X_poly)
    y_prs   = model_params['stage1'].predict(X_poly_s)
    krr_corr = model_params['krr'].predict(X_norm)
    return y_prs + krr_corr


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label):
    rel_err = np.abs(y_true - y_pred) / np.abs(y_pred) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:10s} | R²={r2:+.4f}  RMSE={rmse:.6f}"
          f"  max_rel_err={rel_err.max():.3f}%  mean_rel_err={rel_err.mean():.3f}%")
    return rel_err.max(), r2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid PRS v3 (fixed): Ridge/ElasticNet Poly + KRR-RBF"
    )
    p.add_argument("--response",      required=True,
                   help="Tab-separated .lst file; column 1 used as target (log-space eta)")
    p.add_argument("--design_matrix", required=True,
                   help="Design matrix CSV (no header, comma-separated)")
    p.add_argument("--degree",     type=int,   default=2)
    p.add_argument("--test_size",  type=float, default=0.2)
    p.add_argument("--cv_folds",   type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--stage1",     choices=['ridge', 'elasticnet'], default='ridge',
                   help="Stage-1 model: 'ridge' (default) or 'elasticnet' "
                        "(better for sparse signals with few dominant reactions)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- load design matrix (robust: drops trailing-comma phantom cols) ---
    X_df = pd.read_csv(args.design_matrix, header=None)
    phantom  = X_df.columns[X_df.isna().all()].tolist()
    zero_var = X_df.columns[X_df.std() == 0].tolist()
    if phantom:
        print(f"[INFO] Dropping {len(phantom)} all-NaN columns "
              f"(trailing-comma artefact): {phantom}")
        X_df = X_df.dropna(axis=1, how='all')
    if zero_var:
        print(f"[INFO] Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        X_df = X_df.drop(columns=zero_var)
    X = X_df.values.astype(float)
    print(f"Design matrix loaded : {X.shape}")

    # --- load response ----------------------------------------------------
    y_df = pd.read_csv(args.response, sep='\t', header=None)
    y    = y_df.iloc[:, 1].values.astype(float)
    print(f"Response loaded      : {y.shape}  "
          f"range [{y.min():.4f}, {y.max():.4f}]  std={y.std():.6f}")

    if len(X) != len(y):
        raise ValueError(f"Row mismatch: X={len(X)}, y={len(y)}")

    # --- signal check (before wasting time fitting) -----------------------
    print("\n=== Signal Check ===")
    max_r, best_term = signal_check(X, y)
    print(f"  Max |Pearson r| across all degree-2 polynomial terms: {max_r:.5f}")
    print(f"  Most correlated term: {best_term}")
    if max_r < 0.10:
        print(
            "\n  *** WARNING: Signal is essentially absent (max|r| < 0.10). ***\n"
            "  The polynomial PRS cannot explain this response better than\n"
            "  predicting mean(y).  Possible causes:\n"
            "    1. These 21 reactions are not rate-limiting for this case's\n"
            "       T/P/phi conditions — try a different case number.\n"
            "    2. The design matrix and response file are mismatched.\n"
            "  The model will run, but the parity plot will be flat regardless\n"
            "  of which algorithm is used.\n"
        )
    elif max_r < 0.20:
        print("  [CAUTION] Weak signal (max|r| < 0.20). "
              "Predictions may not improve much over the mean.")
    else:
        print(f"  [OK] Signal detected — model should be learnable.")

    # --- split ------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

    # --- fit --------------------------------------------------------------
    print(f"\nFitting Hybrid PRS v3 (Poly deg={args.degree} + KRR-RBF) "
          f"with stage1={args.stage1} ...")
    model = hybrid_prs_fit(X_tr, y_tr,
                           degree=args.degree,
                           cv_folds=args.cv_folds,
                           stage1=args.stage1)

    # --- predict ----------------------------------------------------------
    y_pred_tr = hybrid_prs_predict(X_tr, model)
    y_pred_te = hybrid_prs_predict(X_te, model)

    # --- metrics ----------------------------------------------------------
    print("\n=== Residual metrics ===")
    max_err_tr, r2_tr = _metrics(y_tr, y_pred_tr, "Train")
    max_err_te, r2_te = _metrics(y_te, y_pred_te, "Test")
    if r2_te < 0.1:
        print("\n  NOTE: R²(test) < 0.1 — the model is not learning above the mean.")
        print("        This is expected when max|r| is below 0.10 (see Signal Check).")

    # --- parity plot + residual histogram (side by side) ------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Parity
    ax = axes[0]
    ax.scatter(y_te, y_pred_te, c='black',        alpha=0.7, s=16, label='Test')
    ax.scatter(y_tr, y_pred_tr, facecolors='none',
               edgecolors='darkorange', alpha=0.4, s=16, label='Train')
    lims = [min(y.min(), y_pred_tr.min(), y_pred_te.min()),
            max(y.max(), y_pred_tr.max(), y_pred_te.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect fit')
    ax.set_xlabel('Actual  η')
    ax.set_ylabel('Predicted  η')
    ax.set_title(
        f"Hybrid PRS v3 — Ridge Poly (deg {args.degree}) + KRR-RBF\n"
        f"R²  Train: {r2_tr:+.4f}   Test: {r2_te:+.4f}\n"
        f"MRE  Train: {max_err_tr:+.4f}   Test: {max_err_te:+.4f}\n"
    )
    ax.legend(fontsize=9)

    # Residuals
    ax2 = axes[1]
    res_tr = y_tr - y_pred_tr
    res_te = y_te - y_pred_te
    bins = np.linspace(min(res_tr.min(), res_te.min()),
                       max(res_tr.max(), res_te.max()), 30)
    ax2.hist(res_tr, bins=bins, alpha=0.6, color='darkorange', label='Train')
    ax2.hist(res_te, bins=bins, alpha=0.6, color='black',      label='Test')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_xlabel('Residual  (actual − predicted)')
    ax2.set_ylabel('Count')
    ax2.set_title(f"Residual distribution\n"
                  f"Train std={res_tr.std():.5f}   Test std={res_te.std():.5f}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("parity_v3_fixed.png", dpi=150)
    plt.show()
    print("Figure saved → parity_v3_fixed.png")


if __name__ == "__main__":
    main()
