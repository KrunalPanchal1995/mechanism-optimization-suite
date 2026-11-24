"""
HYBRID_PRS_v1_fixed.py  —  Fixed Original: Polynomial PRS + tanh nonlinear term
=================================================================================

Corrected bugs vs the original HYBRID_PRS.py
---------------------------------------------
BUG 1 (CRITICAL): Wrong CSV separator.
    The .lst file is tab-separated; pd.read_csv default sep=',' reads each
    entire line as a single string → np.log() crashes with TypeError.
    Fix: sep='\\t'.

BUG 2 (CRITICAL): Double log-transformation.
    sim_data_case-X.lst already stores eta = ln(tau_µs * 10) (log-space).
    The original `y = np.log(df * 10000)` applied log again → nonsense targets.
    Fix: read column 1 directly, no further transformation.

BUG 3: Intercept redundancy / biased BFGS initialisation.
    PolynomialFeatures(include_bias=True) adds a constant column to X_poly.
    LinearRegression(fit_intercept=True) then fits *another* scalar intercept
    on top, creating a redundant parameter that is silently dropped when only
    model.coef_ is passed as initial_params to BFGS.
    Fix: LinearRegression(fit_intercept=False) so the bias is fully absorbed
    into coef_[0] (coefficient for the constant column).

BUG 4: No CLI argument parsing.
    File paths were hard-coded.
    Fix: argparse for --response, --design_matrix, --degree, --test_size.

Usage
-----
    python HYBRID_PRS_v1_fixed.py \\
        --response     sim_data_case-7.lst \\
        --design_matrix DesignMatrix.csv \\
        --degree 2 \\
        --test_size 0.2
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def hybrid_prs_fit(X, y, degree=2, max_iter=1000):
    """
    Fit Polynomial PRS with a tanh-augmented prediction:
        y_pred = X_poly @ w + tanh(X_poly @ w)

    The tanh term adds a bounded nonlinear correction on top of the linear
    projection.  Both the polynomial and the nonlinear correction share the
    same weight vector w, which is optimised jointly by BFGS.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        Target values already in log-space (eta = ln(tau_µs * 10)).
    degree : int
        Polynomial degree.
    max_iter : int
        Maximum BFGS iterations.

    Returns
    -------
    dict with keys 'coefficients' (ndarray) and 'poly' (PolynomialFeatures).
    """
    # --- polynomial expansion -------------------------------------------
    poly = PolynomialFeatures(degree, include_bias=True)
    X_poly = poly.fit_transform(X)          # shape (n, p)  p includes bias

    # --- initial estimate via OLS (fit_intercept=False because bias is in
    #     X_poly[:,0] already; avoids a redundant intercept parameter) -----
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_poly, y)
    initial_params = ols.coef_.copy()       # shape (p,)

    # --- define MSE objective for the hybrid model -----------------------
    def _mse(w, Xp, ytrue):
        f = Xp @ w
        y_pred = f + np.tanh(f)
        return np.mean((ytrue - y_pred) ** 2)

    # --- optimise with BFGS ----------------------------------------------
    result = minimize(
        _mse,
        initial_params,
        args=(X_poly, y),
        method='BFGS',
        options={'maxiter': max_iter, 'disp': False}
    )

    if not result.success:
        print(f"[WARNING] BFGS did not converge: {result.message}", file=sys.stderr)

    return {'coefficients': result.x, 'poly': poly}


def hybrid_prs_predict(X, model_params):
    """
    Predict with the trained hybrid PRS model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    model_params : dict  (from hybrid_prs_fit)

    Returns
    -------
    y_pred : ndarray, shape (n_samples,)
    """
    X = np.asarray(X)
    X_poly = model_params['poly'].transform(X)
    w = model_params['coefficients']
    f = X_poly @ w
    return f + np.tanh(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label):
    rel_err = np.abs(y_true - y_pred) / np.abs(y_pred) * 100
    print(f"  {label:10s} | max rel err: {rel_err.max():.4f}%"
          f"  |  mean rel err: {rel_err.mean():.4f}%"
          f"  |  RMSE: {np.sqrt(np.mean((y_true-y_pred)**2)):.6f}")
    return rel_err.max()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid PRS v1 (fixed original): Polynomial + tanh correction"
    )
    p.add_argument("--response", required=True,
                   help="Path to .lst response file (tab-separated; column 1 used)")
    p.add_argument("--design_matrix", required=True,
                   help="Path to design matrix CSV (no header, comma-separated)")
    p.add_argument("--degree", type=int, default=2,
                   help="Polynomial degree (default: 2)")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Test split fraction (default: 0.2)")
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Max BFGS iterations (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/test split (default: 42)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- load design matrix ---------------------------------------------
    # Trailing commas in the CSV produce a phantom all-NaN last column;
    # dropna(axis=1, how='all') removes it before any further processing.
    X_df = pd.read_csv(args.design_matrix, header=None)
    phantom = X_df.columns[X_df.isna().all()].tolist()
    if phantom:
        print(f"[INFO] Dropping {len(phantom)} all-NaN column(s) "
              f"(trailing-comma artefact): {phantom}")
        X_df = X_df.dropna(axis=1, how='all')
    zero_var = X_df.columns[X_df.std() == 0].tolist()
    if zero_var:
        print(f"[INFO] Dropping {len(zero_var)} zero-variance column(s): {zero_var}")
        X_df = X_df.drop(columns=zero_var)
    X = X_df.values.astype(float)
    print(f"Design matrix loaded : {X.shape}")

    # --- load response (column 1 = eta, already in log-space) ----------
    # BUG 1 fixed: sep='\t'
    # BUG 2 fixed: no further log transformation
    y_df = pd.read_csv(args.response, sep='\t', header=None)
    y = y_df.iloc[:, 1].values.astype(float)
    print(f"Response loaded      : {y.shape}  range [{y.min():.4f}, {y.max():.4f}]")

    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch: design matrix has {len(X)} rows, "
            f"response has {len(y)} rows."
        )

    # --- train/test split -----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # --- fit ------------------------------------------------------------
    print("\nFitting Hybrid PRS v1 (Polynomial + tanh) ...")
    model = hybrid_prs_fit(X_train, y_train, degree=args.degree,
                           max_iter=args.max_iter)

    # --- predict --------------------------------------------------------
    y_pred_train = hybrid_prs_predict(X_train, model)
    y_pred_test  = hybrid_prs_predict(X_test,  model)

    # --- metrics --------------------------------------------------------
    print("\nResidual metrics:")
    max_train = _metrics(y_train, y_pred_train, "Train")
    max_test  = _metrics(y_test,  y_pred_test,  "Test")

    # --- parity plot ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test,  y_pred_test,  c='black', alpha=0.7, s=18, label='Test')
    ax.scatter(y_train, y_pred_train, facecolors='none', edgecolors='green',
               alpha=0.5, s=18, label='Train')
    lims = [min(y.min(), y_pred_train.min(), y_pred_test.min()),
            max(y.max(), y_pred_train.max(), y_pred_test.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect fit')
    ax.set_xlabel('Actual  η')
    ax.set_ylabel('Predicted  η')
    ax.set_title(
        f"Hybrid PRS v1 — Polynomial (deg {args.degree}) + tanh\n"
        f"Max rel err  Train: {max_train:.2f}%   Test: {max_test:.2f}%"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("parity_v1_fixed.png", dpi=150)
    plt.show()
    print("Parity plot saved → parity_v1_fixed.png")


if __name__ == "__main__":
    main()
