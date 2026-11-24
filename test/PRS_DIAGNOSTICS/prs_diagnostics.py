#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prs_diagnostics.py
==================
Diagnostic suite for Polynomial Response Surfaces (PRS) trained by QR least
squares, built to answer ONE question precisely:

    "My full-PRS fits to ~0.3% MRE but my partial-PRS (p-PRS) is stuck at ~20%
     MRE. Is it the design matrix, the sampling strategy, the sample size, or
     the model form -- and will more samples fix it?"

Design philosophy
-----------------
The single most informative number you already have is the TRAINING error.
  * full-PRS : train max-err 0.718 %   (model class is adequate, well sampled)
  * p-PRS    : train max-err 9.898 %   (model class CANNOT fit even the data
                                        it was trained on)
High training error == bias (model-form) limited, NOT variance (data-quantity)
limited.  More samples reduce variance, never bias.  This suite quantifies the
bias/variance split so you stop guessing.

All fitting uses a plain QR-decomposition least-squares solver (optionally with
Tikhonov / ridge via row augmentation, still solved by QR).  No black boxes.

Author: prepared for KP (combustion kinetics / MUQ-SAC p-SAC framework)
"""

from __future__ import annotations
import numpy as np
from itertools import combinations_with_replacement
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- #
#  Plotting is optional / headless-safe                                         #
# ----------------------------------------------------------------------------- #
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:                                   # pragma: no cover
    _HAVE_MPL = False


# ============================================================================= #
#  1.  Polynomial feature expansion (transparent, 1-indexed, x14*x15 naming)     #
# ============================================================================= #
def poly_features(X: np.ndarray, degree: int = 2, include_bias: bool = True):
    """
    Build a polynomial design matrix with explicit, human-readable term names.

    Variables are named x1..xp (1-indexed) so a term shows up as 'x14*x15',
    matching the names your signal-check prints.

    Returns
    -------
    Phi   : (n_samples, n_terms) design matrix
    names : list[str] of length n_terms
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    cols, names = [], []
    if include_bias:
        cols.append(np.ones(n))
        names.append("1")
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(p), d):
            term = np.prod(X[:, combo], axis=1)
            cols.append(term)
            # build name e.g. (3,) -> x3 ; (13,14) -> x14*x15 ; (2,2) -> x3^2
            counts = {}
            for c in combo:
                counts[c] = counts.get(c, 0) + 1
            parts = []
            for c, k in sorted(counts.items()):
                parts.append(f"x{c+1}" if k == 1 else f"x{c+1}^{k}")
            names.append("*".join(parts))
    return np.column_stack(cols), names


# ============================================================================= #
#  2.  QR least-squares solver (optionally ridge-regularised)                    #
# ============================================================================= #
@dataclass
class QRFit:
    beta: np.ndarray                 # coefficients
    names: list                      # term names
    degree: int
    include_bias: bool
    mu: np.ndarray = field(default=None)   # feature centering (non-bias cols)
    sd: np.ndarray = field(default=None)   # feature scaling
    lam: float = 0.0
    cond_R: float = np.nan           # condition number of R (== cond of Phi)


def fit_prs_qr(X, y, degree=2, include_bias=True, lam=0.0, standardize=True):
    """
    Fit a PRS by QR decomposition.

        minimise  || Phi beta - y ||^2  + lam * || beta_nonbias ||^2

    Ridge (lam>0) is applied by AUGMENTING Phi with sqrt(lam)*I rows and still
    solving by QR -- numerically the most stable way to do regularised LS.

    Standardisation puts every (non-bias) column on unit scale so that:
      (a) ridge penalises all terms comparably, and
      (b) the condition number reported is meaningful rather than an artefact
          of raw column magnitudes (x_i^2 terms are otherwise huge).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    Phi, names = poly_features(X, degree, include_bias)

    mu = np.zeros(Phi.shape[1]); sd = np.ones(Phi.shape[1])
    if standardize:
        start = 1 if include_bias else 0
        mu[start:] = Phi[:, start:].mean(0)
        sd[start:] = Phi[:, start:].std(0)
        sd[sd == 0] = 1.0
        Phi = (Phi - mu) / sd
        if include_bias:
            Phi[:, 0] = 1.0                       # keep intercept column = 1

    A, b = Phi, y
    if lam > 0:
        p = Phi.shape[1]
        reg = np.sqrt(lam) * np.eye(p)
        if include_bias:
            reg[0, 0] = 0.0                        # do not penalise intercept
        A = np.vstack([Phi, reg])
        b = np.concatenate([y, np.zeros(p)])

    Q, R = np.linalg.qr(A)                         # thin QR
    beta = np.linalg.solve(R, Q.T @ b)
    cond_R = np.linalg.cond(R)

    return QRFit(beta=beta, names=names, degree=degree, include_bias=include_bias,
                 mu=mu, sd=sd, lam=lam, cond_R=cond_R)


def predict(fit: QRFit, X):
    Phi, _ = poly_features(np.asarray(X, float), fit.degree, fit.include_bias)
    if fit.mu is not None:
        Phi = (Phi - fit.mu) / fit.sd
        if fit.include_bias:
            Phi[:, 0] = 1.0
    return Phi @ fit.beta


# ============================================================================= #
#  3.  Metrics                                                                   #
# ============================================================================= #
def mre(y_true, y_pred):
    """Maximum Relative Error in percent (your 'max error' on the parity plots)."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    return 100.0 * np.max(np.abs(y_true - y_pred) / denom)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def report_metrics(tag, y_tr, p_tr, y_te, p_te):
    print(f"  [{tag}] "
          f"MRE  train={mre(y_tr,p_tr):7.3f}%  test={mre(y_te,p_te):7.3f}%   |   "
          f"R2 train={r2(y_tr,p_tr):.5f}  test={r2(y_te,p_te):.5f}   |   "
          f"RMSE train={rmse(y_tr,p_tr):.4g}  test={rmse(y_te,p_te):.4g}")


# ============================================================================= #
#  STUDY 1.  Bias / variance headline                                            #
# ============================================================================= #
def study_bias_variance(Xtr, ytr, Xte, yte, degree=2, lam=0.0):
    """
    The decision rule:
       train MRE small  & test MRE small  -> good
       train MRE small  & test MRE large  -> VARIANCE (overfit) -> more data /
                                             regularisation help
       train MRE LARGE  (regardless)      -> BIAS (model-form) -> more data will
                                             NOT help; change the basis/sampling
    """
    print("\n=== STUDY 1: bias vs variance headline ===")
    fit = fit_prs_qr(Xtr, ytr, degree=degree, lam=lam)
    ptr, pte = predict(fit, Xtr), predict(fit, Xte)
    report_metrics(f"deg={degree}, lam={lam}", ytr, ptr, yte, pte)
    print(f"  cond(R) = {fit.cond_R:.3e}   (>1e8 ~ collinearity territory)")
    tr, te = mre(ytr, ptr), mre(yte, pte)
    if tr > 3.0:
        print("  -> VERDICT: TRAIN error is high => BIAS-limited (model form or "
              "label noise).\n     More design-matrix rows will NOT fix this.")
    elif te > 3 * tr:
        print("  -> VERDICT: train good, test poor => VARIANCE-limited "
              "(overfit / under-sampled).\n     More rows or ridge should help.")
    else:
        print("  -> VERDICT: balanced fit.")
    return fit


# ============================================================================= #
#  STUDY 2.  Learning curve  ->  answers "will more samples help?"               #
# ============================================================================= #
def study_learning_curve(Xtr, ytr, Xte, yte, degree=2, lam=0.0,
                         fractions=(0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0),
                         n_repeats=3, savepath=None, seed=0):
    """
    Train on growing subsets of the training data; track train & test MRE.
      * Both curves converge to a HIGH plateau  -> bias; more data is futile.
      * Big gap that closes as N grows           -> variance; more data helps.
    This is the direct, quantitative answer to 'should I enlarge the DM?'.
    """
    print("\n=== STUDY 2: learning curve (more-data test) ===")
    rng = np.random.default_rng(seed)
    n = len(ytr)
    rows = []
    for f in fractions:
        m = max(int(f * n), degree * Xtr.shape[1] + 5)
        tr_e, te_e = [], []
        for _ in range(n_repeats):
            idx = rng.choice(n, size=min(m, n), replace=False)
            fit = fit_prs_qr(Xtr[idx], ytr[idx], degree=degree, lam=lam)
            tr_e.append(mre(ytr[idx], predict(fit, Xtr[idx])))
            te_e.append(mre(yte, predict(fit, Xte)))
        rows.append((min(m, n), np.mean(tr_e), np.mean(te_e)))
        print(f"  N={min(m,n):6d}   train MRE={np.mean(tr_e):7.3f}%   "
              f"test MRE={np.mean(te_e):7.3f}%")
    rows = np.array(rows)
    # crude plateau check on the test curve
    last_three = rows[-3:, 2]
    slope = (last_three[-1] - last_three[0]) / max(last_three[0], 1e-9)
    if abs(slope) < 0.1:
        print("  -> test MRE has PLATEAUED: enlarging the DM further is "
              "essentially futile (bias floor).")
    else:
        print("  -> test MRE still decreasing with N: more rows are still buying "
              "you accuracy (variance component present).")
    if _HAVE_MPL and savepath:
        plt.figure(figsize=(6, 4))
        plt.plot(rows[:, 0], rows[:, 1], "o-", label="train MRE")
        plt.plot(rows[:, 0], rows[:, 2], "s-", label="test MRE")
        plt.xlabel("training samples N"); plt.ylabel("MRE [%]")
        plt.title("Learning curve"); plt.legend(); plt.tight_layout()
        plt.savefig(savepath, dpi=130); plt.close()
        print(f"  [saved] {savepath}")
    return rows


# ============================================================================= #
#  STUDY 3.  Polynomial-degree sweep  ->  answers "is the basis too poor?"        #
# ============================================================================= #
def study_degree_sweep(Xtr, ytr, Xte, yte, degrees=(1, 2, 3), lam=1e-6):
    """
    If train MRE drops sharply when you go deg2 -> deg3, the response is more
    nonlinear than a quadratic can hold in the region your sampling visits:
    the degree-2 *form* is the bias source.  (Use a tiny ridge so the high-degree
    fits stay solvable when #terms approaches #rows.)
    """
    print("\n=== STUDY 3: polynomial-degree sweep ===")
    p = Xtr.shape[1]
    TERM_CAP = 4000     # above this, full expansion is memory-infeasible
    for d in degrees:
        n_terms = 1 + sum(
            len(list(combinations_with_replacement(range(p), k)))
            for k in range(1, d + 1))
        if n_terms > TERM_CAP:
            print(f"  degree={d}: terms={n_terms:5d}  -> SKIPPED "
                  f"(full deg-{d} expansion infeasible for p={p}; this is itself "
                  "the message: use SELECTIVE high-order terms, not full deg-3).")
            continue
        fit = fit_prs_qr(Xtr, ytr, degree=d, lam=lam)
        ptr, pte = predict(fit, Xtr), predict(fit, Xte)
        flag = "  (#terms approaching/exceeding #rows!)" if n_terms > 0.5 * len(ytr) else ""
        print(f"  degree={d}: terms={n_terms:5d}  cond(R)={fit.cond_R:.2e}  "
              f"train MRE={mre(ytr,ptr):7.3f}%  test MRE={mre(yte,pte):7.3f}%{flag}")
    print("  -> If train MRE falls a lot from deg2->deg3, your bias is the "
          "quadratic basis; consider selective cubic terms (not full deg-3).")


# ============================================================================= #
#  STUDY 4.  Residual localisation  ->  WHERE do the big errors live?            #
# ============================================================================= #
def study_residual_localization(fit, Xtr, ytr, Xte, yte, nbins=8,
                                savepath=None, n_worst=10):
    """
    Per-bin MRE across the response range + a list of the worst test points.
    In your p-PRS plot the catastrophic scatter sits at the LOW-eta tail; this
    quantifies it and tells you whether ~all your error is a handful of points
    (then MRE, a max-norm metric, is being driven by a localised failure).
    """
    print("\n=== STUDY 4: residual localisation across response range ===")
    pte = predict(fit, Xte)
    err = np.abs(yte - pte) / np.maximum(np.abs(yte), 1e-12) * 100.0
    edges = np.linspace(yte.min(), yte.max(), nbins + 1)
    print("   eta-bin                 n     binMRE%    medErr%")
    for i in range(nbins):
        m = (yte >= edges[i]) & (yte <= edges[i + 1] if i == nbins - 1
                                 else yte < edges[i + 1])
        if m.sum():
            print(f"   [{edges[i]:6.3f},{edges[i+1]:6.3f})  {m.sum():5d}   "
                  f"{err[m].max():8.3f}   {np.median(err[m]):8.3f}")
    order = np.argsort(err)[::-1][:n_worst]
    print(f"\n   {n_worst} worst test points (rel-err %, y_true, y_pred):")
    for j in order:
        print(f"     err={err[j]:7.3f}%   y={yte[j]:7.4f}   yhat={pte[j]:7.4f}")
    frac = err[order].sum() / err.sum()
    print(f"   -> top {n_worst} points carry {100*frac:.1f}% of total abs rel-err."
          "  (High => a localised, not global, failure.)")
    if _HAVE_MPL and savepath:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(yte, pte, s=10, c="k", alpha=.5)
        lim = [min(yte.min(), pte.min()), max(yte.max(), pte.max())]
        ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("actual"); ax[0].set_ylabel("pred")
        ax[0].set_title("parity (test)")
        ax[1].scatter(yte, yte - pte, s=10, c="k", alpha=.5)
        ax[1].axhline(0, c="r", ls="--"); ax[1].set_xlabel("actual eta")
        ax[1].set_ylabel("residual"); ax[1].set_title("residual vs response")
        plt.tight_layout(); plt.savefig(savepath, dpi=130); plt.close()
        print(f"   [saved] {savepath}")


# ============================================================================= #
#  STUDY 5.  Conditioning / collinearity                                         #
# ============================================================================= #
def study_conditioning(Xtr, degree=2, top=10):
    """
    Singular-value spectrum + condition number + VIF of the design matrix.
    A near-singular Phi makes coefficients explode and the fit fragile -- a
    DIFFERENT failure mode from bias, and one that QR exposes but does not cure.
    VIF_j >> 10 flags columns that are near-linear combinations of the others
    (e.g. an interaction x14*x15 collinear with x14 and x15).
    """
    print("\n=== STUDY 5: conditioning / collinearity ===")
    Phi, names = poly_features(Xtr, degree, include_bias=True)
    # standardise non-bias columns for an honest condition number
    P = Phi.copy().astype(float)
    P[:, 1:] = (P[:, 1:] - P[:, 1:].mean(0)) / (P[:, 1:].std(0) + 1e-30)
    sv = np.linalg.svd(P, compute_uv=False)
    print(f"  cond(Phi) = {sv[0]/sv[-1]:.3e}   "
          f"(sigma_max={sv[0]:.3e}, sigma_min={sv[-1]:.3e})")
    print(f"  rank(Phi) ~ {np.sum(sv > sv[0]*1e-12)} / {P.shape[1]} columns")
    # VIF via diagonal of inverse correlation matrix (non-bias terms)
    C = np.corrcoef(P[:, 1:], rowvar=False)
    try:
        vif = np.diag(np.linalg.pinv(C))
        worst = np.argsort(vif)[::-1][:top]
        print(f"  top {top} VIF terms (>10 problematic, >100 severe):")
        for w in worst:
            print(f"     VIF={vif[w]:10.1f}   {names[1+w]}")
    except np.linalg.LinAlgError:
        print("  (VIF unavailable: correlation matrix singular -> severe "
              "collinearity, columns are redundant.)")


# ============================================================================= #
#  STUDY 6.  Leverage & influence (Cook's distance)                              #
# ============================================================================= #
def study_leverage(fit, Xtr, ytr, top=10):
    """
    Hat-matrix leverage h_ii and Cook's distance D_i.  High-leverage tail points
    that the quadratic cannot reach show up here; they are usually the same
    points driving your max-norm MRE.  Distinguishes 'a few pathological rows'
    from 'globally wrong model'.
    """
    print("\n=== STUDY 6: leverage & influence ===")
    Phi, _ = poly_features(Xtr, fit.degree, fit.include_bias)
    if fit.mu is not None:
        Phi = (Phi - fit.mu) / fit.sd
        if fit.include_bias:
            Phi[:, 0] = 1.0
    Q, _ = np.linalg.qr(Phi)
    h = np.sum(Q ** 2, axis=1)                       # leverage = diag(Q Q^T)
    p = Phi.shape[1]
    resid = ytr - predict(fit, Xtr)
    mse = np.sum(resid ** 2) / max(len(ytr) - p, 1)
    cook = (resid ** 2 / (p * mse)) * (h / np.maximum((1 - h) ** 2, 1e-12))
    print(f"  mean leverage = {h.mean():.4f} (= p/n = {p/len(ytr):.4f}); "
          f"max leverage = {h.max():.4f}")
    worst = np.argsort(cook)[::-1][:top]
    print(f"  top {top} influential rows (Cook's D, leverage, residual, y):")
    for w in worst:
        print(f"     D={cook[w]:8.4f}  h={h[w]:.4f}  r={resid[w]:+8.4f}  y={ytr[w]:7.4f}")


# ============================================================================= #
#  STUDY 7.  Subset-adequacy ceiling  (the deep p-SAC question)                  #
# ============================================================================= #
def study_subset_ceiling(X_full, y_full, subset_idx, degree=2, lam=1e-8):
    """
    *Optional but decisive.*  Using your ORIGINAL full-PRS simulation set
    (all n reactions perturbed), ask: how well can a PRS in ONLY the n_ subset
    columns reproduce eta?

        R2_full  : PRS on all n columns
        R2_subset: PRS on just the subset columns
    If R2_subset << R2_full, the sparsity selection has dropped reaction
    directions that genuinely move eta in the sampled region -> there is an
    information ceiling no p-PRS sampling/refinement can beat, because the
    *inputs* are insufficient, not the surrogate.  This is the cleanest test of
    'is my subset selection the problem?'.

    Parameters
    ----------
    X_full : (N, n) full design matrix (all reactions perturbed)
    y_full : (N,)   responses
    subset_idx : list[int] of the n_ retained reaction columns (0-indexed)
    """
    print("\n=== STUDY 7: subset-adequacy ceiling (needs full-PRS data) ===")
    fit_full = fit_prs_qr(X_full, y_full, degree=degree, lam=lam)
    r2_full = r2(y_full, predict(fit_full, X_full))
    Xs = X_full[:, subset_idx]
    fit_sub = fit_prs_qr(Xs, y_full, degree=degree, lam=lam)
    r2_sub = r2(y_full, predict(fit_sub, Xs))
    print(f"  R2 using ALL {X_full.shape[1]} reactions : {r2_full:.5f}")
    print(f"  R2 using ONLY {len(subset_idx)} subset   : {r2_sub:.5f}")
    drop = r2_full - r2_sub
    print(f"  variance lost by sparsification: {100*drop:.2f}% of explainable var")
    if drop > 0.02:
        print("  -> CEILING DETECTED: the subset omits directions that move eta."
              "\n     No p-PRS sampling can recover this; revisit the sparsity "
              "selection (e.g. add state-dependent / tail sensitivities).")
    else:
        print("  -> subset retains the signal; the p-PRS problem is surrogate-"
              "side (basis/sampling/conditioning), not selection.")
    return r2_full, r2_sub


# ============================================================================= #
#  Convenience driver                                                            #
# ============================================================================= #
def run_all(Xtr, ytr, Xte, yte, degree=2, lam=0.0, outdir="."):
    print("#" * 78)
    print(f"# PRS DIAGNOSTIC RUN   p={Xtr.shape[1]} vars   "
          f"Ntrain={len(ytr)}  Ntest={len(yte)}")
    print("#" * 78)
    fit = study_bias_variance(Xtr, ytr, Xte, yte, degree=degree, lam=lam)
    study_learning_curve(Xtr, ytr, Xte, yte, degree=degree, lam=lam,
                         savepath=f"{outdir}/learning_curve.png")
    study_degree_sweep(Xtr, ytr, Xte, yte)
    study_residual_localization(fit, Xtr, ytr, Xte, yte,
                                savepath=f"{outdir}/residuals.png")
    study_conditioning(Xtr, degree=degree)
    study_leverage(fit, Xtr, ytr)
    return fit


if __name__ == "__main__":
    print("This is a library. See demo_run.py for a worked example, or call "
          "run_all(Xtr,ytr,Xte,yte) on your own arrays.")
