"""
sa_correlation_analysis.py

Cheap, simulation-free reaction-reaction correlation analysis built only
from brute-force, one-at-a-time (first-order, central-difference) sensitivity
analysis -- exactly the kind of SA you already compute with
SA_3P_BruteForce.py at each validation condition. No Monte Carlo design
matrix, no second-order (cross/pairwise) finite differences required.

THE IDEA (PCAS -- Principal Component Analysis of Sensitivities, the
Vajda/Valko/Turanyi-style approach from mechanism-reduction literature,
applied here to MUQ-SAC reaction-level perturbations):

A single brute-force SA run at ONE condition gives one number per reaction
-- not enough to define a correlation. But if brute-force SA is already run
at K different operating conditions (T-P-phi grid points, or K different
validation targets), each reaction r has a K-length vector

    S[:, r] = [ d(eta)/d(zeta_r) at condition 1, ..., at condition K ]

Two reactions are "functionally correlated" -- in the sense relevant to
p-PRS truncation -- if these sensitivity patterns move together across
conditions, i.e. Pearson r between columns S[:, i] and S[:, j] is large.
This is a first-order, linear notion: no second derivative, no random
sampling campaign, only brute-force SA you already have (or a modest
N x K grid of one-at-a-time finite differences otherwise).

WHAT THIS DOES NOT TELL YOU: it says nothing about the covariance Sigma
you impose when SAMPLING reactions (a modelling choice; if Sigma is
block-diagonal per-reaction in your SAC sampler, that correlation is
zero by construction -- check Sigma directly, no SA needed for that part).
"""

import numpy as np
from scipy import stats


def build_sensitivity_matrix(sa_results, reaction_ids=None, condition_ids=None):
    """
    Assemble a (K conditions x N reactions) sensitivity matrix from your
    brute-force SA output.

    sa_results can be:
      - a dict: {condition_id: {reaction_id: sensitivity_value, ...}, ...}
      - a 2D array-like already shaped (K, N)
    """
    if isinstance(sa_results, dict):
        condition_ids = list(condition_ids) if condition_ids is not None else sorted(sa_results.keys())
        first = sa_results[condition_ids[0]]
        reaction_ids = list(reaction_ids) if reaction_ids is not None else sorted(first.keys())
        S = np.array(
            [[sa_results[c][r] for r in reaction_ids] for c in condition_ids],
            dtype=float,
        )
    else:
        S = np.asarray(sa_results, dtype=float)
        if S.ndim != 2:
            raise ValueError("sa_results array must be 2D (K conditions x N reactions)")
        K, N = S.shape
        reaction_ids = list(reaction_ids) if reaction_ids is not None else list(range(N))
        condition_ids = list(condition_ids) if condition_ids is not None else list(range(K))
    return S, reaction_ids, condition_ids


def reaction_correlation_matrix(S, min_conditions=10):
    """
    Pearson correlation between every pair of reactions (columns of S),
    using only their brute-force SA pattern across K conditions.
    Returns (rho, pval), both (N, N).
    """
    K, N = S.shape
    if K < min_conditions:
        raise ValueError(
            f"Only {K} conditions in S; need >= {min_conditions} brute-force "
            "SA points before a Pearson estimate is worth trusting. Run "
            "your one-at-a-time SA at a few more T/P/phi points -- still "
            "N x K deterministic sims, nowhere near a Monte Carlo campaign."
        )

    rho = np.corrcoef(S, rowvar=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = rho * np.sqrt((K - 2) / np.clip(1 - rho**2, 1e-12, None))
    pval = 2 * stats.t.sf(np.abs(t_stat), df=K - 2)
    np.fill_diagonal(pval, 0.0)
    return rho, pval


def flag_redundant_pairs(rho, pval, reaction_ids, rho_threshold=0.85, alpha=0.05):
    """Candidate redundant pairs: |rho| >= threshold AND significant at alpha."""
    N = rho.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(rho[i, j]) >= rho_threshold and pval[i, j] <= alpha:
                pairs.append((reaction_ids[i], reaction_ids[j], rho[i, j], pval[i, j]))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def pca_coverage(S, kept_idx, var_threshold=0.95):
    """
    Eigen-decompose the (z-scored) sensitivity matrix to get, purely from
    data already on hand: (a) effectively-independent influence directions
    n_eff across the condition set, and (b) the variance share your kept
    subset captures. The correlation-aware, simulation-free analogue of a
    subset-adequacy ceiling test.
    """
    mu = S.mean(axis=0, keepdims=True)
    sd = S.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Sz = (S - mu) / sd

    _, sv, _ = np.linalg.svd(Sz, full_matrices=False)
    eigvals = sv**2
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)
    n_eff = int(np.searchsorted(cumulative, var_threshold) + 1)

    total_var = np.sum(Sz**2)
    kept_var = np.sum(Sz[:, kept_idx] ** 2)
    kept_share = float(kept_var / total_var)

    return {"explained_variance_ratio": explained, "n_eff": n_eff, "kept_share": kept_share}


def cross_check_dropped_reactions(rho, reaction_ids, kept_idx, dropped_idx, alert_threshold=0.5):
    """
    For every dropped reaction, the strongest correlation with any kept
    reaction. Low max|rho| => not represented by anything kept => a
    concrete, specific candidate source of p-PRS bias.
    """
    report = []
    for j in dropped_idx:
        best = max((abs(rho[j, i]) for i in kept_idx), default=0.0)
        report.append({
            "reaction": reaction_ids[j],
            "max_abs_corr_with_kept": best,
            "well_covered": best >= alert_threshold,
        })
    return sorted(report, key=lambda d: d["max_abs_corr_with_kept"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, N = 40, 20
    latent = rng.normal(size=K)

    S = rng.normal(scale=0.3, size=(K, N))
    S[:, 0] += 1.5 * latent   # reaction 0 driven by a latent factor
    S[:, 1] += 1.4 * latent   # reaction 1 strongly correlated with 0
    # reaction 5 stays pure noise -- independent

    reaction_ids = [f"R{i}" for i in range(N)]

    print("Building sensitivity matrix...")
    Smat, rxn_ids, cond_ids = build_sensitivity_matrix(S, reaction_ids=reaction_ids)

    # quick round-trip check of the dict input path too
    sa_dict = {c: {r: Smat[c, ri] for ri, r in enumerate(rxn_ids)} for c in range(K)}
    Smat2, rxn_ids2, cond_ids2 = build_sensitivity_matrix(sa_dict)
    assert np.allclose(Smat, Smat2[:, [rxn_ids2.index(r) for r in rxn_ids]])
    print("Dict-input round-trip OK.")

    print("\nComputing reaction-reaction correlation matrix...")
    rho, pval = reaction_correlation_matrix(Smat, min_conditions=10)

    print("\nTop candidate redundant pairs (|rho| >= 0.85, p <= 0.05):")
    for r1, r2, rval, pv in flag_redundant_pairs(rho, pval, rxn_ids, rho_threshold=0.85):
        print(f"  {r1} <-> {r2}: rho = {rval:+.3f}  (p = {pv:.2e})")

    kept_idx = [0, 2, 3, 4, 6, 7, 8]
    dropped_idx = [i for i in range(N) if i not in kept_idx]

    cov = pca_coverage(Smat, kept_idx, var_threshold=0.95)
    print(f"\nEffectively independent directions across {K} conditions "
          f"(95% variance): n_eff = {cov['n_eff']}")
    print(f"Variance share captured by the kept subset alone: {cov['kept_share']*100:.1f}%")

    print("\nDropped reactions, worst-covered first:")
    for entry in cross_check_dropped_reactions(rho, rxn_ids, kept_idx, dropped_idx):
        flag = "OK (redundant with a kept reaction)" if entry["well_covered"] else "GAP -- not covered by kept set"
        print(f"  {entry['reaction']}: max|rho| with kept = {entry['max_abs_corr_with_kept']:.3f}  -> {flag}")
