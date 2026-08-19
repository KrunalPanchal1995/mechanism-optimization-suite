"""
correlation_analysis.py  (v2 -- rescue, not pruning)

Two outputs you asked for, both plain CSV:

  1. correlation_matrix_<param>.csv
     One Pearson-r value per reaction pair, computed using your SA values
     across ALL targets/cases at once (need many cases to compute this --
     a single target gives one number per reaction, nothing to correlate).

  2. selection_with_rescue_<param>.csv
     Long format, one row per (case, reaction):
       S_value, abs_S, selected_by_threshold, rescued_by_correlation,
       correlated_with, final_in_PRS
     selected_by_threshold : |S| >= threshold, AT THAT CASE.
     rescued_by_correlation: NOT selected at that case, but correlated
       (using the all-targets correlation matrix from output #1) with a
       reaction that WAS selected at that same case.
     final_in_PRS = selected_by_threshold OR rescued_by_correlation
       -- this is the column you actually want to build p-PRS from.

Handles both your pickle structures (3-param list-valued, or scalar
A-only with a different active reaction subset per case) automatically.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# LOAD + DETECT STRUCTURE  (unchanged from before, this part worked)
# ============================================================

def load_sens_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def detect_structure(sens_dict, param_names=None):
    first_case = next(iter(sens_dict.values()))
    first_val = next(iter(first_case.values()))
    if isinstance(first_val, (list, tuple, np.ndarray)) and len(first_val) > 1:
        n_params = len(first_val)
        if param_names is None:
            default = ["A", "n", "Ea"]
            param_names = default[:n_params] if n_params <= 3 else [f"p{i}" for i in range(n_params)]
        return "vector", list(param_names)
    return "scalar", list(param_names or ["A"])


def to_wide_frames(sens_dict, structure, param_names):
    """(case x reaction) DataFrame per parameter. Missing -> NaN."""
    records = []
    for case_id, rxn_dict in sens_dict.items():
        for rxn_id, val in rxn_dict.items():
            vals = np.asarray(val, dtype=float).ravel() if structure == "vector" \
                else np.asarray([val], dtype=float).ravel()
            row = {"case": str(case_id), "reaction": str(rxn_id)}
            row.update(zip(param_names, vals))
            records.append(row)
    df_long = pd.DataFrame.from_records(records)
    return {p: df_long.pivot_table(index="case", columns="reaction", values=p, aggfunc="first")
            for p in param_names}


# ============================================================
# CORRELATION ACROSS ALL TARGETS  (the thing you actually asked for)
# ============================================================

def correlation_matrix_with_pvalues(wide_df, min_pairs=8, method="pearson"):
    """Pairwise correlation between every column in *wide_df* across shared rows.

    Parameters
    ----------
    wide_df   : DataFrame, rows = cases, columns = features (reactions or
                reaction-parameter pairs). NaNs are handled pairwise.
    min_pairs : int – pairs with fewer shared cases than this are left NaN.
    method    : "pearson" (default) | "spearman"
                Pearson r measures linear co-movement; Spearman rank-r is
                more robust to the noise in first-order central-difference SA
                and catches monotone non-linear coupling.
    """
    cols = wide_df.columns
    N = len(cols)
    X = wide_df.values
    rho   = np.full((N, N), np.nan)
    pval  = np.full((N, N), np.nan)
    npairs = np.zeros((N, N), dtype=int)
    valid  = ~np.isnan(X)

    for i in range(N):
        for j in range(i, N):
            both = valid[:, i] & valid[:, j]
            n = int(both.sum())
            npairs[i, j] = npairs[j, i] = n
            if i == j:
                rho[i, j] = 1.0
                continue
            if n < min_pairs:
                continue
            xi, xj = X[both, i], X[both, j]
            if xi.std() == 0 or xj.std() == 0:
                continue
            if method == "spearman":
                r, p = stats.spearmanr(xi, xj)
            else:   # pearson (default)
                r = float(np.corrcoef(xi, xj)[0, 1])
                df = n - 2
                t_stat = r * np.sqrt(df / max(1e-12, 1 - r**2))
                p = 2 * stats.t.sf(abs(t_stat), df=df)
            rho[i, j]  = rho[j, i]  = float(r)
            pval[i, j] = pval[j, i] = float(p)

    return (pd.DataFrame(rho,    index=cols, columns=cols),
            pd.DataFrame(pval,   index=cols, columns=cols),
            pd.DataFrame(npairs, index=cols, columns=cols))


def min_significant_correlation(n_cases, alpha=0.05):
    df = max(1, n_cases - 2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return float(t_crit / np.sqrt(df + t_crit**2))


def suggest_corr_cutoff(n_cases, alpha=0.05, practical_floor=0.8):
    sig = min_significant_correlation(n_cases, alpha=alpha)
    return max(practical_floor, sig), sig


# ============================================================
# HEATMAP (optional visual, secondary to the CSVs)
# ============================================================

def plot_correlation_heatmap(rho_df, title, outpath, top_n=40, importance=None):
    cols = rho_df.columns
    order = importance.reindex(cols).fillna(0).abs().sort_values(ascending=False).index \
        if importance is not None else cols
    truncated = len(order) > top_n
    keep = order[:top_n] if truncated else order
    sub = rho_df.loc[keep, keep]
    if truncated:
        title = f"{title}  (top {top_n} by max|S|)"
    n = len(sub)
    size = min(20, max(6, 0.28 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(sub.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [str(c)[:30] for c in sub.columns]
    ax.set_xticklabels(labels, rotation=90, fontsize=max(4, 9 - n // 12))
    ax.set_yticklabels(labels, fontsize=max(4, 9 - n // 12))
    if n <= 25:
        for i in range(n):
            for j in range(n):
                v = sub.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ============================================================
# THRESHOLD SELECTION + RESCUE  (the new part)
# ============================================================

def build_selection_table(wide_df, rho_df, threshold, corr_cutoff):
    """
    Long table, one row per (case, reaction):
        S_value, abs_S, selected_by_threshold, rescued_by_correlation,
        correlated_with, final_in_PRS
    """
    rows = []
    for case in wide_df.index:
        s_row = wide_df.loc[case]
        not_selected = s_row[s_row.abs() < threshold].dropna().index.tolist()
        selected = s_row[s_row.abs() >= threshold].dropna().index.tolist()

        for rxn in wide_df.columns:
            s_val = s_row[rxn]
            sel = rxn in selected
            rescued, corr_with = False, []

            if not sel and pd.notna(s_val):
                others = [r for r in selected if r != rxn]
                if others:
                    corrs = rho_df.loc[rxn, others].dropna()
                    hits = corrs[corrs.abs() >= corr_cutoff]
                    if len(hits):
                        rescued = True
                        corr_with = [f"{r} (r={v:+.2f})" for r, v in hits.items()]

            rows.append({
                "case": case, "reaction": rxn,
                "S_value": s_val,
                "abs_S": abs(s_val) if pd.notna(s_val) else np.nan,
                "selected_by_threshold": sel,
                "rescued_by_correlation": rescued,
                "correlated_with": "; ".join(corr_with),
                "final_in_PRS": sel or rescued,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["case", "abs_S"], ascending=[True, False], na_position="last")


def load_correlation_csv(path):
    """Read back a correlation_matrix_<param>.csv exactly as run_pipeline
    wrote it (reaction names on both rows and columns)."""
    return pd.read_csv(path, index_col=0)


def final_active_set_per_target(pickle_path, correlation_csv_path, output_dir,
                                 param_name="A", threshold=0.05, corr_cutoff=0.9, threshold_corr=0.01):
    """
    Cross-checks the ORIGINAL SA pickle against an ALREADY-SAVED
    correlation_matrix_<param>.csv (deliberately decoupled from run_pipeline,
    so you can re-run this with a different corr_cutoff without re-running
    the whole SA load + correlation step):

        selected = reactions with |S| >= threshold, AT THAT TARGET
        rescued  = reactions NOT selected, but with correlation
                   >= corr_cutoff (from the CSV) to a reaction that
                   WAS selected at that SAME target
        final    = selected | rescued only if the selected has |S| > threshold_corr

    Writes final_active_set_<param>.csv (long format) and
    final_active_set_<param>.pkl (dict: target -> sorted reaction list,
    same shape as your sens.pkl). Returns that dict.
    """
    sens_dict = load_sens_pickle(pickle_path)
    structure, detected_names = detect_structure(sens_dict)
    wide_frames = to_wide_frames(sens_dict, structure, detected_names)
    if param_name not in wide_frames:
        raise KeyError(f"'{param_name}' not found; available: {list(wide_frames)}")
    wide = wide_frames[param_name]

    rho = load_correlation_csv(correlation_csv_path)
    common = [r for r in wide.columns if r in rho.index]
    wide, rho = wide[common], rho.loc[common, common]

    final_sets, rows = {}, []
    for case in wide.index:
        s_row = wide.loc[case]
        selected = set(s_row[s_row.abs() >= threshold].dropna().index)
        not_selected = set(s_row[s_row.abs() < threshold].dropna().index)
        rescued = set()
        for rxn in common:
            if rxn in selected or pd.isna(s_row[rxn]):
                continue
            others = [r for r in selected if r != rxn]
            if others and (rho.loc[rxn, others].dropna().abs() >= corr_cutoff).any() and sens_dict[case][rxn] > threshold_corr: 
                rescued.add(rxn)

        final = sorted(selected | rescued)
        final_sets[case] = final
        for rxn in final:
            rows.append({"case": case, "reaction": rxn, "S_value": s_row[rxn],
                         "in_selected": rxn in selected, "in_rescued": rxn in rescued})


    out_df = pd.DataFrame(rows, columns=["case", "reaction", "S_value", "in_selected", "in_rescued"])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"final_active_set_{param_name}.csv")
    pkl_path = os.path.join(output_dir, f"final_active_set_{param_name}.pkl")
    out_df.to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump(final_sets, f)

    avg_final = out_df.groupby("case").size().reindex(wide.index, fill_value=0).mean()
    avg_thresh_only = wide.abs().ge(threshold).fillna(False).sum(axis=1).mean()
    print(f"[{param_name}] avg active-set size per target: "
          f"threshold-only = {avg_thresh_only:.1f}, "
          f"threshold+rescue(>{corr_cutoff}) = {avg_final:.1f}")
    print(f"Wrote: {csv_path}\n       {pkl_path}")
    return final_sets

# ============================================================
# END-TO-END
# ============================================================

def run_pipeline(pickle_path, output_dir, threshold=0.05, corr_cutoff=None,
                  alpha=0.05, min_pairs=8, top_n_heatmap=40, param_names=None,
                  final_set_corr_cutoff=0.9):
    os.makedirs(output_dir, exist_ok=True)
    sens_dict = load_sens_pickle(pickle_path)
    structure, param_names = detect_structure(sens_dict, param_names)
    wide_frames = to_wide_frames(sens_dict, structure, param_names)
    n_cases = len(next(iter(wide_frames.values())).index)

    suggested_cutoff, sig_threshold = suggest_corr_cutoff(n_cases, alpha=alpha)
    if corr_cutoff is None:
        corr_cutoff = suggested_cutoff

    print(f"Structure detected : {structure} -> parameters {param_names}")
    print(f"Cases available    : {n_cases}")
    print(f"Min |r| significant at alpha={alpha} for n={n_cases}: {sig_threshold:.3f}")
    print(f"Correlation cutoff used for rescue: {corr_cutoff:.3f}"
          f"{' (suggested default)' if corr_cutoff == suggested_cutoff else ' (user-specified)'}")

    for p in param_names:
        wide = wide_frames[p]
        rho, pval, npairs = correlation_matrix_with_pvalues(wide, min_pairs=min_pairs)

        #rho.round(4).to_csv(os.path.join(output_dir, f"correlation_matrix_{p}.csv"))

        importance = wide.abs().max(axis=0)
        plot_correlation_heatmap(
            rho, f"Reaction correlation across all targets -- parameter {p}",
            os.path.join(output_dir, f"correlation_heatmap_{p}.png"),
            top_n=top_n_heatmap, importance=importance,
        )

        sel_table = build_selection_table(wide, rho, threshold, corr_cutoff)
        sel_table["S_value"] = sel_table["S_value"].round(6)
        sel_table["abs_S"] = sel_table["abs_S"].round(6)
        sel_table.to_csv(os.path.join(output_dir, f"selection_with_rescue_{p}.csv"), index=False)
        corr_csv_path = os.path.join(output_dir, f"correlation_matrix_{p}.csv")
        rho.round(4).to_csv(corr_csv_path)

        final_sets = final_active_set_per_target(
            pickle_path, corr_csv_path, output_dir,
            param_name=p, threshold=threshold, corr_cutoff=final_set_corr_cutoff,
        )
        n_no_corr = sel_table.groupby("case")["selected_by_threshold"].sum()
        n_with_corr = sel_table.groupby("case")["final_in_PRS"].sum()
        n_rescued = sel_table["rescued_by_correlation"].sum()

        print(f"\n[{p}] threshold-only avg active set per case : {n_no_corr.mean():.1f}")
        print(f"[{p}] threshold + rescue avg active set per case: {n_with_corr.mean():.1f}")
        print(f"[{p}] total (case, reaction) rescue events       : {int(n_rescued)}")

        with open(os.path.join(output_dir, f"correlations_{p}.pkl"), "wb") as f:
            pickle.dump({"rho": rho, "pval": pval, "n_shared_cases": npairs}, f)

    print(f"\nWrote CSVs and heatmaps to: {output_dir}")
    return final_sets


# ============================================================
# COMBINED 3-PARAM PIPELINE (cross-Arrhenius, SA-data-only)
# Used when design_type != "A-facto" in run_script.py
# ============================================================

def build_rxn_metadata_from_unsrt(unsrt_data, param_names=None):
    """
    Build per-reaction label metadata from an already-loaded unsrt_data dict.
    Returns {rxn_eq -> {rIndex, feature_labels}}.

    The unsrt_data is used **for naming only** (rIndex → label prefix like
    "R1093"). No Cholesky/covariance/UQ-prior is read — all correlations
    come purely from the SA sensitivity data.
    """
    if param_names is None:
        param_names = ["A", "n", "Ea"]
    meta = {}
    for rxn_full, obj in unsrt_data.items():
        rxn_eq = rxn_full.split(":")[0]          # strip PLOG index if present
        rIndex = getattr(obj, "rIndex", None) or rxn_eq
        labels = [f"{rIndex}_{p}" for p in param_names]
        meta[rxn_eq] = {"rIndex": rIndex, "feature_labels": labels}
    return meta


def to_feature_frame_combined(sens_dict, param_names, meta):
    """
    Build a (K cases × N*n_params) wide DataFrame.
    Columns: "R1093_A", "R1093_n", "R1093_Ea", ...
    Reactions sorted by rIndex numeric value.
    """
    def _rindex_key(rxn_eq):
        r = meta.get(rxn_eq, {}).get("rIndex", rxn_eq)
        digits = "".join(c for c in r if c.isdigit())
        return int(digits) if digits else float("inf")

    all_rxns = sorted(
        {rxn for case_vals in sens_dict.values() for rxn in case_vals},
        key=_rindex_key,
    )
    feature_cols = []
    rxn_to_labels = {}
    for k, rxn in enumerate(all_rxns):
        labels = meta[rxn]["feature_labels"] if rxn in meta else [f"Rx{k}_{p}" for p in param_names]
        feature_cols.extend(labels)
        rxn_to_labels[rxn] = labels

    rows = {}
    for case_id, rxn_dict in sens_dict.items():
        row = {}
        for rxn, vals in rxn_dict.items():
            vals_arr = np.asarray(vals, dtype=float).ravel()
            for lbl, v in zip(rxn_to_labels.get(rxn, []), vals_arr):
                row[lbl] = v
        rows[str(case_id)] = row

    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=feature_cols)
    df.index.name = "case"
    return df


def combined_correlation_matrix_3p(feature_df, meta, param_names,
                                    min_pairs=8, method="pearson"):
    """
    Pairwise correlation for every (feature_i, feature_j) pair — including
    within-reaction A/n/Ea pairs — computed **entirely from SA data**.

    No analytic prior (Cholesky / UQ covariance) is used.  All entries, both
    within and across reactions, reflect how the sensitivity coefficients
    co-vary across the K validation cases.

    Returns rho_df, pval_df, npairs_df  (no source_df — everything is "data").
    """
    return correlation_matrix_with_pvalues(
        feature_df, min_pairs=min_pairs, method=method)


def _build_combined_selection_detail(feature_df, rho_df,
                                      threshold, corr_cutoff, threshold_corr):
    """
    Long table: one row per (case, feature) explaining selection decisions.
    All correlations used here are SA-data-only (no UQ prior).
    """
    common = [c for c in feature_df.columns if c in rho_df.columns]
    rows = []
    for case in feature_df.index:
        s_row = feature_df.loc[case, common]
        selected = set(s_row[s_row.abs() >= threshold].dropna().index)
        for feat in common:
            s_val = s_row[feat]
            sel = feat in selected
            rescued, corr_with = False, []
            if not sel and pd.notna(s_val) and abs(s_val) > threshold_corr:
                others = [r for r in selected if r != feat]
                if others and feat in rho_df.index:
                    for other_feat, r_val in rho_df.loc[feat, others].dropna().items():
                        if abs(r_val) >= corr_cutoff:
                            corr_with.append(f"{other_feat}(r={r_val:+.2f})")
                    if corr_with:
                        rescued = True
            rows.append({
                "case": case, "feature": feat,
                "S_value": round(float(s_val), 6) if pd.notna(s_val) else None,
                "abs_S": round(abs(float(s_val)), 6) if pd.notna(s_val) else None,
                "selected_by_threshold": sel,
                "rescued_by_correlation": rescued,
                "correlated_with": "; ".join(corr_with),
                "final_in_PRS": sel or rescued,
            })
    return pd.DataFrame(rows).sort_values(
        ["case", "abs_S"], ascending=[True, False], na_position="last")


def final_active_set_combined_3p(feature_df, rho_df, output_dir,
                                  threshold=0.05, corr_cutoff=0.9,
                                  threshold_corr=0.01):
    """
    Per-target active feature sets for the combined cross-Arrhenius pipeline.
    All rescue decisions are based on SA-data correlation only (no UQ prior).
    Returns dict: {str(case) -> sorted list of active feature strings}.
    Writes final_active_set_combined.{csv,pkl} to output_dir.
    """
    common = [c for c in feature_df.columns if c in rho_df.columns]
    final_sets, rows = {}, []

    for case in feature_df.index:
        s_row = feature_df.loc[case, common]
        selected = set(s_row[s_row.abs() >= threshold].dropna().index)
        rescued = set()

        for feat in common:
            if feat in selected or pd.isna(s_row[feat]):
                continue
            if abs(s_row[feat]) <= threshold_corr:
                continue
            if feat not in rho_df.index:
                continue
            others = [r for r in selected if r != feat]
            if not others:
                continue
            for other_feat, r_val in rho_df.loc[feat, others].dropna().items():
                if abs(r_val) >= corr_cutoff:
                    rescued.add(feat)
                    break

        final = sorted(selected | rescued)
        final_sets[case] = final
        for feat in final:
            rows.append({
                "case": case, "feature": feat,
                "S_value": s_row[feat],
                "in_selected": feat in selected,
                "in_rescued": feat in rescued,
            })

    os.makedirs(output_dir, exist_ok=True)
    out_df = pd.DataFrame(
        rows, columns=["case", "feature", "S_value", "in_selected", "in_rescued"])
    csv_path = os.path.join(output_dir, "final_active_set_combined.csv")
    pkl_path = os.path.join(output_dir, "final_active_set_combined.pkl")
    out_df.to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump(final_sets, f)

    avg_final = (out_df.groupby("case").size()
                 .reindex(feature_df.index, fill_value=0).mean())
    avg_thresh = (feature_df[common].abs().ge(threshold)
                  .fillna(False).sum(axis=1).mean())
    print(f"[combined-3p] avg active features/case: "
          f"threshold-only={avg_thresh:.1f}, "
          f"threshold+rescue(>{corr_cutoff})={avg_final:.1f}")
    print(f"Wrote: {csv_path}\n       {pkl_path}")
    return final_sets


def run_combined_3p_pipeline(sens_dict, unsrt_data, output_dir,
                              param_names=None,
                              threshold=0.05,
                              corr_cutoff=0.9,
                              threshold_corr=0.01,
                              min_pairs=8,
                              method="pearson",
                              alpha=0.05):
    """
    Full cross-Arrhenius SA-data correlation pipeline for the 3-param case.
    All correlations — within a reaction (A/n/Ea) and across reactions — are
    derived **purely from the SA sensitivity data**. No Cholesky / UQ prior is
    used. Accepts already-loaded dicts to avoid double-loading pickles.

    Parameters
    ----------
    sens_dict   : dict – {case_id -> {rxn_eq -> [S_A, S_n, S_Ea]}}
    unsrt_data  : dict – {rxn_eq -> object with .rIndex, .activeParameters}
                  Used for naming (rIndex label prefix) only; L/cov not read.
    output_dir  : str  – where to write correlation CSVs and heatmaps
    threshold   : float – primary |S| selection cutoff (default 0.05)
    corr_cutoff : float – rescue trigger: |r| ≥ this with a selected feature
    threshold_corr : float – rescue eligibility floor on |S| (default 0.01)
    method      : "pearson" (default) | "spearman"
                  Pearson is the recommended default — it maps directly onto
                  the linear-coupling assumption behind p-PRS cross-terms.
                  Spearman is a rank-based alternative that is more robust to
                  the noise in first-order central-difference SA coefficients.
    alpha       : float – significance level for reporting min |r|

    Returns
    -------
    dict: {str(case) -> sorted list of active feature strings}
          e.g. {"0": ["R1063_A", "R1063_Ea", "R1071_A", "R1071_n", ...], ...}
          Strings match unsrt_data[rxn].activeParameters format, so they can
          be compared directly against PartialPRS.active_params.
    """
    if param_names is None:
        param_names = ["A", "n", "Ea"]
    os.makedirs(output_dir, exist_ok=True)

    # Build feature labels from unsrt (rIndex only; no L/cov used)
    meta = build_rxn_metadata_from_unsrt(unsrt_data, param_names)
    print(f"\n[combined-3p] {len(meta)} reactions; method={method}")

    feature_df = to_feature_frame_combined(sens_dict, param_names, meta)
    n_cases, n_feats = feature_df.shape
    print(f"[combined-3p] feature matrix: {n_cases} cases × {n_feats} features "
          f"({n_feats // len(param_names)} rxns × {len(param_names)} params)")

    sig_r = min_significant_correlation(n_cases, alpha=alpha)
    print(f"[combined-3p] min |r| significant at α={alpha}, n={n_cases}: {sig_r:.3f}  "
          f"(rescue cutoff={corr_cutoff})")

    # SA-data-only correlation matrix (no prior overwrite)
    rho_df, pval_df, npairs_df = combined_correlation_matrix_3p(
        feature_df, meta, param_names, min_pairs=min_pairs, method=method)

    # Save full correlation matrix
    rho_df.round(4).to_csv(os.path.join(output_dir, "correlation_matrix_combined.csv"))

    # Heatmap
    try:
        importance = feature_df.abs().max(axis=0)
        plot_correlation_heatmap(
            rho_df,
            f"Cross-Arrhenius SA-data correlation ({method.capitalize()})",
            os.path.join(output_dir, "correlation_heatmap_combined.png"),
            top_n=n_feats,
            importance=importance,
        )
    except Exception as _e:
        print(f"[combined-3p] heatmap skipped: {_e}")

    # Per-param selection tables (one per A/n/Ea for easy inspection)
    for p in param_names:
        param_cols = [c for c in feature_df.columns if c.endswith(f"_{p}")]
        if not param_cols:
            continue
        wide_p = feature_df[param_cols].copy()
        wide_p.columns = [c[: -(len(p) + 1)] for c in wide_p.columns]
        rho_p = rho_df.loc[param_cols, param_cols].copy()
        rho_p.index = wide_p.columns
        rho_p.columns = wide_p.columns
        sel_t = build_selection_table(wide_p, rho_p, threshold, corr_cutoff)
        sel_t.to_csv(os.path.join(output_dir, f"selection_with_rescue_{p}.csv"),
                     index=False)

    # Full (reaction,parameter) feature-level selection detail
    sel_combined = _build_combined_selection_detail(
        feature_df, rho_df, threshold, corr_cutoff, threshold_corr)
    sel_combined.to_csv(
        os.path.join(output_dir, "selection_with_rescue_combined.csv"), index=False)

    n_rescued = int(sel_combined["rescued_by_correlation"].sum())
    print(f"[combined-3p] total rescue events (SA-data, |r|≥{corr_cutoff}): {n_rescued}")

    # Save pkl for downstream / diagnostic use
    with open(os.path.join(output_dir, "correlations_combined.pkl"), "wb") as f:
        pickle.dump({"rho": rho_df, "pval": pval_df,
                     "n_shared_cases": npairs_df, "method": method}, f)

    # Final active feature sets (one per case)
    final_sets = final_active_set_combined_3p(
        feature_df, rho_df, output_dir,
        threshold=threshold, corr_cutoff=corr_cutoff,
        threshold_corr=threshold_corr,
    )
    print(f"[combined-3p] all outputs written to: {output_dir}")
    return final_sets


if __name__ == "__main__":
    
    PICKLE_PATH = "sens_parameters.pkl"   # or "../sens_parameters.pkl"
    OUTPUT_DIR  = "sa_pipeline_v2_out"

    sens_dict = load_sens_pickle(PICKLE_PATH)

    # quick sanity check: read case and rxn straight off your real dict
    # before running anything, so you can confirm the structure matches
    case_ids = list(sens_dict.keys())
    first_case = case_ids[0]
    rxn_ids = list(sens_dict[first_case].keys())
    first_rxn = rxn_ids[0]

    print(f"Found {len(case_ids)} cases, e.g. case={first_case!r}")
    print(f"Found {len(rxn_ids)} reactions in that case, e.g. rxn={first_rxn!r}")
    print(f"Sample value for (case={first_case!r}, rxn={first_rxn!r}): "
          f"{sens_dict[first_case][first_rxn]!r}")

    results = run_pipeline(PICKLE_PATH, OUTPUT_DIR, threshold=0.05)
    #check = pd.read_csv("./sa_pipeline_v2_out/selection_with_rescue_A.csv")
    #r12 = check[check["reaction"] == "R12:rxn_name_12"]
    #print("\nR12 rows where it was rescued (designed to be correlated w/ R0, "
    #      "small on its own):")
    #print(r12[r12["rescued_by_correlation"]].head(3).to_string(index=False))
    #assert r12["rescued_by_correlation"].sum() > 0, "expected R12 to get rescued at least once"
    #print("\nSanity check passed.")
