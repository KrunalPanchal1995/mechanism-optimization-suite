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

def correlation_matrix_with_pvalues(wide_df, min_pairs=8):
    """Pearson r between every pair of reaction columns, using only the
    cases both reactions have a value for. Pairs sharing fewer than
    min_pairs cases are left as NaN -- flagged as unreliable, not guessed."""
    cols = wide_df.columns
    N = len(cols)
    X = wide_df.values
    rho = np.full((N, N), np.nan)
    pval = np.full((N, N), np.nan)
    npairs = np.zeros((N, N), dtype=int)
    valid = ~np.isnan(X)

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
            r = float(np.corrcoef(xi, xj)[0, 1])
            df = n - 2
            t_stat = r * np.sqrt(df / max(1e-12, 1 - r**2))
            rho[i, j] = rho[j, i] = r
            pval[i, j] = pval[j, i] = 2 * stats.t.sf(abs(t_stat), df=df)

    return (pd.DataFrame(rho, index=cols, columns=cols),
            pd.DataFrame(pval, index=cols, columns=cols),
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

def plot_correlation_heatmap(rho_df, title, outpath, top_n=40, importance=None, extra_formats=()):
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
    # Round 4 fix (WORKBENCH_AUDIT_2026-08-10.md): VisualAid.postprocess()'s
    # --formats flag used to be silently ignored for this pipeline's own
    # heatmap output -- run_pipeline() had no formats parameter at all, so
    # `--formats png,pdf` produced only a .png, contradicting the shared
    # postprocess CLI contract's documented behavior. Save the SAME already-
    # rendered figure under any additional requested extensions instead of
    # re-plotting per format.
    for fmt in extra_formats:
        fig.savefig(f"{os.path.splitext(outpath)[0]}.{fmt}", dpi=150)
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
                  final_set_corr_cutoff=0.9, formats=("png",)):
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
            extra_formats=[f for f in formats if f != "png"],
        )

        sel_table = build_selection_table(wide, rho, threshold, corr_cutoff)
        sel_table["S_value"] = sel_table["S_value"].round(6)
        sel_table["abs_S"] = sel_table["abs_S"].round(6)
        sel_table.to_csv(os.path.join(output_dir, f"selection_with_rescue_{p}.csv"), index=False)
        corr_csv_path = os.path.join(output_dir, f"correlation_matrix_{p}.csv")
        rho.round(4).to_csv(corr_csv_path)

        final_active_set_per_target(
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
    return wide_frames



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
