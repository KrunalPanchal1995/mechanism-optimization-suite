"""
mso_analysis.py — All Computational Analysis
==============================================
Computes:
  • n_c and per-case cost from active reaction count
  • Cost savings (p-PRS vs Full-PRS staged vs Conventional)
  • Δε per condition and aggregate improvement sums
  • MRE statistics summary (mean, std, max, min)
"""

import numpy as np
import pandas as pd
from itertools import product


# ══════════════════════════════════════════════════════════════════════════════
# PRS coefficient / cost formulas
# ══════════════════════════════════════════════════════════════════════════════

def nc(k: int) -> int:
    """Number of second-order PRS coefficients for k active reactions.
       n_c = 1 + k + k(k+1)/2   (intercept + linear + quadratic/cross terms)
    """
    return 1 + k + k * (k + 1) // 2


def cost_per_case(k: int) -> int:
    """Forward model evaluations required: 4 * n_c."""
    return 4 * nc(k)


# ══════════════════════════════════════════════════════════════════════════════
# Cost analysis for one stage
# ══════════════════════════════════════════════════════════════════════════════

def compute_cost_analysis(cost_df: pd.DataFrame) -> dict:
    """
    Summarise cost data from a cost CSV DataFrame.

    Returns a dict with:
        n_cases, k_total, active_min, active_max,
        pprs_total, full_total, saving_pct,
        cost_per_case_full, nc_full
    """
    n    = len(cost_df)
    k    = int(cost_df['Total_Rxns'].iloc[0])
    amin = int(cost_df['Active_Rxns'].min())
    amax = int(cost_df['Active_Rxns'].max())
    pprs = int(cost_df['p-PRS_Cost'].sum())
    full = int(cost_df['Full_PRS_Cost'].sum())
    cpf  = int(cost_df['Full_PRS_Cost'].iloc[0])   # per-case full cost

    saving = 100.0 * (full - pprs) / full if full > 0 else 0.0

    return dict(
        n_cases          = n,
        k_total          = k,
        active_min       = amin,
        active_max       = amax,
        pprs_total       = pprs,
        full_total       = full,
        saving_pct       = saving,
        cost_per_case_full = cpf,
        nc_full          = nc(k),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Improvement analysis for one stage
# ══════════════════════════════════════════════════════════════════════════════

def delta_eps(nom: float, opt: float) -> float:
    """
    Δε (%) = 100 × (ε_nom − ε_opt) / ε_nom
    Positive  → improvement (objective decreased)
    Negative  → degradation (objective increased)
    """
    if nom == 0:
        return 0.0
    return 100.0 * (nom - opt) / nom


def compute_improvement(errors: dict, stage_labels_in_csv: list) -> dict:
    """
    Compute Δε% for each (φ, P) condition and each optimized stage.

    Parameters
    ----------
    errors : dict  {(phi, P): {stage_label: eps_value}}
    stage_labels_in_csv : list of stage label strings as they appear in the
                          CSV (e.g. ['Nominal','Stage-1','Stage-2'])

    Returns
    -------
    dict {(phi, P): {'eps': {label: value},
                     'delta': {opt_label: delta_pct},
                     'improved': {opt_label: bool}}}
    """
    nominal_label = stage_labels_in_csv[0]
    opt_labels    = stage_labels_in_csv[1:]

    result = {}
    for cond, stage_data in errors.items():
        nom = stage_data.get(nominal_label)
        if nom is None:
            continue
        entry = {'eps': stage_data, 'delta': {}, 'improved': {}}
        for lbl in opt_labels:
            opt = stage_data.get(lbl)
            if opt is None:
                continue
            d = delta_eps(nom, opt)
            entry['delta'][lbl]    = d
            entry['improved'][lbl] = (opt < nom)
        result[cond] = entry

    return result


def compute_aggregate(errors: dict, stage_labels_in_csv: list) -> dict:
    """
    Sum ε across all conditions for each stage.

    Returns
    -------
    dict {stage_label: {'sum': float, 'delta_pct': float, 'factor': float}}
    """
    nominal_label = stage_labels_in_csv[0]
    agg = {lbl: 0.0 for lbl in stage_labels_in_csv}

    for cond_data in errors.values():
        for lbl in stage_labels_in_csv:
            v = cond_data.get(lbl)
            if v is not None:
                agg[lbl] += v

    nom_sum = agg[nominal_label]
    result  = {nominal_label: {'sum': nom_sum, 'delta_pct': 0.0, 'factor': 1.0}}

    for lbl in stage_labels_in_csv[1:]:
        s = agg[lbl]
        d = 100.0 * (nom_sum - s) / nom_sum if nom_sum else 0.0
        f = nom_sum / s if s > 0 else float('inf')
        result[lbl] = {'sum': s, 'delta_pct': d, 'factor': f}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PRS MRE statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_mre_stats(prs_df: pd.DataFrame) -> dict:
    """
    Descriptive statistics of Training_MRE and Testing_MRE.

    Returns dict with keys: train_{mean,std,max,min}, test_{mean,std,max,min},
    also threshold (if column exists), and the raw arrays for plotting.
    """
    tr  = prs_df['Training_MRE'].dropna().values.astype(float)
    te  = prs_df['Testing_MRE'].dropna().values.astype(float)
    thr = prs_df['Threshold'].iloc[0] if 'Threshold' in prs_df.columns else None

    stats = dict(
        train_mean  = float(np.mean(tr)),
        train_std   = float(np.std(tr)),
        train_max   = float(np.max(tr)),
        train_min   = float(np.min(tr)),
        train_median= float(np.median(tr)),
        test_mean   = float(np.mean(te)),
        test_std    = float(np.std(te)),
        test_max    = float(np.max(te)),
        test_min    = float(np.min(te)),
        test_median = float(np.median(te)),
        threshold   = thr,
        train_raw   = tr,
        test_raw    = te,
        n_cases     = len(tr),
    )
    # Fraction of cases where Testing_MRE < Threshold
    if thr is not None:
        stats['frac_below_threshold'] = float(np.mean(te < thr))
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Conventional Full-PRS cost (no reaction classification)
# ══════════════════════════════════════════════════════════════════════════════

def conventional_cost(k_union: int, n_cases_total: int) -> dict:
    """
    Cost for conventional Full-PRS: single PRS on union of all reactions.

    Parameters
    ----------
    k_union       : total unique reactions in HTC ∪ LTC
    n_cases_total : HTC cases + LTC cases

    Returns dict with nc, cost_per_case, total_cost.
    """
    nc_val = nc(k_union)
    cpc    = 4 * nc_val
    total  = cpc * n_cases_total
    return dict(k_union=k_union, nc=nc_val, cost_per_case=cpc, total=total)


# ══════════════════════════════════════════════════════════════════════════════
# Infer stage labels from error dict (which labels appear most consistently?)
# ══════════════════════════════════════════════════════════════════════════════

def infer_stage_labels(errors: dict) -> list:
    """
    Return the ordered list of stage labels common to all conditions.
    The order is the insertion order of the first condition encountered.
    """
    if not errors:
        return []
    # Use the first condition's key order as canonical
    first_cond = next(iter(errors.values()))
    labels = list(first_cond.keys())
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Master analysis orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(all_data: dict, cfg: dict, metadata: dict) -> dict:
    """
    Run all computations and return a nested results dict.

    Structure of returned dict:
    {
      'stages': {
          'HTC': {
              'cost':        {thresh: cost_analysis_dict},
              'improvement': {thresh: improvement_dict},
              'aggregate':   {thresh: aggregate_dict},
              'prs_stats':   mre_stats_dict (or None),
              'stage_labels': [list of labels from CSV],
          },
          'LTC': { ... }
      },
      'combined': {
          'pprs_total': int,
          'full_total': int,
          'combined_saving_pct': float,
          'conventional': {k_union: conventional_dict, ...},
      },
      'thresholds':          list,
      'primary_threshold':   float,
    }
    """
    out = {
        'stages':            {},
        'combined':          {},
        'thresholds':        cfg['thresholds'],
        'primary_threshold': cfg['primary_threshold'],
        'stage_labels_cfg':  cfg['stage_labels'],
        'stage_names_cfg':   cfg['stage_names'],
    }

    combined_pprs = 0
    combined_full = 0
    total_cases   = 0

    for label in cfg['stage_labels']:
        sd    = all_data.get(label, {})
        entry = {'cost': {}, 'improvement': {}, 'aggregate': {},
                 'prs_stats': None, 'stage_labels': []}

        # Stage labels from error CSV
        errors_primary = sd.get('errors', {})
        if errors_primary:
            entry['stage_labels'] = infer_stage_labels(errors_primary)

        # Cost per threshold
        for thresh in cfg['thresholds']:
            cost_df = sd.get('costs', {}).get(thresh)
            if cost_df is not None:
                ca = compute_cost_analysis(cost_df)
                entry['cost'][thresh] = ca
                if thresh == cfg['primary_threshold']:
                    combined_pprs += ca['pprs_total']
                    combined_full += ca['full_total']
                    total_cases   += ca['n_cases']

        # Improvement: we use the primary threshold's error data
        # (all thresholds share the same Nominal; optimized differs per thresh)
        if errors_primary and entry['stage_labels']:
            stage_lbls = entry['stage_labels']
            entry['improvement'][cfg['primary_threshold']] = compute_improvement(
                errors_primary, stage_lbls)
            entry['aggregate'][cfg['primary_threshold']] = compute_aggregate(
                errors_primary, stage_lbls)

        # Threshold comparison (if requested)
        # NOTE: for threshold comparison the error_folder must contain files
        # with columns for each threshold. If the user provides separate folders
        # per threshold, this block can be extended. Currently uses primary folder.

        # PRS stats
        prs_df = sd.get('prs_stats')
        if prs_df is not None:
            entry['prs_stats'] = compute_mre_stats(prs_df)

        out['stages'][label] = entry

    # Combined cost summary
    out['combined'] = dict(
        pprs_total          = combined_pprs,
        full_total          = combined_full,
        total_cases         = total_cases,
        combined_saving_pct = (100.0 * (combined_full - combined_pprs) / combined_full
                               if combined_full > 0 else 0.0),
    )

    # Conventional Full-PRS for range of k_union values
    # k_total for each stage (from primary threshold cost data)
    k_per_stage = []
    for label in cfg['stage_labels']:
        ca_dict = out['stages'][label]['cost'].get(cfg['primary_threshold'])
        if ca_dict:
            k_per_stage.append(ca_dict['k_total'])

    if len(k_per_stage) >= 2:
        k_min = max(k_per_stage)           # full overlap → union = largest set
        k_max = sum(k_per_stage)           # no overlap   → union = sum
        conv_results = {}
        for k_u in sorted(set([k_min, k_max] +
                               [k for k in range(k_min, k_max+1,
                                                  max(1,(k_max-k_min)//3))])):
            cconv = conventional_cost(k_u, total_cases)
            cconv['saving_vs_pprs'] = (
                100.0*(cconv['total'] - combined_pprs)/cconv['total']
                if cconv['total'] > 0 else 0.0)
            cconv['saving_vs_full'] = (
                100.0*(cconv['total'] - combined_full)/cconv['total']
                if cconv['total'] > 0 else 0.0)
            cconv['factor_vs_pprs'] = (
                cconv['total'] / combined_pprs if combined_pprs > 0 else None)
            conv_results[k_u] = cconv
        out['combined']['conventional'] = conv_results
        out['combined']['k_union_min']  = k_min
        out['combined']['k_union_max']  = k_max

    return out
