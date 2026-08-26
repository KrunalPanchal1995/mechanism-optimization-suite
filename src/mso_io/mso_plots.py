"""
mso_plots.py — All Matplotlib Figures
=======================================
Generates:
  S2  cost_comparison         — per-case bar chart + sparsity + total bar
  S4  convergence             — best objective vs iteration  (optional)
  S5  radar_chart             — multi-condition spider plots per stage
  S1  sensitivity_heatmap     — |S_i| heatmap (optional)
      mre_distribution        — box/violin + table of MRE stats

Every plot saves its underlying data as a CSV for reproducibility.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

# ── Style defaults ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       10,
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'legend.fontsize': 9,
    'figure.dpi':      150,
    'lines.linewidth': 1.8,
})

# Colour palette (colour-blind friendly, no reliance on red/green)
_C = {
    'primary':   '#2c5282',   # deep blue
    'secondary': '#744210',   # amber
    'tertiary':  '#276749',   # teal-green
    'neutral':   '#4a5568',   # dark gray
    'light':     '#a0aec0',   # light gray
}


def _save_csv(df: pd.DataFrame, data_dir: str, stem: str):
    path = Path(data_dir) / f"{stem}.csv"
    df.to_csv(path, index=False)
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — Cost comparison (3-panel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cost_comparison(all_data: dict, analysis: dict,
                         cfg: dict, plots_dir: str, data_dir: str) -> str:
    """
    Three panels:
      (a) Per-case cost bars (p-PRS per stage), Full-PRS as dotted line
      (b) Active reactions per case (sparsity) for each stage
      (c) Total cost comparison bar
    """
    primary = cfg['primary_threshold']
    labels  = cfg['stage_labels']

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    all_csv_rows = []

    # ── (a) Per-case cost bars ─────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, :])
    colors = [_C['primary'], _C['secondary'], _C['tertiary']]
    offset_step = 0.35
    max_cases = 0

    for idx, lbl in enumerate(labels):
        cost_dict = analysis['stages'][lbl]['cost'].get(primary)
        if cost_dict is None:
            continue
        df = all_data[lbl]['costs'].get(primary)
        if df is None:
            continue

        n    = len(df)
        x    = np.arange(n)
        off  = (idx - (len(labels)-1)/2) * offset_step
        cost_arr = df['p-PRS_Cost'].values
        full_val  = int(df['Full_PRS_Cost'].iloc[0])
        max_cases = max(max_cases, n)

        ax1.bar(x + off, cost_arr, offset_step*0.85,
                label=f'p-PRS {lbl}', color=colors[idx % len(colors)],
                alpha=0.82, zorder=3)
        ax1.axhline(full_val, color=colors[idx % len(colors)],
                    ls=':', lw=2.0, alpha=0.85,
                    label=f'Full-PRS {lbl} ({full_val:,}/case)')

        for j, (ci, rv) in enumerate(zip(x, df['Active_Rxns'].values)):
            all_csv_rows.append({'stage': lbl, 'case': j,
                                 'pprs_cost': cost_arr[j],
                                 'full_cost': full_val,
                                 'active_rxns': rv})

    ax1.set_xlabel('Optimization Case Index')
    ax1.set_ylabel('Simulations per Case')
    ax1.set_title('Per-Case Simulation Cost: p-PRS vs. Full-PRS Reference',
                  fontweight='bold')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f'{int(v):,}'))
    ax1.legend(ncol=2)
    ax1.grid(axis='y', ls='--', alpha=0.4, zorder=0)
    ax1.annotate('(a)', xy=(0.01,0.96), xycoords='axes fraction',
                 fontsize=12, fontweight='bold')

    # ── (b) Sparsity (active reactions) ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, lbl in enumerate(labels):
        df = all_data[lbl]['costs'].get(primary)
        if df is None:
            continue
        n  = len(df)
        x  = np.arange(1, n+1)
        rxn = df['Active_Rxns'].values
        total = int(df['Total_Rxns'].iloc[0])
        ax2.plot(x, rxn, color=colors[idx % len(colors)],
                 lw=1.6, marker='o', ms=3.5,
                 label=f'{lbl}  ({rxn.min()}–{rxn.max()} / {total})')
        ax2.axhline(total, color=colors[idx % len(colors)],
                    ls=':', lw=1.4, alpha=0.6)

    ax2.set_xlabel('Optimization Case Index')
    ax2.set_ylabel('Active Reactions')
    ax2.set_title('Reaction Sparsity per Case', fontweight='bold')
    ax2.legend()
    ax2.grid(ls='--', alpha=0.40)
    ax2.annotate('(b)', xy=(0.02,0.96), xycoords='axes fraction',
                 fontsize=12, fontweight='bold')

    # ── (c) Total cost comparison ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    bar_labels, totals, c_list = [], [], []
    for idx, lbl in enumerate(labels):
        ca = analysis['stages'][lbl]['cost'].get(primary)
        if ca is None:
            continue
        bar_labels.extend([f'p-PRS\n({lbl})', f'Full-PRS\n({lbl})'])
        totals.extend([ca['pprs_total'], ca['full_total']])
        c_list.extend([colors[idx % len(colors)],
                       matplotlib.colors.to_rgba(
                           colors[idx % len(colors)], alpha=0.40)])

    cc = analysis['combined']
    bar_labels.append('p-PRS\n(Combined)')
    totals.append(cc['pprs_total'])
    c_list.append(_C['neutral'])

    x3 = np.arange(len(bar_labels))
    bars = ax3.bar(x3, totals, color=c_list, width=0.6, zorder=3)
    for bar, val in zip(bars, totals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(totals)*0.01,
                 f'{val:,}', ha='center', va='bottom', fontsize=8.5)

    ax3.set_ylabel('Total Simulations')
    ax3.set_title('Total Optimization Cost', fontweight='bold')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f'{int(v):,}'))
    ax3.set_xticks(x3); ax3.set_xticklabels(bar_labels, fontsize=8.5)
    ax3.grid(axis='y', ls='--', alpha=0.4, zorder=0)
    ax3.annotate('(c)', xy=(0.02,0.96), xycoords='axes fraction',
                 fontsize=12, fontweight='bold')

    fuel = cfg.get('_metadata', {}).get('fuel_name', '')
    fig.suptitle(f'Optimization Cost Analysis{" — " + fuel if fuel else ""}',
                 fontsize=13, fontweight='bold', y=1.01)

    out = str(Path(plots_dir) / 'cost_comparison.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

    if all_csv_rows:
        _save_csv(pd.DataFrame(all_csv_rows), data_dir, 'cost_comparison_data')

    return out


# ══════════════════════════════════════════════════════════════════════════════
# S5 — Radar / Spider chart per stage
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(errors: dict, stage_labels: list,
                     stage_label: str, stage_name: str,
                     plots_dir: str, data_dir: str) -> str:
    """
    Spider chart: spokes = conditions (φ, P); lines = each optimization stage.
    Values are plotted as ε (raw) on a log scale after normalisation to nominal.
    """
    if not errors:
        return None

    conds = sorted(errors.keys(), key=lambda x: (x[0], x[1]))
    N     = len(conds)
    if N < 3:
        return None

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    colors = [_C['neutral'], _C['primary'], _C['secondary'], _C['tertiary']]
    csv_rows = []

    for ci, lbl in enumerate(stage_labels):
        vals = []
        for cond in conds:
            v = errors[cond].get(lbl, np.nan)
            # Normalise by nominal (first stage label) so all are relative
            nom = errors[cond].get(stage_labels[0], 1.0)
            vals.append((v / nom) if nom > 0 else 1.0)
            csv_rows.append({'stage': lbl, 'phi': cond[0], 'P': cond[1],
                             'eps': errors[cond].get(lbl),
                             'eps_norm': vals[-1]})
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[ci % len(colors)],
                lw=2.0, ls='-' if ci > 0 else '--',
                label=lbl, zorder=3)
        ax.fill(angles, vals, alpha=0.06, color=colors[ci % len(colors)])

    # Labels on spokes
    spoke_lbls = [f'φ={c[0]}\nP={c[1]}atm' for c in conds]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spoke_lbls, fontsize=9)
    ax.set_yticklabels([])
    ax.axhline(1.0, color=_C['neutral'], ls=':', lw=1.2, alpha=0.5)

    ax.set_title(f'Multi-Condition Profile — {stage_label} ({stage_name})\n'
                 r'(Normalised by nominal $\varepsilon$)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))

    out = str(Path(plots_dir) / f'radar_{stage_label}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

    _save_csv(pd.DataFrame(csv_rows), data_dir, f'radar_data_{stage_label}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# S1 — Sensitivity heatmap (optional)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_heatmap(sens_df: pd.DataFrame, thresholds: list,
                              stage_label: str,
                              plots_dir: str, data_dir: str) -> str:
    """
    Heatmap of |S_i| for all reactions × conditions.
    Assumes wide format: row = reaction, columns = conditions.
    First column is taken as Reaction_ID.
    """
    if sens_df is None or sens_df.empty:
        return None

    id_col   = sens_df.columns[0]
    val_cols = [c for c in sens_df.columns if c != id_col]
    if not val_cols:
        return None

    mat     = sens_df[val_cols].values.astype(float)
    rxn_ids = sens_df[id_col].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(val_cols)*0.7),
                                    max(5, len(rxn_ids)*0.25)))

    im = ax.imshow(mat, aspect='auto', cmap='YlOrBr',
                   origin='upper', vmin=0)
    plt.colorbar(im, ax=ax, label=r'$|S_i|$ (sensitivity coefficient)')

    # Threshold lines (horizontal — across reaction axis)
    for thresh in thresholds:
        # find reactions at or above thresh
        above = np.where(np.max(mat, axis=1) >= thresh)[0]
        if len(above):
            ax.axhline(above[-1] + 0.5, color='red', lw=1.5, ls='--',
                       label=f'δ = {thresh}')

    ax.set_xticks(range(len(val_cols)))
    ax.set_xticklabels(val_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(rxn_ids)))
    ax.set_yticklabels(rxn_ids, fontsize=7)
    ax.set_xlabel('Condition (φ, P)')
    ax.set_ylabel('Reaction')
    ax.set_title(f'Sensitivity Coefficient Heatmap — {stage_label}',
                 fontweight='bold')
    if thresholds:
        ax.legend(fontsize=8, loc='upper right')

    out = str(Path(plots_dir) / f'sensitivity_heatmap_{stage_label}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

    # Save raw heatmap data
    save_df = sens_df.copy()
    _save_csv(save_df, data_dir, f'sensitivity_heatmap_{stage_label}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MRE distribution (PRS statistics)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mre_distribution(prs_df: pd.DataFrame, mre_stats: dict,
                           stage_label: str,
                           plots_dir: str, data_dir: str) -> str:
    """
    Two-panel figure:
      Left  — Violin + box plot for Training and Testing MRE
      Right — Scatter: Testing MRE vs case index, coloured by threshold pass/fail
    """
    tr  = mre_stats['train_raw']
    te  = mre_stats['test_raw']
    thr = mre_stats.get('threshold')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: violin + box ────────────────────────────────────────
    parts = ax1.violinplot([tr, te], positions=[1, 2],
                            showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(_C['light'])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color(_C['primary'])

    ax1.boxplot([tr, te], positions=[1, 2], widths=0.12,
                medianprops=dict(color=_C['primary'], lw=2),
                whiskerprops=dict(lw=1.2),
                flierprops=dict(marker='x', ms=4, alpha=0.6))

    if thr is not None:
        ax1.axhline(thr, color=_C['secondary'], ls='--', lw=1.6,
                    label=f'Acceptance threshold ({thr}%)')
        ax1.legend()

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Training MRE', 'Testing MRE'])
    ax1.set_ylabel('Maximum Residual Error (%)')
    ax1.set_title(f'MRE Distribution — {stage_label}', fontweight='bold')
    ax1.grid(axis='y', ls='--', alpha=0.4)

    # ── Right: Testing MRE vs case index ─────────────────────────
    n   = len(te)
    idx = np.arange(1, n+1)
    col = [_C['primary'] if v < (thr or np.inf) else _C['secondary']
           for v in te]
    ax2.scatter(idx, te, c=col, s=20, zorder=3, alpha=0.8)
    ax2.plot(idx, te, color=_C['light'], lw=0.8, zorder=2)
    if thr is not None:
        ax2.axhline(thr, color=_C['secondary'], ls='--', lw=1.6,
                    label=f'Threshold = {thr}%')
        ax2.legend()

    ax2.set_xlabel('Optimization Case Index')
    ax2.set_ylabel('Testing MRE (%)')
    ax2.set_title(f'Testing MRE per Case — {stage_label}', fontweight='bold')
    ax2.grid(ls='--', alpha=0.4)

    out = str(Path(plots_dir) / f'mre_distribution_{stage_label}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

    # Save data
    df_csv = pd.DataFrame({'case': np.arange(1, len(tr)+1),
                           'Training_MRE': tr, 'Testing_MRE': te})
    _save_csv(df_csv, data_dir, f'mre_distribution_{stage_label}')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# S4 — Convergence plot (optional)
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(all_data: dict, cfg: dict,
                     plots_dir: str, data_dir: str) -> str:
    """
    Plot best objective function value vs iteration for all available stages.
    Convergence CSV must have columns: Iteration, Best_Objective (and Stage).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [_C['primary'], _C['secondary'], _C['tertiary']]
    found   = False
    csv_out = []

    for idx, lbl in enumerate(cfg['stage_labels']):
        conv_df = all_data.get(lbl, {}).get('convergence')
        if conv_df is None:
            continue
        found = True
        # Detect columns
        iter_col = next((c for c in conv_df.columns
                         if 'iter' in c.lower()), conv_df.columns[0])
        obj_col  = next((c for c in conv_df.columns
                         if 'obj' in c.lower() or 'best' in c.lower()),
                        conv_df.columns[1])

        ax.plot(conv_df[iter_col], conv_df[obj_col],
                color=colors[idx % len(colors)], lw=2.0,
                label=f'Stage {idx+1} — {lbl}')
        for _, row in conv_df.iterrows():
            csv_out.append({'stage': lbl, 'iteration': row[iter_col],
                            'best_objective': row[obj_col]})

    if not found:
        plt.close()
        return None

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Best Objective $\varepsilon$')
    ax.set_title('Optimizer Convergence History', fontweight='bold')
    ax.legend()
    ax.grid(ls='--', alpha=0.4)

    out = str(Path(plots_dir) / 'convergence.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    if csv_out:
        _save_csv(pd.DataFrame(csv_out), data_dir, 'convergence_data')
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(all_data: dict, analysis: dict,
                       cfg: dict, metadata: dict) -> dict:
    """
    Generate all applicable plots.  Returns dict {plot_name: file_path}.
    """
    cfg['_metadata'] = metadata   # convenience for titles

    pd_dir   = cfg['data_dir']
    pl_dir   = cfg['plots_dir']
    primary  = cfg['primary_threshold']
    paths    = {}

    # S2 — cost comparison
    try:
        p = plot_cost_comparison(all_data, analysis, cfg, pl_dir, pd_dir)
        paths['cost_comparison'] = p
        print(f"  ✓ cost_comparison.pdf")
    except Exception as e:
        print(f"  ✗ cost_comparison: {e}")

    # S4 — convergence (optional)
    try:
        p = plot_convergence(all_data, cfg, pl_dir, pd_dir)
        if p:
            paths['convergence'] = p
            print(f"  ✓ convergence.pdf")
    except Exception as e:
        print(f"  ✗ convergence: {e}")

    for lbl in cfg['stage_labels']:
        sd = all_data.get(lbl, {})

        # S5 — radar chart
        errors = sd.get('errors')
        if errors:
            stage_lbls = analysis['stages'][lbl].get('stage_labels', [])
            stage_name = cfg['stage_names'][cfg['stage_labels'].index(lbl)]
            try:
                p = plot_radar_chart(errors, stage_lbls, lbl, stage_name,
                                     pl_dir, pd_dir)
                if p:
                    paths[f'radar_{lbl}'] = p
                    print(f"  ✓ radar_{lbl}.pdf")
            except Exception as e:
                print(f"  ✗ radar_{lbl}: {e}")

        # S1 — sensitivity heatmap (optional)
        sens = sd.get('sensitivity')
        if sens is not None:
            try:
                p = plot_sensitivity_heatmap(sens, cfg['thresholds'],
                                             lbl, pl_dir, pd_dir)
                if p:
                    paths[f'sensitivity_{lbl}'] = p
                    print(f"  ✓ sensitivity_heatmap_{lbl}.pdf")
            except Exception as e:
                print(f"  ✗ sensitivity_heatmap_{lbl}: {e}")

        # MRE distribution
        mre_stats = analysis['stages'][lbl].get('prs_stats')
        prs_df    = sd.get('prs_stats')
        if mre_stats is not None and prs_df is not None:
            try:
                p = plot_mre_distribution(prs_df, mre_stats, lbl,
                                          pl_dir, pd_dir)
                if p:
                    paths[f'mre_{lbl}'] = p
                    print(f"  ✓ mre_distribution_{lbl}.pdf")
            except Exception as e:
                print(f"  ✗ mre_distribution_{lbl}: {e}")

    return paths
