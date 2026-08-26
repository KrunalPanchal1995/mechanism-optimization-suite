"""
mso_latex.py — LaTeX Document Generator
=========================================
Produces a self-contained .tex file with:
  • Methodology section (variables, formulas, definitions)
  • Cost analysis table (p-PRS / Full-PRS staged / Conventional)
  • Per-stage error tables (Nominal → Stage-1 → Stage-2)
  • Threshold comparison table (if multiple δ)
  • Aggregate improvement table
  • PRS statistics table (MRE summary)
  • Auto-generated summary paragraphs
  • Figures via \\includegraphics

No colour commands: bold = improvement, italic = degradation.
"""

from __future__ import annotations
import math
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, decimals=2) -> str:
    """Format a float; use comma-thousands for large numbers."""
    if v is None:
        return '--'
    if abs(v) >= 1000:
        return f'{v:,.{decimals}f}'
    return f'{v:.{decimals}f}'


def _cell(value_str: str, improved: bool | None) -> str:
    """
    Wrap in \\textbf if improved, \\textit if degraded, plain otherwise.
    improved=None → nominal (plain).
    """
    if improved is True:
        return r'\textbf{' + value_str + '}'
    if improved is False:
        return r'\textit{' + value_str + '}'
    return value_str


def _delta_cell(d: float) -> str:
    """Format Δε(%) as $+X.X\\%$ or $-X.X\\%$, bold if positive."""
    sign = '+' if d >= 0 else ''
    s = f'${sign}{d:.1f}\\%$'
    return r'\textbf{' + s + '}' if d > 0 else r'\textit{' + s + '}'


def _inc(path: str, width: str = r'\linewidth') -> str:
    return (f'\\begin{{figure}}[!ht]\\centering\n'
            f'\\includegraphics[width={width}]{{{path}}}\n'
            f'\\end{{figure}}\n')


# ══════════════════════════════════════════════════════════════════════════════
# PREAMBLE
# ══════════════════════════════════════════════════════════════════════════════

def _preamble() -> str:
    return r"""\documentclass[12pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs,multirow,array,amsmath,amssymb}
\usepackage{caption,makecell,adjustbox,graphicx}
\usepackage{hyperref}
\usepackage{parskip}

\captionsetup{font=small,labelfont=bf}

\begin{document}
"""


# ══════════════════════════════════════════════════════════════════════════════
# TITLE / HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _header(metadata: dict, cfg: dict) -> str:
    fuel   = metadata.get('fuel_name', 'Unknown Fuel')
    form   = metadata.get('fuel_formula', '')
    mech   = metadata.get('mechanism_name', '')
    auth   = metadata.get('paper_authors', '')
    jour   = metadata.get('journal', '')
    yr     = metadata.get('year', '')
    note   = metadata.get('study_note', '')

    stages_str = ' $\\rightarrow$ '.join(
        [f'Stage-{i+1} ({lbl})' for i, lbl in
         enumerate(cfg['stage_labels'])])

    lines = [
        r'\begin{center}',
        r'{\LARGE\bfseries MSO Report}\\[6pt]',
        r'{\large p-PRS Multi-Stage Optimization}\\[4pt]',
        f'{{\\large Fuel: {fuel} ({form})}}\\\\[4pt]' if form else
        f'{{\\large Fuel: {fuel}}}\\\\[4pt]',
        f'{{Mechanism: {mech}}}\\\\[2pt]' if mech else '',
        f'{{Stages: {stages_str}}}\\\\[2pt]',
        f'{{Authors: {auth}, {jour} ({yr})}}\\\\[2pt]' if auth else '',
        r'\end{center}',
        r'\vspace{6pt}',
    ]
    if note:
        # expand {fuel_name} placeholder
        note_exp = note.replace('{fuel_name}', fuel)
        lines.append(r'\noindent ' + note_exp.strip())
    lines.append(r'\vspace{4pt}')
    return '\n'.join(l for l in lines if l) + '\n\n'


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _methodology(metadata: dict, cfg: dict, analysis: dict) -> str:
    primary = cfg['primary_threshold']
    thresholds_str = ', '.join(f'$\\delta = {t}$' for t in cfg['thresholds'])

    # Try to get k values from analysis
    k_values = {}
    for lbl in cfg['stage_labels']:
        ca = analysis['stages'][lbl]['cost'].get(primary, {})
        if ca:
            k_values[lbl] = ca.get('k_total', '?')

    k_str = '; '.join(f'$R_\\mathrm{{{lbl}}} = {k}$'
                      for lbl, k in k_values.items())

    return r"""
\section*{A.\enspace Methodology and Statistical Framework}

\subsection*{A.1\enspace Notation and Variable Definitions}

\begin{tabular}{@{}lp{10cm}@{}}
\toprule
\textbf{Symbol} & \textbf{Definition} \\
\midrule
$R$ & Total number of reactions in a given optimization stage \\
$k$ & Number of \emph{active} reactions selected per case ($k \le R$) \\
$\delta$ & Sensitivity threshold: reaction $i$ is active if $|S_i| \ge \delta$ \\
$n_c$ & Number of second-order PRS coefficients (Eq.~\ref{eq:nc}) \\
$n_s$ & Number of forward model evaluations (samples) required \\
$\hat{\varepsilon}$ & PRS-predicted objective function value \\
$\varepsilon$ & True simulator objective function value \\
$\Delta\varepsilon\,(\%)$ & Relative improvement in $\varepsilon$ (Eq.~\ref{eq:delta}) \\
$\text{MRE}$ & Maximum Residual Error of the PRS fit (Eq.~\ref{eq:mre}) \\
\bottomrule
\end{tabular}

\subsection*{A.2\enspace Partial Polynomial Response Surface (p-PRS)}

A second-order Polynomial Response Surface (PRS) is constructed for the
objective function $\varepsilon$ over the space of Arrhenius perturbation
parameters $\mathbf{x} \in \mathbb{R}^k$:
\begin{equation}
  \hat{\varepsilon}(\mathbf{x}) = a_0
    + \sum_{i=1}^{k} a_i\,x_i
    + \sum_{i=1}^{k}\sum_{j \ge i}^{k} a_{ij}\,x_i x_j
\end{equation}
The number of PRS coefficients for $k$ active reactions is:
\begin{equation}\label{eq:nc}
  n_c = 1 + k + \frac{k(k+1)}{2}
\end{equation}
The required sample size (forward model evaluations) is set to four times the
coefficient count to ensure a well-conditioned regression:
\begin{equation}
  n_s = 4\,n_c
\end{equation}
\textbf{Reaction screening.}
At each optimization step a normalized sensitivity coefficient
$S_i = |\partial\varepsilon/\partial x_i|$ is computed.
Reaction $i$ is included in the PRS (classified as \emph{active}) only if
\begin{equation}
  |S_i| \ge \delta
\end{equation}
where $\delta$ is the user-defined threshold (""" + thresholds_str + r""").
This reduces $k \le R$ and hence $n_c$, producing the \emph{partial} PRS (p-PRS)
with lower computational cost than the Full-PRS ($k = R$).

\subsection*{A.3\enspace Multi-Stage Optimization (MSO)}

The MSO pipeline decomposes the full optimization problem by combustion regime.
Each stage targets a distinct set of experimental targets and a distinct set of
sensitive reactions (""" + k_str + r"""):
\begin{enumerate}
  \item \textbf{Stage-1} (HTC) — High-temperature ignition delay times (IDT);
        kinetically governed by chain-branching and H-abstraction reactions.
  \item \textbf{Stage-2} (LTC) — Low-temperature IDT including the NTC region;
        governed by peroxy-radical isomerisation and hydroperoxide decomposition.
\end{enumerate}
Each stage uses the optimized mechanism from the previous stage as its
prior/starting point, enabling sequential improvement across combustion regimes.

\subsection*{A.4\enspace Evaluation Metrics}

\paragraph{Relative improvement in objective function.}
\begin{equation}\label{eq:delta}
  \Delta\varepsilon\,(\%)
    = 100\,\frac{\varepsilon_\mathrm{nom} - \varepsilon_\mathrm{opt}}
                {\varepsilon_\mathrm{nom}}
\end{equation}
Positive $\Delta\varepsilon$ indicates a decrease in objective function
(improvement); negative $\Delta\varepsilon$ indicates an increase (degradation).

\paragraph{Aggregate improvement.}
For a stage with $M$ conditions, the aggregate sum is
$\Sigma\varepsilon = \sum_{j=1}^{M}\varepsilon_j$, and the factor is
$F = \Sigma\varepsilon_\mathrm{nom}/\Sigma\varepsilon_\mathrm{opt}$
($F > 1$ indicates overall improvement).

\paragraph{Computational cost saving.}
\begin{equation}
  \text{CS}\,(\%) = 100\,\frac{C_\mathrm{ref} - C_\mathrm{method}}{C_\mathrm{ref}}
\end{equation}
where $C$ denotes the total number of forward model evaluations.

\paragraph{Maximum Residual Error (MRE).}
The PRS fit accuracy for each optimization case is quantified by:
\begin{equation}\label{eq:mre}
  \text{MRE}\,(\%)
    = 100\,\max_j \frac{|\hat{\varepsilon}_j - \varepsilon_j|}{|\varepsilon_j|}
\end{equation}
A PRS is accepted if $\text{MRE} < \delta_\mathrm{acc}$ where
$\delta_\mathrm{acc}$ is a pre-set accuracy threshold.

"""


# ══════════════════════════════════════════════════════════════════════════════
# COST TABLE  (merged: p-PRS / Full-PRS staged / Conventional)
# ══════════════════════════════════════════════════════════════════════════════

def _cost_table(analysis: dict, cfg: dict) -> str:
    primary = cfg['primary_threshold']
    labels  = cfg['stage_labels']
    cc      = analysis['combined']

    rows_pprs, rows_full = [], []
    for lbl in labels:
        ca = analysis['stages'][lbl]['cost'].get(primary)
        if ca is None:
            continue
        rows_pprs.append((lbl, ca['n_cases'], ca['k_total'],
                          f"{ca['active_min']}--{ca['active_max']}",
                          ca['pprs_total'], ca['saving_pct']))
        rows_full.append((lbl, ca['n_cases'], ca['k_total'],
                          'all', ca['full_total'], None))

    # Combined rows
    pprs_comb = cc['pprs_total']
    full_comb = cc['full_total']
    sav_pprs  = cc['combined_saving_pct']
    sav_full  = (100.0*(list(cc.get('conventional',{}).values())[0]['total']
                        - full_comb)
                 / list(cc.get('conventional',{}).values())[0]['total']
                 if cc.get('conventional') else None)

    # Conventional rows
    conv_rows = []
    if 'conventional' in cc:
        k_min = cc.get('k_union_min')
        k_max = cc.get('k_union_max')
        for k_u, cv in cc['conventional'].items():
            if k_u in (k_min, k_max):
                label_overlap = ('full overlap' if k_u == k_min
                                 else 'no overlap')
                conv_rows.append(
                    (k_u, label_overlap, cv['cost_per_case'],
                     cv['total'], cv['saving_vs_pprs'],
                     cv['saving_vs_full'], cv['factor_vs_pprs']))

    # ── Build LaTeX ────────────────────────────────────────────────
    def _row(cells):
        return '  ' + ' & '.join(str(c) for c in cells) + r' \\' + '\n'

    body = ''
    # p-PRS block
    body += r'  \multirow{' + str(len(rows_pprs)+1) + r'}{*}{\makecell[l]{p-PRS\\($\delta=' + str(primary) + r'$)}}' + '\n'
    for (lbl, n, k, act, cost, sav) in rows_pprs:
        sv = f'\\textbf{{{sav:.1f}}}' if sav is not None else '--'
        body += f'  & {lbl} & {n} & ${k}\\,({act})$ & {cost:,} & {sv} & -- & -- \\\\\n'
    body += (f'  & \\textit{{Combined}} & {cc["total_cases"]} & -- & '
             f'\\textbf{{{pprs_comb:,}}} & \\textbf{{{sav_pprs:.1f}}} & '
             f'-- & -- \\\\\n')
    body += r'  \midrule' + '\n'

    # Full-PRS staged block
    body += r'  \multirow{' + str(len(rows_full)+1) + r'}{*}{\makecell[l]{Full-PRS\\(staged)}}' + '\n'
    for (lbl, n, k, act, cost, _) in rows_full:
        body += f'  & {lbl} & {n} & ${k}\\,({act})$ & {cost:,} & -- & -- & -- \\\\\n'
    sav_full_str = f'\\textbf{{{sav_full:.1f}}}' if sav_full else '--'
    body += (f'  & \\textit{{Combined}} & {cc["total_cases"]} & -- & '
             f'\\textbf{{{full_comb:,}}} & -- & {sav_full_str} & -- \\\\\n')
    body += r'  \midrule' + '\n'

    # Conventional block
    for (k_u, overlap, cpc, tot, sv_p, sv_f, fac) in conv_rows:
        fac_str = f'${fac:.1f}\\times$' if fac else '--'
        body += (f'  Full-PRS (conv., {overlap}) & All & '
                 f'{cc["total_cases"]} & ${k_u}\\,(all)$ & '
                 f'{tot:,} & -- & -- & {fac_str} \\\\\n')

    tex = r"""
\section*{B.\enspace Simulation Cost Analysis}

\begin{table}[!ht]
\centering
\small
\caption{Simulation cost comparison across three strategies.
$R$: total reactions available; active range: min--max per case (p-PRS only).
Savings vs.\ Full-PRS (staged): relative to Full-PRS with same stage structure.
Savings vs.\ Conventional: relative to single-stage Full-PRS on $R_\mathrm{HTC}+R_\mathrm{LTC}$.
Factor: Conventional total $\div$ method total.}
\label{table:cost_comparison}
\setlength{\tabcolsep}{4pt}
\adjustbox{max width=\textwidth}{%
\begin{tabular}{llccrccc}
\toprule
\multirow{2}{*}{\textbf{Method}} &
\multirow{2}{*}{\textbf{Stage}} &
\multirow{2}{*}{\makecell{\textbf{Cases}\\$(n)$}} &
\multirow{2}{*}{\makecell{$\boldsymbol{R}$\\\textbf{(active range)}}} &
\multirow{2}{*}{\makecell{\textbf{Total Cost}\\(simulations)}} &
\multicolumn{3}{c}{\textbf{Cost Saving (\%)}} \\
\cmidrule(lr){6-8}
& & & & & vs.\ Full-PRS (staged) & vs.\ Conventional & Factor \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return tex


# ══════════════════════════════════════════════════════════════════════════════
# ERROR TABLE — one stage
# ══════════════════════════════════════════════════════════════════════════════

def _error_table(errors: dict, stage_lbls: list, improvement: dict,
                 stage_label: str, stage_name: str, caption_extra: str = '') -> str:
    """
    Rows: conditions (φ, P).
    Columns: Nominal ε | Stage-1 ε | Δε | Stage-2 ε | Δε | ...
    Bold = improved over nominal; italic = degraded.
    """
    nominal_lbl = stage_lbls[0]
    opt_lbls    = stage_lbls[1:]
    conds       = sorted(errors.keys(), key=lambda x: (x[0], x[1]))

    # ── Header ────────────────────────────────────────────────────
    ncols_opt = 2 * len(opt_lbls)   # ε + Δε per optimized stage
    ncols_total = 3 + ncols_opt

    col_spec = 'cc' + 'r' + 'rc' * len(opt_lbls)

    header1 = (r'  \multirow{2}{*}{$\boldsymbol{\phi}$} & '
               r'\multirow{2}{*}{\makecell{\textbf{$P$}\\\textbf{(atm)}}} & '
               r'\multirow{2}{*}{\makecell{\textbf{Nominal}\\$\boldsymbol{\varepsilon}$}}')
    for lbl in opt_lbls:
        header1 += f' & \\multicolumn{{2}}{{c}}{{\\textbf{{{lbl}}}}}'
    header1 += r' \\' + '\n'

    header2 = '  & & '
    for lbl in opt_lbls:
        header2 += r' & $\boldsymbol{\varepsilon}$ & $\Delta\varepsilon\,(\%)$'
    header2 += r' \\' + '\n'

    # cmidrule for opt columns
    cmidrules = ''
    start = 4
    for _ in opt_lbls:
        cmidrules += f'  \\cmidrule(lr){{{start}-{start+1}}}\n'
        start += 2

    # ── Data rows ─────────────────────────────────────────────────
    data_rows = ''
    prev_phi  = None
    for cond in conds:
        phi, p = cond
        cdata  = errors[cond]
        nom    = cdata.get(nominal_lbl, None)
        imp_dict   = improvement.get(cond, {})
        delta_dict = imp_dict.get('delta', {})
        impr_dict  = imp_dict.get('improved', {})

        # Alternating row shading via rowcolor is avoided (no colour requested)
        # Use \midrule between phi groups instead
        if prev_phi is not None and phi != prev_phi:
            data_rows += '  \\midrule\n'

        phi_cell = f'  \\multirow{{3}}{{*}}{{{phi}}}' if p == sorted(
            [c[1] for c in conds if c[0] == phi])[0] else '  '

        nom_str = _fmt(nom) if nom is not None else '--'
        row = f'{phi_cell} & {p} & {nom_str}'

        for lbl in opt_lbls:
            opt = cdata.get(lbl)
            if opt is None:
                row += ' & -- & --'
                continue
            imp  = impr_dict.get(lbl)
            eps_cell  = _cell(_fmt(opt), imp)
            delt_cell = _delta_cell(delta_dict.get(lbl, 0.0))
            row += f' & {eps_cell} & {delt_cell}'

        data_rows += row + r' \\' + '\n'
        prev_phi = phi

    tex = (f'\n\\begin{{table}}[!ht]\n\\centering\\small\n'
           f'\\caption{{Objective function $\\varepsilon$ for '
           f'{stage_label} ({stage_name}) optimization.\n'
           f'$\\Delta\\varepsilon\\,(\\%) = 100(\\varepsilon_\\mathrm{{nom}} - '
           f'\\varepsilon_\\mathrm{{opt}})/\\varepsilon_\\mathrm{{nom}}$; '
           f'positive = improvement.\n'
           f'\\textbf{{Bold}} = improved over nominal; '
           f'\\textit{{italic}} = degraded.{caption_extra}}}\n'
           f'\\label{{table:eps_{stage_label.lower()}}}\n'
           f'\\setlength{{\\tabcolsep}}{{4pt}}\n'
           f'\\adjustbox{{max width=\\textwidth}}{{%%\n'
           f'\\begin{{tabular}}{{{col_spec}}}\n'
           f'\\toprule\n'
           + header1 + cmidrules + header2 +
           f'\\midrule\n'
           + data_rows +
           f'\\bottomrule\n'
           f'\\end{{tabular}}}}\n'
           f'\\end{{table}}\n')
    return tex


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _threshold_comparison_table(errors_per_thresh: dict,
                                  nominal_label: str,
                                  thresholds: list,
                                  stage_label: str) -> str:
    """
    errors_per_thresh: {thresh: errors_dict}
    Columns: Nominal | δ=0.05 ε | Δε | δ=0.01 ε | Δε | ...
    """
    # Gather all conditions from first threshold
    first_err = next(iter(errors_per_thresh.values()))
    conds = sorted(first_err.keys(), key=lambda x: (x[0], x[1]))

    header = (r'\multirow{2}{*}{$\boldsymbol{\phi}$} & '
              r'\multirow{2}{*}{\makecell{\textbf{$P$}\\(atm)}} & '
              r'\multirow{2}{*}{\makecell{\textbf{Nominal}\\$\boldsymbol{\varepsilon}$}}')
    for t in thresholds:
        header += f' & \\multicolumn{{2}}{{c}}{{$\\delta={t}$}}'
    header += r' \\' + '\n'

    sub = '& &'
    for _ in thresholds:
        sub += r' & $\boldsymbol{\varepsilon}$ & $\Delta\varepsilon\,(\%)$'
    sub += r' \\' + '\n'

    cmidrules = ''
    start = 4
    for _ in thresholds:
        cmidrules += f'  \\cmidrule(lr){{{start}-{start+1}}}\n'
        start += 2

    col_spec = 'ccr' + 'rc' * len(thresholds)

    rows = ''
    prev_phi = None
    for cond in conds:
        phi, p = cond
        nom = first_err[cond].get(nominal_label)
        if prev_phi is not None and phi != prev_phi:
            rows += '  \\midrule\n'
        phi_cell = (f'  \\multirow{{3}}{{*}}{{{phi}}}'
                    if p == sorted([c[1] for c in conds if c[0]==phi])[0]
                    else '  ')
        row = f'{phi_cell} & {p} & {_fmt(nom)}'
        for t in thresholds:
            err_t = errors_per_thresh.get(t, {})
            cond_t = err_t.get(cond, {})
            # Assume first key is nominal, rest are optimized
            opt_lbl = [k for k in cond_t if k != nominal_label]
            # Take the first optimized stage for threshold comparison
            opt = cond_t.get(opt_lbl[0]) if opt_lbl else None
            if opt is None:
                row += ' & -- & --'
            else:
                imp = opt < nom if nom else None
                d   = 100*(nom - opt)/nom if nom else 0
                row += f' & {_cell(_fmt(opt), imp)} & {_delta_cell(d)}'
        rows += row + r' \\' + '\n'
        prev_phi = phi

    return (f'\n\\begin{{table}}[!ht]\n\\centering\\small\n'
            f'\\caption{{Threshold comparison of objective function $\\varepsilon$ '
            f'for {stage_label} Stage-1 optimization. '
            f'\\textbf{{Bold}} = improved; \\textit{{italic}} = degraded.}}\n'
            f'\\label{{table:thresh_{stage_label.lower()}}}\n'
            f'\\setlength{{\\tabcolsep}}{{4pt}}\n'
            f'\\adjustbox{{max width=\\textwidth}}{{%%\n'
            f'\\begin{{tabular}}{{{col_spec}}}\n'
            f'\\toprule\n{header}{cmidrules}{sub}\\midrule\n'
            f'{rows}\\bottomrule\n\\end{{tabular}}}}\n'
            f'\\end{{table}}\n')


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE IMPROVEMENT TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_table(analysis: dict, cfg: dict) -> str:
    primary = cfg['primary_threshold']
    labels  = cfg['stage_labels']

    # Collect aggregate data
    rows = []
    for lbl in labels:
        agg = analysis['stages'][lbl]['aggregate'].get(primary)
        if agg is None:
            continue
        stage_name = cfg['stage_names'][cfg['stage_labels'].index(lbl)]
        stage_lbls = analysis['stages'][lbl].get('stage_labels', [])
        nom_lbl    = stage_lbls[0] if stage_lbls else 'Nominal'
        opt_lbls   = stage_lbls[1:] if len(stage_lbls) > 1 else []
        rows.append((lbl, stage_name, nom_lbl, opt_lbls, agg))

    if not rows:
        return ''

    # Number of optimized stages (max across all stages)
    max_opt = max(len(r[3]) for r in rows)

    header = (r'\multirow{2}{*}{\textbf{Stage}} & '
              r'\multirow{2}{*}{\makecell{\textbf{Nominal}\\\textbf{$\sum\varepsilon$}}}')
    for i in range(max_opt):
        header += (f' & \\multicolumn{{3}}{{c}}{{\\textbf{{Opt. {i+1}}}}}'
                   if max_opt > 1 else
                   r' & \multicolumn{3}{c}{\textbf{Optimized}}')
    header += r' \\' + '\n'

    sub = '& '
    for _ in range(max_opt):
        sub += r' & $\sum\varepsilon$ & $\Delta\varepsilon\,(\%)$ & Factor'
    sub += r' \\' + '\n'

    cmidrules = ''
    start = 3
    for _ in range(max_opt):
        cmidrules += f'  \\cmidrule(lr){{{start}-{start+2}}}\n'
        start += 3

    col_spec = 'lr' + 'rcc' * max_opt

    data_rows = ''
    for (lbl, sname, nom_lbl, opt_lbls, agg) in rows:
        nom_sum = agg[nom_lbl]['sum']
        row = f'  {lbl} ({sname}) & {_fmt(nom_sum)}'
        for i in range(max_opt):
            if i < len(opt_lbls):
                ol  = opt_lbls[i]
                ag  = agg[ol]
                s   = ag['sum']
                d   = ag['delta_pct']
                f   = ag['factor']
                imp = s < nom_sum
                f_str = (f'$\\sim{round(f)}\\times$' if f < 100
                         else f'$\\sim{f:.1f}\\times$')
                row += (f' & {_cell(_fmt(s), imp)}'
                        f' & {_delta_cell(d)}'
                        f' & {_cell(f_str, imp)}')
            else:
                row += ' & -- & -- & --'
        data_rows += row + r' \\' + '\n'

    return (f'\n\\begin{{table}}[!ht]\n\\centering\\small\n'
            f'\\caption{{Aggregate objective function sums and improvement factors '
            f'for all optimization stages. '
            f'Factor $= \\sum\\varepsilon_\\mathrm{{nom}}/\\sum\\varepsilon_\\mathrm{{opt}}$; '
            f'values $>1$ indicate overall improvement.}}\n'
            f'\\label{{table:aggregate}}\n'
            f'\\setlength{{\\tabcolsep}}{{5pt}}\n'
            f'\\adjustbox{{max width=\\textwidth}}{{%%\n'
            f'\\begin{{tabular}}{{{col_spec}}}\n'
            f'\\toprule\n{header}{cmidrules}{sub}\\midrule\n'
            f'{data_rows}\\bottomrule\n\\end{{tabular}}}}\n'
            f'\\end{{table}}\n')


# ══════════════════════════════════════════════════════════════════════════════
# PRS STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _prs_stats_table(analysis: dict, cfg: dict) -> str:
    rows = ''
    for lbl in cfg['stage_labels']:
        ms = analysis['stages'][lbl].get('prs_stats')
        if ms is None:
            continue
        thr = ms.get('threshold', '--')
        fbt = ms.get('frac_below_threshold')
        fbt_str = f'{100*fbt:.1f}\\%' if fbt is not None else '--'
        rows += (f'  {lbl} & {ms["n_cases"]} & '
                 f'{ms["train_mean"]:.2f} & {ms["train_std"]:.2f} & '
                 f'{ms["train_max"]:.2f} & {ms["train_min"]:.2f} & '
                 f'{ms["test_mean"]:.2f} & {ms["test_std"]:.2f} & '
                 f'{ms["test_max"]:.2f} & {ms["test_min"]:.2f} & '
                 f'{thr} & {fbt_str} \\\\\n')

    if not rows:
        return ''

    return r"""
\section*{D.\enspace PRS Fit Statistics}

\begin{table}[!ht]
\centering
\small
\caption{Summary statistics of PRS Maximum Residual Error (MRE, in \%)
for training and testing sets across all optimization cases.
The acceptance threshold column shows the MRE value below which a PRS
is deemed accurate; the final column gives the fraction of cases satisfying
this criterion.}
\label{table:prs_stats}
\setlength{\tabcolsep}{4pt}
\adjustbox{max width=\textwidth}{%
\begin{tabular}{lcrrrrrrrrrr}
\toprule
\multirow{2}{*}{\textbf{Stage}} &
\multirow{2}{*}{\makecell{\textbf{Cases}\\$(n)$}} &
\multicolumn{4}{c}{\textbf{Training MRE (\%)}} &
\multicolumn{4}{c}{\textbf{Testing MRE (\%)}} &
\multirow{2}{*}{\makecell{\textbf{Accept.}\\\textbf{Threshold}}} &
\multirow{2}{*}{\makecell{\textbf{Cases}\\\textbf{Below Thr.}}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
& & Mean & Std & Max & Min & Mean & Std & Max & Min & & \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}}
\end{table}
"""


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATED SUMMARY PARAGRAPHS
# ══════════════════════════════════════════════════════════════════════════════

def _summary_paragraphs(analysis: dict, cfg: dict, metadata: dict) -> str:
    primary = cfg['primary_threshold']
    labels  = cfg['stage_labels']
    fuel    = metadata.get('fuel_name', 'the fuel')
    cc      = analysis['combined']
    paras   = [r'\section*{E.\enspace Summary of Results}', '']

    # ── Cost paragraph ─────────────────────────────────────────────
    pprs  = cc['pprs_total']
    full  = cc['full_total']
    sav   = cc['combined_saving_pct']
    nc_vals = []
    for lbl in labels:
        ca = analysis['stages'][lbl]['cost'].get(primary, {})
        if ca:
            nc_vals.append(f'$n_c = {1 + ca["k_total"] + ca["k_total"]*(ca["k_total"]+1)//2}$ '
                           f'({lbl}, $R={ca["k_total"]}$)')

    conv_str = ''
    if 'conventional' in cc:
        kmax = cc.get('k_union_max', '?')
        cv   = cc['conventional'].get(kmax, {})
        if cv:
            conv_str = (f'The conventional Full-PRS baseline (single stage, '
                        f'$R_\\mathrm{{union}} = {kmax}$) requires '
                        f'\\textbf{{{cv["total"]:,}}} simulations — '
                        f'a factor of $\\mathbf{{{cv["factor_vs_pprs"]:.1f}\\times}}$ '
                        f'more than the p-PRS combined cost, arising from the '
                        f'$\\mathcal{{O}}(R^2)$ scaling of $n_c$.')

    paras.append(
        r'\noindent\textbf{Computational cost.} '
        f'The p-PRS framework ({"; ".join(nc_vals)}) requires a combined total of '
        f'\\textbf{{{pprs:,}}} forward model evaluations for all stages, '
        f'compared to \\textbf{{{full:,}}} for Full-PRS (staged) — '
        f'a saving of $\\mathbf{{{sav:.1f}\\%}}$. '
        + conv_str
    )

    # ── Per-stage improvement paragraphs ──────────────────────────
    for lbl in labels:
        stage_name  = cfg['stage_names'][cfg['stage_labels'].index(lbl)]
        agg         = analysis['stages'][lbl]['aggregate'].get(primary)
        stage_lbls  = analysis['stages'][lbl].get('stage_labels', [])
        improvement = analysis['stages'][lbl]['improvement'].get(primary, {})

        if agg is None or not stage_lbls:
            continue

        nom_lbl  = stage_lbls[0]
        opt_lbls = stage_lbls[1:]
        nom_sum  = agg[nom_lbl]['sum']

        para = (f'\n\\noindent\\textbf{{{lbl} ({stage_name}) — '
                f'objective function improvement.}} ')

        for oi, ol in enumerate(opt_lbls):
            ag   = agg[ol]
            d    = ag['delta_pct']
            f    = ag['factor']
            # Count improved / degraded conditions
            n_imp  = sum(1 for v in improvement.values()
                         if v['improved'].get(ol, False))
            n_tot  = len(improvement)
            n_deg  = n_tot - n_imp

            # Collect improved and degraded deltas for mean
            imp_d  = [v['delta'][ol] for v in improvement.values()
                      if ol in v['delta'] and v['improved'].get(ol, False)]
            deg_d  = [v['delta'][ol] for v in improvement.values()
                      if ol in v['delta'] and not v['improved'].get(ol, False)]

            mean_imp = sum(imp_d)/len(imp_d) if imp_d else 0
            mean_deg = abs(sum(deg_d)/len(deg_d)) if deg_d else 0

            f_str   = (f'$\\sim{round(f, 1)}\\times$' if f >= 2
                       else f'$\\sim{f:.2f}\\times$')
            dir_str = 'reduces' if d > 0 else 'increases'
            sign    = '+' if d >= 0 else ''

            para += (
                f'{ol} {dir_str} the aggregate {lbl} objective function by '
                f'$\\mathbf{{{sign}{d:.1f}\\%}}$ relative to nominal '
                f'(factor {f_str}; $\\Sigma\\varepsilon_\\mathrm{{nom}} = '
                f'{_fmt(nom_sum)}$, '
                f'$\\Sigma\\varepsilon_\\mathrm{{opt}} = '
                f'{_fmt(ag["sum"])}$). '
            )

            if n_imp > 0 and n_deg > 0:
                para += (
                    f'At the condition level, \\textbf{{{n_imp} of {n_tot}}} '
                    f'conditions improve (mean $\\Delta\\varepsilon = '
                    f'+{mean_imp:.1f}\\%$), while {n_deg} condition'
                    f'{"s" if n_deg > 1 else ""} '
                    f'show a degradation, with the objective function '
                    f'increasing by an average of ${mean_deg:.1f}\\%$ above '
                    f'the nominal value. '
                )
            elif n_imp == n_tot:
                para += (
                    f'All \\textbf{{{n_tot} of {n_tot}}} conditions improve '
                    f'(mean $\\Delta\\varepsilon = +{mean_imp:.1f}\\%$). '
                )
            elif n_imp == 0:
                para += (
                    f'None of the {n_tot} conditions improve; the objective '
                    f'function increases by an average of '
                    f'${mean_deg:.1f}\\%$ above nominal. '
                )

            # Identify the best-improving condition
            best = max(improvement.items(),
                       key=lambda x: x[1]['delta'].get(ol, -999))
            best_cond, best_val = best
            best_d = best_val['delta'].get(ol, 0)
            if best_d > 0:
                para += (
                    f'The largest improvement is at $\\phi={best_cond[0]}$, '
                    f'$P={best_cond[1]}$\\,atm '
                    f'($\\Delta\\varepsilon = +{best_d:.1f}\\%$). '
                )

            para += '\n'

        paras.append(para)

    return '\n\n'.join(paras) + '\n'


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _figures_section(plot_paths: dict, all_data: dict, cfg: dict) -> str:
    lines = [r'\section*{F.\enspace Figures}', '']

    if 'cost_comparison' in plot_paths:
        lines.append(r'\subsection*{F.1\enspace Simulation Cost}')
        lines.append(_inc(plot_paths['cost_comparison'], r'0.95\linewidth'))

    for lbl in cfg['stage_labels']:
        if f'radar_{lbl}' in plot_paths:
            lines.append(f'\\subsection*{{\\enspace Multi-Condition Profile — {lbl}}}')
            lines.append(_inc(plot_paths[f'radar_{lbl}'], r'0.65\linewidth'))

        if f'sensitivity_{lbl}' in plot_paths:
            lines.append(f'\\subsection*{{\\enspace Sensitivity Heatmap — {lbl}}}')
            lines.append(_inc(plot_paths[f'sensitivity_{lbl}'], r'0.95\linewidth'))

        if f'mre_{lbl}' in plot_paths:
            lines.append(f'\\subsection*{{\\enspace PRS MRE Distribution — {lbl}}}')
            lines.append(_inc(plot_paths[f'mre_{lbl}'], r'0.90\linewidth'))

        # IDT comparison plots (user-provided)
        idt_folder = all_data.get(lbl, {}).get('idt_plots_folder')
        if idt_folder:
            idt_plots = sorted(Path(idt_folder).glob('*.pdf'))
            if idt_plots:
                lines.append(f'\\subsection*{{\\enspace IDT Comparison — {lbl}}}')
                for pp in idt_plots:
                    lines.append(_inc(str(pp), r'0.85\linewidth'))

    if 'convergence' in plot_paths:
        lines.append(r'\subsection*{F.\enspace Convergence History}')
        lines.append(_inc(plot_paths['convergence'], r'0.75\linewidth'))

    return '\n'.join(lines) + '\n'


# ══════════════════════════════════════════════════════════════════════════════
# MASTER DOCUMENT ASSEMBLER
# ══════════════════════════════════════════════════════════════════════════════

def generate_latex_document(all_data: dict, analysis: dict,
                             cfg: dict, metadata: dict,
                             plot_paths: dict) -> str:
    primary = cfg['primary_threshold']
    labels  = cfg['stage_labels']

    parts = [
        _preamble(),
        _header(metadata, cfg),
        _methodology(metadata, cfg, analysis),
        _cost_table(analysis, cfg),
        r'\section*{C.\enspace Objective Function Analysis}' + '\n',
    ]

    # Threshold comparison tables (if requested)
    if cfg.get('compare_thresholds') and len(cfg['thresholds']) > 1:
        for lbl in labels:
            errors = all_data.get(lbl, {}).get('errors')
            if errors is None:
                continue
            stage_lbls = analysis['stages'][lbl].get('stage_labels', [])
            if not stage_lbls:
                continue
            nom_lbl = stage_lbls[0]
            # Build a dict keyed by threshold — here we only have primary data
            # (multiple threshold folders would populate this; placeholder)
            parts.append(
                f'% NOTE: Threshold comparison for {lbl} requires separate\n'
                f'% error folders per threshold. Currently only δ={primary} available.\n'
            )

    # Per-stage error tables
    for lbl in labels:
        stage_name = cfg['stage_names'][cfg['stage_labels'].index(lbl)]
        errors     = all_data.get(lbl, {}).get('errors')
        if errors is None:
            continue
        stage_lbls = analysis['stages'][lbl].get('stage_labels', [])
        improvement = analysis['stages'][lbl]['improvement'].get(primary, {})
        if stage_lbls:
            parts.append(
                _error_table(errors, stage_lbls, improvement,
                             lbl, stage_name))

    # Aggregate table
    parts.append(_aggregate_table(analysis, cfg))

    # PRS stats table
    parts.append(_prs_stats_table(analysis, cfg))

    # Summary paragraphs
    parts.append(_summary_paragraphs(analysis, cfg, metadata))

    # Figures
    parts.append(_figures_section(plot_paths, all_data, cfg))

    parts.append(r'\end{document}' + '\n')

    return '\n'.join(parts)
