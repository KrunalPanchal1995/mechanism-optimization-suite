#!/usr/bin/env python3
"""
MSO Report Generator — Main Entry Point
========================================
Run:  python run_report.py

Generates a comprehensive LaTeX report for Multi-Stage Optimization (MSO)
of combustion kinetic mechanisms using the p-PRS framework.

Outputs (in ./mso_output/ by default):
  mso_report.tex        — full LaTeX document
  mso_report.pdf        — compiled PDF (requires pdflatex)
  plots/                — all matplotlib figures (.pdf)
  plot_data/            — CSV snapshots of every plot's underlying data
"""

import os
import sys
import subprocess
from pathlib import Path

# ── Local imports ──────────────────────────────────────────────────────────────
from io.mso_io       import (parse_yaml_metadata, parse_error_csvs_from_folder,
                           parse_cost_csv, parse_prs_stats_csv,
                           parse_sensitivity_csv, parse_convergence_csv)
from io.mso_analysis import run_full_analysis
from io.mso_plots    import generate_all_plots
from io.mso_latex    import generate_latex_document


# ══════════════════════════════════════════════════════════════════════════════
# Interactive helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt, default=None, cast=str, allow_empty=False):
    """Prompt user with optional default; cast the response."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"  {prompt}{suffix}: ").strip()
    if not raw:
        if default is not None:
            return cast(default) if cast is not str else str(default)
        if allow_empty:
            return None
        print("    ✗  Required — please enter a value.")
        return _ask(prompt, default, cast, allow_empty)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"    ✗  Cannot interpret '{raw}' as {cast.__name__}. Try again.")
        return _ask(prompt, default, cast, allow_empty)


def _yn(prompt, default='n'):
    raw = input(f"  {prompt} (y/n) [{default}]: ").strip().lower()
    return (raw if raw in ('y', 'n') else default) == 'y'


def _section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive session
# ══════════════════════════════════════════════════════════════════════════════

def interactive_session():
    """Walk the user through all configuration choices; return cfg dict."""
    print("\n" + "═"*62)
    print("  MSO REPORT GENERATOR")
    print("  p-PRS  ·  Multi-Stage Optimization  ·  LaTeX Report")
    print("═"*62)

    cfg = {}

    # ── 1. Pipeline structure ──────────────────────────────────────
    _section("STEP 1  ·  PIPELINE STRUCTURE")

    cfg['n_stages'] = _ask("Number of optimization stages", default=2, cast=int)

    cfg['stage_names']  = []   # full names  (e.g. "High Temperature Combustion")
    cfg['stage_labels'] = []   # short codes (e.g. "HTC")
    for i in range(cfg['n_stages']):
        name  = _ask(f"Full name for Stage {i+1}  (e.g. High Temperature Combustion)",
                     default=f"Stage-{i+1}")
        label = _ask(f"Short code for Stage {i+1} (e.g. HTC)",
                     default=f"S{i+1}")
        cfg['stage_names'].append(name)
        cfg['stage_labels'].append(label.upper())

    # ── 2. Sensitivity thresholds ──────────────────────────────────
    _section("STEP 2  ·  SENSITIVITY THRESHOLDS")

    cfg['compare_thresholds'] = _yn(
        "Are you comparing multiple sensitivity thresholds?", default='n')

    if cfg['compare_thresholds']:
        n_t = _ask("Number of thresholds to compare", default=2, cast=int)
        cfg['thresholds'] = []
        for i in range(n_t):
            cfg['thresholds'].append(_ask(f"Threshold {i+1}", cast=float))
        cfg['primary_threshold'] = _ask(
            "Which threshold is the final/primary result",
            default=cfg['thresholds'][-1], cast=float)
    else:
        t = _ask("Sensitivity threshold used", default=0.01, cast=float)
        cfg['thresholds']          = [t]
        cfg['primary_threshold']   = t

    # ── 3. Metadata YAML ───────────────────────────────────────────
    _section("STEP 3  ·  METADATA")

    cfg['yaml_path'] = _ask("Path to metadata YAML file", default="./metadata.yaml")
    print(f"\n  ℹ  A template is written to ./metadata_template.yaml if you")
    print(  "     need one — edit it and re-run.")

    # ── 4. Input files per stage ───────────────────────────────────
    _section("STEP 4  ·  INPUT FILES")

    cfg['stage_configs'] = []
    for i, (name, label) in enumerate(zip(cfg['stage_names'],
                                          cfg['stage_labels'])):
        print(f"  Stage {i+1} — {label}  ({name})\n")
        sc = {'name': name, 'label': label}

        sc['error_folder'] = _ask(
            f"Folder containing error CSVs for {label}")

        sc['cost_csvs'] = {}
        for thresh in cfg['thresholds']:
            p = _ask(f"Cost CSV for {label} at δ={thresh}",
                     allow_empty=True, default="")
            if p:
                sc['cost_csvs'][thresh] = p

        sc['prs_stats_csv']  = _ask(
            f"PRS statistics CSV for {label}  (Enter to skip)",
            default="", allow_empty=True) or None

        sc['sensitivity_csv'] = _ask(
            f"Sensitivity coefficients CSV for {label}  (Enter to skip)",
            default="", allow_empty=True) or None

        sc['convergence_csv'] = _ask(
            f"Convergence history CSV for {label}  (Enter to skip)",
            default="", allow_empty=True) or None

        sc['idt_plots_folder'] = _ask(
            f"Folder with IDT comparison plots for {label}  (Enter to skip)",
            default="", allow_empty=True) or None

        cfg['stage_configs'].append(sc)
        print()

    # ── 5. Output ─────────────────────────────────────────────────
    _section("STEP 5  ·  OUTPUT")

    cfg['output_dir']  = _ask("Output directory", default="./mso_output")
    cfg['compile_pdf'] = _yn("Attempt to compile LaTeX → PDF?", default='y')

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX compilation
# ══════════════════════════════════════════════════════════════════════════════

def compile_latex(tex_path: Path, out_dir: Path):
    """Run pdflatex twice for cross-references."""
    try:
        for _ in range(2):
            r = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode',
                 '-output-directory', str(out_dir), str(tex_path)],
                capture_output=True, text=True
            )
        if r.returncode == 0:
            print(f"  ✓ PDF compiled: {tex_path.with_suffix('.pdf')}")
        else:
            print("  ✗ LaTeX compilation errors — inspect the .log file.")
    except FileNotFoundError:
        print("  ✗ pdflatex not found; .tex file saved but not compiled.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = interactive_session()

    # Create output structure
    out   = Path(cfg['output_dir'])
    plots = out / 'plots'
    data  = out / 'plot_data'
    for d in (out, plots, data):
        d.mkdir(parents=True, exist_ok=True)
    cfg['plots_dir'] = str(plots)
    cfg['data_dir']  = str(data)

    # Write YAML template regardless (useful for new users)
    _write_yaml_template(out / 'metadata_template.yaml')

    _section("LOADING DATA")

    metadata = parse_yaml_metadata(cfg['yaml_path'])
    fuel = metadata.get('fuel_name', 'Unknown fuel')
    print(f"  ✓ Metadata: {fuel}")

    all_data = {}
    for sc in cfg['stage_configs']:
        label = sc['label']
        all_data[label] = {}

        # Error CSVs
        if sc.get('error_folder') and Path(sc['error_folder']).is_dir():
            all_data[label]['errors'] = parse_error_csvs_from_folder(
                sc['error_folder'])
            print(f"  ✓ {label} errors: "
                  f"{len(all_data[label]['errors'])} conditions")

        # Cost CSVs
        all_data[label]['costs'] = {}
        for thresh, csv_p in sc.get('cost_csvs', {}).items():
            p = Path(csv_p) if csv_p else None
            if p and p.exists():
                all_data[label]['costs'][thresh] = parse_cost_csv(str(p))
                print(f"  ✓ {label} cost δ={thresh}: "
                      f"{len(all_data[label]['costs'][thresh])} cases")

        # PRS stats
        if sc.get('prs_stats_csv') and Path(sc['prs_stats_csv']).exists():
            all_data[label]['prs_stats'] = parse_prs_stats_csv(
                sc['prs_stats_csv'])
            print(f"  ✓ {label} PRS stats: "
                  f"{len(all_data[label]['prs_stats'])} rows")

        # Sensitivity (optional)
        if sc.get('sensitivity_csv') and Path(sc['sensitivity_csv']).exists():
            all_data[label]['sensitivity'] = parse_sensitivity_csv(
                sc['sensitivity_csv'])
            print(f"  ✓ {label} sensitivity data loaded")

        # Convergence (optional)
        if sc.get('convergence_csv') and Path(sc['convergence_csv']).exists():
            all_data[label]['convergence'] = parse_convergence_csv(
                sc['convergence_csv'])
            print(f"  ✓ {label} convergence data loaded")

        if sc.get('idt_plots_folder'):
            all_data[label]['idt_plots_folder'] = sc['idt_plots_folder']

    _section("RUNNING ANALYSIS")
    analysis = run_full_analysis(all_data, cfg, metadata)
    print("  ✓ Analysis complete")

    _section("GENERATING PLOTS")
    plot_paths = generate_all_plots(all_data, analysis, cfg, metadata)
    print(f"  ✓ {len(plot_paths)} plot(s) generated")

    _section("BUILDING LaTeX DOCUMENT")
    latex_src = generate_latex_document(
        all_data, analysis, cfg, metadata, plot_paths)

    tex_path = out / 'mso_report.tex'
    tex_path.write_text(latex_src, encoding='utf-8')
    print(f"  ✓ LaTeX: {tex_path}")

    if cfg['compile_pdf']:
        compile_latex(tex_path, out)

    print("\n" + "═"*62)
    print("  DONE — output directory: " + str(out.resolve()))
    print("═"*62 + "\n")


# ── YAML template helper ───────────────────────────────────────────────────────

def _write_yaml_template(path: Path):
    template = """\
# ─────────────────────────────────────────────────────────────
#  MSO Report Generator — Metadata Template
#  Fill in all fields; leave blank if unknown.
# ─────────────────────────────────────────────────────────────

fuel_name:        "1,3-Butadiene"
fuel_formula:     "C4H6"
mechanism_name:   "e.g. Narayanaswamy et al. (2014)"
mechanism_species:    45
mechanism_reactions:  235
HTC_T_range:      "950–1350 K"
HTC_P_range:      "10–40 atm"
LTC_T_range:      "650–900 K"
LTC_P_range:      "10–40 atm"
equivalence_ratios:   [0.5, 1.0, 2.0]
pressures_atm:        [10, 20, 40]
paper_authors:    "Panchal et al."
journal:          "Combustion and Flame"
year:             2025
# Additional free-text note that appears verbatim in the report header:
study_note: >
  This report documents the p-PRS-based Multi-Stage Optimization of the
  kinetic mechanism for {fuel_name}. Stage-1 targets high-temperature
  ignition delay times (HTC); Stage-2 targets low-temperature ignition
  delay times (LTC).
"""
    if not path.exists():
        path.write_text(template, encoding='utf-8')


if __name__ == '__main__':
    main()
