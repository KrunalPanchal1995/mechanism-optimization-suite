"""
target_tables.py
----------------
Generate customizable LaTeX target-data tables from a .target CSV file.

One table per target type (Tig / Fls / JSR / Flf / RCM).
Citation keys are resolved from an optional references.bib file.
Output: one .tex file per group + a combined PDF (via pdflatex).

Usage::

    from target_tables import generate_target_tables
    generate_target_tables(
        target_csv="output/targets.target",
        bib_path="references.bib",
        target_types=["Tig", "Fls"],
        columns=["T", "P", "phi", "fuel", "reference"],
        out_dir="latex_tables",
        compile_pdf=True,
    )

CLI::

    python target_tables.py targets.target --bib references.bib \\
        --types Tig Fls --out latex_tables
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Column display names
# ---------------------------------------------------------------------------

_COL_LABELS = {
    "T":        r"$T$ (K)",
    "P":        r"$P$ (atm)",
    "phi":      r"$\phi$",
    "fuel":     "Fuel",
    "oxidizer": "Oxidizer",
    "bath_gas": "Bath gas",
    "type":     "Type",
    "reference": "Ref.",
    "target":   "Observable",
    "value":    "Value",
}


# ---------------------------------------------------------------------------
# BibTeX key extraction
# ---------------------------------------------------------------------------

def _load_bib_keys(bib_path: str) -> Dict[str, str]:
    """Return {citekey: author_year} for quick display."""
    keys = {}
    if not bib_path or not os.path.exists(bib_path):
        return keys
    pattern = re.compile(r"@\w+\{([^,]+),", re.IGNORECASE)
    with open(bib_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                keys[m.group(1).strip()] = m.group(1).strip()
    return keys


# ---------------------------------------------------------------------------
# Parse .target file
# ---------------------------------------------------------------------------

_PIPE_SEP = re.compile(r"\s*\|\s*")

def _read_target_csv(path: str) -> List[dict]:
    """
    Read a pipe-delimited .target file or a regular CSV.
    Returns list of row dicts.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip() and not l.startswith("#")]
    if not lines:
        return rows
    # Detect delimiter
    if "|" in lines[0]:
        header = _PIPE_SEP.split(lines[0].strip(" |"))
        for line in lines[1:]:
            vals = _PIPE_SEP.split(line.strip(" |"))
            rows.append(dict(zip(header, vals + [""] * max(0, len(header) - len(vals)))))
    else:
        reader = csv.DictReader(lines)
        rows = list(reader)
    return rows


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}"))


def _build_table(rows: List[dict], columns: List[str],
                 bib_keys: Dict[str, str], caption: str, label: str) -> str:
    if not rows:
        return f"% No data for {label}\n"

    col_spec = "|" + "c|" * len(columns)
    header_row = " & ".join(
        r"\textbf{" + _COL_LABELS.get(c, c) + "}" for c in columns
    ) + r" \\"

    body_lines = []
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, row.get(c.upper(), ""))
            if c == "reference" and v in bib_keys:
                v = rf"\cite{{{v}}}"
            else:
                v = _escape(str(v))
            vals.append(v)
        body_lines.append(" & ".join(vals) + r" \\")
        body_lines.append(r"\hline")

    body = "\n    ".join(body_lines)
    n = len(rows)

    return rf"""
\begin{{table}}[htbp]
  \centering
  \caption{{{_escape(caption)} ({n} data points)}}
  \label{{{label}}}
  \begin{{tabular}}{{{col_spec}}}
    \hline
    {header_row}
    \hline
    {body}
  \end{{tabular}}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_target_tables(
    target_csv: str,
    bib_path: str = "",
    target_types: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    out_dir: str = "latex_tables",
    compile_pdf: bool = True,
) -> str:
    """
    Generate one .tex per target type and an optional compiled PDF.

    Parameters
    ----------
    target_csv   : path to .target file (pipe- or comma-delimited)
    bib_path     : path to references.bib (optional)
    target_types : list of types to include, e.g. ["Tig","Fls"]. None = all.
    columns      : columns to show. None = default set.
    out_dir      : output directory
    compile_pdf  : run pdflatex on the combined document

    Returns
    -------
    str : path to the generated combined .tex file
    """
    if columns is None:
        columns = ["T", "P", "phi", "fuel", "reference"]
    if target_types is None:
        target_types = ["Tig", "Fls", "JSR", "Flf", "RCM"]

    os.makedirs(out_dir, exist_ok=True)
    rows = _read_target_csv(target_csv)
    bib_keys = _load_bib_keys(bib_path)

    # Group by target type
    groups: Dict[str, List[dict]] = {t: [] for t in target_types}
    for row in rows:
        ttype = row.get("target", row.get("Target", "")).strip()
        if ttype in groups:
            groups[ttype].append(row)

    # Build individual .tex files
    tex_sections = []
    for ttype in target_types:
        grp = groups.get(ttype, [])
        caption = {
            "Tig": "Ignition delay time targets",
            "Fls": "Laminar flame speed targets",
            "JSR": "Jet-stirred reactor targets",
            "Flf": "Counterflow diffusion flame targets",
            "RCM": "Rapid compression machine targets",
        }.get(ttype, f"{ttype} targets")
        label = f"tab:{ttype.lower()}"
        tex_sections.append(
            f"% ── {ttype} ────────────────────────────────────────────────────\n"
            + _build_table(grp, columns, bib_keys, caption, label)
        )

    # Combined document
    bib_section = ""
    if bib_path and os.path.exists(bib_path):
        shutil.copy2(bib_path, os.path.join(out_dir, "references.bib"))
        bib_section = r"""
\bibliographystyle{plainnat}
\bibliography{references}
"""

    combined_tex = rf"""
\documentclass[12pt,a4paper]{{article}}
\usepackage{{booktabs,geometry,natbib}}
\geometry{{margin=2cm}}
\title{{Target Dataset Summary Tables}}
\date{{\today}}
\begin{{document}}
\maketitle
{"".join(tex_sections)}
{bib_section}
\end{{document}}
"""

    combined_path = os.path.join(out_dir, "target_tables.tex")
    with open(combined_path, "w", encoding="utf-8") as fh:
        fh.write(combined_tex)
    print(f"Written: {combined_path}")

    if compile_pdf:
        # Was a direct pdflatex-only call, which silently no-ops ("[pdflatex
        # not found ...]") on a tectonic-only machine (this one, as of
        # 2026-08) even though a usable LaTeX engine is present. Route
        # through the shared tectonic->latexmk->pdflatex ladder instead,
        # same as latex_project.py/run_report.py/mso_studio's reporting.
        try:
            from mso_io.latex_engine import compile_latex
        except ImportError:
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            if _this_dir not in sys.path:
                sys.path.append(_this_dir)
            from mso_io.latex_engine import compile_latex
        pdf_path, engine, log_tail = compile_latex(Path(combined_path), Path(out_dir))
        if pdf_path is not None:
            print(f"Compiled ({engine}): {pdf_path}")
        elif engine == "none":
            print("[no LaTeX engine found on PATH (tried tectonic/latexmk/pdflatex) "
                  "-- .tex written but not compiled]")
        else:
            print(f"[{engine} failed -- .tex written but not compiled; see {log_tail[-200:]!r}]")

    return combined_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Generate LaTeX target-data summary tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("target_csv",    help=".target file path")
    p.add_argument("--bib",         default="",  help="references.bib path")
    p.add_argument("--types", nargs="+",
                   default=["Tig", "Fls", "JSR", "Flf", "RCM"],
                   help="Target types to include")
    p.add_argument("--cols", nargs="+",
                   default=["T", "P", "phi", "fuel", "reference"],
                   help="Columns to include")
    p.add_argument("--out",  default="latex_tables", help="Output directory")
    p.add_argument("--no-pdf", action="store_true", help="Skip pdflatex compilation")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_target_tables(
        target_csv=args.target_csv,
        bib_path=args.bib,
        target_types=args.types,
        columns=args.cols,
        out_dir=args.out,
        compile_pdf=not args.no_pdf,
    )
