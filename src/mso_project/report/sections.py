"""
mso_project.report.sections
=============================
Per-section LaTeX body builders, styled after ``src/mso_io/mso_latex.py``'s
conventions (booktabs tables, no color -- bold/italic only, plain-string
LaTeX assembly, no template engine) but generic over ARTIFACT ROLE rather
than tied to one specific report's fixed data shape.

Each builder has the signature::

    build_<key>(artifacts: List[dict], project: Project,
                 latex_proj: LaTeXProject, rules: ReportRules) -> str

and returns a LaTeX body ready for ``LaTeXProject.append_section(key, title, body)``.
``MAIN_SECTIONS``/``SUPPLEMENTARY_SECTIONS`` map section keys (as used in
``ReportRules.sections_main``/``sections_supplementary``) to these
functions; ``builder.py`` looks up each configured key and skips (with a
warning, not a hard failure) any key without a registered builder, so a
typo'd or forward-looking section key in ``report_rules.yaml`` never
breaks a build.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from ..project import Project
    from ..report.rules import ReportRules
    from ...latex_project import LaTeXProject  # type: ignore


def _escape(text) -> str:
    """Minimal LaTeX special-character escaping -- deliberately
    self-contained rather than importing target_tables.py's/
    mso_studio.reporting's own copies (this module has no other dependency
    on either).

    Backslash MUST be escaped first: every other replacement below
    (``_`` -> ``\\_``, ``{`` -> ``\\{``, ...) itself inserts a literal
    backslash, and if ``\\`` -> ``\\textbackslash{}`` ran after those, it
    would re-escape the backslashes THIS function just inserted (e.g.
    "run_script" -> "run\\_script" -> corrupted into
    "run\\textbackslash{}_script"), not just backslashes present in the
    original input."""
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _csv_table(csv_path: str, caption: str, label: str, max_rows: int = 60) -> str:
    """Render a CSV file as a booktabs LaTeX table. Truncates past
    max_rows with a note (avoids a multi-thousand-row main.tex from a
    large simulation log accidentally routed here)."""
    try:
        with open(csv_path, newline="") as f:
            rows = list(_csv.reader(f))
    except OSError as exc:
        return f"% could not read {csv_path}: {exc}\n"
    if not rows:
        return f"% {csv_path} is empty\n"
    header, body = rows[0], rows[1:]
    truncated = len(body) > max_rows
    body = body[:max_rows]
    col_spec = "l" * len(header)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        f"\\caption{{{_escape(caption)}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        " & ".join(f"\\textbf{{{_escape(h)}}}" for h in header) + r" \\", r"\midrule",
    ]
    for row in body:
        cells = row + [""] * (len(header) - len(row))
        lines.append(" & ".join(_escape(c) for c in cells[:len(header)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if truncated:
        lines.append(f"\\par\\textit{{({len(rows) - 1 - max_rows} further row(s) omitted "
                      f"-- see the full file in the project's tracked artifacts.)}}")
    return "\n".join(lines) + "\n"


def _artifact_listing_table(artifacts: List[dict], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        f"\\caption{{{_escape(caption)}}}", f"\\label{{{label}}}",
        r"\begin{tabular}{lllr}", r"\toprule",
        r"\textbf{Path} & \textbf{Role} & \textbf{Kind} & \textbf{Size (bytes)} \\", r"\midrule",
    ]
    for a in artifacts:
        lines.append(f"{_escape(Path(a['path']).name)} & {_escape(a['role'])} & "
                      f"{_escape(a['kind'])} & {a['size_bytes'] or 0} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _import_figures(artifacts: List[dict], latex_proj, limit: int = None) -> str:
    figs = [a for a in artifacts if a["kind"] in ("png", "pdf")]
    if limit:
        figs = figs[:limit]
    out = []
    for a in figs:
        caption = f"{a['role'].replace('_', ' ')}: {Path(a['path']).stem}"
        out.append(latex_proj.import_figure(a["path"], caption=caption))
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main-report sections
# ---------------------------------------------------------------------------

def build_overview(artifacts, project, latex_proj, rules) -> str:
    runs = project.list_runs()
    pipelines = sorted({r["pipeline"] for r in runs})
    return (
        f"\\textbf{{{_escape(project.info.get('name', ''))}}}\n\n"
        f"{_escape(project.info.get('description', ''))}\n\n"
        f"This report was generated from {len(runs)} tracked run(s) across "
        f"{len(pipelines)} pipeline(s) ({_escape(', '.join(pipelines))}) and "
        f"{len(project.list_artifacts())} tracked artifact(s).\n"
    )


def build_methodology(artifacts, project, latex_proj, rules) -> str:
    lines = []
    for run in project.list_runs():
        lines.append(f"\\textit{{{_escape(run['pipeline'])}}} -- "
                      f"{_escape(run['label'])} ({_escape(run['status'])}, "
                      f"started {_escape(run['started_at'])})\\\\")
    return "\n".join(lines) + "\n" if lines else "No runs tracked yet.\n"


def build_cost_summary(artifacts, project, latex_proj, rules) -> str:
    cost = [a for a in artifacts if a["role"] == "cost_summary" and a["kind"] == "csv"]
    if not cost:
        return "No cost-summary data tracked.\n"
    return "\n".join(_csv_table(a["path"], "Optimization cost", f"tab:cost_{i}")
                      for i, a in enumerate(cost))


def build_fit_quality(artifacts, project, latex_proj, rules) -> str:
    fq = [a for a in artifacts if a["role"] == "fit_quality_summary" and a["kind"] == "csv"]
    if not fq:
        return "No fit-quality data tracked.\n"
    out = [_csv_table(a["path"], "Prediction error summary", f"tab:fit_{i}") for i, a in enumerate(fq)]
    out.append(f"\\par\\textit{{Cases with hold-out error above "
               f"{rules.holdout_error_flag_pct:g}\\% should be flagged for review "
               f"(see the raw table above for exact values).}}")
    return "\n".join(out)


def build_key_figures(artifacts, project, latex_proj, rules) -> str:
    return _import_figures(artifacts, latex_proj, limit=rules.max_figures_main)


def build_outlier_retraining_history(artifacts, project, latex_proj, rules) -> str:
    """Deliberately re-queries role="study_marker" directly (rather than
    using the `artifacts` list already routed to this section by role_routing)
    -- study_marker/archived_study default to role_routing="exclude" (they're
    operational caches, not report-ready tables/figures), but THIS specific
    section's whole purpose is a curated narrative summary of them, which is
    a different, deliberate exception to the general auto-routing rule."""
    markers = [a for a in project.list_artifacts() if a["role"] == "study_marker"]
    if not markers:
        return "No outlier-retraining study history tracked.\n"
    lines = [f"{len(markers)} resampling stud(y/ies) recorded; see "
             f"\\texttt{{{_escape(Path(m['path']).parent.name)}}} for each."
             for m in markers]
    return "\\\\\n".join(lines) + "\n"


def build_conclusions(artifacts, project, latex_proj, rules) -> str:
    return "Write your conclusions here.\n"


# ---------------------------------------------------------------------------
# Supplementary sections
# ---------------------------------------------------------------------------

def build_full_run_table(artifacts, project, latex_proj, rules) -> str:
    runs = project.list_runs()
    if not runs:
        return "No runs tracked.\n"
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\caption{All tracked runs}",
        r"\label{tab:full_run_table}", r"\begin{tabular}{llll}", r"\toprule",
        r"\textbf{Pipeline} & \textbf{Label} & \textbf{Status} & \textbf{Started} \\", r"\midrule",
    ]
    for r in runs:
        lines.append(f"{_escape(r['pipeline'])} & {_escape(r['label'])} & "
                      f"{_escape(r['status'])} & {_escape(r['started_at'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_per_case_diagnostics(artifacts, project, latex_proj, rules) -> str:
    return _import_figures(artifacts, latex_proj)


def build_raw_data_index(artifacts, project, latex_proj, rules) -> str:
    all_artifacts = project.list_artifacts()
    return _artifact_listing_table(all_artifacts, "Every tracked artifact", "tab:raw_data_index")


def build_provenance(artifacts, project, latex_proj, rules) -> str:
    lines = []
    for run in project.list_runs():
        lines.append(f"\\texttt{{{_escape(run['run_id'])}}}: "
                      f"{_escape(run['pipeline'])}/{_escape(run['label'])}, "
                      f"working\\_dir=\\texttt{{{_escape(run['working_dir'])}}}\\\\")
    return "\n".join(lines) + "\n" if lines else "No provenance to report.\n"


MAIN_SECTIONS: Dict[str, "callable"] = {
    "overview": build_overview,
    "methodology": build_methodology,
    "cost_summary": build_cost_summary,
    "fit_quality": build_fit_quality,
    "key_figures": build_key_figures,
    "outlier_retraining_history": build_outlier_retraining_history,
    "conclusions": build_conclusions,
}

SUPPLEMENTARY_SECTIONS: Dict[str, "callable"] = {
    "full_run_table": build_full_run_table,
    "per_case_diagnostics": build_per_case_diagnostics,
    "raw_data_index": build_raw_data_index,
    "provenance": build_provenance,
}
