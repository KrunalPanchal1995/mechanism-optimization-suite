"""
mso_project.report.builder
=============================
``build_report(project, rules, out_dir)`` -- the code that converts a
project's tracked artifacts into a compiled LaTeX main report +
supplementary document + a supplementary Jupyter notebook + a
machine-readable manifest, following ``ReportRules``'s routing.

Routing precedence, applied per artifact:
  1. Its own ``report_include`` column, if not ``"auto"`` (explicit
     per-artifact override -- always wins).
  2. ``rules.routing_for(role, kind)`` (role-routing, else kind fallback).
An artifact routed ``"exclude"`` is left out of the report but is never
touched/unregistered in the project -- routing only affects report
content, never the underlying artifact index.

Each report build is a fresh, reproducible regeneration (not an
incremental diary) -- re-running this against the same rules and the same
tracked artifacts produces the same document, which is the whole point of
keeping the rules in a versioned YAML file rather than re-asking a wizard
every time.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from latex_project import LaTeXProject  # noqa: E402
from mso_io.notebook_gen import (  # noqa: E402
    add_code_cell, add_markdown_cell, new_notebook, save_notebook,
)

from .rules import ReportRules
from .sections import MAIN_SECTIONS, SUPPLEMENTARY_SECTIONS, _escape


@dataclass
class ReportResult:
    main_tex: str
    main_pdf: Optional[str]
    supplementary_tex: str
    supplementary_pdf: Optional[str]
    notebook_path: str
    manifest_path: str
    readme_path: str
    compiled: bool


def _humanize(key: str) -> str:
    return key.replace("_", " ").title()


def _route(artifact: dict, rules: ReportRules) -> str:
    explicit = artifact.get("report_include", "auto")
    if explicit != "auto":
        return explicit
    return rules.routing_for(artifact["role"], artifact["kind"])


def _build_notebook(project, main_artifacts: List[dict],
                     supp_artifacts: List[dict], out_path: str) -> str:
    """A generic supplementary notebook: one markdown overview cell, then
    one code cell per tracked CSV that loads it via pandas for interactive
    exploration -- not mechanism-specific like notebook_gen's higher-level
    build_mechanism_notebook, just this module's own use of the same
    low-level cell primitives."""
    nb = new_notebook()
    add_markdown_cell(nb, (
        f"# {project.info.get('name', 'Project')} — Supplementary Data\n\n"
        f"Generated {datetime.now(timezone.utc).isoformat()}Z. Loads every "
        f"tracked CSV artifact for interactive exploration."))
    add_code_cell(nb, "import pandas as pd\nfrom pathlib import Path")
    for a in main_artifacts + supp_artifacts:
        if a["kind"] != "csv":
            continue
        var = "df_" + "".join(c if c.isalnum() else "_" for c in Path(a["path"]).stem)
        add_markdown_cell(nb, f"## {a['role']}: `{Path(a['path']).name}`")
        add_code_cell(nb, f"{var} = pd.read_csv(r{a['path']!r})\n{var}.head()")
    return save_notebook(nb, out_path)


def build_report(project, rules: Optional[ReportRules] = None,
                  out_dir: Optional[str] = None) -> ReportResult:
    """Build the main report + supplementary document + notebook for
    ``project``. ``rules`` defaults to ``<project.root>/report_rules.yaml``
    if present, else the coded ``DEFAULT_RULES``. ``out_dir`` defaults to
    ``<project.root>/reports/<timestamp>/``."""
    if rules is None:
        rules = ReportRules.load_or_default(str(project.root / "report_rules.yaml"))

    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = str(project.root / "reports" / stamp)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_artifacts = project.list_artifacts()
    main_artifacts, supp_artifacts, excluded = [], [], []
    for a in all_artifacts:
        route = _route(a, rules)
        if route == "main":
            main_artifacts.append(a)
        elif route == "supplementary":
            supp_artifacts.append(a)
        else:
            excluded.append(a)

    # -- main report --------------------------------------------------------
    main_dir = out_root / "main"
    main_proj = LaTeXProject(str(main_dir))
    main_proj.create(rules.title, author=rules.author or "MSO Studio", git_init=False)
    for key in rules.sections_main:
        builder_fn = MAIN_SECTIONS.get(key)
        if builder_fn is None:
            print(f"[mso_project.report] warning: no section builder for "
                  f"main section {key!r} -- skipped.", file=sys.stderr)
            continue
        body = builder_fn(main_artifacts, project, main_proj, rules)
        main_proj.append_section(key, _humanize(key), body)
    if rules.include_notes:
        notes = [a for a in all_artifacts if a["role"] == "user_note"]
        if notes:
            # Notes are free-form user Markdown (add_note() writes "# Title\n\n
            # body"), NOT LaTeX -- a raw, unescaped '#' alone is a fatal LaTeX
            # error (it's TeX's macro-parameter-text character), and any other
            # special character in a user's own words (_, %, $, &, ...) would
            # be just as fatal. verbatim renders it safely and honestly, as
            # what it is: freeform user text, not typeset prose.
            blocks = []
            for n in notes:
                text = Path(n["path"]).read_text()
                blocks.append(f"\\subsection{{{_escape(Path(n['path']).stem)}}}\n"
                               f"\\begin{{verbatim}}\n{text}\\end{{verbatim}}")
            main_proj.append_section("user_notes", "User Notes", "\n\n".join(blocks))

    # -- supplementary document -----------------------------------------
    supp_dir = main_dir / "supplementary"
    supp_proj = LaTeXProject(str(supp_dir))
    supp_proj.create(f"{rules.title} -- Supplementary", author=rules.author or "MSO Studio",
                      git_init=False)
    for key in rules.sections_supplementary:
        builder_fn = SUPPLEMENTARY_SECTIONS.get(key)
        if builder_fn is None:
            print(f"[mso_project.report] warning: no section builder for "
                  f"supplementary section {key!r} -- skipped.", file=sys.stderr)
            continue
        body = builder_fn(supp_artifacts, project, supp_proj, rules)
        supp_proj.append_section(key, _humanize(key), body)

    # -- notebook -------------------------------------------------------
    notebook_path = _build_notebook(project, main_artifacts, supp_artifacts,
                                     str(main_dir / "supplementary_data.ipynb"))

    # -- compile ----------------------------------------------------------
    main_ok = main_proj.compile(open_pdf=False)
    supp_ok = supp_proj.compile(open_pdf=False)
    main_pdf = str(main_dir / "main.pdf") if main_ok and (main_dir / "main.pdf").exists() else None
    supp_pdf = str(supp_dir / "main.pdf") if supp_ok and (supp_dir / "main.pdf").exists() else None

    # -- manifest + README (matches the report/results/README.md +
    #    manifest.json convention this session's research found in the
    #    related external report/ tree) -----------------------------------
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project.info.get("name"),
        "rules_version": rules.rules_version,
        "n_artifacts_main": len(main_artifacts),
        "n_artifacts_supplementary": len(supp_artifacts),
        "n_artifacts_excluded": len(excluded),
        "main_tex": str(main_dir / "main.tex"),
        "main_pdf": main_pdf,
        "supplementary_tex": str(supp_dir / "main.tex"),
        "supplementary_pdf": supp_pdf,
        "notebook": notebook_path,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    readme_path = out_root / "README.md"
    readme_path.write_text(
        f"# {rules.title}\n\n"
        f"Generated {manifest['generated_at']} from project "
        f"`{project.info.get('name')}`.\n\n"
        f"- Main report: `main/main.tex`"
        + (f" (compiled: `main/main.pdf`)" if main_pdf else " (not compiled -- no LaTeX engine found)")
        + f"\n- Supplementary: `main/supplementary/main.tex`"
        + (f" (compiled: `main/supplementary/main.pdf`)" if supp_pdf else " (not compiled)")
        + f"\n- Supplementary data notebook: `main/supplementary_data.ipynb`\n\n"
        f"{len(main_artifacts)} artifact(s) routed to main, "
        f"{len(supp_artifacts)} to supplementary, {len(excluded)} excluded. "
        f"See `manifest.json` for the machine-readable summary.\n")

    return ReportResult(
        main_tex=str(main_dir / "main.tex"), main_pdf=main_pdf,
        supplementary_tex=str(supp_dir / "main.tex"), supplementary_pdf=supp_pdf,
        notebook_path=notebook_path, manifest_path=str(manifest_path),
        readme_path=str(readme_path), compiled=bool(main_ok and supp_ok))
