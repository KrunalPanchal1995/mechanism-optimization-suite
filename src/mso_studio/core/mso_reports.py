"""
mso_studio.core.mso_reports
==============================
Adapter over the suite's richer, module-specific LaTeX generators -- the
"Export options..." alternative next to the generic one-click report
(:mod:`mso_studio.core.reporting`) that every :class:`ModulePage` already
gets for free.

Currently wired: ``src/target_tables.py::generate_target_tables`` --
publication-quality per-target-type LaTeX tables straight from a
``.target`` file, with an optional ``references.bib`` for citation keys.
Available from any module page that already has a ``.target`` file
selected (nominal sim, mechanism comparison).

**Former gap, discovered 2026-07-21, fixed 2026-07-22:** ``src/run_report.py``
used to import its sibling modules as ``from io.mso_io import ...`` /
``io.mso_analysis`` / ``io.mso_plots`` / ``io.mso_latex``, and ``src/io/``
was a real Python package literally named ``io``, colliding with the
stdlib module under every invocation style. Fixed by renaming the package
to ``src/mso_io/`` and updating ``run_report.py``'s imports to
``mso_io.mso_io`` / ``mso_io.mso_analysis`` / ``mso_io.mso_plots`` /
``mso_io.mso_latex``; verified end-to-end via the ``--config <yaml>``
entry point (added alongside the original discovery) with no remaining
``io`` references anywhere else in the tree.

``generate_target_opt.py``'s existing caller (``_append_mso_report_section``)
still passes CLI args (``meta_path --out-dir out_dir``) that don't match
``run_report.py``'s actual interface (interactive session, or
``--config <yaml>``) -- a separate, still-open, pre-existing bug (predates
both this module and the ``io`` collision) that would need a larger dialog
rewrite in that legacy monolith GUI to collect the full ``cfg`` (stage
configs, CSV paths, etc.) rather than just ``meta_path``/``out_dir``; not
fixed here, out of scope for this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from mso_studio.core.backends import REPO_ROOT, science_command, run_backend
from mso_studio.core.nominal import _build_error

TARGET_TABLES_SCRIPT = REPO_ROOT / "src" / "target_tables.py"


@dataclass
class TargetTablesParams:
    """Inputs for ``target_tables.py`` -- see its module docstring for the
    LaTeX table format (one table per target type: Tig/Fls/JSR/Flf/RCM)."""

    target_csv: str
    bib_path: str = ""
    target_types: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    out_dir: str = ""
    compile_pdf: bool = True

    def __post_init__(self) -> None:
        if not self.out_dir:
            self.out_dir = str(Path(self.target_csv).with_suffix("")) + "_latex_tables"


def generate_target_tables(params: TargetTablesParams) -> Path:
    """Run ``target_tables.py`` and return the path to the combined ``.tex``.

    Raises
    ------
    mso_studio.core.backends.BackendError
        If the backend subprocess exits non-zero.
    FileNotFoundError
        If no ``.tex`` file was produced despite a zero exit code.
    """
    args = [str(Path(params.target_csv).resolve()), "--out", params.out_dir]
    if params.bib_path:
        args += ["--bib", params.bib_path]
    if params.target_types:
        args += ["--types", *params.target_types]
    if params.columns:
        args += ["--cols", *params.columns]
    if not params.compile_pdf:
        args.append("--no-pdf")

    command = science_command("target_tables.py", *args)
    result = run_backend(command, cwd=str(REPO_ROOT))
    result.raise_if_failed()

    out_dir = Path(params.out_dir)
    tex_files = sorted(out_dir.glob("*.tex"))
    if not tex_files:
        raise FileNotFoundError(f"target_tables.py ran but produced no .tex file in {out_dir}")
    # Prefer a "combined" file if present, else the first one found.
    combined = [p for p in tex_files if "combined" in p.stem.lower()]
    return combined[0] if combined else tex_files[0]
