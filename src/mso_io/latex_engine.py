"""
mso_io/latex_engine.py
========================
Shared LaTeX->PDF compile ladder: tectonic (single self-contained binary,
auto-fetches packages) > latexmk -pdf > pdflatex (2 passes for cross-
references). Framework-free (stdlib only), used by both
``src/run_report.py`` and ``src/latex_project.py`` -- previously each
hard-required ``pdflatex`` and had no tectonic fallback, so a machine with
only tectonic installed (this one, as of 2026-08 -- ``/home/krithika/
.local/bin/tectonic``) silently produced no PDF from either.

Mirrors the compile ladder already implemented in
``mso_studio.core.reporting._compile`` (the PySide6-side one-click report),
but lives here instead of importing that module directly: ``mso_studio`` is
PySide6-only per the env-isolation policy (dedicated ``.venv-gui/``, never
installed into the shared anaconda env this file's two callers run in).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def compile_latex(
    tex_path: Path, out_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], str, str]:
    """Compile ``tex_path`` to PDF with the best available engine.

    Compiles in place (``cwd=out_dir``, default ``tex_path.parent``) so
    relative ``\\input``/``\\includegraphics`` paths and any sibling
    ``figures/`` directory keep working unchanged.

    Returns ``(pdf_path_or_None, engine_name, log_tail)``. ``engine_name``
    is one of ``"tectonic"``, ``"latexmk"``, ``"pdflatex"``, or ``"none"``
    (no engine found on PATH at all).
    """
    out_dir = Path(out_dir) if out_dir is not None else tex_path.parent
    tex_path = Path(tex_path)
    pdf_path = out_dir / tex_path.with_suffix(".pdf").name

    # Track the last engine actually FOUND and RUN, even if it failed to
    # produce a PDF -- previously, if every found engine failed, this
    # function fell all the way through to `return None, "none", ""`,
    # discarding every engine's real error log and mis-reporting "no LaTeX
    # engine found on PATH" even when one was found and DID run (just
    # failed). Callers (LaTeXProject.compile(), which already branches on
    # `engine == "none"` vs. a real engine name to print the right
    # message) depend on getting the true engine name + log back in this
    # case to be able to show the real failure, not a misleading "not
    # installed" message.
    last_attempt = None  # (engine_name, log_tail)

    tectonic = shutil.which("tectonic")
    if tectonic:
        # Single pass: tectonic resolves cross-references internally.
        log_tail = _run([tectonic, "--outdir", str(out_dir), str(tex_path)], out_dir)
        if pdf_path.exists():
            return pdf_path, "tectonic", log_tail
        last_attempt = ("tectonic", log_tail)

    latexmk = shutil.which("latexmk")
    if latexmk:
        log_tail = _run(
            [latexmk, "-pdf", "-interaction=nonstopmode",
             f"-output-directory={out_dir}", str(tex_path)],
            out_dir,
        )
        if pdf_path.exists():
            return pdf_path, "latexmk", log_tail
        last_attempt = ("latexmk", log_tail)

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        log_tail = ""
        for _pass in range(2):  # 2 passes for cross-references/TOC
            log_tail = _run(
                [pdflatex, "-interaction=nonstopmode",
                 f"-output-directory={out_dir}", str(tex_path)],
                out_dir,
            )
        if pdf_path.exists():
            return pdf_path, "pdflatex", log_tail
        last_attempt = ("pdflatex", log_tail)

    if last_attempt is not None:
        return None, last_attempt[0], last_attempt[1]
    return None, "none", ""


def _run(cmd: list, cwd: Path) -> str:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return ((proc.stdout or "") + (proc.stderr or ""))[-4000:]
