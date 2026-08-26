"""
mso_studio.core.reporting
===========================
The one-click LaTeX report: ``RunRecord`` -> ``.tex`` (Jinja2) -> PDF, with a
three-rung graceful-degradation compile ladder so the feature never dead-ends
on a machine without a LaTeX toolchain (guidelines doc, Part 4.4):

1. **tectonic** -- a single self-contained binary, auto-fetches packages.
2. **System LaTeX** -- ``latexmk -pdf``, falling back to two ``pdflatex`` passes.
3. **No engine available** -- write ``report.tex`` + ``figures/`` into a zip;
   the caller can compile it locally or upload it to Overleaf.

This module is also the integration point for the suite's existing LaTeX
generators (``src/mso_io/mso_latex.py``, ``src/run_report.py``,
``src/latex_project.py``, ``src/target_tables.py``) -- those remain the
richer, module-specific report builders; :func:`build_report` is the
*generic* one-click path available from every ``mso_studio`` module via the
same button and worker.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jinja2
import yaml

from mso_studio.core.runrecord import RunRecord

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

LATEX_SPECIALS = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def tex_escape(value) -> str:
    """Escape LaTeX special characters. Registered as the Jinja2 ``e`` filter.

    Applied to every user-supplied string in the template -- file names
    containing ``_`` or ``%`` are the classic way a one-click report breaks.
    """
    return "".join(LATEX_SPECIALS.get(c, c) for c in str(value))


def _jinja_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        block_start_string=r"\BLOCK{", block_end_string="}",
        variable_start_string=r"\VAR{", variable_end_string="}",
        comment_start_string=r"\#{", comment_end_string="}",
        autoescape=False, trim_blocks=True, lstrip_blocks=True,
    )


@dataclass
class ReportResult:
    """Outcome of :func:`build_report`.

    ``engine`` is one of ``"tectonic"``, ``"latexmk"``, ``"pdflatex"``, or
    ``"bundle"`` (rung 3 -- no PDF, a ``.tex`` + figures zip instead).
    """

    tex_path: Path
    pdf_path: Optional[Path]
    bundle_path: Optional[Path]
    engine: str
    log_tail: str = ""

    @property
    def succeeded_as_pdf(self) -> bool:
        return self.pdf_path is not None


def build_report(record: RunRecord, out_dir: str | Path,
                  template_name: str = "report.tex.j2") -> ReportResult:
    """Render *record* to LaTeX and attempt to compile it to PDF.

    Parameters
    ----------
    record : RunRecord
        The run to report on -- typically ``self.last_run`` in the GUI.
    out_dir : path
        Directory the final ``report.tex`` (and ``report.pdf`` or
        ``report_bundle.zip``) are written into. Created if missing.
    template_name : str
        Template filename under ``mso_studio/templates/``.

    Returns
    -------
    ReportResult
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    figure_paths = {}
    for name, src_path in record.figures.items():
        src = Path(src_path)
        if src.exists() and src.suffix.lower() == ".pdf":
            dest_name = _safe_filename(name) + ".pdf"
            shutil.copy2(src, figures_dir / dest_name)
            figure_paths[name] = f"figures/{dest_name}"

    env = _jinja_env()
    env.filters["e"] = tex_escape
    # Precompute scalar-only results here rather than testing types in the
    # template (Jinja's `is mapping`/`is sequence` tests are not guaranteed
    # across versions) -- a template render failure would be a hard dead end
    # for the "never fails" one-click report promise.
    simple_results = {
        k: v for k, v in record.results.items()
        if v is None or isinstance(v, (str, int, float, bool))
    }
    try:
        template = env.get_template(template_name)
        tex_source = template.render(
            record=record,
            config_yaml=yaml.safe_dump(record.config.to_dict(), sort_keys=False),
            figure_paths=figure_paths,
            simple_results=simple_results,
        )
    except jinja2.TemplateError as exc:
        # Template rendering must never be a dead end (guidelines doc, 4.4) --
        # fall back to a minimal hand-built .tex that at least records what
        # happened and dumps the raw config as verbatim YAML.
        tex_source = _minimal_fallback_tex(record, exc)

    tex_path = out_dir / "report.tex"
    tex_path.write_text(tex_source)

    pdf_path, engine, log_tail = _compile(tex_path, out_dir)
    bundle_path = None
    if pdf_path is None:
        bundle_path = _make_bundle(out_dir, tex_path, figures_dir)

    return ReportResult(
        tex_path=tex_path, pdf_path=pdf_path, bundle_path=bundle_path,
        engine=engine, log_tail=log_tail,
    )


def _minimal_fallback_tex(record: RunRecord, exc: Exception) -> str:
    """Hand-built .tex used only if the Jinja2 template itself fails to render."""
    return (
        "\\documentclass{article}\n\\usepackage{fancyvrb}\n"
        "\\title{MSO Studio Run Report (fallback)}\n\\begin{document}\\maketitle\n"
        f"\\section{{Notice}}\nThe report template failed to render ({tex_escape(str(exc))}); "
        "this is a minimal fallback. See the log for details.\n"
        f"\\section{{Run}}\nModule: {tex_escape(record.config.module)} \\\\ "
        f"Label: {tex_escape(record.config.label)} \\\\ Status: {tex_escape(record.status)}\n"
        "\\section{Configuration}\n\\begin{Verbatim}\n"
        f"{yaml.safe_dump(record.config.to_dict(), sort_keys=False)}\n"
        "\\end{Verbatim}\n\\end{document}\n"
    )


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _compile(tex_path: Path, out_dir: Path) -> tuple[Optional[Path], str, str]:
    """Rungs 1-2 of the compile ladder. Always compiles in a temp dir and
    copies only the final PDF back, keeping the ``.log`` on failure."""
    tectonic = shutil.which("tectonic")
    if tectonic:
        pdf, log_tail = _run_compiler(
            [tectonic, "--outdir", "{outdir}", "{tex}"], tex_path, out_dir,
        )
        if pdf:
            return pdf, "tectonic", log_tail

    latexmk = shutil.which("latexmk")
    if latexmk:
        pdf, log_tail = _run_compiler(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-output-directory={outdir}", "{tex}"],
            tex_path, out_dir,
        )
        if pdf:
            return pdf, "latexmk", log_tail

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        log_tail = ""
        for _pass in range(2):  # two passes for cross-references/TOC
            pdf, log_tail = _run_compiler(
                [pdflatex, "-interaction=nonstopmode", "-output-directory={outdir}", "{tex}"],
                tex_path, out_dir,
            )
        if pdf:
            return pdf, "pdflatex", log_tail

    return None, "none", ""


def _run_compiler(cmd_template: list[str], tex_path: Path, out_dir: Path) -> tuple[Optional[Path], str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_tex = tmp_path / tex_path.name
        shutil.copy2(tex_path, tmp_tex)
        figures_src = tex_path.parent / "figures"
        if figures_src.is_dir():
            shutil.copytree(figures_src, tmp_path / "figures")

        cmd = [c.format(outdir=str(tmp_path), tex=str(tmp_tex)) for c in cmd_template]
        try:
            proc = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, str(exc)

        log_tail = (proc.stdout or "") + (proc.stderr or "")
        produced_pdf = tmp_tex.with_suffix(".pdf")
        if produced_pdf.exists():
            final_pdf = out_dir / "report.pdf"
            shutil.copy2(produced_pdf, final_pdf)
            return final_pdf, log_tail[-4000:]
        return None, log_tail[-4000:]


def _make_bundle(out_dir: Path, tex_path: Path, figures_dir: Path) -> Path:
    """Rung 3: zip ``report.tex`` + ``figures/`` so the feature never dead-ends."""
    bundle_path = out_dir / "report_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tex_path, arcname=tex_path.name)
        if figures_dir.is_dir():
            for fig in figures_dir.glob("*"):
                zf.write(fig, arcname=f"figures/{fig.name}")
    return bundle_path
