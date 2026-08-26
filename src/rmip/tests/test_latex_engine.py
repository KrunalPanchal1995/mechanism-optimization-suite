"""
rmip/tests/test_latex_engine.py
=================================
Regression coverage for the shared tectonic/latexmk/pdflatex compile ladder
(Phase 4, 2026-08 backlog program, bug.md): src/run_report.py and
src/latex_project.py both used to hard-require pdflatex and silently
produce no PDF on a tectonic-only machine (this one, as of 2026-08 --
/home/krithika/.local/bin/tectonic; no pdflatex/latexmk installed). Both now
go through mso_io.latex_engine.compile_latex, mirroring the ladder already
used by mso_studio.core.reporting (the PySide6-side one-click report).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_HAS_TECTONIC = shutil.which("tectonic") is not None
_HAS_ANY_ENGINE = _HAS_TECTONIC or shutil.which("latexmk") or shutil.which("pdflatex")

_MINIMAL_TEX = r"""
\documentclass{article}
\begin{document}
Hello, world.
\end{document}
"""


@pytest.mark.skipif(not _HAS_ANY_ENGINE, reason="no LaTeX engine on PATH")
class TestCompileLatex:
    def test_compiles_a_real_pdf(self, tmp_path):
        from mso_io.latex_engine import compile_latex

        tex_path = tmp_path / "doc.tex"
        tex_path.write_text(_MINIMAL_TEX)

        pdf, engine, log_tail = compile_latex(tex_path, tmp_path)

        assert pdf is not None, log_tail
        assert pdf.exists()
        assert pdf.read_bytes().startswith(b"%PDF")
        assert engine in ("tectonic", "latexmk", "pdflatex")

    @pytest.mark.skipif(not _HAS_TECTONIC, reason="tectonic specifically not on PATH")
    def test_prefers_tectonic_when_available(self, tmp_path):
        from mso_io.latex_engine import compile_latex

        tex_path = tmp_path / "doc.tex"
        tex_path.write_text(_MINIMAL_TEX)

        pdf, engine, _ = compile_latex(tex_path, tmp_path)
        assert engine == "tectonic"
        assert pdf is not None

    def test_no_engine_found_returns_none_gracefully(self, tmp_path, monkeypatch):
        """Simulate a machine with nothing installed -- must return
        (None, "none", "") rather than raising."""
        from mso_io import latex_engine

        monkeypatch.setattr(latex_engine.shutil, "which", lambda name: None)
        tex_path = tmp_path / "doc.tex"
        tex_path.write_text(_MINIMAL_TEX)

        pdf, engine, log_tail = latex_engine.compile_latex(tex_path, tmp_path)
        assert pdf is None
        assert engine == "none"
        assert log_tail == ""


@pytest.mark.skipif(not _HAS_ANY_ENGINE, reason="no LaTeX engine on PATH")
class TestRunReportUsesSharedEngine:
    def test_compile_latex_produces_real_pdf(self, tmp_path, capsys):
        import run_report

        tex_path = tmp_path / "mso_report.tex"
        tex_path.write_text(_MINIMAL_TEX)

        run_report.compile_latex(tex_path, tmp_path)

        pdf = tmp_path / "mso_report.pdf"
        assert pdf.exists()
        assert pdf.read_bytes().startswith(b"%PDF")
        captured = capsys.readouterr()
        assert "PDF compiled" in captured.out


@pytest.mark.skipif(not _HAS_ANY_ENGINE, reason="no LaTeX engine on PATH")
class TestLatexProjectUsesSharedEngine:
    def test_compile_produces_real_pdf(self, tmp_path):
        from latex_project import LaTeXProject

        proj = LaTeXProject(str(tmp_path))
        proj.create("Test Project", author="pytest")
        ok = proj.compile(open_pdf=False)

        assert ok is True
        pdf = tmp_path / "main.pdf"
        assert pdf.exists()
        assert pdf.read_bytes().startswith(b"%PDF")
