"""
latex_project.py
----------------
Local-git LaTeX project manager for the RMIP / MUQ-SAC workflow.

Features
--------
* Create a new LaTeX project folder with ``git init`` and a starter ``main.tex``.
* Add collaborators by email (stored in ``collaborators.json``; git remote wiring
  is printed as instructions — the lab server push/pull requires SSH setup outside
  this script).
* Compile the project with ``pdflatex`` (two-pass) and open the PDF.
* **Incremental section append** — add a new LaTeX section (e.g. "Sensitivity
  analysis results") to ``main.tex`` without re-running the full report wizard.
  The section is inserted before ``\end{document}``.
* Import a simulation PDF figure directly into the figures directory with a
  ``\includegraphics`` snippet ready to copy.

Usage (standalone)::

    python latex_project.py --new   my_project/  "NH3 Combustion Study"
    python latex_project.py --compile my_project/
    python latex_project.py --append  my_project/ section.tex "SA Results"

Usage (Python API)::

    from latex_project import LaTeXProject
    proj = LaTeXProject("/path/to/project")
    proj.create("NH3 Combustion Study", author="KP")
    proj.append_section("sensitivity", title="Sensitivity Analysis", body=latex_text)
    proj.compile()
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import List, Optional

from mso_io.latex_engine import compile_latex as _compile_latex_engine


def _escape_latex(text) -> str:
    """Minimal LaTeX special-character escaping for values substituted into
    ``_MAIN_TEX_TEMPLATE`` (``title``/``author``) -- deliberately
    self-contained rather than importing ``mso_project.report.sections``'s
    own copy of this same idea, matching that function's own stated
    rationale (this module has no other dependency on that package).

    (Round-2 audit fix, B14 -- WORKBENCH_AUDIT_2026-08-10.md) Neither field
    was escaped before this fix: ``report/builder.py``'s own
    ``author=rules.author or "mso_project"`` fallback put the literal,
    underscore-containing Python package name straight into
    ``\\author{{{author}}}``, and ANY real project title containing a LaTeX
    special character (this repo's own dataset-naming convention is full of
    underscores -- H2_O2_TEST, MB_LTC_no_dummy, ...) would hit the same
    "Missing $ inserted." tectonic failure. Backslash MUST be escaped
    first, same reasoning as the sibling copy: every other replacement
    below inserts a literal backslash, and escaping "\\" afterward would
    re-escape those.
    """
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for ch, esc in repl.items():
        s = s.replace(ch, esc)
    return s


# ---------------------------------------------------------------------------
# Starter template
# ---------------------------------------------------------------------------

_MAIN_TEX_TEMPLATE = r"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

% ── Introduction ─────────────────────────────────────────────────────────
\section{{Introduction}}
Write your introduction here.

% ── END SECTIONS (do not remove this marker line) ──────────────────────
\bibliographystyle{{plainnat}}
\bibliography{{references}}

\end{{document}}
"""

_GITIGNORE = """\
*.aux
*.log
*.out
*.toc
*.lot
*.lof
*.bbl
*.blg
*.synctex.gz
*.fdb_latexmk
*.fls
_minted*
"""

_SECTION_TEMPLATE = r"""
% ── {title} ──────────────────────────────────────────────────────────────
\section{{{title}}}
{body}
"""

_END_MARKER = "% ── END SECTIONS (do not remove this marker line) ──────────────────────"


# ---------------------------------------------------------------------------
# LaTeXProject class
# ---------------------------------------------------------------------------

class LaTeXProject:
    """
    Parameters
    ----------
    project_dir : str | Path
        Root directory of the LaTeX project.
    """

    def __init__(self, project_dir):
        self._root = Path(project_dir).resolve()
        self._main_tex = self._root / "main.tex"
        self._collab_file = self._root / "collaborators.json"
        self._figures_dir = self._root / "figures"

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self, title: str, author: str = "KP", git_init: bool = True) -> None:
        """Create the project directory and scaffold files."""
        self._root.mkdir(parents=True, exist_ok=True)
        self._figures_dir.mkdir(exist_ok=True)

        self._main_tex.write_text(
            _MAIN_TEX_TEMPLATE.format(
                title=_escape_latex(title),
                author=_escape_latex(author),
                date=str(date.today()),
            )
        )
        (self._root / ".gitignore").write_text(_GITIGNORE)
        (self._root / "references.bib").write_text(
            "% Add BibTeX references here\n"
        )
        self._collab_file.write_text(json.dumps([], indent=2))

        if git_init:
            result = subprocess.run(
                ["git", "init"], cwd=self._root, capture_output=True, text=True
            )
            if result.returncode == 0:
                subprocess.run(
                    ["git", "add", "."], cwd=self._root, capture_output=True
                )
                subprocess.run(
                    ["git", "commit", "-m", f"Initial commit: {title}"],
                    cwd=self._root, capture_output=True
                )
                print(f"Git repository initialised in {self._root}")
            else:
                print(f"[git init failed: {result.stderr.strip()}]")

        print(f"Project created: {self._root}")

    # ------------------------------------------------------------------
    # Incremental section append
    # ------------------------------------------------------------------

    def append_section(self, key: str, title: str, body: str) -> None:
        """
        Insert a new LaTeX section before the END SECTIONS marker.

        Parameters
        ----------
        key : str
            Short identifier (used as a comment guard to prevent duplicates).
        title : str
            Section title.
        body : str
            LaTeX content of the section (raw LaTeX).
        """
        if not self._main_tex.exists():
            raise FileNotFoundError(f"main.tex not found in {self._root}")

        src = self._main_tex.read_text()
        guard = f"% section-key: {key}"
        if guard in src:
            print(f"Section '{key}' already present — skipping duplicate insert.")
            return

        new_section = f"{guard}\n" + _SECTION_TEMPLATE.format(title=title, body=body)
        if _END_MARKER not in src:
            # Fallback: insert before \end{document}
            src = src.replace(r"\end{document}", new_section + "\n" + r"\end{document}")
        else:
            src = src.replace(_END_MARKER, new_section + "\n" + _END_MARKER)

        self._main_tex.write_text(src)
        self._git_commit(f"Append section: {title}")
        print(f"Section '{title}' appended to main.tex")

    # ------------------------------------------------------------------
    # Figure import
    # ------------------------------------------------------------------

    def import_figure(self, src_path: str, caption: str = "", label: str = "") -> str:
        """
        Copy a figure file into the project's figures/ directory.

        Returns
        -------
        str
            A ready-to-paste ``\\includegraphics`` LaTeX snippet.
        """
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(src_path)
        dest = self._figures_dir / src.name
        shutil.copy2(src, dest)
        rel = dest.relative_to(self._root)
        stem = src.stem
        lbl = label or f"fig:{stem}"
        snippet = textwrap.dedent(f"""
            \\begin{{figure}}[htbp]
              \\centering
              \\includegraphics[width=0.9\\linewidth]{{figures/{src.name}}}
              \\caption{{{caption or stem}}}
              \\label{{{lbl}}}
            \\end{{figure}}
        """).strip()
        print(f"Figure copied to {dest}")
        print("LaTeX snippet:\n" + snippet)
        return snippet

    # ------------------------------------------------------------------
    # Collaborators
    # ------------------------------------------------------------------

    def add_collaborator(self, name: str, email: str,
                          remote_url: Optional[str] = None) -> None:
        """Register a collaborator and print git remote instructions."""
        collabs = json.loads(self._collab_file.read_text())
        if any(c["email"] == email for c in collabs):
            print(f"{email} already registered.")
            return
        collabs.append({"name": name, "email": email, "remote_url": remote_url or ""})
        self._collab_file.write_text(json.dumps(collabs, indent=2))
        self._git_commit(f"Add collaborator: {name} <{email}>")

        if remote_url:
            print(f"\nTo share with {name}:")
            print(f"  git remote add {name.lower().replace(' ', '_')} {remote_url}")
            print(f"  git push {name.lower().replace(' ', '_')} main")
        else:
            print(f"Collaborator {name} <{email}> registered (no remote URL provided).")
            print("When ready: git remote add <alias> <ssh://server/path/to/repo.git>")

    def list_collaborators(self) -> List[dict]:
        if not self._collab_file.exists():
            return []
        return json.loads(self._collab_file.read_text())

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, open_pdf: bool = True) -> bool:
        """
        Compile ``main.tex`` via the shared tectonic/latexmk/pdflatex ladder
        (mso_io.latex_engine) -- this used to hard-require pdflatex and
        silently fail on a tectonic-only machine (this one, as of 2026-08).

        Returns True if successful.
        """
        pdf, engine, log_tail = _compile_latex_engine(self._root / "main.tex", self._root)

        if pdf is None:
            if engine == "none":
                print("No LaTeX engine found on PATH (tried tectonic/latexmk/pdflatex).")
            else:
                print(f"{engine} errors:")
                for line in log_tail.splitlines():
                    if line.startswith("!") or "Error" in line:
                        print(" ", line)
                print("PDF not generated — check main.tex for errors.")
            return False

        print(f"Compiled ({engine}): {pdf}")
        if open_pdf:
            _open_file(pdf)
        return True

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _git_commit(self, message: str) -> None:
        git = shutil.which("git")
        if git is None:
            return
        subprocess.run(["git", "add", "-A"], cwd=self._root, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self._root, capture_output=True
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_file(path: Path) -> None:
    """Open a file with the system default viewer (xdg-open / open)."""
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener:
        subprocess.Popen([opener, str(path)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local-git LaTeX project manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    s_new = sub.add_parser("new", help="Create a new LaTeX project")
    s_new.add_argument("dir",    help="Project directory path")
    s_new.add_argument("title",  help="Document title")
    s_new.add_argument("--author", default="KP")

    s_compile = sub.add_parser("compile", help="Compile main.tex → PDF")
    s_compile.add_argument("dir")
    s_compile.add_argument("--no-open", action="store_true")

    s_append = sub.add_parser("append", help="Append a section to main.tex")
    s_append.add_argument("dir")
    s_append.add_argument("key",   help="Short unique key (no spaces)")
    s_append.add_argument("title", help="Section title")
    s_append.add_argument("--body-file", help="File containing LaTeX section body")

    s_fig = sub.add_parser("figure", help="Import a figure into the project")
    s_fig.add_argument("dir")
    s_fig.add_argument("src",     help="Path to figure file (PDF/PNG/EPS)")
    s_fig.add_argument("--caption", default="")
    s_fig.add_argument("--label",   default="")

    s_collab = sub.add_parser("collab", help="Add a collaborator")
    s_collab.add_argument("dir")
    s_collab.add_argument("name")
    s_collab.add_argument("email")
    s_collab.add_argument("--remote-url", default=None)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.cmd is None:
        _build_parser().print_help()
        return

    proj = LaTeXProject(args.dir)

    if args.cmd == "new":
        proj.create(args.title, author=args.author)

    elif args.cmd == "compile":
        proj.compile(open_pdf=not args.no_open)

    elif args.cmd == "append":
        if args.body_file:
            body = Path(args.body_file).read_text()
        else:
            print("Enter LaTeX body (Ctrl-D to finish):")
            body = sys.stdin.read()
        proj.append_section(args.key, args.title, body)

    elif args.cmd == "figure":
        proj.import_figure(args.src, caption=args.caption, label=args.label)

    elif args.cmd == "collab":
        proj.add_collaborator(args.name, args.email, remote_url=args.remote_url)


if __name__ == "__main__":
    main()
