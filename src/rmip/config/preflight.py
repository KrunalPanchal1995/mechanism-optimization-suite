"""
rmip/config/preflight.py
========================
W9a — dependency preflight: ``rmip doctor`` (P38).

Verifies that the toolchain a given task needs is present and reachable
**before** the job starts, and — the actual point — tells the user the exact
remediation when it isn't (which venv, which command). This is the project's
env-isolation map made self-service: every isolated backend venv, external
binary and optional library is probed here.

A green ``rmip doctor --task rmg`` means "this task will not die on a missing
prerequisite"; a red row is a fix-it instruction, not a stack trace three
hours into a run.

W9c: ``rmip doctor --task rmg --rmg-input input.py`` additionally runs the
W7c static input linter (duplicate library, dual-listed seed, inverted
tolerances) so input-file mistakes also surface at onboarding time.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Known locations of the isolated environments / external tools. These mirror
# the defaults hard-wired in the backend bridges (surrogate.py, b2bdc.py,
# zerork_gsa.py, rmg_interface.py) — kept overridable for tests and for
# machines that lay things out differently.
# ---------------------------------------------------------------------------
CHAOSPY_PYTHON = "/home/krithika/.venvs/chaospy_gpc/bin/python3"
CVXPY_PYTHON = "/home/krithika/.venvs/cvxpy_sdp/bin/python3"
SALIB_PYTHON = "/home/krithika/.venvs/salib_gsa/bin/python3"
RMG_PYTHON = "/home/krithika/anaconda3/envs/rmg_env/bin/python"
RMG_REPO = "/data/OPEN_CODE/RMG_py"

_PROBE_TIMEOUT_S = 30.0


@dataclass
class PrereqResult:
    name: str
    label: str
    present: bool
    required: bool
    detail: str = ""                  # version / path when present, why-not when absent
    remediation: str = ""             # exact fix when absent

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class PreflightReport:
    task: str
    results: List[PrereqResult] = field(default_factory=list)
    lint_findings: List[Any] = field(default_factory=list)   # W9c fold-in

    @property
    def ok(self) -> bool:
        prereqs_ok = all(r.present for r in self.results if r.required)
        lint_ok = not any(getattr(f, "severity", "") == "error"
                          for f in self.lint_findings)
        return prereqs_ok and lint_ok

    @property
    def n_missing_required(self) -> int:
        return sum(1 for r in self.results if r.required and not r.present)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"task": self.task, "ok": self.ok,
                             "results": [r.to_dict() for r in self.results]}
        if self.lint_findings:
            d["lint_findings"] = [f.to_dict() for f in self.lint_findings]
        return d

    def to_text(self) -> str:
        lines = ["=" * 72,
                 f"rmip doctor — task: {self.task}",
                 "=" * 72]
        for r in self.results:
            mark = "✓" if r.present else ("✗" if r.required else "–")
            req = "" if r.required else "  (optional)"
            lines.append(f"  {mark} {r.label}{req}")
            if r.detail:
                lines.append(f"      {r.detail}")
            if not r.present and r.remediation:
                lines.append(f"      fix: {r.remediation}")
        for f in self.lint_findings:
            sev = getattr(f, "severity", "info")
            mark = "✗" if sev == "error" else "▲"
            lines.append(f"  {mark} [input {sev}] {getattr(f, 'message', f)}")
        lines.append("")
        lines.append("  verdict: " + (
            "READY — this task will not fail on a missing prerequisite."
            if self.ok else
            f"NOT READY — {self.n_missing_required} required prerequisite(s) "
            "missing" + (" / input errors above" if self.lint_findings else "")
            + "."))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probes — each returns (present, detail)
# ---------------------------------------------------------------------------
def _probe_import(module: str) -> Tuple[bool, str]:
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "")
        return True, f"{module} {ver}".strip() + f" ({sys.executable})"
    except Exception as exc:                               # noqa: BLE001
        return False, f"import failed: {exc}"


def _probe_venv_import(python: str, module: str) -> Tuple[bool, str]:
    """Probe an isolated venv: exists, then can it import its backend module."""
    if not os.path.exists(python):
        return False, f"venv python not found at {python}"
    try:
        r = subprocess.run(
            [python, "-c",
             f"import {module}; print(getattr({module}, '__version__', 'ok'))"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"probe failed: {exc}"
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        reason = tail[-1] if tail else "import probe failed"
        return False, f"venv exists but '{module}' broken: {reason}"
    return True, f"{module} {r.stdout.strip()} ({python})"


def _probe_binary(*names: str) -> Tuple[bool, str]:
    for n in names:
        path = shutil.which(n)
        if path:
            return True, path
    return False, f"none of {list(names)} on PATH"


def _probe_ck2yaml() -> Tuple[bool, str]:
    path = shutil.which("ck2yaml")
    if path:
        return True, path
    ok, _ = _probe_import("cantera")
    if ok:
        return True, f"python -m cantera.ck2yaml ({sys.executable})"
    return False, "no ck2yaml on PATH and cantera not importable"


def _probe_path(path: str, kind: str = "dir") -> Tuple[bool, str]:
    ok = os.path.isdir(path) if kind == "dir" else os.path.exists(path)
    return ok, path if ok else f"not found: {path}"


def _probe_latex_engine() -> Tuple[bool, str]:
    """Same tectonic > latexmk > pdflatex ladder as
    mso_io.latex_engine.compile_latex, checked here as a presence probe
    (not a reimplementation of the compile logic -- just 'is at least one
    of these on PATH')."""
    for name in ("tectonic", "latexmk", "pdflatex"):
        path = shutil.which(name)
        if path:
            return True, f"{name} ({path})"
    return False, "none of tectonic/latexmk/pdflatex on PATH"


# name -> (label, required-by-default, probe factory, remediation)
# Probe factories are zero-arg lambdas so the module-level path constants can
# be monkeypatched (tests) before any probe runs.
_PREREQS: Dict[str, Tuple[str, Callable[[], Tuple[bool, str]], str]] = {
    "cantera": ("Cantera (shared anaconda env)",
                lambda: _probe_import("cantera"),
                "install Cantera in the shared env — but check numpy pins "
                "first (see feedback-env-isolation): conda install -c cantera cantera"),
    "numpy": ("NumPy", lambda: _probe_import("numpy"),
              "pip install numpy==1.24.4 (the shared-env pinned version)"),
    "yaml": ("PyYAML", lambda: _probe_import("yaml"), "pip install pyyaml"),
    "networkx": ("NetworkX (graph analysis)", lambda: _probe_import("networkx"),
                 "pip install networkx"),
    "tkinter": ("Tkinter (GUI apps)", lambda: _probe_import("tkinter"),
                "install the OS package python3-tk"),
    "ck2yaml": ("ck2yaml (CHEMKIN → YAML converter)", _probe_ck2yaml,
                "comes with Cantera; ensure the cantera env's bin is on PATH "
                "or use: python -m cantera.ck2yaml"),
    "chaospy_venv": ("chaospy venv (gPC surrogate / UQ backend)",
                     lambda: _probe_venv_import(CHAOSPY_PYTHON, "chaospy"),
                     f"create it: python3 -m venv {os.path.dirname(os.path.dirname(CHAOSPY_PYTHON))} "
                     "&& pip install -r /home/krithika/.venvs/chaospy_gpc/requirements.txt "
                     "(pins: chaospy==4.3.21 numpoly==1.2.12 numpy==1.26.4 — "
                     "do NOT let pip pick numpoly 1.3.x, it needs numpy 2.1)"),
    "cvxpy_venv": ("cvxpy venv (B2BDC SDP outer bounds)",
                   lambda: _probe_venv_import(CVXPY_PYTHON, "cvxpy"),
                   f"python3 -m venv {os.path.dirname(os.path.dirname(CVXPY_PYTHON))} "
                   "&& <venv>/bin/pip install cvxpy"),
    "salib_venv": ("SALib venv (Zero-RK GSA backend)",
                   lambda: _probe_venv_import(SALIB_PYTHON, "SALib"),
                   f"python3 -m venv {os.path.dirname(os.path.dirname(SALIB_PYTHON))} "
                   "&& <venv>/bin/pip install SALib "
                   "(NEVER into the shared anaconda env — it upgrades numpy and "
                   "breaks sklearn's ABI)"),
    "rmg_python": ("RMG-Py conda env",
                   lambda: _probe_path(RMG_PYTHON, "file"),
                   "create the rmg_env conda environment per "
                   f"{RMG_REPO}/environment.yml: conda env create -f environment.yml"),
    "rmg_repo": ("RMG-Py source checkout",
                 lambda: _probe_path(RMG_REPO, "dir"),
                 f"git clone https://github.com/ReactionMechanismGenerator/RMG-Py {RMG_REPO}"),
    "gaussian": ("Gaussian (RMG QM thermo, optional)",
                 lambda: _probe_binary("g16", "g09"),
                 "needed only for RMG quantumMechanics blocks; install Gaussian "
                 "or remove the QM section from input.py"),
    "zerork": ("Zero-RK native solver",
               lambda: _probe_binary("constVolumeWSR.x", "constVolumeWSR_mpi.x"),
               "point --zerork-install-dir at a Zero-RK build (bin/ must hold "
               "constVolumeWSR.x); only needed for the zerork-backed gsa-run"),
    # -- Research helper packages (2026-08 workbench audit, gap #4) --
    # rdkit/pysr/pdfplumber/google-genai all already live in the shared
    # anaconda env per mso_studio.core.tools's own docstring; probed here
    # the same way cantera is (plain in-process import, since this module
    # itself only ever runs under SCIENCE_PYTHON via rmip_cli.py doctor).
    "rdkit": ("RDKit (cheminformatics / SMILES)", lambda: _probe_import("rdkit"),
              "conda install -c conda-forge rdkit  (into the shared anaconda env)"),
    "pysr": ("PySR (symbolic regression, needs Julia)", lambda: _probe_import("pysr"),
             "pip install pysr  (into the shared anaconda env; also needs a "
             "Julia install -- see PySR's own setup docs)"),
    "jupyter": ("Jupyter (notebook server)", lambda: _probe_binary("jupyter"),
                "conda install jupyter  (into the shared anaconda env)"),
    "latex_engine": ("LaTeX engine (tectonic / latexmk / pdflatex)",
                      _probe_latex_engine,
                      "install tectonic (single binary, easiest: "
                      "https://tectonic-typesetting.github.io) or a system "
                      "LaTeX distribution (texlive-latex-extra or similar)"),
    "pdfplumber": ("pdfplumber (PDF text/table extraction)",
                   lambda: _probe_import("pdfplumber"),
                   "pip install pdfplumber  (into the shared anaconda env)"),
    "google_genai": ("google-genai (Gemini PDF Q&A, optional)",
                      lambda: _probe_import("google.genai"),
                      "pip install google-genai  (into the shared anaconda env); "
                      "also needs GOOGLE_API_KEY set"),
}

# Task → (required prereqs, optional prereqs). "all" is synthesised below.
TASKS: Dict[str, Tuple[List[str], List[str]]] = {
    "check":      (["cantera", "yaml", "numpy", "ck2yaml"], []),
    "convert":    (["cantera", "ck2yaml"], []),
    "uq":         (["cantera", "numpy", "chaospy_venv"], []),
    "gsa":        (["cantera", "salib_venv"], ["zerork"]),
    "b2bdc":      (["cantera", "cvxpy_venv"], []),
    "rmg":        (["rmg_python", "rmg_repo"], ["gaussian"]),
    "regression": (["cantera", "numpy"], []),
    "reduce":     (["cantera", "networkx"], []),
    "gui":        (["tkinter", "cantera"], []),
    # research helper packages a from-scratch combustion-mechanism workflow
    # needs (plot digitizer is stdlib-only -- csv/io/xml -- so not listed):
    # RMG + RDKit (task 4's explicit asks), PySR, Jupyter, a LaTeX engine,
    # PDF tooling.
    "helper_packages": (["jupyter", "latex_engine"],
                         ["rdkit", "pysr", "pdfplumber", "google_genai",
                          "rmg_python", "rmg_repo"]),
}
TASKS["all"] = (sorted({n for req, _ in TASKS.values() for n in req}),
                sorted({n for _, opt in TASKS.values() for n in opt}))


def check_environment(task: str = "all",
                      rmg_input: Optional[str] = None) -> PreflightReport:
    """Probe every prerequisite the given task needs; never raises.

    Parameters
    ----------
    task : str
        One of :data:`TASKS` (default ``"all"``).
    rmg_input : str, optional
        Path to an RMG ``input.py``; when given (task ``rmg``/``all``), the
        W7c static linter runs too (W9c) so input mistakes surface now.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}' — one of {sorted(TASKS)}")
    required, optional = TASKS[task]
    report = PreflightReport(task=task)
    for name in list(required) + [n for n in optional if n not in required]:
        label, probe, remediation = _PREREQS[name]
        try:
            present, detail = probe()
        except Exception as exc:                           # noqa: BLE001
            present, detail = False, f"probe raised: {exc}"
        report.results.append(PrereqResult(
            name=name, label=label, present=present,
            required=name in required, detail=detail,
            remediation="" if present else remediation))

    if rmg_input:
        try:
            from rmip.generation.rmg_lint import lint_rmg_input
            report.lint_findings = lint_rmg_input(rmg_input)
        except FileNotFoundError:
            from rmip.analysis.mechanism_check import Finding
            report.lint_findings = [Finding(
                check="rmg_lint", severity="error", category="missing-file",
                message=f"RMG input file not found: {rmg_input}")]
    return report


def report_json(report: PreflightReport) -> str:
    return json.dumps(report.to_dict(), indent=2, default=str)
