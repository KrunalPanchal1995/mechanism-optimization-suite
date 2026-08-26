"""
mso_studio.core.tools
========================
Adapters for the Tools module: polynomial/symbolic curve fitting, and
Gemini-backed PDF Q&A / table extraction. Ported from the Tkinter
``src/notebook_gemini/`` panels (``pysr_panel.py``, ``pdf_panel.py``,
``pdf_table_panel.py``) -- these were the only place PySR, PDF Q&A, and PDF
table extraction lived, reachable solely from the ``generate_target_opt.py``
monolith's "Notebook" menu.

Dependency routing: PySR (needs Julia), ``pdfplumber``, and ``google-genai``
all already live in the shared anaconda environment (installed there before
this integration, for the original Tkinter panels) -- never added to the
light GUI venv. Every operation that needs them shells out to
``scripts/tools_worker.py`` via :func:`~mso_studio.core.backends.run_backend`,
same pattern as every other ``core`` adapter. Polynomial fitting is pure
numpy and runs in-process.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON, run_backend

WORKER_SCRIPT = REPO_ROOT / "scripts" / "tools_worker.py"


def gemini_available() -> bool:
    """Cheap upfront check for enabling/disabling Gemini buttons -- the real
    availability (key valid + package importable) is still checked at call
    time in the anaconda subprocess, same graceful-degradation contract as
    the original ``GeminiClient.available`` flag."""
    return bool(os.environ.get("GOOGLE_API_KEY"))


def _call_worker(
    mode: str,
    request: dict,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout: Optional[float] = None,
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / "request.json"
        resp_path = Path(tmp) / "response.json"
        req_path.write_text(json.dumps(request))
        command = [SCIENCE_PYTHON, str(WORKER_SCRIPT), mode, str(req_path), str(resp_path)]
        run_backend(command, on_line=on_line, cancel_event=cancel_event, timeout=timeout)

        if not resp_path.exists():
            raise RuntimeError(f"'{mode}' worker produced no response (check the log).")
        data = json.loads(resp_path.read_text())
        if not data.get("ok"):
            raise RuntimeError(data.get("error", f"'{mode}' failed with no error message."))
        return data["result"]


# ---------------------------------------------------------------------------
# Polynomial fit (pure numpy, in-process)
# ---------------------------------------------------------------------------

@dataclass
class PolyFitResult:
    degree: int
    coefficients: List[float]  # high -> low degree, matches numpy.polyfit order
    rmse: float
    latex_equation: str
    x_fine: List[float]
    y_fine: List[float]


def poly_fit(x: np.ndarray, y: np.ndarray, degree: int) -> PolyFitResult:
    """Least-squares polynomial fit with an RMSE and a ready-to-use LaTeX
    equation string -- ported verbatim from ``pysr_panel.py::_poly_fit``."""
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    y_fit = p(x)
    rmse = float(np.sqrt(np.mean((y - y_fit) ** 2)))

    terms = []
    for i, c in enumerate(coeffs):
        exp = degree - i
        if abs(c) < 1e-15:
            continue
        if exp == 0:
            terms.append(f"{c:.4g}")
        elif exp == 1:
            terms.append(f"{c:.4g}x")
        else:
            terms.append(f"{c:.4g}x^{{{exp}}}")
    latex_eq = " + ".join(terms).replace("+ -", "- ")

    x_fine = np.linspace(x.min(), x.max(), 300)
    y_fine = p(x_fine)
    return PolyFitResult(
        degree=degree, coefficients=[float(c) for c in coeffs], rmse=rmse,
        latex_equation=latex_eq, x_fine=x_fine.tolist(), y_fine=y_fine.tolist(),
    )


# ---------------------------------------------------------------------------
# PySR symbolic regression (subprocess -> anaconda, needs Julia)
# ---------------------------------------------------------------------------

@dataclass
class PySREquation:
    complexity: int
    loss: float
    equation: str


@dataclass
class PySRFitResult:
    equations: List[PySREquation]
    best_equation: str
    x_fine: List[float]
    y_fine: List[float]


def pysr_fit(
    x: np.ndarray, y: np.ndarray, n_iter: int = 40,
    *, on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PySRFitResult:
    """Symbolic regression via PySR, run in the anaconda environment (needs
    Julia). Note on cancellation: PySR's underlying Julia ``.fit()`` call
    cannot be cooperatively interrupted -- ``cancel_event`` here can only
    terminate the OS subprocess outright (same honest limitation the
    original Tkinter panel documented: "a background PySR run may still be
    finishing" is not possible with an in-process thread, but killing the
    subprocess *does* fully stop the computation, which is actually a
    strict improvement over the original).
    """
    result = _call_worker(
        "pysr_fit", {"X": np.asarray(x).tolist(), "Y": np.asarray(y).tolist(), "n_iter": n_iter},
        on_line=on_line, cancel_event=cancel_event, timeout=3600.0,
    )
    return PySRFitResult(
        equations=[PySREquation(**e) for e in result["equations"]],
        best_equation=result["best_equation"],
        x_fine=result["x_fine"], y_fine=result["y_fine"],
    )


# ---------------------------------------------------------------------------
# Gemini PDF Q&A / table cleanup (subprocess -> anaconda)
# ---------------------------------------------------------------------------

def gemini_chat(prompt: str, context: str = "") -> str:
    result = _call_worker("gemini_chat", {"prompt": prompt, "context": context})
    return result["reply"]


def gemini_chat_with_pdf(pdf_path: str, question: str) -> str:
    result = _call_worker("gemini_chat_pdf", {"pdf_path": pdf_path, "question": question})
    return result["reply"]


def gemini_chat_with_image(image_path: str, question: str, mime: str = "image/png") -> str:
    result = _call_worker("gemini_chat_image", {"image_path": image_path, "mime": mime, "question": question})
    return result["reply"]


def gemini_summarise_pdf(pdf_path: str) -> str:
    result = _call_worker("gemini_summarise_pdf", {"pdf_path": pdf_path})
    return result["reply"]


def gemini_clean_table_csv(csv_text: str) -> str:
    result = _call_worker("gemini_clean_table", {"csv_text": csv_text})
    return result["reply"]


# ---------------------------------------------------------------------------
# PDF text / table extraction (subprocess -> anaconda, needs pdfplumber)
# ---------------------------------------------------------------------------

@dataclass
class PdfTextResult:
    n_pages: int
    text: str


def pdf_extract_text(pdf_path: str) -> PdfTextResult:
    result = _call_worker("pdf_extract_text", {"pdf_path": pdf_path})
    return PdfTextResult(n_pages=result["n_pages"], text=result["text"])


@dataclass
class PdfTablesResult:
    tables: List[List[List[str]]]
    page_nums: List[int]


def pdf_extract_tables(pdf_path: str) -> PdfTablesResult:
    result = _call_worker("pdf_extract_tables", {"pdf_path": pdf_path})
    return PdfTablesResult(tables=result["tables"], page_nums=result["page_nums"])


def pdf_page_count(pdf_path: str) -> int:
    result = _call_worker("pdf_page_count", {"pdf_path": pdf_path})
    return result["n_pages"]


def pdf_render_page(pdf_path: str, page_no: int, out_png_path: str, resolution: int = 150) -> str:
    """Rasterize one 1-indexed PDF page to a PNG file (for the digitizer's
    'Open PDF page' action). Routed through pdfplumber/anaconda rather than
    Qt's QtPdf module -- see the worker script's docstring for why."""
    result = _call_worker(
        "pdf_render_page",
        {"pdf_path": pdf_path, "page_no": page_no, "out_png_path": out_png_path, "resolution": resolution},
    )
    return result["out_png_path"]


# ===========================================================================
# Native extension (.so) rebuild (KRUNAL_JULY_23.md point 8)
# ===========================================================================

@dataclass
class BuildResult:
    name: str
    ok: bool
    log: str
    built_artifact: Optional[str] = None
    copied_to: List[str] = field(default_factory=list)


def build_extension(name: str, copy_to_runtime: bool = True) -> BuildResult:
    """Run the extension's real build.sh/setup.py (found under config_tools/,
    each with its own prior-successful build/ dir -- see
    src/mso_io/native_extensions.py's module docstring for why this doesn't
    invent a build system). Runs with the anaconda interpreter/toolchain
    (cmake, pybind11, the g++ that built the existing .so files), never the
    lightweight .venv-gui. Called from a background thread (call_in_thread)
    by NativeExtensionsTab -- this function itself is synchronous."""
    import subprocess
    from mso_io.native_extensions import (
        copy_to_runtime_targets, find_built_artifact, inspect_extension,
    )

    info = inspect_extension(name)
    if not info.has_build_script:
        return BuildResult(name=name, ok=False,
                            log=f"No build.sh/setup.py found under {info.source_dir}")

    cmd = list(info.build_command)
    if cmd[0] == "python3":
        cmd[0] = SCIENCE_PYTHON
    proc = subprocess.run(cmd, cwd=info.source_dir, capture_output=True, text=True, timeout=600)
    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return BuildResult(name=name, ok=False, log=log)

    built = find_built_artifact(info)
    copied: List[str] = []
    if built and copy_to_runtime:
        copied = copy_to_runtime_targets(built, name)
    return BuildResult(name=name, ok=True, log=log, built_artifact=built, copied_to=copied)


# ---------------------------------------------------------------------------
# RDKit (subprocess -> anaconda; not installed in the GUI venv -- verified
# via .venv-gui/bin/python -c "import rdkit" failing ModuleNotFoundError)
# ---------------------------------------------------------------------------

@dataclass
class SmilesValidation:
    name: str
    smiles: str
    valid: bool
    canonical_smiles: Optional[str]
    formula: Optional[str]
    mw: Optional[float]
    error: Optional[str]


def rdkit_validate_smiles(
    rows: List[dict], *, on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[SmilesValidation]:
    """Validate a batch of ``{"name": ..., "smiles": ...}`` rows (e.g. the
    repo's own ``species_smiles.csv``, or a mechanism's species list) via
    RDKit in the anaconda subprocess. One bad SMILES never aborts the
    batch -- each row reports its own ``valid``/``error``."""
    result = _call_worker("rdkit_validate_smiles", {"rows": rows},
                           on_line=on_line, cancel_event=cancel_event, timeout=120.0)
    return [SmilesValidation(**r) for r in result["rows"]]


def rdkit_render_structure(smiles: str, size: int = 300) -> str:
    """Render *smiles* to a temporary PNG (2D structure depiction) via
    RDKit in the anaconda subprocess. Returns the PNG path."""
    out_path = str(Path(tempfile.mkdtemp()) / "structure.png")
    _call_worker("rdkit_render_png", {"smiles": smiles, "out_png_path": out_path, "size": size},
                 timeout=60.0)
    return out_path


def load_species_smiles_csv(path: Optional[str] = None) -> List[dict]:
    """Load ``species_smiles.csv`` (repo root, or *path*) as
    ``[{"name":..., "smiles":...}, ...]`` -- the RDKit tab's default data
    source. Returns ``[]`` (not an error) if the file doesn't exist, since
    a caller may want to validate a mechanism's own species list instead."""
    csv_path = Path(path) if path else REPO_ROOT / "species_smiles.csv"
    if not csv_path.is_file():
        return []
    import csv as _csv
    with csv_path.open(newline="") as fh:
        reader = _csv.DictReader(fh)
        return [{"name": row.get("name", ""), "smiles": row.get("smiles", "")} for row in reader]


# ---------------------------------------------------------------------------
# Jupyter (subprocess -> anaconda; launched detached, not tracked via
# run_backend since a notebook server is a long-lived background process,
# not a run-to-completion backend job)
# ---------------------------------------------------------------------------

def jupyter_available() -> bool:
    """Cheap presence check for enabling/disabling the Launch Jupyter
    button -- mirrors rmg_available()'s degrade-gracefully contract."""
    import shutil as _shutil
    return _shutil.which("jupyter") is not None or Path(SCIENCE_PYTHON).parent.joinpath("jupyter").exists()


def launch_jupyter(root_dir: str) -> int:
    """Start ``jupyter notebook`` rooted at *root_dir* (typically the
    current project directory, so generated notebooks and a mechanism's
    working files are all reachable from the browser tab) as a detached
    background process. Returns the child PID; the GUI does not wait on it
    or capture its output the way run_backend does for run-to-completion
    jobs -- a notebook server is meant to keep running after the button
    click returns."""
    import subprocess as _subprocess

    jupyter_bin = Path(SCIENCE_PYTHON).parent / "jupyter"
    command = [str(jupyter_bin) if jupyter_bin.exists() else "jupyter", "notebook",
               f"--notebook-dir={root_dir}"]
    proc = _subprocess.Popen(
        command, cwd=root_dir,
        stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid
