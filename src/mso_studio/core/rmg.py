"""
mso_studio.core.rmg
=====================
Subprocess adapter into RMG-Py (Reaction Mechanism Generator) -- the "build
a mechanism from scratch" helper package this suite otherwise assumes you
already have a mechanism for. RMG-Py lives in its own conda environment
(``rmg_env``, Python 3.11 -- incompatible with this repo's anaconda
``python3.9`` science stack and unrelated to the GUI's light PySide6 venv),
so this follows exactly the pattern already established by
:mod:`mso_studio.core.backends` / :mod:`mso_studio.core.tools` (PySR needs
Julia via the anaconda subprocess; RMG needs its own interpreter the same
way) -- shell out, stream the log, never import ``rmgpy`` in-process.

Graceful degradation: if ``rmg_env`` or an RMG-Py checkout isn't present on
this machine, :func:`rmg_available` reports that plainly rather than any
caller raising an ImportError deep in a worker thread.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from mso_studio.core.backends import run_backend


def _default_rmg_python() -> Optional[str]:
    """Resolve the ``rmg_env`` conda environment's interpreter. Override
    with ``MSO_STUDIO_RMG_PYTHON`` (mirrors ``MSO_STUDIO_SCIENCE_PYTHON``'s
    override convention in backends.py). Returns ``None`` if not found --
    callers use this to disable RMG features rather than crash."""
    override = os.environ.get("MSO_STUDIO_RMG_PYTHON")
    if override and Path(override).exists():
        return override
    candidates = [
        Path.home() / "anaconda3" / "envs" / "rmg_env" / "bin" / "python",
        Path("/home/krithika/anaconda3/envs/rmg_env/bin/python"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _default_rmg_script() -> Optional[str]:
    """Resolve the RMG-Py checkout's ``rmg.py`` entry point. Prefers
    whichever install ``rmg_env``'s own ``import rmgpy`` actually resolves
    to (checked once, cheaply, via -c) over a hardcoded path -- a machine
    can have more than one RMG-Py checkout on disk (this one does:
    ``~/RMG-Py`` and ``/data/OPEN_CODE/RMG_py``) and only the one on
    ``rmg_env``'s path is guaranteed to match the installed ``rmgpy``
    package version. Override with ``MSO_STUDIO_RMG_SCRIPT``."""
    override = os.environ.get("MSO_STUDIO_RMG_SCRIPT")
    if override and Path(override).exists():
        return override

    py = _default_rmg_python()
    if py:
        try:
            out = subprocess.run(
                [py, "-c", "import rmgpy, os; print(os.path.dirname(rmgpy.__file__))"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                pkg_dir = out.stdout.strip()
                candidate = Path(pkg_dir).parent / "rmg.py"
                if candidate.exists():
                    return str(candidate)
        except (OSError, subprocess.SubprocessError):
            pass

    for c in (Path.home() / "RMG-Py" / "rmg.py",
              Path("/data/OPEN_CODE/RMG_py/rmg.py")):
        if c.exists():
            return str(c)
    return None


def rmg_available() -> tuple[bool, str]:
    """Cheap, side-effect-free presence check for enabling/disabling RMG
    tiles/buttons. Returns ``(available, reason)`` -- ``reason`` is always
    filled in (either "found at ..." or why it's unavailable), same
    contract as ``mso_studio.launcher.project_link.is_available``."""
    py = _default_rmg_python()
    if py is None:
        return False, "no rmg_env conda environment found (looked for ~/anaconda3/envs/rmg_env)"
    script = _default_rmg_script()
    if script is None:
        return False, f"rmg_env found ({py}) but no rmg.py entry point found on disk"
    return True, f"found: {py} running {script}"


@dataclass
class RmgResult:
    """What one RMG job produced. ``chem_annotated_path`` /
    ``species_dictionary_path`` are ``None`` if the job didn't reach that
    stage (a partial/failed run) -- callers check for ``None`` rather than
    assuming success from a zero return code alone, since RMG can exit 0
    after only partially building a model on a tight ``walltime``."""

    output_dir: str
    chem_annotated_path: Optional[str]
    species_dictionary_path: Optional[str]
    cantera_yaml_path: Optional[str]
    log_tail: List[str]
    status: str  # "done" | "failed" | "cancelled"


def run_rmg_job(
    input_py: str,
    out_dir: Optional[str] = None,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout: Optional[float] = None,
) -> RmgResult:
    """Run RMG-Py against an ``input.py`` config (RMG's own config format --
    see ``examples/rmg/minimal/input.py`` in any RMG-Py checkout) in the
    ``rmg_env`` subprocess, streaming the log the same way every other
    ``core`` adapter does via :func:`mso_studio.core.backends.run_backend`.

    RMG writes its own output alongside *input_py* by default (a
    ``chemkin/`` subfolder with ``chem_annotated.inp`` +
    ``species_dictionary.txt``, plus ``cantera/`` if the Cantera writer is
    enabled) -- *out_dir* overrides that via RMG's own ``-o`` flag so a
    caller can point multiple jobs at different scratch directories without
    editing each ``input.py``.

    Raises RuntimeError if rmg_env / rmg.py aren't available (call
    :func:`rmg_available` first to check without raising).
    """
    py = _default_rmg_python()
    script = _default_rmg_script()
    if py is None or script is None:
        available, reason = rmg_available()
        raise RuntimeError(f"RMG-Py is not available: {reason}")

    input_py = str(Path(input_py).resolve())
    command = [py, script, input_py]
    if out_dir:
        out_dir = str(Path(out_dir).resolve())
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        command += ["-o", out_dir]
    else:
        out_dir = str(Path(input_py).parent)

    result = run_backend(command, on_line=on_line, cancel_event=cancel_event, timeout=timeout)

    chem_dir = Path(out_dir) / "chemkin"
    chem_annotated = chem_dir / "chem_annotated.inp"
    species_dict = chem_dir / "species_dictionary.txt"
    cantera_yaml = Path(out_dir) / "cantera" / "chem_annotated.yaml"

    return RmgResult(
        output_dir=out_dir,
        chem_annotated_path=str(chem_annotated) if chem_annotated.exists() else None,
        species_dictionary_path=str(species_dict) if species_dict.exists() else None,
        cantera_yaml_path=str(cantera_yaml) if cantera_yaml.exists() else None,
        log_tail=result.tail,
        status=result.status,
    )


def import_rmg_mechanism(out_dir: str) -> dict:
    """Locate an already-finished RMG job's output under *out_dir* (the
    same layout :func:`run_rmg_job` returns) and hand back paths ready for
    ``mechanism_converter``/``mechanism_check`` -- so an RMG-generated
    mechanism can flow straight into the rest of the optimize workbench
    without the user hunting through ``chemkin/`` by hand.

    Returns ``{"mechanism": ..., "species_dictionary": ..., "cantera_yaml": ...}``
    with any key set to ``None`` if that file isn't present. Never raises --
    an RMG run that stopped early (edge species not yet moved to core) still
    has a valid, if smaller, mechanism worth importing.
    """
    out_dir = Path(out_dir)
    chem_annotated = out_dir / "chemkin" / "chem_annotated.inp"
    species_dict = out_dir / "chemkin" / "species_dictionary.txt"
    cantera_yaml = out_dir / "cantera" / "chem_annotated.yaml"
    return {
        "mechanism": str(chem_annotated) if chem_annotated.exists() else None,
        "species_dictionary": str(species_dict) if species_dict.exists() else None,
        "cantera_yaml": str(cantera_yaml) if cantera_yaml.exists() else None,
    }


def rmgpy_version() -> Optional[str]:
    """Best-effort ``rmgpy.__version__``-equivalent for the doctor tile --
    RMG-Py does not always expose a clean ``__version__``, so this falls
    back to a short git describe of the checkout if the package attribute
    isn't set. Returns ``None`` if rmg_env itself isn't available."""
    py = _default_rmg_python()
    if py is None:
        return None
    try:
        out = subprocess.run(
            [py, "-c",
             "import rmgpy\n"
             "v = getattr(rmgpy, '__version__', None)\n"
             "print(v or 'unknown')"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            v = out.stdout.strip()
            if v and v != "unknown":
                return v
    except (OSError, subprocess.SubprocessError):
        return None

    script = _default_rmg_script()
    if script:
        repo_dir = Path(script).parent
        try:
            out = subprocess.run(["git", "describe", "--always", "--dirty"],
                                  cwd=str(repo_dir), capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                sha = out.stdout.strip()
                if re.match(r"^[0-9a-fA-F]", sha):
                    return f"git:{sha}"
        except (OSError, subprocess.SubprocessError):
            pass
    return "unknown"
