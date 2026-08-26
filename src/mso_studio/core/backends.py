"""
mso_studio.core.backends
=========================
Subprocess bridge from the (light, PySide6-only) GUI virtual environment into
the anaconda ``python3.9`` interpreter that owns the heavy scientific stack
(Cantera, the MUQ-SAC solver bindings, the RMIP toolchain).

Why a subprocess and not an import
-----------------------------------
Per the env-isolation policy for this repo, new/GUI dependencies are never
pip-installed into the shared anaconda environment, and the shared
environment's heavy deps are never pip-installed into the GUI venv either.
The two stacks are bridged the same way the rest of the codebase already
does it (``gui_common.pty_terminal``, ``rmip.simulation.muqsac_bridge``):
launch the target script as a subprocess in the *other* interpreter and talk
to it over argv + stdout/stderr + result files on disk.

This module intentionally does **not** import anything from ``rmip.*`` --
``rmip.simulation`` pulls in Cantera-dependent runners at package-import
time, which would fail in the GUI venv. The environment-sanitization helper
below is a deliberate, small duplication of
``rmip.simulation.subprocess_util.sanitized_subprocess_env`` for that reason.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Repo root: src/mso_studio/core/backends.py -> src/mso_studio/core -> src/mso_studio -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

# Default finite timeout so a wedged solver can't block a worker thread forever.
# Mirrors rmip.simulation.subprocess_util.DEFAULT_SUBPROCESS_TIMEOUT_S.
DEFAULT_TIMEOUT_S = 1800.0

# Substrings (case-insensitive) marking an LD_LIBRARY_PATH entry as belonging
# to a foreign toolchain whose libstdc++ must not shadow the system one.
_FOREIGN_LIBPATH_MARKERS = ("ansys", "/ansys_inc/", "fluent", "matlab")


def _default_anaconda_python() -> str:
    """Resolve the anaconda python3.9 interpreter that owns the science stack."""
    override = os.environ.get("MSO_STUDIO_SCIENCE_PYTHON")
    if override:
        return override
    candidates = [
        Path.home() / "anaconda3" / "bin" / "python3.9",
        Path("/home/krithika/anaconda3/bin/python3.9"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Last resort: whatever's on PATH. Fails loudly and legibly if wrong.
    return "python3.9"


SCIENCE_PYTHON = _default_anaconda_python()


def _is_foreign_libpath(entry: str) -> bool:
    low = entry.lower()
    return any(marker in low for marker in _FOREIGN_LIBPATH_MARKERS)


def sanitized_subprocess_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Copy of ``os.environ`` with foreign ``LD_LIBRARY_PATH`` entries stripped,
    and ``SRC_DIR`` (this repo's ``src/``) prepended onto ``PYTHONPATH``.

    See ``rmip.simulation.subprocess_util.sanitized_subprocess_env`` -- same
    behavior, reimplemented here to avoid importing that Cantera-adjacent
    package from the GUI venv.

    The PYTHONPATH prepend fixes a real bug (found 2026-07-22 running the
    ammonia campaign's Phase 7 GUI sweep, `run_nominal` against Konnov 2019):
    ``science_command`` builds ``[SCIENCE_PYTHON, script, *args]`` with no
    PYTHONPATH setup at all, and ``run_backend`` launches it through this
    function's env -- so any backend script under ``src/`` importing a
    sibling package (``from mso_compare.config import ...``,
    ``from rmip... import ...``) crashes with
    ``ModuleNotFoundError: No module named 'mso_compare'`` unless the
    *calling* process happened to already have ``src/`` on its own
    PYTHONPATH before launching MSO Studio -- true only by accident, never
    guaranteed. Every ``mso_studio.core.*`` module that shells out via
    ``science_command`` (nominal, compare, sensitivity, optimization) hits
    this the same way; fixing it once here covers all of them.
    """
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH")
    if ld:
        kept = [p for p in ld.split(os.pathsep) if p and not _is_foreign_libpath(p)]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    existing_pp = env.get("PYTHONPATH", "")
    parts = [str(SRC_DIR)] + ([existing_pp] if existing_pp else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def resolve_timeout(timeout: Optional[float]) -> Optional[float]:
    return DEFAULT_TIMEOUT_S if timeout is None else timeout


class BackendError(RuntimeError):
    """Raised when a backend subprocess exits non-zero.

    ``str(exc)`` is written directly into GUI error dialogs, so the message
    is user-facing: what ran, how it failed, and where the full log is --
    never a bare traceback (guidelines doc, Part 1.3).
    """

    def __init__(self, command: Sequence[str], return_code: int, log_path: Optional[Path],
                 tail: Sequence[str]):
        self.command = list(command)
        self.return_code = return_code
        self.log_path = log_path
        self.tail = list(tail)
        pretty_cmd = " ".join(shlex.quote(c) for c in command)
        detail = "\n".join(tail[-20:]) if tail else "(no output captured)"
        log_hint = f"\nFull log: {log_path}" if log_path else ""
        super().__init__(
            f"Backend command exited with code {return_code}:\n  {pretty_cmd}\n\n"
            f"Last output:\n{detail}{log_hint}"
        )


def run_backend(
    command: Sequence[str],
    *,
    cwd: Optional[str | os.PathLike] = None,
    env_extra: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    log_path: Optional[Path] = None,
) -> "BackendResult":
    """Run *command* to completion, streaming stdout/stderr line-by-line.

    Parameters
    ----------
    command : sequence of str
        Full argv, e.g. ``[SCIENCE_PYTHON, "src/run_mso_pipeline.py",
        "--config", "mso_config.yaml"]``.
    cwd : path, optional
        Working directory for the subprocess.
    env_extra : dict, optional
        Extra environment variables layered on top of the sanitized copy.
    timeout : float, optional
        Wall-clock ceiling in seconds; ``None`` uses :data:`DEFAULT_TIMEOUT_S`.
    on_line : callable, optional
        Called with each combined stdout/stderr line as it arrives -- this is
        the hook a GUI worker uses to drive a progress log / terminal widget.
    cancel_event : threading.Event, optional
        If set while the process is running, the process is terminated and
        the result's ``status`` is ``"cancelled"``.
    log_path : Path, optional
        If given, every line is also appended here (full, untruncated log).

    Returns
    -------
    BackendResult
    """
    cwd = str(cwd) if cwd is not None else None
    env = sanitized_subprocess_env(env_extra)
    effective_timeout = resolve_timeout(timeout)

    log_fh = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("a", buffering=1)

    tail: List[str] = []
    proc = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    status = "done"
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if len(tail) > 500:
                tail.pop(0)
            if log_fh is not None:
                log_fh.write(line + "\n")
            if on_line is not None:
                on_line(line)
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                status = "cancelled"
                break
        proc.wait(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        status = "timeout"
    finally:
        if log_fh is not None:
            log_fh.close()

    return_code = proc.returncode if proc.returncode is not None else -1
    if status == "done" and return_code != 0:
        status = "failed"

    return BackendResult(
        command=list(command),
        return_code=return_code,
        status=status,
        tail=tail,
        log_path=log_path,
    )


class BackendResult:
    """Outcome of :func:`run_backend`."""

    def __init__(self, command: List[str], return_code: int, status: str,
                 tail: List[str], log_path: Optional[Path]):
        self.command = command
        self.return_code = return_code
        self.status = status  # done | failed | cancelled | timeout
        self.tail = tail
        self.log_path = log_path

    def raise_if_failed(self) -> None:
        if self.status in ("failed", "timeout"):
            raise BackendError(self.command, self.return_code, self.log_path, self.tail)


def science_command(script: str | os.PathLike, *args: str) -> List[str]:
    """Build ``[SCIENCE_PYTHON, script, *args]`` with *script* resolved against ``src/``.

    Accepts either an absolute path or one relative to ``src/`` (e.g.
    ``"run_mso_pipeline.py"`` or ``"MUQ-SAC/run_script.py"``).
    """
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = SRC_DIR / script_path
    return [SCIENCE_PYTHON, str(script_path), *args]
