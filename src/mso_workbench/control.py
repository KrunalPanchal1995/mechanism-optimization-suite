"""
mso_workbench.control
========================
File-based pause protocol -- no sockets, no daemon, so it survives however
the pipeline was launched (plain ``subprocess.Popen``, a real pty via
``gui_common.pty_terminal``, an external terminal spawned by
``generate_target_opt.py``'s Run-Job dialog, or run by hand from a shell:
all four launch paths exist in this repo today, and none of them offers a
reliable place to hold a socket or RPC connection open).

Layout under ``<run_dir>/.workbench/``::

    request      JSON {"action": "pause"|"resume"|"step", "at": "<stage>"|"next", "ts": float}
    state        JSON {"status": "running"|"paused", "stage": str|None, "pid": int, "since": float}
    heartbeat    touched at every breakpoint call -- lets a UI show liveness
                 without depending on log-line scraping
    snapshots/<stage>_<ts>/   one directory per pause, written by snapshot.py

Every read tolerates a missing or partially-written file (returns ``None``
rather than raising) -- this machinery must never be the reason a real
research run crashes, even if a concurrent writer is caught mid-write.
Writes are atomic (write to a temp file, then ``os.replace``) to make that
partial-read case rare in the first place.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

CONTROL_DIRNAME = ".workbench"
REQUEST_FILE = "request"
STATE_FILE = "state"
HEARTBEAT_FILE = "heartbeat"
SNAPSHOTS_DIRNAME = "snapshots"

ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_STEP = "step"
AT_NEXT = "next"

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"

# Env var to kill-switch the whole feature. Any value other than "0"
# (including unset) leaves it enabled -- a run must opt out explicitly.
_ENV_VAR = "MSO_WORKBENCH"

# How often a blocked breakpoint re-checks the request file while paused.
POLL_INTERVAL_S = 0.25


def is_enabled() -> bool:
    return os.environ.get(_ENV_VAR, "1") != "0"


def control_dir(run_dir) -> Path:
    return Path(run_dir) / CONTROL_DIRNAME


def snapshots_dir(run_dir) -> Path:
    return control_dir(run_dir) / SNAPSHOTS_DIRNAME


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        text = path.read_text()
    except (FileNotFoundError, OSError):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_request(run_dir) -> Optional[dict]:
    return _read_json(control_dir(run_dir) / REQUEST_FILE)


def write_request(run_dir, action: str, at: str = AT_NEXT) -> None:
    """Called by the Workbench UI/CLI to ask a running pipeline to pause,
    resume, or step. *at* is a stage id (see stages.py) or "next" for the
    very next boundary reached."""
    _atomic_write_json(
        control_dir(run_dir) / REQUEST_FILE,
        {"action": action, "at": at, "ts": time.time()},
    )


def clear_request(run_dir) -> None:
    path = control_dir(run_dir) / REQUEST_FILE
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def read_state(run_dir) -> Optional[dict]:
    return _read_json(control_dir(run_dir) / STATE_FILE)


def write_state(run_dir, status: str, stage: Optional[str], pid: Optional[int] = None) -> None:
    """Called by the pipeline side to report what it's doing."""
    _atomic_write_json(
        control_dir(run_dir) / STATE_FILE,
        {"status": status, "stage": stage, "pid": pid if pid is not None else os.getpid(),
         "since": time.time()},
    )


def touch_heartbeat(run_dir) -> None:
    path = control_dir(run_dir) / HEARTBEAT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def heartbeat_age_s(run_dir) -> Optional[float]:
    """Seconds since the last heartbeat, or None if there has never been one."""
    path = control_dir(run_dir) / HEARTBEAT_FILE
    try:
        return time.time() - path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None
