"""
mso_workbench.breakpoint
===========================
The hook a pipeline calls at a stage boundary. Disabled entirely by
``MSO_WORKBENCH=0`` (checked first, so an opted-out run pays nothing at
all -- not even a file stat). Otherwise: touch a heartbeat, report the
current stage, and block only if a pause was actually requested for this
stage (or "next"), writing a snapshot of the caller's locals first.

None of this call-site machinery is expensive relative to what it's
instrumenting -- every wired stage in ``run_script.py`` is a once-per-run
or once-per-case boundary (config parsed, samples generated, simulations
launched, PRS fit, optimization run), never a per-sample/per-reaction
inner-loop call, so a small JSON write per call is immaterial next to the
minutes-to-hours of real work each stage represents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from mso_workbench.control import (
    ACTION_PAUSE, ACTION_RESUME, ACTION_STEP, AT_NEXT,
    POLL_INTERVAL_S, STATUS_PAUSED, STATUS_RUNNING,
    clear_request, is_enabled, read_request, touch_heartbeat, write_request, write_state,
)
from mso_workbench.snapshot import write_snapshot


def breakpoint(
    stage_id: str, locals_: Optional[Dict[str, Any]] = None,
    source: Optional[Tuple[str, str]] = None, run_dir: Optional[str] = None,
) -> None:
    """Call at a pipeline stage boundary. *locals_* is whatever local state
    is worth snapshotting if a pause lands here (typically a small dict of
    the caller's own locals -- pass only what's meaningful, not the whole
    frame). *source* is ``(file, function)``, purely descriptive. *run_dir*
    defaults to the current working directory (every call site in
    ``run_script.py`` runs from the run's own directory)."""
    if not is_enabled():
        return

    import os
    run_dir = run_dir or os.getcwd()

    touch_heartbeat(run_dir)
    request = read_request(run_dir)

    if request is None or not _matches(request, stage_id):
        write_state(run_dir, STATUS_RUNNING, stage_id)
        return

    action = request.get("action")
    if action not in (ACTION_PAUSE, ACTION_STEP):
        write_state(run_dir, STATUS_RUNNING, stage_id)
        return

    write_snapshot(run_dir, stage_id, locals_ or {}, source)
    write_state(run_dir, STATUS_PAUSED, stage_id)

    while True:
        time.sleep(POLL_INTERVAL_S)
        touch_heartbeat(run_dir)
        pending = read_request(run_dir)
        if pending is None:
            break  # request cleared externally -> resume
        pending_action = pending.get("action")
        if pending_action == ACTION_RESUME:
            clear_request(run_dir)
            break
        if pending_action == ACTION_STEP:
            # Stepping while paused: re-arm for the very next boundary,
            # then let this one go.
            write_request(run_dir, ACTION_PAUSE, AT_NEXT)
            break
        # Still a pause request (possibly retargeted at a different stage
        # while we sit here) -- keep waiting.

    write_state(run_dir, STATUS_RUNNING, stage_id)


def _matches(request: Dict[str, Any], stage_id: str) -> bool:
    at = request.get("at")
    return at == AT_NEXT or at == stage_id
