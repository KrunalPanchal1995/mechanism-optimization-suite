"""
mso_studio.core.automation
=============================
Adapter over the integrated mechanism-comparison automation
(``src/automation/run_tasks.py`` -- a generic YAML-driven task runner) and
its meta-PRS hyperparameter-refinement toolchain (``src/automation/prs_*.py``).

Integrated 2026-07-21 from the external p_PRS project (see
``src/automation/*.py`` module docstrings for the exact provenance and the
path-hardcoding fixes applied on import). Two important differences from
every other ``core`` adapter in this package:

1. **Long-lived, not one-shot.** A study YAML describes work that runs for
   days (``--daemon``); there is no single "the run finished, here's the
   RunRecord" moment. This module exposes validate/preflight/status/launch
   as separate operations rather than one blocking ``run_*`` call.
2. **A live, independent campaign already exists.** The external p_PRS
   ``_automation/`` directory this code was copied from keeps running its
   own studies on this same machine. ``run_preflight`` below always checks
   thread headroom across *both* directories (see
   ``automation.prs_preflight.automation_dirs``) so a study launched from
   this repo can't accidentally oversubscribe the machine on top of that
   other, already-running campaign.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON, run_backend
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

AUTOMATION_DIR = REPO_ROOT / "src" / "automation"
RUN_TASKS_SCRIPT = AUTOMATION_DIR / "run_tasks.py"

# The one dataset this meta-PRS toolchain was built against -- see
# src/automation/prs_extract_doe.py's docstring. Only applied as a default
# when the caller hasn't already set MSO_PPRS_DATA_ROOT themselves.
_DEFAULT_PPRS_DATA_ROOT = "/data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/MSO_A_FACTOR/p_PRS"

if str(AUTOMATION_DIR) not in sys.path:
    # The automation scripts import each other with bare `from prs_fit import
    # ...`-style statements (they're siblings run/imported from their own
    # directory, not a dotted package) -- match that convention here rather
    # than restructuring code that works correctly today.
    sys.path.insert(0, str(AUTOMATION_DIR))
os.environ.setdefault("MSO_PPRS_DATA_ROOT", _DEFAULT_PPRS_DATA_ROOT)


@dataclass
class CheckResult:
    """Outcome of a read-only check (validate / status / preflight)."""

    ok: bool
    message: str
    detail: dict


def validate_yaml(yaml_path: str | os.PathLike) -> CheckResult:
    """Run ``run_tasks.py <yaml> --validate`` -- checks the YAML and exits;
    nothing is launched (see TASK_SCHEMA.md)."""
    result = run_backend(
        [SCIENCE_PYTHON, str(RUN_TASKS_SCRIPT), str(yaml_path), "--validate"],
        cwd=str(AUTOMATION_DIR),
    )
    ok = result.status == "done" and result.return_code == 0
    return CheckResult(ok=ok, message="\n".join(result.tail), detail={"return_code": result.return_code})


def get_status(yaml_path: str | os.PathLike) -> CheckResult:
    """Run ``run_tasks.py <yaml> --status`` for the human-readable summary,
    plus a structured per-task/per-run breakdown read directly from the
    state file for the GUI's status table."""
    result = run_backend(
        [SCIENCE_PYTHON, str(RUN_TASKS_SCRIPT), str(yaml_path), "--status"],
        cwd=str(AUTOMATION_DIR),
    )
    ok = result.status == "done" and result.return_code == 0
    detail = {"text": "\n".join(result.tail)}
    try:
        detail["tasks"] = _read_state_summary(yaml_path)
    except Exception as exc:  # noqa: BLE001 - the text summary above is enough on its own
        detail["tasks_error"] = str(exc)
    return CheckResult(ok=ok, message="\n".join(result.tail), detail=detail)


def _read_state_summary(yaml_path: str | os.PathLike) -> dict:
    import prs_preflight  # noqa: PLC0415 - deferred until AUTOMATION_DIR is on sys.path
    from run_tasks import Config, State  # noqa: PLC0415

    cfg = Config(str(yaml_path))
    if not cfg.state_file.exists():
        return {}
    state = State(cfg.state_file)
    summary = {}
    for t in cfg.tasks:
        st = state.task(t["id"])
        if t["type"] in ("edit_params", "report"):
            summary[t["id"]] = {"type": t["type"], "status": st.get("status", "pending")}
            continue
        counts: dict[str, int] = {}
        failed = []
        for key, r in st.get("runs", {}).items():
            status = r.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
            if status == "failed":
                failed.append({"key": key, "note": r.get("note", r.get("error", ""))})
        summary[t["id"]] = {"type": t["type"], "counts": counts, "failed": failed}
    return summary


def run_preflight(planned_max_threads: int, cap: int = 160) -> CheckResult:
    """Disk-space + cross-directory thread-headroom check (never launches or
    deletes anything) -- run this before offering to launch a study."""
    import prs_preflight  # noqa: PLC0415

    disk = prs_preflight.check_disk(min_free_gb=200)
    threads = prs_preflight.check_thread_headroom(planned_max_threads, cap=cap)
    ok = disk["ok"] and threads["ok"]
    message = f"{disk['message']}\n{threads['message']}"
    return CheckResult(ok=ok, message=message, detail={"disk": disk, "threads": threads})


def launch_daemon(
    yaml_path: str | os.PathLike,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run ``run_tasks.py <yaml> --daemon``: forks, detaches, and returns
    almost immediately (the study itself then runs unsupervised for days --
    use :func:`get_status` / :func:`validate_yaml` to check on it later).

    Deliberately does NOT call :func:`run_preflight` itself -- the GUI must
    call it first and let the user see the result, per TASK_SCHEMA.md's
    "ALWAYS run --validate first" convention and this project's standing
    live-process caution around launching new studies unsupervised.
    """
    command = [SCIENCE_PYTHON, str(RUN_TASKS_SCRIPT), str(yaml_path), "--daemon"]
    run_config = RunConfig(
        module="automation", label=Path(yaml_path).stem, backend_command=command,
        working_dir=str(AUTOMATION_DIR), inputs={"yaml_path": str(yaml_path)},
    )
    record = new_record(run_config)
    result = run_backend(command, cwd=str(AUTOMATION_DIR), on_line=on_line, cancel_event=cancel_event)
    record.log_tail = result.tail[-200:]
    record.mark_finished("done" if result.status == "done" and result.return_code == 0 else "failed",
                          result.return_code)
    record.results["note"] = (
        "Study detached and running in the background. It keeps running "
        "after this window closes -- check back with Refresh status."
    )
    record.save()
    return record


# ---------------------------------------------------------------------------
# Meta-PRS DoE fit -- read-only surface over prs_extract_doe / prs_fit /
# prs_optimize. Pure numpy/scipy, runs in-process (no subprocess needed).
# ---------------------------------------------------------------------------

@dataclass
class MetaPrsFitResult:
    fuel: str
    n_points: int
    trusted: bool
    loocv_max_abs_pct: float
    loocv_mean_abs_pct: float
    optimum_p_start: float
    optimum_p_final: float
    optimum_predicted_wmre: float
    grid_agrees: bool


def fit_meta_prs(fuel: str, precision: float = 0.001) -> MetaPrsFitResult:
    """Extract the DoE, fit the 2-variable quadratic PRS with its LOOCV trust
    gate, and cheaply optimize the fitted surface -- entirely read-only
    against the (external, already-validated) factorial report .tex tables.
    Writes only small derived CSV/JSON into ``src/automation/prs_doe_data/``.
    """
    import prs_extract_doe  # noqa: PLC0415
    import prs_fit  # noqa: PLC0415
    import prs_optimize  # noqa: PLC0415

    if fuel not in prs_extract_doe.FUEL_REPORTS:
        raise ValueError(
            f"Unknown fuel '{fuel}'. Available: {', '.join(prs_extract_doe.FUEL_REPORTS)}."
        )
    tex_path = prs_extract_doe.FUEL_REPORTS[fuel]
    if not Path(tex_path).is_file():
        raise FileNotFoundError(
            f"Factorial report not found for {fuel}: {tex_path}\n"
            f"(MSO_PPRS_DATA_ROOT={os.environ.get('MSO_PPRS_DATA_ROOT')} -- "
            "set it to the p_PRS project root if the data lives elsewhere.)"
        )
    rows = prs_extract_doe.parse_factorial_report(tex_path)
    prs_extract_doe.write_csv(fuel, rows)

    fit_result, _ = prs_fit.fit_and_save(fuel)
    bounds = prs_optimize.FUEL_BOUNDS.get(fuel, prs_optimize.DEFAULT_BOUNDS)
    optimum = prs_optimize.optimize_prs(fit_result, bounds, precision=precision)

    return MetaPrsFitResult(
        fuel=fuel, n_points=fit_result.n_points, trusted=fit_result.trusted,
        loocv_max_abs_pct=fit_result.loocv_max_abs_pct,
        loocv_mean_abs_pct=fit_result.loocv_mean_abs_pct,
        optimum_p_start=optimum.p_start, optimum_p_final=optimum.p_final,
        optimum_predicted_wmre=optimum.predicted_wmre, grid_agrees=optimum.grid_agrees,
    )


def available_fuels() -> list[str]:
    import prs_extract_doe  # noqa: PLC0415
    return list(prs_extract_doe.FUEL_REPORTS)
