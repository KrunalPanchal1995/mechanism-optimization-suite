"""
mso_studio.core.mechanism_check
==================================
Adapter over the existing ``rmip check`` CLI (``src/rmip/cli.py::_cmd_check``,
which already wraps ``rmip.analysis.mechanism_check.MechanismChecker`` with a
full argv contract including ``--json``) -- unlike the converter port, no new
headless CLI mode was needed here; the CLI this shells out to already
existed and is exercised by ``src/rmip/tests/test_mechanism_check.py``'s
``TestCli`` class.

Shells out via ``science_command``/``run_backend`` rather than importing
``MechanismChecker`` directly, since it depends on Cantera (never installed
in the PySide6-only GUI venv -- see ``backends.py``'s module docstring).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_RMIP_CLI = REPO_ROOT / "rmip_cli.py"


@dataclass
class MechanismCheckParams:
    """User-facing inputs for one health-check run across ``mechanisms``.

    An empty ``checks`` list means "run every check" (``rmip check``'s own
    default, matching ``ALL_CHECKS`` in ``rmip.analysis.mechanism_check``).
    """

    mechanisms: List[str]
    thermo: str = ""
    transport: str = ""
    checks: List[str] = field(default_factory=list)
    T_grid: List[float] = field(default_factory=list)
    P_grid: List[float] = field(default_factory=list)
    target_T_high: float = 2500.0
    collision_exceedance: float = 10.0
    output_dir: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "mechanism_check")
        if not self.label:
            self.label = "+".join(Path(m).stem for m in self.mechanisms[:3])


def run_check(
    params: MechanismCheckParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run ``rmip check`` over ``params.mechanisms`` and return its RunRecord.

    Like ``converter.run_convert``, does NOT raise when the checked
    mechanism(s) have findings (``rmip check`` exits 1 in that case) -- that
    is an expected, informative result, not an infrastructure failure.
    ``status="failed"`` is reserved for the CLI itself crashing (no JSON
    report produced at all).
    """
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{params.label}_check_report.json"

    args = ["check", *params.mechanisms, "--json", str(json_path)]
    if params.thermo:
        args += ["--thermo", params.thermo]
    if params.transport:
        args += ["--transport", params.transport]
    if params.checks:
        args += ["--checks", *params.checks]
    if params.T_grid:
        args += ["--T-grid", *[str(t) for t in params.T_grid]]
    if params.P_grid:
        args += ["--P-grid", *[str(p) for p in params.P_grid]]
    args += ["--target-T-high", str(params.target_T_high)]
    args += ["--collision-exceedance", str(params.collision_exceedance)]
    args.append("--no-color")

    command = science_command(str(_RMIP_CLI), *args)

    run_config = RunConfig(
        module="mechanism_check",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={"mechanisms": ", ".join(params.mechanisms),
                "thermo": params.thermo, "transport": params.transport},
        parameters={
            "checks": params.checks or "all", "T_grid": params.T_grid,
            "P_grid": params.P_grid, "target_T_high": params.target_T_high,
            "collision_exceedance": params.collision_exceedance,
        },
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_{params.label}_check.log"
    result = run_backend(
        command, cwd=str(out_dir), on_line=on_line, cancel_event=cancel_event,
        log_path=log_path,
    )
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record

    if not json_path.exists():
        # The CLI itself crashed (import error, bad argv, ...) rather than
        # completing a check and reporting findings -- a real infra failure.
        record.mark_finished(
            "failed", result.return_code,
            error=f"rmip check produced no report (exit {result.return_code}) -- see log")
        record.save()
        return record

    reports = json.loads(json_path.read_text())
    if isinstance(reports, dict):
        reports = [reports]
    # CheckReport.to_dict()'s shape: {"counts": {"error": N, "warning": M,
    # "info": K}, ...} -- n_errors/n_warnings are properties on the live
    # object, not dict keys.
    total_errors = sum(r.get("counts", {}).get("error", 0) for r in reports)
    total_warnings = sum(r.get("counts", {}).get("warning", 0) for r in reports)
    record.results["reports"] = reports
    record.results["n_errors"] = total_errors
    record.results["n_warnings"] = total_warnings
    record.results["json_path"] = str(json_path)
    record.tables["check_report"] = str(json_path)

    record.mark_finished("done", result.return_code)
    record.save()
    return record
