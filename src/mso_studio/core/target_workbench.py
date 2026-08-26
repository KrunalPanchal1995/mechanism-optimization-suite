"""
mso_studio.core.target_workbench
===================================
Adapter for the Target Workbench module (part of the Optimize Workbench):
curate a .target/.add pair before it reaches Addendum -- load, sanity-check,
generate NTC-aware dummy points, prune for T/P/phi coverage, then write a
new versioned pair. Ported from generate_target_opt.py's Tkinter
``_tab_target_workbench`` (WS-D-adjacent monolith split).

Every operation shells out to ``scripts/target_workbench_worker.py`` via
SCIENCE_PYTHON (same pattern as ``mso_studio.core.tools``'s
``_call_worker``) -- ``VisualAid.TargetChecker.check_parses`` imports
``combustion_target_class``, which needs sklearn/scipy, neither installed
in the light GUI venv.
"""
from __future__ import annotations

import json
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON, run_backend

WORKER_SCRIPT = REPO_ROOT / "scripts" / "target_workbench_worker.py"


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


@dataclass
class TargetRowInfo:
    line_no: int
    dataset_id: str
    target_type: str
    temperature_K: Optional[float]
    pressure_atm: Optional[float]
    phi_raw: Optional[str]
    observed: Optional[float]
    deviation: Optional[float]
    data_weight: Optional[float]
    composition_sum: float
    is_dummy: bool


@dataclass
class LoadResult:
    rows: List[TargetRowInfo]
    dataset_ids: List[str]


def load_target_set(target_path: str, add_path: str = "") -> LoadResult:
    result = _call_worker("load", {"target_path": target_path, "add_path": add_path})
    rows = [TargetRowInfo(
        line_no=r["line_no"], dataset_id=r["dataset_id"], target_type=r["target_type"],
        temperature_K=r["temperature_K"], pressure_atm=r["pressure_atm"],
        phi_raw=r["phi_raw"], observed=r["observed"], deviation=r["deviation"],
        data_weight=r["data_weight"], composition_sum=r["composition_sum"],
        is_dummy=r["is_dummy"],
    ) for r in result["rows"]]
    return LoadResult(rows=rows, dataset_ids=result["dataset_ids"])


@dataclass
class Finding:
    severity: str
    line_no: Optional[int]
    dataset_id: Optional[str]
    check: str
    message: str
    suggested_fix: Optional[str]


def run_checks(target_path: str, add_path: str = "") -> List[Finding]:
    result = _call_worker("check", {"target_path": target_path, "add_path": add_path})
    return [Finding(**f) for f in result["findings"]]


@dataclass
class DummyGenerationResult:
    rows: List[dict]          # {dataset_ID, Temperature_K, observed, ...}
    template_dataset_id: str


def generate_dummy_points(
    target_path: str, add_path: str, dataset_id: str, points_per_zone: int = 15,
) -> DummyGenerationResult:
    result = _call_worker("generate_dummy", {
        "target_path": target_path, "add_path": add_path,
        "dataset_id": dataset_id, "points_per_zone": points_per_zone,
    })
    return DummyGenerationResult(
        rows=result["rows"], template_dataset_id=result["template"]["dataset_id"])


@dataclass
class PruneResult:
    keep_line_nos: List[int]
    drop_line_nos: List[int]
    coverage: float


def coverage_prune(target_path: str, add_path: str, n_drop: int, seed: int = 0) -> PruneResult:
    result = _call_worker("coverage_prune", {
        "target_path": target_path, "add_path": add_path, "n_drop": n_drop, "seed": seed,
    })
    return PruneResult(**result)


@dataclass
class WriteCuratedResult:
    new_target_path: str
    new_add_path: str
    log_path: str
    gate_errors: List[dict] = field(default_factory=list)


def write_curated(
    target_path: str, add_path: str, out_stem: str,
    keep_line_nos: List[int],
    dummy_rows: Optional[List[dict]] = None,
    add_updates: Optional[dict] = None,
    reason_log: Optional[List[dict]] = None,
    out_dir: Optional[str] = None,
) -> WriteCuratedResult:
    result = _call_worker("write_curated", {
        "target_path": target_path, "add_path": add_path, "out_stem": out_stem,
        "keep_line_nos": keep_line_nos, "dummy_rows": dummy_rows or [],
        "add_updates": add_updates or {}, "reason_log": reason_log or [],
        "out_dir": out_dir,
    })
    return WriteCuratedResult(**result)


def render_plot(
    target_path: str, add_path: str, kind: str, out_png_path: str,
    dataset_id: str = "", dropped_line_nos: Optional[List[int]] = None,
) -> str:
    result = _call_worker("plot", {
        "target_path": target_path, "add_path": add_path, "kind": kind,
        "out_png_path": out_png_path, "dataset_id": dataset_id,
        "dropped_line_nos": dropped_line_nos or [],
    })
    return result["png_path"]
