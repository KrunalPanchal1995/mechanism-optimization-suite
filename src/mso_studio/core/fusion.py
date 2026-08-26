"""
mso_studio.core.fusion
=========================
Adapter over ``mean_value_fusion_gui.py``'s ``--fuse`` headless CLI mode
(KRUNAL_JULY_23.md point 6's fix -- automatic glossary backfill + the
degenerate-empty-union guard -- lives in ``_run_fusion()``, which this CLI
calls, so it's fixed here too).

Shells out rather than importing ``rmip.stubs.FusionEngine``/
``GraphComparator`` directly, since those are Cantera/rdkit-adjacent (never
installed in the PySide6-only GUI venv -- see ``backends.py``'s docstring).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record

_FUSION_SCRIPT = REPO_ROOT / "mean_value_fusion_gui.py"
_DEFAULT_GLOSSARY = str(REPO_ROOT / "species_smiles.csv")

_SUMMARY_LINE = re.compile(r"^(\w+)=(.*)$")


@dataclass
class FusionParams:
    """User-facing inputs for one mechanism-fusion run.

    Parameters mirror ``mean_value_fusion_gui._run_fusion()``'s keyword
    arguments 1:1.
    """

    mechanism_ids: List[int]
    db_path: str
    tol_A: float = 0.01
    tol_n: float = 0.001
    tol_Ea: float = 100.0
    output_name: str = "Fused_Mech"
    output_dir: str = ""
    write_cantera: bool = False
    priority_order: Optional[List[int]] = None
    glossary_path: Optional[str] = _DEFAULT_GLOSSARY
    label: str = ""
    extra_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "outputs" / "fusion")
        if not self.label:
            self.label = self.output_name


def run_fusion(
    params: FusionParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run one mechanism fusion and return its :class:`RunRecord`.

    Does NOT raise on a degenerate/empty union or a data-quality gap that
    blocks writing the fused mechanism (e.g. a reaction with no usable rate
    data) -- those are expected, informative outcomes (see
    ``FusionDegenerateError`` in the Tk app), not infrastructure errors.
    ``status="failed"`` with the tool's own ``ERROR:`` line as
    ``record.error`` in that case.
    """
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--fuse", "--db", params.db_path,
        "--mechs", *[str(m) for m in params.mechanism_ids],
        "--tol-A", str(params.tol_A), "--tol-n", str(params.tol_n),
        "--tol-Ea", str(params.tol_Ea),
        "--output-name", params.output_name, "--output-dir", str(out_dir),
        "--glossary", params.glossary_path or "",
    ]
    if params.write_cantera:
        args.append("--write-cantera")
    if params.priority_order:
        args += ["--priority-order", *[str(m) for m in params.priority_order]]

    command = science_command(str(_FUSION_SCRIPT), *args)

    run_config = RunConfig(
        module="mechanism_fusion",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={"RMIP database": params.db_path,
                "mechanism IDs": ", ".join(str(m) for m in params.mechanism_ids),
                "glossary": params.glossary_path or ""},
        parameters={
            "tol_A": params.tol_A, "tol_n": params.tol_n, "tol_Ea": params.tol_Ea,
            "write_cantera": params.write_cantera,
            "priority_order": params.priority_order,
        },
        extra=dict(params.extra_overrides),
    )
    record = new_record(run_config)

    log_path = out_dir / f"mso_studio_{params.label}_fuse.log"
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

    summary = _parse_summary(result.tail)
    if result.status != "done" or "fused_mech_id" not in summary:
        error_lines = [ln for ln in result.tail if ln.startswith("ERROR:")]
        error = error_lines[-1][len("ERROR: "):] if error_lines else \
            f"fusion failed (exit {result.return_code}) -- see log"
        record.mark_finished("failed", result.return_code, error=error)
        record.save()
        return record

    record.results.update(summary)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _parse_summary(lines: List[str]) -> Dict[str, object]:
    """Parse the plain ``key=value`` summary lines _run_fusion's --fuse mode
    prints (see mean_value_fusion_gui.py's main())."""
    out: Dict[str, object] = {}
    for line in lines:
        m = _SUMMARY_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key in ("fused_mech_id",):
            out[key] = int(value)
        elif key == "balanced":
            out[key] = value == "True"
        elif key == "n_species":
            out[key] = int(value)
        elif key == "output":
            out[key] = value
        elif key == "uncertainty_csv":
            out["uncertainty_csv_path"] = value
        elif key == "n_uncertainty_reactions":
            out[key] = int(value)
        elif key.startswith("n_"):
            pass  # "n_verbatim=3 n_copy=8 n_fuse=0" is one combined line, handled below
    # the counts line is space-separated key=value pairs on one line, e.g.
    # "n_verbatim=3 n_copy=8 n_fuse=0" -- _SUMMARY_LINE only matched the
    # first token above, so re-scan any line containing "n_verbatim="
    for line in lines:
        if "n_verbatim=" in line:
            for tok in line.split():
                k, _, v = tok.partition("=")
                if k in ("n_verbatim", "n_copy", "n_fuse"):
                    out[k] = int(v)
    return out
