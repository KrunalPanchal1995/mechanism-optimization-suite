"""
mso_studio.core.sensitivity
=============================
Adapters over the two distinct sensitivity-analysis backends in this repo:

* **Brute-force kinetics SA** (``src/Sensitivity_Analysis/sens_3_params.py``)
  -- 3-parameter (ln A, n, Ea/R) forward/central-difference log sensitivity,
  driven by a ``target.opt`` YAML. Mirrors ``brute_force_sa_gui.py``.
* **Ignition-delay sensitivity** (``rmip idt-sens``, backed by
  ``rmip.simulation.idt_sensitivity.IdtSensitivityAnalyzer``) -- d ln(tau)/d
  ln(k) per reaction, brute-force or CVODES, for one scalar T/P/phi
  condition against a Cantera mechanism. Mirrors ``idt_sensitivity_gui.py``.
"""
from __future__ import annotations

import json
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record
from mso_studio.core.nominal import _build_error


# ---------------------------------------------------------------------------
# Brute-force kinetics SA (target.opt driven)
# ---------------------------------------------------------------------------

@dataclass
class BruteForceSAParams:
    """Inputs for the 3-parameter brute-force sensitivity run.

    Parameters
    ----------
    target_opt_path : str
        Path to a fully-populated ``target.opt`` YAML (``Locations``,
        ``Counts``, ``Stats.SA_scheme`` etc. already set).
    output_dir : str
        Working directory the backend is launched from; ``sens_parameters.pkl``
        and the ``SA_3P_run_*.log`` land here.
    """

    target_opt_path: str
    output_dir: str = ""
    label: str = field(default="")
    plot_style_json: str = ""   # KRUNAL_JULY_23.md point 4 -- optional PlotStyle
                                  # JSON path, forwarded to sens_3_params.py via
                                  # SENS3P_PLOT_STYLE_JSON (see brute_force_sa_gui.py)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(Path(self.target_opt_path).resolve().parent)
        if not self.label:
            self.label = Path(self.target_opt_path).stem


def run_brute_force_sa(
    params: BruteForceSAParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run ``sens_3_params.py <target.opt>`` and return its :class:`RunRecord`."""
    target_opt = Path(params.target_opt_path).resolve()
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = science_command("Sensitivity_Analysis/sens_3_params.py", str(target_opt))

    run_config = RunConfig(
        module="sensitivity_brute_force",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={"target_opt_path": str(target_opt)},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_sa_{params.label}.log"
    env_extra = {"SENS3P_PLOT_STYLE_JSON": params.plot_style_json} if params.plot_style_json else None
    result = run_backend(
        command, cwd=str(out_dir), on_line=on_line, cancel_event=cancel_event, log_path=log_path,
        env_extra=env_extra,
    )
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record
    if result.status != "done":
        record.mark_finished("failed", result.return_code, error=str(_build_error(result)))
        record.save()
        result.raise_if_failed()
        return record  # unreachable

    _collect_sa_results(record, out_dir)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _collect_sa_results(record: RunRecord, out_dir: Path) -> None:
    pkl_path = out_dir / "sens_parameters.pkl"
    if pkl_path.exists():
        record.results["sens_parameters_path"] = str(pkl_path)
        try:
            with pkl_path.open("rb") as fh:
                data = pickle.load(fh)
            record.results["n_targets"] = len(data) if hasattr(data, "__len__") else None
        except (pickle.UnpicklingError, EOFError, OSError):
            record.warnings.append(f"Could not unpickle {pkl_path} for a results preview.")
    logs = sorted(out_dir.glob("SA_3P_run_*.log"))
    if logs:
        record.results["sa_log_path"] = str(logs[-1])


# ---------------------------------------------------------------------------
# Ignition-delay (IDT) sensitivity
# ---------------------------------------------------------------------------

@dataclass
class IdtSensitivityParams:
    """Inputs for ``rmip idt-sens`` -- see ``rmip_cli.py idt-sens --help``.

    Parameters
    ----------
    mechanism : str
        Cantera-format mechanism YAML.
    fuel : str
        Fuel species, e.g. ``"NC7H16"``.
    T_K, P_atm, phi : float
        Initial temperature [K], pressure [atm], equivalence ratio.
    oxidizer : str
        Oxidizer composition string (default air).
    reactor : {"constV", "constP"}
    method : {"brute", "cvodes"}
    idt_definition : {"dTdt", "threshold", "species"}
    species : str, optional
        Required when ``idt_definition == "species"``.
    norm : {"log", "raw"}
    diff : {"central", "forward"}
    delta : float
        Multiplier perturbation fraction (brute only).
    top_n : int
        Number of top reactions to keep in ``record.results``.
    workers : int
        Parallel processes for brute-force.
    """

    mechanism: str
    fuel: str
    T_K: float
    P_atm: float
    phi: float
    oxidizer: str = "O2:1.0, N2:3.76"
    reactor: str = "constV"
    method: str = "brute"
    idt_definition: str = "dTdt"
    species: Optional[str] = None
    threshold_dT: float = 400.0
    norm: str = "log"
    diff: str = "central"
    delta: float = 0.05
    top_n: int = 20
    workers: int = 1
    output_dir: str = ""
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "SIM_RUNS" / "idt_sensitivity")
        if not self.label:
            self.label = f"{Path(self.mechanism).stem}_phi{self.phi}_T{self.T_K:g}_P{self.P_atm:g}"


def run_idt_sensitivity(
    params: IdtSensitivityParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run ``rmip_cli.py idt-sens ...`` and return its :class:`RunRecord`."""
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{params.label}.json"

    args = [
        str(Path(params.mechanism).resolve()),
        "--fuel", params.fuel,
        "--oxidizer", params.oxidizer,
        "--T", str(params.T_K),
        "--P", str(params.P_atm),
        "--phi", str(params.phi),
        "--reactor", params.reactor,
        "--method", params.method,
        "--idt-def", params.idt_definition,
        "--threshold-dT", str(params.threshold_dT),
        "--norm", params.norm,
        "--diff", params.diff,
        "--delta", str(params.delta),
        "--top", str(params.top_n),
        "--workers", str(params.workers),
        "--json", str(json_path),
    ]
    if params.species:
        args += ["--species", params.species]

    command = science_command(REPO_ROOT / "rmip_cli.py", "idt-sens", *args)

    run_config = RunConfig(
        module="idt_sensitivity",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={"mechanism": params.mechanism},
        parameters={
            "fuel": params.fuel, "T_K": params.T_K, "P_atm": params.P_atm, "phi": params.phi,
            "reactor": params.reactor, "method": params.method,
            "idt_definition": params.idt_definition,
        },
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_idt_sens_{params.label}.log"
    result = run_backend(
        command, cwd=str(out_dir), on_line=on_line, cancel_event=cancel_event, log_path=log_path,
    )
    record.log_path = str(log_path)
    record.log_tail = result.tail[-200:]

    if result.status == "cancelled":
        record.mark_finished("cancelled", result.return_code)
        record.save()
        return record
    if result.status != "done":
        record.mark_finished("failed", result.return_code, error=str(_build_error(result)))
        record.save()
        result.raise_if_failed()
        return record  # unreachable

    if json_path.exists():
        record.results["idt_sensitivity"] = json.loads(json_path.read_text())
        record.tables["idt_sensitivity_json"] = str(json_path)
    record.mark_finished("done", result.return_code)
    record.save()
    return record
