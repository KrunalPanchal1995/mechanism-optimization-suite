"""
mso_studio.core.optimization
==============================
Adapter over conventional mechanism optimization
(``src/MUQ-SAC/run_script.py``): PRS-surrogate posterior optimization of
Arrhenius A-factors against ignition-delay/flame-speed targets.

``run_script.py`` has **no** ``if __name__ == "__main__":`` guard -- it runs
its pipeline as top-level module code on import. It must always be launched
as a subprocess (never imported), exactly as
``rmip.simulation.muqsac_bridge.MuqsacBridge`` already does.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mso_studio.core.backends import run_backend, REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record
from mso_studio.core.nominal import _build_error

RUN_SCRIPT_PATH = REPO_ROOT / "src" / "MUQ-SAC" / "run_script.py"

#: Result files run_script.py writes into its cwd on success (see
#: src/MUQ-SAC/run_script.py + OptimizationTool2_0.py / MechManipulator2_0.py).
_RESULT_FILES = {
    "optimization_cost": "optimization_cost.csv",
    "prediction_errors": "Prediction_errors.csv",
    "sens_parameters": "sens_parameters.pkl",
    "solution_zeta": "solution_zeta.save",
    "posterior_cov": "posterior_cov.pkl",
    "optimized_mechanism": "Optimized_mech.yaml",
}


@dataclass
class OptimizationParams:
    """Inputs for a conventional (MUQ-SAC) optimization run.

    Parameters
    ----------
    target_opt_path : str
        Path to a fully-populated ``target.opt`` YAML (mechanism, targets,
        uncertainty data, PRS/annealing hyperparameters under ``Stats``).
    output_dir : str
        Working directory the optimizer is launched from -- this is where
        ``Optimized_mech.yaml`` and every other result file land.
    """

    target_opt_path: str
    output_dir: str = ""
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(Path(self.target_opt_path).resolve().parent)
        if not self.label:
            self.label = Path(self.target_opt_path).stem


def run_optimization(
    params: OptimizationParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run ``MUQ-SAC/run_script.py <target.opt>`` and return its :class:`RunRecord`.

    This can be a long optimization (GA + L-BFGS-B / SLSQP over many PRS
    evaluations) -- always invoke from a worker thread, never the GUI thread.
    """
    target_opt = Path(params.target_opt_path).resolve()
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [SCIENCE_PYTHON, str(RUN_SCRIPT_PATH), str(target_opt)]

    run_config = RunConfig(
        module="optimization",
        label=params.label,
        backend_command=command,
        working_dir=str(out_dir),
        inputs={"target_opt_path": str(target_opt)},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_optimization_{params.label}.log"
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

    _collect_results(record, out_dir)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _collect_results(record: RunRecord, out_dir: Path) -> None:
    for key, filename in _RESULT_FILES.items():
        p = out_dir / filename
        if p.exists():
            if p.suffix == ".csv":
                record.tables[key] = str(p)
            else:
                record.results[key] = str(p)

    plots_dir = out_dir / "Opt" / "Plots" / "Posterior"
    if plots_dir.is_dir():
        for i, fig_path in enumerate(sorted(plots_dir.glob("*.pdf"))):
            record.figures[f"posterior/{fig_path.stem}"] = str(fig_path)

    optimized_mech = out_dir / "Optimized_mech.yaml"
    if optimized_mech.exists():
        record.results["optimized_mechanism_path"] = str(optimized_mech)
