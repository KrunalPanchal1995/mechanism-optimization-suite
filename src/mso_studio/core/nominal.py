"""
mso_studio.core.nominal
=========================
Adapter over :mod:`mso_compare.run_nominal_sim` (a single nominal simulation
run). See ``src/mso_compare/README.md`` for the underlying pipeline.

This module writes a throwaway single-run ``mso_config.yaml`` matching the
schema in ``src/mso_compare/config.py`` / ``template_mso_pipeline.yaml``,
then shells out to ``run_nominal_sim.py --config ... --run ...`` in the
anaconda interpreter (see :mod:`mso_studio.core.backends`).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record


@dataclass
class NominalSimParams:
    """User-facing inputs for one nominal simulation run.

    Parameters
    ----------
    label : str
        Run label, e.g. ``"nominal_htc"``. Also used as the config key and
        the RunRecord label.
    mechanism, targets, thermo_file, trans_file : str
        Paths to the mechanism, ``.target``, thermo, and transport files.
    targets_count : int
        Number of target lines to read from *targets*.
    base_opt_file : str
        Base ``.opt`` YAML template providing all other defaults.
    muqsac_src_dir : str
        Path to ``src/MUQ-SAC`` (added to the backend's ``sys.path``).
    output_dir : str
        Where results are written. Defaults to ``SIM_RUNS/<label>`` under
        the repo root, matching the existing Tkinter ``nominal_sim_gui.py``
        convention.
    uncertainty_data : str, optional
        Dummy path acceptable -- not used for a nominal run.
    """

    label: str
    mechanism: str
    targets: str
    thermo_file: str
    trans_file: str
    targets_count: int
    base_opt_file: str
    muqsac_src_dir: str = str(REPO_ROOT / "src" / "MUQ-SAC")
    output_dir: str = ""
    uncertainty_data: str = ""
    extra_overrides: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(REPO_ROOT / "SIM_RUNS" / self.label)
        if not self.uncertainty_data:
            # Nominal runs don't use uncertainty data; point at the targets
            # file itself so config.SimRunConfig.validate() has *something*
            # that exists rather than warning about a made-up dummy path.
            self.uncertainty_data = self.targets

    def to_sim_config_yaml(self) -> dict:
        """Render the single-``simulations`` entry ``mso_config.yaml`` dict."""
        return {
            "project": {"name": f"mso_studio_{self.label}", "working_dir": str(REPO_ROOT), "fuel": ""},
            "muqsac_src_dir": self.muqsac_src_dir,
            "base_opt_file": self.base_opt_file,
            "simulations": {
                self.label: {
                    "mechanism": self.mechanism,
                    "targets": self.targets,
                    "targets_count": self.targets_count,
                    "thermo_file": self.thermo_file,
                    "trans_file": self.trans_file,
                    "uncertainty_data": self.uncertainty_data,
                    "output_dir": self.output_dir,
                    "extra_overrides": self.extra_overrides,
                }
            },
        }


def run_nominal(
    params: NominalSimParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run one nominal simulation and return its :class:`RunRecord`.

    Raises
    ------
    mso_studio.core.backends.BackendError
        If the backend subprocess exits non-zero. The message is
        user-facing (see :class:`~mso_studio.core.backends.BackendError`);
        the full stdout/stderr is at ``record.log_path`` regardless.
    """
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "_mso_studio_config.yaml"
    config_path.write_text(yaml.safe_dump(params.to_sim_config_yaml(), sort_keys=False))

    command = science_command(
        "mso_compare/run_nominal_sim.py", "--config", str(config_path), "--run", params.label,
    )

    run_config = RunConfig(
        module="nominal_sim",
        label=params.label,
        backend_command=command,
        working_dir=str(REPO_ROOT),
        inputs={
            "mechanism": params.mechanism,
            "targets": params.targets,
            "thermo_file": params.thermo_file,
            "trans_file": params.trans_file,
            "base_opt_file": params.base_opt_file,
        },
        parameters={"targets_count": params.targets_count, "output_dir": params.output_dir},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = out_dir / f"mso_studio_{params.label}.log"
    result = run_backend(
        command, cwd=str(REPO_ROOT), on_line=on_line, cancel_event=cancel_event, log_path=log_path,
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
        return record  # unreachable, raise_if_failed always raises here

    _collect_results(record, out_dir, params.label)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _build_error(result):
    from mso_studio.core.backends import BackendError
    return BackendError(result.command, result.return_code, result.log_path, result.tail)


def _collect_results(record: RunRecord, out_dir: Path, label: str) -> None:
    """Populate ``record.tables``/``results`` from the files ``sim_runner`` writes.

    Layout (``sim_runner.run_nominal_simulation``)::

        output_dir/Plot/Dataset/{Tig,RCM,JSR,Fls}/<dataset_id>.csv
        output_dir/target_data.txt
    """
    dataset_root = out_dir / "Plot" / "Dataset"
    n_tables = 0
    for stage_dir in sorted(dataset_root.glob("*")):
        if not stage_dir.is_dir():
            continue
        for csv_path in sorted(stage_dir.glob("*.csv")):
            key = f"{stage_dir.name}/{csv_path.stem}"
            record.tables[key] = str(csv_path)
            n_tables += 1

    target_data = out_dir / "target_data.txt"
    if target_data.exists():
        record.results["target_data_path"] = str(target_data)

    record.results["n_datasets"] = n_tables
    record.results["output_dir"] = str(out_dir)
