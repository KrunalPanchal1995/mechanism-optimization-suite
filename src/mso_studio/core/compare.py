"""
mso_studio.core.compare
=========================
Adapter over the full 4-phase MSO pipeline (``src/run_mso_pipeline.py`` ->
:class:`mso_compare.pipeline.MSO_Pipeline`): simulate -> merge -> error
metrics -> plots.

Unlike :mod:`mso_studio.core.nominal` (which synthesizes a throwaway
single-run config), this adapter takes an existing ``mso_config.yaml`` path
-- the compare pipeline's config format is rich enough (up to six run
labels: nominal/stage1/stage2 x htc/ltc) that hand-building it from GUI
widgets belongs in the GUI module, not here. See
``src/mso_compare/template_mso_pipeline.yaml`` for the schema.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from mso_studio.core.backends import science_command, run_backend, REPO_ROOT
from mso_studio.core.runrecord import RunConfig, RunRecord, new_record
from mso_studio.core.nominal import _build_error  # shared BackendError builder


@dataclass
class ComparePipelineParams:
    """User-facing inputs for one 4-phase MSO-compare run.

    Parameters
    ----------
    config_path : str
        Path to an ``mso_config.yaml`` (see
        ``src/mso_compare/template_mso_pipeline.yaml``).
    regime : {"htc", "ltc", "both"}
        Which regime Phases 2-4 process.
    skip_simulations, skip_merge, skip_error_metrics, skip_plots : bool
        Mirrors the identically-named ``run_mso_pipeline.py`` flags -- lets
        a user re-run only the later phases against existing outputs.
    dry_run : bool
        Print the plan without executing.
    """

    config_path: str
    regime: str = "both"
    skip_simulations: bool = False
    skip_merge: bool = False
    skip_error_metrics: bool = False
    skip_plots: bool = False
    dry_run: bool = False
    label: str = field(default="")

    def __post_init__(self) -> None:
        if not self.label:
            self.label = Path(self.config_path).stem


def run_compare(
    params: ComparePipelineParams,
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> RunRecord:
    """Run the 4-phase MSO compare pipeline and return its :class:`RunRecord`.

    Raises
    ------
    mso_studio.core.backends.BackendError
        If the backend subprocess exits non-zero.
    """
    config_path = Path(params.config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    working_dir = raw.get("project", {}).get("working_dir") or str(config_path.parent)

    args = ["--config", str(config_path), "--regime", params.regime]
    if params.skip_simulations:
        args.append("--skip-simulations")
    if params.skip_merge:
        args.append("--skip-merge")
    if params.skip_error_metrics:
        args.append("--skip-error-metrics")
    if params.skip_plots:
        args.append("--skip-plots")
    if params.dry_run:
        args.append("--dry-run")
    command = science_command("run_mso_pipeline.py", *args)

    run_config = RunConfig(
        module="mso_compare",
        label=params.label,
        backend_command=command,
        working_dir=working_dir,
        inputs={"config_path": str(config_path)},
        parameters={"regime": params.regime, "dry_run": params.dry_run},
    )
    record = new_record(run_config)
    record.capture_input_hashes()

    log_path = Path(working_dir) / f"mso_studio_compare_{params.label}.log"
    result = run_backend(
        command, cwd=working_dir, on_line=on_line, cancel_event=cancel_event, log_path=log_path,
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

    _collect_results(record, Path(working_dir), raw)
    record.mark_finished("done", result.return_code)
    record.save()
    return record


def _collect_results(record: RunRecord, working_dir: Path, raw: dict) -> None:
    """Populate ``record.tables``/``figures`` from the pipeline's output dirs.

    Reads the ``error_metrics`` / ``plots`` sections of the raw config to
    find where Phase 3/4 wrote their CSVs and PDFs (paths there are relative
    to ``project.working_dir``, matching how ``run_mso_pipeline.py`` itself
    resolves them when launched with that directory as ``cwd``).
    """
    err_cfg = raw.get("error_metrics", {})
    plot_cfg = raw.get("plots", {})

    for regime_key in ("htc_output_dir", "ltc_output_dir"):
        err_dir = err_cfg.get(regime_key)
        if err_dir:
            p = (working_dir / err_dir).resolve()
            for csv_path in sorted(p.rglob("*.csv")):
                key = str(csv_path.relative_to(p)) if p.exists() else csv_path.name
                record.tables[f"error_metrics/{regime_key.split('_')[0]}/{key}"] = str(csv_path)

        plot_dir = plot_cfg.get(regime_key)
        if plot_dir:
            p = (working_dir / plot_dir).resolve()
            for pdf_path in sorted(p.rglob("*.pdf")):
                key = str(pdf_path.relative_to(p)) if p.exists() else pdf_path.name
                record.figures[f"plots/{regime_key.split('_')[0]}/{key}"] = str(pdf_path)

    record.results["working_dir"] = str(working_dir)
    record.results["n_tables"] = len(record.tables)
    record.results["n_figures"] = len(record.figures)
