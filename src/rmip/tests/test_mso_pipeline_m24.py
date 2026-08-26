"""
Regression coverage for M24: mso_compare.pipeline.MSO_Pipeline.run_simulations()
used to hand every configured run to a shared-process ThreadPoolExecutor, but
sim_runner.run_nominal_simulation() changes the *process-wide* current working
directory (os.chdir) and swaps the *process-wide* root logger's FileHandler
around each run's solve. Two runs racing on either one could cross-contaminate
each other's outputs.

The fix (src/mso_compare/pipeline.py) switched to a ProcessPoolExecutor so each
run gets its own OS process -- os.chdir()/logging state in one run's process
can never be observed by another run's process. This test proves the isolation
structurally (every worker reports the cwd it actually observed while "inside"
its own run matched its own directory), not by hoping a timing race doesn't
fire.
"""
import os
import time
from pathlib import Path

import pytest

from mso_compare import pipeline as mso_pipeline
from mso_compare.config import PipelineConfig, SimRunConfig, MergeConfig, ErrorMetricsConfig, PlotConfig


def _fake_run_nominal_simulation(sim_run_cfg, base_opt_file, muqsac_dir="", dry_run=False, raw=None):
    """Stand-in for sim_runner.run_nominal_simulation that mimics its real
    chdir-around-the-solve shape, staggered so concurrent workers overlap."""
    my_dir = Path(sim_run_cfg.output_dir) / "NOMINAL"
    my_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(my_dir)
    try:
        time.sleep(0.2)
        observed_cwd = Path.cwd()
    finally:
        os.chdir(original_cwd)

    status = "done" if observed_cwd == my_dir else "CWD_CORRUPTED"
    (my_dir / "observed_cwd.txt").write_text(f"{status}: expected {my_dir}, saw {observed_cwd}")
    return {
        "label": sim_run_cfg.label,
        "output_dir": str(sim_run_cfg.output_dir),
        "plot_dir": str(my_dir),
        "status": status,
    }


def _make_config(tmp_path, n_runs=4):
    sims = {}
    for i in range(n_runs):
        label = f"run{i}"
        sims[label] = SimRunConfig(
            label=label,
            mechanism="dummy.inp",
            targets="dummy.target",
            targets_count=1,
            thermo_file="dummy.therm",
            trans_file="dummy.trans",
            uncertainty_data="",
            output_dir=str(tmp_path / label),
        )
    return PipelineConfig(
        project_name="m24_test",
        working_dir=str(tmp_path),
        fuel="H2",
        base_opt_file=str(tmp_path / "base.opt"),
        muqsac_src_dir="",
        simulations=sims,
        merge=MergeConfig(),
        error_metrics=ErrorMetricsConfig(),
        plots=PlotConfig(),
    )


def test_concurrent_runs_do_not_cross_contaminate_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(mso_pipeline, "run_nominal_simulation", _fake_run_nominal_simulation)
    # Force every run into the same worker pool regardless of host CPU count.
    monkeypatch.setattr(mso_pipeline, "_thread_cap", lambda: 4)

    cfg = _make_config(tmp_path, n_runs=4)
    pipe = mso_pipeline.MSO_Pipeline(cfg)
    pipe.run_simulations()

    for label, sim_cfg in cfg.simulations.items():
        marker = Path(sim_cfg.output_dir) / "NOMINAL" / "observed_cwd.txt"
        assert marker.exists(), f"{label} never ran"
        content = marker.read_text()
        assert content.startswith("done"), f"{label} observed a corrupted cwd: {content}"
