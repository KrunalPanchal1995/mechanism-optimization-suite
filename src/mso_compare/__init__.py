"""
mso_compare
===========
Unified framework for MSO (Multi-Stage Optimization) mechanism comparison.

Modules
-------
config          — YAML pipeline configuration loader and dataclasses
utils           — Path helpers, dataset-ID parsing, .opt file overrides, logging
sim_runner      — Nominal simulation runner (rewrite of run_nominal_sim.py)
merge           — Merges per-run CSV results into organised (φ, P) structures
error_metrics   — Computes and writes per-condition chi-squared error CSVs
plotter         — Unified flexible IDT comparison plotter
pipeline        — End-to-end MSO pipeline orchestrator
"""

__version__ = "1.0.0"
__author__  = "MSO Framework"

from .config       import load_config, PipelineConfig, SimRunConfig
from .pipeline     import MSO_Pipeline
from .sim_runner   import run_nominal_simulation
from .merge        import merge_simulation_results
from .error_metrics import generate_error_csvs, compute_errors_from_df
from .plotter      import plot_comparison, plot_two_threshold_comparison

__all__ = [
    "load_config",
    "PipelineConfig",
    "SimRunConfig",
    "MSO_Pipeline",
    "run_nominal_simulation",
    "merge_simulation_results",
    "generate_error_csvs",
    "compute_errors_from_df",
    "plot_comparison",
    "plot_two_threshold_comparison",
]
