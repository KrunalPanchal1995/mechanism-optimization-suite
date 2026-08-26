"""Footprint for src/mso_compare/ (pipeline.py/merge.py/sim_runner.py) and
nominal_sim_gui.py's own invocation of it. Four independent, YAML-config-
driven output roots, all ``./``-relative by default:
SIM_RUNS/<label>/, MERGE_OUTPUT/{HTC,LTC}/, ERROR_METRICS/{HTC,LTC}/,
PLOTS/{HTC,LTC}/.
"""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("SIM_RUNS/**/target_data.txt", "sim_run_summary"),
    FootprintEntry("SIM_RUNS/**/*.log", "sim_run_log"),
    FootprintEntry("SIM_RUNS/**/*.csv", "sim_run_data"),
    FootprintEntry("MERGE_OUTPUT/**/*.csv", "merged_comparison"),
    FootprintEntry("ERROR_METRICS/**/*.csv", "error_metric"),
    FootprintEntry("PLOTS/**/*.png", "comparison_figure"),
]
