"""Footprint for brute_force_sa_gui.py. Default field is outputs/sa
(CWD-relative). The GUI writes its own config/target files directly into
`out`, then shells out to sens_3_params.py with cwd=out -- so that tool's
real SA outputs (sens_parameters.pkl / rxn_list*.csv, the same shape
run_script.py's own Partial-PRS SA branch produces) land in the same
directory and are covered by the sensitivity_cache/reaction_index patterns
below, reusing run_script's role names since they are the same artifacts."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("target.opt", "sa_config"),
    FootprintEntry("plot_style.json", "sa_plot_style"),
    FootprintEntry("addendum_working.add", "sa_addendum"),
    FootprintEntry("targets_selected.target", "sa_target_set"),
    FootprintEntry("_filtered_*.target", "sa_target_set"),
    FootprintEntry("sens_parameters.pkl", "sensitivity_cache"),
    FootprintEntry("sens_3p_parameters.pkl", "sensitivity_cache"),
    FootprintEntry("rxn_list*.csv", "reaction_index"),
    FootprintEntry("SA_tool*.out", "sensitivity_log"),
]
