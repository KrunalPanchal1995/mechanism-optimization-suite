"""Footprint for nominal_sim_gui.py / src/MUQ-SAC/run_nominal_sim.py.
Strongly CWD-relative (chdir-based): writes under a "NOMINAL/" directory it
creates in the invocation's working_dir."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("NOMINAL/**/target_data.txt", "nominal_sim_summary"),
    FootprintEntry("NOMINAL/Plot/Dataset/**/*.csv", "nominal_sim_data"),
    FootprintEntry("NOMINAL/**/*.log", "nominal_sim_log"),
]
