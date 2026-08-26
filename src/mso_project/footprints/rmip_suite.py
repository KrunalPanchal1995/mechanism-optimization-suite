"""Footprint for rmip_suite_gui.py and src/rmip/analysis/* broadly
(comparison.py, benchmarking.py, unified_tree.py, sensitivity, b2bdc/
bayesian/surrogate work-dir JSON requests). Tabs shell out to rmip_cli.py
with --out/--out-dir (relative defaults -> CWD-relative). The canonical
state for this pipeline lives in the shared SQLite --db (conventionally
"project.db") -- flat files below are its exports/artifacts, not the
source of truth, but are still worth tracking as report-ready figures/
tables."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("**/comparison_report.json", "rmip_comparison_report"),
    FootprintEntry("**/kinetic_diff.csv", "rmip_diff_table"),
    FootprintEntry("**/thermo_diff.csv", "rmip_diff_table"),
    FootprintEntry("**/*.png", "rmip_figure"),
    FootprintEntry("**/unified_tree*.csv", "rmip_unified_tree"),
    FootprintEntry("**/unified_tree*.json", "rmip_unified_tree"),
    FootprintEntry("**/*sensitivity*.csv", "rmip_sensitivity_table"),
    FootprintEntry("**/project.db", "rmip_database"),
]
