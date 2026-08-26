"""Footprint for forward_uq.py / forward_uq_gui.py.

forward_uq.py's own CSVs/pkl land in bare CWD regardless of --output_dir
(a quirk of the script, not this footprint); plots respect --output_dir
(default "ForwardUQ", relative to CWD unless the GUI resolves it absolute
and runs the subprocess with cwd=that dir, in which case CWD==output_dir
and everything lands together). Patterns cover both.
"""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("forward_uq_statistics.csv", "forward_uq_summary"),
    FootprintEntry("sobol_indices.csv", "sobol_indices"),
    FootprintEntry("unsrt.pkl", "uncertainty_cache"),
    FootprintEntry("target.opt", "forward_uq_config"),
    FootprintEntry("ForwardUQ/**/*.png", "forward_uq_figure"),
    FootprintEntry("ForwardUQ/**/*.pdf", "forward_uq_figure"),
    FootprintEntry("*/**/*.png", "forward_uq_figure"),  # any other --output_dir name
    FootprintEntry("*/**/*.pdf", "forward_uq_figure"),
]
