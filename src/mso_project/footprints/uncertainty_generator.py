"""Footprint for uncertainty_generator_gui.py. Default fields are
outputs/uncertainty.xml and outputs/uncertainty_bounds.csv (CWD-relative)."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("**/*uncertainty_bounds*.csv", "uncertainty_bounds_table"),
    FootprintEntry("**/*.xml", "uncertainty_xml"),
]
