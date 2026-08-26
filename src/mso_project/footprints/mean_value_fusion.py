"""Footprint for mean_value_fusion_gui.py. Default field is outputs/fusion
(CWD-relative). Note: this pipeline ALSO mutates the shared RMIP SQLite
project.db (a new fused-mechanism row) -- that's rmip_suite's concern
(role "rmip_database"), not tracked again here to avoid double-counting the
same physical .db file under two roles if both footprints are ever scanned
over the same directory."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("*_uncertainty_classification.csv", "fusion_uncertainty_table"),
    FootprintEntry("*.inp", "fused_mechanism_ck"),
    FootprintEntry("*.dat", "fused_mechanism_ck"),
]
