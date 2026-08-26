"""Footprint for mechanism_converter_gui.py. Default out_dir field is
outputs/converted (CWD-relative). Optional ingest target is a separate
RMIP SQLite db (default <repo>/outputs/project.db) -- tracked here as
"mechanism_database" since, unlike mean_value_fusion's db mutation, this
pipeline's db is usually a fresh/dedicated file, not the same shared db
other rmip_suite tabs write to."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("**/fixed_*.inp", "repaired_mechanism_ck"),
    FootprintEntry("**/*.yaml", "converted_mechanism"),
    FootprintEntry("**/project.db", "mechanism_database"),
]
