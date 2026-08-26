"""Footprint for fuse_kld.py. Everything lands directly in the caller-
supplied --out-dir (no further subdirectory nesting)."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("*_union_spec.json", "fusion_union_spec"),
    FootprintEntry("*_mso_bounds.json", "fusion_bounds"),
    FootprintEntry("*_fusion_summary.json", "fusion_summary"),
    FootprintEntry("*_priority_provenance.json", "fusion_provenance"),
    FootprintEntry("*.yaml", "fused_mechanism_yaml"),
    FootprintEntry("*.inp", "fused_mechanism_ck"),
    FootprintEntry("*.dat", "fused_mechanism_ck"),
]
