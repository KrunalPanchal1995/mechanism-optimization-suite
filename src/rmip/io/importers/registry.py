"""
rmip/io/importers/registry.py
=============================
W3 — format detection + dispatch for the importer framework (P5).

``detect_format`` sniffs a file; ``import_mechanism`` routes to the matching
backend. New formats drop in by adding a backend to ``_IMPORTERS`` — callers
(CLI `rmip import`, converter GUI) never change.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from rmip.analysis.mechanism_check import CheckReport, Finding
from rmip.io.importers.base import MechanismImporter, convert_chemkin_to_yaml
from rmip.io.importers.opensmoke import OpenSmokeImporter

# sniff order matters: most-specific first
_IMPORTERS = [OpenSmokeImporter]


def detect_format(path: str) -> str:
    """Return 'cantera-yaml' | 'opensmoke' | 'chemkin' | 'unknown'."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        return "cantera-yaml"
    for imp in _IMPORTERS:
        if imp.sniff(path):
            return imp.format_name
    try:
        with open(path, "r", errors="replace") as fh:
            up = fh.read().upper()
    except OSError:
        return "unknown"
    if ("SPECIES" in up or "SPEC" in up) and ("REACTIONS" in up or "REAC" in up):
        return "chemkin"
    if "ELEMENTS" in up or "THERMO" in up:
        return "chemkin"
    return "unknown"


def import_mechanism(path: str, out_yaml: str, fmt: str = "auto", *,
                     thermo: Optional[str] = None,
                     transport: Optional[str] = None,
                     gas_only: bool = False
                     ) -> Tuple[Optional[str], CheckReport]:
    """Import ``path`` → ``out_yaml``; returns ``(yaml_path_or_None, report)``."""
    if fmt == "auto":
        fmt = detect_format(path)

    if fmt == "opensmoke":
        return OpenSmokeImporter().import_(
            path, out_yaml, thermo=thermo, transport=transport,
            gas_only=gas_only)

    if fmt == "chemkin":
        report = CheckReport(mechanism=path, ran_checks=["import"])
        report.loaded = True
        yaml_path, findings = convert_chemkin_to_yaml(
            path, out_yaml, thermo=thermo, transport=transport)
        report.findings = findings
        return yaml_path, report

    report = CheckReport(mechanism=path, ran_checks=["import"])
    report.findings.append(Finding(
        check="import", severity="error", category="unknown-format",
        message=f"Unsupported/unknown format '{fmt}' for {os.path.basename(path)}.",
    ))
    return None, report
