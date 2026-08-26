"""
rmip/io/importers/opensmoke.py
==============================
W3 — importer for OpenSmoke-flavoured CHEMKIN mechanisms (P5).

The community case (`Cantera google groups/W3…`, thread ``chlJWiEaMHQ``): an
OpenSmoke polyethylene-pyrolysis mechanism failed to convert because OpenSmoke's
``LIQUID``/``MATERIAL`` keywords encode a heterogeneous model — bulk condensed
reactants producing gas products with an implicit transport assumption — that
has **no faithful analogue** in Cantera's site/interface phase model. That is a
*modeling* gap, not a syntax gap, so this importer:

1. identifies the condensed-phase content (``MATERIAL`` headers, species inside
   ``LIQUID``/``SOLID`` blocks, and every reaction referencing them);
2. **refuses** to convert while that content is present, with an ERROR finding
   naming each unrepresentable construct — unless the caller passes
   ``gas_only=True`` to *explicitly acknowledge* the approximation;
3. with ``gas_only=True``, strips the condensed content, records every dropped
   species/reaction as a WARNING finding (the auditable assumption report), and
   converts the remaining gas-phase subset via the shared W2-preflight +
   ck2yaml pipeline.

Format note: OpenSmoke condensed-phase syntax is loosely specified in the wild
(CRECK-style solid-fuel files vary); this parser handles the common shapes —
``MATERIAL <name>`` header lines and ``LIQUID``/``SOLID`` … ``END`` blocks
listing condensed species — and is deliberately conservative: anything it does
not recognise stays in the file and is caught by the W2 preflight.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Set, Tuple

from rmip.analysis.mechanism_check import CheckReport, Finding
from rmip.io.importers.base import MechanismImporter, convert_chemkin_to_yaml

log = logging.getLogger("rmip.io.importers.opensmoke")

_CONDENSED_OPENERS = {"LIQUID", "SOLID"}
_MATERIAL_RE = re.compile(r"^\s*MATERIAL\b(.*)$", re.IGNORECASE)


class OpenSmokeImporter(MechanismImporter):
    format_name = "opensmoke"

    # ------------------------------------------------------------------
    @classmethod
    def sniff(cls, path: str) -> bool:
        try:
            with open(path, "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return False
        up = "\n".join(ln.split("!", 1)[0] for ln in text.split("\n")).upper()
        has_ck = "SPECIES" in up and "REACTIONS" in up
        has_ext = bool(_MATERIAL_RE.search(up, re.MULTILINE)) or any(
            re.search(rf"^\s*{kw}\b", up, re.MULTILINE)
            for kw in _CONDENSED_OPENERS)
        # a MATERIAL/LIQUID/SOLID extension inside a CHEMKIN-shaped file
        return has_ck and has_ext

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> Tuple[Set[str], List[str], List[Finding]]:
        """Return (condensed_species, material_names, findings) for the raw text."""
        condensed: Set[str] = set()
        materials: List[str] = []
        findings: List[Finding] = []
        in_block: Optional[str] = None
        for i, line in enumerate(text.split("\n"), start=1):
            code = line.split("!", 1)[0].strip()
            if not code:
                continue
            m = _MATERIAL_RE.match(code)
            if m and in_block is None:
                name = m.group(1).strip() or "(unnamed)"
                materials.append(name)
                findings.append(Finding(
                    check="import", severity="warning", category="material",
                    message=f"OpenSmoke MATERIAL '{name}' — Cantera has no "
                            "bulk-material analogue; its species will be treated "
                            "as condensed-phase content.",
                    location=f"line {i}", details={"line": i, "name": name}))
                continue
            first = code.split()[0].upper()
            if in_block is None and first in _CONDENSED_OPENERS:
                in_block = first
                continue
            if in_block is not None:
                if first == "END":
                    in_block = None
                    continue
                for tok in code.split():
                    condensed.add(tok)
        for sp in sorted(condensed):
            findings.append(Finding(
                check="import", severity="warning", category="condensed-species",
                message=f"Condensed-phase species '{sp}' (LIQUID/SOLID block) — "
                        "Cantera's site/interface model cannot represent "
                        "OpenSmoke's implicit bulk→gas transport assumption "
                        "(see W5).",
                location=sp, details={"species": sp}))
        return condensed, materials, findings

    # ------------------------------------------------------------------
    def strip_to_gas(self, text: str) -> Tuple[str, Set[str], List[Finding]]:
        """Remove MATERIAL/LIQUID/SOLID content + reactions referencing it.

        Returns (gas_text, condensed_species, findings-with-drops).
        """
        condensed, _materials, findings = self.analyze(text)
        out_lines: List[str] = []
        in_block: Optional[str] = None
        in_reactions = False
        for i, line in enumerate(text.split("\n"), start=1):
            code = line.split("!", 1)[0].strip()
            first = code.split()[0].upper() if code else ""
            if _MATERIAL_RE.match(code) and in_block is None:
                continue                                   # drop MATERIAL header
            if in_block is None and first in _CONDENSED_OPENERS:
                in_block = first
                continue
            if in_block is not None:
                if first == "END":
                    in_block = None
                continue                                   # drop block content
            if first in ("REACTIONS", "REAC"):
                in_reactions = True
            elif in_reactions and first == "END":
                in_reactions = False
            elif in_reactions and code and "=" in code:
                toks = set(re.split(r"[+=<>()\s/]+", code.upper()))
                hit = {sp for sp in condensed if sp.upper() in toks}
                if hit:
                    findings.append(Finding(
                        check="import", severity="warning",
                        category="dropped-reaction",
                        message=(f"Dropped heterogeneous reaction '{code.split()[0]}' "
                                 f"(references condensed species {sorted(hit)})."),
                        location=f"line {i}",
                        details={"line": i, "equation": code,
                                 "condensed": sorted(hit)}))
                    out_lines.append("! [rmip import: dropped heterogeneous] "
                                     + line)
                    continue
            out_lines.append(line)
        # also drop condensed names from SPECIES declarations, if present
        text_out = "\n".join(out_lines)
        for sp in condensed:
            text_out = re.sub(rf"(?m)^(\s*){re.escape(sp)}(\s+|$)", r"\1", text_out)
        return text_out, condensed, findings

    # ------------------------------------------------------------------
    def import_(self, path: str, out_yaml: str, *,
                thermo: Optional[str] = None, transport: Optional[str] = None,
                gas_only: bool = False) -> Tuple[Optional[str], CheckReport]:
        report = CheckReport(mechanism=path, ran_checks=["import"])
        report.loaded = True
        with open(path, "r", errors="replace") as fh:
            text = fh.read()

        if not gas_only:
            condensed, materials, findings = self.analyze(text)
            report.findings = findings
            if condensed or materials:
                report.findings.append(Finding(
                    check="import", severity="error",
                    category="unrepresentable-content",
                    message=(f"{len(condensed)} condensed-phase species / "
                             f"{len(materials)} MATERIAL block(s) cannot be "
                             "faithfully represented in Cantera. Re-run with "
                             "--gas-only to acknowledge dropping them and "
                             "convert the gas-phase subset."),
                    details={"condensed_species": sorted(condensed),
                             "materials": materials}))
                return None, report
            # no condensed content after all → plain CHEMKIN path
            gas_text = text
        else:
            gas_text, condensed, findings = self.strip_to_gas(text)
            report.findings = findings
            report.meta["dropped_species"] = sorted(condensed)

        # write the gas-phase CHEMKIN next to the output and convert
        gas_inp = os.path.splitext(out_yaml)[0] + ".gas.inp"
        os.makedirs(os.path.dirname(os.path.abspath(gas_inp)) or ".",
                    exist_ok=True)
        with open(gas_inp, "w") as fh:
            fh.write(gas_text if gas_text.endswith("\n") else gas_text + "\n")

        yaml_path, conv_findings = convert_chemkin_to_yaml(
            gas_inp, out_yaml, thermo=thermo, transport=transport)
        report.findings.extend(conv_findings)
        report.meta["gas_inp"] = gas_inp
        return yaml_path, report
