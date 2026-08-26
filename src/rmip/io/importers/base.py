"""
rmip/io/importers/base.py
=========================
W3 — importer ABC + the shared CHEMKIN→YAML conversion helper.

An importer translates a foreign kinetics format into Cantera YAML **with an
explicit assumption report**: every modeling decision (and every construct that
cannot be faithfully represented) is a `Finding` in the returned `CheckReport`
— never a silent lossy conversion. See
``Cantera google groups/W3_foreign_format_importers.md`` (P5).
"""
from __future__ import annotations

import abc
import logging
import os
import shutil
import subprocess
from typing import List, Optional, Tuple

from rmip.analysis.mechanism_check import CheckReport, Finding

log = logging.getLogger("rmip.io.importers")


class MechanismImporter(abc.ABC):
    """One foreign-format backend."""

    format_name: str = "abstract"

    @classmethod
    @abc.abstractmethod
    def sniff(cls, path: str) -> bool:
        """True if ``path`` looks like this backend's format."""

    @abc.abstractmethod
    def import_(self, path: str, out_yaml: str, *,
                thermo: Optional[str] = None, transport: Optional[str] = None,
                gas_only: bool = False) -> Tuple[Optional[str], CheckReport]:
        """Convert ``path`` → ``out_yaml``.

        Returns ``(yaml_path, report)``; ``yaml_path is None`` means the import
        was **refused** (unrepresentable content not acknowledged) — the report
        says exactly why.
        """


# ---------------------------------------------------------------------------
# shared: preflight + ck2yaml + load-verify for a CHEMKIN text
# ---------------------------------------------------------------------------
def convert_chemkin_to_yaml(inp_path: str, out_yaml: str, *,
                            thermo: Optional[str] = None,
                            transport: Optional[str] = None,
                            report: Optional[CheckReport] = None,
                            ) -> Tuple[Optional[str], List[Finding]]:
    """Run the W2 preflight, then ck2yaml, then verify the YAML loads.

    Returns ``(yaml_path_or_None, findings)`` — findings carry the preflight
    results plus any conversion/load error.
    """
    from rmip.io.chemkin_preflight import ChemkinPreflight

    findings: List[Finding] = []
    pf = ChemkinPreflight(inp_path, thermo=thermo, transport=transport)
    pf_report = pf.run()
    findings.extend(pf_report.findings)

    exe = shutil.which("ck2yaml")
    cmd = ([exe] if exe else ["python", "-m", "cantera.ck2yaml"])
    cmd += [f"--input={inp_path}", f"--output={out_yaml}", "--permissive"]
    if thermo:
        cmd.append(f"--thermo={thermo}")
    if transport:
        cmd.append(f"--transport={transport}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as exc:
        findings.append(Finding(
            check="import", severity="error", category="conversion-failure",
            message=f"ck2yaml invocation failed: {exc}",
            location=os.path.basename(inp_path)))
        return None, findings
    if r.returncode != 0 or not os.path.exists(out_yaml):
        tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-8:])
        findings.append(Finding(
            check="import", severity="error", category="conversion-failure",
            message="ck2yaml failed on the (already preflighted) input — "
                    "see detail.",
            location=os.path.basename(inp_path), details={"ck2yaml_tail": tail}))
        return None, findings

    # verify the produced YAML loads
    try:
        import warnings
        import cantera as ct
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gas = ct.Solution(out_yaml)
        findings.append(Finding(
            check="import", severity="info", category="converted",
            message=f"Converted and verified: {gas.n_species} species, "
                    f"{gas.n_reactions} reactions.",
            location=os.path.basename(out_yaml)))
    except Exception as exc:                               # noqa: BLE001
        findings.append(Finding(
            check="import", severity="error", category="load-failure",
            message=f"ck2yaml produced a YAML that Cantera rejects: {exc}",
            location=os.path.basename(out_yaml)))
        return None, findings
    return out_yaml, findings
