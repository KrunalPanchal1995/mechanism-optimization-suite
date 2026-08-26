"""
rmip/analysis/thermo_provenance.py
==================================
W4 — Thermodynamic-data provenance & cross-reference checker.

Closes community problems **P10** (Arkane-computed thermochemistry for K-salts
"way off" vs JANAF — ΔHf/ΔG systematically wrong while Cp and S were close) and
**P27** (community dispute over which O3 NASA-polynomial dataset is "correct");
surfaces the **P19** symptom (charged species — cross-linked to the existing
charge-balance check). See ``Cantera google groups/W4_thermo_transport_provenance.md``.

For each species in a mechanism, the checker compares the NASA-7-derived
ΔHf(298.15), S°(298.15) and Cp(298.15) against an authoritative reference and
flags outliers with the **numeric discrepancy** and a classification:

* ``systematic-enthalpy`` — ΔHf diverges while S and Cp match (the P10
  signature: a wrong reference-atom enthalpy / formation offset, not a bad fit);
* ``distributed`` — several properties diverge (a genuinely bad polynomial).

References
----------
* **Built-in subset** (default): ATcT/CODATA-derived ΔHf°(298)/S°(298)/Cp(298)
  for the common combustion species — physical constants, redistributable.
  Verified against Cantera's shipped ``h2o2.yaml``/``gri30.yaml``: clean
  mechanisms deviate ≤ 1.9 kJ/mol, so the default 8 kJ/mol tolerance yields
  zero false positives on good data.
* **User-supplied file** (``refs=`` / ``--refs``): a CHEMKIN-format THERMO file
  (e.g. Burcat's ``THERM.DAT`` or an ATcT export) — parsed as NASA-7 records
  and evaluated the same way, so the full database can be used without this
  repo vendoring it.

Matching is deliberately conservative: a mechanism species is compared only when
its **name matches (case-insensitive) and its elemental composition matches**
the reference entry — composition-only matching would confuse isomers.

Repair philosophy (per the W4 plan): provenance is a human call — outliers are
**warnings with choice fixes**, never auto-fixed. The offered repair for the
systematic case, ``shift_enthalpy_to_ref``, shifts the NASA-7 ``a6`` of both
temperature ranges by −ΔH/R, which moves ΔHf onto the reference while leaving
Cp(T) and S(T) untouched (a6 appears only in H).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rmip.analysis.mechanism_check import Finding, FixOption

log = logging.getLogger("rmip.analysis.thermo_provenance")

R_GAS = 8.314_462_618          # J mol⁻¹ K⁻¹
T_REF = 298.15                 # K


# ===========================================================================
# Reference data model
# ===========================================================================
@dataclass
class RefEntry:
    name: str
    hf298_kJmol: float
    s298_JmolK: float
    cp298_JmolK: Optional[float] = None
    composition: Optional[Dict[str, float]] = None   # None → skip comp guard
    source: str = "builtin-ATcT/CODATA"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "hf298_kJmol": self.hf298_kJmol,
                "s298_JmolK": self.s298_JmolK, "cp298_JmolK": self.cp298_JmolK,
                "composition": self.composition, "source": self.source}


def _e(name, hf, s, cp, comp) -> Tuple[str, RefEntry]:
    return name, RefEntry(name=name, hf298_kJmol=hf, s298_JmolK=s,
                          cp298_JmolK=cp, composition=comp)


# Curated ATcT/CODATA values (ΔHf°298 kJ/mol, S°298 J/mol/K, Cp°298 J/mol/K).
# These are physical constants from the public ATcT/CODATA/JANAF tables.
BUILTIN_REFS: Dict[str, RefEntry] = dict([
    _e("H2",    0.0,     130.68, 28.84, {"H": 2}),
    _e("O2",    0.0,     205.15, 29.38, {"O": 2}),
    _e("N2",    0.0,     191.61, 29.12, {"N": 2}),
    _e("AR",    0.0,     154.85, 20.79, {"Ar": 1}),
    _e("HE",    0.0,     126.15, 20.79, {"He": 1}),
    _e("H",     217.998, 114.72, 20.79, {"H": 1}),
    _e("O",     249.229, 161.06, 21.91, {"O": 1}),
    _e("OH",    37.49,   183.74, 29.89, {"H": 1, "O": 1}),
    _e("H2O",   -241.826, 188.84, 33.59, {"H": 2, "O": 1}),
    _e("HO2",   12.27,   229.10, 34.89, {"H": 1, "O": 2}),
    _e("H2O2",  -135.46, 232.95, 43.10, {"H": 2, "O": 2}),
    _e("CO",    -110.53, 197.66, 29.14, {"C": 1, "O": 1}),
    _e("CO2",   -393.51, 213.79, 37.13, {"C": 1, "O": 2}),
    _e("CH4",   -74.52,  186.37, 35.69, {"C": 1, "H": 4}),
    _e("CH3",   146.37,  194.20, 38.69, {"C": 1, "H": 3}),
    _e("C2H2",  228.27,  200.93, 44.05, {"C": 2, "H": 2}),
    _e("C2H4",  52.45,   219.32, 43.56, {"C": 2, "H": 4}),
    _e("C2H6",  -83.91,  229.16, 52.49, {"C": 2, "H": 6}),
    _e("CH2O",  -109.19, 218.76, 35.39, {"C": 1, "H": 2, "O": 1}),
    _e("HCO",   41.80,   224.65, 34.60, {"C": 1, "H": 1, "O": 1}),
    _e("CH3OH", -200.71, 239.90, 44.06, {"C": 1, "H": 4, "O": 1}),
    _e("NO",    91.12,   210.76, 29.86, {"N": 1, "O": 1}),
    _e("NO2",   34.02,   240.04, 36.97, {"N": 1, "O": 2}),
    _e("N2O",   81.60,   219.96, 38.60, {"N": 2, "O": 1}),
    _e("NH3",   -45.55,  192.77, 35.65, {"N": 1, "H": 3}),
])


# ===========================================================================
# NASA-7 property evaluation (298.15 K, low-T branch)
# ===========================================================================
def nasa7_props_298(coeffs_low: Sequence[float]) -> Tuple[float, float, float]:
    """Return (Hf298 [kJ/mol], S298 [J/mol/K], Cp298 [J/mol/K]) from the low-T
    NASA-7 branch. For standard formation-referenced polynomials H(298.15) *is*
    the enthalpy of formation."""
    a = coeffs_low
    T = T_REF
    H = R_GAS * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4
                     + a[4]*T**4/5 + a[5]/T) / 1000.0
    S = R_GAS * (a[0]*math.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3
                 + a[4]*T**4/4 + a[6])
    Cp = R_GAS * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)
    return H, S, Cp


# ===========================================================================
# The checker
# ===========================================================================
@dataclass
class SpeciesComparison:
    name: str
    hf_mech: float
    hf_ref: float
    s_mech: float
    s_ref: float
    cp_mech: float
    cp_ref: Optional[float]
    source: str
    flagged: bool = False
    classification: str = "ok"

    @property
    def d_hf(self) -> float:
        return self.hf_mech - self.hf_ref

    @property
    def d_s(self) -> float:
        return self.s_mech - self.s_ref

    @property
    def d_cp_rel(self) -> Optional[float]:
        if self.cp_ref in (None, 0):
            return None
        return (self.cp_mech - self.cp_ref) / self.cp_ref

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "hf_mech_kJmol": self.hf_mech,
                "hf_ref_kJmol": self.hf_ref, "d_hf_kJmol": self.d_hf,
                "s_mech_JmolK": self.s_mech, "s_ref_JmolK": self.s_ref,
                "d_s_JmolK": self.d_s, "cp_mech_JmolK": self.cp_mech,
                "cp_ref_JmolK": self.cp_ref, "d_cp_rel": self.d_cp_rel,
                "source": self.source, "flagged": self.flagged,
                "classification": self.classification}


class ThermoProvenanceChecker:
    """Cross-check a mechanism's species thermo against reference data.

    Parameters
    ----------
    refs : dict[str, RefEntry] | None
        Reference entries keyed by upper-case species name. Default: the
        built-in ATcT/CODATA subset. Use :func:`load_chemkin_thermo_refs` to
        load a user-supplied Burcat/ATcT CHEMKIN THERMO file.
    tol_hf_kJ, tol_s_J, tol_cp_rel : float
        Flagging tolerances for |ΔHf| [kJ/mol], |ΔS| [J/mol/K] and the relative
        Cp deviation. Defaults chosen so clean literature mechanisms
        (h2o2/GRI-3.0) produce zero flags while genuine offsets are caught.
    """

    def __init__(self, refs: Optional[Dict[str, RefEntry]] = None,
                 tol_hf_kJ: float = 8.0, tol_s_J: float = 10.0,
                 tol_cp_rel: float = 0.08) -> None:
        self.refs = refs if refs is not None else dict(BUILTIN_REFS)
        self.tol_hf_kJ = tol_hf_kJ
        self.tol_s_J = tol_s_J
        self.tol_cp_rel = tol_cp_rel

    # ------------------------------------------------------------------
    def compare_gas(self, gas) -> List[SpeciesComparison]:
        """Compare every matchable species of a loaded ``cantera.Solution``."""
        out: List[SpeciesComparison] = []
        for sp in gas.species():
            th = sp.thermo
            if type(th).__name__ != "NasaPoly2":
                continue
            ref = self._match(sp.name, dict(sp.composition))
            if ref is None:
                continue
            c = list(th.coeffs)                    # [Tmid, high7, low7]
            out.append(self._compare_one(sp.name, c[8:15], ref))
        return out

    def _match(self, name: str, comp: Dict[str, float]) -> Optional[RefEntry]:
        ref = self.refs.get(name.upper())
        if ref is None:
            return None
        if ref.composition is not None:
            want = {k.upper(): float(v) for k, v in ref.composition.items()}
            have = {k.upper(): float(v) for k, v in comp.items() if v}
            if want != have:
                log.debug("composition mismatch for %s: %s vs ref %s",
                          name, have, want)
                return None                        # isomer / different species
        return ref

    def _compare_one(self, name: str, coeffs_low: Sequence[float],
                     ref: RefEntry) -> SpeciesComparison:
        hf, s, cp = nasa7_props_298(coeffs_low)
        cmpn = SpeciesComparison(
            name=name, hf_mech=hf, hf_ref=ref.hf298_kJmol,
            s_mech=s, s_ref=ref.s298_JmolK,
            cp_mech=cp, cp_ref=ref.cp298_JmolK, source=ref.source)
        hf_bad = abs(cmpn.d_hf) > self.tol_hf_kJ
        s_bad = abs(cmpn.d_s) > self.tol_s_J
        cp_bad = (cmpn.d_cp_rel is not None
                  and abs(cmpn.d_cp_rel) > self.tol_cp_rel)
        if hf_bad or s_bad or cp_bad:
            cmpn.flagged = True
            # the P10 signature: enthalpy off, shape (S, Cp) fine
            cmpn.classification = ("systematic-enthalpy"
                                   if hf_bad and not s_bad and not cp_bad
                                   else "distributed")
        return cmpn

    # ------------------------------------------------------------------
    def findings_for_gas(self, gas) -> List[Finding]:
        """Emit `Finding`s (mechanism_check model) for a loaded Solution."""
        comparisons = self.compare_gas(gas)
        findings: List[Finding] = []
        n_flagged = 0
        for c in comparisons:
            if not c.flagged:
                continue
            n_flagged += 1
            sysm = c.classification == "systematic-enthalpy"
            fixes = [FixOption(
                id="shift_enthalpy_to_ref", kind="choice", recommended=sysm,
                label=f"Shift ΔHf to reference ({c.hf_ref:+.2f} kJ/mol)",
                description="Shift NASA-7 a6 of both T-ranges by −ΔH/R — moves "
                            "the formation enthalpy onto the reference while "
                            "leaving Cp(T) and S(T) untouched. Recommended only "
                            "for the systematic-enthalpy case.",
                params={"species": c.name,
                        "delta_H_J_mol": c.d_hf * 1000.0}),
                FixOption(id="ignore", kind="choice",
                          label="Keep as-is (acknowledge)",
                          description="Reference datasets can legitimately "
                                      "disagree (the O3 dispute, P27) — keep the "
                                      "mechanism's value.",
                          params={"species": c.name})]
            findings.append(Finding(
                check="thermo_provenance", severity="warning",
                category="thermo-provenance",
                message=(f"{c.name}: ΔHf(298)={c.hf_mech:+.2f} kJ/mol vs "
                         f"reference {c.hf_ref:+.2f} (Δ={c.d_hf:+.2f}); "
                         f"ΔS={c.d_s:+.2f} J/mol/K"
                         + (f"; ΔCp={c.d_cp_rel:+.1%}" if c.d_cp_rel is not None
                            else "")
                         + f" — {c.classification} [{c.source}]"),
                location=c.name,
                details=c.to_dict(),
                fix_options=fixes))
        if comparisons:
            findings.append(Finding(
                check="thermo_provenance", severity="info",
                category="thermo-provenance-summary",
                message=(f"Cross-checked {len(comparisons)} species against "
                         f"reference thermo data; {n_flagged} outlier(s)."),
                details={"n_matched": len(comparisons), "n_flagged": n_flagged}))
        return findings


# ===========================================================================
# User-supplied reference file (CHEMKIN THERMO format, e.g. Burcat THERM.DAT)
# ===========================================================================
def load_chemkin_thermo_refs(path: str,
                             source: Optional[str] = None
                             ) -> Dict[str, RefEntry]:
    """Parse a CHEMKIN-format THERMO file into reference entries.

    Best-effort NASA-7 record reader (4-line records flagged 1/2/3/4 in column
    80). Composition fields are not parsed — user-file matching is name-based —
    so isomer-ambiguous names in a user file are the user's responsibility.
    """
    src = source or f"user-file:{os.path.basename(path)}"
    refs: Dict[str, RefEntry] = {}
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if len(line) >= 80 and line[79] == "1" and line[:1] not in ("!", " ") \
                or (len(line) >= 80 and line[79] == "1" and line[:16].strip()):
            name = line[:18].split()
            if not name:
                i += 1
                continue
            name = name[0]
            try:
                nums: List[float] = []
                for j in (1, 2, 3):
                    row = lines[i + j]
                    for k in range(0, 75, 15):
                        txt = row[k:k + 15].strip()
                        if txt:
                            nums.append(float(txt))
                high, low = nums[0:7], nums[7:14]
                if len(low) == 7:
                    hf, s, cp = nasa7_props_298(low)
                    refs[name.upper()] = RefEntry(
                        name=name, hf298_kJmol=hf, s298_JmolK=s,
                        cp298_JmolK=cp, composition=None, source=src)
                i += 4
                continue
            except (ValueError, IndexError):
                pass
        i += 1
    log.info("loaded %d reference thermo records from %s", len(refs), path)
    return refs


# ===========================================================================
# Text rendering (for the CLI)
# ===========================================================================
def render_comparison_table(comparisons: List[SpeciesComparison]) -> str:
    lines = [
        f"{'species':<12} {'Hf_mech':>9} {'Hf_ref':>9} {'ΔHf':>7}  "
        f"{'ΔS':>6}  {'ΔCp':>7}  class",
        "-" * 72,
    ]
    for c in sorted(comparisons, key=lambda x: -abs(x.d_hf)):
        dcp = f"{c.d_cp_rel:+.1%}" if c.d_cp_rel is not None else "   n/a"
        mark = " ⚠" if c.flagged else ""
        lines.append(
            f"{c.name:<12} {c.hf_mech:>+9.2f} {c.hf_ref:>+9.2f} "
            f"{c.d_hf:>+7.2f}  {c.d_s:>+6.2f}  {dcp:>7}  "
            f"{c.classification}{mark}")
    return "\n".join(lines)
