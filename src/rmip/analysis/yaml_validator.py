"""
rmip/analysis/yaml_validator.py
===============================
W1 — Cantera YAML schema validator & version-migration linter.

Closes community problems **P1** (LLM-generated YAML uses the stale CTI keyword
`stoichiometric` instead of `fixed-stoichiometry`, and "ask the model to check
it" didn't catch it) and **P29** (CTI→YAML keyword renames are a recurring silent
trap). See ``Cantera google groups/W1_yaml_validator_migration.md``.

AI- and human-authored Cantera YAML routinely carries deprecated CTI-era phase
`thermo:` models and renamed reaction `type:` values. Cantera only fails at
*runtime* with an opaque, banner-wrapped ``CanteraError`` ("Unknown reaction rate
type 'PLOG'"), and it fails on the *first* offender, so a file with several stale
keywords is fixed one frustrating reload at a time.

This linter walks the YAML **as a dict** (so it works even when the file will not
load) and flags every deprecated construct at once, each with a one-click
migration to the current spelling. The rename map is curated and verified against
the installed Cantera; the fixer half lives in
``rmip.analysis.mechanism_fixer`` (handler ``migrate_keyword``).

The deprecated spellings below were verified to be *rejected* by Cantera 3.0.1
while their `current` replacements load, on the shipped ``h2o2.yaml``:

    thermo:  ideal_gas → ideal-gas          stoichiometric → fixed-stoichiometry
    type:    PLOG / plog / pdep_arrhenius → pressure-dependent-Arrhenius
             three_body / ThreeBody / threebody → three-body
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rmip.analysis.mechanism_check import Finding, FixOption

log = logging.getLogger("rmip.analysis.yaml_validator")


# ===========================================================================
# Rename catalogue
# ===========================================================================
@dataclass(frozen=True)
class RenameRule:
    context: str          # "phase-thermo" | "reaction-type"
    deprecated: str       # the stale token (case-sensitive match)
    current: str          # the current replacement
    note: str = ""        # short human explanation


# Curated deprecated→current map. Reaction-type and the two phase-thermo entries
# are verified against Cantera 3.0.1 (see module docstring / probe). The remaining
# thermo/kinetics entries are the well-established CTI→YAML renames from Cantera's
# migration guide; they are only ever *flagged*, never guessed — a token is called
# deprecated solely if it appears here.
KNOWN_RENAMES: Tuple[RenameRule, ...] = (
    # ---- reaction types (verified) ----
    RenameRule("reaction-type", "PLOG", "pressure-dependent-Arrhenius",
               "CTI/CHEMKIN PLOG spelling"),
    RenameRule("reaction-type", "plog", "pressure-dependent-Arrhenius",
               "lower-case PLOG spelling"),
    RenameRule("reaction-type", "pdep_arrhenius", "pressure-dependent-Arrhenius",
               "old CTI class name"),
    RenameRule("reaction-type", "pdep-Arrhenius", "pressure-dependent-Arrhenius",
               "near-miss spelling"),
    RenameRule("reaction-type", "three_body", "three-body",
               "underscore spelling"),
    RenameRule("reaction-type", "threebody", "three-body", "collapsed spelling"),
    RenameRule("reaction-type", "ThreeBody", "three-body", "CTI class name"),
    RenameRule("reaction-type", "chemically_activated", "chemically-activated",
               "underscore spelling"),
    RenameRule("reaction-type", "chebyshev", "Chebyshev", "case-mismatch"),
    # ---- phase thermo models ----
    RenameRule("phase-thermo", "ideal_gas", "ideal-gas", "underscore spelling"),
    RenameRule("phase-thermo", "stoichiometric", "fixed-stoichiometry",
               "CTI stoichiometric-solid model (the P1 case)"),
    RenameRule("phase-thermo", "Stoichiometric", "fixed-stoichiometry",
               "CTI class name"),
    RenameRule("phase-thermo", "const_density", "constant-density",
               "underscore spelling"),
    RenameRule("phase-thermo", "ConstDensity", "constant-density",
               "CTI class name"),
    RenameRule("phase-thermo", "IdealSolidSolution", "ideal-condensed",
               "CTI class name"),
    RenameRule("phase-thermo", "Surface", "ideal-surface", "CTI class name"),
    RenameRule("phase-thermo", "Edge", "edge", "CTI class name"),
)

_RENAME_INDEX: Dict[Tuple[str, str], RenameRule] = {
    (r.context, r.deprecated): r for r in KNOWN_RENAMES
}

# Best-effort set of reaction `type:` values valid in the current major Cantera.
# Used only for a *soft* "unrecognised type" INFO — never a hard error — because
# this set drifts across versions. Absent/`elementary` types are implicit.
CURRENT_REACTION_TYPES = frozenset({
    "elementary", "three-body", "falloff", "chemically-activated",
    "pressure-dependent-Arrhenius", "Chebyshev", "Blowers-Masel",
    "two-temperature-plasma", "electron-collision-plasma", "linear-Burke",
    "interface-Arrhenius", "sticking-Arrhenius", "Arrhenius",
})


# ===========================================================================
# Validator
# ===========================================================================
class YamlValidator:
    """Flag deprecated/renamed constructs in a Cantera YAML mechanism.

    Parameters
    ----------
    data : dict | None
        A parsed Cantera YAML document. If ``None``, provide ``path``.
    path : str | None
        Path to a YAML file to parse (used when ``data`` is not given).
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None,
                 path: Optional[str] = None) -> None:
        if data is None and path is not None:
            data = _load_yaml(path)
        self.data: Dict[str, Any] = data or {}
        self.path = path

    # ------------------------------------------------------------------
    def validate(self) -> List[Finding]:
        """Return a Finding per deprecated/unrecognised construct."""
        findings: List[Finding] = []
        findings.extend(self._check_phases())
        findings.extend(self._check_reactions())
        return findings

    def _check_phases(self) -> List[Finding]:
        out: List[Finding] = []
        phases = self.data.get("phases")
        if not isinstance(phases, list):
            return out
        for i, ph in enumerate(phases):
            if not isinstance(ph, dict):
                continue
            thermo = ph.get("thermo")
            if not isinstance(thermo, str):
                continue
            rule = _RENAME_INDEX.get(("phase-thermo", thermo))
            if rule is not None:
                out.append(self._rename_finding(
                    rule, location=f"phase[{i}] '{ph.get('name', '?')}'.thermo",
                    params={"context": "phase-thermo", "phase_index": i,
                            "current": rule.current, "deprecated": rule.deprecated}))
        return out

    def _check_reactions(self) -> List[Finding]:
        out: List[Finding] = []
        rxns = self.data.get("reactions")
        if not isinstance(rxns, list):
            return out
        for i, rx in enumerate(rxns):
            if not isinstance(rx, dict):
                continue
            rtype = rx.get("type")
            if not isinstance(rtype, str):
                continue
            eq = rx.get("equation", f"reaction[{i}]")
            rule = _RENAME_INDEX.get(("reaction-type", rtype))
            if rule is not None:
                out.append(self._rename_finding(
                    rule, location=eq,
                    params={"context": "reaction-type", "reaction_index": i,
                            "current": rule.current, "deprecated": rule.deprecated}))
            elif rtype not in CURRENT_REACTION_TYPES:
                out.append(Finding(
                    check="schema_deprecation", severity="info",
                    category="unrecognized-type",
                    message=(f"Reaction type '{rtype}' is not a recognised current "
                             "Cantera type and has no known migration — verify it "
                             "against your Cantera version."),
                    location=eq, details={"reaction_index": i, "type": rtype}))
        return out

    @staticmethod
    def _rename_finding(rule: RenameRule, location: str,
                        params: Dict[str, Any]) -> Finding:
        return Finding(
            check="schema_deprecation", severity="error",
            category="deprecated-keyword",
            message=(f"Deprecated keyword '{rule.deprecated}' — use "
                     f"'{rule.current}' ({rule.note}). Cantera will refuse to load "
                     "this file until it is migrated."),
            location=location,
            details={"deprecated": rule.deprecated, "current": rule.current,
                     "context": rule.context},
            fix_options=[FixOption(
                id="migrate_keyword", kind="auto", recommended=True,
                label=f"Migrate '{rule.deprecated}' → '{rule.current}'",
                description=f"Replace the deprecated {rule.context} keyword with "
                            "its current spelling.",
                params=params)],
        )

    # ------------------------------------------------------------------
    def migrate(self) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Return (migrated_dict, changes) applying every known rename.

        A pure-dict convenience for callers that want the whole migration at
        once; the per-finding path (via ``mechanism_fixer``) is used by the GUI.
        """
        import copy
        data = copy.deepcopy(self.data)
        changes: List[Dict[str, str]] = []
        for ph in data.get("phases", []) or []:
            if isinstance(ph, dict) and isinstance(ph.get("thermo"), str):
                rule = _RENAME_INDEX.get(("phase-thermo", ph["thermo"]))
                if rule:
                    changes.append({"where": f"phase '{ph.get('name','?')}'",
                                    "from": ph["thermo"], "to": rule.current})
                    ph["thermo"] = rule.current
        for rx in data.get("reactions", []) or []:
            if isinstance(rx, dict) and isinstance(rx.get("type"), str):
                rule = _RENAME_INDEX.get(("reaction-type", rx["type"]))
                if rule:
                    changes.append({"where": rx.get("equation", "reaction"),
                                    "from": rx["type"], "to": rule.current})
                    rx["type"] = rule.current
        return data, changes


# ===========================================================================
def _load_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        # rmip.analysis.yaml_safe, not plain yaml.safe_load: stock PyYAML
        # resolves a bare species name "NO" as a YAML-1.1 bool (see
        # yaml_safe.py docstring), silently corrupting species-name lists.
        from rmip.analysis.yaml_safe import safe_load_path
        d = safe_load_path(path)
        return d if isinstance(d, dict) else None
    except Exception:                                      # noqa: BLE001
        return None
