"""
rmip/analysis/mechanism_fixer.py
================================
Apply the repairs proposed by :mod:`rmip.analysis.mechanism_check` to a
mechanism and emit a corrected Cantera YAML.

The checker attaches :class:`~rmip.analysis.mechanism_check.FixOption` objects to
each fixable finding. This module consumes ``(finding, option_id)`` pairs and
edits the **parsed YAML dict** in place, then writes a valid YAML that Cantera
can re-load. Everything works on the YAML text (not a live Solution) so it can
repair mechanisms that do not even load yet (unmarked duplicates, negative PLOG).

Two classes of repair, mirroring the ``FixOption.kind`` field:

* **auto**   — the single obvious answer (thermo Tmid refit, mark-duplicate,
               drop a negative PLOG piece). Safe to apply unattended via
               ``--fix auto``.
* **choice** — an ambiguous defect with several pre-built answers the user must
               pick (collision-limit: scale-A / delete / ignore; Ea unit trap:
               reinterpret / ignore).

Reaction indices in a fix's ``params`` refer to the position in the mechanism's
top-level ``reactions`` list — which is also the Cantera reaction order — so the
index reported by the checker maps unambiguously to a YAML entry.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.analysis.mechanism_fixer")

J_PER_CAL = 4.184
_COLLISION_LIMIT_298_CGS = 1.0e14      # cm^3/mol/s at 298 K (Arrhenius A units)


@dataclass
class FixResult:
    option_id: str
    location: str
    status: str                         # "applied" | "skipped" | "failed"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"option_id": self.option_id, "location": self.location,
                "status": self.status, "message": self.message}


class MechanismFixer:
    """Load a mechanism YAML, apply repairs, write the corrected YAML.

    Parameters
    ----------
    yaml_path : str
        Path to a Cantera YAML (for CHEMKIN inputs, convert first with the
        checker, whose ``yaml_path`` attribute points at the converted file).
    """

    def __init__(self, yaml_path: str) -> None:
        # Not plain yaml.safe_load: stock PyYAML's YAML-1.1 bool resolver
        # turns a bare species name "NO" (nitric oxide -- present in every
        # NOx mechanism) into Python False, which write() then re-emits as a
        # literal YAML `false` mixed into a species-name list. Cantera's own
        # (stricter) reader rejects that outright with a confusing
        # "Could not parse species declaration of type 'vector<AnyValue>'"
        # error that gives no hint the real cause is a mangled species name.
        # Found 2026-07-22 auto-fixing Glarborg 2018's negative-PLOG findings.
        from rmip.analysis.yaml_safe import safe_load_path
        self.src_path = yaml_path
        self.data: Dict[str, Any] = safe_load_path(yaml_path)
        if not isinstance(self.data, dict):
            raise ValueError(f"{yaml_path} is not a Cantera YAML document")
        self.results: List[FixResult] = []
        self._pending_deletes: set[int] = set()

    # ------------------------------------------------------------------
    @property
    def reactions(self) -> List[Dict[str, Any]]:
        return self.data.setdefault("reactions", [])

    @property
    def species(self) -> List[Dict[str, Any]]:
        return self.data.setdefault("species", [])

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def apply(self, fixes: List[Dict[str, Any]]) -> List[FixResult]:
        """Apply a list of fixes.

        Each fix is ``{"option_id": str, "params": dict, "location": str}``.
        In-place edits run first; reaction deletions are deferred and applied
        last (in descending index order) so earlier indices stay valid.
        """
        handlers = {
            "migrate_keyword": self._fix_migrate_keyword,
            "refit_thermo": self._fix_refit_thermo,
            "shift_enthalpy_to_ref": self._fix_shift_enthalpy,
            "mark_duplicate": self._fix_mark_duplicate,
            "plog_drop_negative": self._fix_plog_drop_negative,
            "scale_A_to_limit": self._fix_scale_A,
            "ea_reinterpret_j": lambda p, l: self._fix_ea_reinterpret(p, l, J_PER_CAL),
            "ea_reinterpret_j_kmol": lambda p, l: self._fix_ea_reinterpret(p, l, 1000.0 * J_PER_CAL),
            "remove_species": self._fix_remove_species,
            "delete_reaction": self._fix_delete_reaction,
            "rename_species_ref": self._fix_rename_species_ref,
            "ignore": self._fix_ignore,
        }
        for fx in fixes:
            oid = fx.get("option_id", "")
            params = fx.get("params", {}) or {}
            loc = fx.get("location", "")
            handler = handlers.get(oid)
            if handler is None:
                self.results.append(FixResult(oid, loc, "skipped",
                                              f"no handler for '{oid}'"))
                continue
            try:
                handler(params, loc)
            except Exception as exc:                       # noqa: BLE001
                log.exception("fix '%s' failed", oid)
                self.results.append(FixResult(oid, loc, "failed", str(exc)))

        self._flush_deletes()
        return self.results

    def write(self, out_path: str) -> str:
        import yaml
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        # tag provenance without disturbing Cantera's schema
        gen = self.data.get("generator", "")
        self.data["generator"] = (gen + " + rmip mechanism-fixer").strip(" +")

        # Preserve ck2yaml's own convention of literal block scalars ('key: |-')
        # for multi-line strings (bibliography 'note:'/'description:' fields).
        # Plain yaml.safe_dump re-serializes any multi-line str as a quoted
        # flow scalar instead, which (a) is far less readable than the
        # original and (b) defeats SyntaxLinter's YAML-block-scalar
        # recognition -- its continuation lines have no '|'/'>' marker to key
        # off, so bibliography text like "J Phys Chem A 2013." gets scanned
        # as mechanism data and misfires the "truncated numeric field" check
        # hundreds of times over (found 2026-07-22 sweeping the Glarborg 2018
        # health-check report through mechanism_fixer's round-trip).
        class _Dumper(yaml.SafeDumper):
            pass

        def _str_presenter(dumper, data):
            style = "|" if "\n" in data else None
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

        _Dumper.add_representer(str, _str_presenter)

        with open(out_path, "w") as fh:
            # width=100000: a long *single-line* note/description (no
            # embedded '\n', so _str_presenter above leaves it a plain
            # scalar) still gets implicitly line-folded by PyYAML's default
            # 80-column width -- same false-positive syntax_lint hazard as
            # the literal-block case above, just for a different reason.
            # An effectively-unbounded width stops PyYAML from ever folding
            # a plain scalar across lines.
            yaml.dump(self.data, fh, Dumper=_Dumper, sort_keys=False,
                      default_flow_style=False, width=100000)
        return out_path

    # ------------------------------------------------------------------
    # Individual fixes
    # ------------------------------------------------------------------
    def _fix_ignore(self, params, loc) -> None:
        self.results.append(FixResult("ignore", loc, "applied",
                                      "acknowledged; left unchanged"))

    def _fix_migrate_keyword(self, params, loc) -> None:
        """W1: replace a deprecated CTI→YAML keyword with its current spelling."""
        ctx = params.get("context")
        current = params.get("current")
        deprecated = params.get("deprecated", "?")
        if ctx == "phase-thermo":
            phases = self.data.get("phases", [])
            i = params.get("phase_index")
            if not (isinstance(i, int) and 0 <= i < len(phases)):
                self.results.append(FixResult("migrate_keyword", loc, "skipped",
                                              "phase index out of range"))
                return
            phases[i]["thermo"] = current
        elif ctx == "reaction-type":
            rxn = self._reaction_at(params.get("reaction_index"))
            if rxn is None:
                self.results.append(FixResult("migrate_keyword", loc, "skipped",
                                              "reaction index out of range"))
                return
            rxn["type"] = current
        else:
            self.results.append(FixResult("migrate_keyword", loc, "skipped",
                                          f"unknown migration context '{ctx}'"))
            return
        self.results.append(FixResult(
            "migrate_keyword", loc, "applied",
            f"'{deprecated}' → '{current}'"))

    def _fix_refit_thermo(self, params, loc) -> None:
        from rmip.analysis.thermo_fitter import refit_from_coeffs
        name = params.get("species") or loc
        sp = self._find_species(name)
        if sp is None:
            self.results.append(FixResult("refit_thermo", name, "skipped",
                                          "species not found"))
            return
        thermo = sp.get("thermo", {})
        if str(thermo.get("model", "")).upper() not in ("NASA7", "NASA"):
            self.results.append(FixResult("refit_thermo", name, "skipped",
                                          "not a NASA-7 thermo record"))
            return
        ranges = thermo.get("temperature-ranges", [])
        data = thermo.get("data", [])
        if len(ranges) < 3 or len(data) < 2:
            self.results.append(FixResult("refit_thermo", name, "skipped",
                                          "single-range or malformed NASA record"))
            return
        tmin, tmid, tmax = float(ranges[0]), float(ranges[1]), float(ranges[-1])
        low = [float(x) for x in data[0]]
        high = [float(x) for x in data[1]]
        fit = refit_from_coeffs(low, high, tmin, tmid, tmax)
        new_low, new_high = fit.as_coeffs_pair()
        thermo["data"] = [list(new_low), list(new_high)]
        self.results.append(FixResult(
            "refit_thermo", name, "applied",
            f"refit NASA-7 at Tmid={tmid:.0f} K (continuity restored)"))

    def _fix_shift_enthalpy(self, params, loc) -> None:
        """W4: shift NASA-7 a6 (both T-ranges) by −ΔH/R so ΔHf(298) lands on the
        reference value while Cp(T) and S(T) are untouched (a6 only enters H)."""
        R = 8.314_462_618
        name = params.get("species") or loc
        delta_H = params.get("delta_H_J_mol")           # mech − ref, J/mol
        sp = self._find_species(name)
        if sp is None or delta_H is None:
            self.results.append(FixResult("shift_enthalpy_to_ref", name,
                                          "skipped", "species/delta missing"))
            return
        thermo = sp.get("thermo", {})
        data = thermo.get("data", [])
        if str(thermo.get("model", "")).upper() != "NASA7" or len(data) < 2:
            self.results.append(FixResult("shift_enthalpy_to_ref", name,
                                          "skipped", "not a 2-range NASA-7 record"))
            return
        shift = -float(delta_H) / R                     # Δa6 [K]
        for rng in data[:2]:
            rng[5] = float(rng[5]) + shift
        self.results.append(FixResult(
            "shift_enthalpy_to_ref", name, "applied",
            f"a6 shifted by {shift:+.4g} K (ΔHf moved {-delta_H/1000.0:+.2f} "
            "kJ/mol onto the reference)"))

    def _fix_mark_duplicate(self, params, loc) -> None:
        idxs = params.get("reaction_indices", [])
        n = 0
        for i in idxs:
            if 0 <= i < len(self.reactions):
                self.reactions[i]["duplicate"] = True
                n += 1
        self.results.append(FixResult("mark_duplicate", loc, "applied",
                                      f"flagged {n} reaction(s) duplicate"))

    def _fix_plog_drop_negative(self, params, loc) -> None:
        i = params.get("reaction_index")
        rxn = self._reaction_at(i)
        if rxn is None:
            self.results.append(FixResult("plog_drop_negative", loc, "skipped",
                                          "reaction index out of range"))
            return
        rates = rxn.get("rate-constants", [])
        kept = [rc for rc in rates if _num(rc.get("A")) is None or _num(rc.get("A")) >= 0]
        dropped = len(rates) - len(kept)
        if dropped == 0:
            self.results.append(FixResult("plog_drop_negative", loc, "skipped",
                                          "no negative pieces found"))
            return
        if len(kept) < 1:
            self.results.append(FixResult("plog_drop_negative", loc, "failed",
                                          "dropping would leave no PLOG pieces"))
            return
        rxn["rate-constants"] = kept
        self.results.append(FixResult("plog_drop_negative", loc, "applied",
                                      f"dropped {dropped} negative pressure piece(s)"))

    def _fix_scale_A(self, params, loc) -> None:
        i = params.get("reaction_index")
        direction = params.get("direction", "forward")
        ratio = float(params.get("ratio", 0) or 0)
        rxn = self._reaction_at(i)
        if rxn is None:
            self.results.append(FixResult("scale_A_to_limit", loc, "skipped",
                                          "reaction index out of range"))
            return
        # Only a pressure-independent Arrhenius in the FORWARD direction can be
        # scaled unambiguously by editing A. Everything else is manual.
        if rxn.get("type") not in (None, "Arrhenius", "arrhenius"):
            self.results.append(FixResult(
                "scale_A_to_limit", loc, "skipped",
                f"reaction type '{rxn.get('type')}' — scale A manually"))
            return
        if direction != "forward":
            self.results.append(FixResult(
                "scale_A_to_limit", loc, "skipped",
                "reverse-direction violation is driven by thermo/Kc — fix the "
                "thermo, not A (scaling forward A would corrupt Keq)"))
            return
        rc = rxn.get("rate-constant")
        if not isinstance(rc, dict) or "A" not in rc:
            self.results.append(FixResult("scale_A_to_limit", loc, "skipped",
                                          "no Arrhenius A to scale"))
            return
        if ratio <= 1.0:
            self.results.append(FixResult("scale_A_to_limit", loc, "skipped",
                                          "reported exceedance ≤ 1"))
            return
        old = _num(rc["A"])
        rc["A"] = old / ratio
        self.results.append(FixResult(
            "scale_A_to_limit", loc, "applied",
            f"scaled A {old:.3g} → {rc['A']:.3g} (÷{ratio:.3g}) to the limit"))

    def _fix_ea_reinterpret(self, params, loc, divisor: float) -> None:
        oid = ("ea_reinterpret_j_kmol" if divisor > 100.0
               else "ea_reinterpret_j")
        i = params.get("reaction_index")
        rxn = self._reaction_at(i)
        if rxn is None:
            self.results.append(FixResult(oid, loc, "skipped",
                                          "reaction index out of range"))
            return
        rc = rxn.get("rate-constant")
        if not isinstance(rc, dict) or "Ea" not in rc:
            self.results.append(FixResult(oid, loc, "skipped",
                                          "no Ea to reinterpret"))
            return
        old = _num(rc["Ea"])
        rc["Ea"] = old / divisor
        unit = "J/kmol→cal/mol" if divisor > 100.0 else "J/mol→cal/mol"
        self.results.append(FixResult(
            oid, loc, "applied",
            f"Ea {old:.4g} → {rc['Ea']:.4g} (÷{divisor:.4g}; {unit})"))

    def _fix_remove_species(self, params, loc) -> None:
        name = params.get("species") or loc
        before = len(self.species)
        self.data["species"] = [s for s in self.species if s.get("name") != name]
        # also drop from every phase's species list
        for ph in self.data.get("phases", []):
            spec = ph.get("species")
            if isinstance(spec, list):
                ph["species"] = [s for s in spec if s != name]
        removed = before - len(self.data["species"])
        self.results.append(FixResult(
            "remove_species", name, "applied" if removed else "skipped",
            f"removed species '{name}'" if removed else "species not found"))

    def _fix_delete_reaction(self, params, loc) -> None:
        idxs = params.get("reaction_indices")
        if idxs is None and params.get("reaction_index") is not None:
            idxs = [params["reaction_index"]]
        for i in idxs or []:
            if 0 <= i < len(self.reactions):
                self._pending_deletes.add(i)
        self.results.append(FixResult("delete_reaction", loc, "applied",
                                      f"queued {len(idxs or [])} reaction(s) for deletion"))

    def _fix_rename_species_ref(self, params, loc) -> None:
        """W8a/P35: rewrite every reference to a case-variant species name
        (equations, third-body efficiencies, phase species lists) to the
        spelling declared in the species section."""
        frm = params.get("from")
        to = params.get("to")
        if not frm or not to:
            self.results.append(FixResult("rename_species_ref", loc, "skipped",
                                          "missing from/to name"))
            return
        n_edits = 0

        def _rename_eq(eq: str) -> str:
            nonlocal n_edits
            # equations are space-tokenised ('2 H2O', '(+M)' are own tokens);
            # also catch the falloff form '(+AR)'.
            out = []
            for tok in eq.split(" "):
                if tok == frm:
                    tok = to
                    n_edits += 1
                elif tok == f"(+{frm})":
                    tok = f"(+{to})"
                    n_edits += 1
                out.append(tok)
            return " ".join(out)

        for key, val in list(self.data.items()):
            if key != "reactions" and not str(key).endswith("-reactions"):
                continue
            if not isinstance(val, list):
                continue
            for rxn in val:
                if not isinstance(rxn, dict):
                    continue
                if isinstance(rxn.get("equation"), str):
                    rxn["equation"] = _rename_eq(rxn["equation"])
                eff = rxn.get("efficiencies")
                if isinstance(eff, dict) and frm in eff:
                    eff[to] = eff.pop(frm)
                    n_edits += 1

        for ph in self.data.get("phases", []):
            spec = ph.get("species")
            if isinstance(spec, list):
                for j, s in enumerate(spec):
                    if s == frm:
                        spec[j] = to
                        n_edits += 1

        self.results.append(FixResult(
            "rename_species_ref", loc,
            "applied" if n_edits else "skipped",
            f"renamed {n_edits} reference(s) '{frm}' → '{to}'"))

    def _flush_deletes(self) -> None:
        if not self._pending_deletes:
            return
        for i in sorted(self._pending_deletes, reverse=True):
            if 0 <= i < len(self.reactions):
                del self.reactions[i]
        self._pending_deletes.clear()
        # A deleted reaction may have been one of exactly two declared
        # `duplicate: true` reactions sharing an equation (e.g. a dual-PLOG
        # subtraction channel removed by delete_reaction). Cantera's
        # Kinetics::checkDuplicates rejects a lone survivor still flagged
        # duplicate ("No duplicate found for declared duplicate reaction").
        # Clear the flag wherever exactly one instance of an equation
        # remains. Found 2026-07-22 deleting Glarborg 2018's all-negative
        # 'C2H3 + CH2O <=> C2H4 + HCO' PLOG duplicate.
        by_eq: Dict[str, List[Dict[str, Any]]] = {}
        for rxn in self.reactions:
            eq = rxn.get("equation")
            if eq is not None:
                by_eq.setdefault(eq, []).append(rxn)
        for eq, rxns in by_eq.items():
            if len(rxns) == 1 and rxns[0].get("duplicate") is True:
                rxns[0].pop("duplicate", None)

    # ------------------------------------------------------------------
    def _find_species(self, name: str) -> Optional[Dict[str, Any]]:
        for s in self.species:
            if s.get("name") == name:
                return s
        return None

    def _reaction_at(self, i) -> Optional[Dict[str, Any]]:
        if i is None or not (0 <= i < len(self.reactions)):
            return None
        return self.reactions[i]


def _num(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.split()[0])
        except (ValueError, IndexError):
            return None
    return None


# ===========================================================================
# High-level helper: collect the auto-fixes from a report
# ===========================================================================
def collect_auto_fixes(report) -> List[Dict[str, Any]]:
    """Return the list of auto-applicable fixes from a CheckReport."""
    fixes: List[Dict[str, Any]] = []
    for f in report.findings:
        auto = f.auto_fix
        if auto is not None:
            fixes.append({"option_id": auto.id, "params": auto.params,
                          "location": f.location})
    return fixes
