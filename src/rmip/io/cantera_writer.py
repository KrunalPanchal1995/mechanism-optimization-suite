"""
rmip/io/cantera_writer.py
==========================
Phase-4: Direct DB → Cantera 3.x YAML writer.

Why this exists
---------------
The existing ChemkinWriter.convert_to_cantera_yaml() uses the ck2yaml
round-trip subprocess, which cannot embed RMIP uncertainty metadata.
This writer goes directly from the DB to YAML and optionally adds per-reaction
``rmip-uncertainty`` keys that Cantera ignores on load but preserve the data
for KP's downstream tooling.

Unit convention in the emitted YAML
-------------------------------------
Cantera YAML units header:
    {activation-energy: J/mol, length: m, time: s, quantity: mol}

Pre-exponential A is written in SI units (m^n / (mol^(n-1) · s)).
Ea is in J/mol.

Validation
----------
When Cantera is importable, the writer calls cantera.Solution(output_yaml)
after writing to confirm the file is valid.  On failure it logs a warning
but does not raise.

Usage
-----
>>> from rmip.io.cantera_writer import CanteraYAMLWriter
>>> writer = CanteraYAMLWriter("project.db")
>>> writer.write(4, "unified.yaml")
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("rmip.io.cantera_writer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YAML_UNITS_HEADER = """\
units:
  activation-energy: J/mol
  length: m
  time: s
  quantity: mol
"""

# Elements in preferred CHEMKIN order. M6 (review.md): must use Cantera's
# canonical element-symbol case ("Ar", not "AR") -- Cantera element lookup
# is case-sensitive and species `composition:` blocks are built from the
# same [A-Z][a-z]? regex as _build_phases_section below with no case
# transform, so a mismatched-case `elements:` declaration made the exported
# YAML unloadable for any mechanism with a 2-letter element (EXEC: GRI-
# wetCO round trip failed "element not found: AR").
_ELEMENT_ORDER = ["C", "H", "O", "N", "S", "Ar", "He", "Cl", "F"]


class CanteraYAMLWriter:
    """
    Write an RMIP DB mechanism to Cantera YAML format.

    Parameters
    ----------
    db_path : str | Path
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def write(
        self,
        mechanism_id: int,
        output_yaml: str,
        uncertainty_bands: "Optional[List[Any]]" = None,
        embed_uncertainty: bool = True,
    ) -> str:
        """
        Write mechanism to Cantera YAML.

        Parameters
        ----------
        mechanism_id : int
        output_yaml : str
            Output file path.
        uncertainty_bands : list[UncertaintyBand] | None
            Uncertainty bands from UnifiedTreeBuilder.build(); if provided and
            embed_uncertainty=True, per-reaction rmip-uncertainty keys are added.
        embed_uncertainty : bool
            Whether to embed rmip-uncertainty metadata (default True).

        Returns
        -------
        str
            Path to written file.
        """
        Path(output_yaml).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            mech_row     = conn.execute(
                "SELECT * FROM mechanisms WHERE id=?", (mechanism_id,)
            ).fetchone()
            if mech_row is None:
                raise ValueError(f"Mechanism id={mechanism_id} not found")

            species_rows  = conn.execute(
                "SELECT * FROM species WHERE mechanism_id=? ORDER BY id",
                (mechanism_id,),
            ).fetchall()
            reaction_rows = conn.execute(
                "SELECT * FROM reactions WHERE mechanism_id=? ORDER BY id",
                (mechanism_id,),
            ).fetchall()
            stoich_map    = self._load_stoich(conn, mechanism_id)
            eff_map       = self._load_efficiencies(conn, mechanism_id)

        # Build band lookup: canonical_key → UncertaintyBand
        band_map: Dict[str, Any] = {}
        if uncertainty_bands and embed_uncertainty:
            for b in uncertainty_bands:
                band_map[b.canonical_key] = b

        # Build YAML text
        lines: List[str] = []
        lines.append(f"description: >")
        mech_name = dict(mech_row).get("name", f"mechanism_{mechanism_id}")
        lines.append(f"  RMIP unified mechanism — {mech_name}")
        lines.append("")
        lines.extend(_YAML_UNITS_HEADER.splitlines())
        lines.append("")

        # Phases
        lines.extend(self._build_phases_section(mech_name, species_rows))
        lines.append("")

        # Species
        lines.extend(self._build_species_section(species_rows))
        lines.append("")

        # Reactions
        lines.extend(self._build_reactions_section(
            reaction_rows, stoich_map, eff_map, band_map
        ))
        lines.append("")

        yaml_text = "\n".join(lines) + "\n"
        Path(output_yaml).write_text(yaml_text, encoding="utf-8")
        log.info("Wrote Cantera YAML → %s", output_yaml)

        # Validate via round-trip if Cantera is available
        self._validate_yaml(output_yaml)

        return output_yaml

    # -------------------------------------------------------------------------
    # YAML section builders
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_phases_section(
        name: str,
        species_rows: List[sqlite3.Row],
    ) -> List[str]:
        """Build the 'phases' block."""
        # Infer elements from formulae
        elements: set = set()
        for sp in species_rows:
            formula = dict(sp).get("formula") or ""
            for el in re.findall(r"[A-Z][a-z]?", formula):
                # M6: keep the regex-captured case as-is (matches
                # _parse_composition's identical regex with no case
                # transform below) -- .upper() turned "Ar" into "AR",
                # mismatching the `composition:` block's "Ar" key.
                elements.add(el)
        elements.update({"C", "H", "O", "N"})

        ordered = [e for e in _ELEMENT_ORDER if e in elements]
        ordered += sorted(elements - set(_ELEMENT_ORDER))
        elem_str = "[" + ", ".join(ordered) + "]"

        sp_names = [dict(sp).get("name", "") for sp in species_rows]
        sp_list_yaml = _yaml_str_list(sp_names, indent=4)

        lines = [
            "phases:",
            f"- name: {_yaml_safe(name)}",
            "  thermo: ideal-gas",
            "  elements: " + elem_str,
            "  species: " + sp_list_yaml,
            "  kinetics: gas",
            "  state: {T: 300.0, P: 1.01325e+05}",
        ]
        return lines

    @staticmethod
    def _build_species_section(
        species_rows: List[sqlite3.Row],
    ) -> List[str]:
        """Build the 'species' block."""
        lines = ["species:"]
        for sp in species_rows:
            sp = dict(sp)
            name    = sp.get("name", "UNK")
            formula = sp.get("formula") or ""
            comp    = _parse_composition(formula)

            lines.append(f"- name: {_yaml_safe(name)}")
            if comp:
                comp_str = "{" + ", ".join(f"{el}: {n}" for el, n in comp.items()) + "}"
                lines.append(f"  composition: {comp_str}")
            else:
                lines.append("  composition: {}")

            # NASA-7 thermo
            try:
                low  = json.loads(sp.get("coeffs_low_json")  or "null")
                high = json.loads(sp.get("coeffs_high_json") or "null")
                tmin = sp.get("tmin_K") or 200.0
                tmid = sp.get("tmid_K") or 1000.0
                tmax = sp.get("tmax_K") or 5000.0
                if low and high and len(low) == 7 and len(high) == 7:
                    lines.append("  thermo:")
                    lines.append("    model: NASA7")
                    lines.append(f"    temperature-ranges: [{tmin:.1f}, {tmid:.1f}, {tmax:.1f}]")
                    lines.append("    data:")
                    # M8 (review.md): Cantera's YAML NASA7 format pairs the
                    # data blocks positionally with temperature-ranges --
                    # the FIRST block covers [Tmin, Tmid] (low-T) and the
                    # SECOND covers [Tmid, Tmax] (high-T), confirmed against
                    # this repo's own FFCM1.yaml (e.g. N2: low-T block's a1
                    # ~3.53 matches a rigid-rotor diatomic at 300 K; high-T
                    # block's a1 ~2.95 only makes sense above ~1000 K).
                    # These were swapped -- Cantera evaluated the high-T
                    # polynomial below Tmid and vice versa, which is exactly
                    # what surfaced as spurious "discontinuity in cp/R" (and
                    # h/RT, s/R) warnings at Tmid on every multi-range
                    # species this writer ever exported.
                    lines.append("    - " + _yaml_coeff_list(low))
                    lines.append("    - " + _yaml_coeff_list(high))
            except (TypeError, ValueError, json.JSONDecodeError):
                lines.append("  thermo: {}")

            # Lennard-Jones transport (Phase-3 follow-on) — emitted only when the
            # species carries it in the DB (backward-compatible: absent columns /
            # NULLs simply omit the block).
            geom = sp.get("transport_geometry")
            well = sp.get("well_depth_K")
            diam = sp.get("diameter_A")
            if geom is not None and well is not None and diam is not None:
                lines.append("  transport:")
                lines.append("    model: gas")
                lines.append(f"    geometry: {geom}")
                lines.append(f"    well-depth: {float(well):.4f}")
                lines.append(f"    diameter: {float(diam):.4f}")
                dip = sp.get("dipole_D") or 0.0
                pol = sp.get("polarizability_A3") or 0.0
                rot = sp.get("rot_relax") or 0.0
                if float(dip):
                    lines.append(f"    dipole: {float(dip):.4f}")
                if float(pol):
                    lines.append(f"    polarizability: {float(pol):.4f}")
                if float(rot):
                    lines.append(f"    rotational-relaxation: {float(rot):.4f}")

        return lines

    def _build_reactions_section(
        self,
        reaction_rows: List[sqlite3.Row],
        stoich_map: Dict[int, List[tuple]],
        eff_map: Dict[int, Dict[str, float]],
        band_map: Dict[str, Any],
    ) -> List[str]:
        """Build the 'reactions' block."""
        lines = ["reactions:"]
        for rxn in reaction_rows:
            rxn = dict(rxn)
            rxn_lines = self._format_reaction_yaml(rxn, stoich_map, eff_map, band_map)
            lines.extend(rxn_lines)
        return lines

    def _format_reaction_yaml(
        self,
        rxn: dict,
        stoich_map: Dict[int, List[tuple]],
        eff_map: Dict[int, Dict[str, float]],
        band_map: Dict[str, Any],
    ) -> List[str]:
        """Format one reaction as Cantera YAML entries."""
        rxn_id = rxn["id"]
        ktype  = (rxn.get("kinetics_type") or "arrhenius").lower()
        formula = rxn.get("formula_string") or self._build_formula(rxn, stoich_map)

        A   = rxn.get("A_si") or 0.0
        n   = rxn.get("n") or 0.0
        Ea  = rxn.get("Ea_si") or 0.0

        lines: List[str] = []

        if ktype in ("arrhenius", "multi_arrhenius", "unknown"):
            lines.append(f"- equation: {formula}")
            lines.append(f"  rate-constant: {{A: {A:.6E}, b: {n:.4f}, Ea: {Ea:.4f}}}")

        elif ktype == "third_body":
            lines.append(f"- equation: {formula}")
            lines.append(f"  type: three-body")
            lines.append(f"  rate-constant: {{A: {A:.6E}, b: {n:.4f}, Ea: {Ea:.4f}}}")
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append("  efficiencies: " + _yaml_dict_inline(effs))

        elif ktype in ("troe", "falloff", "lindemann"):
            kdata = self._parse_kdata(rxn)
            lines.append(f"- equation: {formula}")
            lines.append(f"  type: falloff")
            lines.append(f"  high-P-rate-constant: {{A: {A:.6E}, b: {n:.4f}, Ea: {Ea:.4f}}}")
            low = kdata.get("low", {})
            if low:
                Al = low.get("A_si", 0.0)
                nl = low.get("n", 0.0)
                El = low.get("Ea_si", 0.0)
                lines.append(f"  low-P-rate-constant: {{A: {Al:.6E}, b: {nl:.4f}, Ea: {El:.4f}}}")
            if ktype == "troe" and "alpha" in kdata:
                alpha = kdata["alpha"]
                T1    = kdata.get("T1", 1e30)
                T3    = kdata.get("T3", 1e-30)
                T2    = kdata.get("T2")
                if T2 is not None:
                    lines.append(f"  Troe: {{A: {alpha}, T3: {T3:.3E}, T1: {T1:.3E}, T2: {T2:.3E}}}")
                else:
                    lines.append(f"  Troe: {{A: {alpha}, T3: {T3:.3E}, T1: {T1:.3E}}}")
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append("  efficiencies: " + _yaml_dict_inline(effs))

        elif ktype == "plog":
            kdata = self._parse_kdata(rxn)
            lines.append(f"- equation: {formula}")
            lines.append(f"  type: pressure-dependent-Arrhenius")
            plog_lines: List[str] = []

            if "pressures" in kdata and "arrhenius" in kdata:
                # SI format from parsers: pressures in Pa
                for P_Pa, arr in zip(kdata["pressures"], kdata["arrhenius"]):
                    Ap  = arr.get("A_si") or arr.get("A", 0.0)
                    np_ = arr.get("n", 0.0)
                    Ep  = arr.get("Ea_si") or arr.get("Ea", 0.0)
                    plog_lines.append(
                        f"    - {{P: {P_Pa:.4E}, A: {Ap:.6E}, b: {np_:.4f}, Ea: {Ep:.4f}}}"
                    )
            elif "plog_rates" in kdata:
                # Legacy format: A in CGS, Ea in kcal/mol → convert to SI
                for rate in kdata["plog_rates"]:
                    P_atm = rate.get("P_atm", 1.0)
                    Ap_cgs = rate.get("A", 0.0)
                    np_   = rate.get("n", 0.0)
                    Ep_k  = rate.get("Ea", 0.0)
                    Ap_si = Ap_cgs / 1e6
                    Ep_si = Ep_k * 4184.0
                    plog_lines.append(
                        f"    - {{P: {P_atm * 101325:.4E}, "
                        f"A: {Ap_si:.6E}, b: {np_:.4f}, Ea: {Ep_si:.4f}}}"
                    )
            if plog_lines:
                lines.append("  rates:")
                lines.extend(plog_lines)
            else:
                # Fallback: write as plain Arrhenius
                lines[-1] = f"- equation: {formula}"
                lines.append(f"  rate-constant: {{A: {A:.6E}, b: {n:.4f}, Ea: {Ea:.4f}}}")
                log.warning("PLOG reaction %s: no rate entries found — wrote as Arrhenius",
                            rxn.get("canonical_key", ""))

        elif ktype == "chebyshev":
            kdata = self._parse_kdata(rxn)
            Tmin  = kdata.get("Tmin", 300.0)
            Tmax  = kdata.get("Tmax", 3000.0)
            Pmin  = kdata.get("Pmin", 101.325)    # Pa
            Pmax  = kdata.get("Pmax", 1.013e7)    # Pa
            coeffs = kdata.get("coeffs", [])
            lines.append(f"- equation: {formula}")
            lines.append(f"  type: Chebyshev")
            lines.append(f"  temperature-range: [{Tmin:.1f}, {Tmax:.1f}]")
            lines.append(f"  pressure-range: [{Pmin:.3E}, {Pmax:.3E}]")
            if coeffs:
                lines.append("  data:")
                for row in coeffs:
                    row_str = "[" + ", ".join(f"{c:.6E}" for c in row) + "]"
                    lines.append(f"  - {row_str}")

        else:
            log.warning("Unknown kinetics type '%s' for rxn id=%d — writing as Arrhenius",
                        ktype, rxn_id)
            lines.append(f"- equation: {formula}")
            lines.append(f"  rate-constant: {{A: {A:.6E}, b: {n:.4f}, Ea: {Ea:.4f}}}")

        # Duplicate flag
        if rxn.get("duplicate"):
            lines.append("  duplicate: true")

        # Embed uncertainty metadata
        ck = rxn.get("canonical_key", "")
        if ck and ck in band_map:
            band = band_map[ck]
            f_worst = band.f_worst()
            f_mean  = band.f_mean()
            sigma_1k = band.sigma_lnk_at(1000.0)
            unc_str = (
                f"{{f_worstT: {f_worst:.3f}, f_meanT: {f_mean:.3f}, "
                f"sigma_lnk_1000K: {sigma_1k:.4f}, P_ref_atm: {band_map.get('__P_ref', 1.0)}}}"
            )
            note_str = (
                f"RMIP unified; {rxn.get('fusion_class','FUSE')}; "
                f"uf_worstT={f_worst:.2f}"
            )
            # Try embedding as a YAML mapping key (Cantera tolerates extras on reactions)
            try:
                # Test if we can embed — always do it, Cantera ≥2.6 tolerates unknown keys
                lines.append(f"  note: \"{note_str}\"")
                lines.append(f"  rmip-uncertainty: {unc_str}")
            except Exception:
                # Fallback: pack into note only
                lines[-2] = f"  note: \"{note_str}; {unc_str}\""

        return lines

    # -------------------------------------------------------------------------
    # DB loaders
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_stoich(
        conn: sqlite3.Connection,
        mech_id: int,
    ) -> Dict[int, List[tuple]]:
        """Return {rxn_id: [(sp_name, coeff, role), ...]}."""
        rows = conn.execute(
            """
            SELECT s.reaction_id, sp.name, s.stoich_coeff, s.role
            FROM stoichiometry s
            JOIN species sp ON sp.id = s.species_id
            WHERE sp.mechanism_id = ?
            ORDER BY s.reaction_id, s.role, s.id
            """,
            (mech_id,),
        ).fetchall()
        result: Dict[int, List[tuple]] = {}
        for row in rows:
            result.setdefault(row["reaction_id"], []).append(
                (row["name"], row["stoich_coeff"], row["role"])
            )
        return result

    @staticmethod
    def _load_efficiencies(
        conn: sqlite3.Connection,
        mech_id: int,
    ) -> Dict[int, Dict[str, float]]:
        """Return {rxn_id: {sp_name: eff}}."""
        rows = conn.execute(
            """
            SELECT tbe.reaction_id,
                   COALESCE(sp.name, tbe.species_name) AS name,
                   tbe.efficiency
            FROM third_body_efficiencies tbe
            LEFT JOIN species sp ON sp.id = tbe.species_id
            JOIN reactions r ON r.id = tbe.reaction_id
            WHERE r.mechanism_id = ?
            """,
            (mech_id,),
        ).fetchall()
        result: Dict[int, Dict[str, float]] = {}
        for row in rows:
            result.setdefault(row["reaction_id"], {})[row["name"]] = row["efficiency"]
        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_kdata(rxn: dict) -> dict:
        raw = rxn.get("kinetics_data_json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _build_formula(rxn: dict, stoich_map: Dict[int, List[tuple]]) -> str:
        """Build 'A+B<=>C+D' from stoich if formula_string is absent."""
        rxn_id = rxn["id"]
        stoich = stoich_map.get(rxn_id, [])
        reactants = [(name, coef) for name, coef, role in stoich if role == "reactant"]
        products  = [(name, coef) for name, coef, role in stoich if role == "product"]

        def side(pairs):
            parts = []
            for name, coef in pairs:
                parts.append(f"{int(coef)}{name}" if coef != 1.0 else name)
            return "+".join(parts)

        arrow = "<=>" if rxn.get("reversible", 1) else "=>"
        return f"{side(reactants)}{arrow}{side(products)}"

    @staticmethod
    def _validate_yaml(yaml_path: str) -> None:
        """Round-trip load via cantera.Solution; log warning on failure."""
        try:
            import cantera as ct
            ct.Solution(yaml_path)
            log.debug("Cantera round-trip validation passed for %s", yaml_path)
        except ImportError:
            log.debug("Cantera not available — skipping YAML validation")
        except Exception as exc:
            log.warning(
                "Cantera YAML validation failed for %s: %s — "
                "file was written but may contain issues",
                yaml_path, exc,
            )


# ---------------------------------------------------------------------------
# YAML utility functions
# ---------------------------------------------------------------------------

def _yaml_safe(value: str) -> str:
    """Wrap a string in quotes if it contains special YAML characters."""
    if any(c in value for c in ':{}[],&*#?|-<>=!%@`"\'\\'):
        return f'"{value}"'
    return value


def _yaml_coeff_list(coeffs: List[float]) -> str:
    """Format a list of NASA-7 coefficients as a compact YAML array."""
    return "[" + ", ".join(f"{c:.8E}" for c in coeffs) + "]"


def _yaml_str_list(items: List[str], indent: int = 0) -> str:
    """Format a list of strings as a YAML inline list."""
    return "[" + ", ".join(items) + "]"


def _yaml_dict_inline(d: Dict[str, float]) -> str:
    """Format a dict as a YAML inline mapping."""
    pairs = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(d.items()))
    return "{" + pairs + "}"


def _parse_composition(formula: str) -> Dict[str, int]:
    """Parse Hill-notation formula into {element: count} dict."""
    result: Dict[str, int] = {}
    for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            result[el] = result.get(el, 0) + (int(cnt) if cnt else 1)
    return result
