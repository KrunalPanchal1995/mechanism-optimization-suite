"""
rmip/io/chemkin_writer.py
==========================
Exports an RMIP DB mechanism to CHEMKIN-II format (.inp + optional therm.dat).

What is written
---------------
ELEMENTS block:   unique elements across all species (inferred from formula)
SPECIES block:    all species in the mechanism
THERMO block:     NASA-7 polynomials in CHEMKIN-III format
REACTIONS block:  kinetics for all reactions

Supported kinetics types
------------------------
  arrhenius        → plain A n Ea line
  third_body       → A n Ea + efficiency lines
  troe             → LOW / TROE lines
  plog             → PLOG lines
  chebyshev        → CHEB lines
  multi_arrhenius  → duplicate Arrhenius lines (DUPLICATE flag)
  lindemann        → LOW line only (no TROE)
  falloff          → alias for troe

Transport data
--------------
NOT written (design decision D6).

Cantera YAML conversion
-----------------------
convert_to_cantera_yaml() calls cantera.ck2yaml (via subprocess) to produce a
Cantera YAML from the written CHEMKIN file.  This is used internally by the
simulation runners (IDTRunner, LFSRunner) to load the mechanism in Cantera.

Units in output
---------------
  A  : as stored in reactions.A_orig / A_units (if A_units == 'SI', converts)
  Ea : kcal/mol (CHEMKIN convention); stored SI → multiply by 1/4184.
  n  : dimensionless

Usage
-----
>>> from rmip.io.chemkin_writer import ChemkinWriter
>>> writer = ChemkinWriter("project.db")
>>> writer.write(mechanism_id=1, output_path="output/mech.inp")
>>> yaml_path = writer.convert_to_cantera_yaml("output/mech.inp")
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("rmip.io.chemkin_writer")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# J/mol → kcal/mol  (CHEMKIN convention for activation energy)
J_MOL_TO_KCAL_MOL = 1.0 / 4184.0

# SI pre-exponential: if A_units is "SI", convert to CHEMKIN CGS-molar units
# (cm³, mol, s).  The conversion factor depends on reaction order:
#   bimolecular : 1 m³ mol⁻¹ s⁻¹ = 1e6 cm³ mol⁻¹ s⁻¹
#   unimolecular: 1 s⁻¹           = 1 s⁻¹
# We default to bimolecular (factor 1e6) when order is unknown.
SI_TO_CGS_BIMOL = 1.0e6  # m³ mol⁻¹ s⁻¹ → cm³ mol⁻¹ s⁻¹

# Line width for CHEMKIN formatted files
CHEMKIN_LINE_WIDTH = 80

# NASA-7 output format
_NASA7_FMT1 = "{:>24.8E}{:>24.8E}{:>24.8E}{:>24.8E}{:>24.8E}"
_NASA7_FMT2 = "{:>24.8E}{:>24.8E}{:>24.8E}{:>24.8E}{:>24.8E}"
_NASA7_FMT3 = "{:>24.8E}{:>24.8E}{:>24.8E}"

# CHEMKIN elements recognised (extend as needed)
_ALL_ELEMENTS = {
    "C", "H", "O", "N", "S", "AR", "HE", "CL", "F", "SI", "P",
    "FE", "NA", "CA", "MG", "AL", "K",
}


# ─────────────────────────────────────────────────────────────────────────────
# ChemkinWriter
# ─────────────────────────────────────────────────────────────────────────────

class ChemkinWriter:
    """
    Writes an RMIP DB mechanism to CHEMKIN-II format.

    Parameters
    ----------
    db_path : str | Path
    use_original_units : bool
        If True, use A_orig / A_units / Ea_orig / Ea_units for output (no
        unit conversion).  If False (default), use A_si / Ea_si and convert
        to CHEMKIN CGS-molar units.
    """

    def __init__(
        self,
        db_path: str | Path,
        use_original_units: bool = True,
    ) -> None:
        self._db_path = str(db_path)
        self._use_orig = use_original_units

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def write(
        self,
        mechanism_id: int,
        output_path:  str,
        thermo_path:  Optional[str] = None,
        include_thermo_in_mech: bool = True,
    ) -> None:
        """
        Write mechanism to ``output_path`` in CHEMKIN-II format.

        Parameters
        ----------
        mechanism_id          : int
        output_path           : str — path for the .inp file
        thermo_path           : str | None — if given, write thermo to a separate file
        include_thermo_in_mech: bool — embed THERMO block in .inp (default: True)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            species_rows  = self._load_species(conn, mechanism_id)
            reaction_rows = self._load_reactions(conn, mechanism_id)
            stoich_map    = self._load_stoichiometry(conn, mechanism_id)
            eff_map       = self._load_efficiencies(conn, mechanism_id)

        lines: List[str] = []

        # Header comment
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            mrow = conn.execute(
                "SELECT name, description FROM mechanisms WHERE id=?", (mechanism_id,)
            ).fetchone()
        mname = mrow["name"] if mrow else f"mechanism_{mechanism_id}"
        mdesc = mrow["description"] if mrow else ""

        lines += [
            f"! Mechanism: {mname}",
            f"! Generated by RMIP ChemkinWriter",
            f"! {mdesc}",
            "!",
        ]

        # ELEMENTS
        lines += self._write_elements(species_rows)

        # SPECIES
        lines += self._write_species(species_rows)

        # THERMO
        if include_thermo_in_mech:
            lines += self._write_thermo(species_rows)

        # REACTIONS
        lines += self._write_reactions(reaction_rows, stoich_map, eff_map)

        lines.append("END\n")

        with open(output_path, "w") as fh:
            fh.write("\n".join(lines))

        log.info("CHEMKIN file written: %s (%d species, %d reactions)",
                 output_path, len(species_rows), len(reaction_rows))

        # Separate thermo file
        if thermo_path:
            thermo_lines = ["THERMO ALL"] + self._write_thermo(species_rows)[1:-1] + ["END\n"]
            with open(thermo_path, "w") as fh:
                fh.write("\n".join(thermo_lines))
            log.info("Thermo file written: %s", thermo_path)

    def convert_to_cantera_yaml(
        self,
        chemkin_path: str,
        thermo_path:  Optional[str] = None,
        output_yaml:  Optional[str] = None,
        ck2yaml_exe:  str = "ck2yaml",
    ) -> str:
        """
        Convert the written CHEMKIN file to a Cantera YAML file using ck2yaml.

        Parameters
        ----------
        chemkin_path : str — path to .inp file
        thermo_path  : str | None — separate thermo file (if used)
        output_yaml  : str | None — path for YAML; default = chemkin_path.replace('.inp', '.yaml')
        ck2yaml_exe  : str — ck2yaml executable name (must be on PATH or full path)

        Returns
        -------
        str — path to the generated YAML file
        """
        if output_yaml is None:
            output_yaml = str(Path(chemkin_path).with_suffix(".yaml"))

        cmd = [ck2yaml_exe, f"--input={chemkin_path}"]
        if thermo_path:
            cmd.append(f"--thermo={thermo_path}")
        cmd.append(f"--output={output_yaml}")
        # Transport not included (D6)

        log.info("Running ck2yaml: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except FileNotFoundError:
            raise RuntimeError(
                f"ck2yaml not found: '{ck2yaml_exe}'.  "
                "Install Cantera or add ck2yaml to PATH."
            )

        if proc.returncode != 0:
            log.error("ck2yaml failed:\n%s", proc.stderr)
            raise RuntimeError(f"ck2yaml failed (exit {proc.returncode}):\n{proc.stderr}")

        log.info("Cantera YAML written: %s", output_yaml)
        return output_yaml

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — DB loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_species(self, conn: sqlite3.Connection, mech_id: int) -> List[dict]:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM species WHERE mechanism_id=? ORDER BY id", (mech_id,)
            )
        ]

    def _load_reactions(self, conn: sqlite3.Connection, mech_id: int) -> List[dict]:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM reactions WHERE mechanism_id=? ORDER BY id", (mech_id,)
            )
        ]

    def _load_stoichiometry(
        self, conn: sqlite3.Connection, mech_id: int
    ) -> Dict[int, List[Tuple[str, float, str]]]:
        """Returns {reaction_id: [(species_name, stoich, role), ...]}"""
        rows = conn.execute(
            """
            SELECT r.id AS rxn_id, sp.name, s.stoich_coeff, s.role
              FROM stoichiometry s
              JOIN reactions r ON r.id = s.reaction_id
              JOIN species   sp ON sp.id = s.species_id
             WHERE r.mechanism_id = ?
            ORDER BY r.id, s.role, sp.name
            """,
            (mech_id,),
        ).fetchall()
        out: Dict[int, List[Tuple[str, float, str]]] = {}
        for row in rows:
            out.setdefault(row["rxn_id"], []).append(
                (row["name"], row["stoich_coeff"], row["role"])
            )
        return out

    def _load_efficiencies(
        self, conn: sqlite3.Connection, mech_id: int
    ) -> Dict[int, Dict[str, float]]:
        """Returns {reaction_id: {species_name: efficiency}}"""
        # Check if table exists
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='third_body_efficiencies'"
        ).fetchone()
        if not tbl:
            return {}

        rows = conn.execute(
            """
            SELECT r.id AS rxn_id, sp.name, t.efficiency
              FROM third_body_efficiencies t
              JOIN reactions r ON r.id = t.reaction_id
              JOIN species sp ON sp.id = t.species_id
             WHERE r.mechanism_id = ?
            """,
            (mech_id,),
        ).fetchall()
        out: Dict[int, Dict[str, float]] = {}
        for row in rows:
            out.setdefault(row["rxn_id"], {})[row["name"]] = row["efficiency"]
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — CHEMKIN section writers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_elements(self, species_rows: List[dict]) -> List[str]:
        """Infer elements from species formulas and write ELEMENTS block."""
        element_set: set = set()
        for sp in species_rows:
            formula = sp.get("formula") or ""
            for el in re.findall(r"[A-Z][a-z]?", formula):
                element_set.add(el.upper())

        # Ensure at least C H O N
        element_set.update({"C", "H", "O", "N"})
        element_set &= _ALL_ELEMENTS   # keep only known elements

        # Canonical order
        order = ["C", "H", "O", "N", "S", "AR", "HE", "CL", "F"]
        ordered = [e for e in order if e in element_set]
        ordered += sorted(element_set - set(order))

        lines = ["ELEMENTS"]
        lines += [f"  {' '.join(ordered)}"]
        lines += ["END", ""]
        return lines

    def _write_species(self, species_rows: List[dict]) -> List[str]:
        """Write SPECIES block."""
        lines = ["SPECIES"]
        names = [sp["name"] for sp in species_rows]
        # 5 species per line
        for i in range(0, len(names), 5):
            lines.append("  " + "  ".join(f"{n:<15}" for n in names[i:i+5]))
        lines += ["END", ""]
        return lines

    def _write_thermo(self, species_rows: List[dict]) -> List[str]:
        """Write THERMO block in CHEMKIN-III / NASA-7 format."""
        lines = ["THERMO ALL"]
        lines.append(f"   300.00   1000.00   5000.00")  # default Trange header

        for sp in species_rows:
            therm_lines = self._format_nasa7(sp)
            if therm_lines:
                lines.extend(therm_lines)
            else:
                log.debug("No thermo for species %s — skipping", sp.get("name"))

        lines += ["END", ""]
        return lines

    def _format_nasa7(self, sp: dict) -> List[str]:
        """Format NASA-7 thermo entry for one species (4 lines)."""
        try:
            low  = json.loads(sp.get("coeffs_low_json")  or "null")
            high = json.loads(sp.get("coeffs_high_json") or "null")
        except (json.JSONDecodeError, TypeError):
            return []

        if not low or not high or len(low) < 7 or len(high) < 7:
            return []

        name   = (sp.get("name") or "UNK")[:16].ljust(16)
        formula = self._format_formula(sp.get("formula") or "")
        tmin   = sp.get("tmin_K") or 200.0
        tmid   = sp.get("tmid_K") or 1000.0
        tmax   = sp.get("tmax_K") or 5000.0

        # Line 1: name, date, formula, phase, T ranges, #coefficients
        line1 = (
            f"{name}{'':10}"
            f"{formula:<20}"
            f"G{tmin:10.3f}{tmax:10.3f}{tmid:8.3f}      1"
        )
        # Line 2: high-T coefficients 1-5
        line2 = "".join(f"{c:15.8E}" for c in high[:5]) + "    2"
        # Line 3: high-T 6-7 + low-T 1-3
        line3 = "".join(f"{c:15.8E}" for c in (high[5:7] + low[:3])) + "    3"
        # Line 4: low-T 4-7
        line4 = "".join(f"{c:15.8E}" for c in low[3:7]) + "                   4"

        return [line1, line2, line3, line4]

    @staticmethod
    def _format_formula(formula: str) -> str:
        """Format Hill-notation formula into CHEMKIN 4-element 2-char columns."""
        # Parse element counts
        counts: Dict[str, int] = {}
        for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
            if el:
                counts[el] = counts.get(el, 0) + (int(cnt) if cnt else 1)
        # CHEMKIN format: up to 4 elements, each as "El##" in 2+3 chars
        order = ["C", "H", "O", "N"]
        other = sorted(k for k in counts if k not in order)
        elements = [(e, counts[e]) for e in order if e in counts]
        elements += [(e, counts[e]) for e in other]
        out = ""
        for el, cnt in elements[:4]:
            out += f"{el:<2}{cnt:>3}"
        return out.ljust(20)

    def _write_reactions(
        self,
        reaction_rows: List[dict],
        stoich_map:    Dict[int, List[Tuple[str, float, str]]],
        eff_map:       Dict[int, Dict[str, float]],
    ) -> List[str]:
        """Write REACTIONS block."""
        lines = ["REACTIONS  CAL/MOLE", ""]

        duplicates_seen: set = set()

        for rxn in reaction_rows:
            rxn_lines = self._format_reaction(rxn, stoich_map, eff_map, duplicates_seen)
            lines.extend(rxn_lines)
            lines.append("")

        lines += ["END", ""]
        return lines

    def _format_reaction(
        self,
        rxn:            dict,
        stoich_map:     Dict[int, List[Tuple[str, float, str]]],
        eff_map:        Dict[int, Dict[str, float]],
        duplicates_seen: set,
    ) -> List[str]:
        """Format a single reaction into CHEMKIN lines."""
        rxn_id = rxn["id"]
        ktype  = (rxn.get("kinetics_type") or "arrhenius").lower()

        # ── Determine A / n / Ea for output ──────────────────────────────
        if self._use_orig and rxn.get("A_orig") is not None:
            A_out  = rxn["A_orig"]
            Ea_out = rxn.get("Ea_orig") or 0.0
            # Convert Ea to kcal/mol if stored in SI
            ea_units = (rxn.get("Ea_units") or "").upper()
            if "J" in ea_units and "CAL" not in ea_units:
                Ea_out *= J_MOL_TO_KCAL_MOL
        else:
            A_out  = (rxn.get("A_si") or 0.0) * SI_TO_CGS_BIMOL
            Ea_out = (rxn.get("Ea_si") or 0.0) * J_MOL_TO_KCAL_MOL

        n_out = rxn.get("n") or 0.0

        # ── Build reaction string from stoichiometry ──────────────────────
        stoich = stoich_map.get(rxn_id, [])
        rxn_str = self._format_rxn_string(rxn, stoich)

        lines: List[str] = []

        # Comment
        if rxn.get("rmg_comment"):
            lines.append(f"! {rxn['rmg_comment']}")

        # Duplicate flag (place BEFORE the reaction line)
        ck = rxn.get("canonical_key", "")
        if rxn.get("duplicate"):
            if ck not in duplicates_seen:
                duplicates_seen.add(ck)
            lines.append("DUPLICATE")

        # ── Kinetics-type-specific output ─────────────────────────────────
        if ktype in ("arrhenius", "multi_arrhenius", "unknown"):
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")

        elif ktype in ("third_body",):
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append(self._format_efficiencies(effs))

        elif ktype in ("troe", "falloff", "lindemann"):
            kdata = self._parse_kdata(rxn)
            # High-pressure limit
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            # Low-pressure limit
            low = kdata.get("low_rate", {})
            if low:
                A_low  = low.get("A", 0.0)
                n_low  = low.get("n", 0.0)
                Ea_low = low.get("Ea", 0.0)
                if not self._use_orig:
                    A_low  *= SI_TO_CGS_BIMOL
                    Ea_low *= J_MOL_TO_KCAL_MOL
                lines.append(f"  LOW / {A_low:.6E}  {n_low:.3f}  {Ea_low:.3f} /")
            # Troe parameters
            troe = kdata.get("troe_params", {})
            if troe and ktype == "troe":
                a  = troe.get("A", 0.0)
                t3 = troe.get("T3", 1e30)
                t1 = troe.get("T1", 1e30)
                t2 = troe.get("T2")
                if t2 is not None:
                    lines.append(f"  TROE / {a:.4f}  {t3:.3E}  {t1:.3E}  {t2:.3E} /")
                else:
                    lines.append(f"  TROE / {a:.4f}  {t3:.3E}  {t1:.3E} /")
            # Efficiencies
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append(self._format_efficiencies(effs))

        elif ktype == "plog":
            kdata = self._parse_kdata(rxn)
            # Placeholder high-P line required by CHEMKIN parser
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            for entry in kdata.get("plog_rates", []):
                P_atm  = entry.get("P_atm", 1.0)
                A_p    = entry.get("A", 0.0)
                n_p    = entry.get("n", 0.0)
                Ea_p   = entry.get("Ea", 0.0)
                if not self._use_orig:
                    A_p  *= SI_TO_CGS_BIMOL
                    Ea_p *= J_MOL_TO_KCAL_MOL
                lines.append(f"  PLOG / {P_atm:.4E}  {A_p:.6E}  {n_p:.3f}  {Ea_p:.3f} /")

        elif ktype == "chebyshev":
            kdata = self._parse_kdata(rxn)
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            T_lims = kdata.get("T_limits", [300, 3000])
            P_lims = kdata.get("P_limits", [0.001, 100])
            n_T = kdata.get("n_T", 3)
            n_P = kdata.get("n_P", 3)
            coeffs = kdata.get("coefficients", [])
            lines.append(f"  TCHEB / {T_lims[0]:.1f}  {T_lims[1]:.1f} /")
            lines.append(f"  PCHEB / {P_lims[0]:.4E}  {P_lims[1]:.4E} /")
            lines.append(f"  CHEB / {n_T}  {n_P} /")
            for row in coeffs:
                row_str = "  ".join(f"{c:.6E}" for c in row)
                lines.append(f"  CHEB / {row_str} /")

        else:
            # Unknown / unsupported — write as plain Arrhenius with a warning comment
            log.warning(
                "Unsupported kinetics type '%s' for reaction id=%d — writing as Arrhenius",
                ktype, rxn_id,
            )
            lines.append(f"! WARNING: kinetics_type={ktype}, written as plain Arrhenius")
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")

        return lines

    def _format_rxn_string(self, rxn: dict, stoich: List[Tuple[str, float, str]]) -> str:
        """Build 'A+B<=>C+D' or 'A+B(+M)<=>C+D(+M)' string."""
        if rxn.get("formula_string"):
            # Use stored formula string — most faithful representation
            return rxn["formula_string"]

        reactants = [
            (name, coef)
            for name, coef, role in stoich
            if role in ("reactant", "reactant_reverse")
        ]
        products = [
            (name, coef)
            for name, coef, role in stoich
            if role in ("product", "product_reverse")
        ]

        def species_str(pairs: List[Tuple[str, float]]) -> str:
            parts = []
            for name, coef in pairs:
                if coef == 1.0:
                    parts.append(name)
                elif coef == int(coef):
                    parts.append(f"{int(coef)}{name}")
                else:
                    parts.append(f"{coef:.2f}{name}")
            return "+".join(parts)

        arrow = "<=>" if rxn.get("reversible", 1) else "=>"
        rxn_str = f"{species_str(reactants)}{arrow}{species_str(products)}"

        has_tb = rxn.get("has_third_body") or rxn.get("pressure_dependent")
        ktype  = (rxn.get("kinetics_type") or "").lower()
        if has_tb and ktype in ("troe", "falloff", "lindemann"):
            # Add (+M)
            rxn_str = rxn_str.replace("<=>", "(+M)<=>").replace("(+M)<=>", "(+M)<=>", 1)
            if "(+M)" not in rxn_str.split("<=>")[1]:
                rxn_str = rxn_str + "(+M)"
        elif has_tb and ktype == "third_body":
            rxn_str += "+M"

        return rxn_str

    @staticmethod
    def _format_efficiencies(effs: Dict[str, float]) -> str:
        """Format third-body efficiency line."""
        parts = [f"{sp}/{eff:.3f}/" for sp, eff in sorted(effs.items())]
        return "  " + "  ".join(parts)

    @staticmethod
    def _parse_kdata(rxn: dict) -> dict:
        """Parse kinetics_data_json field."""
        raw = rxn.get("kinetics_data_json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
