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
  Ea : cal/mol (REACTIONS CAL/MOLE header); stored SI (J/mol) → multiply by 1/4.184.
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

# J/mol → cal/mol  (this writer's REACTIONS header declares CAL/MOLE, matching
# _chemkin_parser.py's _normalize_ea_to_cal(), which always normalizes
# Ea_orig to cal/mol at parse time regardless of the source file's header --
# B15: this used to be 1/4184 (a kcal/mol factor) written under a CAL/MOLE
# header, silently shrinking every Ea by 1000x whenever A_orig was NULL
# (i.e. every fusion/optimize/unify product and every yaml-sourced mechanism).
J_MOL_TO_CAL_MOL = 1.0 / 4.184

# SI pre-exponential: if A_units is "SI", convert to CHEMKIN CGS-molar units
# (cm³, mol, s).  The conversion factor depends on reaction order:
#   unimolecular  (1/s):              1 s⁻¹            → 1 s⁻¹        (factor 1)
#   bimolecular   (m³/mol/s):         1 m³ mol⁻¹ s⁻¹   → 1e6 cm³ mol⁻¹ s⁻¹
#   termolecular  (m⁶/mol²/s):        1 m⁶ mol⁻² s⁻¹   → 1e12 cm⁶ mol⁻² s⁻¹
SI_TO_CGS_BIMOL = 1.0e6  # m³ mol⁻¹ s⁻¹ → cm³ mol⁻¹ s⁻¹ (kept for reference/back-compat)


def _si_to_cgs_factor(reaction_order: int) -> float:
    """SI (m, mol, s) → CGS (cm, mol, s) A-factor conversion for the given order."""
    return 10.0 ** (6 * (reaction_order - 1))

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
    force : bool
        If False (default), raise ValueError when a reaction whose
        kinetics require a representative Arrhenius A (i.e. every type
        except chebyshev, which has none by design) has no usable A value
        -- writing `0.000000E+00` for "we have no rate data" silently
        misrepresents it as "rate is exactly zero" (M7, review.md). Pass
        True to write the fabricated zero anyway (e.g. for inspecting an
        otherwise-broken mechanism).
    """

    def __init__(
        self,
        db_path: str | Path,
        use_original_units: bool = True,
        force: bool = False,
    ) -> None:
        self._db_path = str(db_path)
        self._use_orig = use_original_units
        self._force = force

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def write_transport(self, mechanism_id: int, output_path: str) -> int:
        """
        Write a CHEMKIN transport file from the mechanism's stored transport
        columns (Phase-3 follow-on).  Returns the number of species written.
        Species with no transport data in the DB are skipped.
        """
        import sqlite3
        from rmip.io.transport_io import TransportData, write_chemkin_transport

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT name, transport_geometry, well_depth_K, diameter_A, "
                "dipole_D, polarizability_A3, rot_relax, transport_source "
                "FROM species WHERE mechanism_id=?", (mechanism_id,)
            ).fetchall()
        finally:
            conn.close()

        records = {}
        for r in rows:
            if r["transport_geometry"] is None or r["well_depth_K"] is None:
                continue
            records[r["name"]] = TransportData(
                geometry=r["transport_geometry"], well_depth_K=r["well_depth_K"],
                diameter_A=r["diameter_A"], dipole_D=r["dipole_D"] or 0.0,
                polarizability_A3=r["polarizability_A3"] or 0.0,
                rot_relax=r["rot_relax"] or 0.0,
                source=r["transport_source"] or "")
        text = write_chemkin_transport(
            records, header=f"Transport for mechanism_id={mechanism_id}")
        with open(output_path, "w") as fh:
            fh.write(text)
        log.info("Wrote CHEMKIN transport for %d species → %s",
                 len(records), output_path)
        return len(records)

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

        # REACTIONS (its own terminating "END" closes the file -- CHEMKIN's
        # REACTIONS block is always the last section, no further top-level
        # "END" is needed. A previously-unconditional extra "END" here
        # produced a duplicate terminator that Cantera's own ck2yaml rejects
        # outright ("Section starts with unrecognized keyword") -- silently
        # tolerated everywhere else this writer's output was consumed
        # (Zero-RK's parser, this repo's own tests) until ck2yaml was
        # actually exercised against it end-to-end.
        lines += self._write_reactions(reaction_rows, stoich_map, eff_map)

        with open(output_path, "w") as fh:
            fh.write("\n".join(lines))

        log.info("CHEMKIN file written: %s (%d species, %d reactions)",
                 output_path, len(species_rows), len(reaction_rows))

        # Separate thermo file
        if thermo_path:
            # _write_thermo() already appends its own ["END", ""] terminator
            # -- slicing [1:-1] only dropped its trailing "", leaving "END"
            # in the body and producing a duplicate "END" line once this
            # caller's own terminator was appended below.
            thermo_lines = ["THERMO ALL"] + self._write_thermo(species_rows)[1:-2] + ["END", ""]
            with open(thermo_path, "w") as fh:
                fh.write("\n".join(thermo_lines))
            log.info("Thermo file written: %s", thermo_path)

    def convert_to_cantera_yaml(
        self,
        chemkin_path: str,
        thermo_path:  Optional[str] = None,
        output_yaml:  Optional[str] = None,
        ck2yaml_exe:  str = "ck2yaml",
        verify_loads: bool = True,
    ) -> str:
        """
        Convert the written CHEMKIN file to a Cantera YAML file using ck2yaml.

        Parameters
        ----------
        chemkin_path : str — path to .inp file
        thermo_path  : str | None — separate thermo file (if used)
        output_yaml  : str | None — path for YAML; default = chemkin_path.replace('.inp', '.yaml')
        ck2yaml_exe  : str — ck2yaml executable name (must be on PATH or full path)
        verify_loads : bool — after ck2yaml succeeds, attempt a real
            ``cantera.Solution()`` load of the output and raise if it fails
            (default True). ck2yaml's own exit code does NOT catch every
            defect — most notably "Undeclared duplicate reactions" (bug.md:
            the suite's own canonical fused Keromnes/Li/Davis mechanism was
            written this way, ck2yaml exited 0, and the broken .yaml sat in
            the repo for weeks before anyone noticed it couldn't actually be
            loaded). Cantera's own duplicate/consistency checks only run at
            ``Solution()`` construction, not during ck2yaml's YAML-writing
            pass — this is exactly the gap that let a broken mechanism
            through as a silent "success". Set False only for callers that
            deliberately want to inspect a possibly-invalid YAML (e.g. a
            diagnostic tool), never for a normal production write.

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

        if verify_loads:
            try:
                import warnings

                import cantera as ct
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ct.Solution(output_yaml)
            except Exception as exc:                      # noqa: BLE001
                raise RuntimeError(
                    f"ck2yaml wrote {output_yaml!r} (exit 0) but it does not "
                    f"load in Cantera: {exc}\n"
                    "This is a data defect ck2yaml's own exit code does not "
                    "catch (e.g. an undeclared duplicate reaction from a "
                    "third-body/canonicalization mismatch — see bug.md's "
                    "canonical-key-rebuild migration). Run "
                    "`rmip check` on the source mechanism(s) for diagnostics "
                    "before re-fusing/re-converting."
                )

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

        # NOTE: standard CHEMKIN NASA-7 fixed columns are name(1-18) + date(19-24)
        # + formula(25-44) + phase(45) + Tlow(46-55) + Thigh(56-65) + Tmid(66-73/74).
        # This was previously name(16)+blank(10) = 26 wide, 2 columns too wide,
        # which pushed the phase char off column 45 and broke Zero-RK's thermo-
        # block species lookup for every species (confirmed empirically -- every
        # species came back "undefined" against a correctly-read SPECIES block --
        # see ZERORK_GSA_PLAN.md Phase 1).
        name   = (sp.get("name") or "UNK")[:18].ljust(18)
        formula = self._format_formula(sp.get("formula") or "")
        tmin   = sp.get("tmin_K") or 200.0
        tmid   = sp.get("tmid_K") or 1000.0
        tmax   = sp.get("tmax_K") or 5000.0

        # Degenerate/single-range species (Tmid at or past one end of
        # [Tmin, Tmax] -- e.g. noble gases like AR sourced verbatim from
        # some source thermo sets record Tmid==Thigh=6000, meaning there is
        # really only ONE valid NASA-7 fit, not two). ck2yaml treats
        # Tmid==Tmin/Tmid==Tmax as "one range" and requires the low/high
        # coefficient blocks to be IDENTICAL in that case -- if they differ
        # (as they did here: this writer always emitted two separately-
        # sourced sets regardless), it raises InputError
        # ("Only one temperature range defined but two distinct sets of
        # coefficients given") deep inside Cantera's own ck2yaml, crashing
        # the fusion export with a confusing AttributeError while formatting
        # that error message (see cantera/ck2yaml.py's read_NASA7_entry).
        # Bug found stress-testing the launcher (Test_run_23_JULY/
        # mean_value_fusion_gui/SUMMARY.md). Fix: when the range is
        # degenerate, write the SAME (low) coefficients into both blocks so
        # ck2yaml's high_coeffs == low_coeffs branch accepts it cleanly.
        if tmid <= tmin or tmid >= tmax:
            high = low

        # Line 1: name, date, formula, phase, T ranges, #coefficients
        # NOTE: tmid's field is 9 wide (not the standard 8) so there's always a
        # whitespace gap before it -- at exactly 8 wide, tmid=1000.000 (the
        # overwhelmingly common case) exactly fills the field with no leading
        # space, so it runs directly into Thigh's field with zero separators
        # (e.g. "3500.0001000.000"). Cantera's ck2yaml tolerates this; Zero-RK's
        # stricter token-based CHEMKIN parser does not (confirmed empirically).
        line1 = (
            f"{name}"
            f"{'':6}"
            f"{formula:<20}"
            f"G{tmin:10.3f}{tmax:10.3f}{tmid:9.3f}     1"
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
        # Parse element counts. NOTE: symbols are upper-cased here to match
        # the convention _write_elements uses for the file's ELEMENTS block
        # (and what real CHEMKIN files in the wild actually use, e.g. "AR"
        # not "Ar") -- Zero-RK's parser does an exact-string match between a
        # species' declared elements and the ELEMENTS block, so a case
        # mismatch here (title-case "Ar" vs uppercase "AR") is a hard parse
        # failure ("undeclared element Ar"), confirmed empirically.
        counts: Dict[str, int] = {}
        for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
            if el:
                el = el.upper()
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

        for rxn in reaction_rows:
            rxn_lines = self._format_reaction(rxn, stoich_map, eff_map)
            lines.extend(rxn_lines)
            lines.append("")

        lines += ["END", ""]
        return lines

    def _format_reaction(
        self,
        rxn:            dict,
        stoich_map:     Dict[int, List[Tuple[str, float, str]]],
        eff_map:        Dict[int, Dict[str, float]],
    ) -> List[str]:
        """Format a single reaction into CHEMKIN lines."""
        rxn_id = rxn["id"]
        ktype  = (rxn.get("kinetics_type") or "arrhenius").lower()

        # ── Stoichiometric reaction order (sum of reactant coefficients,
        #    not species count) — drives the SI→CGS A-factor conversion.
        #    "(+M)" fall-off partners are stripped before storage, so this
        #    is the intrinsic (high-P-limit) order; explicit third-body
        #    reactions store "M" as a real reactant, so it's already
        #    included in the sum.
        stoich = stoich_map.get(rxn_id, [])
        rxn_order = int(round(sum(
            coef for _, coef, role in stoich
            if role in ("reactant", "reactant_reverse")
        ))) or 1

        # ── Determine A / n / Ea for output ──────────────────────────────
        use_orig_path = self._use_orig and rxn.get("A_orig") is not None
        A_missing = (rxn.get("A_orig") if use_orig_path else rxn.get("A_si")) is None

        # M7: chebyshev has no representative single-Arrhenius A by design
        # (its rate law is the CHEB coefficient table) -- everything else
        # (arrhenius/third_body/troe/lindemann/plog/unknown) needs one, and
        # writing a fabricated 0.000000E+00 for "no data" silently claims
        # the rate constant is exactly zero.
        ktype_needs_A = (rxn.get("kinetics_type") or "arrhenius").lower() != "chebyshev"
        if A_missing and ktype_needs_A and not self._force:
            raise ValueError(
                f"Reaction id={rxn_id} ('{rxn.get('formula_string')}'): no "
                f"usable A value (A_si and A_orig both NULL) -- refusing to "
                f"write a fabricated 0.000000E+00. Pass force=True to "
                f"ChemkinWriter() to write it anyway."
            )

        if use_orig_path:
            A_out  = rxn["A_orig"]
            Ea_out = rxn.get("Ea_orig") or 0.0
            # _chemkin_parser.py normalizes Ea_orig to cal/mol unconditionally
            # at parse time (see _normalize_ea_to_cal), so Ea_units should
            # always read "cal/mol" here -- this is a defensive fallback for
            # any row written by something other than that parser.
            ea_units = (rxn.get("Ea_units") or "").upper()
            if "KCAL" in ea_units:
                Ea_out *= 1000.0
            elif "J" in ea_units and "CAL" not in ea_units:
                Ea_out *= J_MOL_TO_CAL_MOL
        else:
            A_out  = (rxn.get("A_si") or 0.0) * _si_to_cgs_factor(rxn_order)
            Ea_out = (rxn.get("Ea_si") or 0.0) * J_MOL_TO_CAL_MOL

        n_out = rxn.get("n") or 0.0

        # ── Build reaction string from stoichiometry ──────────────────────
        rxn_str = self._format_rxn_string(rxn, stoich)

        lines: List[str] = []

        # Comment
        if rxn.get("rmg_comment"):
            lines.append(f"! {rxn['rmg_comment']}")

        # ── Kinetics-type-specific output ─────────────────────────────────
        if ktype in ("arrhenius", "multi_arrhenius", "unknown"):
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")

        elif ktype in ("third_body",):
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append(self._format_efficiencies(effs))

        elif ktype in ("troe", "falloff", "lindemann"):
            # NOTE: this used to read kdata["low_rate"]/["troe_params"] with
            # inner keys "A"/"Ea" -- _chemkin_parser.py's _build_kinetics_data
            # actually stores {"low": {...}, "high": {...}} (lindemann) or
            # {"alpha", "T1", "T2", "T3", "low", "high"} (troe), inner keys
            # "A_si"/"n"/"Ea_si". The old key names never matched anything the
            # parser writes, so `low`/`troe` here were always {} and every
            # falloff reaction silently lost its LOW/TROE block on export
            # (confirmed empirically -- see ZERORK_GSA_PLAN.md Phase 1). Also
            # note despite the "_si" suffix, these inner values are the raw
            # CHEMKIN-original-unit numbers as parsed (no SI conversion is
            # actually applied to them anywhere) -- so they're passed through
            # as-is regardless of self._use_orig; there is no true-SI source
            # to convert *from* for this sub-block, unlike the main A_out/
            # Ea_out path above which does have a real SI representation.
            kdata = self._parse_kdata(rxn)
            # High-pressure limit
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            # Low-pressure limit
            low = kdata.get("low", {})
            if low:
                A_low  = low.get("A_si", 0.0)
                n_low  = low.get("n", 0.0)
                Ea_low = low.get("Ea_si", 0.0)
                lines.append(f"  LOW / {A_low:.6E}  {n_low:.3f}  {Ea_low:.3f} /")
            elif not self._force:
                # A pressure-dependent (troe/falloff/lindemann) reaction
                # with NO low-pressure-limit data at all used to silently
                # write only the high-P line -- CHEMKIN-legal-looking
                # output that ck2yaml then rejects with "Reaction equation
                # implies pressure dependence but no alternate rate
                # parameters (such as HIGH or LOW) were given", and
                # *formatting that error message* crashes deep inside
                # Cantera's own ck2yaml with
                # AttributeError: 'NoneType' object has no attribute
                # 'reaction_string_suffix' (reaction.kinetics is still None
                # when the error string is built) -- a confusing crash with
                # no indication of which reaction or repo-side file caused
                # it. Fail fast here instead, same escape hatch as the
                # A_missing check above. Bug found stress-testing the
                # launcher (Test_run_23_JULY/mean_value_fusion_gui/SUMMARY.md).
                raise ValueError(
                    f"Reaction id={rxn_id} ('{rxn.get('formula_string')}'): "
                    f"declared pressure-dependent (kinetics_type={ktype!r}) "
                    f"but has no low-pressure-limit (LOW) data -- refusing "
                    f"to write a high-P-only pressure-dependent reaction "
                    f"(Cantera's ck2yaml will reject it with a confusing "
                    f"crash). Pass force=True to ChemkinWriter() to write "
                    f"it anyway."
                )
            # Troe parameters
            if ktype == "troe" and kdata.get("alpha") is not None:
                a  = kdata.get("alpha", 0.0)
                t3 = kdata.get("T3", 1e30)
                t1 = kdata.get("T1", 1e30)
                t2 = kdata.get("T2")
                if t2 is not None:
                    lines.append(f"  TROE / {a:.4f}  {t3:.3E}  {t1:.3E}  {t2:.3E} /")
                else:
                    lines.append(f"  TROE / {a:.4f}  {t3:.3E}  {t1:.3E} /")
            # Efficiencies
            effs = eff_map.get(rxn_id, {})
            if effs:
                lines.append(self._format_efficiencies(effs))

        elif ktype == "plog":
            # NOTE: same key-name fix as the falloff branch above -- the
            # parser stores {"pressures": [Pa,...], "arrhenius": [{"A_si",
            # "n","Ea_si"}, ...]} (parallel lists), not a single "plog_rates"
            # list of {"P_atm","A","n","Ea"} dicts.
            kdata = self._parse_kdata(rxn)
            # Placeholder high-P line required by CHEMKIN parser
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            pressures_pa = kdata.get("pressures", [])
            arrhenius_entries = kdata.get("arrhenius", [])
            for P_pa, entry in zip(pressures_pa, arrhenius_entries):
                P_atm  = P_pa / 101325.0
                A_p    = entry.get("A_si", 0.0)
                n_p    = entry.get("n", 0.0)
                Ea_p   = entry.get("Ea_si", 0.0)
                lines.append(f"  PLOG / {P_atm:.4E}  {A_p:.6E}  {n_p:.3f}  {Ea_p:.3f} /")

        elif ktype == "chebyshev":
            # NOTE: same key-name fix -- parser stores flat
            # {"Tmin","Tmax","Pmin","Pmax","coeffs"} (Pmin/Pmax in Pa), not
            # "T_limits"/"P_limits"/"n_T"/"n_P"/"coefficients". n_T/n_P are
            # derived from the coefficient matrix shape since the parser
            # doesn't store them separately.
            kdata = self._parse_kdata(rxn)
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")
            tmin = kdata.get("Tmin") or 300.0
            tmax = kdata.get("Tmax") or 3000.0
            pmin_atm = (kdata.get("Pmin") or 101325.0) / 101325.0
            pmax_atm = (kdata.get("Pmax") or 10132500.0) / 101325.0
            coeffs = kdata.get("coeffs", [])
            n_T = len(coeffs)
            n_P = len(coeffs[0]) if coeffs else 0
            lines.append(f"  TCHEB / {tmin:.1f}  {tmax:.1f} /")
            lines.append(f"  PCHEB / {pmin_atm:.4E}  {pmax_atm:.4E} /")
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
            # (Round-3 audit fix, B13 -- WORKBENCH_AUDIT_2026-08-10.md) This
            # branch discards the reaction's falloff data (SRI and any
            # other kinetics type this writer doesn't model), but was still
            # writing the ORIGINAL equation string verbatim -- which, for a
            # falloff reaction, still carries "(+M)"/"(+species)" third-body
            # notation. The result is syntactically pressure-dependent
            # CHEMKIN with no LOW/TROE/SRI auxiliary line, which ck2yaml
            # correctly rejects on re-conversion ("Reaction equation
            # implies pressure [dependence]..."). Since this branch has
            # already committed to writing plain Arrhenius, the equation
            # string must also stop claiming to be falloff -- same regex
            # already used elsewhere in this file (see _format_rxn_string's
            # re-attachment of "(+M)" a few hundred lines below) to detect
            # this exact notation, used here in reverse to remove it.
            rxn_str = re.sub(r'\s*\(\+[^)]+\)', '', rxn_str).strip()
            lines.append(f"! WARNING: kinetics_type={ktype}, written as plain Arrhenius")
            lines.append(f"{rxn_str:<52} {A_out:.6E}  {n_out:.3f}  {Ea_out:.3f}")

        # Duplicate flag: CHEMKIN spec places DUPLICATE AFTER the reaction
        # it marks (and after that reaction's LOW/TROE/PLOG/efficiency
        # sub-lines) -- B7 (review.md): this used to be emitted BEFORE the
        # reaction line, which the parser's own (now-also-fixed) DUPLICATE
        # handling reads as marking the *following* reaction. Round-tripped
        # files marked the wrong member of each duplicate pair and ck2yaml
        # rejected them outright.
        if rxn.get("duplicate"):
            lines.append("DUPLICATE")

        return lines

    def _format_rxn_string(self, rxn: dict, stoich: List[Tuple[str, float, str]]) -> str:
        """Build 'A+B<=>C+D' or 'A+B(+M)<=>C+D(+M)' string."""
        if stoich:
            # Build from `stoich` (species names via a proper FK join against
            # THIS mechanism's own species table, see _load_stoichiometry) --
            # NOT from the cached `formula_string` column. formula_string is
            # written once at parse time using the SOURCE mechanism's own
            # species spelling; after fusion, the species this reaction's
            # stoichiometry rows actually point to may have been canonicalized
            # to a DIFFERENT representative name in the fused SPECIES block
            # (e.g. one source mechanism's "H2O" and another's differently-
            # spelled synonym both resolve to one chosen species row) --
            # `formula_string` does not know about that and goes stale,
            # producing reaction lines that reference species names never
            # declared in SPECIES. Confirmed empirically fusing 3 dummy
            # mechanisms with deliberately different spellings per shared
            # species (2026-07-17 GUI stress test): ck2yaml rejected the
            # fused .inp outright because reactions referenced undeclared
            # species from each source's original spelling.
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

            # Falloff/Troe/Lindemann reactions strip their "(+M)"/"(+AR)"/etc
            # collider partner before storing stoichiometry (see the class
            # comment), and that annotation has no other persisted home
            # besides the cached formula_string -- re-apply it here (same
            # text, both sides) so a stoich-rebuilt string doesn't silently
            # drop it. Only the annotation text is borrowed, never the
            # species spelling, so this can't reintroduce the staleness bug
            # above.
            stored = rxn.get("formula_string") or ""
            m = re.search(r'\(\+[^)]+\)', stored)
            if m and not re.search(r'\(\+[^)]+\)', rxn_str):
                tag = m.group(0)
                arrow_found = "<=>" if "<=>" in rxn_str else "=>"
                head, _, tail = rxn_str.partition(arrow_found)
                rxn_str = f"{head}{tag}{arrow_found}{tail}{tag}"
        elif rxn.get("formula_string"):
            # No stoichiometry rows available (shouldn't normally happen) --
            # fall back to the cached string rather than emit "<=>" alone.
            rxn_str = rxn["formula_string"]
        else:
            rxn_str = "<=>"

        has_tb = rxn.get("has_third_body") or rxn.get("pressure_dependent")
        ktype  = (rxn.get("kinetics_type") or "").lower()
        # Any "(+X)" collider annotation counts, not just literally "(+M)"
        # -- a specific-collider reaction's formula_string already carries
        # its own "(+AR)"/"(+HE)"/etc (see _chemkin_parser.py's
        # _build_formula_string); checking only for "(+M)" here would
        # wrongly ALSO append a generic "(+M)" on top of it.
        has_any_collider_tag = bool(re.search(r'\(\+[^)]+\)', rxn_str))
        if has_tb and ktype in ("troe", "falloff", "lindemann"):
            if not has_any_collider_tag:
                # "<=>" contains "=>" as a substring, so check for it first —
                # partitioning on the wrong (shorter) arrow would split the
                # reactant side mid-arrow and corrupt the string.
                arrow = "<=>" if "<=>" in rxn_str else "=>"
                head, _, tail = rxn_str.partition(arrow)
                head += "(+M)"
                tail += "(+M)"
                rxn_str = head + arrow + tail
        elif has_tb and ktype == "third_body" and "+M" not in rxn_str and "M" not in rxn_str:
            # Bare "+M" (no parens) belongs on BOTH sides of the equation
            # ("H2+M<=>2H+M"), not just appended once at the end -- that
            # used to produce an asymmetric "H2<=>2H+M" (M on the product
            # side only), which ck2yaml rejects ("Third bodies do not
            # match: 'None' and 'M'"). Found stress-testing fusion on
            # Keromnes/Li/Davis 2013-2015-2005 (2026-07-14): every
            # bare-third-body reaction's fused/priority-picked output hit
            # this.
            arrow = "<=>" if "<=>" in rxn_str else "=>"
            head, _, tail = rxn_str.partition(arrow)
            rxn_str = f"{head}+M{arrow}{tail}+M"

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
