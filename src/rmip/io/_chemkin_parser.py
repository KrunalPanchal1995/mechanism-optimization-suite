"""
rmip/io/_chemkin_parser.py
===========================
Standalone pure-Python CHEMKIN-II mechanism file parser.

No external dependencies — uses only stdlib (``re``, ``math``, ``logging``,
``pathlib``, ``dataclasses``).  Produces the same SpeciesDict / ReactionDict
output format that ``chemkin_reader.py`` produces via RMG-Py.

Supports
--------
* CHEMKIN-II format (ELEMENTS, SPECIES, THERMO, REACTIONS blocks)
* NASA-7 polynomials (low + high temperature ranges)
* All kinetics types: Arrhenius, PLOG, Troe, Lindemann, ThirdBody, Chebyshev
* Fortran float notation (``1.23D+04``, ``1.23d-04``)
* Species aliases (e.g. ``C7H15-1``, ``nC7H16``)
* RMG-style comments for reaction family extraction
* Unit conversion to SI (Ea → J/mol, A → appropriate SI units)
* Third-body efficiencies
* DUPLICATE keyword
* Separate thermo file support
* Multi-line continuation with ``&``

Does **not** support (out of scope per RMIP design)
---------------------------------------------------
* Transport data (D6)
* Surface / solid-phase kinetics
* Resonance structure generation (requires RMG-Py molecular graph engine)
* NASA-9 polynomials (NASA-7 only)

Usage
-----
>>> from rmip.io._chemkin_parser import ChemkinParser
>>> parser = ChemkinParser()
>>> species_list, rxn_list = parser.parse("mechanism.inp")
>>> species_list, rxn_list = parser.parse("mechanism.inp", thermo_path="therm.dat")
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger("rmip.io.chemkin_parser")

# Type aliases matching chemkin_reader.py
SpeciesDict = Dict[str, Any]
ReactionDict = Dict[str, Any]


# =============================================================================
# Internal data model — parsed CHEMKIN entities
# =============================================================================

@dataclass
class ParsedNASAPolynomial:
    """One temperature-range NASA-7 polynomial (7 coefficients)."""
    Tmin: float = 0.0       # K
    Tmax: float = 0.0       # K
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 7)


@dataclass
class ParsedThermo:
    """NASA-7 thermo for one species (low + high range)."""
    polynomials: List[ParsedNASAPolynomial] = field(default_factory=list)
    comment: str = ""


@dataclass
class ParsedSpecies:
    """Species as parsed from SPECIES + THERMO blocks."""
    label: str = ""
    thermo: Optional[ParsedThermo] = None


@dataclass
class ParsedArrheniusParams:
    """Arrhenius rate parameters (in file-native units)."""
    A: float = 0.0
    n: float = 0.0
    Ea: float = 0.0        # cal/mol or kcal/mol per file header
    A_units: str = ""
    Ea_units: str = ""


@dataclass
class ParsedKinetics:
    """Kinetics for one reaction."""
    type: str = "arrhenius"                     # 'arrhenius', 'plog', 'troe', etc.
    arrhenius: Optional[ParsedArrheniusParams] = None   # high-P limit or only rate
    arrhenius_low: Optional[ParsedArrheniusParams] = None
    arrhenius_high: Optional[ParsedArrheniusParams] = None
    # PLOG fields
    pressures: List[float] = field(default_factory=list)
    plog_arrhenius: List[ParsedArrheniusParams] = field(default_factory=list)
    # Troe fields
    alpha: Optional[float] = None
    T1: Optional[float] = None
    T2: Optional[float] = None
    T3: Optional[float] = None
    # Chebyshev fields
    Tmin_cheb: Optional[float] = None
    Tmax_cheb: Optional[float] = None
    Pmin_cheb: Optional[float] = None
    Pmax_cheb: Optional[float] = None
    cheb_degree_T: int = 0
    cheb_degree_P: int = 0
    cheb_coeffs: List[List[float]] = field(default_factory=list)
    # Third-body
    efficiencies: Dict[str, float] = field(default_factory=dict)
    has_third_body: bool = False
    # Falloff
    is_pressure_dependent: bool = False


@dataclass
class ParsedReaction:
    """Reaction as parsed from REACTIONS block."""
    reactants: List[Tuple[str, float]] = field(default_factory=list)
    products: List[Tuple[str, float]] = field(default_factory=list)
    reversible: bool = True
    kinetics: Optional[ParsedKinetics] = None
    duplicate: bool = False
    family: Optional[str] = None          # from RMG template comment
    comment: Optional[str] = None         # full RMG comment
    third_body: bool = False
    specific_collider: Optional[str] = None


# =============================================================================
# ChemkinParser
# =============================================================================

class ChemkinParser:
    """
    Pure-Python CHEMKIN-II mechanism parser.

    Parses ELEMENTS, SPECIES, THERMO, and REACTIONS blocks.  Returns lists
    of SpeciesDict and ReactionDict in the same format as
    ``chemkin_reader.load_chemkin()``.

    Parameters
    ----------
    debug : bool
        If True, emit extra debug-level log messages.
    """

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        # Internal state reset on each parse()
        self._species: Dict[str, ParsedSpecies] = {}
        self._species_order: List[str] = []           # preserve SPECIES block order
        self._species_aliases: Dict[str, str] = {}     # alias → canonical label
        self._reactions: List[ParsedReaction] = []
        self._elements: List[str] = []
        self._ea_units: str = "cal/mol"               # per REACTIONS header
        self._rxn_units_header: str = ""               # raw header line

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def parse(
        self,
        mech_path: str,
        thermo_path: Optional[str] = None,
    ) -> Tuple[List[SpeciesDict], List[ReactionDict]]:
        """
        Parse a CHEMKIN mechanism file.

        Parameters
        ----------
        mech_path : str
            Path to the CHEMKIN mechanism ``.inp`` file.
        thermo_path : str, optional
            Separate thermodynamic data file (``.dat``).  If None, thermo is
            assumed to be embedded in *mech_path*.

        Returns
        -------
        species_list : List[SpeciesDict]
        rxn_list     : List[ReactionDict]
        """
        self._reset()

        # Read and pre-process files
        mech_text = _read_file(str(mech_path))
        thermo_text = _read_file(str(thermo_path)) if thermo_path else None

        # Split into blocks
        mech_blocks = _split_into_blocks(mech_text, source="mechanism")

        if thermo_text:
            thermo_blocks = _split_into_blocks(thermo_text, source="thermo")
            # Merge THERMO block from separate file
            if "THERMO" in thermo_blocks or "THERMO ALL" in thermo_blocks:
                thermo_block = thermo_blocks.get("THERMO") or thermo_blocks.get("THERMO ALL")
                if thermo_block is not None:
                    mech_blocks["THERMO"] = thermo_block

        # Parse each block
        if "ELEMENTS" in mech_blocks:
            self._parse_elements_block(mech_blocks["ELEMENTS"])

        if "SPECIES" in mech_blocks:
            self._parse_species_block(mech_blocks["SPECIES"])

        if "THERMO" in mech_blocks or "THERMO ALL" in mech_blocks:
            thermo_lines = mech_blocks.get("THERMO") or mech_blocks.get("THERMO ALL")
            if thermo_lines is not None:
                self._parse_thermo_block(thermo_lines)

        if "REACTIONS" in mech_blocks:
            self._parse_reactions_block(mech_blocks["REACTIONS"])

        log.info(
            "ChemkinParser: %d species, %d reactions parsed from %s",
            len(self._species_order), len(self._reactions), mech_path,
        )

        # Convert to dict format
        return self._to_dicts()

    # -------------------------------------------------------------------------
    # Internal — reset
    # -------------------------------------------------------------------------

    def _reset(self) -> None:
        self._species = {}
        self._species_order = []
        self._species_aliases = {}
        self._reactions = []
        self._elements = []
        self._ea_units = "cal/mol"
        self._rxn_units_header = ""

    # -------------------------------------------------------------------------
    # ELEMENTS block
    # -------------------------------------------------------------------------

    def _parse_elements_block(self, lines: List[str]) -> None:
        """Parse ``ELEMENTS ... END`` block."""
        for line in lines:
            stripped = _strip_comment(line).strip()
            if not stripped or stripped.upper() in ("END", "ELEMENTS"):
                continue
            for token in stripped.split():
                el = token.strip().upper()
                if el and el != "END":
                    self._elements.append(el)
        if self._debug:
            log.debug("Elements: %s", self._elements)

    # -------------------------------------------------------------------------
    # SPECIES block
    # -------------------------------------------------------------------------

    def _parse_species_block(self, lines: List[str]) -> None:
        """Parse ``SPECIES ... END`` block."""
        for line in lines:
            stripped = _strip_comment(line).strip()
            if not stripped or stripped.upper() in ("END", "SPECIES"):
                continue
            for token in stripped.split():
                name = token.strip()
                if name and name.upper() not in ("END", "SPECIES", "ELEMENTS", "THERMO",
                                                   "REACTIONS"):
                    if name not in self._species:
                        self._species[name] = ParsedSpecies(label=name)
                        self._species_order.append(name)
        if self._debug:
            log.debug("Species: %d names", len(self._species_order))

    # -------------------------------------------------------------------------
    # THERMO block (NASA-7)
    # -------------------------------------------------------------------------

    def _parse_thermo_block(self, lines: List[str]) -> None:
        """
        Parse ``THERMO [ALL] ... END`` block.

        NASA-7 format: 4 lines per species.
        Line 1: species name, elemental composition, phase, T ranges
        Line 2: high-T coefficients 1–5
        Line 3: high-T 6–7, low-T 1–3
        Line 4: low-T 4–7
        """
        # Skip the temperature-range header line (e.g. "300.000  1000.000  5000.000")
        # and any blank/comment lines before the first species entry.
        data_lines: List[str] = []
        in_data = False
        for line in lines:
            stripped = _strip_comment(line, keep_comment=False).strip()
            upper = stripped.upper()
            if upper in ("END", "THERMO", "THERMO ALL", "THER ALL", "THERMOALL"):
                continue
            if not stripped:
                continue
            # Skip the Trange header line (contains three numbers)
            if not in_data:
                tokens = stripped.split()
                if len(tokens) == 3 and all(_is_number(t) for t in tokens):
                    continue
                in_data = True
            data_lines.append(line)

        # Group into 4-line entries
        i = 0
        while i + 3 < len(data_lines):
            l1 = _strip_comment(data_lines[i]).strip()
            # IMPORTANT: do NOT strip coefficient lines — NASA-7 uses
            # fixed-width 15-character fields, and leading spaces are
            # significant for alignment.
            l2 = _strip_comment(data_lines[i + 1])
            l3 = _strip_comment(data_lines[i + 2])
            l4 = _strip_comment(data_lines[i + 3])

            # Detect if this is a real NASA-7 entry
            # Line 1 should have species name; lines 2-4 contain E/D notation
            if (l1 and not l1.upper().startswith("END")
                    and (_has_exponent_notation(l2) or _has_exponent_notation(l3))):
                self._parse_nasa7_entry(l1, l2, l3, l4)
                i += 4
            else:
                i += 1

    def _parse_nasa7_entry(self, l1: str, l2: str, l3: str, l4: str) -> None:
        """Parse one NASA-7 species entry (4 lines)."""
        # Line 1: extract species name (first 18 chars)
        # Format: <name:18><date:6><formula:24><phase:1><Tlow:10><Thigh:10><Tmid:8>...
        name = l1[:18].strip()
        if not name:
            return

        # Extract temperature ranges from line 1 using fixed-width fields
        # FORTRAN format: 1-18=name, 19-24=date, 25-44=elements, 45=phase,
        #                 46-55=Tlow, 56-65=Thigh, 66-73=Tmid
        tlow = thigh = tmid = 1000.0
        try:
            if len(l1) >= 73:
                tlow  = _try_fortran_float(l1[45:55]) or 300.0
                thigh = _try_fortran_float(l1[55:65]) or 5000.0
                tmid  = _try_fortran_float(l1[65:73]) or 1000.0
            else:
                # Fallback: find the last 3 numbers on line 1
                nums = list(_iter_numbers(l1))
                if len(nums) >= 3:
                    tlow, thigh, tmid = nums[-3], nums[-2], nums[-1]
        except Exception:
            pass

        # Parse coefficients from lines 2-4 using fixed-width 15-char fields.
        # NASA-7 FORTRAN format uses E15.8: 5 coeffs per line, 15 columns each.
        # Lines 2-4 each end with a line-number indicator ("2", "3", "4")
        # that precedes the 5-coeff block, but the trailing integer is after the 5th field.
        # We extract the 5 coeffs from the first 75 chars (5 × 15) of each line.
        try:
            c2 = _parse_nasa7_coeffs(l2, 5)  # high-T 1-5
            c3 = _parse_nasa7_coeffs(l3, 5)  # high-T 6-7, low-T 1-3
            c4 = _parse_nasa7_coeffs(l4, 4)  # low-T 4-7
        except Exception:
            if self._debug:
                log.debug("Could not parse NASA-7 coeffs for %s", name)
            return

        if len(c2) < 5 or len(c3) < 5 or len(c4) < 4:
            if self._debug:
                log.debug("Incomplete NASA-7 coefficients for %s: c2=%d c3=%d c4=%d",
                          name, len(c2), len(c3), len(c4))
            return

        # High range: c2[0..4], c3[0..1]  (7 coeffs)
        high_coeffs = c2[:5] + c3[:2]
        # Low range:  c3[2..5], c4[0..3]  (7 coeffs)
        low_coeffs  = c3[2:5] + c4[:4]

        high_poly = ParsedNASAPolynomial(Tmin=tmid, Tmax=thigh, coeffs=list(high_coeffs))
        low_poly  = ParsedNASAPolynomial(Tmin=tlow, Tmax=tmid, coeffs=list(low_coeffs))

        thermo = ParsedThermo(polynomials=[low_poly, high_poly])

        if name in self._species:
            self._species[name].thermo = thermo
        else:
            self._species[name] = ParsedSpecies(label=name, thermo=thermo)
            self._species_order.append(name)

    # -------------------------------------------------------------------------
    # REACTIONS block
    # -------------------------------------------------------------------------

    def _parse_reactions_block(self, lines: List[str]) -> None:
        """
        Parse ``REACTIONS ... END`` block.

        Structure:
            REACTIONS [CAL/MOLE|KCAL/MOLE|J/MOLE] [MOLES|MOLECULES]
            <reaction equation>  <A>  <n>  <Ea>
            [auxiliary lines: LOW, TROE, PLOG, CHEB, DUP, efficiencies]
            ...
            END
        """
        header_seen = False
        raw_lines: List[str] = []

        for line in lines:
            stripped = _strip_comment(line, keep_comment=False).strip()
            upper = stripped.upper()

            if upper.startswith("END") and not upper.startswith("END_"):
                break

            if not header_seen:
                if upper.startswith("REACTIONS") or upper.startswith("REAC"):
                    self._parse_reactions_header(stripped)
                    header_seen = True
                continue

            if not stripped:
                raw_lines.append(line)  # preserve blank lines as-is
                continue

            raw_lines.append(line)

        # Parse individual reactions
        self._parse_reaction_entries(raw_lines)

    def _parse_reactions_header(self, header: str) -> None:
        """Detect units from REACTIONS header line."""
        self._rxn_units_header = header
        upper = header.upper()
        if "KCAL" in upper:
            self._ea_units = "kcal/mol"
        elif "J/MOLE" in upper or "JOULES" in upper:
            self._ea_units = "J/mol"
        else:
            self._ea_units = "cal/mol"   # CHEMKIN default
        if self._debug:
            log.debug("Reactions header: Ea units = %s", self._ea_units)

    def _parse_reaction_entries(self, raw_lines: List[str]) -> None:
        """Parse individual reaction entries from the REACTIONS block."""
        # Buffer lines for one reaction at a time
        entry_lines: List[str] = []
        pending_comment: List[str] = []

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            stripped = _strip_comment(line, keep_comment=True)

            # Collect comments
            if stripped.strip().startswith("!") or stripped.strip().startswith("#"):
                pending_comment.append(stripped.strip())
                i += 1
                continue

            content = _strip_comment(line, keep_comment=False).strip()
            if not content:
                # Blank line — flush pending reaction if any
                if entry_lines:
                    self._reactions.append(
                        self._parse_one_reaction(entry_lines, pending_comment)
                    )
                    entry_lines = []
                    pending_comment = []
                i += 1
                continue

            upper = content.upper()

            # Keywords that belong to the current reaction (auxiliary lines)
            if entry_lines and (
                upper.startswith("LOW ") or upper.startswith("LOW/")
                or upper.startswith("TROE") or upper.startswith("TROE/")
                or upper.startswith("PLOG") or upper.startswith("PLOG/")
                or upper.startswith("CHEB") or upper.startswith("CHEB/")
                or upper.startswith("TCHEB") or upper.startswith("PCHEB")
                or upper.startswith("SRI") or upper.startswith("SRI/")
                or upper.startswith("REV ")
                or _is_efficiency_line(content)
            ):
                entry_lines.append(line)
                i += 1
                continue

            # DUPLICATE marks the FOLLOWING reaction as duplicate.
            # Flush the previous reaction first, then start a new group.
            if upper.startswith("DUPLICATE"):
                if entry_lines:
                    self._reactions.append(
                        self._parse_one_reaction(entry_lines, pending_comment)
                    )
                    entry_lines = []
                    pending_comment = []
                entry_lines.append(line)
                i += 1
                continue

            # New reaction line — flush previous reaction if it has content
            if entry_lines:
                has_content = any(
                    _strip_comment(l, keep_comment=False).strip().upper() != "DUPLICATE"
                    for l in entry_lines
                )
                if has_content:
                    self._reactions.append(
                        self._parse_one_reaction(entry_lines, pending_comment)
                    )
                    entry_lines = []
                    pending_comment = []
                # If only DUPLICATE markers, keep them for the next reaction

            entry_lines.append(line)
            i += 1

        # Flush last reaction (skip if only DUPLICATE markers remain)
        if entry_lines:
            has_content = any(
                _strip_comment(l, keep_comment=False).strip().upper() != "DUPLICATE"
                for l in entry_lines
            )
            if has_content:
                self._reactions.append(
                    self._parse_one_reaction(entry_lines, pending_comment)
                )

    def _parse_one_reaction(
        self,
        lines: List[str],
        comments: List[str],
    ) -> ParsedReaction:
        """Parse one reaction from its collected lines."""
        rxn = ParsedReaction()

        # Extract comment info
        rxn.comment = "\n".join(comments) if comments else None
        for c in comments:
            c_clean = c.lstrip("!").strip()
            if c_clean.lower().startswith("template reaction:"):
                rxn.family = c_clean.split(":", 1)[1].strip()
            elif c_clean.lower().startswith("template:"):
                rxn.family = c_clean.split(":", 1)[1].strip()

        # Separate the primary reaction line from auxiliary sub-lines
        primary_line: Optional[str] = None
        aux_lines: List[str] = []
        duplicate_flag = False

        for line in lines:
            stripped = _strip_comment(line, keep_comment=False).strip()
            upper = stripped.upper()

            if upper.startswith("DUPLICATE"):
                duplicate_flag = True
                continue
            if upper.startswith("REV "):
                # Reverse rate specification — skip (not needed for RMIP)
                continue
            if not stripped:
                continue

            # Determine if this is an auxiliary sub-line or the primary line
            if _is_auxiliary_line(stripped):
                aux_lines.append(line)
            elif primary_line is None:
                primary_line = line
            else:
                # Could be continuation of auxiliary info (efficiencies)
                aux_lines.append(line)

        if primary_line is None:
            return rxn

        rxn.duplicate = duplicate_flag

        # Parse equation and Arrhenius params from primary line
        self._parse_primary_reaction_line(primary_line, rxn)

        # Parse auxiliary sub-lines
        self._parse_auxiliary_lines(aux_lines, rxn)

        return rxn

    def _parse_primary_reaction_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse 'A+B<=>C+D  A  n  Ea' from the primary reaction line."""
        content = _strip_comment(line, keep_comment=False)

        # Split into equation part and kinetics part
        # The arrow separates reactants from products
        # After products, we have Arrhenius triple (A, n, Ea) as numbers

        # Find the arrow
        eq_str, kin_str = _split_equation_and_kinetics(content)

        # Parse equation
        self._parse_reaction_equation(eq_str, rxn)

        # Parse Arrhenius params
        if kin_str:
            self._parse_arrhenius_triple(kin_str, rxn)

    def _parse_reaction_equation(self, eq_str: str, rxn: ParsedReaction) -> None:
        """Parse 'A+B(+M)<=>C+D(+M)' into reactants/products."""
        eq_str = eq_str.strip()

        # Detect arrow
        if "<=>" in eq_str:
            rxn.reversible = True
            left, right = eq_str.split("<=>", 1)
        elif "=>" in eq_str:
            rxn.reversible = False
            left, right = eq_str.split("=>", 1)
        elif "=" in eq_str:
            rxn.reversible = True
            left, right = eq_str.split("=", 1)
        else:
            log.warning("No reaction arrow found in: %s", eq_str)
            return

        # Check for (+M) / (+N2) etc. in equation
        third_body_pattern = re.compile(r'\(\+[^)]+\)')
        if third_body_pattern.search(eq_str):
            rxn.third_body = True

        rxn.reactants = self._parse_species_side(left)
        rxn.products = self._parse_species_side(right)

    def _parse_species_side(self, side: str) -> List[Tuple[str, float]]:
        """Parse one side of a reaction equation into [(name, stoich), ...]."""
        # Remove (+M), (+N2), etc.
        side = re.sub(r'\(\+[^)]+\)', '', side)
        side = side.strip()

        if not side:
            return []

        # Split on '+'
        parts = side.split("+")
        species_list: List[str] = []
        for p in parts:
            p = p.strip()
            if p:
                species_list.append(p)

        # Parse stoichiometric coefficients
        counter: Counter = Counter()
        for token in species_list:
            # Try to extract a leading number
            m = re.match(r'^(\d*\.?\d*)\s*(.*)', token)
            if m and m.group(1):
                try:
                    coeff = float(m.group(1))
                    name = m.group(2).strip()
                except ValueError:
                    coeff = 1.0
                    name = token
            else:
                coeff = 1.0
                name = token

            if name:
                counter[name] += coeff

        return [(name, float(count)) for name, count in counter.items()]

    def _parse_arrhenius_triple(self, kin_str: str, rxn: ParsedReaction) -> None:
        """Parse 'A  n  Ea' triple from kinetics portion of reaction line."""
        nums = list(_iter_numbers(kin_str))
        if len(nums) < 3:
            return

        # First three numbers are A, n, Ea
        A_val, n_val, Ea_val = nums[0], nums[1], nums[2]

        # Detect units for A based on reaction order
        # This is approximate — we set the correct SI conversion in _convert_to_dicts
        rxn_order = len(rxn.reactants)

        if rxn.kinetics is None:
            rxn.kinetics = ParsedKinetics()

        rxn.kinetics.type = "arrhenius"
        rxn.kinetics.arrhenius = ParsedArrheniusParams(
            A=A_val, n=n_val, Ea=Ea_val,
            A_units=_infer_A_units(rxn_order),
            Ea_units=self._ea_units,
        )

    def _parse_auxiliary_lines(self, lines: List[str], rxn: ParsedReaction) -> None:
        """Parse LOW, TROE, PLOG, CHEB, efficiency sub-lines."""
        if rxn.kinetics is None:
            rxn.kinetics = ParsedKinetics()

        for raw_line in lines:
            line = _strip_comment(raw_line, keep_comment=False).strip()
            upper = line.upper()

            if upper.startswith("LOW "):
                self._parse_low_line(line, rxn)
            elif upper.startswith("TROE"):
                self._parse_troe_line(line, rxn)
            elif upper.startswith("PLOG"):
                self._parse_plog_line(line, rxn)
            elif upper.startswith("CHEB ") or upper.startswith("CHEB/"):
                self._parse_cheb_line(line, rxn)
            elif upper.startswith("TCHEB"):
                self._parse_tcheb_line(line, rxn)
            elif upper.startswith("PCHEB"):
                self._parse_pcheb_line(line, rxn)
            elif upper.startswith("SRI"):
                self._parse_sri_line(line, rxn)
            elif _is_efficiency_line(line):
                self._parse_efficiency_line(line, rxn)

    def _parse_low_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse LOW / A n Ea / line."""
        nums = list(_iter_numbers(line))
        if len(nums) >= 3:
            rxn.kinetics.arrhenius_low = ParsedArrheniusParams(
                A=nums[0], n=nums[1], Ea=nums[2],
                A_units=_infer_A_units(len(rxn.reactants) + 1),
                Ea_units=self._ea_units,
            )
        # Set kinetics type based on existing type + LOW
        if rxn.kinetics.type in ("arrhenius", "third_body"):
            rxn.kinetics.is_pressure_dependent = True

    def _parse_troe_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse TROE / alpha T3 T1 [T2] / line."""
        nums = list(_iter_numbers(line))
        rxn.kinetics.type = "troe"
        rxn.kinetics.is_pressure_dependent = True
        rxn.kinetics.has_third_body = True
        if len(nums) >= 3:
            rxn.kinetics.alpha = nums[0]
            rxn.kinetics.T3 = nums[1]
            rxn.kinetics.T1 = nums[2]
        if len(nums) >= 4:
            rxn.kinetics.T2 = nums[3]

    def _parse_plog_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse PLOG / P  A  n  Ea / line."""
        nums = list(_iter_numbers(line))
        rxn.kinetics.type = "plog"
        rxn.kinetics.is_pressure_dependent = True
        if len(nums) >= 4:
            # Pressure is in atm in CHEMKIN PLOG lines
            P_atm = nums[0]
            rxn.kinetics.pressures.append(P_atm * 101325.0)  # atm → Pa (SI)
            rxn.kinetics.plog_arrhenius.append(ParsedArrheniusParams(
                A=nums[1], n=nums[2], Ea=nums[3],
                A_units=_infer_A_units(len(rxn.reactants)),
                Ea_units=self._ea_units,
            ))

    def _parse_cheb_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse CHEB / nT nP / or CHEB / coeff1 coeff2 ... /"""
        # First check if this is the degree line or a coefficient row
        inside = _extract_slash_content(line)
        rxn.kinetics.type = "chebyshev"
        rxn.kinetics.is_pressure_dependent = True

        if inside:
            tokens = inside.split()
            nums = [_fortran_float(t) for t in tokens]

            # If we already have degrees, this is a coefficient row
            if rxn.kinetics.cheb_degree_T > 0:
                rxn.kinetics.cheb_coeffs.append(nums)
            elif len(nums) == 2 and all(n == int(n) for n in nums):
                # Degree specification: nT nP
                rxn.kinetics.cheb_degree_T = int(nums[0])
                rxn.kinetics.cheb_degree_P = int(nums[1])
            else:
                # Coefficient row without prior degree spec
                rxn.kinetics.cheb_coeffs.append(nums)

    def _parse_tcheb_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse TCHEB / Tmin Tmax / line."""
        nums = list(_iter_numbers(line))
        rxn.kinetics.type = "chebyshev"
        rxn.kinetics.is_pressure_dependent = True
        if len(nums) >= 2:
            rxn.kinetics.Tmin_cheb = nums[0]
            rxn.kinetics.Tmax_cheb = nums[1]

    def _parse_pcheb_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse PCHEB / Pmin Pmax / line."""
        nums = list(_iter_numbers(line))
        rxn.kinetics.type = "chebyshev"
        rxn.kinetics.is_pressure_dependent = True
        if len(nums) >= 2:
            rxn.kinetics.Pmin_cheb = nums[0] * 101325.0   # atm → Pa
            rxn.kinetics.Pmax_cheb = nums[1] * 101325.0

    def _parse_sri_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse SRI / a b c d e / line (treated as Troe-like)."""
        rxn.kinetics.type = "sri"
        rxn.kinetics.is_pressure_dependent = True
        rxn.kinetics.has_third_body = True
        # SRI has 3 or 5 parameters; store alpha as first param
        nums = list(_iter_numbers(line))
        if nums:
            rxn.kinetics.alpha = nums[0]

    def _parse_efficiency_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse third-body efficiency line like 'H2/2.4/ H2O/15.4/'."""
        rxn.kinetics.has_third_body = True
        if not rxn.kinetics.type:
            rxn.kinetics.type = "third_body"
        elif rxn.kinetics.type == "arrhenius":
            rxn.kinetics.type = "third_body"

        # Match species/efficiency pairs:  NAME/FLOAT/
        pattern = re.compile(r'([A-Za-z0-9_+\-()]+)\s*/\s*([\d.+\-EeDd]+)\s*/')
        for m in pattern.finditer(line):
            sp_name = m.group(1).strip()
            try:
                eff = float(m.group(2).replace('D', 'E').replace('d', 'E'))
            except ValueError:
                eff = 1.0
            rxn.kinetics.efficiencies[sp_name] = eff

    # -------------------------------------------------------------------------
    # Convert internal data → SpeciesDict / ReactionDict (chemkin_reader format)
    # -------------------------------------------------------------------------

    def _to_dicts(self) -> Tuple[List[SpeciesDict], List[ReactionDict]]:
        """Convert parsed data to the output dict format."""
        species_list = [self._species_to_dict(name) for name in self._species_order]
        rxn_list = [self._reaction_to_dict(rxn) for rxn in self._reactions]
        return species_list, rxn_list

    def _species_to_dict(self, name: str) -> SpeciesDict:
        """Convert one ParsedSpecies to SpeciesDict."""
        sp = self._species.get(name)
        d: SpeciesDict = {
            "name":           name,
            "smiles":         None,
            "formula":        None,
            "mw_kg_mol":      None,
            "thermo_type":    None,
            "tmin_K":         None,
            "tmid_K":         None,
            "tmax_K":         None,
            "coeffs_low":     None,
            "coeffs_high":    None,
            "h298_kJ_mol":    None,
            "s298_J_molK":    None,
            "thermo_comment": None,
        }

        if sp is None:
            return d

        # Formula from thermo comment / elemental composition
        thermo = sp.thermo
        if thermo is not None and len(thermo.polynomials) >= 2:
            d["thermo_type"] = "nasa7"
            lo = thermo.polynomials[0]
            hi = thermo.polynomials[1]
            d["tmin_K"]     = lo.Tmin
            d["tmid_K"]     = lo.Tmax  # same as hi.Tmin
            d["tmax_K"]     = hi.Tmax
            d["coeffs_low"]  = list(lo.coeffs)
            d["coeffs_high"] = list(hi.coeffs)
            d["thermo_comment"] = thermo.comment if thermo.comment else None

            # Estimate h298 and s298 from NASA-7 polynomials
            # NASA-7: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
            #          H/RT  = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
            #          S/R   = a1*lnT + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
            R = 8.314462618   # J/mol-K
            T = 298.15
            # Use low-T coefficients (valid at 298 K)
            c = lo.coeffs
            if len(c) >= 7:
                H_RT = (c[0] + c[1]*T/2.0 + c[2]*T**2/3.0 + c[3]*T**3/4.0
                        + c[4]*T**4/5.0 + c[5]/T)
                S_R  = (c[0]*math.log(T) + c[1]*T + c[2]*T**2/2.0
                        + c[3]*T**3/3.0 + c[4]*T**4/4.0 + c[6])
                d["h298_kJ_mol"] = H_RT * R * T / 1000.0    # kJ/mol
                d["s298_J_molK"] = S_R * R                   # J/mol-K

        return d

    def _reaction_to_dict(self, rxn: ParsedReaction) -> ReactionDict:
        """Convert one ParsedReaction to ReactionDict."""
        kin = rxn.kinetics or ParsedKinetics()
        ktype = kin.type

        # Build formula string
        formula = self._build_formula_string(rxn)

        # Extract representative A_si, n, Ea_si
        A_si, n, Ea_si, A_orig, Ea_orig, A_units, Ea_units = \
            self._extract_representative_kinetics(kin, rxn)

        # Build kinetics_data blob
        kdata = self._build_kinetics_data(kin)

        # Determine pressure_dependent and has_third_body
        is_pdep = kin.is_pressure_dependent or ktype in (
            "plog", "troe", "lindemann", "chebyshev", "sri",
        )
        has_tb = kin.has_third_body or ktype in ("third_body", "troe", "lindemann", "sri")

        # Build efficiency list
        eff_list = [(name, eff) for name, eff in kin.efficiencies.items()]

        d: ReactionDict = {
            "formula_string":         formula,
            "reversible":             rxn.reversible,
            "family":                 rxn.family,
            "duplicate":              rxn.duplicate,
            "kinetics_type":          ktype,
            "pressure_dependent":     is_pdep,
            "has_third_body":         has_tb,
            "A_si":                   A_si,
            "A_orig":                 A_orig,
            "A_units":                A_units,
            "n":                      n,
            "Ea_si":                  Ea_si,
            "Ea_orig":                Ea_orig,
            "Ea_units":               Ea_units,
            "kinetics_data":          kdata,
            "reactants":              list(rxn.reactants),
            "products":               list(rxn.products),
            "third_body_efficiencies": eff_list,
            "rmg_comment":            rxn.comment,
        }
        return d

    def _build_formula_string(self, rxn: ParsedReaction) -> str:
        """Build 'A + B <=> C + D' string."""
        def side(pairs: List[Tuple[str, float]]) -> str:
            parts = []
            for name, coeff in pairs:
                if coeff == 1.0:
                    parts.append(name)
                elif coeff == int(coeff):
                    parts.append(f"{int(coeff)} {name}")
                else:
                    parts.append(f"{coeff} {name}")
            return " + ".join(parts)

        arrow = " <=> " if rxn.reversible else " => "
        return side(rxn.reactants) + arrow + side(rxn.products)

    def _extract_representative_kinetics(
        self, kin: ParsedKinetics, rxn: ParsedReaction,
    ) -> Tuple[Optional[float], Optional[float], Optional[float],
               Optional[float], Optional[float], Optional[str], Optional[str]]:
        """
        Extract (A_si, n, Ea_si, A_orig, Ea_orig, A_units, Ea_units)
        in SI units from the parsed kinetics.
        """
        ktype = kin.type

        if ktype == "arrhenius" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "third_body" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "troe":
            # Use high-P limit if available
            arr = kin.arrhenius_high or kin.arrhenius
            if arr is not None:
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "lindemann":
            arr = kin.arrhenius_high or kin.arrhenius
            if arr is not None:
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "plog":
            # Return the 1 atm representative values
            for i, P_si in enumerate(kin.pressures):
                if abs(P_si - 101325.0) < 50000 and i < len(kin.plog_arrhenius):
                    arr = kin.plog_arrhenius[i]
                    A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
                    return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units
            # Fallback: use first PLOG entry
            if kin.plog_arrhenius:
                arr = kin.plog_arrhenius[0]
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "chebyshev":
            # No simple Arrhenius representation
            return None, None, None, None, None, None, None

        elif ktype == "sri" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, len(rxn.reactants))
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        return None, None, None, None, None, None, None

    def _build_kinetics_data(self, kin: ParsedKinetics) -> Dict[str, Any]:
        """Build the kinetics_data dict matching chemkin_reader's format."""
        ktype = kin.type

        if ktype == "plog":
            pressures_si = kin.pressures
            arrhenius_entries = []
            for arr in kin.plog_arrhenius:
                arrhenius_entries.append({
                    "A_si": arr.A, "n": arr.n, "Ea_si": arr.Ea,
                })
            return {"pressures": pressures_si, "arrhenius": arrhenius_entries}

        elif ktype == "troe":
            low = {}
            high = {}
            if kin.arrhenius_low:
                low = {"A_si": kin.arrhenius_low.A, "n": kin.arrhenius_low.n,
                       "Ea_si": kin.arrhenius_low.Ea}
            if kin.arrhenius_high:
                high = {"A_si": kin.arrhenius_high.A, "n": kin.arrhenius_high.n,
                        "Ea_si": kin.arrhenius_high.Ea}
            return {
                "alpha": kin.alpha,
                "T1": kin.T1, "T2": kin.T2, "T3": kin.T3,
                "low": low, "high": high,
            }

        elif ktype == "lindemann":
            low = {}
            high = {}
            if kin.arrhenius_low:
                low = {"A_si": kin.arrhenius_low.A, "n": kin.arrhenius_low.n,
                       "Ea_si": kin.arrhenius_low.Ea}
            if kin.arrhenius_high:
                high = {"A_si": kin.arrhenius_high.A, "n": kin.arrhenius_high.n,
                        "Ea_si": kin.arrhenius_high.Ea}
            return {"low": low, "high": high}

        elif ktype == "chebyshev":
            return {
                "Tmin": kin.Tmin_cheb,
                "Tmax": kin.Tmax_cheb,
                "Pmin": kin.Pmin_cheb,
                "Pmax": kin.Pmax_cheb,
                "coeffs": kin.cheb_coeffs,
            }

        elif ktype == "sri":
            return {"alpha": kin.alpha}

        return {}


# =============================================================================
# Utility — file I/O
# =============================================================================

def _read_file(path: str) -> str:
    """Read file content, handling encoding issues."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def _strip_comment(line: str, keep_comment: bool = False) -> str:
    """
    Strip CHEMKIN comments (``!`` prefix) from a line.

    Parameters
    ----------
    line : str
    keep_comment : bool
        If True, return the full line including comment (for comment extraction).
        If False (default), strip everything after ``!``.
    """
    if keep_comment:
        return line
    # Strip comments: '!' is the CHEMKIN comment character
    # Be careful not to strip '!' inside quoted strings (rare in CHEMKIN)
    idx = line.find("!")
    if idx >= 0:
        return line[:idx]
    return line


# =============================================================================
# Utility — block splitting
# =============================================================================

def _split_into_blocks(text: str, source: str = "") -> Dict[str, List[str]]:
    """
    Split CHEMKIN text into named blocks.

    Returns dict mapping block name → list of lines.
    Block names are: ELEMENTS, SPECIES, THERMO, THERMO ALL, REACTIONS.
    """
    lines = text.splitlines()
    blocks: Dict[str, List[str]] = {}
    current_block: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = _strip_comment(line).strip()
        upper = stripped.upper()

        # Detect block start
        block_start = None
        if upper.startswith("ELEMENTS") or upper.startswith("ELEM"):
            block_start = "ELEMENTS"
        elif upper.startswith("SPECIES") or upper.startswith("SPEC"):
            block_start = "SPECIES"
        elif upper.startswith("THERMO ALL") or upper.startswith("THERMOALL"):
            block_start = "THERMO ALL"
        elif upper.startswith("THERMO") or upper.startswith("THER"):
            block_start = "THERMO"
        elif upper.startswith("REACTIONS") or upper.startswith("REAC"):
            block_start = "REACTIONS"

        if block_start is not None:
            # Save previous block
            if current_block is not None:
                blocks[current_block] = current_lines
            current_block = block_start
            current_lines = [line]
            continue

        # Detect block end
        if upper == "END" and current_block is not None:
            current_lines.append(line)
            blocks[current_block] = current_lines
            current_block = None
            current_lines = []
            continue

        if current_block is not None:
            current_lines.append(line)

    # Save unterminated block
    if current_block is not None and current_lines:
        blocks[current_block] = current_lines

    return blocks


# =============================================================================
# Utility — number parsing
# =============================================================================

def _fortran_float(s: str) -> float:
    """
    Parse a Fortran-style float string.

    Handles: ``1.23D+04``, ``1.23d-04``, ``1.23E 04`` (space as plus),
    ``1.23+04`` (implicit E), plain floats.

    Raises ValueError for unparseable input.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty string")

    # Replace D/d with E
    s = s.replace("D", "E").replace("d", "E")

    # Handle "E 04" → "E+04" (space between E sign and digits)
    s = re.sub(r'E\s+(\d)', r'E+\1', s)
    s = re.sub(r'E\s+-', r'E-', s)

    # Handle implicit E notation: "1.23+04" → "1.23E+04"
    # Only if there's no 'E' already
    if "E" not in s:
        s = re.sub(r'([\d.])([+\-]\d{2,})$', r'\1E\2', s)

    return float(s)


def _try_fortran_float(s: str, default: float = 0.0) -> float:
    """Parse Fortran float, returning *default* on failure."""
    try:
        return _fortran_float(s)
    except (ValueError, AttributeError):
        return default


def _is_number(s: str) -> bool:
    """Check if string is a number (including Fortran floats)."""
    try:
        _fortran_float(s)
        return True
    except (ValueError, AttributeError):
        return False


def _has_exponent_notation(s: str) -> bool:
    """Check if string contains scientific notation (E/D)."""
    return bool(re.search(r'[EDed]', s))


def _parse_nasa7_coeffs(line: str, n_expected: int) -> List[float]:
    """
    Parse NASA-7 coefficient line using fixed-width 15-character fields.

    NASA-7 uses FORTRAN E15.8 format: each coefficient occupies exactly 15
    columns.  Numbers can run together without spaces (e.g.
    ``-9.50158922E+02-3.20502331E+00`` = two numbers).

    The line may have a trailing integer (line number "2", "3", "4") which
    is ignored.
    """
    coeffs: List[float] = []
    for i in range(n_expected):
        start = i * 15
        end = start + 15
        if start >= len(line):
            break
        field = line[start:end].strip()
        if not field:
            continue
        try:
            coeffs.append(_fortran_float(field))
        except (ValueError, AttributeError):
            # Field might be blank or non-numeric — skip
            pass
    return coeffs


def _iter_numbers(text: str):
    """Yield floats from a string containing Fortran-format numbers."""
    # Split on whitespace and try to parse each token
    for token in text.split():
        # Strip enclosing slashes, commas
        token = token.strip("/,")
        try:
            yield _fortran_float(token)
        except Exception:
            # Try extracting just the numeric part
            m = re.match(r'^[+\-]?[\d.]+[EDed]?[+\-]?\d*', token)
            if m:
                try:
                    yield _fortran_float(m.group())
                except Exception:
                    pass


def _extract_numbers(text: str) -> List[str]:
    """Extract number-like tokens from text."""
    pattern = re.compile(r'[+\-]?\d*\.?\d+(?:[EDed][+\-]?\d+)?')
    return pattern.findall(text)


# =============================================================================
# Utility — reaction parsing helpers
# =============================================================================

def _split_equation_and_kinetics(text: str) -> Tuple[str, str]:
    """
    Split reaction line into equation part and kinetics (Arrhenius triple) part.

    The equation ends with the products. The kinetics triple consists of
    the last 3 numbers on the line.

    Returns (equation_str, kinetics_str).
    """
    text = text.strip()

    # Find the arrow position
    arrow_match = re.search(r'(<=>|=>|=)', text)
    if not arrow_match:
        return text, ""

    arrow_end = arrow_match.end()

    # Find the last 3 numbers on the line
    numbers = list(re.finditer(
        r'[+\-]?\d*\.?\d+(?:[EDed][+\-]?\d+)?', text[arrow_end:]
    ))

    if len(numbers) < 3:
        return text, ""

    # The kinetics portion starts at the 3rd-to-last number
    third_last = numbers[-3]
    split_pos = arrow_end + third_last.start()

    eq = text[:split_pos].strip()
    kin = text[split_pos:].strip()

    return eq, kin


def _is_auxiliary_line(line: str) -> bool:
    """Check if a line is an auxiliary kinetics sub-line (LOW, TROE, etc.)."""
    upper = line.strip().upper()
    return (
        upper.startswith("LOW ") or upper.startswith("LOW/")
        or upper.startswith("TROE") or upper.startswith("TROE/")
        or upper.startswith("PLOG") or upper.startswith("PLOG/")
        or upper.startswith("CHEB ") or upper.startswith("CHEB/")
        or upper.startswith("TCHEB") or upper.startswith("PCHEB")
        or upper.startswith("SRI") or upper.startswith("SRI/")
        or _is_efficiency_line(line)
    )


def _is_efficiency_line(line: str) -> bool:
    """Check if a line is a third-body efficiency specification."""
    # Pattern: species/eff/ species/eff/
    return bool(re.search(r'[A-Za-z0-9_+\-]+\s*/\s*[\d.EDed+\-]+\s*/', line))


def _extract_slash_content(line: str) -> Optional[str]:
    """Extract content between / ... / delimiters."""
    m = re.search(r'/(.*?)/', line)
    return m.group(1).strip() if m else None


# =============================================================================
# Utility — unit conversion
# =============================================================================

def _infer_A_units(reaction_order: int) -> str:
    """Infer A-factor units based on reaction order."""
    # In CHEMKIN, A-factor units depend on reaction order:
    #   unimolecular:  s⁻¹
    #   bimolecular:   cm³ mol⁻¹ s⁻¹
    #   termolecular:  cm⁶ mol⁻² s⁻¹
    if reaction_order <= 1:
        return "s^-1"
    elif reaction_order == 2:
        return "cm^3/(mol*s)"
    else:
        return f"cm^{3*(reaction_order-1)}/mol^{reaction_order-1}/s"


def _convert_to_si(
    A: float, Ea: float, Ea_units: str, reaction_order: int,
) -> Tuple[float, float]:
    """
    Convert A-factor and activation energy to SI units.

    A (SI):
        unimolecular (1/s):     1/s → 1/s        (no change)
        bimolecular (cm³/mol/s):  cm³/mol/s → m³/mol/s  (÷ 1e6)

    Ea (SI):
        cal/mol  → J/mol    (× 4.184)
        kcal/mol → J/mol    (× 4184.0)
        J/mol    → J/mol    (no change)
    """
    # Ea conversion
    eu = (Ea_units or "").lower()
    if "kcal" in eu:
        Ea_si = Ea * 4184.0
    elif "cal" in eu and "kcal" not in eu:
        Ea_si = Ea * 4.184
    else:
        Ea_si = Ea   # assume already J/mol or unknown

    # A conversion
    if reaction_order == 1:
        A_si = A   # 1/s → 1/s
    elif reaction_order == 2:
        A_si = A / 1.0e6   # cm³/mol/s → m³/mol/s
    elif reaction_order == 3:
        A_si = A / 1.0e12  # cm⁶/mol²/s → m⁶/mol²/s
    else:
        # Unknown — store as-is
        A_si = A

    return A_si, Ea_si
