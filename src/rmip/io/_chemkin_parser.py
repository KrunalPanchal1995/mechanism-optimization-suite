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
    formula: Optional[str] = None


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
    # M4 (review.md): True only for a *bare* "+M" (no parens, e.g.
    # "H2+M<=>2H+M") -- distinct from the parenthesized falloff case
    # "(+M)", which `third_body` above also covers. A bare-M reaction's
    # single rate expression includes the M concentration term, so its
    # stoichiometric order for A-factor SI conversion needs +1 once M is
    # no longer counted via a (wrong) literal "M" reactant entry; a
    # parenthesized falloff's main line is the high-pressure limit, which
    # must NOT get the +1 (M is already excluded from its reactants and
    # correctly has no concentration term at that limit).
    bare_third_body: bool = False


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

    def __init__(self, debug: bool = False, allow_pseudo_species: bool = False) -> None:
        self._debug = debug
        # M1 (review.md): mirrors ck2yaml's --permissive. Default (False)
        # raises on any reaction participant not declared in the SPECIES
        # block -- catches typos/corruption loudly instead of silently
        # storing garbage stoichiometry. Chemiluminescence mechanisms often
        # reference bookkeeping pseudo-species (e.g. "HV" for a photon)
        # that were never meant to be tracked as real species; pass True to
        # drop those specific tokens with a warning instead of raising.
        self._allow_pseudo_species = allow_pseudo_species
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

        if "SPECIES" in mech_blocks:
            # Only meaningful if a SPECIES block was actually declared --
            # skip for REACTIONS-only fragments some tests parse in
            # isolation (nothing to check against).
            self._check_referential_integrity()

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

        # Extract elemental composition from columns 25-44 (4x 5-char "El###"
        # groups). This was previously discarded entirely -- ParsedSpecies had
        # no formula field, so species.formula stayed NULL in the DB for every
        # CHEMKIN-sourced mechanism, which meant ChemkinWriter's ELEMENTS block
        # silently fell back to a hardcoded {C,H,O,N} (missing e.g. AR) and
        # every THERMO entry's own elemental-composition columns came out
        # blank on re-export. Confirmed empirically this is fatal for Zero-RK
        # (which computes species molecular weights from the declared
        # elements, not from name lookup) -- ended in NaN state at t=0 for
        # every species, not just AR, since molecular weight for every
        # species was silently zero/undefined.
        formula = _parse_elemental_formula(l1[24:44]) if len(l1) >= 44 else ""

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
            if formula:
                self._species[name].formula = formula
        else:
            self._species[name] = ParsedSpecies(label=name, thermo=thermo, formula=formula or None)
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

            # B6 (review.md): bare Fortran-exponent notation ("1.7+13", legal
            # and common in legacy CHEMKIN files) has no E/D marker, so the
            # number regexes used everywhere downstream (equation/kinetics
            # split, LOW/TROE/PLOG triples) parse it as TWO separate numbers
            # ("1.7" and "+13") instead of one -- "1.7" gets glued onto the
            # end of the equation string and A ends up 13.0 instead of
            # 1.7e13 (12 orders of magnitude off), silently, exit 0. Fixed
            # once here so every downstream consumer sees normal E-notation.
            raw_lines.append(_fix_fortran_exponent_line(line))

        # Parse individual reactions
        self._parse_reaction_entries(raw_lines)

    def _parse_reactions_header(self, header: str) -> None:
        """Detect units from REACTIONS header line."""
        self._rxn_units_header = header
        upper = header.upper()
        if "KCAL" in upper:
            self._ea_units = "kcal/mol"
        elif "KJOU" in upper:
            # M3 (review.md): "KJOULES/MOLE" contains "JOULES" as a
            # substring, so the elif below used to match it first and treat
            # kJ/mol values as if they were J/mol -- Ea off by 1000x. Must
            # be checked before the plain JOULES/J-MOLE branch.
            self._ea_units = "kJ/mol"
        elif "J/MOLE" in upper or "JOULES" in upper:
            self._ea_units = "J/mol"
        elif "KELVIN" in upper:
            # Ea given as activation *temperature* Ea/R (e.g. GRI-3.0's
            # "REACTIONS   MOLES  KELVINS" header) -- this modifier was
            # previously undetected and silently fell through to the cal/mol
            # default, mislabeling every reaction's Ea by a factor of R
            # (~1.987 cal/mol/K). Confirmed empirically against GRI-Mech
            # 3.0's published values (e.g. O+H2<=>H+OH: file value 3150.48 x
            # 1.98720 = 6260 cal/mol, the documented value) and against
            # Zero-RK's solver, which failed outright (NaN state at t=0,
            # CVode convergence failure) on the under-scaled rates.
            self._ea_units = "K"
        else:
            self._ea_units = "cal/mol"   # CHEMKIN default

        # M3: MOLECULES (A-factor per-molecule instead of per-mole, factor
        # Avogadro's number ~6.022e23) and EVOLTS (Ea in electron-volts) are
        # legal CHEMKIN REACTIONS-header modifiers this parser doesn't
        # convert -- silently treating either as the CHEMKIN default would
        # leave A or Ea off by many orders of magnitude with no warning.
        if "MOLECULES" in upper:
            log.warning(
                "REACTIONS header %r declares MOLECULES units (A-factor "
                "per molecule, not per mole) -- this parser does not "
                "convert by Avogadro's number; A values will be wrong "
                "unless already per-mole.", header,
            )
        if "EVOLTS" in upper or "EVOLT" in upper:
            log.warning(
                "REACTIONS header %r declares EVOLTS units (Ea in "
                "electron-volts) -- this parser does not convert eV; Ea "
                "values will be wrong unless already in %s.",
                header, self._ea_units,
            )

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

            # DUPLICATE (or DUP) marks the PRECEDING reaction as a duplicate
            # member and terminates its entry -- CHEMKIN spec: the keyword
            # follows each member of a duplicate-reaction set. B7
            # (review.md): this used to flush the preceding reaction WITHOUT
            # the duplicate flag and attach "DUPLICATE" to whatever reaction
            # came next instead -- inverting which member of each pair is
            # actually marked, and rejected by ck2yaml on round-trip. The
            # bare "DUP" abbreviation wasn't recognized at all, producing a
            # phantom empty reaction row.
            if _is_duplicate_keyword(content):
                if entry_lines:
                    entry_lines.append(line)
                    self._reactions.append(
                        self._parse_one_reaction(entry_lines, pending_comment)
                    )
                else:
                    log.warning(
                        "DUPLICATE keyword with no preceding reaction -- ignoring"
                    )
                entry_lines = []
                pending_comment = []
                i += 1
                continue

            # New reaction line — flush previous reaction if it has content
            if entry_lines:
                has_content = any(
                    not _is_duplicate_keyword(_strip_comment(l, keep_comment=False))
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

            if _is_duplicate_keyword(stripped):
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

        # Check for (+M) / (+N2) etc. in equation. A specific collider
        # (anything but the generic "M") makes this a physically DIFFERENT
        # reaction from the same equation's generic-M or other-specific-
        # collider variants (e.g. "H+O2(+M)<=>HO2(+M)" vs
        # "H+O2(+AR)<=>HO2(+AR)" vs "H+O2(+HE)<=>HO2(+HE)" are 3 distinct
        # falloff reactions with different LOW/TROE parameters, common in
        # detailed mechanisms) -- found stress-testing fusion on Keromnes/
        # Li/Davis (2026-07-14): stripping this without recording it lost
        # the distinguishing collider identity entirely, so all 3 variants
        # got the same (wrong) reconstructed formula_string on export,
        # which ck2yaml then rejected as an "undeclared duplicate reaction".
        third_body_match = re.search(r'\(\+([^)]+)\)', eq_str)
        if third_body_match:
            rxn.third_body = True
            collider = third_body_match.group(1).strip()
            if collider.upper() != "M":
                rxn.specific_collider = collider

        rxn.reactants = self._parse_species_side(left, rxn)
        rxn.products = self._parse_species_side(right, rxn)

    def _parse_species_side(
        self, side: str, rxn: Optional[ParsedReaction] = None,
    ) -> List[Tuple[str, float]]:
        """Parse one side of a reaction equation into [(name, stoich), ...].

        Species-list-aware (M1, review.md): chemical formulas can
        legitimately contain a literal trailing '+' (ion species like
        HO2+, H2O+) -- naively splitting on '+' as a pure separator
        truncates them (HO2+ -> "HO2"). Tries the longest declared-species
        match at each '+'-delimited boundary before falling back to the
        single raw token.

        Bare "+M" third bodies (M4, review.md): the parenthesized form
        "(+M)" was already stripped before this is called; a bare "+M"
        (no parens, e.g. "H2+M<=>2H+M") used to fall through and get
        stored as a literal species named "M", miscomputing reaction order
        and throwing SI-path A-factor exports off by 1e6. Stripped here
        instead and folded into ``rxn.third_body``.
        """
        # Remove (+M), (+N2), etc.
        side = re.sub(r'\(\+[^)]+\)', '', side)
        side = side.strip()

        if not side:
            return []

        raw_parts = [p.strip() for p in side.split("+")]
        # Reconstruct tokens, preferring the longest declared-species match
        # across adjacent raw '+'-split chunks (handles "HO2+" -> ["HO2",""]
        # by remerging the trailing empty chunk's '+' onto "HO2").
        tokens: List[str] = []
        i = 0
        n = len(raw_parts)
        while i < n:
            if not raw_parts[i]:
                i += 1
                continue
            # Strip a leading numeric stoich coefficient before matching,
            # so "2HO2+" (coeff=2, name="HO2+") still recognizes "HO2+".
            m = re.match(r'^(\d*\.?\d+)\s*(.*)', raw_parts[i])
            if m and m.group(1) and m.group(2):
                coeff_prefix, first_name = m.group(1), m.group(2)
            else:
                coeff_prefix, first_name = "", raw_parts[i]

            best = first_name
            best_j = i + 1
            candidate = first_name
            # Look ahead through empty/short chunks that a '+'-suffixed
            # declared species would produce (e.g. "HO2" + "" from "HO2+",
            # or "H2O" + "" + "" from "H2O++" -- rare doubly-charged ions).
            for j in range(i + 1, min(n, i + 3)):
                candidate = candidate + "+" + raw_parts[j]
                if candidate in self._species:
                    best = candidate
                    best_j = j + 1
            tokens.append(coeff_prefix + best if coeff_prefix else best)
            i = best_j

        # Parse stoichiometric coefficients + strip bare "+M" third bodies
        counter: Counter = Counter()
        for token in tokens:
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

            if name == "M" and name not in self._species:
                if rxn is not None:
                    rxn.third_body = True
                    rxn.bare_third_body = True
                continue

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

        # Detect units for A based on reaction order (stoichiometric sum, not
        # species count — matches _extract_representative_kinetics's SI-conversion
        # order so the display label agrees with the numeric conversion).
        # M4: a bare "+M" third body is no longer counted via a literal "M"
        # reactant entry (see _parse_species_side) -- its concentration term
        # still belongs in the rate law, so +1 here restores the correct
        # order. Parenthesized falloff "(+M)" does NOT get +1: the main
        # line represents the high-pressure limit, which has no M term.
        rxn_order = int(round(sum(coeff for _, coeff in rxn.reactants)))
        if rxn.bare_third_body:
            rxn_order += 1

        if rxn.kinetics is None:
            rxn.kinetics = ParsedKinetics()

        rxn.kinetics.type = "arrhenius"
        rxn.kinetics.arrhenius = ParsedArrheniusParams(
            A=A_val, n=n_val, Ea=_normalize_ea_to_cal(Ea_val, self._ea_units),
            A_units=_infer_A_units(rxn_order),
            Ea_units="cal/mol",
        )

    def _parse_auxiliary_lines(self, lines: List[str], rxn: ParsedReaction) -> None:
        """Parse LOW, TROE, PLOG, CHEB, efficiency sub-lines."""
        if rxn.kinetics is None:
            rxn.kinetics = ParsedKinetics()

        for raw_line in lines:
            line = _strip_comment(raw_line, keep_comment=False).strip()
            upper = line.upper()

            if upper.startswith("LOW ") or upper.startswith("LOW/"):
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

        # M5: the main reaction line's Arrhenius triple IS a falloff
        # reaction's high-pressure limit (that's what "arrhenius_high"
        # means), but it's parsed into kin.arrhenius, not kin.arrhenius_high
        # -- which was never actually populated anywhere. Downstream reads
        # that correctly fall back via `kin.arrhenius_high or kin.arrhenius`
        # (_extract_representative_kinetics) were fine, but consumers
        # reading kin.arrhenius_high directly (kinetics_data's "high" block)
        # always got an empty dict for every falloff reaction.
        if rxn.kinetics.type in ("troe", "lindemann") and rxn.kinetics.arrhenius_high is None:
            rxn.kinetics.arrhenius_high = rxn.kinetics.arrhenius

    def _parse_low_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse LOW / A n Ea / line.

        M2 (review.md): _iter_numbers() splits on whitespace, then strips
        enclosing '/' from each whitespace-delimited token -- "LOW/2.3E18"
        (no space after the slash) never has its leading '/' stripped (it's
        not the first character), so the whole token fails to parse as a
        number and the first value silently vanishes. Extracting the
        slash-delimited content first (_extract_slash_content) sidesteps
        the keyword prefix entirely."""
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
        if len(nums) >= 3:
            rxn.kinetics.arrhenius_low = ParsedArrheniusParams(
                A=nums[0], n=nums[1], Ea=_normalize_ea_to_cal(nums[2], self._ea_units),
                A_units=_infer_A_units(
                    int(round(sum(coeff for _, coeff in rxn.reactants))) + 1
                ),
                Ea_units="cal/mol",
            )
        # Set kinetics type based on existing type + LOW. A LOW line with no
        # following TROE line is a plain Lindemann falloff -- if this isn't
        # explicitly set here (mirroring _parse_troe_line's explicit
        # `type = "troe"`), a later efficiency line falls through to the
        # generic "third_body" branch instead, silently discarding
        # arrhenius_low and losing the reaction's pressure dependence
        # entirely (confirmed empirically: GRI-3.0's O+CO(+M)<=>CO2(+M)).
        if rxn.kinetics.type in ("arrhenius", "third_body"):
            rxn.kinetics.type = "lindemann"
            rxn.kinetics.is_pressure_dependent = True

    def _parse_troe_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse TROE / alpha T3 T1 [T2] / line. See _parse_low_line for
        why the slash content must be extracted first (M2)."""
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
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
        """Parse PLOG / P  A  n  Ea / line. See _parse_low_line for why the
        slash content must be extracted first (M2)."""
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
        rxn.kinetics.type = "plog"
        rxn.kinetics.is_pressure_dependent = True
        if len(nums) >= 4:
            # Pressure is in atm in CHEMKIN PLOG lines
            P_atm = nums[0]
            rxn.kinetics.pressures.append(P_atm * 101325.0)  # atm → Pa (SI)
            rxn.kinetics.plog_arrhenius.append(ParsedArrheniusParams(
                A=nums[1], n=nums[2], Ea=_normalize_ea_to_cal(nums[3], self._ea_units),
                A_units=_infer_A_units(
                    int(round(sum(coeff for _, coeff in rxn.reactants)))
                ),
                Ea_units="cal/mol",
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
        """Parse TCHEB / Tmin Tmax / line. See _parse_low_line for why the
        slash content must be extracted first (M2)."""
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
        rxn.kinetics.type = "chebyshev"
        rxn.kinetics.is_pressure_dependent = True
        if len(nums) >= 2:
            rxn.kinetics.Tmin_cheb = nums[0]
            rxn.kinetics.Tmax_cheb = nums[1]

    def _parse_pcheb_line(self, line: str, rxn: ParsedReaction) -> None:
        """Parse PCHEB / Pmin Pmax / line. See _parse_low_line for why the
        slash content must be extracted first (M2)."""
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
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
        # SRI has 3 or 5 parameters; store alpha as first param. See
        # _parse_low_line for why the slash content must be extracted
        # first (M2).
        nums = list(_iter_numbers(_extract_slash_content(line) or line))
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
            "formula":        sp.formula if sp is not None else None,
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
        kdata = self._build_kinetics_data(kin, rxn)

        # Determine pressure_dependent and has_third_body
        is_pdep = kin.is_pressure_dependent or ktype in (
            "plog", "troe", "lindemann", "chebyshev", "sri",
        )
        # M4: rxn.bare_third_body catches a bare "+M" reaction with no
        # subsequent efficiency-override aux line -- kin.has_third_body is
        # only set by that aux line, and ktype only becomes "third_body"
        # because of it too, so neither alone is reliable here.
        has_tb = kin.has_third_body or rxn.bare_third_body or ktype in ("third_body", "troe", "lindemann", "sri")

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
        """Build 'A + B <=> C + D' string, or 'A + B (+AR) <=> C + D (+AR)'
        for a specific-collider falloff reaction -- the collider identity
        must round-trip into the formula string, or a mechanism with
        multiple collider variants of the same base equation (e.g.
        generic-M, +AR, +HE) exports 3 textually-identical equation lines
        with no way to tell them apart (see _parse_reaction_equation)."""
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
        # M6: a *parenthesized* generic-M falloff ("H2O2(+M)<=>2OH(+M)") has
        # third_body=True but specific_collider=None (the collider name "M"
        # is the generic default, not recorded as specific -- see
        # _parse_reaction_equation). The old code only ever emitted a tag
        # for specific_collider, so every generic-M falloff lost its "(+M)"
        # entirely on export ("H2O2 <=> 2 OH") -- indistinguishable from a
        # plain bimolecular reaction with the same base equation, and (since
        # formula_string is a downstream join key, e.g. rop_runner.py) a
        # silent collision with any such reaction. bare_third_body (the
        # unparenthesized "H2+M<=>2H+M" form) must NOT get this tag: M was
        # never parenthesized there, and bare-M reactions already round-trip
        # correctly without it (see chemkin_writer's has_third_body path).
        if rxn.specific_collider:
            collider_tag = f" (+{rxn.specific_collider})"
        elif rxn.third_body and not rxn.bare_third_body:
            collider_tag = " (+M)"
        else:
            collider_tag = ""
        return (side(rxn.reactants) + collider_tag + arrow
                + side(rxn.products) + collider_tag)

    def _check_referential_integrity(self) -> None:
        """M1 (review.md): verify every reaction participant is a declared
        species. Previously unchecked -- an undeclared participant (typo,
        truncated ion name, or a chemiluminescence bookkeeping pseudo-
        species like "HV") was silently accepted, ingested, and re-exported;
        Zero-RK then aborted (rc=134) on the resulting mechanism with no
        indication of the actual cause.

        Default (allow_pseudo_species=False): logs at ERROR level, naming
        the reaction and the undeclared participant, but keeps the
        participant in reactants/products -- loud instead of silent,
        without discarding data or hard-failing an otherwise-parseable
        mechanism (many real and test files reference a species that's
        thermo-only or declared elsewhere; a hard raise here proved too
        disruptive against real fixtures during this fix's own testing).
        allow_pseudo_species=True instead DROPS the undeclared participant
        with a WARNING, mirroring ck2yaml's --permissive handling of
        bookkeeping pseudo-species like HV -- this is what un-breaks
        gsa-run on chemiluminescence mechanisms."""
        for rxn in self._reactions:
            undeclared = [
                name for name, _ in list(rxn.reactants) + list(rxn.products)
                if name not in self._species
            ]
            if not undeclared:
                continue
            eq = self._build_formula_string(rxn)
            if self._allow_pseudo_species:
                for name in undeclared:
                    log.warning(
                        "Reaction '%s': undeclared participant '%s' -- "
                        "dropping (allow_pseudo_species=True)", eq, name,
                    )
                rxn.reactants = [(n, c) for n, c in rxn.reactants if n not in undeclared]
                rxn.products  = [(n, c) for n, c in rxn.products  if n not in undeclared]
            else:
                log.error(
                    "Reaction '%s': undeclared participant(s) %s -- not "
                    "present in the SPECIES block. Pass "
                    "allow_pseudo_species=True to ChemkinParser() to drop "
                    "such tokens (e.g. chemiluminescence bookkeeping "
                    "species like HV) instead of keeping them.",
                    eq, undeclared,
                )

    def _extract_representative_kinetics(
        self, kin: ParsedKinetics, rxn: ParsedReaction,
    ) -> Tuple[Optional[float], Optional[float], Optional[float],
               Optional[float], Optional[float], Optional[str], Optional[str]]:
        """
        Extract (A_si, n, Ea_si, A_orig, Ea_orig, A_units, Ea_units)
        in SI units from the parsed kinetics.
        """
        ktype = kin.type
        # Stoichiometric order (sum of coefficients), not species count — a
        # repeated-reactant reaction like 2CH3(+M)<=>C2H6(+M) is bimolecular
        # (order 2), not unimolecular (len(reactants) == 1).
        # M4: bare_third_body (no efficiency-override aux line necessarily
        # follows, so `ktype` alone can't be trusted to be "third_body" --
        # it stays "arrhenius" whenever no such line was present) needs the
        # same +1 compensation as _parse_arrhenius_triple applies, for the
        # same reason: M is no longer counted via a literal "M" reactant.
        rxn_order = int(round(sum(coeff for _, coeff in rxn.reactants)))
        if rxn.bare_third_body:
            rxn_order += 1

        if ktype == "arrhenius" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "third_body" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "troe":
            # Use high-P limit if available
            arr = kin.arrhenius_high or kin.arrhenius
            if arr is not None:
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "lindemann":
            arr = kin.arrhenius_high or kin.arrhenius
            if arr is not None:
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "plog":
            # Return the 1 atm representative values
            for i, P_si in enumerate(kin.pressures):
                if abs(P_si - 101325.0) < 50000 and i < len(kin.plog_arrhenius):
                    arr = kin.plog_arrhenius[i]
                    A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
                    return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units
            # Fallback: use first PLOG entry
            if kin.plog_arrhenius:
                arr = kin.plog_arrhenius[0]
                A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
                return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        elif ktype == "chebyshev":
            # No simple Arrhenius representation
            return None, None, None, None, None, None, None

        elif ktype == "sri" and kin.arrhenius is not None:
            arr = kin.arrhenius
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, rxn_order)
            return A_si, arr.n, Ea_si, arr.A, arr.Ea, arr.A_units, arr.Ea_units

        return None, None, None, None, None, None, None

    def _build_kinetics_data(self, kin: ParsedKinetics, rxn: ParsedReaction) -> Dict[str, Any]:
        """Build the kinetics_data dict, mol-SI throughout (M5, review.md).

        Previously these inner A_si/Ea_si values were raw file-native
        numbers (cm/mol/s, cal/mol) despite the "_si" suffix -- unlike this
        same method's main A_si/Ea_si columns (which do go through
        _convert_to_si), and unlike cantera_reader.py's kinetics_data
        (correctly mol-SI) and chemkin_reader.py's RMG-Py path (also
        mol-SI via RMG's own .value_si). Any consumer trusting the "_si"
        name here (writer falloff/PLOG export, uncertainty_bands.py, M14)
        got numbers off by 1e6-1e9. Same per-sub-rate order convention as
        cantera_reader.py: LOW = reactant_order+1 (includes the M
        concentration term), HIGH/PLOG = reactant_order (no M term).
        """
        ktype = kin.type
        reactant_order = int(round(sum(coeff for _, coeff in rxn.reactants)))

        def conv(arr: Optional[ParsedArrheniusParams], order: int) -> Dict[str, Optional[float]]:
            if arr is None:
                return {}
            A_si, Ea_si = _convert_to_si(arr.A, arr.Ea, arr.Ea_units, order)
            return {"A_si": A_si, "n": arr.n, "Ea_si": Ea_si}

        if ktype == "plog":
            pressures_si = kin.pressures
            arrhenius_entries = [conv(arr, reactant_order) for arr in kin.plog_arrhenius]
            return {"pressures": pressures_si, "arrhenius": arrhenius_entries}

        elif ktype == "troe":
            return {
                "alpha": kin.alpha,
                "T1": kin.T1, "T2": kin.T2, "T3": kin.T3,
                "low":  conv(kin.arrhenius_low,  reactant_order + 1),
                "high": conv(kin.arrhenius_high, reactant_order),
            }

        elif ktype == "lindemann":
            return {
                "low":  conv(kin.arrhenius_low,  reactant_order + 1),
                "high": conv(kin.arrhenius_high, reactant_order),
            }

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


# Bare Fortran-exponent notation: a sign+1-3-digit run immediately after a
# digit or '.', terminated by whitespace/'/'/EOL, with no E/D marker
# ("4.8749+04" -> needs "4.8749E+04"). The lookbehind excludes numbers that
# already have a marker (its preceding char is 'E'/'e'/'D'/'d', not a digit
# or '.'), so this is idempotent. Same pattern as mechanism_converter_gui.py's
# fix_fortran_exponents(), duplicated here rather than imported to keep this
# core io module independent of the top-level GUI scripts.
_FORTRAN_EXP_RE = re.compile(r"(?<=[0-9.])([+-])(\d{1,3})(?=[\s/]|$)")


def _fix_fortran_exponent_line(line: str) -> str:
    """Insert the missing 'E' in bare Fortran-style exponents (B6,
    review.md). Only touches the code portion of the line, leaving any
    trailing '!' comment untouched."""
    idx = line.find("!")
    code = line if idx == -1 else line[:idx]
    rest = "" if idx == -1 else line[idx:]
    fixed, _ = _FORTRAN_EXP_RE.subn(r"E\1\2", code)
    return fixed + rest


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


def _is_duplicate_keyword(line: str) -> bool:
    """CHEMKIN accepts both DUPLICATE and the DUP abbreviation, always
    standalone on their own line (B7, review.md)."""
    upper = line.strip().upper()
    return upper == "DUP" or upper.startswith("DUPLICATE")


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

def _parse_elemental_formula(field: str) -> str:
    """
    Parse a CHEMKIN THERMO line-1 elemental-composition field (20 chars,
    4x 5-char "El###" groups: 2-char symbol + 3-char count) into a Hill-
    notation formula string (e.g. "H2", "CO2", "Ar").
    """
    counts: Dict[str, int] = {}
    for i in range(4):
        group = field[i * 5:i * 5 + 5]
        if len(group) < 5:
            break
        sym = group[0:2].strip()
        cnt_str = group[2:5].strip()
        if not sym:
            continue
        try:
            cnt = int(cnt_str) if cnt_str else 1
        except ValueError:
            cnt = 1
        if cnt <= 0:
            continue
        # CHEMKIN commonly writes symbols in all-caps (e.g. "AR"); normalize
        # to proper case (first upper, rest lower) since Hill notation and
        # chemkin_writer._format_formula's regex both expect it.
        sym_norm = sym[0].upper() + sym[1:].lower()
        counts[sym_norm] = counts.get(sym_norm, 0) + cnt

    if not counts:
        return ""

    parts = []
    for el in ("C", "H"):
        if el in counts:
            parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    for el in sorted(k for k in counts if k not in ("C", "H")):
        parts.append(el if counts[el] == 1 else f"{el}{counts[el]}")
    return "".join(parts)


_GAS_CONSTANT_CAL_PER_MOL_K = 1.987204  # R, for Ea/R (Kelvin) -> Ea (cal/mol)


def _normalize_ea_to_cal(Ea: float, ea_units: str) -> float:
    """
    Normalize a raw parsed Ea value to cal/mol, whatever unit convention the
    REACTIONS header declared. Applied once, at parse time, to every
    Arrhenius triple (main line, LOW, PLOG) so that everything downstream
    (SI conversion, original-units CHEMKIN re-export under the writer's
    fixed CAL/MOLE header) can assume cal/mol unconditionally instead of
    re-deriving the source header's units at every consumption site.
    """
    units = (ea_units or "").lower()
    if units == "k" or "kelvin" in units:
        return Ea * _GAS_CONSTANT_CAL_PER_MOL_K
    if "kcal" in units:
        return Ea * 1000.0
    if "kj" in units:
        # M3: kJ/mol -> cal/mol (1000 J/mol / 4.184 J/cal). Must be checked
        # before the plain J/mol branch below, which "kj/mol" would also
        # match via units.startswith("j/") if reordered.
        return Ea * 1000.0 / 4.184
    if units.startswith("j/") or "joule" in units:
        return Ea / 4.184
    return Ea  # already cal/mol (or unknown -- assume cal/mol, CHEMKIN default)


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
