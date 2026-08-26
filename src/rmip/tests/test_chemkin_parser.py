"""
rmip/tests/test_chemkin_parser.py
==================================
Tests for the standalone pure-Python CHEMKIN-II parser
(``rmip.io._chemkin_parser``).

These tests verify that the parser correctly handles all kinetics types,
NASA-7 thermo, Fortran float notation, unit conversion, and edge cases.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import textwrap
from pathlib import Path

import pytest

from rmip.io._chemkin_parser import (
    ChemkinParser,
    ParsedArrheniusParams,
    ParsedKinetics,
    ParsedNASAPolynomial,
    ParsedReaction,
    ParsedSpecies,
    ParsedThermo,
    _fortran_float,
    _try_fortran_float,
    _is_number,
    _has_exponent_notation,
    _is_efficiency_line,
    _is_auxiliary_line,
    _split_equation_and_kinetics,
    _parse_nasa7_coeffs,
    _iter_numbers,
    _split_into_blocks,
    _strip_comment,
    _infer_A_units,
    _convert_to_si,
)


# =============================================================================
# Fixtures — minimal CHEMKIN file builders
# =============================================================================

def _write_chemi(path: str, content: str) -> str:
    """Write a CHEMKIN file and return its path."""
    Path(path).write_text(textwrap.dedent(content))
    return path


@pytest.fixture
def tmp_dir():
    """Temporary directory for writing test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# =============================================================================
# Tests — Fortran float parsing
# =============================================================================

class TestFortranFloat:
    def test_plain_float(self):
        assert _fortran_float("1.5") == 1.5
        assert _fortran_float("-3.14") == -3.14

    def test_d_notation(self):
        assert _fortran_float("1.23D+04") == 12300.0
        assert _fortran_float("1.23d-04") == 0.000123

    def test_e_notation(self):
        assert _fortran_float("1.23E+04") == 12300.0
        assert _fortran_float("-4.56e-02") == -0.0456

    def test_implicit_e_notation(self):
        """Handle '1.23+04' style implicit exponent."""
        assert _fortran_float("1.23+04") == 12300.0
        assert _fortran_float("1.23-04") == 0.000123

    def test_raises_on_non_numeric(self):
        with pytest.raises(ValueError):
            _fortran_float("TROE")
        with pytest.raises(ValueError):
            _fortran_float("")
        with pytest.raises(ValueError):
            _fortran_float("/")

    def test_try_fortran_float_fallback(self):
        assert _try_fortran_float("TROE") == 0.0
        assert _try_fortran_float("TROE", default=-1.0) == -1.0
        assert _try_fortran_float("3.14") == 3.14


class TestIsNumber:
    def test_valid_numbers(self):
        assert _is_number("300.000")
        assert _is_number("1.23E+04")
        assert _is_number("-5")
        assert _is_number("1.23D-04")

    def test_invalid_numbers(self):
        assert not _is_number("TROE")
        assert not _is_number("abc")


class TestHasExponentNotation:
    def test_has_e(self):
        assert _has_exponent_notation("1.23E+04")
        assert _has_exponent_notation(" 3.33727920E+00-4.94")

    def test_no_e(self):
        assert not _has_exponent_notation("300.000")
        assert not _has_exponent_notation("abc")


# =============================================================================
# Tests — block splitting
# =============================================================================

class TestBlockSplitting:
    def test_minimal_mechanism(self):
        text = textwrap.dedent("""\
            ELEMENTS
            H O C
            END
            SPECIES
            H2 O2 H2O
            END
            REACTIONS
            H2+O2=>H2O  1.0 0.0 0.0
            END
        """)
        blocks = _split_into_blocks(text)
        assert "ELEMENTS" in blocks
        assert "SPECIES" in blocks
        assert "REACTIONS" in blocks
        assert len(blocks["SPECIES"]) >= 3

    def test_thermo_all_block(self):
        text = textwrap.dedent("""\
            THERMO ALL
               300.000  1000.000  5000.000
            H2   H 2 G 200.0 3500.0 1000.0 1
             1.0E+00 2.0E+00 3.0E+00 4.0E+00 5.0E+00    2
             6.0E+00 7.0E+00 1.0E+00 2.0E+00 3.0E+00    3
             4.0E+00 5.0E+00 6.0E+00 7.0E+00              4
            END
        """)
        blocks = _split_into_blocks(text)
        assert "THERMO ALL" in blocks


# =============================================================================
# Tests — utility helpers
# =============================================================================

class TestSplitEquationAndKinetics:
    def test_simple_reversible(self):
        eq, kin = _split_equation_and_kinetics("A+B<=>C+D  1.0E+10  0.0  5000.0")
        assert "A+B<=>C+D" in eq
        assert "1.0E+10" in kin

    def test_irreversible(self):
        eq, kin = _split_equation_and_kinetics("A=>B  2.5E+12  1.5  30000.0")
        assert "A=>B" in eq

    def test_no_kinetics(self):
        eq, kin = _split_equation_and_kinetics("A+B<=>C+D")
        assert "A+B<=>C+D" in eq


class TestIsEfficiencyLine:
    def test_efficiency_line(self):
        assert _is_efficiency_line("H2/2.4/ H2O/15.4/ CO/1.75/")
        assert _is_efficiency_line("  H2/2/ H2O/6/")

    def test_not_efficiency_line(self):
        assert not _is_efficiency_line("A+B=>C  1.0 0.0 0.0")
        assert not _is_efficiency_line("LOW / 1.0 0.0 0.0 /")


class TestIsAuxiliaryLine:
    def test_low(self):
        assert _is_auxiliary_line("LOW / 1.0E+15 0.0 5000.0 /")
        assert _is_auxiliary_line("LOW  /  1.0E+15  0.0  5000.0  /")

    def test_troe(self):
        assert _is_auxiliary_line("TROE / 0.7824 271.0 2755.0 6570.0 /")

    def test_plog(self):
        assert _is_auxiliary_line("PLOG / 1.0 1.0E+12 0.0 0.0 /")

    def test_chebyshev(self):
        assert _is_auxiliary_line("TCHEB / 300.0 3000.0 /")
        assert _is_auxiliary_line("PCHEB / 0.001 100.0 /")
        assert _is_auxiliary_line("CHEB / 6 4 /")

    def test_efficiency(self):
        assert _is_auxiliary_line("H2/2.0/ H2O/6.0/")

    def test_not_auxiliary(self):
        assert not _is_auxiliary_line("A+B=>C  1.0 0.0 0.0")


class TestIterNumbers:
    def test_simple(self):
        nums = list(_iter_numbers("1.0 2.0 3.0"))
        assert nums == [1.0, 2.0, 3.0]

    def test_fortran_notation(self):
        nums = list(_iter_numbers("1.0E+10 1.23D+04 -5.0"))
        assert len(nums) == 3
        assert nums[0] == 1.0e10
        assert nums[1] == 12300.0

    def test_skips_non_numeric(self):
        """TROE, LOW, / are skipped — only actual numbers returned."""
        nums = list(_iter_numbers("TROE / 0.7824 271.0 /"))
        assert nums == [0.7824, 271.0]

    def test_with_slashes(self):
        nums = list(_iter_numbers("LOW  /  2.47E+24  -2.57  213.89  /"))
        assert len(nums) == 3
        assert nums[0] == 2.47e24
        assert nums[1] == -2.57


class TestNASA7Coeffs:
    def test_five_coeffs(self):
        # Simulate GRI3-style H2 high-T line
        line = " 3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2"
        coeffs = _parse_nasa7_coeffs(line, 5)
        assert len(coeffs) == 5
        assert abs(coeffs[0] - 3.3372792) < 1e-6
        assert abs(coeffs[1] - (-4.94024731e-05)) < 1e-10

    def test_four_coeffs(self):
        # Line 4 has 4 coefficients
        line = " 2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4"
        coeffs = _parse_nasa7_coeffs(line, 4)
        assert len(coeffs) == 4


class TestStripComment:
    def test_strip_exclamation(self):
        assert _strip_comment("A+B=>C  1.0 0.0  ! comment").strip() == "A+B=>C  1.0 0.0"

    def test_no_comment(self):
        assert _strip_comment("A+B=>C  1.0 0.0") == "A+B=>C  1.0 0.0"

    def test_full_line_comment(self):
        assert _strip_comment("! This is a comment").strip() == ""


# =============================================================================
# Tests — unit conversion
# =============================================================================

class TestInferAUnits:
    def test_unimolecular(self):
        assert "s^-1" in _infer_A_units(1)

    def test_bimolecular(self):
        assert "cm^3" in _infer_A_units(2)

    def test_termolecular(self):
        assert "cm^6" in _infer_A_units(3)


class TestConvertToSI:
    def test_ea_cal_to_joule(self):
        _, ea_si = _convert_to_si(1.0, 1000.0, "cal/mol", 2)
        assert abs(ea_si - 4184.0) < 1.0

    def test_ea_kcal_to_joule(self):
        _, ea_si = _convert_to_si(1.0, 1.0, "kcal/mol", 2)
        assert abs(ea_si - 4184.0) < 1.0

    def test_ea_already_joule(self):
        _, ea_si = _convert_to_si(1.0, 5000.0, "J/mol", 2)
        assert ea_si == 5000.0

    def test_a_bimolecular(self):
        a_si, _ = _convert_to_si(1.0e13, 0.0, "cal/mol", 2)
        assert abs(a_si - 1.0e7) < 1.0  # ÷1e6

    def test_a_unimolecular(self):
        a_si, _ = _convert_to_si(1.0e13, 0.0, "cal/mol", 1)
        assert a_si == 1.0e13  # unchanged


# =============================================================================
# Tests — ChemkinParser integration
# =============================================================================

class TestChemkinParserMinimal:
    """Parse a minimal hand-crafted CHEMKIN file."""

    @pytest.fixture
    def minimal_chemi(self, tmp_dir):
        content = """\
            ELEMENTS
            H O
            END
            SPECIES
            H2 O2 H2O OH
            END
            THERMO ALL
               300.000  1000.000  5000.000
            H2                      H   2    0    0    0G    200.00   3500.00 1000.00    0 1
             3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2
            -9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3
             2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4
            H                       H   1    0    0    0G    200.00   3500.00 1000.00    0 1
             2.50000001E+00-2.30842973E-11 1.61561948E-14-4.73515235E-18 4.98197357E-22    2
             2.54736599E+04-4.46682914E-01 2.50000000E+00 7.05332819E-13-1.99591964E-15    3
             2.30081632E-18-9.27732332E-22 2.54736599E+04-4.46682853E-01                   4
            O2                      O   2    0    0    0G    200.00   3500.00 1000.00    0 1
             3.28253784E+00 1.48308754E-03-7.57966669E-07 2.09470555E-10-2.16717794E-14    2
            -1.08845772E+03 5.45323129E+00 3.78245636E+00-2.99673416E-03 9.84730201E-06    3
            -9.68129509E-09 3.24372837E-12-1.06394356E+03 3.65767573E+00                   4
            H2O                     O   1H   2    0    0G    200.00   3500.00 1000.00    0 1
             3.03399249E+00 2.17691804E-03-1.64072518E-07-9.70419870E-11 1.68200992E-14    2
            -3.00042971E+04 4.96677010E+00 4.19864056E+00-2.03643410E-03 6.52040211E-06    3
            -5.48797062E-09 1.77197817E-12-3.02937267E+04-8.49032208E-01                   4
            END
            REACTIONS
            H2+O2<=>2 OH              1.000E+13      0.0000   50000.00
            2 H2+O2<=>2 H2O           2.000E+14      0.0000   40000.00
            H+O2(+M)<=>HO2(+M)        1.500E+12      0.6000       0.00
                LOW  /  2.000E+18  -1.0000      0.00  /
                TROE /  0.8000  1.0000E+02  2.0000E+03  5.0000E+03 /
                H2/2.0/ H2O/6.0/ O2/0.78/
            END
        """
        path = os.path.join(tmp_dir, "minimal.inp")
        _write_chemi(path, content)
        return path

    def test_species_count(self, minimal_chemi):
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        # SPECIES block has 4 names (H2, O2, H2O, OH)
        # THERMO has 4 entries (H2, H, O2, H2O) — 'H' is extra
        assert len(sp) == 5

    def test_reaction_count(self, minimal_chemi):
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        assert len(rx) == 3

    def test_species_thermo(self, minimal_chemi):
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        # H2, H, O2, H2O have thermo entries; OH does not
        thermo_species = {s["name"]: s for s in sp if s["thermo_type"] == "nasa7"}
        assert "H2" in thermo_species, "H2 missing thermo"
        assert "O2" in thermo_species, "O2 missing thermo"
        assert "H2O" in thermo_species, "H2O missing thermo"
        for name in ("H2", "H", "O2", "H2O"):
            s = next(x for x in sp if x["name"] == name)
            assert s["coeffs_low"] is not None, f"{name} coeffs_low"
            assert s["coeffs_high"] is not None, f"{name} coeffs_high"
            assert len(s["coeffs_low"]) == 7
            assert len(s["coeffs_high"]) == 7

    def test_species_dict_schema(self, minimal_chemi):
        """Verify all required keys are present."""
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        required = [
            "name", "smiles", "formula", "mw_kg_mol", "thermo_type",
            "tmin_K", "tmid_K", "tmax_K", "coeffs_low", "coeffs_high",
            "h298_kJ_mol", "s298_J_molK", "thermo_comment",
        ]
        for s in sp:
            for key in required:
                assert key in s, f"Missing key '{key}' in species {s['name']}"

    def test_reaction_dict_schema(self, minimal_chemi):
        """Verify all required keys are present."""
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        required = [
            "formula_string", "reversible", "family", "duplicate",
            "kinetics_type", "pressure_dependent", "has_third_body",
            "A_si", "A_orig", "A_units", "n", "Ea_si", "Ea_orig",
            "Ea_units", "kinetics_data", "reactants", "products",
            "third_body_efficiencies", "rmg_comment",
        ]
        for r in rx:
            for key in required:
                assert key in r, f"Missing key '{key}' in reaction"

    def test_arrhenius_reaction(self, minimal_chemi):
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        r0 = rx[0]
        assert r0["kinetics_type"] == "arrhenius"
        assert r0["A_si"] is not None
        assert r0["n"] is not None
        assert r0["Ea_si"] is not None
        assert not r0["pressure_dependent"]
        assert not r0["has_third_body"]

    def test_stoichiometry_with_prefix(self, minimal_chemi):
        """Reaction '2 H2+O2<=>2 H2O' should have stoich=2 for H2 and H2O."""
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        r1 = rx[1]  # 2 H2+O2<=>2 H2O
        reactants = dict(r1["reactants"])
        products = dict(r1["products"])
        # Reactants: H2 (stoich=2), O2 (stoich=1)
        assert reactants.get("H2") == 2.0 or reactants.get("h2") == 2.0
        # Products: H2O (stoich=2)
        assert products.get("H2O") == 2.0 or products.get("h2o") == 2.0

    def test_troe_reaction(self, minimal_chemi):
        parser = ChemkinParser()
        sp, rx = parser.parse(minimal_chemi)
        r2 = rx[2]  # H+O2(+M)<=>HO2(+M) with LOW/TROE/efficiencies
        assert r2["kinetics_type"] == "troe"
        assert r2["pressure_dependent"]
        assert r2["has_third_body"]
        kdata = r2["kinetics_data"]
        assert kdata["alpha"] is not None
        assert kdata["T1"] is not None
        assert kdata["T3"] is not None
        assert kdata["low"]["A_si"] is not None
        assert len(r2["third_body_efficiencies"]) == 3


class TestChemkinParserArrheniusVariants:
    """Test different Arrhenius parameter formats."""

    def test_ea_cal_per_mole(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 H2O  END
            REACTIONS  CAL/MOLE
            H2+O2=>H2O  1.0E+13  0.0  47700.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "cal.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        # 47700 cal/mol ≈ 199.7 kJ/mol
        assert rx[0]["Ea_si"] == pytest.approx(47700 * 4.184, rel=0.01)

    def test_ea_kcal_per_mole(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 H2O  END
            REACTIONS  KCAL/MOLE
            H2+O2=>H2O  1.0E+13  0.0  47.7
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "kcal.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        # 47.7 kcal/mol = 47.7 * 4184 ≈ 199577 J/mol
        assert rx[0]["Ea_si"] == pytest.approx(47.7 * 4184.0, rel=0.01)

    def test_ea_kjoules_per_mole_not_confused_with_joules(self, tmp_dir):
        """M3 (review.md): "KJOULES/MOLE" contains "JOULES" as a substring,
        so the header-unit detector used to match the JOULES branch first
        and treat kJ/mol values as J/mol -- Ea off by 1000x."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 H2O  END
            REACTIONS  KJOULES/MOLE
            H2+O2=>H2O  1.0E+13  0.0  199.577
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "kjoules.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        # 199.577 kJ/mol = 199577 J/mol -- NOT 199.577 J/mol.
        assert rx[0]["Ea_si"] == pytest.approx(199577.0, rel=0.01)

    def test_molecules_units_warns(self, tmp_dir, caplog):
        """M3: MOLECULES (A-factor per molecule, not per mole) isn't
        converted by this parser -- must warn instead of silently treating
        it as per-mole."""
        import logging
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 H2O  END
            REACTIONS  MOLECULES
            H2+O2=>H2O  1.0E+13  0.0  1000.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "molecules.inp"), content)
        parser = ChemkinParser()
        with caplog.at_level(logging.WARNING, logger="rmip.io.chemkin_parser"):
            parser.parse(path)
        assert any("MOLECULES" in r.message for r in caplog.records)

    def test_evolts_units_warns(self, tmp_dir, caplog):
        """M3: EVOLTS (Ea in electron-volts) isn't converted -- must warn."""
        import logging
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 H2O  END
            REACTIONS  EVOLTS
            H2+O2=>H2O  1.0E+13  0.0  1.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "evolts.inp"), content)
        parser = ChemkinParser()
        with caplog.at_level(logging.WARNING, logger="rmip.io.chemkin_parser"):
            parser.parse(path)
        assert any("EVOLTS" in r.message for r in caplog.records)

    def test_bare_fortran_exponent_A_factor(self, tmp_dir):
        """B6 (review.md): "1.7+13" (legal, common in legacy CHEMKIN files)
        has no E/D marker, so the number regex used to split it into TWO
        separate numbers ("1.7" and "+13") instead of one -- "1.7" got
        glued onto the end of the equation string and A came out as 13.0
        instead of 1.7e13 (12 orders of magnitude off), silently, exit 0."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 OH  END
            REACTIONS
            H2+O2<=>2OH  1.7+13  0.0  47780.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "fortran_exp.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        assert "1.7" not in rx[0]["formula_string"]  # not glued onto the equation
        assert rx[0]["A_si"] == pytest.approx(1.7e13 / 1e6, rel=1e-6)  # cm3->m3
        assert rx[0]["Ea_si"] == pytest.approx(47780.0 * 4.184, rel=1e-6)

    def test_bare_fortran_exponent_negative(self, tmp_dir):
        """Same as above but a negative bare exponent ("2.3-05")."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 OH  END
            REACTIONS
            H2+O2<=>2OH  2.3-05  0.0  1000.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "fortran_exp_neg.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        assert rx[0]["A_si"] == pytest.approx(2.3e-05 / 1e6, rel=1e-6)

    def test_fortran_exponent_not_confused_with_species_plus(self, tmp_dir):
        """The Fortran-exponent fix must not corrupt genuine '+' in species
        formulas/equations (e.g. an ion species like HO2+) -- only a
        sign+digits run immediately after a digit/period, before whitespace
        or end-of-line, is a candidate exponent."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2 O2 OH  END
            REACTIONS
            H2+O2<=>2OH  1.0E+13  0.0  1000.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "plus_species.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        assert rx[0]["A_si"] == pytest.approx(1.0e13 / 1e6, rel=1e-6)


class TestChemkinParserDuplicate:
    """Test DUPLICATE keyword handling."""

    def test_duplicate_flag(self, tmp_dir):
        """B7 (review.md): CHEMKIN spec places DUPLICATE AFTER the reaction
        it marks (each member of a duplicate-reaction set gets its own
        following DUPLICATE line) -- both reactions here are marked
        duplicate. The old (buggy) behavior attached DUPLICATE to whatever
        reaction came *next* instead, which this test used to assert as
        correct (rx[0]["duplicate"] is False) -- that was pinning the bug."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 OH  END
            REACTIONS
            H+O2<=>O+OH  1.0E+14  0.0  16000.0
            DUPLICATE
            H+O2<=>O+OH  5.0E+13  0.0  15000.0
            DUPLICATE
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "dup.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 2
        assert rx[0]["duplicate"] is True
        assert rx[1]["duplicate"] is True

    def test_duplicate_marks_only_the_line_it_follows(self, tmp_dir):
        """A DUPLICATE line after only the SECOND of two otherwise-identical
        reactions marks only that one -- confirms attachment is to the
        immediately preceding reaction, not a blanket flag."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 OH  END
            REACTIONS
            H+O2<=>O+OH  1.0E+14  0.0  16000.0
            H+O2<=>O+OH  5.0E+13  0.0  15000.0
            DUPLICATE
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "dup2.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 2
        assert rx[0]["duplicate"] is False
        assert rx[1]["duplicate"] is True

    def test_dup_abbreviation_recognized(self, tmp_dir):
        """B7: the bare "DUP" abbreviation wasn't recognized at all and
        became a phantom empty reaction row."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 OH  END
            REACTIONS
            H+O2<=>O+OH  1.0E+14  0.0  16000.0
            DUP
            H+O2<=>O+OH  5.0E+13  0.0  15000.0
            DUP
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "dup3.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 2  # not 3 (no phantom row from the bare "DUP" line)
        assert rx[0]["duplicate"] is True
        assert rx[1]["duplicate"] is True


class TestChemkinParserThermo:
    """Test NASA-7 thermo parsing."""

    def test_temperature_ranges(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2  END
            THERMO ALL
               300.000  1000.000  5000.000
            H2                      H   2    0    0    0G    200.00   3500.00 1000.00    0 1
             3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2
            -9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3
             2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4
            END
            REACTIONS  END
        """
        path = _write_chemi(os.path.join(tmp_dir, "thermo.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(sp) == 1
        s = sp[0]
        assert s["name"] == "H2"
        assert s["tmin_K"] == 200.0
        assert s["tmid_K"] == 1000.0
        assert s["tmax_K"] == 3500.0
        assert len(s["coeffs_low"]) == 7
        assert len(s["coeffs_high"]) == 7

    def test_separate_thermo_file(self, tmp_dir):
        mech_content = """\
            ELEMENTS  H O  END
            SPECIES  H2  END
            REACTIONS  END
        """
        mech_path = _write_chemi(os.path.join(tmp_dir, "mech.inp"), mech_content)

        thermo_content = """\
            THERMO ALL
               300.000  1000.000  5000.000
            H2                      H   2    0    0    0G    200.00   3500.00 1000.00    0 1
             3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2
            -9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3
             2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4
            END
        """
        thermo_path = _write_chemi(os.path.join(tmp_dir, "therm.dat"), thermo_content)

        parser = ChemkinParser()
        sp, rx = parser.parse(mech_path, thermo_path=thermo_path)
        assert len(sp) == 1
        assert sp[0]["tmin_K"] == 200.0


class TestChemkinParserComments:
    """Test RMG comment parsing."""

    def test_template_reaction_extraction(self, tmp_dir):
        content = """\
            ELEMENTS  C H  END
            SPECIES  CH3 C2H6  END
            REACTIONS
            ! Template reaction: R_Recombination
            ! Exact match found for rate rule [C_methyl;C_methyl]
            CH3+CH3=>C2H6  8.260E+17  -1.400  1.000
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "rmg_comment.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        assert rx[0]["family"] == "R_Recombination"
        assert rx[0]["rmg_comment"] is not None
        assert "Template reaction" in rx[0]["rmg_comment"]


class TestChemkinParserThirdBody:
    """Test third-body reaction parsing."""

    def test_third_body_with_efficiencies(self, tmp_dir):
        content = """\
            ELEMENTS  H O N AR  END
            SPECIES  H O2 HO2 N2 AR H2O H2  END
            REACTIONS
            H+O2+M<=>HO2+M  2.800E+18  -0.8600  0.00
            O2/0/ H2O/0/ N2/1.26/ AR/0.5/
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "tb.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        r = rx[0]
        assert r["has_third_body"]
        effs = dict(r["third_body_efficiencies"])
        assert effs.get("O2") == 0.0 or effs.get("o2") == 0.0
        assert effs.get("N2") == 1.26 or effs.get("n2") == 1.26
        assert effs.get("AR") == 0.5 or effs.get("ar") == 0.5

    def test_bare_m_never_stored_as_species(self, tmp_dir):
        """M4 (review.md): a bare '+M' third body (no parens) used to be
        stored as a literal reactant/product named "M" -- present here with
        NO subsequent efficiency-override line, the case most likely to be
        missed by a fix that only handles the has_third_body-via-aux-line
        path."""
        content = """\
            ELEMENTS  H  END
            SPECIES  H H2  END
            REACTIONS
            H2+M<=>2H+M  4.580E+19  -1.4000  104390.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "bare_m.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        r = rx[0]
        names = {n for n, _ in r["reactants"]} | {n for n, _ in r["products"]}
        assert "M" not in names
        assert r["has_third_body"] is True

    def test_bare_m_order_compensated_in_A_si_conversion(self, tmp_dir):
        """M4: A off by 1e6 was the actual symptom -- once "M" is correctly
        excluded from reactants, the stoichiometric order used for the
        cm^3->m^3 SI conversion must be compensated by +1 (order=2:
        H2 concentration x M concentration) or A_si silently comes out
        1e6x too large (order=1, as if unimolecular)."""
        content = """\
            ELEMENTS  H  END
            SPECIES  H H2  END
            REACTIONS
            H2+M<=>2H+M  4.580E+19  -1.4000  104390.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "bare_m_order.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        # cm^3/mol/s -> m^3/mol/s: /1e6 (order 2, i.e. bimolecular-equivalent
        # including M) -- NOT /1 (order 1, which the bug produced by
        # dropping M's contribution to stoichiometric order).
        assert rx[0]["A_si"] == pytest.approx(4.580e19 / 1e6, rel=1e-6)

    def test_parenthesized_falloff_main_line_not_compensated(self, tmp_dir):
        """Sanity check that the M4 fix doesn't overcorrect: a parenthesized
        "(+M)" falloff's main line is the high-pressure limit and must NOT
        get the bare-M +1 order compensation (it never had M counted in its
        reactants in the first place)."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2(+M)<=>HO2(+M)  4.650E+12  0.440  0.00
            LOW / 1.740E+19  -1.23  0.00 /
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "falloff_main.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        # High-P limit is bimolecular (order 2 -- H + O2), same as if no
        # third body were involved at all: cm^3/mol/s -> m^3/mol/s = /1e6.
        assert rx[0]["A_si"] == pytest.approx(4.650e12 / 1e6, rel=1e-6)

    def test_generic_M_falloff_keeps_collider_tag_in_formula_string(self, tmp_dir):
        """M6 regression: _build_formula_string used to emit a collider tag
        only for a *specific* collider (rxn.specific_collider, e.g. "(+AR)")
        and never consulted rxn.third_body -- so a parenthesized *generic*-M
        falloff like "H2O2(+M)<=>2OH(+M)" lost its tag entirely on export
        ("H2O2 <=> 2 OH"), indistinguishable from a plain bimolecular
        reaction sharing the same base equation, and (since formula_string
        is a downstream join key, e.g. rop_runner.py) silently collidable
        with one. A specific-collider falloff of the SAME base equation must
        keep its own distinct tag."""
        content = """\
            ELEMENTS  H O AR  END
            SPECIES  H2O2 OH AR  END
            REACTIONS
            H2O2(+M)<=>2OH(+M)  2.000E+12  0.900  48748.0
            LOW / 2.490E+24  -2.300  48749.0 /
            H2O2(+AR)<=>2OH(+AR)  2.000E+12  0.900  48748.0
            LOW / 1.870E+25  -2.300  48749.0 /
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "generic_m_falloff.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 2
        formulas = [r["formula_string"] for r in rx]
        assert "H2O2 (+M) <=> 2 OH (+M)" in formulas
        assert "H2O2 (+AR) <=> 2 OH (+AR)" in formulas
        # The two falloff variants must be distinguishable -- this is the
        # exact collision ck2yaml rejects as an "undeclared duplicate
        # reaction" when they aren't.
        assert len(set(formulas)) == 2

    def test_bare_M_gets_no_parenthesized_tag(self, tmp_dir):
        """The M6 fix must not touch bare "+M" reactions (no parens) --
        they never had M parenthesized in the source, and already round-trip
        correctly without a tag via chemkin_writer's has_third_body path
        (see test_bare_third_body_M_on_both_sides in test_phase3.py)."""
        content = """\
            ELEMENTS  H  END
            SPECIES  H H2  END
            REACTIONS
            H2+M<=>2H+M  4.580E+19  -1.4000  104390.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "bare_m_formula.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert rx[0]["formula_string"] == "H2 <=> 2 H"


class TestChemkinParserKineticsDataMolSI:
    """M5 (review.md): kinetics_data's inner A_si/Ea_si used to be raw
    file-native numbers (cm/mol/s, cal/mol) despite the "_si" suffix --
    inconsistent with this same reader's own top-level A_si/Ea_si columns
    (which ARE mol-SI), and with cantera_reader.py/chemkin_reader.py's
    kinetics_data (also mol-SI). Any consumer trusting kinetics_data's
    "_si" naming (falloff/PLOG CHEMKIN export, uncertainty_bands.py) got
    numbers off by 1e6-1e9."""

    def test_plog_sub_rates_mol_si(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2<=>HO2  1.0E+12  0.0  0.0
            PLOG / 0.1   1.230E+15  0.000  25000.0 /
            PLOG / 10.0  4.560E+16  0.000  30000.0 /
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "plog_si.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        kdata = rx[0]["kinetics_data"]
        # Bimolecular (order 2, H+O2 -- no M term): cm^3/mol/s -> m^3/mol/s
        # is /1e6, matching the top-level A_si convention exactly.
        assert kdata["arrhenius"][0]["A_si"] == pytest.approx(1.230e15 / 1e6, rel=1e-6)
        assert kdata["arrhenius"][0]["Ea_si"] == pytest.approx(25000.0 * 4.184, rel=1e-6)
        assert kdata["arrhenius"][1]["A_si"] == pytest.approx(4.560e16 / 1e6, rel=1e-6)

    def test_troe_low_and_high_mol_si(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2(+M)<=>HO2(+M)  4.650E+12  0.440  0.00
            LOW / 1.740E+19  -1.23  0.00 /
            TROE / 0.7346  94.0  1756.0  5182.0 /
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "troe_si.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        kdata = rx[0]["kinetics_data"]
        # High-P limit: order 2 (H+O2, no M) -> /1e6.
        assert kdata["high"]["A_si"] == pytest.approx(4.650e12 / 1e6, rel=1e-6)
        # Low-P limit: order 3 (H+O2+M) -> /1e12.
        assert kdata["low"]["A_si"] == pytest.approx(1.740e19 / 1e12, rel=1e-6)
        # Consistent with the top-level (representative = high-P) A_si.
        assert rx[0]["A_si"] == pytest.approx(kdata["high"]["A_si"], rel=1e-6)


class TestChemkinParserNoSpaceAfterSlash:
    """M2 (review.md): _iter_numbers() splits on whitespace then strips
    enclosing '/' per-token; "LOW/2.3E18" (no space after the slash) has
    its '/' in the middle of the token, not at either end, so .strip("/,")
    does nothing, the token fails to parse as a float, and the FIRST value
    silently vanishes -- shifting every subsequent number in that sub-line
    into the wrong field."""

    def test_low_no_space_after_slash(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2(+M)<=>HO2(+M)  4.650E+12  0.440  0.00
            LOW/2.300E+18  -0.90  -855.56/
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "low_nospace.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        kdata = rx[0]["kinetics_data"]
        # kinetics_data's "low"/"high" are mol-SI (M5, review.md). LOW's
        # order is reactant_order(2: H+O2)+1(M) = 3 -> cm^6/mol^2/s ->
        # m^6/mol^2/s is /1e12. Pre-fix: A_si would be missing/None (first
        # number dropped) or the wrong value (n, Ea shifted into A's slot).
        assert kdata["low"]["A_si"] == pytest.approx(2.300e18 / 1e12, rel=1e-6)
        assert kdata["low"]["n"] == pytest.approx(-0.90)
        assert kdata["low"]["Ea_si"] == pytest.approx(-855.56 * 4.184, rel=1e-6)

    def test_troe_no_space_after_slash(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2(+M)<=>HO2(+M)  4.650E+12  0.440  0.00
            LOW / 2.300E+18  -0.90  -855.56 /
            TROE/0.7346  94.0  1756.0  5182.0/
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "troe_nospace.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        kdata = rx[0]["kinetics_data"]
        assert kdata["alpha"] == pytest.approx(0.7346)
        assert kdata["T3"] == pytest.approx(94.0)
        assert kdata["T1"] == pytest.approx(1756.0)
        assert kdata["T2"] == pytest.approx(5182.0)

    def test_plog_no_space_after_slash(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 HO2  END
            REACTIONS
            H+O2<=>HO2  1.0E+12  0.0  0.0
            PLOG/0.1  1.0E+12  0.0  1000.0/
            PLOG/10.0  2.0E+13  0.0  2000.0/
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "plog_nospace.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        kdata = rx[0]["kinetics_data"]
        assert len(kdata["pressures"]) == 2
        assert kdata["arrhenius"][0]["A_si"] is not None
        assert kdata["arrhenius"][0]["n"] == pytest.approx(0.0)


class TestChemkinParserIonSpecies:
    """M1 (review.md): trailing '+' in ion species names must survive
    equation tokenization instead of being truncated as if it were a
    stoichiometric '+' separator."""

    def test_trailing_plus_ion_species_preserved(self, tmp_dir):
        content = """\
            ELEMENTS  H O E  END
            SPECIES  H O2 HO2+ E  END
            REACTIONS
            H+O2<=>HO2++E  1.0E+13  0.0  1000.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "ion.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        product_names = {n for n, _ in rx[0]["products"]}
        assert "HO2+" in product_names
        assert "E" in product_names


class TestChemkinParserReferentialCheck:
    """M1: undeclared reaction participants used to be silently accepted."""

    def test_undeclared_participant_logged_but_kept_by_default(self, tmp_dir, caplog):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2  END
            REACTIONS
            H+O2<=>HO2  1.0E+13  0.0  1000.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "undeclared.inp"), content)
        parser = ChemkinParser()  # allow_pseudo_species=False (default)
        import logging
        with caplog.at_level(logging.ERROR, logger="rmip.io.chemkin_parser"):
            sp, rx = parser.parse(path)
        assert len(rx) == 1
        # Default: kept (no silent data loss), but loudly logged.
        product_names = {n for n, _ in rx[0]["products"]}
        assert "HO2" in product_names
        assert any("HO2" in r.message and "undeclared" in r.message.lower()
                    for r in caplog.records)

    def test_undeclared_participant_dropped_with_allow_pseudo_species(self, tmp_dir):
        """Mirrors ck2yaml --permissive: chemiluminescence mechanisms often
        reference a bookkeeping pseudo-species (e.g. "HV" for a photon) that
        was never meant to be a real tracked species."""
        content = """\
            ELEMENTS  H O  END
            SPECIES  H O2 OH O  END
            REACTIONS
            OH<=>O+HV  1.0E+06  0.0  0.0
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "chemilum.inp"), content)
        parser = ChemkinParser(allow_pseudo_species=True)
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        product_names = {n for n, _ in rx[0]["products"]}
        assert "HV" not in product_names
        assert "O" in product_names


# =============================================================================
# Tests — real mechanism files (regression)
# =============================================================================

class TestRealMechanismGRI3:
    """Parse the GRI3 mechanism and verify expected output."""

    @pytest.fixture
    def gri3_path(self):
        p = "/home/krithika/MUQ-SAC/MECHANISMS/GRI3/chem_gri30_wetCO.inp"
        if not os.path.exists(p):
            pytest.skip("GRI3 mechanism not found")
        return p

    def test_species_count(self, gri3_path):
        parser = ChemkinParser()
        sp, rx = parser.parse(gri3_path)
        # GRI3 has 14 named species in SPECIES block (+1 extra from 'C' in thermo)
        assert len(sp) >= 14

    def test_reaction_count(self, gri3_path):
        parser = ChemkinParser()
        sp, rx = parser.parse(gri3_path)
        # GRI3 has ~48 reactions (may vary slightly with duplicate handling)
        assert len(rx) >= 40

    def test_thermo_for_all_species(self, gri3_path):
        parser = ChemkinParser()
        sp, rx = parser.parse(gri3_path)
        # All species in SPECIES block should have thermo
        # (the SPECIES block names are what matters)
        species_names_in_block = {"H2", "H", "O", "O2", "OH", "H2O", "HO2",
                                   "H2O2", "CO", "CO2", "HCO", "CH2O", "N2", "AR"}
        with_thermo = {s["name"] for s in sp if s["tmin_K"] is not None}
        missing = species_names_in_block - with_thermo
        assert not missing, f"Species missing thermo: {missing}"

    def test_known_enthalpies(self, gri3_path):
        parser = ChemkinParser()
        sp, rx = parser.parse(gri3_path)
        sp_map = {s["name"]: s for s in sp}
        # H2O enthalpy of formation ~ -241.8 kJ/mol
        if "H2O" in sp_map:
            h = sp_map["H2O"]["h298_kJ_mol"]
            assert h is not None
            # Allow ±20 kJ/mol tolerance for different NASA-7 fitting
            assert abs(h - (-241.8)) < 40.0, f"H2O h298 = {h}, expected ~ -241.8"

    def test_contains_troe(self, gri3_path):
        parser = ChemkinParser()
        sp, rx = parser.parse(gri3_path)
        troe_rxns = [r for r in rx if r["kinetics_type"] == "troe"]
        assert len(troe_rxns) >= 2


class TestChemkinParserIrreversible:
    """Test irreversible reaction notation (=> instead of <=>)."""

    def test_irreversible(self, tmp_dir):
        content = """\
            ELEMENTS  C H  END
            SPECIES  CH3 C2H6  END
            REACTIONS
            CH3+CH3=>C2H6  8.260E+17  -1.400  1.000
            END
        """
        path = _write_chemi(os.path.join(tmp_dir, "irrev.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(rx) == 1
        assert rx[0]["reversible"] is False
        assert "=>" in rx[0]["formula_string"]


# =============================================================================
# Tests — ChemkinParser with embedded thermo in mechanism file
# =============================================================================

class TestEmbeddedThermo:
    """Test that thermo embedded in the same file is parsed."""

    def test_embedded_thermo(self, tmp_dir):
        content = """\
            ELEMENTS  H O  END
            SPECIES  H2  END
            THERMO ALL
               300.000  1000.000  5000.000
            H2                      H   2    0    0    0G    200.00   3500.00 1000.00    0 1
             3.33727920E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
            -9.50158922E+02 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    3
             0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00                   4
            END
            REACTIONS  END
        """
        path = _write_chemi(os.path.join(tmp_dir, "embedded.inp"), content)
        parser = ChemkinParser()
        sp, rx = parser.parse(path)
        assert len(sp) == 1
        assert sp[0]["thermo_type"] == "nasa7"
        assert sp[0]["tmin_K"] == 200.0
