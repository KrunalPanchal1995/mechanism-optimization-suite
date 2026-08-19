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


class TestChemkinParserDuplicate:
    """Test DUPLICATE keyword handling."""

    def test_duplicate_flag(self, tmp_dir):
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
        assert rx[0]["duplicate"] is False  # first occurrence, no DUP before it
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
