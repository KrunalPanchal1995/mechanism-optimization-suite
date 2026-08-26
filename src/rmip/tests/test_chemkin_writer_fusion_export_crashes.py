"""
Regression tests for two mean_value_fusion_gui.py Cantera-YAML-export
crashes found stress-testing the launcher (2026-08,
Test_run_23_JULY/mean_value_fusion_gui/SUMMARY.md):

1. A degenerate NASA-7 thermo range (Tmid at or past one end of
   [Tmin, Tmax] -- e.g. a noble gas like AR sourced verbatim from some
   thermo sets, Tmid==Thigh) used to still emit two DIFFERENT coefficient
   sets. Cantera's ck2yaml treats Tmid==Tmax as "one range" and requires
   the two sets to be identical in that case; if they differ, it raises
   InputError -- and *formatting that error message* crashes inside
   Cantera's own ck2yaml with AttributeError: 'NoneType' object has no
   attribute 'reaction_string_suffix' (see chemkin_writer.py::_format_nasa7
   for the full trace). Fixed by writing the same (low) coefficients into
   both blocks when the range is degenerate.

2. A pressure-dependent (troe/falloff/lindemann) reaction with no
   low-pressure-limit data used to silently write only the high-P
   Arrhenius line, producing CHEMKIN text ck2yaml rejects with a similarly
   confusing crash. Fixed by failing fast with a clear repo-side ValueError
   instead (same force= escape hatch as the existing A_missing check).
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

import pytest  # noqa: E402

from rmip.io.chemkin_writer import ChemkinWriter  # noqa: E402


def _writer(force: bool = False) -> ChemkinWriter:
    return ChemkinWriter(":memory:", force=force)


class TestDegenerateNasa7Range:
    def _species(self, tmid: float, tmax: float, low, high) -> dict:
        return {
            "name": "AR",
            "formula": "Ar",
            "tmin_K": 200.0,
            "tmid_K": tmid,
            "tmax_K": tmax,
            "coeffs_low_json": json.dumps(low),
            "coeffs_high_json": json.dumps(high),
        }

    def test_tmid_equals_tmax_forces_identical_coefficient_blocks(self):
        """The exact repro: Tmid==Thigh=6000 with two DIFFERENT source
        coefficient sets -- must no longer emit them as distinct."""
        low = [2.5, 0.0, 0.0, 0.0, 0.0, -745.4, 4.4]
        high = [2.5, 1e-9, -2e-9, 3e-9, -1e-9, -745.4, 4.4]  # deliberately different
        sp = self._species(tmid=6000.0, tmax=6000.0, low=low, high=high)

        lines = _writer()._format_nasa7(sp)
        assert len(lines) == 4

        written_high, written_low = _reconstruct_coeffs(lines)
        assert written_high == pytest.approx(written_low)

    def test_normal_two_range_species_is_unaffected(self):
        """A genuine two-range species (Tmin < Tmid < Tmax) keeps its two
        distinct coefficient sets -- this fix must not collapse real data."""
        low = [3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 3.5]
        high = [2.9, 1.5e-3, -5e-7, 8e-11, -4e-15, -900.0, 6.0]
        sp = self._species(tmid=1000.0, tmax=5000.0, low=low, high=high)

        lines = _writer()._format_nasa7(sp)
        written_high, written_low = _reconstruct_coeffs(lines)
        assert written_high == pytest.approx(high)
        assert written_low == pytest.approx(low)

    def test_tmid_at_or_below_tmin_also_treated_as_degenerate(self):
        low = [2.5, 0.0, 0.0, 0.0, 0.0, -745.4, 4.4]
        high = [9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9]
        sp = self._species(tmid=200.0, tmax=5000.0, low=low, high=high)

        lines = _writer()._format_nasa7(sp)
        written_high, written_low = _reconstruct_coeffs(lines)
        assert written_high == pytest.approx(written_low)


def _parse_coeffs(line: str, n: int):
    """Parse the first n 15-char-wide coefficient fields of a CHEMKIN NASA-7
    line (inverse of chemkin_writer's f"{c:15.8E}" formatting). Coefficients
    always come first, before the trailing card-number marker, whose width
    differs per line (5 chars on lines 2/3, 20 on line 4) -- so this only
    ever reads the first n*15 characters and never needs to know that width."""
    return [float(line[i:i + 15]) for i in range(0, n * 15, 15)]


def _reconstruct_coeffs(lines):
    """Reconstruct (high[0:7], low[0:7]) exactly as ck2yaml parses them
    (cantera/ck2yaml.py's read_NASA7_entry): high[:5] on line 2,
    high[5:7]+low[:3] on line 3, low[3:7] on line 4."""
    line2 = _parse_coeffs(lines[1], 5)
    line3 = _parse_coeffs(lines[2], 5)
    line4 = _parse_coeffs(lines[3], 4)
    high = line2 + line3[:2]
    low = line3[2:5] + line4
    return high, low


class TestMissingLowPressureLimitData:
    _STOICH = {1: [("A", 1.0, "reactant"), ("B", 1.0, "product")]}
    _EFF: dict = {}

    def _rxn(self, ktype: str, kdata: dict) -> dict:
        return {
            "id": 1,
            "kinetics_type": ktype,
            "A_orig": 1.0e13,
            "n": 0.0,
            "Ea_orig": 0.0,
            "Ea_units": "cal/mol",
            "formula_string": "A<=>B",
            "kinetics_data_json": json.dumps(kdata),
        }

    @pytest.mark.parametrize("ktype", ["troe", "falloff", "lindemann"])
    def test_raises_clear_error_when_low_data_missing(self, ktype):
        rxn = self._rxn(ktype, kdata={})  # no "low" key at all

        with pytest.raises(ValueError, match="no low-pressure-limit"):
            _writer()._format_reaction(rxn, self._STOICH, self._EFF)

    def test_force_true_writes_it_anyway(self):
        """The escape hatch (matching the existing A_missing pattern) --
        force=True writes the high-P-only reaction without a LOW line,
        same as before this fix, for deliberately inspecting broken data."""
        rxn = self._rxn("troe", kdata={})
        lines = _writer(force=True)._format_reaction(rxn, self._STOICH, self._EFF)
        assert not any(line.strip().startswith("LOW") for line in lines)

    def test_present_low_data_writes_normally_and_does_not_raise(self):
        rxn = self._rxn("troe", kdata={"low": {"A_si": 1e10, "n": 0.0, "Ea_si": 0.0}})
        lines = _writer()._format_reaction(rxn, self._STOICH, self._EFF)
        assert any(line.strip().startswith("LOW") for line in lines)
