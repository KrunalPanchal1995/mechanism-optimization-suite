"""
Regression coverage for WORKBENCH_AUDIT_2026-08-10.md defect B13: when
ChemkinWriter._format_reaction()'s "unsupported kinetics type (e.g. SRI) ->
write as plain Arrhenius" branch discards a falloff reaction's pressure-
dependent data, it must also strip that reaction's "(+M)"/"(+species)"
collider notation from the written equation string -- otherwise the output
is syntactically pressure-dependent CHEMKIN with no LOW/TROE/SRI auxiliary
line, which ck2yaml correctly rejects on re-conversion.

Round 3 fixed and manually verified this against a real FFCM1+GRI3
--write-cantera fusion (two real kinetics_type='sri' reactions,
CH3CHO<=>CH4+CO and CH3CHO<=>CH3+HCO), but shipped with no committed unit
test -- this file closes that gap, and extends the check to a
specific-collider tag ("(+AR)", not just the generic "(+M)"), which the
fix's own regex (`\\(\\+[^)]+\\)`) already generalizes to but was never
exercised by a test.
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from rmip.io.chemkin_writer import ChemkinWriter  # noqa: E402

_STOICH = {1: [("A", 1.0, "reactant"), ("M", 1.0, "reactant"),
                ("B", 1.0, "product"), ("M", 1.0, "product")]}
_EFF: dict = {}


def _writer(force: bool = False) -> ChemkinWriter:
    return ChemkinWriter(":memory:", force=force)


def _sri_rxn(formula_string: str) -> dict:
    """A reaction with an unsupported kinetics_type (SRI isn't modeled by
    this writer -- same real shape as the FFCM1 SRI reactions Round 3's
    manual reproduction used), whose stored formula_string still carries
    its original falloff collider notation (as a real parsed SRI/falloff
    reaction's would, before the discard this branch performs)."""
    return {
        "id": 1,
        "kinetics_type": "sri",
        "A_orig": 1.0e13,
        "n": 0.0,
        "Ea_orig": 0.0,
        "Ea_units": "cal/mol",
        "formula_string": formula_string,
        "kinetics_data_json": json.dumps({}),
        "has_third_body": True,
    }


class TestUnsupportedKineticsTypeStripsColliderNotation:
    def test_generic_plus_m_notation_is_stripped(self):
        rxn = _sri_rxn("A+M<=>B+M")  # no (+X) tag at all in the source string
        lines = _writer()._format_reaction(rxn, _STOICH, _EFF)
        rxn_line = lines[-1]
        assert "(+" not in rxn_line

    def test_specific_collider_tag_is_stripped_not_just_generic_plus_m(self):
        """The regex is `\\(\\+[^)]+\\)`, which matches ANY "(+X)" tag, not
        a literal "(+M)" only -- confirms that generality actually holds
        for a specific-collider reaction like FFCM1's real
        CH3CHO<=>CH4+CO (+AR)-style notation."""
        rxn = _sri_rxn("A(+AR)<=>B(+AR)")
        lines = _writer()._format_reaction(rxn, _STOICH, _EFF)
        rxn_line = lines[-1]
        assert "(+AR)" not in rxn_line
        assert "(+" not in rxn_line

    def test_written_line_carries_the_unsupported_kinetics_warning(self):
        rxn = _sri_rxn("A(+AR)<=>B(+AR)")
        lines = _writer()._format_reaction(rxn, _STOICH, _EFF)
        assert any("kinetics_type=sri" in line and "WARNING" in line for line in lines)


class TestSupportedFalloffKeepsColliderNotation:
    """Control case: a troe/lindemann reaction WITH real LOW data must be
    completely unaffected by the B13 fix -- it's handled by an earlier,
    separate branch (`elif ktype in ("troe", "falloff", "lindemann")`) that
    never reaches the stripping code at all."""

    def test_troe_reaction_with_low_data_keeps_plus_m(self):
        rxn = {
            "id": 2,
            "kinetics_type": "troe",
            "A_orig": 1.0e13,
            "n": 0.0,
            "Ea_orig": 0.0,
            "Ea_units": "cal/mol",
            "formula_string": "A(+M)<=>B(+M)",
            "has_third_body": True,
            "kinetics_data_json": json.dumps({
                "low": {"A_si": 1e10, "n": 0.0, "Ea_si": 0.0},
                "alpha": 0.5, "T3": 100.0, "T1": 1000.0,
            }),
        }
        lines = _writer()._format_reaction(rxn, _STOICH, _EFF)
        rxn_line = lines[0]
        assert "(+M)" in rxn_line
        assert any(line.strip().startswith("LOW") for line in lines)
        assert any(line.strip().startswith("TROE") for line in lines)

    def test_lindemann_reaction_with_specific_collider_keeps_it(self):
        rxn = {
            "id": 3,
            "kinetics_type": "lindemann",
            "A_orig": 1.0e13,
            "n": 0.0,
            "Ea_orig": 0.0,
            "Ea_units": "cal/mol",
            "formula_string": "A(+AR)<=>B(+AR)",
            "has_third_body": True,
            "kinetics_data_json": json.dumps({
                "low": {"A_si": 1e10, "n": 0.0, "Ea_si": 0.0},
            }),
        }
        lines = _writer()._format_reaction(rxn, _STOICH, _EFF)
        rxn_line = lines[0]
        assert "(+AR)" in rxn_line
