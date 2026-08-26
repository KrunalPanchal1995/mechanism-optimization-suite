"""
rmip/tests/test_cantera_reader.py
==================================
Regression coverage for B1/B2 (review.md, 2026-07-14): the Cantera-3 reader
used to dispatch reaction kinetics on old Cantera-2 class names
(``ElementaryReaction``, ``ThreeBodyReaction``, ...) that Cantera >=3.0
removed, so every reaction fell through to NULL kinetics for every
yaml-parsed mechanism in the suite. Separately, even a correct read
returned Cantera 3's kmol-based SI (m, kmol, s, J/kmol) instead of this
suite's mol-based SI (m, mol, s, J/mol) convention, throwing A off by
1000**(order-1) and Ea off by 1000x.

Ran with real Cantera against the repo's FFCM1.yaml (38 species, 291
reactions covering plain Arrhenius, three-body, Troe/Lindemann/SRI
falloff). No test previously existed for this module at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cantera")

_REPO = Path(__file__).resolve().parents[3]
_FFCM1 = _REPO / "MECHANISM" / "FFCM1" / "FFCM1.yaml"

pytestmark = pytest.mark.skipif(not _FFCM1.exists(), reason="FFCM1.yaml fixture not present")


def _load():
    from rmip.io.cantera_reader import load_cantera_yaml
    return load_cantera_yaml(str(_FFCM1))


def _by_equation(rxn_list, equation):
    for r in rxn_list:
        if r["formula_string"] == equation:
            return r
    raise KeyError(equation)


class TestB1KineticsPopulated:
    def test_no_reaction_has_null_kinetics(self):
        """B1: every reaction used to fall through to A_si=None. A hollow
        Chebyshev reaction (this suite has no representative-Arrhenius
        summary for it) is the only legitimate NULL; FFCM1 has none."""
        _, rxn_list = _load()
        assert len(rxn_list) == 291
        hollow = [r["formula_string"] for r in rxn_list if r["A_si"] is None]
        assert hollow == []

    def test_kinetics_type_distribution_matches_known_mechanism_shape(self):
        _, rxn_list = _load()
        counts: dict = {}
        for r in rxn_list:
            counts[r["kinetics_type"]] = counts.get(r["kinetics_type"], 0) + 1
        assert counts == {
            "arrhenius": 249, "third_body": 12,
            "troe": 25, "lindemann": 3, "sri": 2,
        }


class TestB2MolSIUnits:
    def test_plain_arrhenius_A_and_Ea_in_mol_si(self):
        """H + O2 <=> O + OH, rate-constant {A: 9.841e13, b: 0.0, Ea: 1.531e4}
        under `units: {length: cm, quantity: mol, activation-energy: cal/mol}`.
        Known mol-SI value: A = 9.841e13 cm^3/mol/s * 1e-6 = 9.841e7 m^3/mol/s;
        Ea = 1.531e4 cal/mol * 4.184 = 64057.04 J/mol."""
        _, rxn_list = _load()
        r = _by_equation(rxn_list, "H + O2 <=> O + OH")
        assert r["A_si"] == pytest.approx(9.841e7, rel=1e-6)
        assert r["n"] == pytest.approx(0.0)
        assert r["Ea_si"] == pytest.approx(64057.04, rel=1e-6)
        assert r["has_third_body"] is False
        assert r["pressure_dependent"] is False

    def test_third_body_order_plus_one_conversion(self):
        """H2 + M <=> 2 H + M, rate-constant {A: 4.58e19, b: -1.4, Ea: 1.0439e5}
        (cm^3/mol/s, cal/mol). Third-body order=2 (H2 + M), so A is divided
        by 1e6 (cm^3->m^3) same as a bimolecular reaction: 4.58e13 m^3/mol/s.
        Regression pin: naively using kmol->mol order=reactant_order (1,
        ignoring the +1 for M) would give 4.58e16 instead."""
        _, rxn_list = _load()
        r = _by_equation(rxn_list, "H2 + M <=> 2 H + M")
        assert r["has_third_body"] is True
        assert r["A_si"] == pytest.approx(4.58e13, rel=1e-6)
        assert r["Ea_si"] == pytest.approx(1.0439e5 * 4.184, rel=1e-6)
        effs = dict(r["third_body_efficiencies"])
        assert effs["H2O"] == pytest.approx(12.02)

    def test_falloff_low_and_high_rate_kinetics_data_populated(self):
        """H2O2 (+M) <=> 2 OH (+M): Troe falloff. low_rate order = 3
        (bimolecular order 2 + 1 for M), high_rate order = 2 (no M)."""
        _, rxn_list = _load()
        r = _by_equation(rxn_list, "H2O2 (+M) <=> 2 OH (+M)")
        assert r["kinetics_type"] == "troe"
        assert r["pressure_dependent"] is True
        assert r["has_third_body"] is True
        kdata = r["kinetics_data"]
        assert kdata["low"]["A_si"] is not None
        assert kdata["high"]["A_si"] is not None
        assert kdata["falloff"] == "troe"
        assert len(kdata["troe_params"]) == 4
        # High-pressure limit is the representative A/n/Ea for this row.
        assert r["A_si"] == pytest.approx(kdata["high"]["A_si"])

    def test_A_orig_left_null_not_mislabeled(self):
        """B2: A_orig/Ea_orig used to be set to the SI value under an
        'original units' label -- CHEMKIN export was then off by ~1e9.
        Cantera doesn't expose the file's as-typed units, so these must
        stay NULL rather than lying about what unit they're in."""
        _, rxn_list = _load()
        r = _by_equation(rxn_list, "H + O2 <=> O + OH")
        assert r["A_orig"] is None
        assert r["Ea_orig"] is None
        assert r["A_units"] is None
        assert r["Ea_units"] is None
