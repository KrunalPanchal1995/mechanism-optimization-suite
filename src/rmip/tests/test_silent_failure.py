"""
W8a — silent-failure detector tests (P33 dead/decoupled, P35 case-variant).

The point of these checks is the failure mode that produces NO error: a
mechanism that loads cleanly but computes nothing. So the fixtures here are
all *loadable* (except the case-variant one, which is exactly the pre-load
raw-scan case).
"""
import pytest

pytest.importorskip("cantera")

from rmip.analysis.mechanism_check import (
    MechanismChecker, _equation_species,
)
from rmip.analysis.mechanism_fixer import MechanismFixer


_NASA7 = """\
    model: NASA7
    temperature-ranges: [200.0, 1000.0, 3500.0]
    data:
    - [2.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 4.0]
    - [2.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 4.0]"""


def _species_block(name: str, composition: str) -> str:
    return (f"- name: {name}\n"
            f"  composition: {{{composition}}}\n"
            f"  thermo:\n{_NASA7}\n")


def _mech_yaml(dead_A: float, live_A: float = 1.0e10) -> str:
    """Small loadable mechanism: one live reaction, one reaction with A=dead_A
    whose product HO2 appears nowhere else (decoupled when the reaction is
    dead)."""
    return (
        "phases:\n"
        "- name: gas\n"
        "  thermo: ideal-gas\n"
        "  elements: [H, O, N]\n"
        "  species: [H2, O2, H2O, HO2, N2]\n"
        "  kinetics: gas\n"
        "  reactions: all\n"
        "  state: {T: 300.0, P: 1 atm}\n"
        "\n"
        "species:\n"
        + _species_block("H2", "H: 2")
        + _species_block("O2", "O: 2")
        + _species_block("H2O", "H: 2, O: 1")
        + _species_block("HO2", "H: 1, O: 2")
        + _species_block("N2", "N: 2")
        + "\n"
        "reactions:\n"
        "- equation: 2 H2 + O2 => 2 H2O\n"
        f"  rate-constant: {{A: {live_A}, b: 0.0, Ea: 10000.0}}\n"
        "- equation: H2 + 2 O2 => 2 HO2\n"
        f"  rate-constant: {{A: {dead_A}, b: 0.0, Ea: 0.0}}\n")


@pytest.fixture()
def dead_mech(tmp_path):
    p = tmp_path / "dead.yaml"
    p.write_text(_mech_yaml(dead_A=0.0))
    return str(p)


@pytest.fixture()
def all_dead_mech(tmp_path):
    p = tmp_path / "all_dead.yaml"
    p.write_text(_mech_yaml(dead_A=0.0, live_A=0.0))
    return str(p)


# ---------------------------------------------------------------------------
# P33 — dead reactions / decoupled species
# ---------------------------------------------------------------------------
class TestZeroRates:
    def test_dead_reaction_flagged(self, dead_mech):
        report = MechanismChecker(dead_mech).run(checks=["zero_rates"])
        assert report.loaded
        dead = [f for f in report.findings if f.category == "dead-reaction"]
        assert len(dead) == 1
        assert "HO2" in dead[0].location
        assert dead[0].details["reaction_index"] == 1
        # the dead reaction is deletable
        assert any(o.id == "delete_reaction" for o in dead[0].fix_options)

    def test_decoupled_species_flagged(self, dead_mech):
        report = MechanismChecker(dead_mech).run(checks=["zero_rates"])
        dec = [f for f in report.findings if f.category == "decoupled-species"]
        assert [f.location for f in dec] == ["HO2"]

    def test_all_rates_zero_is_error(self, all_dead_mech):
        report = MechanismChecker(all_dead_mech).run(checks=["zero_rates"])
        cats = {f.category for f in report.findings}
        assert "all-rates-zero" in cats
        assert not report.ok               # silent success no more (P33)
        # aggregate finding replaces per-reaction spam
        assert "dead-reaction" not in cats

    def test_healthy_mechanism_clean(self):
        report = MechanismChecker("h2o2.yaml").run(checks=["zero_rates"])
        assert report.loaded
        assert report.findings == []

    def test_live_low_A_not_flagged(self, tmp_path):
        # a slow-but-real reaction must not be mistaken for dead
        p = tmp_path / "slow.yaml"
        p.write_text(_mech_yaml(dead_A=1.0e-3))
        report = MechanismChecker(str(p)).run(checks=["zero_rates"])
        assert report.findings == []


# ---------------------------------------------------------------------------
# P35 — cross-section case-variant / undeclared references
# ---------------------------------------------------------------------------
def _case_variant_yaml() -> str:
    # species declared 'AR'; the third-body efficiencies and phase list
    # reference 'Ar' — case-sensitive Cantera never connects them.
    return (
        "phases:\n"
        "- name: gas\n"
        "  thermo: ideal-gas\n"
        "  elements: [H, O, Ar]\n"
        "  species: [H2, O2, H2O, Ar]\n"
        "  kinetics: gas\n"
        "  reactions: all\n"
        "  state: {T: 300.0, P: 1 atm}\n"
        "\n"
        "species:\n"
        + _species_block("H2", "H: 2")
        + _species_block("O2", "O: 2")
        + _species_block("H2O", "H: 2, O: 1")
        + _species_block("AR", "Ar: 1")
        + "\n"
        "reactions:\n"
        "- equation: 2 H2 + O2 => 2 H2O\n"
        "  rate-constant: {A: 1.0e+10, b: 0.0, Ea: 10000.0}\n"
        "- equation: 2 H2 + O2 + M => 2 H2O + M\n"
        "  type: three-body\n"
        "  rate-constant: {A: 1.0e+10, b: 0.0, Ea: 0.0}\n"
        "  efficiencies: {Ar: 0.7}\n")


class TestNameConsistency:
    def test_case_variant_flagged_pre_load(self, tmp_path):
        p = tmp_path / "case.yaml"
        p.write_text(_case_variant_yaml())
        report = MechanismChecker(str(p)).run()
        # Cantera refuses this file (phase lists unknown 'Ar') — the raw
        # scan must still catch and explain the actual defect.
        assert not report.loaded
        hits = [f for f in report.findings
                if f.category == "species-case-mismatch"]
        assert hits, "case-variant reference not detected on raw YAML"
        assert {f.details["referenced"] for f in hits} == {"Ar"}
        assert all(f.details["declared"] == "AR" for f in hits)
        assert all(f.auto_fix is not None for f in hits)

    def test_rename_fix_makes_mechanism_loadable(self, tmp_path):
        import cantera as ct
        p = tmp_path / "case.yaml"
        p.write_text(_case_variant_yaml())
        checker = MechanismChecker(str(p))
        report = checker.run(checks=["name_consistency"])
        fixes = [{"option_id": f.auto_fix.id, "params": f.auto_fix.params,
                  "location": f.location}
                 for f in report.findings if f.auto_fix is not None]
        assert fixes
        fixer = MechanismFixer(checker.yaml_path)
        results = fixer.apply(fixes)
        assert all(r.status == "applied" for r in results)
        out = str(tmp_path / "case.fixed.yaml")
        fixer.write(out)
        gas = ct.Solution(out)             # loads now
        assert "AR" in gas.species_names
        recheck = MechanismChecker(out).run(checks=["name_consistency"])
        assert recheck.findings == []

    def test_undeclared_reference_flagged(self, tmp_path):
        text = _case_variant_yaml().replace("efficiencies: {Ar: 0.7}",
                                            "efficiencies: {XE: 0.7}") \
                                   .replace("species: [H2, O2, H2O, Ar]",
                                            "species: [H2, O2, H2O, AR]")
        p = tmp_path / "undeclared.yaml"
        p.write_text(text)
        report = MechanismChecker(str(p)).run(checks=["name_consistency"])
        cats = {f.category: f for f in report.findings}
        assert "undeclared-species-ref" in cats
        assert cats["undeclared-species-ref"].details["referenced"] == "XE"

    def test_case_twins_flagged(self, tmp_path):
        # both 'AR' and 'Ar' declared as species → suspicious twins
        text = _case_variant_yaml().replace(
            "- name: AR",
            "- name: Ar\n  composition: {Ar: 1}\n  thermo:\n" + _NASA7
            + "\n- name: AR")
        p = tmp_path / "twins.yaml"
        p.write_text(text)
        report = MechanismChecker(str(p)).run(checks=["name_consistency"])
        assert any(f.category == "species-case-twins" for f in report.findings)

    def test_clean_mechanism_no_findings(self):
        report = MechanismChecker("h2o2.yaml").run(checks=["name_consistency"])
        assert report.findings == []


# ---------------------------------------------------------------------------
# equation tokenizer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eq,expected", [
    ("2 H2 + O2 <=> 2 H2O", ["H2", "O2", "H2O"]),
    ("H + O2 (+M) <=> HO2 (+M)", ["H", "O2", "HO2"]),
    ("H + O2 (+AR) <=> HO2 (+AR)", ["H", "AR", "O2", "AR", "HO2"]),
    ("H2 + M <=> 2 H + M", ["H2", "H"]),
    ("NC7H16 => C7H15-2 + H", ["NC7H16", "C7H15-2", "H"]),
])
def test_equation_species(eq, expected):
    assert _equation_species(eq) == expected
