"""
Tests for W4 — thermo-data provenance & cross-reference checker.

Covers `Cantera google groups/W4_thermo_transport_provenance.md` (P10/P27):
zero false positives on clean mechanisms, detection + systematic/distributed
classification of injected enthalpy errors, the `shift_enthalpy_to_ref` fix,
composition-guarded matching, user-supplied CHEMKIN THERMO reference files,
and the `rmip thermo-check` CLI.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_thermo_provenance.py -v
"""
import os
import sys
import warnings

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

warnings.filterwarnings("ignore")
ct = pytest.importorskip("cantera")
import yaml as _yaml                                                # noqa: E402

from rmip.analysis.thermo_provenance import (                       # noqa: E402
    ThermoProvenanceChecker, BUILTIN_REFS, nasa7_props_298,
    load_chemkin_thermo_refs,
)
from rmip.analysis.mechanism_check import MechanismChecker, ALL_CHECKS  # noqa: E402
from rmip.analysis import mechanism_fixer as fixer_mod              # noqa: E402

_H2O2 = os.path.join(ct.__path__[0], "data", "h2o2.yaml")
_GRI = os.path.join(ct.__path__[0], "data", "gri30.yaml")
R = 8.314462618


@pytest.fixture()
def hf_bad_yaml(tmp_path):
    """h2o2.yaml with OH's ΔHf shifted +25 kJ/mol via a6 (both ranges)."""
    raw = _yaml.safe_load(open(_H2O2))
    for sp in raw["species"]:
        if sp["name"] == "OH":
            for rng in sp["thermo"]["data"]:
                rng[5] = float(rng[5]) + 25000.0 / R
    p = tmp_path / "hf_bad.yaml"
    _yaml.safe_dump(raw, open(p, "w"), sort_keys=False)
    return str(p)


class TestCleanMechanisms:
    @pytest.mark.parametrize("mech", [_H2O2, _GRI])
    def test_no_false_positives(self, mech):
        gas = ct.Solution(mech)
        comparisons = ThermoProvenanceChecker().compare_gas(gas)
        assert comparisons, "expected some matched species"
        assert not [c for c in comparisons if c.flagged], \
            [(c.name, c.d_hf) for c in comparisons if c.flagged]

    def test_gri_matches_many_species(self):
        gas = ct.Solution(_GRI)
        comparisons = ThermoProvenanceChecker().compare_gas(gas)
        assert len(comparisons) >= 20


class TestDetection:
    def test_injected_hf_error_flagged_systematic(self, hf_bad_yaml):
        gas = ct.Solution(hf_bad_yaml)
        flagged = [c for c in ThermoProvenanceChecker().compare_gas(gas)
                   if c.flagged]
        assert len(flagged) == 1
        c = flagged[0]
        assert c.name == "OH"
        assert c.classification == "systematic-enthalpy"
        # injected +25 plus OH's inherent ~+1.86 offset
        assert 24.0 < c.d_hf < 30.0
        # a6 shift must not touch S or Cp
        assert abs(c.d_s) < 1.0
        assert abs(c.d_cp_rel) < 0.01

    def test_composition_guard_blocks_isomer_matches(self):
        """A species named like a ref but with different composition is skipped."""
        gas = ct.Solution(_H2O2)
        checker = ThermoProvenanceChecker()
        assert checker._match("OH", {"H": 1.0, "O": 1.0}) is not None
        assert checker._match("OH", {"C": 1.0, "H": 4.0}) is None

    def test_props_at_298_sane_for_h2(self):
        gas = ct.Solution(_H2O2)
        c = list(gas.species("H2").thermo.coeffs)
        hf, s, cp = nasa7_props_298(c[8:15])
        assert abs(hf) < 0.5                     # ΔHf(H2) ≡ 0
        assert abs(s - 130.68) < 1.0
        assert abs(cp - 28.84) < 1.0


class TestFix:
    def test_shift_enthalpy_repairs_the_outlier(self, hf_bad_yaml, tmp_path):
        rep = MechanismChecker(hf_bad_yaml).run(checks=["thermo_provenance"])
        flagged = [f for f in rep.findings if f.category == "thermo-provenance"]
        assert len(flagged) == 1
        opt = next(o for o in flagged[0].fix_options
                   if o.id == "shift_enthalpy_to_ref")
        assert opt.recommended                    # systematic case
        fixer = fixer_mod.MechanismFixer(hf_bad_yaml)
        results = fixer.apply([{"option_id": opt.id, "params": opt.params,
                                "location": flagged[0].location}])
        assert results[0].status == "applied"
        out = str(tmp_path / "fixed.yaml")
        fixer.write(out)
        gas = ct.Solution(out)
        flagged2 = [c for c in ThermoProvenanceChecker().compare_gas(gas)
                    if c.flagged]
        assert not flagged2


class TestUserRefs:
    def test_load_chemkin_thermo_refs(self, tmp_path):
        """A CHEMKIN THERMO file parses into usable reference entries."""
        from rmip.tests.fixtures.chemkin._make_opensmoke_fixture import (
            _nasa7_card,
        )
        gas = ct.Solution(_H2O2)
        sp = gas.species("H2O")
        c = list(sp.thermo.coeffs)
        card = _nasa7_card("H2O", dict(sp.composition), c[8:15], c[1:8],
                           sp.thermo.min_temp, c[0], sp.thermo.max_temp)
        p = tmp_path / "refs.dat"
        p.write_text("THERMO ALL\n   200.000  1000.000  6000.000\n"
                     + card + "\nEND\n")
        refs = load_chemkin_thermo_refs(str(p))
        assert "H2O" in refs
        assert abs(refs["H2O"].hf298_kJmol - (-241.826)) < 2.0


class TestIntegration:
    def test_registered_in_check_battery(self):
        assert "thermo_provenance" in ALL_CHECKS

    def test_check_battery_flags_outlier_as_warning(self, hf_bad_yaml):
        rep = MechanismChecker(hf_bad_yaml).run(checks=["thermo_provenance"])
        w = [f for f in rep.findings if f.category == "thermo-provenance"]
        assert len(w) == 1 and w[0].severity == "warning"
        assert rep.ok                             # warnings never fail the verdict

    def test_thermo_check_cli(self, hf_bad_yaml, tmp_path):
        from rmip.cli import main
        out = tmp_path / "cmp.json"
        rc = main(["thermo-check", hf_bad_yaml, "--json", str(out)])
        assert rc == 1                            # outlier present
        import json
        d = json.load(open(out))
        assert any(row["flagged"] for row in d)
        assert main(["thermo-check", _GRI]) == 0  # clean → 0
