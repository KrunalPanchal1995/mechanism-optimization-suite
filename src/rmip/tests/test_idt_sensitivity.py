"""
Tests for W6 — first-class ignition-delay sensitivity.

Covers `Cantera google groups/W6_idt_sensitivity.md` (P20): base IDT, brute vs
CVODES agreement, log/raw normalisation relationship, central/forward diff,
IDT-definition selection, the no-ignition guard, and the `rmip idt-sens` CLI.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_idt_sensitivity.py -v
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

from rmip.simulation.idt_sensitivity import IdtSensitivityAnalyzer   # noqa: E402

_MECH = os.path.join(ct.__path__[0], "data", "h2o2.yaml")


@pytest.fixture(scope="module")
def analyzer():
    return IdtSensitivityAnalyzer(_MECH, fuel="H2", T_K=1000, P_atm=10, phi=1.0)


class TestBase:
    def test_base_idt_is_positive_finite(self, analyzer):
        tau = analyzer.base_idt()
        assert tau is not None and 0 < tau < 1.0

    def test_no_ignition_raises(self):
        a = IdtSensitivityAnalyzer(_MECH, fuel="H2", T_K=1000, P_atm=10, phi=1.0,
                                   max_time=1e-6)      # far too short to ignite
        with pytest.raises(RuntimeError):
            a.run_brute()


class TestMethodsAgree:
    def test_brute_cvodes_top5_overlap(self, analyzer):
        rb = analyzer.run_brute(delta=0.05, norm="log")
        rc = analyzer.run_cvodes(norm="log")
        bset = {e for e, _ in rb.top(5)}
        cset = {e for e, _ in rc.top(5)}
        assert len(bset & cset) >= 4
        assert rb.top(1)[0][0] == rc.top(1)[0][0]      # same #1 reaction

    def test_values_close(self, analyzer):
        rb = analyzer.run_brute(delta=0.05, norm="log")
        rc = analyzer.run_cvodes(norm="log")
        eq = rb.top(1)[0][0]
        vb, vc = rb.coefficients[eq], rc.coefficients[eq]
        assert abs(vb - vc) / abs(vb) < 0.10           # within 10%


class TestPhysics:
    def test_chain_branching_is_negative(self, analyzer):
        """Speeding H+O2<=>O+OH (chain branching) shortens IDT → S<0."""
        r = analyzer.run_brute(delta=0.05, norm="log")
        branch = next((s for eq, s in r.coefficients.items()
                       if eq.replace(" ", "") == "H+O2<=>O+OH"), None)
        assert branch is not None and branch < 0

    def test_third_body_termination_is_positive(self, analyzer):
        """H+O2+M<=>HO2+M (chain termination) lengthens IDT → S>0."""
        r = analyzer.run_brute(delta=0.05, norm="log")
        term = [s for eq, s in r.coefficients.items()
                if "HO2 + M" in eq and "H + O2 + M" in eq]
        assert term and term[0] > 0


class TestNormalization:
    def test_log_vs_raw_relationship(self, analyzer):
        """raw sensitivity ≈ τ · log sensitivity (to first order)."""
        rl = analyzer.run_brute(delta=0.05, norm="log")
        rr = analyzer.run_brute(delta=0.05, norm="raw")
        eq = rl.top(1)[0][0]
        expected_raw = rl.idt_s * rl.coefficients[eq]
        assert abs(rr.coefficients[eq] - expected_raw) / abs(expected_raw) < 0.1


class TestDiffAndDefinition:
    def test_central_and_forward_agree_on_top(self, analyzer):
        rc = analyzer.run_brute(delta=0.05, diff="central", norm="log")
        rf = analyzer.run_brute(delta=0.05, diff="forward", norm="log")
        assert rc.top(1)[0][0] == rf.top(1)[0][0]

    def test_species_definition_runs(self):
        a = IdtSensitivityAnalyzer(_MECH, fuel="H2", T_K=1000, P_atm=10, phi=1.0,
                                   idt_definition="species", species="OH")
        tau = a.base_idt()
        assert tau is not None and tau > 0

    def test_threshold_definition_runs(self):
        a = IdtSensitivityAnalyzer(_MECH, fuel="H2", T_K=1000, P_atm=10, phi=1.0,
                                   idt_definition="threshold", threshold_dT=400)
        r = a.run_brute(delta=0.05, norm="log", n_top=5)
        assert len(r.coefficients) == 5


class TestSerialization:
    def test_to_dict_and_render(self, analyzer):
        r = analyzer.run_cvodes(norm="log", n_top=10)
        d = r.to_dict()
        assert d["observable"] == "IDT" and d["method"] == "cvodes"
        assert len(d["coefficients"]) == 10
        assert "log-norm" in r.render_text(n=5)


class TestCli:
    def test_idt_sens_cli(self, tmp_path):
        from rmip.cli import main
        out = tmp_path / "s.json"
        rc = main(["idt-sens", _MECH, "--fuel", "H2", "--T", "1000",
                   "--P", "10", "--phi", "1.0", "--method", "cvodes",
                   "--top", "5", "--json", str(out)])
        assert rc == 0 and out.exists()
        import json
        d = json.load(open(out))
        assert d["observable"] == "IDT"
