"""
W8b — version-behaviour regression harness tests (P28/P15).

Uses a two-case mini-suite (h2o2 equilibrium + IDT — seconds, no flame solve)
so the subprocess round-trip, golden bless/diff cycle and drift detection are
all exercised end-to-end without the full canonical suite's runtime.
"""
import copy
import json

import pytest

pytest.importorskip("cantera")

from rmip.analysis import version_regression as vr

MINI_SUITE = [
    {"id": "h2o2-equilibrium-phi1", "mechanism": "h2o2.yaml",
     "task": "equilibrium", "T_K": 300.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "H2:1.0", "track_species": ["H2O", "OH"]},
    {"id": "h2o2-idt-1100K-1atm", "mechanism": "h2o2.yaml",
     "task": "idt", "T_K": 1100.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "H2:1.0", "t_end_s": 0.05},
]


@pytest.fixture(scope="module")
def payload():
    """One real subprocess suite run, shared by every test in the module."""
    return vr.run_suite(cases=MINI_SUITE, timeout_s=600.0)


class TestRunSuite:
    def test_payload_shape(self, payload):
        assert set(payload["results"]) == {c["id"] for c in MINI_SUITE}
        assert payload["meta"]["cantera_version"]
        for entry in payload["results"].values():
            assert entry["ok"], entry["error"]

    def test_observables_physical(self, payload):
        eq = payload["results"]["h2o2-equilibrium-phi1"]["observables"]
        # stoichiometric H2/air HP-equilibrium: adiabatic flame ~2380 K
        assert 2200.0 < eq["T_ad_K"] < 2600.0
        assert 0.2 < eq["X_H2O"] < 0.4
        idt = payload["results"]["h2o2-idt-1100K-1atm"]["observables"]["idt_s"]
        assert 1e-6 < idt < 0.05

    def test_failing_case_reported_not_raised(self):
        bad = [{"id": "nope", "mechanism": "does_not_exist_zz.yaml",
                "task": "equilibrium", "T_K": 300.0, "P_atm": 1.0,
                "phi": 1.0, "fuel": "H2:1.0"}]
        out = vr.run_suite(cases=bad, timeout_s=300.0)
        entry = out["results"]["nope"]
        assert not entry["ok"]
        assert entry["error"]


class TestGoldenCycle:
    def test_bless_then_match(self, payload, tmp_path):
        golden = str(tmp_path / "golden.json")
        vr.bless(payload, golden_path=golden)
        with open(golden) as fh:
            doc = json.load(fh)
        assert doc["meta"]["blessed_at"]

        report = vr.diff_against_golden(payload, golden_path=golden,
                                        cases=MINI_SUITE)
        assert report.ok
        assert all(e.status == "ok" for e in report.entries)
        assert "PASS" in report.to_text()

    def test_drift_detected_with_magnitude(self, payload, tmp_path):
        golden = str(tmp_path / "golden.json")
        vr.bless(payload, golden_path=golden)
        # simulate a version whose IDT shifted +20%
        with open(golden) as fh:
            doc = json.load(fh)
        gid = "h2o2-idt-1100K-1atm"
        true_idt = doc["results"][gid]["observables"]["idt_s"]
        doc["results"][gid]["observables"]["idt_s"] = true_idt * 1.2
        with open(golden, "w") as fh:
            json.dump(doc, fh)

        report = vr.diff_against_golden(payload, golden_path=golden,
                                        cases=MINI_SUITE)
        assert not report.ok
        drifted = [e for e in report.entries if e.status == "drift"]
        assert [e.observable for e in drifted] == ["idt_s"]
        d = drifted[0]
        # |current - golden| / golden = 0.2/1.2
        assert d.rel_delta == pytest.approx(0.2 / 1.2, rel=1e-6)
        assert d.tolerance == vr.DEFAULT_TOLERANCES["idt"]
        assert "FAIL" in report.to_text()

    def test_tolerance_override_absorbs_drift(self, payload, tmp_path):
        golden = str(tmp_path / "golden.json")
        vr.bless(payload, golden_path=golden)
        with open(golden) as fh:
            doc = json.load(fh)
        gid = "h2o2-idt-1100K-1atm"
        doc["results"][gid]["observables"]["idt_s"] *= 1.03
        with open(golden, "w") as fh:
            json.dump(doc, fh)
        strict = vr.diff_against_golden(payload, golden_path=golden,
                                        cases=MINI_SUITE,
                                        tolerances={"idt": 0.001})
        loose = vr.diff_against_golden(payload, golden_path=golden,
                                       cases=MINI_SUITE,
                                       tolerances={"idt": 0.10})
        assert not strict.ok
        assert loose.ok

    def test_bless_refuses_failed_cases(self, payload, tmp_path):
        broken = copy.deepcopy(payload)
        broken["results"]["h2o2-idt-1100K-1atm"] = {
            "ok": False, "observables": {}, "error": "boom"}
        with pytest.raises(RuntimeError, match="refusing to bless"):
            vr.bless(broken, golden_path=str(tmp_path / "g.json"))

    def test_missing_golden_is_explicit(self, payload, tmp_path):
        with pytest.raises(FileNotFoundError, match="--bless"):
            vr.diff_against_golden(payload,
                                   golden_path=str(tmp_path / "absent.json"))


class TestDiffResultSets:
    """The generic engine — also the behavioural half of the RMG diff (P15)."""

    def test_new_and_missing_cases(self):
        ref = {"a": {"ok": True, "observables": {"x": 1.0}}}
        cur = {"b": {"ok": True, "observables": {"x": 1.0}}}
        entries = vr.diff_result_sets(cur, ref)
        by_case = {e.case_id: e.status for e in entries}
        assert by_case == {"a": "missing", "b": "new"}

    def test_failed_current_case(self):
        ref = {"a": {"ok": True, "observables": {"x": 1.0}}}
        cur = {"a": {"ok": False, "observables": {}, "error": "died"}}
        (entry,) = vr.diff_result_sets(cur, ref)
        assert entry.status == "failed"
        assert "died" in entry.detail

    def test_two_rmg_style_result_sets(self):
        # two "builds" of the same target observable → behavioural drift
        run_a = {"tign": {"ok": True, "observables": {"idt_s": 1.00e-3}}}
        run_b = {"tign": {"ok": True, "observables": {"idt_s": 1.30e-3}}}
        entries = vr.diff_result_sets(run_a, run_b,
                                      case_tasks={"tign": "idt"})
        (e,) = entries
        assert e.status == "drift"
        assert e.rel_delta == pytest.approx(0.3 / 1.3, rel=1e-6)
