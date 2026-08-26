"""
W9 — onboarding tests: dependency preflight (doctor, P38), task→recipe
router (how-do-i, P36), and the W9c input-hygiene fold-in.
"""
import os
import sys

import pytest

from rmip.config import preflight
from rmip.config.preflight import TASKS, check_environment
from rmip.config.recipes import RECIPES, find_recipes, render_recipes

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# W9a — doctor
# ---------------------------------------------------------------------------
class TestPreflight:
    def test_present_backend_reported(self):
        pytest.importorskip("cantera")
        report = check_environment("check")
        by_name = {r.name: r for r in report.results}
        assert by_name["cantera"].present
        assert "cantera" in by_name["cantera"].detail
        assert by_name["cantera"].remediation == ""       # nothing to fix

    def test_missing_backend_reported_with_remediation(self, monkeypatch):
        monkeypatch.setattr(preflight, "CHAOSPY_PYTHON",
                            "/nonexistent/venv/bin/python3")
        report = check_environment("uq")
        chaospy = next(r for r in report.results if r.name == "chaospy_venv")
        assert not chaospy.present
        assert chaospy.required
        assert "not found" in chaospy.detail
        assert "numpoly" in chaospy.remediation      # the exact-pins fix
        assert not report.ok
        assert "NOT READY" in report.to_text()

    def test_optional_prereq_does_not_fail_report(self, monkeypatch):
        pytest.importorskip("cantera")
        # gaussian is optional for rmg; simulate everything required present
        monkeypatch.setattr(preflight, "RMG_PYTHON", sys.executable)
        monkeypatch.setattr(preflight, "RMG_REPO",
                            os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.setattr(preflight, "_probe_binary",
                            lambda *names: (False, "simulated absent"))
        report = check_environment("rmg")
        gauss = next(r for r in report.results if r.name == "gaussian")
        assert not gauss.present and not gauss.required
        assert report.ok                       # optional absence ≠ failure

    def test_every_task_prereq_is_registered(self):
        for task, (req, opt) in TASKS.items():
            for name in req + opt:
                assert name in preflight._PREREQS, (task, name)

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="unknown task"):
            check_environment("frobnicate")

    def test_to_dict_roundtrip(self):
        report = check_environment("regression")
        d = report.to_dict()
        assert d["task"] == "regression"
        assert {r["name"] for r in d["results"]} >= {"cantera", "numpy"}


# ---------------------------------------------------------------------------
# W9c — input-hygiene fold-in
# ---------------------------------------------------------------------------
class TestDoctorRmgLint:
    def test_dual_listed_library_surfaces_in_doctor(self):
        dual = os.path.join(_FIXTURES, "rmg_input_dual.py")
        if not os.path.exists(dual):
            pytest.skip("rmg_input_dual.py fixture not present")
        report = check_environment("rmg", rmg_input=dual)
        cats = {getattr(f, "category", "") for f in report.lint_findings}
        assert "dual-listed-library" in cats
        assert not report.ok                   # input error → NOT READY
        assert "dual" in report.to_text().lower() or "BOTH" in report.to_text()

    def test_missing_input_file_reported(self):
        report = check_environment("rmg", rmg_input="/no/such/input.py")
        assert any(getattr(f, "category", "") == "missing-file"
                   for f in report.lint_findings)
        assert not report.ok


# ---------------------------------------------------------------------------
# W9b — recipe router
# ---------------------------------------------------------------------------
class TestRecipes:
    @pytest.mark.parametrize("query,expect_cmd", [
        ("idt sensitivity", "idt-sens"),            # the W6 recipe
        ("ignition delay sensitivity", "idt-sens"),
        ("reduce a mechanism", " reduce "),
        ("check my file", " check "),
        ("reaction path diagram", " flux "),
        ("convert chemkin to yaml", "preflight"),
        ("did the flame speed change after upgrading cantera", "regression"),
        ("which prerequisites are missing", "doctor"),
    ])
    def test_known_tasks_route_deterministically(self, query, expect_cmd):
        top = find_recipes(query)[0]
        assert expect_cmd in top.command + " ", \
            f"{query!r} routed to {top.command!r}"

    def test_every_recipe_uses_the_launcher(self):
        # env-isolation rule: never `python src/rmip/cli.py` directly
        for r in RECIPES:
            assert "src/rmip/cli.py" not in r.command
            assert r.description

    def test_render_includes_command_and_gui(self):
        text = render_recipes(find_recipes("idt sensitivity"),
                              query="idt sensitivity")
        assert "rmip_cli.py" in text
        assert "idt_sensitivity_gui.py" in text

    def test_nonsense_still_answers_closest(self):
        assert find_recipes("zzz qqq totally unrelated") != []


# ---------------------------------------------------------------------------
# CLI smoke — the commands exist and wire through
# ---------------------------------------------------------------------------
class TestCli:
    def test_how_do_i(self, capsys):
        from rmip.cli import main
        rc = main(["how-do-i", "idt", "sensitivity"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "idt-sens" in out

    def test_doctor_runs(self, capsys):
        pytest.importorskip("cantera")
        from rmip.cli import main
        rc = main(["doctor", "--task", "check"])
        out = capsys.readouterr().out
        assert "rmip doctor" in out
        assert rc == 0                 # shared env has the check toolchain
