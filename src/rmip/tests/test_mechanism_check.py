"""
Tests for the mechanism health-check engine + fixer + GUI surface.

Covers the MECHANISM_CHECK_RESEARCH.md phase:
  * MechanismChecker detects each injected fault class on the bundled fixtures.
  * MechanismFixer repairs them and the corrected YAML re-checks clean/loadable.
  * The `check` CLI subcommand and the GUI headless self-test run end-to-end.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_mechanism_check.py -v
(via the launcher path so `src/io` does not shadow stdlib `io`.)
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

pytest.importorskip("cantera")

from rmip.analysis.mechanism_check import (           # noqa: E402
    MechanismChecker, CheckReport, Finding, ALL_CHECKS, collision_limit_kmol,
)
from rmip.analysis import mechanism_fixer as fixer_mod  # noqa: E402
from rmip.tests.fixtures._make_broken_mech import (            # noqa: E402
    build_broken_dicts, build_integrator_failure_dict,
)
from rmip.analysis.syntax_lint import SyntaxLinter, LintIssue    # noqa: E402

_FX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    """Materialise the three broken fixtures into a temp dir (fresh per run)."""
    import yaml
    d = tmp_path_factory.mktemp("mechfx")
    mech, dup, plog = build_broken_dicts()
    paths = {}
    for name, data in (("mech", mech), ("dup", dup), ("plog", plog)):
        p = d / f"broken_{name}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        paths[name] = str(p)
    return paths


@pytest.fixture(scope="module")
def integrator_failure_path(tmp_path_factory):
    """A mechanism engineered so CVODES itself fails mid-integration (P7)."""
    import yaml
    d = tmp_path_factory.mktemp("mechfx_p7")
    p = d / "broken_integrator.yaml"
    with open(p, "w") as fh:
        yaml.safe_dump(build_integrator_failure_dict(), fh, sort_keys=False)
    return str(p)


# ===========================================================================
# Detection
# ===========================================================================
class TestDetection:
    def test_clean_mechanism_passes(self):
        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        rep = MechanismChecker(gri, T_grid_K=[1000, 1800],
                               P_atm_grid=[1.0]).run()
        assert rep.loaded
        assert rep.n_errors == 0
        assert isinstance(rep.to_json(), str)

    def test_loadable_broken_has_all_fault_classes(self, fixtures):
        rep = MechanismChecker(fixtures["mech"]).run()
        assert rep.loaded
        cats = {f.category for f in rep.findings}
        assert "collision-limit" in cats
        assert "thermo-discontinuity" in cats
        assert "unused-species" in cats
        assert "ea-magnitude" in cats
        assert rep.n_errors > 0

    def test_thermo_discontinuity_is_auto_fixable(self, fixtures):
        rep = MechanismChecker(fixtures["mech"]).run(
            checks=["thermo_discontinuity"])
        disc = [f for f in rep.findings if f.category == "thermo-discontinuity"]
        assert len(disc) == 1
        assert disc[0].location == "OH"
        assert disc[0].auto_fix is not None
        assert disc[0].auto_fix.id == "refit_thermo"

    def test_unmarked_duplicate_detected_despite_load_failure(self, fixtures):
        rep = MechanismChecker(fixtures["dup"]).run()
        assert rep.loaded is False
        dups = [f for f in rep.findings if f.category == "unmarked-duplicate"]
        assert len(dups) == 1
        assert dups[0].auto_fix is not None          # "mark_duplicate"

    def test_negative_plog_detected_despite_load_failure(self, fixtures):
        rep = MechanismChecker(fixtures["plog"]).run()
        assert rep.loaded is False
        neg = [f for f in rep.findings if f.category == "plog-negative-A"]
        assert len(neg) == 1
        assert any(o.id == "plog_drop_negative" for o in neg[0].fix_options)

    def test_collision_limit_helper_scales_with_T(self):
        assert collision_limit_kmol(298.15) == pytest.approx(1e11, rel=1e-6)
        assert collision_limit_kmol(1192.6) == pytest.approx(2e11, rel=1e-3)

    def test_subset_of_checks_runs(self, fixtures):
        rep = MechanismChecker(fixtures["mech"]).run(checks=["unit_sanity"])
        assert {f.check for f in rep.findings} <= {"unit_sanity"}

    def test_report_serialization_roundtrips(self, fixtures):
        rep = MechanismChecker(fixtures["mech"]).run()
        d = rep.to_dict()
        assert d["counts"]["error"] == rep.n_errors
        assert "findings" in d and isinstance(d["findings"], list)


# ===========================================================================
# Repair
# ===========================================================================
class TestRepair:
    def _apply(self, path, fixes, out):
        chk = MechanismChecker(path)
        chk.run()
        assert chk.yaml_path is not None
        fixer = fixer_mod.MechanismFixer(chk.yaml_path)
        results = fixer.apply(fixes)
        fixer.write(out)
        chk.cleanup()
        return results

    def test_mark_duplicate_makes_it_loadable(self, fixtures, tmp_path):
        rep = MechanismChecker(fixtures["dup"]).run()
        fixes = fixer_mod.collect_auto_fixes(rep)
        assert fixes and fixes[0]["option_id"] == "mark_duplicate"
        out = str(tmp_path / "dup.fixed.yaml")
        self._apply(fixtures["dup"], fixes, out)
        rep2 = MechanismChecker(out).run()
        assert rep2.loaded
        assert not any(f.category == "unmarked-duplicate" for f in rep2.findings)

    def test_drop_negative_plog_makes_it_loadable(self, fixtures, tmp_path):
        rep = MechanismChecker(fixtures["plog"]).run()
        neg = next(f for f in rep.findings if f.category == "plog-negative-A")
        opt = next(o for o in neg.fix_options if o.id == "plog_drop_negative")
        out = str(tmp_path / "plog.fixed.yaml")
        self._apply(fixtures["plog"],
                    [{"option_id": opt.id, "params": opt.params,
                      "location": neg.location}], out)
        rep2 = MechanismChecker(out).run()
        assert rep2.loaded
        assert not any(f.category == "plog-negative-A" for f in rep2.findings)

    def test_thermo_refit_removes_discontinuity(self, fixtures, tmp_path):
        rep = MechanismChecker(fixtures["mech"]).run()
        fixes = fixer_mod.collect_auto_fixes(rep)      # includes refit_thermo
        assert any(f["option_id"] == "refit_thermo" for f in fixes)
        out = str(tmp_path / "mech.fixed.yaml")
        self._apply(fixtures["mech"], fixes, out)
        rep2 = MechanismChecker(out).run(checks=["thermo_discontinuity"])
        assert not any(f.category == "thermo-discontinuity" for f in rep2.findings)

    def test_ea_reinterpret_reduces_magnitude(self, fixtures, tmp_path):
        rep = MechanismChecker(fixtures["mech"]).run(checks=["unit_sanity"])
        f = next(x for x in rep.findings if x.category == "ea-magnitude")
        opt = next(o for o in f.fix_options if o.id == "ea_reinterpret_j_kmol")
        out = str(tmp_path / "mech_ea.fixed.yaml")
        self._apply(fixtures["mech"],
                    [{"option_id": opt.id, "params": opt.params,
                      "location": f.location}], out)
        rep2 = MechanismChecker(out).run(checks=["unit_sanity"])
        assert not any(x.category == "ea-magnitude" for x in rep2.findings)

    def test_scale_A_only_applies_to_forward_arrhenius(self, fixtures, tmp_path):
        rep = MechanismChecker(fixtures["mech"]).run(checks=["collision_limit"])
        fwd = [f for f in rep.findings
               if f.category == "collision-limit"
               and f.details.get("direction") == "forward"]
        assert fwd, "expected a forward collision-limit violation to scale"
        opt = next(o for o in fwd[0].fix_options if o.id == "scale_A_to_limit")
        out = str(tmp_path / "mech_scale.fixed.yaml")
        results = self._apply(
            fixtures["mech"],
            [{"option_id": opt.id, "params": opt.params,
              "location": fwd[0].location}], out)
        assert any(r.status == "applied" for r in results)


# ===========================================================================
# CLI + GUI surfaces
# ===========================================================================
class TestCli:
    def test_check_cli_runs_and_reports_nonzero(self, fixtures):
        from rmip.cli import main
        rc = main(["check", fixtures["dup"]])
        assert rc == 1                                 # duplicate → load fails

    def test_check_cli_fix_all_repairs(self, fixtures, tmp_path):
        from rmip.cli import main
        rc = main(["check", fixtures["dup"], "--fix", "auto",
                   "--fixed-out", str(tmp_path)])
        out = tmp_path / "broken_dup.fixed.yaml"
        assert out.exists()
        rep2 = MechanismChecker(str(out)).run()
        assert rep2.loaded


class TestGuiSurface:
    def test_headless_selftest_exits_clean(self):
        import subprocess
        proc = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "mechanism_check_gui.py"),
             "--no-gui"], capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "broken_mech.yaml" in proc.stdout

    def test_run_checks_helper(self, fixtures):
        import mechanism_check_gui as gui
        res = gui.run_checks(list(fixtures.values()))
        assert len(res) == 3
        for _p, (chk, rep) in res.items():
            assert isinstance(rep, CheckReport)
            chk.cleanup()

    def test_finding_for_iid_roundtrips_through_tree_iid_scheme(self, fixtures):
        """KRUNAL_JULY_23.md point 3: the detail pane looks a Finding up from
        the Treeview row iid ("{id(rep)}:{fid}", see _add_report_to_tree).
        Confirm that round-trip is correct, and that a stale/garbage iid
        degrades to (None, None) rather than raising."""
        import mechanism_check_gui as gui
        chk = MechanismChecker(fixtures["mech"])
        rep = chk.run()
        reports = {fixtures["mech"]: rep}
        assert rep.findings, "fixture should have at least one finding"
        for fid, f in enumerate(rep.findings):
            path, found = gui.finding_for_iid(reports, f"{id(rep)}:{fid}")
            assert path == fixtures["mech"]
            assert found is f
        # mechanism-header row (no ':') and a since-cleared/garbage iid
        assert gui.finding_for_iid(reports, "not-a-real-iid") == (None, None)
        assert gui.finding_for_iid(reports, "999999999:0") == (None, None)
        assert gui.finding_for_iid(reports, f"{id(rep)}:9999") == (None, None)
        chk.cleanup()

    def test_build_detail_text_surfaces_the_real_load_error_copyably(self):
        """The exact complaint: 'you cannot even copy the error msg. it says
        LOAD failed[.] ... make the errors thrown more descriptive[,] give
        examples on how they can fix the issue raised.' build_detail_text is
        the plain-text (Tk-Text-widget-ready, hence copyable) rendering of a
        Finding — confirm it surfaces Finding.details['error'] (previously
        shown nowhere in the GUI) and a worked fix example."""
        import mechanism_check_gui as gui
        f = Finding(
            check="load", severity="error", category="load-failure",
            message="Mechanism failed to load in Cantera — see error detail. "
                    "File-level checks below may still apply.",
            location="128-957.inp",
            details={"error": "Unexpected token \"C2H4O1-2+\" in reaction "
                              "expression \"C2H5+O2<=>C2H4O1-2+OH\""},
        )
        text = gui.build_detail_text("/some/path/128-957.inp", f)
        assert "128-957.inp" in text
        assert "LOAD-FAILURE" in text.upper() or "load-failure" in text
        assert "ERROR" in text.upper()
        # the real underlying error is present verbatim, not just "LOAD FAILED"
        assert 'Unexpected token "C2H4O1-2+"' in text
        # a worked example is included for this category
        assert "How to fix this" in text
        assert "Fortran-style exponent" in text or "SPECIES block" in text

    def test_build_detail_text_handles_no_example_and_no_error_gracefully(self):
        import mechanism_check_gui as gui
        f = Finding(check="x", severity="info", category="totally-unmapped-category",
                    message="just a message", location="line 1", details={})
        text = gui.build_detail_text(None, f)
        assert "just a message" in text
        assert "How to fix this" not in text
        assert "Underlying tool error" not in text

    def test_fix_examples_cover_common_conversion_categories(self):
        import mechanism_check_gui as gui
        for cat in ("load-failure", "species-case-mismatch",
                    "undeclared-species-ref", "thermo-discontinuity",
                    "unmarked-duplicate", "ea-magnitude"):
            assert gui._fix_example_for(cat), f"no fix example for {cat}"
        assert gui._fix_example_for("no-such-category") is None

    def test_build_run_status_before_and_after_a_check(self, fixtures):
        """KRUNAL_JULY_23.md point 1. Real tk.Tk() root, no synthetic
        clicks -- same convention as the rest of this suite's Tk-backed
        checks."""
        import tkinter as tk
        import mechanism_check_gui as gui

        root = tk.Tk()
        try:
            app = gui.MechanismCheckApp(root)
            for p in fixtures.values():
                app.mech_list.insert("end", p)

            before = app._build_run_status()
            assert before.tool == "Mechanism Health-Check"
            assert before.status == "unknown"
            assert set(before.inputs.values()) == set(fixtures.values())

            for p in fixtures.values():
                chk = MechanismChecker(p)
                app.reports[p] = chk.run()
                chk.cleanup()

            after = app._build_run_status()
            assert after.status in ("success", "failed")
            assert "error(s)" in after.summary
            assert any(os.path.basename(p) in after.log for p in fixtures.values())
        finally:
            root.destroy()


# ===========================================================================
# P7 — sim-setup sanity (bounded const-V ignition smoke test)
# ===========================================================================
class TestSimSetupSanity:
    def test_check_sim_setup_sanity_flags_integrator_failure(
            self, integrator_failure_path):
        # Engineered fixture (build_integrator_failure_dict): OH's high-T
        # NASA-7 quartic coefficient is set to an absurd, deeply-negative
        # value. It LOADS fine in Cantera (only the usual NasaPoly2
        # discontinuity warning) but drives CVODES to a non-finite RHS on
        # the very first step at combustion temperature -- verified
        # interactively before being encoded as a fixture.
        rep = MechanismChecker(integrator_failure_path).run(
            checks=["sim_setup_sanity"])
        bad = [f for f in rep.findings if f.category == "integrator-failure"]
        assert len(bad) == 1
        assert bad[0].check == "sim_setup_sanity"
        assert bad[0].severity == "warning"
        assert not bad[0].fix_options            # not auto-repairable
        assert "T_K" in bad[0].details and "P_atm" in bad[0].details

    def test_check_sim_setup_sanity_clean_on_good_fixtures(self, fixtures):
        # The other broken-fixture classes (unmarked duplicate, negative
        # PLOG, collision-limit/thermo/Ea faults) do not make the ODE
        # integrator itself fail -- confirms this check doesn't spuriously
        # fire on faults that belong to other categories.
        rep = MechanismChecker(fixtures["mech"]).run(checks=["sim_setup_sanity"])
        assert rep.findings == []

    def test_check_sim_setup_sanity_bounded_cost(self, fixtures):
        import time
        t0 = time.monotonic()
        MechanismChecker(fixtures["mech"]).run(checks=["sim_setup_sanity"])
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"sim_setup_sanity took {elapsed:.1f}s, expected <10s"


# ===========================================================================
# P8 — CSP ultrafast-timescale model screening
# ===========================================================================
class TestCspTimescales:
    def test_check_csp_timescales_flags_ultrafast_mode(self, fixtures):
        # fixtures["mech"] already carries an ~1e6x collision-limit A-factor
        # boost (build_broken_dicts fault 1) -- reused here rather than
        # inventing a second absurd-A-factor fixture, since it already
        # produces multiple sub-1e-12s eigenmodes (cross-validating P2's
        # collision-limit screen against P8's CSP timescale screen).
        rep = MechanismChecker(fixtures["mech"]).run(checks=["csp_timescales"])
        ultra = [f for f in rep.findings if f.category == "ultrafast-timescale"]
        assert ultra, "expected at least one ultrafast-timescale finding"
        for f in ultra:
            assert f.check == "csp_timescales"
            assert f.severity == "warning"
            assert f.details["tau_s"] < 1e-12

    def test_check_csp_timescales_no_spurious_flag_on_clean_mechanism(self):
        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        rep = MechanismChecker(gri, T_grid_K=[1000, 1800],
                               P_atm_grid=[1.0]).run(checks=["csp_timescales"])
        assert not any(f.category == "ultrafast-timescale" for f in rep.findings)

    def test_check_csp_timescales_species_count_guard(self):
        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        rep = MechanismChecker(gri, csp_max_species=0).run(
            checks=["csp_timescales"])
        assert len(rep.findings) == 1
        f = rep.findings[0]
        assert f.check == "csp_timescales"
        assert f.severity == "info"
        assert f.category == "csp-skipped"
        assert f.details["csp_max_species"] == 0


# ===========================================================================
# P9 — syntax linter (line/column raw text scan)
# ===========================================================================
class TestSyntaxLint:
    def test_check_syntax_lint_line_column(self, tmp_path):
        p = tmp_path / "snippet.txt"
        p.write_text(
            "! a comment line, ignored\n"
            "H+H+M<=>H2+M   1.7-13  0.0  0.0\n"
        )
        issues = SyntaxLinter(str(p)).lint()
        exp = [i for i in issues if i.severity == "error"
               and "1.7-13" in i.message]
        assert len(exp) == 1
        assert exp[0].line == 2
        # column is the 1-indexed start of '1.7-13' on line 2
        line2 = "H+H+M<=>H2+M   1.7-13  0.0  0.0\n"
        expected_col = line2.index("1.7-13") + 1
        assert exp[0].column == expected_col

    def test_check_syntax_lint_stray_tab_and_truncated_numeric(self, tmp_path):
        p = tmp_path / "snippet2.txt"
        p.write_text("A\t1.0E   2.\n")
        issues = SyntaxLinter(str(p)).lint()
        assert any(i.severity == "warning" and "tab" in i.message.lower()
                   for i in issues)
        truncated = [i for i in issues if "truncated" in i.message.lower()]
        assert len(truncated) == 2          # '1.0E' and '2.'

    def test_check_syntax_lint_runs_on_load_failure(self, tmp_path):
        # A file that is neither valid CHEMKIN nor valid YAML -- Cantera's
        # load fails outright -- but the raw text still carries a
        # classic missing-exponent defect. syntax_lint is wired into the
        # pre-load fallback path in MechanismChecker.run() precisely so it
        # still reports here, unlike checks that need a parsed YAML/Solution.
        p = tmp_path / "garbage.yaml"
        p.write_text(
            "this is not { valid yaml or chemkin\n"
            "some reaction line  1.7-13  0.0  0.0\n"
        )
        rep = MechanismChecker(str(p)).run()
        assert rep.loaded is False
        lint_findings = [f for f in rep.findings if f.check == "syntax_lint"]
        assert lint_findings, "expected syntax_lint findings despite load failure"
        assert any("1.7-13" in f.message for f in lint_findings)

    def test_check_syntax_lint_wired_into_checker(self, fixtures):
        # end-to-end through MechanismChecker.check_syntax_lint on an
        # already-valid YAML dump (yaml.safe_dump's float formatting always
        # includes 'e', so no findings are expected here -- this just
        # proves the plumbing (Finding.location/details) is correct).
        rep = MechanismChecker(fixtures["mech"]).run(checks=["syntax_lint"])
        assert {f.check for f in rep.findings} <= {"syntax_lint"}
        for f in rep.findings:
            assert f.category == "syntax"
            assert f.location.startswith("line ")


# ===========================================================================
# Wiring sanity for the three new (P7/P8/P9) checks
# ===========================================================================
class TestNewChecksWiring:
    def test_all_checks_includes_new_p7_p8_p9(self, fixtures):
        assert {"sim_setup_sanity", "csp_timescales", "syntax_lint"} <= set(ALL_CHECKS)

        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        rep = MechanismChecker(gri, T_grid_K=[1000, 1800],
                               P_atm_grid=[1.0]).run()       # no checks= filter
        assert not any(f.category == "check-error" for f in rep.findings)

        rep2 = MechanismChecker(fixtures["mech"]).run()      # no checks= filter
        assert not any(f.category == "check-error" for f in rep2.findings)
        ran = {f.check for f in rep2.findings}
        # sim_setup_sanity is expected to be clean on this fixture, so it
        # may legitimately contribute zero findings; just confirm the other
        # two new checks actually produced something on this known-bad file.
        assert "csp_timescales" in ran
        assert rep2.loaded is True


# ===========================================================================
# M6 follow-on — third-body-implied duplicate reactions (bug.md)
# ===========================================================================
class TestThirdBodyDuplicates:
    """check_duplicates only matches identical equation-string signatures --
    it cannot see that a bare-M reaction and an explicit self-collider form
    of the SAME physical reaction (found stress-testing fusion on Keromnes/
    Li/Davis) become duplicates once Cantera expands the third body at load
    time. This is exactly the class of bug that broke the suite's own
    canonical fused mechanism (bug.md)."""

    def _rep(self, tmp_path, reactions, name="tb.yaml"):
        import yaml
        p = tmp_path / name
        p.write_text(yaml.safe_dump({"reactions": reactions}, sort_keys=False))
        return MechanismChecker(str(p)).run(checks=["third_body_duplicates"])

    def test_bare_m_and_explicit_self_collider_flagged(self, tmp_path):
        rep = self._rep(tmp_path, [
            {"equation": "H2 + M <=> 2 H + M"},
            {"equation": "2 H + H2 <=> 2 H2"},
        ])
        found = [f for f in rep.findings if f.check == "third_body_duplicates"]
        assert len(found) == 1
        assert found[0].category == "third-body-implied-duplicate"
        assert found[0].details["reaction_indices"] == [0, 1]
        assert "H2 + M <=> 2 H + M" in found[0].message
        assert "2 H + H2 <=> 2 H2" in found[0].message

    def test_unrelated_reactions_not_flagged(self, tmp_path):
        rep = self._rep(tmp_path, [
            {"equation": "H2 + M <=> 2 H + M"},
            {"equation": "CO + O <=> CO2"},
            {"equation": "H + OH <=> H2O"},
        ])
        assert not [f for f in rep.findings if f.check == "third_body_duplicates"]

    def test_already_textually_identical_not_double_reported(self, tmp_path):
        """Two reactions with the SAME textual signature are check_duplicates'
        job (already correctly handled there) -- must not also fire here."""
        rep = self._rep(tmp_path, [
            {"equation": "H2 + M <=> 2 H + M"},
            {"equation": "M + H2 <=> M + 2 H"},
        ])
        assert not [f for f in rep.findings if f.check == "third_body_duplicates"]

    def test_generic_m_falloff_and_specific_collider_not_confused(self, tmp_path):
        """A generic-M falloff and its (+AR) sibling of the SAME base
        equation are physically DIFFERENT reactions (different LOW/TROE
        parameters) -- must not be flagged as third-body-implied duplicates
        of each other, since neither has an explicit self-collider form."""
        rep = self._rep(tmp_path, [
            {"equation": "H2O2 (+M) <=> 2 OH (+M)"},
            {"equation": "H2O2 (+AR) <=> 2 OH (+AR)"},
        ])
        assert not [f for f in rep.findings if f.check == "third_body_duplicates"]

    def test_real_keromnes_li_davis_repro(self, tmp_path):
        """The exact repro that broke the suite's own canonical fused
        mechanism (bug.md): Keromnes' generic-M form of H2 dissociation vs.
        Li/Davis-style explicit self-third-body form of the same step."""
        rep = self._rep(tmp_path, [
            {"equation": "H2 + M <=> 2 H + M"},          # Keromnes-style
            {"equation": "2 H + H2 <=> H2 + H2"},          # Li/Davis-style
        ])
        found = [f for f in rep.findings if f.check == "third_body_duplicates"]
        assert len(found) == 1

    def test_already_marked_duplicate_true_not_reflagged(self, tmp_path):
        """The exact same pair, but ALREADY correctly resolved (both marked
        duplicate: true, e.g. the output of the canonical-key-rebuild
        migration's re-fuse — Cantera sums their rates and loads fine).
        Nothing actionable left to report."""
        rep = self._rep(tmp_path, [
            {"equation": "H2 + M <=> 2 H + M", "duplicate": True},
            {"equation": "2 H + H2 <=> 2 H2", "duplicate": True},
        ])
        assert not [f for f in rep.findings if f.check == "third_body_duplicates"]

    def test_efficiency_override_species_with_default_species_not_confused(self, tmp_path):
        """A generic-M reaction that zeroes out species X's efficiency, plus
        a SEPARATE explicit self-collider reaction for a DIFFERENT species Y
        (not overridden) -- must flag only the Y pairing, not X."""
        rep = self._rep(tmp_path, [
            {"equation": "H + O2 + M <=> HO2 + M",
             "efficiencies": {"AR": 0.0}},
            {"equation": "H + O2 + AR <=> HO2 + AR"},   # X: explicitly excluded
            {"equation": "H + 2 O2 <=> HO2 + O2"},      # Y: NOT excluded
        ])
        found = [f for f in rep.findings if f.check == "third_body_duplicates"]
        assert len(found) == 1
        assert "H + 2 O2 <=> HO2 + O2" in found[0].message
        assert "H + O2 + AR <=> HO2 + AR" not in found[0].message

    def test_real_gri30_no_false_positives(self):
        """Sanity regression against a real, well-vetted mechanism: GRI-3.0
        deliberately writes several per-species-collider reaction clusters
        (e.g. H+O2+M with O2/H2O/N2/AR efficiencies explicitly zeroed, each
        with its own separate explicit reaction) plus reactions that
        coincidentally share a stripped-M core with an unrelated bare
        reaction (e.g. NNH<=>N2+H vs NNH+M<=>N2+H+M) -- none of these are
        real third-body-implied duplicates and must not be flagged."""
        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        rep = MechanismChecker(gri).run(checks=["third_body_duplicates"])
        assert not [f for f in rep.findings if f.check == "third_body_duplicates"]

    def test_third_body_duplicates_in_all_checks_and_load_failure_path(self, tmp_path):
        """Wired into ALL_CHECKS and the pre-load raw-YAML fallback path
        (MechanismChecker.run()'s early return for a file Cantera refuses to
        load) -- same treatment as check_duplicates."""
        assert "third_body_duplicates" in ALL_CHECKS
        import yaml
        p = tmp_path / "unloadable.yaml"
        p.write_text(yaml.safe_dump({
            "reactions": [
                {"equation": "H2 + M <=> 2 H + M"},
                {"equation": "2 H + H2 <=> 2 H2"},
            ],
        }, sort_keys=False))
        rep = MechanismChecker(str(p)).run()  # no checks= filter -- Cantera can't load this
        assert rep.loaded is False
        found = [f for f in rep.findings if f.check == "third_body_duplicates"]
        assert len(found) == 1


# ===========================================================================
# Bug fix (2026-08-05, found via the launcher stress-test/code-review pass):
# check_duplicates false-positives on legitimate irreversible forward/
# backward reaction pairs (Test_run_23_JULY/mechanism_check_gui/SUMMARY.md).
# ===========================================================================
class TestIrreversibleDuplicateFalsePositive:
    def test_signature_distinguishes_irreversible_direction(self):
        """The exact bug: 'A + B => C' and 'C => A + B' are two distinct,
        independently-parameterized one-way reactions -- not duplicates of
        each other -- but the old signature collapsed both sides into an
        order-insensitive frozenset unconditionally, regardless of whether
        the reaction was even reversible, so they hashed identically."""
        from rmip.analysis.mechanism_check import _yaml_reaction_signature

        forward = _yaml_reaction_signature("A + B => C")
        backward = _yaml_reaction_signature("C => A + B")
        assert forward != backward

    def test_signature_still_collapses_reversible_reactions(self):
        """A genuinely reversible reaction IS symmetric -- 'A + B <=> C' and
        'C <=> A + B' represent the identical physical equilibrium and must
        still collide (this is check_duplicates' actual job for the common
        case, and must not regress)."""
        from rmip.analysis.mechanism_check import _yaml_reaction_signature

        assert _yaml_reaction_signature("A + B <=> C") == _yaml_reaction_signature("C <=> A + B")

    def test_signature_still_collapses_species_order_within_a_side(self):
        """Unrelated to reversibility: 'A + B' and 'B + A' on the same side
        must still collide regardless of direction."""
        from rmip.analysis.mechanism_check import _yaml_reaction_signature

        assert (_yaml_reaction_signature("A + B => C")
                == _yaml_reaction_signature("B + A => C"))

    def test_check_duplicates_no_false_positive_on_irreversible_pair(self, tmp_path):
        """Integration-level repro of the original bug, through the real
        check_duplicates() path used by MechanismCheckPage/mechanism_check_gui."""
        import yaml
        p = tmp_path / "fwd_back.yaml"
        p.write_text(yaml.safe_dump({
            "reactions": [
                {"equation": "H2 + O => H + OH"},
                {"equation": "H + OH => H2 + O"},
            ],
        }, sort_keys=False))
        rep = MechanismChecker(str(p)).run()  # unloadable minimal fixture -- raw-YAML path
        assert rep.loaded is False
        dups = [f for f in rep.findings if f.category == "unmarked-duplicate"]
        assert dups == []

    def test_check_duplicates_still_flags_a_real_irreversible_duplicate(self, tmp_path):
        """Same equation, same direction, twice -- still a genuine duplicate
        and must still be flagged."""
        import yaml
        p = tmp_path / "real_dup.yaml"
        p.write_text(yaml.safe_dump({
            "reactions": [
                {"equation": "H2 + O => H + OH"},
                {"equation": "H2 + O => H + OH"},
            ],
        }, sort_keys=False))
        rep = MechanismChecker(str(p)).run()
        assert rep.loaded is False
        dups = [f for f in rep.findings if f.category == "unmarked-duplicate"]
        assert len(dups) == 1

    def test_check_duplicates_still_flags_a_real_reversible_duplicate(self, tmp_path):
        import yaml
        p = tmp_path / "real_rev_dup.yaml"
        p.write_text(yaml.safe_dump({
            "reactions": [
                {"equation": "H2 + O <=> H + OH"},
                {"equation": "H + OH <=> H2 + O"},  # same reversible reaction, written backwards
            ],
        }, sort_keys=False))
        rep = MechanismChecker(str(p)).run()
        assert rep.loaded is False
        dups = [f for f in rep.findings if f.category == "unmarked-duplicate"]
        assert len(dups) == 1
