"""
W7 — RMG explainability / supervisor / linter / reproducibility-diff tests.

Per the W7 plan, no live RMG is needed anywhere here:
- W7a explainer runs on a recorded RMG.log fixture whose lines follow the
  *verbatim* format strings from rmgpy (solver/base.pyx, rmg/model.py);
- W7b supervisor is driven with fake subprocesses via command_template;
- W7c linter is static (ast);
- W7d diff runs on two recorded chemkin/species_dictionary fixtures.
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rmip.generation.rmg_explainer import RmgRunExplainer      # noqa: E402
from rmip.generation.rmg_lint import lint_rmg_input            # noqa: E402
from rmip.generation.rmg_diff import (                         # noqa: E402
    canonical_graph_key, reproducibility_diff)
from rmip.generation.rmg_supervisor import RmgSupervisor       # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
RUN = str(FIXTURES / "rmg_run")
DIFF_A = str(FIXTURES / "rmg_diff_a")
DIFF_B = str(FIXTURES / "rmg_diff_b")
INPUT_DUAL = str(FIXTURES / "rmg_input_dual.py")


# ===========================================================================
# W7a — explainer (P17 / P26)
# ===========================================================================
@pytest.fixture(scope="module")
def explanation():
    return RmgRunExplainer(RUN).explain()


def test_every_core_species_attributed(explanation):
    by_species = {p.species: p for p in explanation.promotions}
    # Flux promotions carry time, ratio and tolerance.
    p = by_species["C2H5(3)"]
    assert p.mechanism == "flux" and p.iteration == 1
    assert p.rate_ratio == pytest.approx(4.82)
    assert p.tolerance == pytest.approx(0.1)
    assert p.time_s == pytest.approx(2.1e-4)
    assert by_species["CH3(4)"].iteration == 2
    assert by_species["C2H4(5)"].mechanism == "flux"
    # The two non-flux promotion mechanisms are distinguished.
    assert by_species["H(6)"].mechanism == "singularity"
    assert by_species["C3H7(9)"].mechanism == "resurrection"
    assert by_species["C3H7(9)"].rate_ratio == pytest.approx(0.42)
    # Initial/seed species are attributed as such, never flux-ranked first.
    assert by_species["ethane(1)"].mechanism == "seed/initial"
    assert "never flux-promoted" in by_species["ethane(1)"].why()


def test_promotions_ranked_by_flux(explanation):
    flux_ranked = [p for p in explanation.promotions
                   if p.mechanism != "seed/initial"]
    assert flux_ranked[0].species == "C2H5(3)"          # 4.82 is the largest
    ratios = [p.rate_ratio or 0 for p in flux_ranked]
    assert ratios == sorted(ratios, reverse=True)
    # seed/initial species trail the ranking.
    assert explanation.promotions[-1].mechanism == "seed/initial"


def test_edge_inspection(explanation):
    edge = {e.species: e for e in explanation.edge}
    # A species that got close (interruption line) but never promoted.
    assert edge["C4H10(8)"].fate == "remained-edge"
    assert edge["C4H10(8)"].max_rate_ratio == pytest.approx(0.18)
    # Pruned and forbidden fates.
    assert edge["C4H9(10)"].fate == "pruned"
    assert edge["CH2(S)(11)"].fate == "forbidden"
    # Edge chemkin species with no measurable flux are merged in (P26).
    assert edge["CH4(12)"].max_rate_ratio is None
    assert edge["C2H3(13)"].fate == "remained-edge"
    # CH3(4) had an interruption sighting in iteration 1 but was later
    # promoted — it must NOT appear as edge.
    assert "CH3(4)" not in edge


def test_run_metadata(explanation):
    assert explanation.iterations == 3
    assert explanation.completed
    assert explanation.final_core == {"species": 8, "reactions": 20}
    assert explanation.final_edge == {"species": 12, "reactions": 30}


def test_report_rendering(explanation):
    txt = explanation.render_text()
    assert "Why each core species is here" in txt
    assert "C2H5(3)" in txt and "rate ratio 4.82" in txt
    assert "considered but rejected" in txt
    d = explanation.to_dict()
    assert json.dumps(d)                    # JSON-serializable
    assert d["promotions"][0]["why"]


def test_missing_log_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="RMG.log"):
        RmgRunExplainer(str(tmp_path))


# ===========================================================================
# W7c — input linter (P39)
# ===========================================================================
def test_dual_listed_library_flagged():
    findings = lint_rmg_input(INPUT_DUAL)
    dual = [f for f in findings if f.category == "dual-listed-library"]
    assert len(dual) == 1
    assert dual[0].details["library"] == "GRI-Mech3.0"
    assert dual[0].severity == "error"
    assert "#2979" in dual[0].message


def test_duplicate_within_list_flagged():
    findings = lint_rmg_input(INPUT_DUAL)
    dups = [f for f in findings if f.category == "duplicate-library"]
    assert len(dups) == 1
    assert dups[0].details == {"library": "BurkeH2O2",
                               "list": "thermoLibraries", "count": 2}


def test_inverted_tolerances_flagged():
    findings = lint_rmg_input(INPUT_DUAL)
    tol = [f for f in findings if f.category == "tolerance-order"]
    assert len(tol) == 1
    assert tol[0].details["toleranceMoveToCore"] == 0.5


def test_clean_input_passes(tmp_path):
    clean = tmp_path / "input.py"
    clean.write_text(textwrap.dedent("""
        database(
            thermoLibraries=['primaryThermoLibrary'],
            reactionLibraries=[('BurkeH2O2inN2', False)],
            seedMechanisms=['GRI-Mech3.0'],
            kineticsDepositories=['training'],
        )
        model(toleranceMoveToCore=0.1, toleranceInterruptSimulation=0.5)
    """))
    assert lint_rmg_input(str(clean)) == []


def test_syntax_error_reported(tmp_path):
    bad = tmp_path / "input.py"
    bad.write_text("database(thermoLibraries=[")
    findings = lint_rmg_input(str(bad))
    assert findings[0].category == "syntax-error"
    assert findings[0].severity == "error"


def test_dynamic_list_reported_unlintable(tmp_path):
    dyn = tmp_path / "input.py"
    dyn.write_text(textwrap.dedent("""
        libs = ['a'] * 2
        database(thermoLibraries=libs)
        model(toleranceMoveToCore=0.1)
    """))
    findings = lint_rmg_input(str(dyn))
    assert any(f.category == "unlintable" for f in findings)


# ===========================================================================
# W7d — reproducibility diff (P15)
# ===========================================================================
def test_graph_key_invariant_under_renumbering():
    a = "H2O(4)\n1 O u0 p2 c0 {2,S} {3,S}\n2 H u0 p0 c0 {1,S}\n3 H u0 p0 c0 {1,S}"
    b = "X(99)\n1 H u0 p0 c0 {3,S}\n2 H u0 p0 c0 {3,S}\n3 O u0 p2 c0 {1,S} {2,S}"
    assert canonical_graph_key(a) == canonical_graph_key(b)


def test_graph_key_distinguishes_radical_state():
    oh = "OH\nmultiplicity 2\n1 O u1 p2 c0 {2,S}\n2 H u0 p0 c0 {1,S}"
    oh_anion_like = "OH2\n1 O u0 p2 c0 {2,S}\n2 H u0 p0 c0 {1,S}"
    assert canonical_graph_key(oh) != canonical_graph_key(oh_anion_like)


@pytest.fixture(scope="module")
def diff():
    return reproducibility_diff(DIFF_A, DIFF_B)


def test_species_alignment_across_renaming(diff):
    # OH(3) vs OH(7): same structure, different label — matched as renamed.
    assert ("OH(3)", "OH(7)") in diff.renamed
    # H2O has renumbered atoms in B — still matched (not renamed).
    assert diff.common_species == 6          # H2, O2, OH, H2O, H, AR
    assert diff.only_in_a == ["HO2(5)"]
    assert diff.only_in_b == ["H2O2(9)"]


def test_reaction_alignment_and_rate_change(diff):
    assert diff.common_reactions == 2
    assert len(diff.rate_changes) == 1
    rc = diff.rate_changes[0]
    assert "H2O(4)" in rc.equation_a
    # A: 2e13 -> 3e13
    assert rc.dlog10_A == pytest.approx(0.176, abs=1e-3)
    assert [r for r in diff.reactions_only_in_a] == \
        ["H(6)+O2(2)+M=HO2(5)+M"]
    assert [r for r in diff.reactions_only_in_b] == \
        ["OH(7)+OH(7)=H2O2(9)"]
    assert not diff.identical


def test_diff_self_is_identical():
    d = reproducibility_diff(DIFF_A, DIFF_A)
    assert d.identical
    assert d.renamed == []
    assert d.common_species == 7


def test_diff_rendering(diff):
    txt = diff.render_text()
    assert "DIVERGED" in txt
    assert "OH(3)  ->  OH(7)" in txt
    assert json.dumps(diff.to_dict())


# ===========================================================================
# W7b — supervisor (P30 / P16); fake subprocesses only
# ===========================================================================
def _fake_input(tmp_path, tol=0.1):
    p = tmp_path / "input.py"
    p.write_text(f"model(toleranceMoveToCore={tol})\n")
    return str(p)


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("import sys, os, time\n" + textwrap.dedent(body))
    return str(p)


def test_supervisor_completed(tmp_path):
    script = _script(tmp_path, "ok.py", """
        out = sys.argv[1]
        with open(os.path.join(out, "RMG.log"), "w") as fh:
            fh.write("    The model core has 3 species and 2 reactions\\n")
            fh.write("    The model core has 5 species and 7 reactions\\n")
            fh.write("MODEL GENERATION COMPLETED\\n")
    """)
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{output}"],
        poll_interval_s=0.1)
    report = sup.run(_fake_input(tmp_path), str(tmp_path / "out"))
    assert report.ok and report.outcome == "completed"
    assert report.iterations_completed == 2
    assert (tmp_path / "out" / "supervisor_report.json").is_file()
    assert (tmp_path / "out" / "supervisor_report.txt").is_file()


def test_supervisor_wall_timeout_kills_and_reports(tmp_path):
    script = _script(tmp_path, "hang.py", """
        out = sys.argv[1]
        # keep writing so stall detection never fires — only the wall clock
        while True:
            with open(os.path.join(out, "RMG.log"), "a") as fh:
                fh.write("tick\\n")
            time.sleep(0.05)
    """)
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{output}"],
        wall_timeout_s=1.0, poll_interval_s=0.1)
    t0 = time.monotonic()
    report = sup.run(_fake_input(tmp_path), str(tmp_path / "out"))
    assert report.outcome == "wall-timeout"
    assert time.monotonic() - t0 < 15.0          # actually killed, promptly
    assert "RMG.log" in " ".join(report.partial_outputs)


def test_supervisor_stall_detection(tmp_path):
    script = _script(tmp_path, "stall.py", """
        out = sys.argv[1]
        with open(os.path.join(out, "RMG.log"), "w") as fh:
            fh.write("    The model core has 3 species and 2 reactions\\n")
        time.sleep(600)     # alive but going nowhere: the P30 signature
    """)
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{output}"],
        stall_timeout_s=1.0, poll_interval_s=0.1)
    report = sup.run(_fake_input(tmp_path), str(tmp_path / "out"))
    assert report.outcome == "stalled"
    assert "P30" in report.reason
    assert report.iterations_completed == 1      # partial progress captured


def test_supervisor_memory_budget(tmp_path):
    script = _script(tmp_path, "hog.py", """
        out = sys.argv[1]
        with open(os.path.join(out, "RMG.log"), "w") as fh:
            fh.write("allocating\\n")
        ballast = bytearray(400 * 1024 * 1024)   # ~400 MB RSS
        ballast[::4096] = b"x" * len(ballast[::4096])
        time.sleep(600)
    """)
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{output}"],
        mem_budget_mb=200.0, poll_interval_s=0.2)
    report = sup.run(_fake_input(tmp_path), str(tmp_path / "out"))
    assert report.outcome == "memory-budget"
    assert report.peak_rss_mb > 200.0
    assert "P16" in report.reason


def test_supervisor_tolerance_relaxation_recovers(tmp_path):
    """Stage 1 stalls at the tight tolerance; the supervisor rewrites
    toleranceMoveToCore x10 and stage 2 completes."""
    script = _script(tmp_path, "tol.py", """
        inp, out = sys.argv[1], sys.argv[2]
        tol = float(open(inp).read().split("toleranceMoveToCore=")[1]
                    .split(")")[0])
        log = os.path.join(out, "RMG.log")
        if tol < 0.5:
            open(log, "w").write("starting\\n")
            time.sleep(600)                      # stall
        else:
            with open(log, "w") as fh:
                fh.write("    The model core has 9 species and 9 reactions\\n")
                fh.write("MODEL GENERATION COMPLETED\\n")
    """)
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{input}", "{output}"],
        stall_timeout_s=1.0, poll_interval_s=0.1,
        relax_on_stall=True, relax_factor=10.0, max_relaxations=2)
    report = sup.run(_fake_input(tmp_path, tol=0.1), str(tmp_path / "out"))
    assert report.outcome == "completed"
    assert len(report.stages) == 2
    assert report.stages[0].outcome == "stalled"
    assert report.stages[1].tolerance_move_to_core == pytest.approx(1.0)


def test_supervisor_staged_batch_isolates_failures(tmp_path):
    """One pathological stage cannot sink the batch (staged conditions)."""
    script = _script(tmp_path, "mixed.py", """
        inp, out = sys.argv[1], sys.argv[2]
        log = os.path.join(out, "RMG.log")
        if "bad" in inp:
            open(log, "w").write("starting\\n")
            time.sleep(600)
        else:
            with open(log, "w") as fh:
                fh.write("MODEL GENERATION COMPLETED\\n")
    """)
    good1 = tmp_path / "cond_1.py"; good1.write_text("model(toleranceMoveToCore=0.1)\n")
    bad = tmp_path / "cond_bad.py"; bad.write_text("model(toleranceMoveToCore=0.1)\n")
    good2 = tmp_path / "cond_3.py"; good2.write_text("model(toleranceMoveToCore=0.1)\n")
    sup = RmgSupervisor(
        command_template=[sys.executable, script, "{input}", "{output}"],
        stall_timeout_s=1.0, poll_interval_s=0.1)
    reports = sup.run_staged([str(good1), str(bad), str(good2)],
                             str(tmp_path / "batch"))
    assert [r.outcome for r in reports] == \
        ["completed", "stalled", "completed"]


def test_supervisor_missing_rmg_env_is_explained(tmp_path):
    sup = RmgSupervisor(rmg_python=str(tmp_path / "no" / "python"))
    with pytest.raises(RuntimeError, match="RMG interpreter not found"):
        sup.run(_fake_input(tmp_path), str(tmp_path / "out"))
