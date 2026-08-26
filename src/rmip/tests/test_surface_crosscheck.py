"""
W5b — surface cross-solver validator tests (P21 detector).

The RMG env is not required: the diff engine is exercised through the
offline ``rmg_json`` path (the documented mode for machines without RMG).
- Self-agreement guard: feeding the Cantera side's own values back as the
  "RMG" result must produce zero flags.
- A constructed Keq inconsistency (the W5 plan's acceptance case) must be
  flagged and ranked first.
Live-mode plumbing is covered by error-path tests (missing CHEMKIN input /
missing RMG env give clean, explained errors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rmip.simulation.surface_crosscheck import (        # noqa: E402
    SurfaceCrossChecker, canonical_equation)

MECH = str(Path(__file__).parent / "fixtures" / "char_oxidation.yaml")
CASE = dict(T_K=1400.0, P_atm=1.0, X="O2:0.21, N2:0.79")


def _cantera_result():
    return SurfaceCrossChecker(MECH, **CASE).cantera_side()


def _write_rmg_json(tmp_path, payload) -> str:
    p = tmp_path / "rmg_result.json"
    p.write_text(json.dumps(payload))
    return str(p)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------
def test_canonical_equation_reorders_and_unifies():
    a = canonical_equation("O(s) + C(b) => CO + (s)")
    b = canonical_equation("C(b)+O(s) => (s)  +  CO")
    assert a == b
    # Different arrows stay distinct from reversibility's point of view…
    assert "=>" in a
    # …but a reversible form of the same reaction differs only in the arrow.
    c = canonical_equation("C(b) + O(s) <=> CO + (s)")
    assert c.replace("<=>", "=>") == a


# ---------------------------------------------------------------------------
# Self-agreement regression guard
# ---------------------------------------------------------------------------
def test_cantera_vs_itself_is_consistent(tmp_path):
    ct_res = _cantera_result()
    rmg_json = _write_rmg_json(tmp_path, ct_res)
    report = SurfaceCrossChecker(MECH, **CASE).run(rmg_json=rmg_json)
    assert not report.divergent
    assert len(report.reactions) == 2         # both surface reactions matched
    assert not report.unmatched_cantera and not report.unmatched_rmg
    assert all(abs(r.delta) < 1e-12 for r in report.reactions)
    # Coverages were provided, so they are diffed too — and agree.
    assert report.coverages and all(not c.flagged for c in report.coverages)


# ---------------------------------------------------------------------------
# Constructed Keq inconsistency is flagged (acceptance case)
# ---------------------------------------------------------------------------
def test_constructed_keq_divergence_flagged(tmp_path):
    rmg_res = _cantera_result()
    rmg_res["reactions"][1]["log10_keq"] += 2.0       # factor-100 Keq error
    rmg_json = _write_rmg_json(tmp_path, rmg_res)
    report = SurfaceCrossChecker(MECH, **CASE).run(rmg_json=rmg_json)
    assert report.divergent
    flagged = [r for r in report.reactions if r.flagged]
    assert len(flagged) == 1
    assert flagged[0].delta == pytest.approx(2.0, abs=1e-9)
    # Ranked first by |delta|.
    assert report.ranked_reactions()[0] is flagged[0]


def test_coverage_divergence_flagged(tmp_path):
    rmg_res = _cantera_result()
    names = list(rmg_res["coverages"])
    rmg_res["coverages"][names[0]] += 0.3
    rmg_res["coverages"][names[1]] -= 0.3
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    assert any(c.flagged for c in report.coverages)


def test_below_tolerance_not_flagged(tmp_path):
    rmg_res = _cantera_result()
    rmg_res["reactions"][0]["log10_keq"] += 0.3       # below the 0.5 default
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    assert not any(r.flagged for r in report.reactions)


# ---------------------------------------------------------------------------
# Alignment: unmatched entries surface, species_map translates RMG names
# ---------------------------------------------------------------------------
def test_unmatched_reactions_reported(tmp_path):
    rmg_res = _cantera_result()
    dropped = rmg_res["reactions"].pop(0)
    rmg_res["reactions"].append(
        {"equation": "X(s) + CO => XCO(s)", "log10_keq": 1.0})
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    assert canonical_equation(dropped["equation"]) in [
        canonical_equation(e) for e in report.unmatched_cantera]
    assert report.unmatched_rmg == ["X(s) + CO => XCO(s)"]


def test_species_map_translates_rmg_names(tmp_path):
    ct_res = _cantera_result()
    rmg_res = {
        "reactions": [
            {"equation": r["equation"].replace("O(s)", "OX")
                                      .replace("(s)", "SITE")
                                      .replace("C(b)", "CB"),
             "log10_keq": r["log10_keq"]}
            for r in ct_res["reactions"]],
        "coverages": {},
    }
    # NB replace order above turns O(s)->OX first, so (s)->SITE is safe.
    checker = SurfaceCrossChecker(
        MECH, species_map={"OX": "O(s)", "SITE": "(s)", "CB": "C(b)"}, **CASE)
    report = checker.run(rmg_json=_write_rmg_json(tmp_path, rmg_res))
    assert len(report.reactions) == 2
    assert not report.unmatched_cantera and not report.unmatched_rmg
    assert not any(r.flagged for r in report.reactions)


def test_keq_only_mode_notes_missing_coverages(tmp_path):
    rmg_res = _cantera_result()
    rmg_res["coverages"] = {}
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    assert report.coverages == []
    assert any("no coverages" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Findings + rendering
# ---------------------------------------------------------------------------
def test_to_findings_ranked(tmp_path):
    rmg_res = _cantera_result()
    rmg_res["reactions"][0]["log10_keq"] += 1.0
    rmg_res["reactions"][1]["log10_keq"] -= 3.0
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    findings = report.to_findings()
    keq_findings = [f for f in findings if f.category == "keq-divergence"]
    assert len(keq_findings) == 2
    assert abs(keq_findings[0].details["log10_keq_rmg"]
               - keq_findings[0].details["log10_keq_cantera"]) \
        == pytest.approx(3.0, abs=1e-9)
    assert all(f.severity == "warning" for f in findings)


def test_render_text_and_dict(tmp_path):
    rmg_res = _cantera_result()
    rmg_res["reactions"][1]["log10_keq"] += 2.0
    report = SurfaceCrossChecker(MECH, **CASE).run(
        rmg_json=_write_rmg_json(tmp_path, rmg_res))
    txt = report.render_text()
    assert "DIVERGENT" in txt and "⚠" in txt
    d = report.to_dict()
    assert d["divergent"] is True
    assert d["reactions"][0]["flagged"] is True     # ranked first


# ---------------------------------------------------------------------------
# Live-mode error paths (no RMG env needed — they must fail *cleanly*)
# ---------------------------------------------------------------------------
def test_live_mode_without_chemkin_input_is_explained():
    with pytest.raises(RuntimeError, match="CHEMKIN-format"):
        SurfaceCrossChecker(MECH, **CASE).run()


def test_live_mode_with_missing_rmg_env_is_explained(tmp_path):
    fake = tmp_path / "chem.inp"
    fake.write_text("! placeholder\n")
    checker = SurfaceCrossChecker(
        MECH, rmg_chemkin_gas=str(fake), rmg_chemkin_surf=str(fake),
        rmg_python=str(tmp_path / "nonexistent" / "python"), **CASE)
    with pytest.raises(RuntimeError, match="RMG interpreter not found"):
        checker.run()
