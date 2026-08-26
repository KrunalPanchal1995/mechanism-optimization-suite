"""
Tests for mso_studio.core.fusion -- the PySide6 port of Mechanism Fusion
(KRUNAL_JULY_23.md: PySide6 conversion). Covers ``_parse_summary`` (the
plain-text -> dict bridge), command construction, and a real end-to-end
fusion through the actual two-interpreter bridge.
"""
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.fusion import FusionParams, run_fusion, _parse_summary


def test_parse_summary_extracts_all_fields():
    lines = [
        "some noise before it", "fused_mech_id=19", "n_species=14",
        "n_verbatim=3 n_copy=8 n_fuse=0", "balanced=True",
        "glossary_backfill mech=2 species_resolved=0 reactions_resolved=0",
        "output=/tmp/out/Fused.inp",
    ]
    summary = _parse_summary(lines)
    assert summary == {
        "fused_mech_id": 19, "n_species": 14, "n_verbatim": 3, "n_copy": 8,
        "n_fuse": 0, "balanced": True, "output": "/tmp/out/Fused.inp",
    }


def test_parse_summary_empty_on_no_match():
    assert _parse_summary(["random log line", "another one"]) == {}


def test_fusion_params_defaults():
    params = FusionParams(mechanism_ids=[2, 3, 4], db_path="/a/project.db")
    assert params.output_dir.endswith("outputs/fusion")
    assert params.label == "Fused_Mech"


def test_run_fusion_builds_expected_argv(monkeypatch, tmp_path):
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        from mso_studio.core.backends import BackendResult
        return BackendResult(
            command=list(command), return_code=0, status="done",
            tail=["fused_mech_id=7", "n_species=10",
                  "n_verbatim=2 n_copy=3 n_fuse=1", "balanced=True",
                  f"output={tmp_path}/Fused_Mech.inp"],
            log_path=None,
        )

    monkeypatch.setattr("mso_studio.core.fusion.run_backend", fake_run_backend)

    params = FusionParams(
        mechanism_ids=[2, 3, 4], db_path="/a/project.db",
        output_dir=str(tmp_path), priority_order=[2, 3, 4], glossary_path="",
    )
    record = run_fusion(params)

    cmd = captured["command"]
    assert cmd[0] == SCIENCE_PYTHON
    assert str(REPO_ROOT / "mean_value_fusion_gui.py") in cmd
    assert "--fuse" in cmd
    assert "--mechs" in cmd and "2" in cmd and "3" in cmd and "4" in cmd
    assert "--priority-order" in cmd
    assert record.status == "done"
    assert record.results["fused_mech_id"] == 7
    assert record.results["n_species"] == 10


def test_run_fusion_reports_degenerate_error_without_raising(monkeypatch, tmp_path):
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(
            command=list(command), return_code=1, status="failed",
            tail=["ERROR: Fusion of mechanisms [2, 3] produced an empty union"],
            log_path=None,
        )

    monkeypatch.setattr("mso_studio.core.fusion.run_backend", fake_run_backend)
    params = FusionParams(mechanism_ids=[2, 3], db_path="/a/project.db", output_dir=str(tmp_path))
    record = run_fusion(params)   # must NOT raise
    assert record.status == "failed"
    assert "empty union" in record.error


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_run_fusion_real_h2_fixture_end_to_end(tmp_path):
    """The actual bridge: .venv-gui -> anaconda python3.9 ->
    mean_value_fusion_gui.py --fuse, against the same H2-combustion DB
    fixture the Tk app's own test suite uses (test_stubs_fusion.py)."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "src" / "rmip" / "tests"))
    from test_stubs_fusion import _build_h2_fixture  # local test helper

    db, mid1, mid2 = _build_h2_fixture(tmp_path)
    params = FusionParams(
        mechanism_ids=[mid1, mid2], db_path=db, output_dir=str(tmp_path / "out"),
        output_name="RealBridgeFused", glossary_path="",
    )
    lines = []
    record = run_fusion(params, on_line=lines.append)

    assert record.status == "done"
    assert record.results["n_species"] == 6
    assert (Path(tmp_path) / "out" / "RealBridgeFused.inp").exists()

    # KRUNAL_JULY_23.md point 6 -- the per-reaction uncertainty CSV path
    # must survive the real subprocess bridge + _parse_summary, not just
    # exist in the anaconda-side dict (mean_value_fusion_gui.py's own
    # test suite already covers _run_fusion() directly).
    assert record.results.get("n_uncertainty_reactions", 0) >= 1
    csv_path = record.results.get("uncertainty_csv_path")
    assert csv_path and Path(csv_path).exists()
