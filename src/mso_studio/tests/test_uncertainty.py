"""
Tests for mso_studio.core.uncertainty -- the PySide6 port of the
Uncertainty Generator (KRUNAL_JULY_23.md: PySide6 conversion). Covers both
headless CLI modes (--generate-xml, --compute-bounds) this shells out to.
"""
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.uncertainty import (
    GenerateXmlParams, ComputeBoundsParams, run_generate_xml, run_compute_bounds,
)

_FIXTURES = REPO_ROOT / "src" / "rmip" / "tests" / "fixtures" / "gui_stress"


def test_run_generate_xml_builds_expected_argv(monkeypatch, tmp_path):
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=0, status="done",
                             tail=[], log_path=None)

    monkeypatch.setattr("mso_studio.core.uncertainty.run_backend", fake_run_backend)

    params = GenerateXmlParams(
        reactions=[{"name": "H+O2<=>O+OH", "type": "pressure_independent",
                   "temperatures": [500, 2500], "uncertainties": [0.3, 0.3]}],
        output_path=str(tmp_path / "out.xml"),
    )
    record = run_generate_xml(params)

    cmd = captured["command"]
    assert cmd[0] == SCIENCE_PYTHON
    assert str(REPO_ROOT / "uncertainty_generator_gui.py") in cmd
    assert "--generate-xml" in cmd
    assert "--reactions-json" in cmd and "--output" in cmd
    assert record.status == "done"
    assert record.results["output_path"] == str(tmp_path / "out.xml")


def test_run_compute_bounds_parses_ok_and_error_lines(monkeypatch, tmp_path):
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        out_csv = tmp_path / "bounds.csv"
        out_csv.write_text("rxn,A_nominal\n")
        return BackendResult(
            command=list(command), return_code=1, status="failed",
            tail=["OK H+O2<=>O+OH: A_nominal=9.841e+13 factor=1.49182 "
                  "A_lb=6.59662e+13 A_ub=1.4681e+14",
                  "ERROR X+Y<=>Z: 'no'"],
            log_path=None,
        )

    monkeypatch.setattr("mso_studio.core.uncertainty.run_backend", fake_run_backend)
    params = ComputeBoundsParams(xml_path="/a/u.xml", mechanism="/a/m.yaml",
                                 output_csv=str(tmp_path / "bounds.csv"))
    record = run_compute_bounds(params)   # must NOT raise despite return_code=1

    assert record.status == "done"        # partial success is still "done"
    assert record.results["n_ok"] == 1
    assert record.results["n_err"] == 1
    rows = record.results["rows"]
    assert rows[0]["rxn"] == "H+O2<=>O+OH"
    assert rows[0]["A_nominal"] == pytest.approx(9.841e13)
    assert rows[1]["error"] == "'no'"


def test_run_compute_bounds_no_output_is_infra_failure(monkeypatch, tmp_path):
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=2, status="failed",
                             tail=["Traceback...", "ModuleNotFoundError"], log_path=None)

    monkeypatch.setattr("mso_studio.core.uncertainty.run_backend", fake_run_backend)
    params = ComputeBoundsParams(xml_path="/a/u.xml", mechanism="/a/m.yaml",
                                 output_csv=str(tmp_path / "bounds.csv"))
    record = run_compute_bounds(params)
    assert record.status == "failed"
    assert record.error


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_real_generate_then_compute_bounds_end_to_end(tmp_path):
    """The actual bridge: .venv-gui -> anaconda python3.9 -> both headless
    CLI modes, chained exactly as a user would (generate an XML, then
    collapse it), against the known-good mech_std.yaml fixture (verified
    A_nominal=9.841E+13, same fixture test_gui_stress_sweep.py's
    test_uncertainty_generator_roundtrip uses)."""
    if not _FIXTURES.joinpath("mech_std.yaml").exists():
        pytest.skip("gui_stress fixtures not present")

    xml_path = tmp_path / "generated.xml"
    gen_params = GenerateXmlParams(
        reactions=[{"name": "H + O2 <=> O + OH", "type": "pressure_independent",
                   "perturbation_type": "factor", "data_mode": "constant;array",
                   "temperatures": [500.0, 1500.0, 2500.0],
                   "uncertainties": [0.40, 0.25, 0.15]}],
        output_path=str(xml_path),
    )
    gen_record = run_generate_xml(gen_params)
    assert gen_record.status == "done"
    assert xml_path.is_file()

    bounds_params = ComputeBoundsParams(
        xml_path=str(xml_path), mechanism=str(_FIXTURES / "mech_std.yaml"),
        output_csv=str(tmp_path / "bounds.csv"),
    )
    bounds_record = run_compute_bounds(bounds_params)
    assert bounds_record.status == "done"
    assert bounds_record.results["n_ok"] == 1
    assert bounds_record.results["n_err"] == 0
    row = bounds_record.results["rows"][0]
    assert row["A_nominal"] == pytest.approx(9.841e13, rel=1e-3)
    assert row["A_lb"] < row["A_nominal"] < row["A_ub"]
