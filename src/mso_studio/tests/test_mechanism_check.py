"""
Tests for mso_studio.core.mechanism_check -- the PySide6 port of the
Mechanism Health-Check (KRUNAL_JULY_23.md: user asked to convert all
Tkinter GUIs to PySide6). Unlike the converter port, no new headless CLI
was needed -- this shells out to the EXISTING ``rmip check`` CLI
(src/rmip/cli.py::_cmd_check), which already has a full --json contract.
"""
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.mechanism_check import MechanismCheckParams, run_check


def test_params_defaults_output_dir_and_label():
    params = MechanismCheckParams(mechanisms=["/a/128-957.yaml"])
    assert params.output_dir.endswith("outputs/mechanism_check")
    assert params.label == "128-957"


def test_params_label_joins_multiple_mechanisms():
    params = MechanismCheckParams(mechanisms=["/a/m1.yaml", "/a/m2.yaml"])
    assert params.label == "m1+m2"


def test_run_check_builds_expected_argv(monkeypatch, tmp_path):
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        json_path = tmp_path / f"{captured.get('label', 'x')}_check_report.json"
        # simulate rmip_cli.py writing a real JSON report
        import json
        json_path.write_text(json.dumps({
            "mechanism": "/a/m.yaml", "loaded": True, "ran_checks": ["load"],
            "counts": {"error": 0, "warning": 2, "info": 1}, "meta": {}, "findings": [],
        }))
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=0, status="done",
                             tail=[], log_path=None)

    monkeypatch.setattr("mso_studio.core.mechanism_check.run_backend", fake_run_backend)

    params = MechanismCheckParams(
        mechanisms=["/a/m.yaml"], checks=["load", "duplicates"],
        T_grid=[300.0, 1000.0], output_dir=str(tmp_path), label="m",
    )
    captured["label"] = "m"
    record = run_check(params)

    cmd = captured["command"]
    assert cmd[0] == SCIENCE_PYTHON
    assert str(REPO_ROOT / "rmip_cli.py") in cmd
    assert "check" in cmd and "/a/m.yaml" in cmd
    assert "--checks" in cmd and "load" in cmd and "duplicates" in cmd
    assert "--T-grid" in cmd and "300.0" in cmd
    assert record.status == "done"
    assert record.results["n_errors"] == 0
    assert record.results["n_warnings"] == 2


def test_run_check_findings_do_not_count_as_infra_failure(monkeypatch, tmp_path):
    """rmip check exits 1 when a mechanism has errors -- that's a normal,
    informative result, not a crash; status must still be 'done'."""
    def fake_run_backend(command, **kwargs):
        import json
        json_path = tmp_path / "m_check_report.json"
        json_path.write_text(json.dumps({
            "mechanism": "/a/m.yaml", "loaded": False, "ran_checks": ["load"],
            "counts": {"error": 3, "warning": 0, "info": 0}, "meta": {}, "findings": [],
        }))
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=1, status="failed",
                             tail=["some findings printed"], log_path=None)

    monkeypatch.setattr("mso_studio.core.mechanism_check.run_backend", fake_run_backend)
    params = MechanismCheckParams(mechanisms=["/a/m.yaml"], output_dir=str(tmp_path), label="m")
    record = run_check(params)
    assert record.status == "done"
    assert record.results["n_errors"] == 3


def test_run_check_no_json_produced_is_infra_failure(monkeypatch, tmp_path):
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=2, status="failed",
                             tail=["Traceback...", "ImportError: no module"], log_path=None)

    monkeypatch.setattr("mso_studio.core.mechanism_check.run_backend", fake_run_backend)
    params = MechanismCheckParams(mechanisms=["/a/m.yaml"], output_dir=str(tmp_path), label="m")
    record = run_check(params)
    assert record.status == "failed"
    assert record.error


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_run_check_real_konnov_2019_end_to_end(tmp_path):
    """The actual bridge: .venv-gui -> anaconda python3.9 -> rmip_cli.py
    check. Confirms KRUNAL_JULY_23.md point 3's clarity fixes are reachable
    from the report this adapter reads (findings JSON), and that the
    KONNOV_2019 mechanism -- the exact one Krunal reported as confusing --
    produces a real, parseable report through this path."""
    konnov = REPO_ROOT / "MECHANISM" / "KONNOV_2019_AMMONIA_HYDROGEN" / "128-957.inp"
    thermo = REPO_ROOT / "MECHANISM" / "KONNOV_2019_AMMONIA_HYDROGEN" / "128-957.dat"
    if not konnov.exists():
        pytest.skip("KONNOV_2019_AMMONIA_HYDROGEN not present")

    params = MechanismCheckParams(
        mechanisms=[str(konnov)], thermo=str(thermo), output_dir=str(tmp_path),
        checks=["load"],
    )
    record = run_check(params)
    assert record.status == "done"
    assert "reports" in record.results
    assert len(record.results["reports"]) == 1
