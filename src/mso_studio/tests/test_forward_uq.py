"""
Tests for mso_studio.core.forward_uq -- the PySide6 port of Forward UQ
(KRUNAL_JULY_23.md: PySide6 conversion). ``forward_uq.py`` already has a
full argparse CLI, so no new headless mode was needed here -- unlike the
converter/health-check/fusion/uncertainty-generator ports.
"""
import sys
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON, BackendError
from mso_studio.core.forward_uq import ForwardUqParams, run_forward_uq


def test_params_defaults_output_dir_and_label():
    params = ForwardUqParams(target_opt="/a/case1.opt")
    assert params.output_dir.endswith("outputs/forward_uq")
    assert params.label == "case1"


def test_run_forward_uq_builds_expected_argv(monkeypatch, tmp_path):
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=0, status="done",
                             tail=["Forward UQ complete"], log_path=None)

    monkeypatch.setattr("mso_studio.core.forward_uq.run_backend", fake_run_backend)

    target_opt = tmp_path / "case1.opt"
    target_opt.write_text("Locations: {}\n")
    params = ForwardUqParams(
        target_opt=str(target_opt), n_samples=1000, method="lhs", seed=7,
        output_dir=str(tmp_path / "out"),
    )
    record = run_forward_uq(params)

    cmd = captured["command"]
    assert cmd[0] == SCIENCE_PYTHON
    assert str(REPO_ROOT / "forward_uq.py") in cmd
    assert str(target_opt) in cmd
    assert "--n_samples" in cmd and "1000" in cmd
    assert "--method" in cmd and "lhs" in cmd
    assert "--seed" in cmd and "7" in cmd
    assert "--output_dir" in cmd
    assert record.status == "done"


def test_run_forward_uq_raises_backend_error_on_failure(monkeypatch, tmp_path):
    """Unlike converter/check/fusion, forward UQ has no meaningful partial-
    success concept -- a non-zero exit is an infrastructure-style failure
    and must raise (matching core.nominal.run_nominal's own pattern)."""
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=1, status="failed",
                             tail=["FileNotFoundError: no PRS cache"], log_path=None)

    monkeypatch.setattr("mso_studio.core.forward_uq.run_backend", fake_run_backend)
    target_opt = tmp_path / "case1.opt"
    target_opt.write_text("Locations: {}\n")
    params = ForwardUqParams(target_opt=str(target_opt), output_dir=str(tmp_path / "out"))
    with pytest.raises(BackendError):
        run_forward_uq(params)


def test_run_forward_uq_collects_statistics_csv(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stats = out_dir / "forward_uq_statistics.csv"
    stats.write_text("case_id,eta_mean\ncase-0,1.2\n")

    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=0, status="done",
                             tail=["Forward UQ complete"], log_path=None)

    monkeypatch.setattr("mso_studio.core.forward_uq.run_backend", fake_run_backend)
    target_opt = tmp_path / "case1.opt"
    target_opt.write_text("Locations: {}\n")
    params = ForwardUqParams(target_opt=str(target_opt), output_dir=str(out_dir))
    record = run_forward_uq(params)

    assert record.results["statistics_path"] == str(stats)
    assert record.tables["forward_uq_statistics"] == str(stats)


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_run_forward_uq_real_bridge_with_synthetic_prs_cache(tmp_path):
    """The actual bridge: .venv-gui -> anaconda python3.9 -> forward_uq.py,
    against the same synthetic PRS-cache fixture
    test_gui_stress_sweep.py::test_forward_uq_full_run_with_synthetic_prs_cache
    uses, reused here rather than re-derived."""
    fixtures_dir_src = REPO_ROOT / "src" / "rmip" / "tests" / "fixtures" / "gui_stress"
    unsrt_xml = REPO_ROOT / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"
    if not fixtures_dir_src.exists() or not unsrt_xml.exists():
        pytest.skip("gui_stress fixtures not present")

    import shutil
    fixtures_dir = tmp_path / "fixtures_copy"
    shutil.copytree(fixtures_dir_src, fixtures_dir)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "MUQ-SAC"))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(fixtures_dir_src))
    import importlib
    bootstrap = importlib.import_module("bootstrap_forward_uq_cache")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    text = (fixtures_dir / "target.opt").read_text().replace(
        str(fixtures_dir_src), str(fixtures_dir))
    target_opt = work_dir / "target.opt"
    target_opt.write_text(text)
    bootstrap.build(work_dir, str(unsrt_xml), str(fixtures_dir / "mech_std.yaml"), n_cases=2)

    # forward_uq.py resolves Opt/ as os.path.join(os.getcwd(), "Opt") --
    # this adapter (like the original Tk GUI's _run_uq()) launches with
    # cwd=output_dir, so output_dir must be the SAME directory the PRS
    # cache (Opt/) was bootstrapped into, not a fresh empty subdirectory --
    # forward_uq's own outputs (forward_uq_statistics.csv, results/*.pdf)
    # get added alongside the existing Opt/ cache, not into an isolated dir.
    params = ForwardUqParams(
        target_opt=str(target_opt), n_samples=512, output_dir=str(work_dir),
    )
    record = run_forward_uq(params)

    assert record.status == "done"
    assert "statistics_path" in record.results
