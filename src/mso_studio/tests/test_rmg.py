"""
Tests for mso_studio.core.rmg -- the RMG-Py subprocess adapter (2026-08
workbench audit, helper-package gap #4). No Qt import here (core/ rule),
and no test requires rmg_env/RMG-Py to actually be installed -- every path
must degrade to a clear, non-raising (available, reason) instead.
"""
from mso_studio.core import rmg as rmg_core


def test_rmg_available_never_raises():
    available, reason = rmg_core.rmg_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str) and reason


def test_default_rmg_python_env_override(monkeypatch, tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv("MSO_STUDIO_RMG_PYTHON", str(fake_python))
    assert rmg_core._default_rmg_python() == str(fake_python)


def test_default_rmg_python_missing_override_falls_through(monkeypatch):
    monkeypatch.setenv("MSO_STUDIO_RMG_PYTHON", "/no/such/interpreter")
    # Falls through to the real candidate search (may or may not find one
    # on this machine) -- must not raise or return the bogus override path.
    result = rmg_core._default_rmg_python()
    assert result != "/no/such/interpreter"


def test_default_rmg_script_env_override(monkeypatch, tmp_path):
    fake_script = tmp_path / "rmg.py"
    fake_script.write_text("# fake\n")
    monkeypatch.setenv("MSO_STUDIO_RMG_SCRIPT", str(fake_script))
    assert rmg_core._default_rmg_script() == str(fake_script)


def test_rmg_available_false_when_no_python_found(monkeypatch):
    monkeypatch.setattr(rmg_core, "_default_rmg_python", lambda: None)
    available, reason = rmg_core.rmg_available()
    assert available is False
    assert "rmg_env" in reason


def test_rmg_available_false_when_python_found_but_no_script(monkeypatch):
    monkeypatch.setattr(rmg_core, "_default_rmg_python", lambda: "/usr/bin/python3")
    monkeypatch.setattr(rmg_core, "_default_rmg_script", lambda: None)
    available, reason = rmg_core.rmg_available()
    assert available is False
    assert "rmg.py" in reason


def test_run_rmg_job_raises_cleanly_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(rmg_core, "_default_rmg_python", lambda: None)
    input_py = tmp_path / "input.py"
    input_py.write_text("# fake rmg input\n")
    try:
        rmg_core.run_rmg_job(str(input_py))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "not available" in str(exc)


def test_import_rmg_mechanism_missing_dir_returns_all_none(tmp_path):
    result = rmg_core.import_rmg_mechanism(str(tmp_path / "no_such_output"))
    assert result == {"mechanism": None, "species_dictionary": None, "cantera_yaml": None}


def test_import_rmg_mechanism_finds_real_files(tmp_path):
    chem_dir = tmp_path / "chemkin"
    chem_dir.mkdir()
    (chem_dir / "chem_annotated.inp").write_text("! fake mechanism\n")
    (chem_dir / "species_dictionary.txt").write_text("! fake species dict\n")
    result = rmg_core.import_rmg_mechanism(str(tmp_path))
    assert result["mechanism"] == str(chem_dir / "chem_annotated.inp")
    assert result["species_dictionary"] == str(chem_dir / "species_dictionary.txt")
    assert result["cantera_yaml"] is None


def test_rmgpy_version_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(rmg_core, "_default_rmg_python", lambda: None)
    assert rmg_core.rmgpy_version() is None


def test_run_rmg_job_reports_failed_status_not_a_raised_exception_when_subprocess_crashes(
    monkeypatch, tmp_path,
):
    """Round 4 real-environment stress test (WORKBENCH_AUDIT_2026-08-10.md):
    a real end-to-end run_rmg_job() against this machine's actual rmg_env
    failed immediately -- rmgpy.rmg.main's own `from cantera import
    ck2yaml` raised ImportError inside the rmg_env subprocess itself
    (GLIBCXX_3.4.32 not found; the env's cantera wheel needs a newer
    libstdc++ than this system's GLIBCXX_3.4.28 -- an rmg_env conda
    environment/ABI issue, not a bug in this adapter or the repo). The
    adapter's OWN behavior under that real failure is what matters here and
    is correct: status='failed', every path field None, nothing raised --
    confirmed live. This test pins that contract with a synthetic subprocess
    failure (any nonzero-exit interpreter) so a regression here is caught
    without depending on the rmg_env environment itself being installed."""
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text("#!/bin/sh\necho 'boom: some real subprocess failure' >&2\nexit 1\n")
    fake_python.chmod(0o755)
    monkeypatch.setattr(rmg_core, "_default_rmg_python", lambda: str(fake_python))
    monkeypatch.setattr(rmg_core, "_default_rmg_script", lambda: "unused.py")

    input_py = tmp_path / "input.py"
    input_py.write_text("# fake rmg input\n")
    out_dir = tmp_path / "out"

    result = rmg_core.run_rmg_job(str(input_py), out_dir=str(out_dir))

    assert result.status == "failed"
    assert result.chem_annotated_path is None
    assert result.species_dictionary_path is None
    assert result.cantera_yaml_path is None
    assert any("boom" in line for line in result.log_tail)
