"""
Tests for mso_studio.core.converter -- the PySide6 port of the CHEMKIN
converter (KRUNAL_JULY_23.md: the user asked to convert all Tkinter GUIs to
PySide6, in addition to points 1-8). Covers:

* ConverterParams defaults (fast, no subprocess).
* run_convert()'s command construction via a monkeypatched run_backend (fast,
  no real subprocess -- proves the argv shape without needing the anaconda
  interpreter or a real mechanism).
* A REAL end-to-end conversion of the KONNOV_2019 mechanism through the
  actual two-interpreter bridge (.venv-gui -> anaconda python3.9 ->
  mechanism_converter_gui.py --convert), skipped if the anaconda interpreter
  isn't present on this machine. This is the single most important check:
  it proves the exact bridge Krunal's original bug report exercised (a
  CHEMKIN conversion invoked from a GUI) now genuinely produces a fused
  YAML with the species-corruption bug (point 2) fixed, from the PySide6
  side too.
"""
import os
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.converter import ConverterParams, run_convert


def test_converter_params_defaults_output_dir_and_label():
    params = ConverterParams(mechanism="/some/dir/128-957.inp")
    assert params.output_dir.endswith("outputs/converted")
    assert params.label == "128-957"


def test_converter_params_explicit_values_not_overridden():
    params = ConverterParams(mechanism="/a/mech.inp", output_dir="/custom/out",
                              label="my_label")
    assert params.output_dir == "/custom/out"
    assert params.label == "my_label"


def test_run_convert_builds_expected_argv(monkeypatch, tmp_path):
    """Fast, no real subprocess -- monkeypatch run_backend to capture the
    argv mso_studio.core.converter builds for it."""
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=0, status="done",
                             tail=[], log_path=None)

    monkeypatch.setattr("mso_studio.core.converter.run_backend", fake_run_backend)

    params = ConverterParams(
        mechanism="/m/128-957.inp", thermo="/m/128-957.dat",
        output_dir=str(tmp_path), make_yaml=True, make_pre=False,
        permissive=True, fix_exp=True, fix_crlf=True,
    )
    record = run_convert(params)

    cmd = captured["command"]
    assert cmd[0] == SCIENCE_PYTHON
    assert str(REPO_ROOT / "mechanism_converter_gui.py") in cmd
    assert "--convert" in cmd
    assert "--input" in cmd and "/m/128-957.inp" in cmd
    assert "--thermo" in cmd and "/m/128-957.dat" in cmd
    assert "--make-yaml" in cmd
    assert "--permissive" in cmd
    assert "--fix-exp" in cmd and "--fix-crlf" in cmd
    assert record.status == "done"
    assert record.config.module == "mechanism_converter"

    # explicit no_-flags map to the negative CLI switch
    params2 = ConverterParams(mechanism="/m/x.inp", output_dir=str(tmp_path),
                              fix_exp=False, make_yaml=False)
    run_convert(params2)
    cmd2 = captured["command"]
    assert "--no-fix-exp" in cmd2
    assert "--no-make-yaml" in cmd2


def test_run_convert_reports_failure_without_raising(monkeypatch, tmp_path):
    def fake_run_backend(command, **kwargs):
        from mso_studio.core.backends import BackendResult
        return BackendResult(
            command=list(command), return_code=1, status="failed",
            tail=["some preflight output",
                  'ERROR: ck2yaml failed (exit 1) -- full output above.'],
            log_path=None,
        )

    monkeypatch.setattr("mso_studio.core.converter.run_backend", fake_run_backend)
    params = ConverterParams(mechanism="/m/broken.inp", output_dir=str(tmp_path))
    record = run_convert(params)   # must NOT raise
    assert record.status == "failed"
    assert record.error == "ck2yaml failed (exit 1) -- full output above."


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_run_convert_real_konnov_2019_end_to_end(tmp_path):
    """The actual bridge Krunal's bug report exercised, run from PySide6:
    .venv-gui -> anaconda python3.9 -> mechanism_converter_gui.py --convert.
    Confirms point 2's fix (species-name corruption) holds through this path
    too, not just when convert() is called directly in-process."""
    konnov = REPO_ROOT / "MECHANISM" / "KONNOV_2019_AMMONIA_HYDROGEN" / "128-957.inp"
    thermo = REPO_ROOT / "MECHANISM" / "KONNOV_2019_AMMONIA_HYDROGEN" / "128-957.dat"
    if not konnov.exists():
        pytest.skip("KONNOV_2019_AMMONIA_HYDROGEN not present")

    params = ConverterParams(
        mechanism=str(konnov), thermo=str(thermo), output_dir=str(tmp_path),
        make_yaml=False, make_db=False, permissive=True,
    )
    lines = []
    record = run_convert(params, on_line=lines.append)

    full_log = "\n".join(lines)
    assert "C2H4O1E-2" not in full_log
    assert "C2H3O1E-2" not in full_log
    assert 'Unexpected token "C2H4O1-2+"' not in full_log
    assert record.status == "done"
