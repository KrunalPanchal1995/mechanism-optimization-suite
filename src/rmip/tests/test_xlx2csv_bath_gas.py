"""
Regression coverage for WORKBENCH_AUDIT_2026-08-10.md defect B15, extended
in Round 4: a header that declares "bg1" in Data_structure but omits the
"Bath_gas" field entirely used to sail through xlx2csv.py silently
(normalize_species_list(None) == [], so the per-row column-existence check
never runs) and write literal "Bath_gas -- x->None = None" into every
output row -- which then crashed several rows and one whole file later,
inside combustion_target_class.py, as json.loads("None") (Python's None
spelled out is not valid JSON), a confusing error with no indication the
real problem is a missing header field several layers upstream.

Imports xlx2csv.py by file path (config_tools/ isn't a Python package under
src/, so it isn't importable by dotted name) -- same vendored-import
convention used elsewhere in this suite for sibling converter scripts.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_XLX2CSV_DIR = _REPO / "config_tools" / "converters_io" / "XLX2CSV"
_REAL_FIXTURE_DIR = _XLX2CSV_DIR / "Flame_Speed"


def _load_xlx2csv():
    spec = importlib.util.spec_from_file_location(
        "test_xlx2csv_bath_gas_vendored_xlx2csv", str(_XLX2CSV_DIR / "xlx2csv.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def xlx2csv_cli(monkeypatch):
    """main() does os.chdir(sys.argv[1]) internally -- always restore the
    real cwd afterward so this test can't leak state into later tests."""
    original_cwd = os.getcwd()
    mod = _load_xlx2csv()
    yield mod
    os.chdir(original_cwd)


class TestMissingBathGasHeaderField:
    def test_bg1_in_data_structure_but_no_bath_gas_field_raises_clearly(
        self, tmp_path, xlx2csv_cli, monkeypatch,
    ):
        work_dir = tmp_path / "missing_bath_gas"
        work_dir.mkdir()
        (work_dir / "fls.csv").write_bytes((_REAL_FIXTURE_DIR / "fls.csv").read_bytes())
        header_text = (_REAL_FIXTURE_DIR / "fls.header").read_text()
        assert "Bath_gas: {bg1: N2}" in header_text  # sanity: the real fixture has it
        broken_header = header_text.replace("Bath_gas: {bg1: N2}", "#Bath_gas: {bg1: N2}")
        (work_dir / "fls.header").write_text(broken_header)

        monkeypatch.setattr(sys, "argv", ["xlx2csv.py", str(work_dir)])
        with pytest.raises(KeyError, match="no 'Bath_gas' field"):
            xlx2csv_cli.main()

    def test_real_fixture_with_bath_gas_present_is_unaffected(
        self, tmp_path, xlx2csv_cli, monkeypatch,
    ):
        """Regression guard: the fix must not touch the ordinary case where
        Bath_gas IS given -- this is the exact real fixture G3's own
        verification used. Copied into tmp_path (not run in place) so the
        test doesn't write an output/ directory into the real repo tree."""
        work_dir = tmp_path / "real_fixture"
        work_dir.mkdir()
        (work_dir / "fls.csv").write_bytes((_REAL_FIXTURE_DIR / "fls.csv").read_bytes())
        (work_dir / "fls.header").write_bytes((_REAL_FIXTURE_DIR / "fls.header").read_bytes())

        monkeypatch.setattr(sys, "argv", ["xlx2csv.py", str(work_dir)])
        rc = xlx2csv_cli.main()
        assert rc in (0, None)
        assert (work_dir / "output" / "fls.out").exists()
        out_text = (work_dir / "output" / "fls.out").read_text()
        assert "x->{'bg1': 'N2'}" in out_text
