"""
rmip/tests/test_csv2xml_cli.py
=================================
Regression coverage for WORKBENCH_AUDIT_2026-08-10.md Round 4, defect R4:
config_tools/converters_io/CSV2XML/csv2xml.py's CLI only caught
RespecthWriterError, so three ordinary user mistakes (omitting the
documented-optional --x-units/--y-units flags, a missing --csv path, a
non-numeric data row) all produced a raw Python traceback instead of a
clean ``ERROR: ...`` message and exit code 1.

Imports csv2xml.py by file path (same vendored-import convention
``rmip.io.respecth_reader``/``mso_studio.core.csv2xml`` already use for
xml2csv.py) and calls its ``main(argv)`` directly, capturing stdout/stderr,
rather than shelling out -- faster and gives a real traceback in the test
output if one of these regresses.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_CSV2XML_DIR = _REPO / "config_tools" / "converters_io" / "CSV2XML"


def _load_csv2xml():
    spec = importlib.util.spec_from_file_location(
        "test_csv2xml_cli_vendored_csv2xml", str(_CSV2XML_DIR / "csv2xml.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def csv2xml_cli():
    return _load_csv2xml()


def _write_pts_csv(path: Path) -> None:
    path.write_text("T_K,tau_ms\n1000,50\n1100,40\n")


class TestCsv2XmlCliErrorPaths:
    def test_omitting_x_units_and_y_units_does_not_crash(self, tmp_path, csv2xml_cli, capsys):
        """Both flags are documented as optional (module docstring, and
        write_respecth_xml's own metadata.get(..., 'K'/'') defaults) --
        omitting them used to raise TypeError: cannot serialize None."""
        csv_path = tmp_path / "pts.csv"
        _write_pts_csv(csv_path)
        out_path = tmp_path / "out.xml"

        rc = csv2xml_cli.main([
            "--csv", str(csv_path), "--output", str(out_path),
            "--target-type", "Tig", "--x-quantity", "temperature",
            "--phi", "1.0", "--fuel", "H2:1.0",
        ])

        assert rc == 0
        assert out_path.exists()
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err

    def test_missing_csv_file_exits_cleanly(self, tmp_path, csv2xml_cli, capsys):
        rc = csv2xml_cli.main([
            "--csv", str(tmp_path / "does_not_exist.csv"), "--output", str(tmp_path / "out.xml"),
            "--target-type", "Tig", "--x-quantity", "temperature",
            "--x-units", "K", "--y-units", "ms", "--phi", "1.0", "--fuel", "H2:1.0",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "ERROR" in captured.err

    def test_non_numeric_data_row_exits_cleanly(self, tmp_path, csv2xml_cli, capsys):
        csv_path = tmp_path / "badrow.csv"
        csv_path.write_text("T_K,tau_ms\n1000,50\nnot_a_number,40\n")

        rc = csv2xml_cli.main([
            "--csv", str(csv_path), "--output", str(tmp_path / "out.xml"),
            "--target-type", "Tig", "--x-quantity", "temperature",
            "--x-units", "K", "--y-units", "ms", "--phi", "1.0", "--fuel", "H2:1.0",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err
        assert "data row 2" in captured.err

    def test_documented_example_still_works(self, tmp_path, csv2xml_cli):
        """Regression guard on the module docstring's own worked example --
        the R4 fix must not break the ordinary, everything-specified case."""
        csv_path = tmp_path / "pts.csv"
        _write_pts_csv(csv_path)
        out_path = tmp_path / "out.xml"

        rc = csv2xml_cli.main([
            "--csv", str(csv_path), "--output", str(out_path),
            "--target-type", "Tig", "--x-quantity", "temperature",
            "--x-units", "K", "--y-units", "ms", "--pressure-atm", "2.2",
            "--phi", "1.0", "--fuel", "H2:1.0", "--bath-gas-ratio", "199",
            "--bath-gas-species", "AR",
        ])

        assert rc == 0
        assert out_path.exists()
        assert 'units="K"' in out_path.read_text()
