"""
rmip/tests/test_respecth_writer.py
=====================================
Regression coverage for WORKBENCH_AUDIT_2026-08-10.md's defect B11: the plot
digitizer's own minimal RKD XML writer (mso_studio.core.digitizer.
export_rkd_xml) emits a schema config_tools/converters_io/XML2CSV/
xml2csv.py rejects outright ("Missing commonProperties or dataGroup...
skipping") -- every digitized dataset was silently dropped, not converted.

rmip.io.respecth_writer.write_respecth_xml is the fix (F1 in that same
audit's Part F: "CSV -> ReSpecTh/RKD XML converter"). These tests prove it
by round-tripping THROUGH THE REAL, UNMODIFIED xml2csv.py -- not by
asserting anything about the XML's shape in isolation, since "well-formed
XML" and "XML xml2csv.py actually accepts" turned out to be different
things (that gap is exactly what B11 was).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rmip.io.respecth_writer import write_respecth_xml, RespecthWriterError  # noqa: E402


def _load_xml2csv():
    """Import xml2csv.py by file path -- same pattern
    rmip.io.respecth_reader._load_xml2csv already uses for the same
    module."""
    import importlib.util

    xml2csv_dir = _REPO / "config_tools" / "converters_io" / "XML2CSV"
    if str(xml2csv_dir) not in sys.path:
        sys.path.append(str(xml2csv_dir))
    spec = importlib.util.spec_from_file_location(
        "test_respecth_writer_vendored_xml2csv", str(xml2csv_dir / "xml2csv.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows, header=("x", "y")):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _write_csv_no_header(path: Path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for row in rows:
            w.writerow(row)


class TestWriteRespecthXmlAcceptedByRealXml2Csv:
    """The core claim: a CSV this writer touches becomes a REAL, non-skipped
    target row via the actual xml2csv.py converter -- not a synthetic
    stand-in for it."""

    def test_temperature_varying_tig(self, tmp_path):
        csv_path = tmp_path / "tig.csv"
        _write_csv(csv_path, [(1450.0, 0.056), (1350.0, 0.185), (1250.0, 0.75), (1150.0, 3.5)])

        xml_path = tmp_path / "xmldir" / "tig.xml"
        xml_path.parent.mkdir()
        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 2.2, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}, "bath_gas_ratio": 3.76, "bath_gas_species": "N2"},
            "dataset_id": "tig",
        }
        write_respecth_xml(str(csv_path), metadata, str(xml_path))

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str(xml_path.parent), str(out_dir), fuel_list=["NH3"])

        assert result["n_files_ok"] == 1
        assert result["n_files_skipped"] == 0
        assert result["n_rows"] == 4
        target_text = Path(result["target_path"]).read_text()
        lines = target_text.strip().splitlines()
        assert len(lines) == 4
        # Every T and observed value round-trips exactly.
        assert "T -- 1450.0" in lines[0] and "observed -- 0.056" in lines[0]
        assert "T -- 1150.0" in lines[3] and "observed -- 3.5" in lines[3]
        assert "Phi -- 1.0" in lines[0]

    def test_phi_varying_fls(self, tmp_path):
        """The harder branch: composition (and therefore phi) varies per
        row. Confirms the phi VALUE this writer was told to encode is the
        phi VALUE xml2csv.py's own independent composition -> phi
        calculation re-derives -- proving the shared chem_data.py
        stoichiometry lookup keeps both sides consistent."""
        csv_path = tmp_path / "fls.csv"
        rows = [(0.7, 10.5), (0.8, 14.2), (0.9, 17.3), (1.0, 18.5)]
        _write_csv(csv_path, rows)

        xml_path = tmp_path / "xmldir" / "fls.xml"
        xml_path.parent.mkdir()
        metadata = {
            "target_type": "Fls", "x_quantity": "phi",
            "x_units": "unitless", "y_units": "cm/s",
            "fixed": {"pressure_atm": 1.0, "temperature_K": 298.0},
            "composition": {"fuel": {"NH3": 1.0}, "bath_gas_ratio": 3.76, "bath_gas_species": "N2"},
            "dataset_id": "fls",
        }
        write_respecth_xml(str(csv_path), metadata, str(xml_path))

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str(xml_path.parent), str(out_dir), fuel_list=["NH3"])

        assert result["n_rows"] == 4
        lines = Path(result["target_path"]).read_text().strip().splitlines()
        for line, (phi, su) in zip(lines, rows):
            assert f"Phi -- {phi}" in line
            assert f"observed -- {su}" in line

    def test_real_digitizer_csv_with_duplicate_headers(self, tmp_path):
        """THE original B11 reproduction: the real, recorded digitizer
        output (Test_run_23_JULY/mso_studio/gui_widget_evidence/
        digitized_points.csv) has DUPLICATE column headers ('T_K,T_K') --
        csv.DictReader silently collapses these to one column, losing the
        first value entirely. This is the exact file that produced ZERO
        target rows via the digitizer's own export_rkd_xml + xml2csv.py in
        the original Round-2 audit session; it must now produce one real
        row via this writer instead."""
        real_csv = (_REPO / "Test_run_23_JULY" / "mso_studio" / "gui_widget_evidence"
                    / "digitized_points.csv")
        if not real_csv.exists():
            pytest.skip("recorded digitizer evidence file not present in this checkout")

        xml_path = tmp_path / "xmldir" / "real.xml"
        xml_path.parent.mkdir()
        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}, "bath_gas_ratio": 3.76, "bath_gas_species": "N2"},
            "dataset_id": "real_digitizer",
        }
        write_respecth_xml(str(real_csv), metadata, str(xml_path))

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str(xml_path.parent), str(out_dir), fuel_list=["NH3"])

        assert result["n_files_ok"] == 1
        assert result["n_rows"] == 1  # digitized_points.csv has exactly one data row
        line = Path(result["target_path"]).read_text().strip()
        # The x column value (50.0) must survive -- this is exactly what a
        # DictReader-based reader would have silently dropped.
        assert "T -- 50.0" in line
        assert "observed -- 25.0" in line


class TestWriteRespecthXmlValidation:
    def test_rejects_unknown_target_type(self, tmp_path):
        csv_path = tmp_path / "x.csv"
        _write_csv(csv_path, [(1.0, 2.0)])
        with pytest.raises(RespecthWriterError):
            write_respecth_xml(str(csv_path), {"target_type": "Nope", "x_quantity": "temperature"},
                                str(tmp_path / "out.xml"))

    def test_rejects_missing_fuel(self, tmp_path):
        csv_path = tmp_path / "x.csv"
        _write_csv(csv_path, [(1.0, 2.0)])
        with pytest.raises(RespecthWriterError):
            write_respecth_xml(
                str(csv_path),
                {"target_type": "Tig", "x_quantity": "temperature", "composition": {}},
                str(tmp_path / "out.xml"),
            )

    def test_rejects_inert_fuel_with_zero_stoichiometry(self, tmp_path):
        """chem_data.compute_nu_o2's atom-balance fallback (the same one
        xml2csv.py itself uses) returns 0.0 for a genuinely inert species --
        no C/H/S content to oxidize. 'AR' as a "fuel" has no combustion
        chemistry, so this must raise rather than silently emit a
        div-by-zero-adjacent composition. (An arbitrary made-up name like
        "XYZ123" is NOT a reliable test case here: the same naive
        atom-balance parser can coincidentally extract nonzero C/H/O counts
        from letters that happen to look like element symbols.)"""
        csv_path = tmp_path / "x.csv"
        _write_csv(csv_path, [(1.0, 2.0)])
        with pytest.raises(RespecthWriterError):
            write_respecth_xml(
                str(csv_path),
                {
                    "target_type": "Fls", "x_quantity": "phi",
                    "composition": {"fuel": {"AR": 1.0}},
                },
                str(tmp_path / "out.xml"),
            )


class TestReadCsvPointsHeaderDetection:
    """Round 4 regression: _read_csv_points() used to unconditionally treat
    row 0 as a header, silently discarding the first digitized point of any
    hand-prepared, headerless CSV (a 3-row headerless numeric CSV produced
    only 2 <dataPoint> elements). WORKBENCH_AUDIT_2026-08-10.md Round 4,
    defect R3."""

    def test_headerless_csv_keeps_every_row(self, tmp_path):
        csv_path = tmp_path / "headerless.csv"
        _write_csv_no_header(csv_path, [(1000.0, 50.0), (1100.0, 40.0), (1200.0, 30.0)])

        xml_path = tmp_path / "out.xml"
        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(xml_path))
        assert xml_str.count("<dataPoint>") == 3  # was 2 before the fix

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str(xml_path.parent), str(out_dir), fuel_list=["NH3"])
        assert result["n_rows"] == 3

    def test_header_row_with_duplicate_names_still_treated_as_header(self, tmp_path):
        """The pre-existing duplicate-header case (T_K,T_K) must not
        regress: a non-numeric row 0 is still consumed as a header, not
        data."""
        csv_path = tmp_path / "dup_header.csv"
        _write_csv(csv_path, [(50.0, 25.0)], header=("T_K", "T_K"))

        xml_path = tmp_path / "out.xml"
        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(xml_path))
        assert xml_str.count("<dataPoint>") == 1  # not 2 -- the header row isn't data


class TestBadDataRowErrorMessage:
    """Round 4 regression: a non-numeric data row used to raise a bare
    ValueError with no file/row context. WORKBENCH_AUDIT_2026-08-10.md
    Round 4, defect R4."""

    def test_non_numeric_row_raises_respecth_writer_error_with_row_number(self, tmp_path):
        csv_path = tmp_path / "badrow.csv"
        _write_csv(csv_path, [(1000.0, 50.0), ("not_a_number", 40.0)])

        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        with pytest.raises(RespecthWriterError, match=r"data row 2"):
            write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))


class TestAdversarialInputs:
    """Round 4, Phase 2 stress-testing: real defects found by probing F1
    with inputs beyond what the original B11 fix's own tests covered."""

    def test_whitespace_only_row_is_skipped_like_a_blank_line(self, tmp_path):
        """A line that's only whitespace (no comma at all) parses as a
        non-empty single-cell row (['   ']), not [] -- it used to reach the
        data-row loop and raise "not valid (x, y) data" instead of being
        silently skipped like a genuinely blank line."""
        csv_path = tmp_path / "blank_rows.csv"
        csv_path.write_text("T_K,tau_ms\n1000,50\n\n   \n1100,40\n")

        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))
        assert xml_str.count("<dataPoint>") == 2

    def test_three_column_csv_is_treated_as_symmetric_uncertainty(self, tmp_path):
        """has_err was gated at len(header) >= 4, so a (x, y, dy) 3-column
        CSV had its dy column silently discarded -- no error, no warning,
        the uncertainty the user explicitly provided just vanished."""
        csv_path = tmp_path / "threecol.csv"
        csv_path.write_text("T_K,tau_ms,dy\n1000,50,5\n1100,40,4\n")

        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))
        assert "evaluated standard deviation" in xml_str

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str((tmp_path / "out.xml").parent), str(out_dir), fuel_list=["NH3"])
        assert result["n_rows"] == 2

    def test_fuel_species_colliding_with_bath_gas_name_is_additive(self, tmp_path):
        """A diluted-fuel blend that itself contains the bath-gas species
        (e.g. {'H2': 1.0, 'N2': 0.5} paired with the default N2 bath gas --
        an ordinary syngas-type case) used to have its fuel-side N2 mole
        fraction silently OVERWRITTEN by the bath-gas N2 fraction via
        {**mole_fracs, bath_gas_species: bath_x}, instead of the two being
        added together. Covers both the constant-composition
        (x_quantity='temperature') and per-row (x_quantity='phi') branches,
        which had the same bug independently."""
        csv_path = tmp_path / "n2fuel.csv"
        csv_path.write_text("T_K,tau_ms\n1000,50\n1100,40\n")
        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"H2": 1.0, "N2": 0.5}, "bath_gas_ratio": 3.76,
                             "bath_gas_species": "N2"},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))
        assert xml_str.count('preferredKey="N2"') == 1  # not 2 (duplicate component)

        import re
        amounts = re.findall(
            r'<speciesLink preferredKey="(H2|N2|O2)"/>\s*<amount[^>]*>([\d.eE+-]+)</amount>',
            xml_str)
        fracs = {sp: float(v) for sp, v in amounts}
        assert set(fracs) == {"H2", "N2", "O2"}
        assert abs(sum(fracs.values()) - 1.0) < 1e-6

    def test_fuel_species_colliding_with_bath_gas_name_phi_varying(self, tmp_path):
        csv_path = tmp_path / "n2fuel_phi.csv"
        csv_path.write_text("phi,Su\n0.8,10\n1.0,15\n")
        metadata = {
            "target_type": "Fls", "x_quantity": "phi",
            "x_units": "unitless", "y_units": "cm/s",
            "fixed": {"pressure_atm": 1.0, "temperature_K": 298.0},
            "composition": {"fuel": {"H2": 1.0, "N2": 0.5}, "bath_gas_ratio": 3.76,
                             "bath_gas_species": "N2"},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))
        assert xml_str.count('label="x_N2"') == 1  # not 2 declared <property> elements

        mod = _load_xml2csv()
        out_dir = tmp_path / "out"
        result = mod.main(str((tmp_path / "out.xml").parent), str(out_dir), fuel_list=["H2"])
        assert result["n_rows"] == 2

    def test_bom_prefixed_headerless_csv_keeps_every_row(self, tmp_path):
        """A BOM (routine from Excel/Windows CSV exports) glued to the
        first cell of a headerless numeric CSV used to defeat the R3
        header-vs-data numeric-detection check (float("\\ufeff1000") raises),
        silently resurrecting the exact first-point-dropped bug R3 fixed."""
        csv_path = tmp_path / "bom_headerless.csv"
        csv_path.write_bytes(b"\xef\xbb\xbf1000,50\n1100,40\n1200,30\n")

        metadata = {
            "target_type": "Tig", "x_quantity": "temperature",
            "x_units": "K", "y_units": "ms",
            "fixed": {"pressure_atm": 1.0, "phi": 1.0},
            "composition": {"fuel": {"NH3": 1.0}},
        }
        xml_str = write_respecth_xml(str(csv_path), metadata, str(tmp_path / "out.xml"))
        assert xml_str.count("<dataPoint>") == 3


class TestCsv2XmlCliHeaderYamlBathGasOverride:
    """Round 4, Phase 2: --bath-gas-ratio/--bath-gas-species were silently
    ignored whenever --header was also given -- neither flag was checked
    anywhere in the --header merge block, even though the module
    docstring's own worked example passes both alongside --header."""

    def _load_csv2xml_cli(self):
        import importlib.util

        csv2xml_dir = _REPO / "config_tools" / "converters_io" / "CSV2XML"
        spec = importlib.util.spec_from_file_location(
            "test_respecth_writer_vendored_csv2xml", str(csv2xml_dir / "csv2xml.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_cli_flags_override_header_yaml_bath_gas(self, tmp_path):
        csv_path = tmp_path / "pts.csv"
        csv_path.write_text("T_K,tau_ms\n1000,50\n1100,40\n")
        header_yaml = tmp_path / "header.yaml"
        header_yaml.write_text(
            "target_type: Tig\nx_quantity: temperature\nx_units: K\ny_units: ms\n"
            "fixed:\n  pressure_atm: 1.0\n  phi: 1.0\n"
            "composition:\n  fuel: {NH3: 1.0}\n  bath_gas_ratio: 3.76\n  bath_gas_species: N2\n"
        )
        out_path = tmp_path / "out.xml"
        cli = self._load_csv2xml_cli()

        rc = cli.main([
            "--csv", str(csv_path), "--header", str(header_yaml), "--output", str(out_path),
            "--bath-gas-ratio", "199", "--bath-gas-species", "AR",
        ])

        assert rc == 0
        xml_text = out_path.read_text()
        assert 'preferredKey="AR"' in xml_text
        assert 'preferredKey="N2"' not in xml_text

    def test_header_yaml_bath_gas_used_when_not_overridden(self, tmp_path):
        csv_path = tmp_path / "pts.csv"
        csv_path.write_text("T_K,tau_ms\n1000,50\n1100,40\n")
        header_yaml = tmp_path / "header.yaml"
        header_yaml.write_text(
            "target_type: Tig\nx_quantity: temperature\nx_units: K\ny_units: ms\n"
            "fixed:\n  pressure_atm: 1.0\n  phi: 1.0\n"
            "composition:\n  fuel: {NH3: 1.0}\n  bath_gas_ratio: 3.76\n  bath_gas_species: N2\n"
        )
        out_path = tmp_path / "out.xml"
        cli = self._load_csv2xml_cli()

        rc = cli.main(["--csv", str(csv_path), "--header", str(header_yaml), "--output", str(out_path)])

        assert rc == 0
        assert 'preferredKey="N2"' in out_path.read_text()
