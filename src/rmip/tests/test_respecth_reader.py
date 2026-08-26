"""
rmip/tests/test_respecth_reader.py
==================================
Tests for ReSpecTh/PrIMe (RKD XML) ingestion (`rmip/io/respecth_reader.py` +
`ExpDataReader.load_respecth_xml`), which bridges the legacy `.target` pipeline
and the modern rmip SQLite DB.

Covers:
  * the `.target` tokeniser + rmip-canonical unit conversion (pure, no DB),
  * the exp_datasets.sim_type CHECK-widening migration,
  * an end-to-end import against committed ReSpecTh sample XML → both endpoints.

Run with:
    python -m pytest src/rmip/tests/test_respecth_reader.py -v
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from rmip.io.respecth_reader import (
    RESPECTH_SIM_TYPE_MAP,
    _observed_factor,
    _pressure_to_atm,
    _tokenize_target_line,
    datasets_from_target_file,
)

# Committed ReSpecTh samples used by xml2csv's own test harness.
_SAMPLE_DIR = (
    Path(__file__).resolve().parents[3]
    / "config_tools" / "converters_io" / "XML2CSV"
    / "DATASET" / "xml2csv_code_testing"
)
_needs_samples = pytest.mark.skipif(
    not _SAMPLE_DIR.is_dir() or not list(_SAMPLE_DIR.glob("*.xml")),
    reason="ReSpecTh sample XML dir not present",
)


# ---------------------------------------------------------------------------
# Pure unit-conversion + tokeniser
# ---------------------------------------------------------------------------

def test_observed_factor_idt_time_units():
    assert _observed_factor("IDT", "us")  == (1e-6, "s")
    assert _observed_factor("IDT", "ms")  == (1e-3, "s")
    assert _observed_factor("IDT", "s")   == (1.0,  "s")
    # unknown unit -> factor 1.0 (with a warning), still canonical 's'
    assert _observed_factor("IDT", "weird") == (1.0, "s")


def test_observed_factor_lfs_and_speciation():
    assert _observed_factor("LFS", "cm/s") == (1e-2, "m/s")
    assert _observed_factor("LFS", "m/s")  == (1.0, "m/s")
    assert _observed_factor("JSR", "ppm")  == (1e-6, "mole_fraction")


def test_pressure_to_atm():
    assert _pressure_to_atm(1.0, "atm") == pytest.approx(1.0)
    assert _pressure_to_atm(1.0, "bar") == pytest.approx(1.0 / 1.01325)
    assert _pressure_to_atm(760.0, "torr") == pytest.approx(1.0)
    assert _pressure_to_atm(101325.0, "Pa") == pytest.approx(1.0)


def test_tokenize_target_line():
    line = ("1\t|myset\t| target -- Tig\t| T -- 1200\t| P -- 2.5\t| "
            "Phi -- 1.0\t| observed -- 10.0\t| deviation -- 1.0\t| "
            "units -- {'P': 'atm', 'T': 'K', 'observed': 'ms'}\n")
    dataset_id, fields = _tokenize_target_line(line)
    assert dataset_id == "myset"
    assert fields["target"] == "Tig"
    assert fields["T"] == "1200"
    assert fields["observed"] == "10.0"
    assert "'observed': 'ms'" in fields["units"]


def test_datasets_from_synthetic_target(tmp_path):
    """A hand-written .target with a known unit → exact canonical conversion."""
    target = tmp_path / "syn.target"
    target.write_text(
        "1\t|setA\t| target -- Tig\t| T -- 1200\t| P -- 2.5\t| Phi -- 1.0\t| "
        "observed -- 10.0\t| deviation -- 1.0\t| "
        "units -- {'P': 'atm', 'T': 'K', 'observed': 'ms'}\n"
        "2\t|setA\t| target -- Tig\t| T -- 1300\t| P -- 2.5\t| Phi -- 1.0\t| "
        "observed -- 5.0\t| deviation -- 0.5\t| "
        "units -- {'P': 'atm', 'T': 'K', 'observed': 'ms'}\n",
        encoding="utf-8",
    )
    dsets = datasets_from_target_file(str(target), default_fuel_name="NH3")
    assert len(dsets) == 1
    ds = dsets[0]
    assert ds.sim_type == "IDT"
    assert ds.fuel_name == "NH3"
    assert len(ds.points) == 2
    p0 = ds.points[0]
    assert p0["T_K"] == 1200.0
    assert p0["P_atm"] == pytest.approx(2.5)
    assert p0["value"] == pytest.approx(10.0 * 1e-3)      # ms -> s
    assert p0["uncertainty"] == pytest.approx(1.0 * 1e-3)  # same factor
    assert p0["units"] == "s"


def test_unknown_target_tag_skipped(tmp_path):
    target = tmp_path / "bad.target"
    target.write_text(
        "1\t|s\t| target -- WAT\t| T -- 1\t| P -- 1\t| observed -- 1\t| "
        "units -- {'P': 'atm'}\n",
        encoding="utf-8",
    )
    assert datasets_from_target_file(str(target)) == []


def test_sim_type_map_covers_classifier_tags():
    for tag in ("Tig", "RCM", "Fls", "Flf", "Flf_a", "JSR", "PFR", "Flw"):
        assert tag in RESPECTH_SIM_TYPE_MAP


# ---------------------------------------------------------------------------
# Migration: exp_datasets.sim_type widening
# ---------------------------------------------------------------------------

def test_exp_dataset_sim_type_widened(tmp_path):
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations
    from rmip.database.schema_phase3 import insert_exp_dataset

    db = str(tmp_path / "widen.db")
    conn = init_db(db)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    run_phase6_migrations(conn)
    # JSR/PFR must now be accepted (were rejected by the original IDT/LFS CHECK)
    ds_id = insert_exp_dataset(conn, name="jsr_set", sim_type="JSR")
    assert ds_id > 0
    row = conn.execute("SELECT sim_type FROM exp_datasets WHERE id=?", (ds_id,)).fetchone()
    conn.close()
    assert row[0] == "JSR"


# ---------------------------------------------------------------------------
# End-to-end against committed ReSpecTh samples
# ---------------------------------------------------------------------------

@_needs_samples
def test_import_respecth_end_to_end(tmp_path):
    from rmip.io.exp_data_reader import ExpDataReader

    db = str(tmp_path / "e2e.db")
    # bootstrap full schema the way the CLI does
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations
    conn = init_db(db); upgrade_db(conn)
    run_phase3_migrations(conn); run_phase6_migrations(conn); conn.close()

    out_dir = tmp_path / "legacy_out"
    out_dir.mkdir()
    reader = ExpDataReader(db)
    result = reader.load_respecth_xml(
        str(_SAMPLE_DIR), out_dir=str(out_dir), fuel_list=["H2"], solver="cantera",
    )

    # Endpoint A: legacy artefacts on disk
    assert Path(result["target_path"]).exists()
    assert (out_dir / "converted.target").exists()
    assert (out_dir / "addundem.add").exists()

    # Endpoint B: DB rows
    assert result["n_points"] > 0
    assert len(result["dataset_ids"]) >= 1

    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    dsets = [dict(r) for r in conn.execute("SELECT * FROM exp_datasets")]
    pts = [dict(r) for r in conn.execute("SELECT * FROM exp_data_points")]
    conn.close()

    assert {d["sim_type"] for d in dsets} <= {"IDT", "LFS", "JSR", "PFR"}
    assert all(d["fuel_name"] == "H2" for d in dsets)
    # IDT points must be canonicalised to seconds and be physically small
    idt_ids = {d["id"] for d in dsets if d["sim_type"] == "IDT"}
    idt_pts = [p for p in pts if p["dataset_id"] in idt_ids]
    assert idt_pts, "expected at least one IDT dataset in the samples"
    for p in idt_pts:
        assert p["units"] == "s"
        assert 0 < p["value"] < 1.0   # seconds, not micro/milli magnitudes
        assert p["T_K"] > 300


@_needs_samples
def test_import_respecth_single_file(tmp_path):
    """A single .xml file (not a dir) is staged and ingested correctly."""
    from rmip.io.exp_data_reader import ExpDataReader
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations

    sample = sorted(_SAMPLE_DIR.glob("*.xml"))[0]
    db = str(tmp_path / "single.db")
    conn = init_db(db); upgrade_db(conn)
    run_phase3_migrations(conn); run_phase6_migrations(conn); conn.close()

    out_dir = tmp_path / "out"; out_dir.mkdir()
    reader = ExpDataReader(db)
    result = reader.load_respecth_xml(
        str(sample), out_dir=str(out_dir), fuel_list=["H2"],
    )
    assert result["conversion"]["n_files_ok"] == 1
    assert result["n_points"] > 0
