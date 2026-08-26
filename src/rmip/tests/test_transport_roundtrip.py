"""
rmip/tests/test_transport_roundtrip.py
======================================
Tests for the transport parse→DB→writer round-trip (Phase-3 follow-on, Point 2a).

Covers both endpoints:
  * Cantera: h2o2.yaml carries a transport block → parsed into the species DB
    columns → re-emitted by CanteraYAMLWriter → reloads identically in Cantera.
  * CHEMKIN: a transport file is now honored at parse time (previously ignored)
    and ChemkinWriter.write_transport re-emits from the DB columns.

Run with:
    python -m pytest src/rmip/tests/test_transport_roundtrip.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")

from rmip.config import RMIPConfig  # noqa: E402
from rmip.core.parser import MechanismParser  # noqa: E402


# ---------------------------------------------------------------------------
# Cantera endpoint
# ---------------------------------------------------------------------------

def _parse_h2o2(tmp_path):
    db_path = str(tmp_path / "tp.db")
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=str(Path(ct.__path__[0]) / "data" / "h2o2.yaml"))
    parser.conn.close()
    return db_path, mech_id


def test_cantera_parse_populates_transport_columns(tmp_path):
    db_path, mech_id = _parse_h2o2(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, transport_geometry, well_depth_K, diameter_A, "
            "transport_source FROM species WHERE mechanism_id=?", (mech_id,)
        ).fetchall()
    by = {r["name"]: r for r in rows}
    assert by["H2"]["transport_geometry"] == "linear"
    assert by["H2"]["well_depth_K"] == pytest.approx(38.0)
    assert by["H2"]["diameter_A"] == pytest.approx(2.92)
    assert by["H"]["transport_geometry"] == "atom"
    assert all(r["transport_source"] == "cantera"
               for r in rows if r["well_depth_K"] is not None)


def test_cantera_writer_roundtrips_transport(tmp_path):
    from rmip.io.cantera_writer import CanteraYAMLWriter
    db_path, mech_id = _parse_h2o2(tmp_path)
    out = tmp_path / "rt.yaml"
    CanteraYAMLWriter(db_path).write(mech_id, str(out))

    orig = ct.Solution(str(Path(ct.__path__[0]) / "data" / "h2o2.yaml"))
    rt = ct.Solution(str(out))
    for name in ("H2", "O", "OH", "H2O", "HO2"):
        o, r = orig.species(name).transport, rt.species(name).transport
        assert r.well_depth == pytest.approx(o.well_depth, rel=1e-3)
        assert r.diameter == pytest.approx(o.diameter, rel=1e-3)


def test_chemkin_writer_emits_transport(tmp_path):
    from rmip.io.chemkin_writer import ChemkinWriter
    from rmip.io.transport_io import parse_chemkin_transport
    db_path, mech_id = _parse_h2o2(tmp_path)
    out = tmp_path / "tran.dat"
    n = ChemkinWriter(db_path).write_transport(mech_id, str(out))
    assert n == 10
    records = parse_chemkin_transport(str(out))
    assert records["H2"].well_depth_K == pytest.approx(38.0)
    assert records["H2"].diameter_A == pytest.approx(2.92)
    assert records["H"].geometry == "atom"


# ---------------------------------------------------------------------------
# CHEMKIN endpoint — transport_path now honored at parse time
# ---------------------------------------------------------------------------

_MINI_CHEMKIN = (
    "ELEMENTS\nH O\nEND\n\n"
    "SPECIES\nH2 O OH\nEND\n\n"
    "THERMO ALL\n"
    "   200.000  1000.000  5000.000\n"
    "H2                      H   2               G  200.000 5000.000 1000.000    1\n"
    " 3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2\n"
    "-9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3\n"
    " 2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4\n"
    "O                       O   1               G  200.000 5000.000 1000.000    1\n"
    " 2.56942078E+00-8.59741137E-05 4.19484589E-08-1.00177799E-11 1.22833691E-15    2\n"
    " 2.92175791E+04 4.78433864E+00 3.16826710E+00-3.27931884E-03 6.64306396E-06    3\n"
    "-6.12806624E-09 2.11265971E-12 2.91222592E+04 2.05193346E+00                   4\n"
    "OH                      H   1O   1          G  200.000 5000.000 1000.000    1\n"
    " 2.86472886E+00 1.05650448E-03-2.59082758E-07 3.05218674E-11-1.33195876E-15    2\n"
    " 3.68362875E+03 5.70164073E+00 4.12530561E+00-3.22544939E-03 6.52764691E-06    3\n"
    "-5.79853643E-09 2.00592054E-12 3.38153812E+03-6.90432960E-01                   4\n"
    "END\n\n"
    "REACTIONS\n"
    "H2 + O = OH + H    2.0E+14 0.0 70300.0\n"
    "END\n"
)

_MINI_TRANSPORT = (
    "H2                 1    38.000     2.920     0.000     0.790   280.000\n"
    "O                  0    80.000     2.750     0.000     0.000     0.000\n"
    "OH                 1    80.000     2.750     0.000     0.000     0.000\n"
)


def test_chemkin_parse_honors_transport_path(tmp_path):
    mech = tmp_path / "mech.inp"
    mech.write_text(_MINI_CHEMKIN)
    tran = tmp_path / "tran.dat"
    tran.write_text(_MINI_TRANSPORT)

    db_path = str(tmp_path / "ck.db")
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    mech_id = parser.parse(mech_path=str(mech), transport_path=str(tran))
    parser.conn.close()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        by = {r["name"]: r for r in conn.execute(
            "SELECT name, transport_geometry, well_depth_K, diameter_A, "
            "transport_source FROM species WHERE mechanism_id=?", (mech_id,))}
    assert by["H2"]["well_depth_K"] == pytest.approx(38.0)
    assert by["H2"]["diameter_A"] == pytest.approx(2.92)
    assert by["H2"]["transport_geometry"] == "linear"
    assert by["O"]["transport_geometry"] == "atom"
    assert by["H2"]["transport_source"] == "chemkin"
