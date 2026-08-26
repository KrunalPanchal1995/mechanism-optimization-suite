"""
rmip/tests/test_transport.py
============================
Tests for transport-property I/O (`rmip/io/transport_io.py`) and the RDKit
structure→Lennard-Jones estimator (`rmip/analysis/transport_estimator.py`).

Run with:
    python -m pytest src/rmip/tests/test_transport.py -v
"""
from __future__ import annotations

import pytest

from rmip.io.transport_io import (
    TransportData,
    parse_chemkin_transport,
    write_chemkin_transport,
)


# ---------------------------------------------------------------------------
# TransportData + CHEMKIN I/O
# ---------------------------------------------------------------------------

_SAMPLE = """\
! a real-ish CHEMKIN transport snippet
AR                 0   136.500     3.330     0.000     0.000     0.000
H2                 1    38.000     2.920     0.000     0.790   280.000
H2O                2   572.400     2.605     1.844     0.000     4.000
CH3                1   144.000     3.800     0.000     0.000     0.000 ! radical
"""


def test_parse_chemkin_transport():
    d = parse_chemkin_transport(_SAMPLE)
    assert set(d) == {"AR", "H2", "H2O", "CH3"}
    assert d["AR"].geometry == "atom"
    assert d["H2"].geometry == "linear"
    assert d["H2O"].geometry == "nonlinear"
    assert d["H2O"].well_depth_K == pytest.approx(572.4)
    assert d["H2O"].diameter_A == pytest.approx(2.605)
    assert d["H2O"].dipole_D == pytest.approx(1.844)
    assert d["H2"].polarizability_A3 == pytest.approx(0.79)
    assert d["H2"].rot_relax == pytest.approx(280.0)
    assert d["AR"].source == "chemkin"


def test_parse_skips_keywords_and_short_lines():
    text = "TRANSPORT\nAR 0 136.5 3.33 0 0 0\nBADLINE 2 3\nEND\n"
    d = parse_chemkin_transport(text)
    assert set(d) == {"AR"}


def test_chemkin_roundtrip():
    d = parse_chemkin_transport(_SAMPLE)
    text = write_chemkin_transport(d, header="regenerated")
    d2 = parse_chemkin_transport(text)
    assert set(d2) == set(d)
    for name in d:
        assert d2[name].geom_int() == d[name].geom_int()
        assert d2[name].well_depth_K == pytest.approx(d[name].well_depth_K)
        assert d2[name].diameter_A == pytest.approx(d[name].diameter_A)


def test_to_cantera_dict_omits_zero_optionals():
    td = TransportData("nonlinear", 572.4, 2.605, dipole_D=1.844)
    cd = td.to_cantera_dict()
    assert cd["geometry"] == "nonlinear"
    assert cd["well-depth"] == pytest.approx(572.4)
    assert cd["dipole"] == pytest.approx(1.844)
    assert "polarizability" not in cd          # zero → omitted
    assert "rotational-relaxation" not in cd

    td2 = TransportData("atom", 136.5, 3.33)
    assert set(td2.to_cantera_dict()) == {"model", "geometry", "well-depth", "diameter"}


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

rdkit = pytest.importorskip("rdkit")
from rmip.analysis.transport_estimator import estimate_transport  # noqa: E402


# (smiles, literature sigma [Å], literature eps/kB [K]); non-polar species only
_LITERATURE = [
    ("C",    3.758, 148.6),   # CH4
    ("CC",   4.443, 215.7),   # C2H6
    ("CCC",  4.982, 237.1),   # C3H8 (Sandia transport DB)
    ("N#N",  3.621,  97.5),   # N2
]


@pytest.mark.parametrize("smiles,sigma_lit,eps_lit", _LITERATURE)
def test_estimator_matches_literature(smiles, sigma_lit, eps_lit):
    td = estimate_transport(smiles, name=smiles)
    assert td is not None
    # sigma within 12 %, eps within 30 % — this is an engineering estimate
    assert td.diameter_A == pytest.approx(sigma_lit, rel=0.12)
    assert td.well_depth_K == pytest.approx(eps_lit, rel=0.30)
    assert td.source.startswith("estimated:")


def test_estimator_geometry():
    assert estimate_transport("[Ar]").geometry == "atom"
    assert estimate_transport("[OH]").geometry == "linear"      # 2 atoms
    assert estimate_transport("O").geometry == "nonlinear"      # H2O, 3 atoms
    assert estimate_transport("O=C=O").geometry == "linear"     # CO2 (in known set)


def test_estimator_saturates_radicals():
    """A radical and its saturated parent get near-identical LJ (CH3 ≈ CH4)."""
    ch3 = estimate_transport("[CH3]")
    ch4 = estimate_transport("C")
    assert ch3 is not None and ch4 is not None
    assert ch3.diameter_A == pytest.approx(ch4.diameter_A, rel=0.05)
    assert ch3.well_depth_K == pytest.approx(ch4.well_depth_K, rel=0.05)


def test_estimator_bad_smiles_returns_none():
    assert estimate_transport("not a smiles!!!") is None
    assert estimate_transport("") is None


def test_from_cantera_dict():
    d = {"model": "gas", "geometry": "nonlinear", "diameter": 2.605,
         "well-depth": 572.4, "dipole": 1.844, "rotational-relaxation": 4.0}
    td = TransportData.from_cantera_dict(d)
    assert td.geometry == "nonlinear"
    assert td.well_depth_K == pytest.approx(572.4)
    assert td.diameter_A == pytest.approx(2.605)
    assert td.dipole_D == pytest.approx(1.844)
    assert td.rot_relax == pytest.approx(4.0)
    assert td.source == "cantera"


# ---------------------------------------------------------------------------
# DB migration + CLI integration
# ---------------------------------------------------------------------------

def test_migration_adds_transport_columns(tmp_path):
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations

    db = str(tmp_path / "t.db")
    conn = init_db(db); upgrade_db(conn)
    run_phase3_migrations(conn); run_phase6_migrations(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(species)").fetchall()}
    conn.close()
    assert {"transport_geometry", "well_depth_K", "diameter_A", "dipole_D",
            "polarizability_A3", "rot_relax", "transport_source"} <= cols


def test_cmd_transport_end_to_end(tmp_path):
    """Full CLI path: read known transport from Cantera + estimate an extra."""
    ct = pytest.importorskip("cantera")
    import os
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from rmip.database.schema import init_db, insert_species
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations
    from rmip.cli import _cmd_transport

    db = str(tmp_path / "cli.db")
    conn = init_db(db); upgrade_db(conn)
    run_phase3_migrations(conn); run_phase6_migrations(conn)
    yaml = os.path.join(ct.__path__[0], "data", "h2o2.yaml")
    conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, loaded_at) VALUES (?,?,?,?)",
        ("h2o2", "cantera_yaml", yaml, datetime.now(timezone.utc).isoformat()),
    )
    mid = conn.execute("SELECT id FROM mechanisms").fetchone()[0]
    for nm, smi in [("H2", "[H][H]"), ("O2", "O=O"), ("H2O", "O")]:
        insert_species(conn, mechanism_id=mid, name=nm, canonical_smiles=smi)
    insert_species(conn, mechanism_id=mid, name="NC7H16", canonical_smiles="CCCCCCC")
    conn.commit(); conn.close()

    tran = str(tmp_path / "out.tran")
    args = SimpleNamespace(mech_id=mid, db=db, transport_file=None,
                           estimate=True, export_chemkin=tran, dry_run=False)
    rc = _cmd_transport(args)
    assert rc == 0
    assert os.path.exists(tran)

    # the extra species must have been estimated + persisted
    conn = init_db(db)
    row = conn.execute(
        "SELECT transport_geometry, diameter_A, transport_source "
        "FROM species WHERE name='NC7H16'").fetchone()
    # H2O must have been read from Cantera (known), not estimated
    h2o = conn.execute(
        "SELECT transport_source FROM species WHERE name='H2O'").fetchone()
    conn.close()
    assert row[0] == "nonlinear"
    assert 5.0 < row[1] < 7.5           # n-heptane sigma ~6.3 Å
    assert row[2].startswith("estimated:")
    assert h2o[0] is None                # known-from-Cantera → not written as estimate
