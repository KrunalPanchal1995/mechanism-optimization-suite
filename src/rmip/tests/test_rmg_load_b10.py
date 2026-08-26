"""
rmip/tests/test_rmg_load_b10.py
================================
Regression test for B10 (bug.md / review.md deep review).

`RMGInterface.load_rmg_output` used to call a nonexistent
``MechanismParser.parse_and_store(mechanism_path=...)`` (the real API is
``MechanismParser(RMIPConfig(...)).parse(mech_path=...)``), and constructed the
parser with a bare db-path string instead of an ``RMIPConfig``. Both raised at
runtime, so the whole "load an already-generated RMG CHEMKIN mechanism" path --
and the ``rmip complete`` CLI that depends on it -- was dead.

This test loads a real (dummy H2/O2) CHEMKIN mechanism through
``load_rmg_output`` and asserts it lands in the DB with the right species and
reaction counts, so the AttributeError/kwarg mismatch can't silently return.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rmip.generation.rmg_interface import RMGInterface

_FX = Path(__file__).resolve().parent / "fixtures" / "gui_stress"


def test_load_rmg_output_stores_mechanism(tmp_path):
    db = tmp_path / "rmg.db"
    iface = RMGInterface(str(db))

    mech_id = iface.load_rmg_output(
        chemkin_path=str(_FX / "mech_A.inp"),
        thermo_path=str(_FX / "thermo_A.dat"),
        db_name="RMG_NC7H16",
    )

    # Must return a real DB id, not raise AttributeError / return None.
    assert isinstance(mech_id, int) and mech_id > 0

    conn = sqlite3.connect(str(db))
    try:
        name = conn.execute(
            "SELECT name FROM mechanisms WHERE id=?", (mech_id,)
        ).fetchone()
        assert name is not None and name[0] == "RMG_NC7H16"

        n_species = conn.execute(
            "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (mech_id,)
        ).fetchone()[0]
        n_rxn = conn.execute(
            "SELECT COUNT(*) FROM reactions WHERE mechanism_id=?", (mech_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    # mech_A.inp is the checked-in 9-species / 8-reaction dummy fixture.
    assert n_species == 9
    assert n_rxn == 8


def test_load_rmg_output_with_embedded_thermo(tmp_path):
    """thermo embedded in the .inp (thermo_path=None) must also work -- the
    `rmip complete` path passes thermo_path=None because RMG embeds thermo in
    its chem.inp rather than emitting a separate therm.dat."""
    # Build a combined mechanism: mech_A with thermo_A's THERMO block inlined
    # before REACTIONS (valid CHEMKIN), then load it with thermo_path=None.
    mech = (_FX / "mech_A.inp").read_text()
    thermo = (_FX / "thermo_A.dat").read_text()
    combined = mech.replace("REACTIONS", thermo.rstrip() + "\n\nREACTIONS", 1)
    combined_path = tmp_path / "chem_embedded.inp"
    combined_path.write_text(combined)

    db = tmp_path / "rmg2.db"
    iface = RMGInterface(str(db))
    mech_id = iface.load_rmg_output(
        chemkin_path=str(combined_path),
        thermo_path=None,
        db_name="RMG_embedded",
    )
    assert isinstance(mech_id, int) and mech_id > 0
