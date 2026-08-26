"""
rmip/tests/test_error_map.py
============================
Tests for the skeletal-vs-detailed error atlas
(``rmip/analysis/error_map.py``, FEATURE_ROADMAP #11a).

A detailed h2o2 mechanism is parsed into a temp DB; two skeletal variants are
materialised — a benign one (drop the inert AR, chemically identical) and an
aggressive one (drop the HO2/H2O2 low-temperature chemistry) — and the error map
must show the aggressive one breaking where the benign one does not.

Run with:
    python -m pytest src/rmip/tests/test_error_map.py -v
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ct = pytest.importorskip("cantera")

from rmip.analysis.error_map import ErrorMapper, parse_grid, _parse_axis  # noqa: E402
from rmip.reduction.materialize import materialize_subset, mechanism_lists  # noqa: E402
from rmip.types import ErrorMapResult  # noqa: E402


# ---------------------------------------------------------------------------
# Grid parsing (pure)
# ---------------------------------------------------------------------------

def test_parse_axis_linspace():
    xs = _parse_axis("800:1600:5")
    assert xs == [800.0, 1000.0, 1200.0, 1400.0, 1600.0]


def test_parse_axis_explicit_and_scalar():
    assert _parse_axis("1,2,3") == [1.0, 2.0, 3.0]
    assert _parse_axis("1.5") == [1.5]


def test_parse_grid_cartesian_product():
    conds = parse_grid("T=1000:1400:3,P=1:10:2,phi=1.0", fuel="H2:1.0")
    assert len(conds) == 3 * 2 * 1
    assert {c.fuel for c in conds} == {"H2:1.0"}
    assert sorted({c.T_K for c in conds}) == [1000.0, 1200.0, 1400.0]
    assert sorted({c.P_atm for c in conds}) == [1.0, 10.0]


def test_parse_grid_defaults_missing_axes():
    conds = parse_grid("phi=0.5,1.0,1.5", fuel="H2:1.0", default_T=1100.0)
    assert len(conds) == 3
    assert all(c.T_K == 1100.0 and c.P_atm == 1.0 for c in conds)


# ---------------------------------------------------------------------------
# End-to-end error atlas
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o2_skeletals(tmp_path_factory):
    """Detailed h2o2 + a benign skeletal (drop AR) + an aggressive one (drop HO2/H2O2)."""
    from rmip.config import RMIPConfig
    from rmip.core.parser import MechanismParser

    db_path = str(tmp_path_factory.mktemp("errmap") / "em.db")
    parser = MechanismParser(RMIPConfig(db_path=db_path))
    detailed = parser.parse(mech_path=str(Path(ct.__path__[0]) / "data" / "h2o2.yaml"))
    parser.conn.close()

    names, formulas = mechanism_lists(db_path, detailed)
    benign = materialize_subset(
        db_path, detailed, species=[n for n in names if n != "AR"],
        reactions=formulas, new_name="benign")
    aggressive = materialize_subset(
        db_path, detailed,
        species=[n for n in names if n not in ("HO2", "H2O2", "AR")],
        reactions=formulas, new_name="aggressive")
    return db_path, detailed, benign, aggressive


def test_benign_skeletal_is_accurate(h2o2_skeletals):
    db_path, detailed, benign, _ = h2o2_skeletals
    conds = parse_grid("T=1100:1400:3,phi=1.0", fuel="H2:1.0")
    res = ErrorMapper(db_path).map_error(detailed, benign, "IDT", conds,
                                         tolerance=0.10, save_db=False)
    assert isinstance(res, ErrorMapResult)
    assert res.n_evaluated >= 3
    # dropping the inert AR (absent from the mixture) changes nothing
    assert res.max_rel_err < 0.02
    assert res.frac_within_tol == 1.0


def test_aggressive_skeletal_breaks_somewhere(h2o2_skeletals):
    db_path, detailed, benign, aggressive = h2o2_skeletals
    conds = parse_grid("T=1100:1400:3,phi=1.0", fuel="H2:1.0")
    benign_res = ErrorMapper(db_path).map_error(detailed, benign, "IDT", conds,
                                                tolerance=0.10, save_db=False)
    aggr_res = ErrorMapper(db_path).map_error(detailed, aggressive, "IDT", conds,
                                              tolerance=0.10, save_db=False)
    assert aggr_res.n_evaluated >= 1
    # removing HO2/H2O2 chemistry shifts IDT — strictly worse than the benign cut
    assert aggr_res.max_rel_err > benign_res.max_rel_err
    assert aggr_res.worst_cell is not None


def test_error_map_persists_and_saves(h2o2_skeletals, tmp_path):
    import json
    db_path, detailed, benign, _ = h2o2_skeletals
    conds = parse_grid("T=1200:1300:2,phi=1.0", fuel="H2:1.0")
    res = ErrorMapper(db_path).map_error(detailed, benign, "IDT", conds,
                                         save_db=True)
    assert res.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT n_conditions, sim_type FROM error_map_runs WHERE id=?",
            (res.run_id,)).fetchone()
    assert row[0] == 2 and row[1] == "IDT"

    out = tmp_path / "em.json"
    res.save(str(out))
    d = json.loads(out.read_text())
    assert len(d["cells"]) == 2
