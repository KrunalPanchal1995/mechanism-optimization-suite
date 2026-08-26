"""
rmip/tests/test_graph_builder_m9.py
======================================
Regression coverage for the M9 (review.md) gap in graph_builder.py, the last
of the M9 remaining-sites survey (bug.md): two CHEMKIN DUPLICATE reactions
(same canonical_key, different kinetics) within one mechanism collided
wherever this module built an identity-keyed dict from `canonical_key`
alone --

- `build_union_graphs`: `rx_key_to_node[i][_rx_key(data)] = node_id` kept
  only the last-seen duplicate, so ghost-node computation for the OTHER
  mechanisms' panels lost one of the two rows entirely.
- `_assign_union_positions`: the compact `union_G` used for shared layout
  collapsed both duplicate rows to one node, so both ended up drawn at the
  identical (x, y) position -- visually indistinguishable/overlapping.
- `_assign_global_reaction_labels`: `global_id[_rx_key(data)]` assigned the
  SAME "R<n>" label to both duplicate rows (the literal "R-label collision"
  named in bug.md) -- two chemically distinct reactions displayed under one
  shared label with no way to tell them apart in the UI.

Fix: `occurrence_index` (0 for the first row at a given canonical_key
within a mechanism, 1 for the second, etc, in `id`/insertion order) is now
stamped onto every reaction node in `build_graph` and folded into every
`_rx_key()` identity string, so duplicate rows get distinct identities
throughout.
"""
from __future__ import annotations

import sqlite3


def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    return init_db(path)


def _insert_mechanism(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO mechanisms (name, format, filepath, description,
                                n_species, n_reactions, loaded_at, metadata_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (name, "chemkin", "/fake/path.inp", f"Test mechanism {name}", 0, 0,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, inchi_key=None):
    cur = conn.execute(
        """
        INSERT INTO species (mechanism_id, name, inchi_key, formula,
                             coeffs_low_json, coeffs_high_json)
        VALUES (?,?,?,?,?,?)
        """,
        (mech_id, name, inchi_key, name, "[1,0,0,0,0,0,0]", "[1,0,0,0,0,0,0]"),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id, formula_string, canonical_key, A_si):
    cur = conn.execute(
        """
        INSERT INTO reactions
            (mechanism_id, formula_string, canonical_key,
             reversible, kinetics_type, A_si, n, Ea_si, duplicate)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (mech_id, formula_string, canonical_key, 1, "arrhenius", A_si, 0.0, 50000.0, 1),
    )
    return cur.lastrowid


def _add_stoich(conn, rxn_id, sp_id, coeff, role):
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


def _build_db_with_duplicate(tmp_path):
    """m1 has a DUPLICATE-marked reaction (two rows, same canonical_key,
    different A_si). m2 has no reactions at all, just to force a real
    union (ghost-node) computation across >1 mechanism."""
    db_path = str(tmp_path / "m9_graph.db")
    conn = _make_db(db_path)
    m1 = _insert_mechanism(conn, "M1")
    m2 = _insert_mechanism(conn, "M2")

    a = _insert_species(conn, m1, "A")
    b = _insert_species(conn, m1, "B")
    _insert_species(conn, m2, "A")
    _insert_species(conn, m2, "B")

    r0 = _insert_reaction(conn, m1, "A<=>B", "R_DUP", A_si=1e10)
    r1 = _insert_reaction(conn, m1, "A<=>B", "R_DUP", A_si=2e10)
    for rid in (r0, r1):
        _add_stoich(conn, rid, a, 1.0, "reactant")
        _add_stoich(conn, rid, b, 1.0, "product")

    conn.commit()
    conn.close()
    return db_path, m1, m2, r0, r1


class TestM9OccurrenceIndexStampedOnBuild:
    def test_duplicate_reactions_get_distinct_occurrence_index(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, m1, m2, r0, r1 = _build_db_with_duplicate(tmp_path)
        gb = GraphBuilder(db_path)
        G = gb.build_graph(m1, assign_positions=False)

        occs = sorted(
            d["occurrence_index"] for _, d in G.nodes(data=True)
            if d.get("type") == "reaction"
        )
        assert occs == [0, 1]


class TestM9GlobalReactionLabelsNotCollided:
    def test_duplicate_rows_get_distinct_global_r_labels(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, m1, m2, r0, r1 = _build_db_with_duplicate(tmp_path)
        gb = GraphBuilder(db_path)
        graphs = gb.build_union_graphs([m1, m2])
        G1 = graphs[0]

        labels_by_dbid = {
            d["db_id"]: d.get("reaction_label")
            for _, d in G1.nodes(data=True)
            if d.get("type") == "reaction" and not d.get("ghost")
        }
        assert set(labels_by_dbid) == {r0, r1}
        # Pre-fix: both duplicate rows shared the same `_rx_key` (plain
        # canonical_key), so `global_id.get(k)` returned the SAME "R<n>"
        # label for both -- the literal R-label collision.
        assert labels_by_dbid[r0] != labels_by_dbid[r1]
        assert None not in labels_by_dbid.values()


class TestM9UnionGraphKeepsBothDuplicateRows:
    def test_both_duplicate_rows_survive_union_ghost_computation(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, m1, m2, r0, r1 = _build_db_with_duplicate(tmp_path)
        gb = GraphBuilder(db_path)
        graphs = gb.build_union_graphs([m1, m2])
        G2 = graphs[1]  # m2 has no own reactions -> both should be ghosts

        ghost_reactions = [
            d for _, d in G2.nodes(data=True)
            if d.get("type") == "reaction" and d.get("ghost")
        ]
        # Pre-fix: rx_key_to_node[0][_rx_key(data)] collapsed both m1 rows
        # onto one key, so only ONE ghost reaction (not two) would appear
        # in m2's panel, silently hiding the second DUPLICATE row from
        # every side-by-side comparison view.
        ghost_dbids = {d["db_id"] for d in ghost_reactions}
        assert ghost_dbids == {r0, r1}


class TestM9UnionPositionsDoNotOverlap:
    def test_duplicate_rows_get_distinct_layout_positions(self, tmp_path):
        from rmip.core.graph_builder import GraphBuilder
        db_path, m1, m2, r0, r1 = _build_db_with_duplicate(tmp_path)
        gb = GraphBuilder(db_path)
        graphs = gb.build_union_graphs([m1, m2], include_spring=False)
        G1 = graphs[0]

        pos_by_dbid = {
            d["db_id"]: (d["pos_x_circle"], d["pos_y_circle"])
            for _, d in G1.nodes(data=True)
            if d.get("type") == "reaction"
        }
        # Pre-fix: both duplicate rows resolved to the same union_G node
        # key, so pos_circle.get(k) returned the identical (x, y) for both
        # -- overlapping diamonds in every rendered panel.
        assert pos_by_dbid[r0] != pos_by_dbid[r1]
