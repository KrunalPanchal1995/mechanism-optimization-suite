"""
rmip/tests/test_phase5c_3d.py
=============================
Test suite for the Phase-5C 3D planetary layout and visualizer.

Covered
-------
  GraphBuilder
    - _fibonacci_sphere() — unit norm, count, determinism
    - _planetary_3d_positions() — stacked-disc layout: ring radius and
      z-layer grow with carbon number, C0 species on the innermost
      (non-zero) bottom ring, alpha/family angular ordering, reaction
      placement between layers
    - build_graph() assigns pos_x_3d / pos_y_3d / pos_z_3d (+ _fam) / carbon_n
    - build_union_graphs() — same identity key → same 3D position in
      every panel (own and ghost nodes)

  NetworkVisualizer3D
    - build_figure() returns a Plotly figure with the expected traces
    - export_html() writes a self-contained file
    - constructor rejects graphs without 3D positions

  CLI
    - visualize3d subcommand is registered

Run with:
    python -m pytest rmip/tests/test_phase5c_3d.py -v
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import pytest
import networkx as nx

plotly = pytest.importorskip("plotly", reason="plotly required for 3D tests")

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (mirrors test_phase5.py helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    return init_db(path)


def _insert_mechanism(conn, name, n_sp=3, n_rxn=2):
    cur = conn.execute(
        "INSERT INTO mechanisms (name, format, filepath, description, "
        "n_species, n_reactions, loaded_at, metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (name, "chemkin", "/fake.inp", f"Test {name}", n_sp, n_rxn,
         "2024-01-01T00:00:00Z", "{}"),
    )
    return cur.lastrowid


def _insert_species(conn, mech_id, name, inchi_key, formula="H2"):
    cur = conn.execute(
        "INSERT INTO species (mechanism_id, name, inchi_key, formula, "
        "canonical_smiles, glossary_matched, tmin_K, tmid_K, tmax_K) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mech_id, name, inchi_key, formula, None, 0, 200, 1000, 5000),
    )
    return cur.lastrowid


def _insert_reaction(conn, mech_id, formula_str, can_key=None):
    cur = conn.execute(
        "INSERT INTO reactions (mechanism_id, formula_string, canonical_key, "
        "reversible, A_si, n, Ea_si, kinetics_type, pressure_dependent, "
        "has_third_body) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (mech_id, formula_str, can_key, 1, 1e10, 0.0, 50000.0,
         "arrhenius", 0, 0),
    )
    return cur.lastrowid


def _insert_stoich(conn, rxn_id, sp_id, coeff, role):
    conn.execute(
        "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
        "VALUES (?,?,?,?)",
        (rxn_id, sp_id, coeff, role),
    )


def _tmp_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        return tf.name


def _small_mech(conn, name, extra_species=()):
    """Insert a small H2/CH4/C2H4 mechanism; return mech_id."""
    mid = _insert_mechanism(conn, name)
    sp = {}
    for nm, formula in [("H2", "H2"), ("CH4", "CH4"), ("C2H4", "C2H4"),
                        *extra_species]:
        sp[nm] = _insert_species(conn, mid, nm, f"IK_{nm}", formula=formula)
    rxn = _insert_reaction(conn, mid, "H2 + CH4 <=> C2H4",
                           can_key="CK_H2_CH4_C2H4")
    _insert_stoich(conn, rxn, sp["H2"], 1.0, "reactant")
    _insert_stoich(conn, rxn, sp["CH4"], 1.0, "reactant")
    _insert_stoich(conn, rxn, sp["C2H4"], 1.0, "product")
    conn.commit()
    return mid


# ─────────────────────────────────────────────────────────────────────────────
# Fibonacci sphere
# ─────────────────────────────────────────────────────────────────────────────

class TestFibonacciSphere:

    def test_unit_norm_and_count(self):
        from rmip.core.graph_builder import GraphBuilder
        for n in (1, 2, 5, 50):
            pts = GraphBuilder._fibonacci_sphere(n)
            assert len(pts) == n
            for x, y, z in pts:
                assert math.isclose(math.sqrt(x*x + y*y + z*z), 1.0,
                                    rel_tol=1e-9)

    def test_deterministic(self):
        from rmip.core.graph_builder import GraphBuilder
        assert GraphBuilder._fibonacci_sphere(17) == \
               GraphBuilder._fibonacci_sphere(17)


# ─────────────────────────────────────────────────────────────────────────────
# Planetary 3D positions
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanetary3DPositions:

    def _union(self, nodes, edges=()):
        G = nx.DiGraph()
        for k, typ, formula in nodes:
            G.add_node(k, node_type=typ, formula=formula)
        for u, v in edges:
            G.add_edge(u, v)
        return G

    @staticmethod
    def _rxy(pos, k):
        x, y, _ = pos[k]
        return math.hypot(x, y)

    def test_ring_radius_and_layer_grow_with_carbon_number(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union([
            ("nm:H2",   "species", "H2"),
            ("nm:CH4",  "species", "CH4"),
            ("nm:C2H4", "species", "C2H4"),
            ("nm:C5",   "species", "C5H10O2"),
        ])
        pos = gb._planetary_3d_positions(union_G, delta_r=300)

        # C0 → innermost ring (non-zero radius: it is a ring, not a point)
        # in the bottom plane (z = 0)
        assert math.isclose(self._rxy(pos, "nm:H2"), 150.0, rel_tol=1e-6)
        assert pos["nm:H2"][2] == 0.0
        # Ring radius strictly increases with carbon number
        assert (self._rxy(pos, "nm:H2") < self._rxy(pos, "nm:CH4")
                < self._rxy(pos, "nm:C2H4") < self._rxy(pos, "nm:C5"))
        # Default shell_gap = 2.0: Cn ring radius = delta_r * (0.5 + 2*n)
        assert math.isclose(self._rxy(pos, "nm:CH4"),  300 * 2.5,  rel_tol=1e-6)
        assert math.isclose(self._rxy(pos, "nm:C2H4"), 300 * 4.5,  rel_tol=1e-6)
        assert math.isclose(self._rxy(pos, "nm:C5"),   300 * 10.5, rel_tol=1e-6)
        # Default layer_gap = 1.0: Cn ring in the z = delta_r * n plane
        assert math.isclose(pos["nm:CH4"][2],  300.0,  rel_tol=1e-6)
        assert math.isclose(pos["nm:C2H4"][2], 600.0,  rel_tol=1e-6)
        assert math.isclose(pos["nm:C5"][2],   1500.0, rel_tol=1e-6)

    def test_shell_gap_controls_ring_spacing(self):
        """shell_gap scales the ring radii but leaves the C0 core unchanged."""
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union([
            ("nm:H2",  "species", "H2"),
            ("nm:CH4", "species", "CH4"),
        ])
        compact = gb._planetary_3d_positions(union_G, delta_r=300, shell_gap=1.0)
        assert math.isclose(self._rxy(compact, "nm:CH4"), 300 * 1.5, rel_tol=1e-6)

        wide = gb._planetary_3d_positions(union_G, delta_r=300, shell_gap=4.0)
        assert math.isclose(self._rxy(wide, "nm:CH4"), 300 * 4.5, rel_tol=1e-6)
        assert math.isclose(self._rxy(compact, "nm:H2"),
                            self._rxy(wide, "nm:H2"), rel_tol=1e-6)

    def test_layer_gap_controls_vertical_spacing(self):
        """layer_gap=0 flattens all rings into one plane."""
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union([
            ("nm:H2",  "species", "H2"),
            ("nm:CH4", "species", "CH4"),
        ])
        flat = gb._planetary_3d_positions(union_G, delta_r=300, layer_gap=0.0)
        assert flat["nm:H2"][2] == flat["nm:CH4"][2] == 0.0

        tall = gb._planetary_3d_positions(union_G, delta_r=300, layer_gap=2.0)
        assert math.isclose(tall["nm:CH4"][2], 600.0, rel_tol=1e-6)

    def test_family_order_differs_from_alpha(self):
        """Family ordering groups hydrocarbons before oxygenates on a ring."""
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union([
            ("nm:C2H5OH", "species", "C2H5OH"),   # oxygenate
            ("nm:C2H6",   "species", "C2H6"),     # hydrocarbon
            ("nm:CH3CHO", "species", "CH3CHO"),   # oxygenate
        ])
        alpha  = gb._planetary_3d_positions(union_G, order="alpha")
        family = gb._planetary_3d_positions(union_G, order="family")
        # Same rings (same xy-radius and z) in both orderings
        for k in alpha:
            assert math.isclose(self._rxy(alpha, k), self._rxy(family, k),
                                rel_tol=1e-9)
            assert alpha[k][2] == family[k][2]
        # But angular placement differs: alphabetical puts C2H5OH first
        # (angle 0), family ordering puts the hydrocarbon C2H6 there.
        assert alpha["nm:C2H5OH"] != family["nm:C2H5OH"]
        assert alpha["nm:C2H5OH"] == family["nm:C2H6"]

    def test_formula_preferred_over_name(self):
        """A species named 'MB' with formula C5H10O2 must land on the C5 ring."""
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union([("nm:MB", "species", "C5H10O2")])
        pos = gb._planetary_3d_positions(union_G, delta_r=300)
        assert math.isclose(self._rxy(pos, "nm:MB"), 300 * 10.5, rel_tol=1e-6)
        assert math.isclose(pos["nm:MB"][2], 1500.0, rel_tol=1e-6)

    def test_reaction_not_at_origin(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._union(
            [("nm:H2", "species", "H2"),
             ("nm:O2", "species", "O2"),
             ("ck:R1", "reaction", None)],
            edges=[("nm:H2", "ck:R1"), ("nm:O2", "ck:R1")],
        )
        pos = gb._planetary_3d_positions(union_G, delta_r=300)
        x, y, z = pos["ck:R1"]
        # Even if the neighbour centroid collapses, the reaction is pushed
        # off the origin so it remains visible.
        assert math.sqrt(x*x + y*y + z*z) > 1.0

    def test_shell_members_share_radius(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        names = [f"nm:SP{i}" for i in range(12)]
        union_G = self._union([(k, "species", "C3H8") for k in names])
        pos = gb._planetary_3d_positions(union_G, delta_r=200)
        radii = {round(math.sqrt(x*x + y*y + z*z), 6)
                 for x, y, z in (pos[k] for k in names)}
        assert len(radii) == 1
        # And the points are not all identical
        assert len({pos[k] for k in names}) == 12


# ─────────────────────────────────────────────────────────────────────────────
# GraphBuilder integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGraph3DAttributes:

    def test_build_graph_assigns_3d_positions(self):
        db = _tmp_db()
        conn = _make_db(db)
        mid = _small_mech(conn, "M1")

        from rmip.core.graph_builder import GraphBuilder
        G = GraphBuilder(db).build_graph(mid)
        for _, data in G.nodes(data=True):
            assert isinstance(data["pos_x_3d"], float)
            assert isinstance(data["pos_y_3d"], float)
            assert isinstance(data["pos_z_3d"], float)
            assert isinstance(data["pos_x_3d_fam"], float)
            assert isinstance(data["pos_y_3d_fam"], float)
            assert isinstance(data["pos_z_3d_fam"], float)
            if data["type"] == "species":
                assert isinstance(data["carbon_n"], int)

    def test_union_graphs_share_3d_positions(self):
        db = _tmp_db()
        conn = _make_db(db)
        m1 = _small_mech(conn, "M1")
        m2 = _small_mech(conn, "M2", extra_species=[("C3H8", "C3H8")])

        from rmip.core.graph_builder import GraphBuilder
        g1, g2 = GraphBuilder(db).build_union_graphs([m1, m2])

        def pos_by_identity(G):
            out = {}
            for _, d in G.nodes(data=True):
                key = (f"ik:{d['inchi_key']}" if d.get("inchi_key")
                       else d.get("canonical_key") or d.get("formula_string"))
                out[key] = (d["pos_x_3d"], d["pos_y_3d"], d["pos_z_3d"])
            return out

        p1, p2 = pos_by_identity(g1), pos_by_identity(g2)
        assert set(p1) == set(p2)          # union node inventory identical
        for k in p1:
            assert p1[k] == p2[k], f"3D position differs across panels for {k}"


# ─────────────────────────────────────────────────────────────────────────────
# NetworkVisualizer3D
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkVisualizer3D:

    def _graph(self):
        db = _tmp_db()
        conn = _make_db(db)
        mid = _small_mech(conn, "M1")
        from rmip.core.graph_builder import GraphBuilder
        return GraphBuilder(db).build_graph(mid)

    def test_build_figure_traces(self):
        from rmip.analysis.visualization3d import NetworkVisualizer3D
        viz = NetworkVisualizer3D(self._graph())
        fig = viz.build_figure()
        names = [t.name for t in fig.data]
        assert "Species (colour = carbon number)" in names
        assert "Reactions" in names
        assert "Edges" in names
        assert "Carbon orbits" in names

    def test_toggles_remove_traces(self):
        from rmip.analysis.visualization3d import NetworkVisualizer3D
        viz = NetworkVisualizer3D(self._graph())
        fig = viz.build_figure(show_edges=False, show_orbits=False)
        names = [t.name for t in fig.data]
        assert "Edges" not in names
        assert "Carbon orbits" not in names

    def test_rejects_graph_without_positions(self):
        from rmip.analysis.visualization3d import NetworkVisualizer3D
        G = nx.DiGraph()
        G.add_node("S_1", type="species", name="H2")
        with pytest.raises(ValueError, match="pos_x_3d"):
            NetworkVisualizer3D(G)

    def test_export_html(self, tmp_path):
        from rmip.analysis.visualization3d import NetworkVisualizer3D
        viz = NetworkVisualizer3D(self._graph())
        out = tmp_path / "g3d.html"
        viz.export_html(str(out))
        assert out.exists() and out.stat().st_size > 1000

    def test_at_most_two_graphs(self):
        from rmip.analysis.visualization3d import NetworkVisualizer3D
        G = self._graph()
        with pytest.raises(ValueError, match="at most 2"):
            NetworkVisualizer3D([G, G, G])


# ─────────────────────────────────────────────────────────────────────────────
# CLI registration
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIVisualize3D:

    def test_subcommand_registered(self):
        from rmip.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["visualize3d", "1", "2", "--db", "x.db", "--union",
             "--label", "id", "--port", "8051"]
        )
        assert args.mech_id == [1, 2]
        assert args.union is True
        assert args.label == "id"
        assert args.port == 8051
        assert args.shell_gap == 2.0
        assert args.layer_gap == 1.0
        assert args.order == "alpha"

    def test_layout_flags(self):
        from rmip.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(
            ["visualize3d", "1", "--db", "x.db", "--shell-gap", "3.5",
             "--layer-gap", "0.5", "--order", "family"]
        )
        assert args.shell_gap == 3.5
        assert args.layer_gap == 0.5
        assert args.order == "family"
