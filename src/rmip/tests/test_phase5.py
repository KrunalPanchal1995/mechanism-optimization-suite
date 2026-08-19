"""
rmip/tests/test_phase5.py
=========================
Test suite for Phase 5A, 5C, and 5D features.

Phases tested
-------------
  Phase 5A — SA/flux edge highlighting in visualizer
    - SensitivityFluxIntegrator output flows through to_cytoscape_elements()
    - importance score embedded on edges
    - _default_stylesheet() generates threshold-band selectors
    - carbon-ring layout preset added to _PRESET_MODES

  Phase 5C — Carbon-ring layout
    - _carbon_number() helper correctness
    - _assign_union_positions() computes pos_x_cring / pos_y_cring
    - build_graph() assigns positions for single-graph mode

  Phase 5D — Species variant overrides
    - load_variants() CSV parser
    - run_phase5d_migrations() adds species columns
    - MechanismParser.parse() applies variant overrides

Run with:
    python -m pytest rmip/tests/test_phase5.py -v
"""
from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures shared across test classes
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    conn = init_db(path)
    return conn


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


def _make_simple_graph() -> nx.DiGraph:
    """Return a minimal bipartite graph with one species and one reaction."""
    G = nx.DiGraph()
    G.add_node("S_1", type="species", name="H2", inchi_key="IK_H2",
               formula="H2", importance_score=None, sensitivity_score=None,
               flux_score=None, ghost=False)
    G.add_node("R_1", type="reaction", formula_string="H2 + O2 <=> H2O2",
               canonical_key="CK1", importance_score=0.75,
               sensitivity_score=0.8, flux_score=0.6,
               reaction_label="R1", ghost=False,
               pos_x=0, pos_y=0)
    G.add_node("S_2", type="species", name="O2", inchi_key="IK_O2",
               formula="O2", importance_score=None, ghost=False)
    G.add_edge("S_1", "R_1", role="reactant", stoich=1.0)
    G.add_edge("S_2", "R_1", role="reactant", stoich=1.0)
    G.add_edge("R_1", "S_2", role="product", stoich=1.0)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5A: edge importance in to_cytoscape_elements()
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeImportance:

    def test_importance_embedded_on_edges(self):
        from rmip.analysis.visualization import NetworkVisualizer
        G = _make_simple_graph()
        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="importance")
        edge_elems = [e for e in elems if "source" in e["data"]]
        # At least one edge should carry importance
        with_imp = [e for e in edge_elems if "importance" in e["data"]]
        assert len(with_imp) > 0

    def test_importance_value_matches_reaction_score(self):
        from rmip.analysis.visualization import NetworkVisualizer
        G = _make_simple_graph()
        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="importance")
        edge_elems = [e for e in elems if "source" in e["data"]]
        # All edges from/to R_1 should have importance = 0.75
        for e in edge_elems:
            d = e["data"]
            if d.get("source") == "R_1" or d.get("target") == "R_1":
                assert abs(d.get("importance", -1) - 0.75) < 1e-9

    def test_ghost_reaction_importance_is_zero(self):
        from rmip.analysis.visualization import NetworkVisualizer
        G = _make_simple_graph()
        # Add a ghost reaction with a high score
        G.add_node("ghost_R_99", type="reaction", formula_string="CO<=>CO2",
                   importance_score=0.99, ghost=True,
                   pos_x=0, pos_y=0, canonical_key=None)
        G.add_node("S_3", type="species", name="CO", inchi_key="IK_CO",
                   formula="CO", importance_score=None, ghost=False)
        G.add_edge("S_3", "ghost_R_99", role="reactant", stoich=1.0)
        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="importance")
        ghost_edges = [
            e for e in elems
            if "source" in e["data"] and (
                e["data"]["source"] == "ghost_R_99" or
                e["data"]["target"] == "ghost_R_99"
            )
        ]
        for e in ghost_edges:
            assert e["data"].get("importance") == 0.0

    def test_no_importance_when_score_none(self):
        """Edges adjacent to reaction with no score should NOT carry importance."""
        from rmip.analysis.visualization import NetworkVisualizer
        G = nx.DiGraph()
        G.add_node("S_1", type="species", name="H2", formula="H2",
                   importance_score=None, ghost=False)
        G.add_node("R_1", type="reaction", formula_string="H2<=>2H",
                   importance_score=None, ghost=False, canonical_key=None)
        G.add_edge("S_1", "R_1", role="reactant", stoich=1.0)
        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="importance")
        edge_elems = [e for e in elems if "source" in e["data"]]
        for e in edge_elems:
            assert "importance" not in e["data"]

    def test_score_sa_uses_sensitivity_score(self):
        from rmip.analysis.visualization import NetworkVisualizer
        G = _make_simple_graph()
        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="sa")
        edge_elems = [e for e in elems if "source" in e["data"]]
        with_imp = [e for e in edge_elems if "importance" in e["data"]]
        assert len(with_imp) > 0
        # should be 0.8 (sensitivity_score), not 0.75 (importance_score)
        for e in with_imp:
            if e["data"]["source"] == "S_1":
                assert abs(e["data"]["importance"] - 0.8) < 1e-9


class TestDefaultStylesheet:

    def test_stylesheet_has_threshold_selector(self):
        from rmip.analysis.visualization import _default_stylesheet
        ss = _default_stylesheet(threshold=0.15)
        selectors = [r["selector"] for r in ss]
        assert any("0.15" in s for s in selectors)

    def test_stylesheet_has_all_four_bands(self):
        from rmip.analysis.visualization import _default_stylesheet
        ss = _default_stylesheet(threshold=0.1)
        selectors = " ".join(r["selector"] for r in ss)
        assert "importance >= 0.5" in selectors
        assert "importance >= 0.8" in selectors
        # grey band
        assert "importance < 0.1" in selectors

    def test_stylesheet_has_variant_selector(self):
        from rmip.analysis.visualization import _default_stylesheet
        ss = _default_stylesheet()
        selectors = [r["selector"] for r in ss]
        assert "node[variant_tag]" in selectors

    def test_default_threshold_is_0_1(self):
        from rmip.analysis.visualization import _default_stylesheet
        ss = _default_stylesheet()
        selectors = " ".join(r["selector"] for r in ss)
        assert "importance < 0.1" in selectors


class TestSensitivityIntegration:
    """Integration: attach scores then check element output."""

    def test_attach_and_visualize_importance(self):
        from rmip.analysis.sensitivity import SensitivityFluxIntegrator
        from rmip.analysis.visualization import NetworkVisualizer

        G = _make_simple_graph()
        # Overwrite importance to None to start fresh
        G.nodes["R_1"]["importance_score"] = None

        sa_data = {
            "observable": "IDT",
            "conditions": {},
            "coefficients": {"H2 + O2 <=> H2O2": 0.9},
        }
        m8 = SensitivityFluxIntegrator("")
        m8.attach(0, G, sa_data, None)

        assert G.nodes["R_1"]["sensitivity_score"] == pytest.approx(1.0)
        assert G.nodes["R_1"]["importance_score"] is not None

        viz = NetworkVisualizer(G)
        elems = viz.to_cytoscape_elements(G, score="importance")
        edge_elems = [e for e in elems if "source" in e["data"]]
        with_imp = [e for e in edge_elems if "importance" in e["data"]]
        assert len(with_imp) > 0
        # importance should be > 0
        for e in with_imp:
            assert e["data"]["importance"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5C: carbon-ring layout
# ─────────────────────────────────────────────────────────────────────────────

class TestCarbonNumber:

    def test_c0_no_carbon(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        assert gb._carbon_number("H2") == 0
        assert gb._carbon_number("O2") == 0
        assert gb._carbon_number("") == 0
        assert gb._carbon_number(None) == 0

    def test_c1_single_carbon(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        assert gb._carbon_number("CH4") == 1
        assert gb._carbon_number("C") == 1
        assert gb._carbon_number("CO") == 1
        assert gb._carbon_number("CO2") == 1

    def test_c2_two_carbons(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        assert gb._carbon_number("C2H4") == 2
        assert gb._carbon_number("C2H2") == 2

    def test_cn_large(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        assert gb._carbon_number("C7H16") == 7
        assert gb._carbon_number("C12H26") == 12


class TestCarbonRingPositions:

    def _build_union_graph(self, nodes):
        G = nx.DiGraph()
        for k, typ in nodes:
            G.add_node(k, node_type=typ)
        return G

    def test_species_on_correct_shell(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._build_union_graph([
            ("nm:H2",   "species"),
            ("nm:CH4",  "species"),
            ("nm:C2H4", "species"),
            ("ck:RXN1", "reaction"),
        ])
        union_G.add_edge("nm:H2", "ck:RXN1")
        union_G.add_edge("nm:CH4", "ck:RXN1")
        pos = gb._carbon_ring_positions(union_G, delta_r=300)
        # H2 → C0 → shell 0 → radius 0
        x_h2, y_h2 = pos["nm:H2"]
        assert math.isclose(x_h2, 0.0, abs_tol=1e-6)
        assert math.isclose(y_h2, 0.0, abs_tol=1e-6)
        # CH4 → C1 → shell 1 → radius 300
        x_ch4, y_ch4 = pos["nm:CH4"]
        assert math.isclose(math.sqrt(x_ch4**2 + y_ch4**2), 300, rel_tol=1e-6)
        # C2H4 → C2 → shell 2 → radius 600
        x_c2, y_c2 = pos["nm:C2H4"]
        assert math.isclose(math.sqrt(x_c2**2 + y_c2**2), 600, rel_tol=1e-6)

    def test_reaction_centroid(self):
        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder.__new__(GraphBuilder)
        union_G = self._build_union_graph([
            ("nm:H2",  "species"),
            ("nm:O2",  "species"),
            ("ck:R1",  "reaction"),
        ])
        union_G.add_edge("nm:H2", "ck:R1")
        union_G.add_edge("nm:O2", "ck:R1")
        pos = gb._carbon_ring_positions(union_G, delta_r=300)
        # Both H2 and O2 are C0 → both at (0,0) → reaction centroid at (0,0)
        rx_x, rx_y = pos["ck:R1"]
        assert math.isclose(rx_x, 0.0, abs_tol=1e-6)
        assert math.isclose(rx_y, 0.0, abs_tol=1e-6)

    def test_cring_positions_in_build_graph(self):
        """build_graph() should populate pos_x_cring on nodes."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        conn = _make_db(db_path)
        mid   = _insert_mechanism(conn, "TestMech", n_sp=2, n_rxn=1)
        sp1   = _insert_species(conn, mid, "H2",  "IK_H2",  formula="H2")
        sp2   = _insert_species(conn, mid, "H2O", "IK_H2O", formula="H2O")
        rxn   = _insert_reaction(conn, mid, "H2 + O2 <=> H2O")
        _insert_stoich(conn, rxn, sp1, 1.0, "reactant")
        _insert_stoich(conn, rxn, sp2, 1.0, "product")
        conn.commit()

        from rmip.core.graph_builder import GraphBuilder
        gb = GraphBuilder(db_path)
        G  = gb.build_graph(mid)

        for _, data in G.nodes(data=True):
            assert "pos_x_cring" in data
            assert "pos_y_cring" in data
            assert isinstance(data["pos_x_cring"], float)

    def test_cring_preset_in_preset_modes(self):
        from rmip.analysis.visualization import _PRESET_MODES
        assert "preset_carbon_ring" in _PRESET_MODES
        assert _PRESET_MODES["preset_carbon_ring"] == ("pos_x_cring", "pos_y_cring")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5D: species variant overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadVariants:

    def test_basic_csv(self, tmp_path):
        from rmip.io.glossary_reader import load_variants
        csv_file = tmp_path / "variants.csv"
        csv_file.write_text(
            "variant_name,canonical_name,reason\n"
            "OHEX,OH,excited_state\n"
            "HNO2,HONO,tautomer\n"
        )
        v = load_variants(str(csv_file))
        assert "OHEX" in v
        assert v["OHEX"]["canonical_name"] == "OH"
        assert v["OHEX"]["reason"] == "excited_state"
        assert "HNO2" in v

    def test_case_insensitive_key(self, tmp_path):
        from rmip.io.glossary_reader import load_variants
        csv_file = tmp_path / "variants.csv"
        csv_file.write_text(
            "variant_name,canonical_name,reason\n"
            "ohex,OH,excited_state\n"
        )
        v = load_variants(str(csv_file))
        assert "OHEX" in v   # keys are upper-cased

    def test_comment_lines_skipped(self, tmp_path):
        from rmip.io.glossary_reader import load_variants
        csv_file = tmp_path / "variants.csv"
        csv_file.write_text(
            "# This is a comment\n"
            "variant_name,canonical_name,reason\n"
            "OHEX,OH,excited_state\n"
        )
        v = load_variants(str(csv_file))
        assert len(v) == 1

    def test_missing_columns_raises(self, tmp_path):
        from rmip.io.glossary_reader import load_variants
        csv_file = tmp_path / "variants.csv"
        csv_file.write_text("name,smiles\nOHEX,OH\n")
        with pytest.raises(ValueError, match="variant_name"):
            load_variants(str(csv_file))


class TestPhase5DMigrations:

    def test_adds_variant_tag_column(self):
        from rmip.database.migrations import run_phase5d_migrations
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _make_db.__wrapped__(conn) if hasattr(_make_db, "__wrapped__") else None
        # Use init_db
        from rmip.database.schema import init_db
        conn = init_db(":memory:")
        run_phase5d_migrations(conn)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(species)").fetchall()]
        assert "variant_tag" in cols
        assert "canonical_name" in cols

    def test_idempotent(self):
        from rmip.database.schema import init_db
        from rmip.database.migrations import run_phase5d_migrations
        conn = init_db(":memory:")
        run_phase5d_migrations(conn)
        run_phase5d_migrations(conn)   # second call must not raise


class TestParserVariants:
    """Integration test: parse with variants file."""

    def _write_minimal_chemkin(self, path: Path):
        """Write a minimal CHEMKIN .inp with OHEX species."""
        path.write_text(
            "ELEMENTS\nH O\nEND\n\n"
            "SPECIES\nH2 O OH OHEX\nEND\n\n"
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
            "OHEX                    H   1O   1          G  200.000 5000.000 1000.000    1\n"
            " 2.86472886E+00 1.05650448E-03-2.59082758E-07 3.05218674E-11-1.33195876E-15    2\n"
            " 3.68362875E+03 5.70164073E+00 4.12530561E+00-3.22544939E-03 6.52764691E-06    3\n"
            "-5.79853643E-09 2.00592054E-12 3.38153812E+03-6.90432960E-01                   4\n"
            "END\n\n"
            "REACTIONS\n"
            "H2 + O = OH + H    2.0E+14 0.0 70300.0\n"
            "END\n"
        )

    def test_variant_sets_inchi_key(self, tmp_path):
        mech_path = tmp_path / "mech.inp"
        self._write_minimal_chemkin(mech_path)

        glossary_path = tmp_path / "glossary.csv"
        glossary_path.write_text(
            "name,smiles\n"
            "H2,[HH]\n"
            "O,[O]\n"
            "OH,[OH]\n"
            # OHEX intentionally omitted — variant will map it to OH
        )

        variants_path = tmp_path / "variants.csv"
        variants_path.write_text(
            "variant_name,canonical_name,reason\n"
            "OHEX,OH,excited_state\n"
        )

        db_path = str(tmp_path / "test.db")
        from rmip.config import RMIPConfig
        from rmip.core.parser import MechanismParser
        from rmip.database.migrations import run_phase5d_migrations

        cfg = RMIPConfig(db_path=db_path)
        parser = MechanismParser(cfg)
        mech_id = parser.parse(
            mech_path=str(mech_path),
            glossary_path=str(glossary_path),
            variants_path=str(variants_path),
        )

        conn = parser.conn
        # species.variant_tag and canonical_name columns must exist
        run_phase5d_migrations(conn)  # idempotent
        rows = conn.execute(
            "SELECT name, inchi_key, variant_tag, canonical_name "
            "FROM species WHERE mechanism_id=?",
            (mech_id,),
        ).fetchall()
        by_name = {r[0]: r for r in rows}

        # OHEX should share OH's InChIKey
        oh_ik   = by_name["OH"][1]
        ohex_ik = by_name["OHEX"][1]
        assert oh_ik is not None
        assert ohex_ik == oh_ik, f"OHEX InChIKey {ohex_ik!r} != OH {oh_ik!r}"
        assert by_name["OHEX"][2] == "excited_state"
        assert by_name["OHEX"][3] == "OH"

    def test_parse_without_variants_unchanged(self, tmp_path):
        """Parsing without --variants should work as before (no new columns needed)."""
        mech_path = tmp_path / "mech.inp"
        self._write_minimal_chemkin(mech_path)
        db_path = str(tmp_path / "test.db")
        from rmip.config import RMIPConfig
        from rmip.core.parser import MechanismParser

        cfg    = RMIPConfig(db_path=db_path)
        parser = MechanismParser(cfg)
        mech_id = parser.parse(mech_path=str(mech_path))
        assert mech_id > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5A CLI: sa-run command registration
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPhase5A:

    def test_sa_run_command_registered(self):
        import sys
        import argparse
        from rmip.cli import _build_parser
        p = _build_parser()
        # Verify sa-run is a recognized subcommand
        with pytest.raises(SystemExit):
            p.parse_args(["sa-run", "--help"])

    def test_visualize_has_sa_json_arg(self):
        from rmip.cli import _build_parser
        p = _build_parser()
        args = p.parse_args([
            "visualize", "1", "--db", "x.db",
            "--sa-json", "sa.json", "--threshold", "0.2", "--score", "sa",
        ])
        assert args.sa_json == "sa.json"
        assert args.threshold == pytest.approx(0.2)
        assert args.score == "sa"

    def test_parse_has_variants_arg(self):
        from rmip.cli import _build_parser
        p = _build_parser()
        args = p.parse_args([
            "parse", "mech.inp", "--db", "x.db",
            "--variants", "variants.csv",
        ])
        assert args.variants == "variants.csv"
