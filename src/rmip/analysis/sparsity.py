"""
rmip/analysis/sparsity.py
=========================
Module 4: Sparsity / Missing Chemistry Detector

What it does
------------
Given the bipartite DiGraph from M2 (GraphBuilder), this module:

  1. Identifies **dead-end sources** — species that appear ONLY as reactants
     (in-degree 0 in the reaction network sense: no reaction produces them).
     These are expected for true reactants (fuel, O₂) but suspicious for
     intermediates.

  2. Identifies **dead-end sinks** — species that appear ONLY as products
     (out-degree 0): never consumed.  Terminal species like CO₂, H₂O are
     intentional sinks; unusual intermediates flagged here are chemistry gaps.

  3. Finds **orphan reactions** — reactions where EVERY reactant is a dead-end
     source (the reaction is isolated from the rest of the network).

  4. Finds **isolated nodes** — nodes with no edges at all.

  5. Reports **connected components** (weakly connected in the bipartite graph).

  6. Computes **coverage fraction** — the fraction of species nodes that have
     a directed path to any designated "product" species (CO₂, H₂O, N₂, etc.)
     via a forward BFS.

  7. Cross-mechanism **SparsityComparison** — for N mechanisms, reports which
     species / reactions appear in one but not another (by InChI key /
     canonical_key respectively).

  8. Writes M4 attributes back onto graph nodes:
       Species: ``is_dead_end_source``, ``is_dead_end_sink``, ``can_reach_products``
       Reaction: ``in_orphan_chain``

Node-ID convention (from GraphBuilder):
  Species  →  "S_{db_id}"
  Reaction →  "R_{db_id}"

Usage
-----
>>> from rmip.analysis.sparsity import SparsityDetector
>>> detector = SparsityDetector("project.db")
>>> G   = GraphBuilder("project.db").build_graph(1)
>>> rep = detector.analyze(mechanism_id=1, G=G)
>>> print(rep.n_dead_end_sinks)
>>> detector.update_graph_attributes(G, rep)
>>> cmp = detector.compare_all([1, 2, 3])
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

import networkx as nx

from rmip.types import (
    SparsityComparison,
    SparsityReport,
    SpeciesClass,
    ReactionClass,
)

logger = logging.getLogger(__name__)

# Species that are expected final products — having them as sinks is intentional
_DEFAULT_PRODUCT_LABELS: Set[str] = {
    "CO2", "H2O", "N2", "CO", "O2", "H2", "NO", "NO2", "SO2",
    "AR", "HE", "H2O2",
}


class SparsityDetector:
    """
    M4: Sparsity / Missing Chemistry Detector.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    known_product_inchi_keys : set[str], optional
        InChI keys of species considered terminal products.
        If not provided, the detector uses a heuristic: species with
        canonical_smiles in ``_DEFAULT_PRODUCT_LABELS`` plus any species
        with out-degree 0 whose formula matches common combustion products.
    """

    def __init__(
        self,
        db_path: str,
        known_product_inchi_keys: Optional[Set[str]] = None,
    ) -> None:
        self._db_path  = db_path
        self._prod_ik  = known_product_inchi_keys or set()

    # ------------------------------------------------------------------
    # Primary Analysis
    # ------------------------------------------------------------------

    def analyze(self, mechanism_id: int, G: nx.DiGraph) -> SparsityReport:
        """
        Analyse the sparsity of one mechanism's bipartite reaction graph.

        Parameters
        ----------
        mechanism_id : int
            DB id of the mechanism.
        G : nx.DiGraph
            Bipartite DiGraph from `GraphBuilder.build_graph()`.

        Returns
        -------
        SparsityReport
        """
        species_nodes  = {n for n, d in G.nodes(data=True) if d.get("type") == "species"}
        reaction_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "reaction"}

        # Edges in bipartite graph:
        #   species → reaction  (species is reactant)
        #   reaction → species  (species is product)
        dead_end_sources = []
        dead_end_sinks   = []

        for sn in species_nodes:
            in_edges  = G.in_degree(sn)    # edges from reactions → species
            out_edges = G.out_degree(sn)   # edges from species → reactions
            if in_edges == 0:
                dead_end_sources.append(sn)
            if out_edges == 0:
                dead_end_sinks.append(sn)

        # Isolated nodes (no edges at all)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]

        # Orphan reactions: all reactant edges come from dead-end sources
        des_set = set(dead_end_sources)
        orphan_reactions = []
        for rn in reaction_nodes:
            # predecessors of a reaction node = its reactant species
            reactants = list(G.predecessors(rn))
            if reactants and all(r in des_set for r in reactants):
                orphan_reactions.append(rn)

        # Connected components (undirected view for connectivity)
        components = [
            list(comp)
            for comp in nx.weakly_connected_components(G)
        ]

        # Coverage fraction: fraction of species with path to any product-like node
        prod_nodes   = self._identify_product_nodes(G, mechanism_id, dead_end_sinks)
        reachable    = self._species_reaching_products(G, prod_nodes, species_nodes)
        coverage     = len(reachable) / max(len(species_nodes), 1)

        report = SparsityReport(
            mechanism_id          = mechanism_id,
            n_species             = len(species_nodes),
            n_reactions           = len(reaction_nodes),
            dead_end_source_nodes = sorted(dead_end_sources),
            dead_end_sink_nodes   = sorted(dead_end_sinks),
            isolated_nodes        = sorted(isolated),
            orphan_reaction_nodes = sorted(orphan_reactions),
            connected_components  = [sorted(c) for c in components],
            coverage_fraction     = coverage,
        )
        logger.info(
            "Sparsity [mech %d]: %d species, %d reactions, "
            "%d dead-end sources, %d dead-end sinks, coverage=%.1f%%",
            mechanism_id,
            report.n_species, report.n_reactions,
            report.n_dead_end_sources, report.n_dead_end_sinks,
            coverage * 100,
        )
        return report

    # ------------------------------------------------------------------
    # Cross-Mechanism Comparison
    # ------------------------------------------------------------------

    def compare_all(
        self,
        mechanism_ids: List[int],
    ) -> SparsityComparison:
        """
        Compare the sparsity signature of N mechanisms.

        For each mechanism, retrieves its species (by InChI key) and reactions
        (by canonical_key) directly from the DB — no graph required.

        Returns
        -------
        SparsityComparison
            ``species_exclusive[mech_id]``  → InChI keys only in that mechanism
            ``reactions_exclusive[mech_id]`` → canonical_keys only in that mechanism
            ``species_common``              → InChI keys in ALL mechanisms
            ``reactions_common``            → canonical_keys in ALL mechanisms
        """
        conn = self._connect()
        try:
            mech_species   = {}
            mech_reactions = {}
            for mid in mechanism_ids:
                mech_species[mid]   = set(self._get_inchi_keys(conn, mid))
                mech_reactions[mid] = set(self._get_canonical_keys(conn, mid))

        finally:
            conn.close()

        all_species   = set.union(*mech_species.values())   if mech_species else set()
        all_reactions = set.union(*mech_reactions.values()) if mech_reactions else set()

        species_common   = set.intersection(*mech_species.values())   if mech_species else set()
        reactions_common = set.intersection(*mech_reactions.values()) if mech_reactions else set()

        species_exclusive   = {
            mid: sorted(mech_species[mid] - species_common)
            for mid in mechanism_ids
        }
        reactions_exclusive = {
            mid: sorted(mech_reactions[mid] - reactions_common)
            for mid in mechanism_ids
        }

        return SparsityComparison(
            mechanism_ids         = list(mechanism_ids),
            species_exclusive     = species_exclusive,
            reactions_exclusive   = reactions_exclusive,
            species_common        = sorted(species_common),
            reactions_common      = sorted(reactions_common),
        )

    # ------------------------------------------------------------------
    # Graph Attribute Annotation
    # ------------------------------------------------------------------

    def update_graph_attributes(
        self,
        G:      nx.DiGraph,
        report: SparsityReport,
    ) -> nx.DiGraph:
        """
        Write M4 analysis results back onto graph node attributes.

        Sets on species nodes:
            ``is_dead_end_source`` : bool
            ``is_dead_end_sink``   : bool
            ``can_reach_products`` : bool

        Sets on reaction nodes:
            ``in_orphan_chain`` : bool

        Returns the same graph (mutated in-place).
        """
        des_set     = set(report.dead_end_source_nodes)
        sink_set    = set(report.dead_end_sink_nodes)
        orphan_set  = set(report.orphan_reaction_nodes)

        for node, data in G.nodes(data=True):
            if data.get("type") == "species":
                data["is_dead_end_source"] = node in des_set
                data["is_dead_end_sink"]   = node in sink_set
                # can_reach_products is set by _species_reaching_products;
                # we store it here from the report's coverage insight
                # (detailed per-node info is in the graph from the coverage BFS)
                if "can_reach_products" not in data:
                    data["can_reach_products"] = not (node in sink_set)

            elif data.get("type") == "reaction":
                data["in_orphan_chain"] = node in orphan_set

        return G

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(self, report: SparsityReport, output_path: str) -> None:
        """Save the SparsityReport as a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Sparsity report saved: %s", path)

    def print_summary(self, report: SparsityReport) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  SPARSITY REPORT  —  Mechanism ID {report.mechanism_id}")
        print(f"{'='*60}")
        print(f"  Species           : {report.n_species}")
        print(f"  Reactions         : {report.n_reactions}")
        print(f"  Dead-end sources  : {report.n_dead_end_sources}")
        print(f"  Dead-end sinks    : {report.n_dead_end_sinks}")
        print(f"  Isolated nodes    : {report.n_isolated}")
        print(f"  Orphan reactions  : {len(report.orphan_reaction_nodes)}")
        print(f"  Connected compts  : {report.n_components}")
        print(f"  Coverage fraction : {report.coverage_fraction:.1%}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_inchi_keys(self, conn: sqlite3.Connection, mech_id: int) -> List[str]:
        rows = conn.execute(
            "SELECT inchi_key FROM species WHERE mechanism_id=? AND inchi_key IS NOT NULL",
            (mech_id,),
        ).fetchall()
        return [r["inchi_key"] for r in rows]

    def _get_canonical_keys(self, conn: sqlite3.Connection, mech_id: int) -> List[str]:
        rows = conn.execute(
            "SELECT canonical_key FROM reactions WHERE mechanism_id=? AND canonical_key IS NOT NULL",
            (mech_id,),
        ).fetchall()
        return [r["canonical_key"] for r in rows]

    def _identify_product_nodes(
        self,
        G:           nx.DiGraph,
        mech_id:     int,
        sink_nodes:  List[str],
    ) -> Set[str]:
        """
        Return the set of species nodes considered final combustion products.
        Uses a two-pass heuristic:
          1. Nodes whose `name` matches _DEFAULT_PRODUCT_LABELS.
          2. Nodes in the DB whose formula suggests CO₂/H₂O/N₂.
        """
        prod_nodes: Set[str] = set()
        for node, data in G.nodes(data=True):
            if data.get("type") != "species":
                continue
            name    = str(data.get("name", "")).upper()
            formula = str(data.get("formula", "")).upper()
            # Match by name
            if name in _DEFAULT_PRODUCT_LABELS:
                prod_nodes.add(node)
            # Match by formula
            elif formula in {"CO2", "H2O", "N2", "H2O2"}:
                prod_nodes.add(node)
            # InChI key match from user-supplied product set
            elif data.get("inchi_key") in self._prod_ik:
                prod_nodes.add(node)

        # If no products identified, fall back to sink nodes
        if not prod_nodes:
            logger.warning(
                "No product nodes identified; using all dead-end sinks as products"
            )
            prod_nodes = set(sink_nodes)

        return prod_nodes

    def _species_reaching_products(
        self,
        G:             nx.DiGraph,
        product_nodes: Set[str],
        species_nodes: Set[str],
    ) -> Set[str]:
        """
        BFS from each product node in the REVERSED graph to find all
        species that have a directed path (forward) to at least one product.
        """
        G_rev    = G.reverse(copy=False)
        reachable: Set[str] = set()

        for prod in product_nodes:
            if prod not in G_rev:
                continue
            for node in nx.bfs_tree(G_rev, prod).nodes():
                if node in species_nodes:
                    reachable.add(node)
                    # Also mark can_reach_products directly on the graph
                    G.nodes[node]["can_reach_products"] = True

        # Mark unreachable species
        for sn in species_nodes - reachable:
            if "can_reach_products" not in G.nodes[sn]:
                G.nodes[sn]["can_reach_products"] = False

        return reachable
