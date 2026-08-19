"""
rmip/core/graph_builder.py
==========================
Module 2: Reaction Network Graph Builder.

Constructs a bipartite directed graph from the RMIP database where:
  - Species are nodes: ``S_{db_id}``
  - Reactions are nodes: ``R_{db_id}``
  - Directed edges: species→reaction (reactant) and reaction→species (product)

Usage::

    from rmip.core.graph_builder import GraphBuilder

    gb    = GraphBuilder("project.db")
    G     = gb.build_graph(mech_id=1, include_reverse_edges=True)
    stats = gb.get_stats(G)
    gb.export_graphml(G, "mechanism.graphml")
    gb.export_json(G, "mechanism.json")

    # Subgraph / pathway queries
    sub   = gb.get_species_subgraph(G, ["NC7H16", "C7H15"])
    paths = gb.find_paths_between_species(G, "NC7H16", "CO2")
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from rmip.database.schema import get_connection, get_reactions_for_mechanism, get_species_for_mechanism

log = logging.getLogger("rmip.core.graph_builder")


def _canon_formula(formula_string: str) -> str:
    """
    Return a mechanism-agnostic canonical form of a CHEMKIN formula string
    for use as a reaction identity key.

    Normalisation steps
    -------------------
    1. Collapse whitespace.
    2. Unify arrow notation  (``=``, ``=>``, ``<=>``)  →  ``<=>``.
    3. Sort species tokens alphabetically within each side.

    Examples
    --------
    ``"H + O2 <=> OH + O"``  and  ``"O2 + H = O + OH"``
    both become  ``"H + O2 <=> O + OH"``.

    Stoichiometric coefficients (``"2 OH"``) are kept as-is and sorted
    lexicographically along with the species name, so  ``"2 OH"``  sorts
    before  ``"HO2"``.
    """
    fs = " ".join(formula_string.split())
    # Determine and strip arrow
    for arrow in (" <=> ", "<=>", " => ", "=>", " = ", "="):
        if arrow in fs:
            lhs, rhs = fs.split(arrow, 1)
            break
    else:
        return fs.lower()
    # Sort each side's species tokens
    def _sort_side(side: str) -> str:
        tokens = [t.strip() for t in side.split("+") if t.strip()]
        return " + ".join(sorted(tokens))
    return _sort_side(lhs) + " <=> " + _sort_side(rhs)


class GraphBuilder:
    """
    Builds a bipartite directed reaction network graph from the RMIP DB.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn    = get_connection(db_path)

    # ------------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------------

    def build_graph(
        self,
        mech_id:              int,
        include_reverse_edges: bool = True,
    ) -> nx.DiGraph:
        """
        Build and return the bipartite reaction network for mechanism *mech_id*.

        Parameters
        ----------
        mech_id : int
            Database mechanism id.
        include_reverse_edges : bool
            If True, add reverse edges for reversible reactions.

        Returns
        -------
        nx.DiGraph
        """
        G = nx.DiGraph()
        G.graph["mechanism_id"] = mech_id
        G.graph["db_path"]      = self._db_path

        # ---- Add species nodes ----
        species_rows = get_species_for_mechanism(self._conn, mech_id)
        for sp in species_rows:
            nid = f"S_{sp['id']}"
            G.add_node(nid,
                type             = "species",
                db_id            = sp["id"],
                name             = sp["name"],
                canonical_smiles = sp["canonical_smiles"],
                inchi_key        = sp["inchi_key"],
                formula          = sp["formula"],
                mw_kg_mol        = sp["mw_kg_mol"],
                glossary_matched = bool(sp["glossary_matched"]),
                mechanism_id     = mech_id,
                # M4 attributes (set by SparsityDetector)
                is_dead_end_source = None,
                is_dead_end_sink   = None,
                can_reach_products = None,
            )

        # ---- Build species name→node-id lookup ----
        sp_dbid_to_nid: Dict[int, str] = {
            sp["id"]: f"S_{sp['id']}" for sp in species_rows
        }

        # ---- Add reaction nodes + edges ----
        rxn_rows = get_reactions_for_mechanism(self._conn, mech_id)
        for rx in rxn_rows:
            rnid = f"R_{rx['id']}"
            G.add_node(rnid,
                type             = "reaction",
                db_id            = rx["id"],
                formula_string   = rx["formula_string"],
                reversible       = bool(rx["reversible"]),
                kinetics_type    = rx["kinetics_type"],
                family           = rx["family"],
                A_si             = rx["A_si"],
                n                = rx["n"],
                Ea_si            = rx["Ea_si"],
                pressure_dependent = bool(rx["pressure_dependent"]),
                has_third_body   = bool(rx["has_third_body"]),
                mechanism_id     = mech_id,
                canonical_key    = rx["canonical_key"],
                reaction_label   = rx["reaction_label"] if "reaction_label" in rx.keys() else None,
                # M8 attributes (set by SensitivityFluxIntegrator)
                sensitivity_score = None,
                flux_score        = None,
                importance_score  = None,
            )

            # Fetch stoichiometry
            stoich_rows = self._conn.execute(
                """SELECT s.*, sp.id AS sp_db_id
                   FROM stoichiometry s
                   JOIN species sp ON sp.id = s.species_id
                   WHERE s.reaction_id = ?""",
                (rx["id"],),
            ).fetchall()

            for st in stoich_rows:
                sp_nid = sp_dbid_to_nid.get(st["sp_db_id"])
                if sp_nid is None:
                    continue
                coeff = st["stoich_coeff"]
                role  = st["role"]

                if role == "reactant":
                    G.add_edge(sp_nid, rnid, stoich=coeff, role="reactant")
                    if include_reverse_edges and rx["reversible"]:
                        G.add_edge(rnid, sp_nid, stoich=coeff, role="reactant_reverse")
                elif role == "product":
                    G.add_edge(rnid, sp_nid, stoich=coeff, role="product")
                    if include_reverse_edges and rx["reversible"]:
                        G.add_edge(sp_nid, rnid, stoich=coeff, role="product_reverse")

        log.info(
            "Built graph for mech_id=%d: %d nodes, %d edges",
            mech_id, G.number_of_nodes(), G.number_of_edges(),
        )
        # Assign preset layout positions (spring, circle, bipartite, carbon-ring)
        # so single-panel mode also offers the preset layout dropdown.
        self._assign_union_positions([G])
        return G

    def build_union_graphs(
        self,
        mech_ids: List[int],
        include_reverse_edges: bool = True,
    ) -> List[nx.DiGraph]:
        """
        Build one graph per mechanism where every graph has the same node set
        (the union of all species and reactions across *mech_ids*).

        Nodes that belong to other mechanisms but not to this one are added as
        isolated "ghost" nodes (attribute ``ghost=True``).  They have no edges
        so only the mechanism's own chemistry is drawn, but the complete node
        inventory is visible in every panel for easy side-by-side comparison.

        Species are matched by ``inchi_key`` (if not NULL) then by ``name``.
        Reactions are matched by ``canonical_key`` (if not NULL) then by
        ``formula_string``.

        Parameters
        ----------
        mech_ids : list[int]
            Mechanism IDs to include.
        include_reverse_edges : bool
            Passed through to each :meth:`build_graph` call.

        Returns
        -------
        list[nx.DiGraph]
            One graph per entry in *mech_ids*, same order.
        """
        graphs = [self.build_graph(mid, include_reverse_edges) for mid in mech_ids]

        def _sp_key(data: dict) -> str:
            ik = data.get("inchi_key")
            return f"ik:{ik}" if ik else f"nm:{data.get('name', '')}"

        def _rx_key(data: dict) -> str:
            ck = data.get("canonical_key")
            return f"ck:{ck}" if ck else f"cf:{_canon_formula(data.get('formula_string', ''))}"

        # Build per-graph identity→node_id mappings
        sp_key_to_node: List[Dict[str, str]] = [{} for _ in graphs]
        rx_key_to_node: List[Dict[str, str]] = [{} for _ in graphs]
        for i, G in enumerate(graphs):
            for node_id, data in G.nodes(data=True):
                if data.get("type") == "species":
                    sp_key_to_node[i][_sp_key(data)] = node_id
                elif data.get("type") == "reaction":
                    rx_key_to_node[i][_rx_key(data)] = node_id

        # Union of all identity keys across all graphs
        all_sp_keys = set().union(*(m.keys() for m in sp_key_to_node))
        all_rx_keys = set().union(*(m.keys() for m in rx_key_to_node))

        # For each graph add ghost nodes for absent species/reactions
        for i, G in enumerate(graphs):
            for k in all_sp_keys:
                if k not in sp_key_to_node[i]:
                    src_data, src_mech = self._find_source(
                        k, sp_key_to_node, graphs, mech_ids
                    )
                    if src_data is None:
                        continue
                    # "S" prefix avoids collision with same db_id from reactions table
                    ghost_nid = f"ghost_S_{src_data.get('db_id','?')}_{src_mech}"
                    node_attrs = {
                        **src_data,
                        "ghost": True,
                        "ghost_from_mech_id": src_mech,
                        "mechanism_id": mech_ids[i],
                    }
                    G.add_node(ghost_nid, **node_attrs)

            for k in all_rx_keys:
                if k not in rx_key_to_node[i]:
                    src_data, src_mech = self._find_source(
                        k, rx_key_to_node, graphs, mech_ids
                    )
                    if src_data is None:
                        continue
                    # "R" prefix avoids collision with same db_id from species table
                    ghost_nid = f"ghost_R_{src_data.get('db_id','?')}_{src_mech}"
                    node_attrs = {
                        **src_data,
                        "ghost": True,
                        "ghost_from_mech_id": src_mech,
                        "mechanism_id": mech_ids[i],
                    }
                    G.add_node(ghost_nid, **node_attrs)

            n_ghost = len([n for n, d in G.nodes(data=True) if d.get("ghost")])
            log.info(
                "Union graph for mech_id=%d: %d nodes (%d own + %d ghost), %d edges",
                mech_ids[i],
                G.number_of_nodes(),
                G.number_of_nodes() - n_ghost,
                n_ghost,
                G.number_of_edges(),
            )

        # Compute one shared layout and distribute positions to all graphs
        self._assign_union_positions(graphs)
        # Assign global union R-IDs so the same reaction carries the same label
        # in every panel regardless of per-mechanism sequential numbering.
        self._assign_global_reaction_labels(graphs)
        return graphs

    def _assign_union_positions(self, graphs: List[nx.DiGraph]) -> None:
        """
        Build a compact true-union graph (one node per unique species / reaction,
        keyed by chemical identity), run ``spring_layout`` on it, and write
        ``pos_x`` / ``pos_y`` to *every* node in every per-mechanism graph.

        Identity key:  inchi_key → "ik:<key>", else name → "nm:<name>"  (species)
                       canonical_key → "ck:<key>", else formula → "fs:<str>" (reaction)

        Ghost nodes copy ``**src_data`` when created, so they carry the same
        ``inchi_key`` / ``canonical_key`` attributes as their source.  The
        identity-key lookup therefore returns the same (x, y) for both the own
        node and its ghost counterpart — guaranteeing identical structure in all
        comparison panels.
        """

        def _sp_key(data: dict) -> str:
            ik = data.get("inchi_key")
            return f"ik:{ik}" if ik else f"nm:{data.get('name', '')}"

        def _rx_key(data: dict) -> str:
            ck = data.get("canonical_key")
            return f"ck:{ck}" if ck else f"cf:{_canon_formula(data.get('formula_string', ''))}"

        def _node_key(data: dict) -> str:
            return _sp_key(data) if data.get("type") == "species" else _rx_key(data)

        # ── Build compact union graph (own nodes only define topology) ──────────
        union_G: nx.DiGraph = nx.DiGraph()
        for G in graphs:
            for _, data in G.nodes(data=True):
                if data.get("ghost"):
                    continue
                k = _node_key(data)
                if k not in union_G:
                    union_G.add_node(k, node_type=data.get("type", "unknown"))

        for G in graphs:
            own_key = {
                nid: _node_key(data)
                for nid, data in G.nodes(data=True)
                if not data.get("ghost")
            }
            for src, tgt in G.edges():
                sk = own_key.get(src)
                tk = own_key.get(tgt)
                if sk and tk and not union_G.has_edge(sk, tk):
                    union_G.add_edge(sk, tk)

        # ── Spring layout on the union graph ─────────────────────────────────────
        pos_spring = nx.spring_layout(
            union_G, seed=42, k=2.0, iterations=150, scale=900
        )

        # ── Circle layout: species first (alphabetical), then reactions ───────────
        sp_keys = sorted(
            [k for k in union_G.nodes() if union_G.nodes[k].get("node_type") == "species"]
        )
        rx_keys = sorted(
            [k for k in union_G.nodes() if union_G.nodes[k].get("node_type") != "species"]
        )
        sorted_circle = sp_keys + rx_keys
        n_total = max(len(sorted_circle), 1)
        R = 900.0
        pos_circle = {
            k: (R * math.cos(2 * math.pi * i / n_total),
                R * math.sin(2 * math.pi * i / n_total))
            for i, k in enumerate(sorted_circle)
        }

        # ── Bipartite layout: species top row, reactions bottom row ──────────────
        canvas_w = 1800.0
        sp_step  = canvas_w / max(len(sp_keys), 1)
        rx_step  = canvas_w / max(len(rx_keys), 1)
        pos_bipartite: Dict[str, tuple] = {}
        for i, k in enumerate(sp_keys):
            pos_bipartite[k] = ((i + 0.5) * sp_step - canvas_w / 2, 500.0)
        for i, k in enumerate(rx_keys):
            pos_bipartite[k] = ((i + 0.5) * rx_step - canvas_w / 2, -500.0)

        # ── Carbon-ring layout (Phase 5C) ────────────────────────────────────────
        pos_cring = self._carbon_ring_positions(union_G)

        # ── Ghost importance: degree in union graph (higher = more connected) ─────
        ghost_imp = dict(union_G.degree())

        # ── Distribute all positions + importance to every graph node ─────────────
        # Ghost nodes carry the same identity attributes as their source, so
        # _node_key(ghost_data) == _node_key(own_data) → same coordinates.
        for G in graphs:
            for _, data in G.nodes(data=True):
                k = _node_key(data)
                xs, ys   = pos_spring.get(k, (0.0, 0.0))
                xc, yc   = pos_circle.get(k, (0.0, 0.0))
                xb, yb   = pos_bipartite.get(k, (0.0, 0.0))
                xr, yr   = pos_cring.get(k, (0.0, 0.0))
                data["pos_x"]           = float(xs)
                data["pos_y"]           = float(ys)
                data["pos_x_circle"]    = float(xc)
                data["pos_y_circle"]    = float(yc)
                data["pos_x_bipartite"] = float(xb)
                data["pos_y_bipartite"] = float(yb)
                data["pos_x_cring"]     = float(xr)
                data["pos_y_cring"]     = float(yr)
                data["ghost_importance"] = int(ghost_imp.get(k, 0))

    @staticmethod
    def _carbon_number(formula: str) -> int:
        """Extract carbon count from an empirical formula string (e.g. 'C2H4' → 2)."""
        if not formula:
            return 0
        m = re.search(r'C(\d*)', formula)
        if not m:
            return 0
        digits = m.group(1)
        return int(digits) if digits else 1

    def _carbon_ring_positions(
        self,
        union_G: nx.DiGraph,
        delta_r: float = 300.0,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute concentric-ring (x, y) positions for all nodes in *union_G*.

        Species nodes are placed on rings keyed by their carbon number:
          - Shell 0 (centre): species with C₀ (H, O, N, inorganic)
          - Shell k:          species with Cₖ count (k ≥ 1)

        Within each shell, species are sorted alphabetically and spaced
        uniformly.  Reaction nodes are placed at the centroid of their
        adjacent species nodes' ring positions.

        Parameters
        ----------
        delta_r : float
            Radial separation between consecutive carbon shells [px].
        """
        pos: Dict[str, Tuple[float, float]] = {}

        # Separate species and reaction keys
        sp_keys = [k for k in union_G.nodes() if union_G.nodes[k].get("node_type") == "species"]
        rx_keys = [k for k in union_G.nodes() if union_G.nodes[k].get("node_type") != "species"]

        # Group species by carbon-number shell.
        # The node key is "ik:<InChIKey>" or "nm:<name>".  We can't extract
        # formula from the identity key alone, so we look it up from the first
        # graph that carries this node (own, not ghost).  As a fallback we use
        # the name portion of the identity key to guess the carbon count.
        shell_map: Dict[str, int] = {}  # identity key → carbon number

        for k in sp_keys:
            # Try to derive formula from node key
            name_part = k.split(":", 1)[-1] if ":" in k else k
            cn = self._carbon_number(name_part)
            shell_map[k] = cn

        # Group keys by shell
        shells: Dict[int, List[str]] = {}
        for k, cn in shell_map.items():
            shells.setdefault(cn, []).append(k)

        # Assign angular positions within each shell
        for cn, keys in shells.items():
            r = cn * delta_r if cn > 0 else 0.0
            sorted_keys = sorted(keys)
            n = len(sorted_keys)
            if n == 1:
                # Single node at top of ring
                pos[sorted_keys[0]] = (r, 0.0)
            else:
                for i, k in enumerate(sorted_keys):
                    theta = 2 * math.pi * i / n
                    pos[k] = (r * math.cos(theta), r * math.sin(theta))

        # Place reaction nodes at centroid of adjacent species
        for rx_k in rx_keys:
            neighbours = list(union_G.predecessors(rx_k)) + list(union_G.successors(rx_k))
            sp_positions = [pos[nb] for nb in neighbours if nb in pos]
            if sp_positions:
                cx = sum(p[0] for p in sp_positions) / len(sp_positions)
                cy = sum(p[1] for p in sp_positions) / len(sp_positions)
                pos[rx_k] = (cx, cy)
            else:
                pos[rx_k] = (0.0, 0.0)

        return pos

    def _assign_global_reaction_labels(self, graphs: List[nx.DiGraph]) -> None:
        """
        Overwrite ``reaction_label`` on every reaction node (own and ghost) in
        every graph with a globally consistent R-ID.

        Ordering: mechanism-1's own reactions first (in DB/insertion order),
        then reactions unique to mechanism-2 (in their order), etc.  This means
        a reaction shared across all mechanisms gets the same global R-label as
        it had in mechanism-1, avoiding arbitrary re-numbering for the primary
        mechanism.
        """
        def _rx_key(data: dict) -> str:
            ck = data.get("canonical_key")
            return f"ck:{ck}" if ck else f"cf:{_canon_formula(data.get('formula_string', ''))}"

        # First-appearance order: own reactions from each graph in insertion order
        seen: set = set()
        ordered: List[str] = []
        for G in graphs:
            for _, data in G.nodes(data=True):
                if data.get("type") == "reaction" and not data.get("ghost"):
                    k = _rx_key(data)
                    if k not in seen:
                        seen.add(k)
                        ordered.append(k)

        global_id = {k: f"R{i + 1}" for i, k in enumerate(ordered)}

        # Write global R-ID to every reaction node (own and ghost)
        for G in graphs:
            for _, data in G.nodes(data=True):
                if data.get("type") == "reaction":
                    k = _rx_key(data)
                    gid = global_id.get(k)
                    if gid:
                        data["reaction_label"] = gid

    @staticmethod
    def _find_source(
        key: str,
        key_to_node_maps: List[Dict[str, str]],
        graphs: List[nx.DiGraph],
        mech_ids: List[int],
    ) -> Tuple[Optional[dict], Optional[int]]:
        """Return (node_data_dict, mech_id) for the first graph that has *key*."""
        for j, mapping in enumerate(key_to_node_maps):
            if key in mapping:
                nid = mapping[key]
                return dict(graphs[j].nodes[nid]), mech_ids[j]
        return None, None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, G: nx.DiGraph) -> Dict[str, Any]:
        species_nodes  = [n for n, d in G.nodes(data=True) if d["type"] == "species"]
        reaction_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "reaction"]

        # Species degree (in + out)
        sp_degrees = sorted(
            [(G.nodes[n]["name"], G.degree(n)) for n in species_nodes],
            key=lambda x: x[1], reverse=True,
        )

        # Isolated species
        isolated = [n for n in species_nodes if G.degree(n) == 0]

        # Connected components
        wcc = list(nx.weakly_connected_components(G))
        scc = list(nx.strongly_connected_components(G))

        # Reaction family distribution
        family_dist: Dict[str, int] = {}
        for n in reaction_nodes:
            fam = G.nodes[n].get("family") or "Unknown"
            family_dist[fam] = family_dist.get(fam, 0) + 1

        avg_sp_deg = (sum(d for _, d in sp_degrees) / len(sp_degrees)
                      if sp_degrees else 0.0)

        return {
            "n_nodes":           G.number_of_nodes(),
            "n_edges":           G.number_of_edges(),
            "n_species":         len(species_nodes),
            "n_reactions":       len(reaction_nodes),
            "n_isolated_species": len(isolated),
            "n_components_weak":  len(wcc),
            "n_components_strong": len(scc),
            "largest_scc_size":   max((len(c) for c in scc), default=0),
            "is_dag":            nx.is_directed_acyclic_graph(G),
            "avg_species_degree": round(avg_sp_deg, 2),
            "max_species_degree": sp_degrees[0][1] if sp_degrees else 0,
            "most_connected_species": [(n, d) for n, d in sp_degrees[:10]],
            "family_distribution":    family_dist,
            "mechanism_id":      G.graph.get("mechanism_id"),
        }

    # ------------------------------------------------------------------
    # Subgraph / pathway queries
    # ------------------------------------------------------------------

    def get_species_subgraph(
        self, G: nx.DiGraph, species_names: List[str]
    ) -> nx.DiGraph:
        """Return induced subgraph containing only the named species (and their reactions)."""
        name_to_nid = {d["name"]: n for n, d in G.nodes(data=True)
                       if d["type"] == "species"}
        target_sp = {name_to_nid[nm] for nm in species_names if nm in name_to_nid}
        # Include all reaction nodes adjacent to the target species
        nbrs: set = set()
        for sp_nid in target_sp:
            nbrs.update(G.predecessors(sp_nid))
            nbrs.update(G.successors(sp_nid))
        return G.subgraph(target_sp | nbrs).copy()

    def find_paths_between_species(
        self,
        G:       nx.DiGraph,
        source:  str,
        target:  str,
        cutoff:  int = 20,
    ) -> List[List[str]]:
        """
        Find all simple paths from species *source* to species *target*.

        Returns a list of node-id paths.  Each path alternates
        species → reaction → species → ···.
        """
        name_to_nid = {d["name"]: n for n, d in G.nodes(data=True)
                       if d["type"] == "species"}
        src_nid = name_to_nid.get(source)
        tgt_nid = name_to_nid.get(target)
        if src_nid is None or tgt_nid is None:
            log.warning("Species '%s' or '%s' not in graph", source, target)
            return []
        try:
            return list(nx.all_simple_paths(G, src_nid, tgt_nid, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graphml(self, G: nx.DiGraph, path: str) -> None:
        """Export the graph to GraphML format."""
        # GraphML requires primitive attribute types; convert None → ""
        H = G.copy()
        for _, data in H.nodes(data=True):
            for k, v in data.items():
                if v is None:
                    data[k] = ""
        for _, _, data in H.edges(data=True):
            for k, v in data.items():
                if v is None:
                    data[k] = ""
        nx.write_graphml(H, path)
        log.info("Exported GraphML: %s", path)

    def export_json(self, G: nx.DiGraph, path: str) -> None:
        """Export the graph to Cytoscape-compatible JSON format."""
        data = nx.cytoscape_data(G)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        log.info("Exported JSON: %s", path)
