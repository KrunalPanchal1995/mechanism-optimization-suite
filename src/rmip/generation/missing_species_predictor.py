"""
rmip/generation/missing_species_predictor.py
=============================================
Phase-4 Module: Missing Species Predictor

Deterministic identification of candidate missing species and reactions,
feeding an RMG seed spec for pathway generation via the existing RMGInterface.

Two evidence sources
--------------------
1. Cross-mechanism gap (high confidence)
   Species and reactions present in ANY contributing mechanism but absent from
   the unified mechanism.  These are known chemistry → must-include list for RMG.

2. Topological gap (heuristic)
   Run SparsityDetector on the unified mechanism.  Dead-end species (never
   produced or never consumed) and orphan reaction reactants signal missing
   producing/consuming pathways → seed/expansion list for RMG reaction families.

Usage
-----
>>> from rmip.generation.missing_species_predictor import MissingSpeciesPredictor
>>> predictor = MissingSpeciesPredictor("project.db")
>>> report = predictor.predict(unified_mech_id=4, contributing_mech_ids=[1, 2, 3])
>>> seed = predictor.to_rmg_seed(report, "rmg_seed/seed.json")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rmip.types import MissingSpeciesReport, RMGSeedSpec

log = logging.getLogger("rmip.generation.missing_species_predictor")


class MissingSpeciesPredictor:
    """
    Identify candidate missing species and reactions for a unified mechanism.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def predict(
        self,
        unified_mech_id: int,
        contributing_mech_ids: List[int],
    ) -> MissingSpeciesReport:
        """
        Identify species and reactions missing from the unified mechanism.

        Parameters
        ----------
        unified_mech_id : int
            DB id of the unified mechanism.
        contributing_mech_ids : List[int]
            DB ids of source mechanisms used to build the unified mechanism.

        Returns
        -------
        MissingSpeciesReport
        """
        log.info(
            "MissingSpeciesPredictor.predict(): unified=%d, contributing=%s",
            unified_mech_id, contributing_mech_ids,
        )

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # ── 1. Cross-mechanism gap ────────────────────────────────────
            gap_species, gap_reactions = self._find_cross_mech_gaps(
                conn, unified_mech_id, contributing_mech_ids
            )

            # ── 2. Topological gap ────────────────────────────────────────
            dead_end_species, orphan_species = self._find_topological_gaps(
                conn, unified_mech_id
            )

        # ── 3. Build RMG seed lists ───────────────────────────────────────
        evidence: Dict[str, str] = {}

        must_include_smiles: List[str] = []
        for sp_info in gap_species:
            smiles = sp_info.get("canonical_smiles") or sp_info.get("name", "")
            if smiles and smiles not in must_include_smiles:
                must_include_smiles.append(smiles)
            ik = sp_info.get("inchi_key", sp_info.get("name", ""))
            evidence[ik] = (
                f"cross-mech gap: present in mech {sp_info.get('source_mech_id')} "
                f"but absent from unified mech {unified_mech_id}"
            )

        seed_smiles: List[str] = []
        for ik in dead_end_species:
            evidence[ik] = (
                f"topological gap (dead-end): "
                f"never produced or never consumed in unified mech {unified_mech_id}"
            )
            # We store SMILES in seed list; SMILES are fetched at to_rmg_seed time

        for ik in orphan_species:
            if ik not in evidence:
                evidence[ik] = (
                    f"topological gap (orphan reactant): "
                    f"reactant in an orphan reaction in unified mech {unified_mech_id}"
                )

        # Collect SMILES for dead-end + orphan species from DB
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            all_topo_ik = list(set(dead_end_species) | set(orphan_species))
            if all_topo_ik:
                placeholders = ",".join("?" * len(all_topo_ik))
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT inchi_key, canonical_smiles, name
                    FROM species
                    WHERE mechanism_id = ? AND inchi_key IN ({placeholders})
                    """,
                    [unified_mech_id] + all_topo_ik,
                ).fetchall()
                for row in rows:
                    smiles = row["canonical_smiles"] or row["name"]
                    if smiles and smiles not in seed_smiles:
                        seed_smiles.append(smiles)

        for ck in gap_reactions:
            evidence[ck] = (
                f"cross-mech gap: reaction present in contributing mechs "
                f"but absent from unified mech {unified_mech_id}"
            )

        report = MissingSpeciesReport(
            unified_mech_id       = unified_mech_id,
            contributing_mech_ids = contributing_mech_ids,
            gap_species           = gap_species,
            gap_reactions         = gap_reactions,
            dead_end_species      = dead_end_species,
            orphan_species        = orphan_species,
            seed_smiles           = seed_smiles,
            must_include_smiles   = must_include_smiles,
            evidence              = evidence,
        )

        log.info(
            "Missing species report: %d gap species, %d gap reactions, "
            "%d dead-end species, %d orphan species",
            len(gap_species), len(gap_reactions),
            len(dead_end_species), len(orphan_species),
        )
        return report

    def to_rmg_seed(
        self,
        report: MissingSpeciesReport,
        out_path: str,
    ) -> RMGSeedSpec:
        """
        Convert a MissingSpeciesReport to an RMGSeedSpec and write it to JSON.

        Parameters
        ----------
        report : MissingSpeciesReport
        out_path : str
            Path to write the seed JSON file.

        Returns
        -------
        RMGSeedSpec
        """
        spec = RMGSeedSpec(
            seed_smiles           = report.seed_smiles,
            must_include_smiles   = report.must_include_smiles,
            source_unified_mech_id = report.unified_mech_id,
            evidence              = report.evidence,
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        spec.save_json(out_path)
        log.info("RMG seed spec → %s  (%d seed, %d must-include)",
                 out_path, len(spec.seed_smiles), len(spec.must_include_smiles))
        return spec

    # -------------------------------------------------------------------------
    # Internal — cross-mechanism gap detection
    # -------------------------------------------------------------------------

    def _find_cross_mech_gaps(
        self,
        conn: sqlite3.Connection,
        unified_mech_id: int,
        contributing_mech_ids: List[int],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Find species/reactions in contributing mechanisms but absent from unified.

        Returns
        -------
        gap_species : list of dicts with keys: inchi_key, name, canonical_smiles, source_mech_id
        gap_reactions : list of canonical_keys
        """
        if not contributing_mech_ids:
            return [], []

        # Species in unified
        unified_sp = {
            row["inchi_key"]
            for row in conn.execute(
                "SELECT inchi_key FROM species WHERE mechanism_id=? AND inchi_key IS NOT NULL",
                (unified_mech_id,),
            ).fetchall()
        }
        # Fall back to name if no inchi_key
        unified_sp_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM species WHERE mechanism_id=?",
                (unified_mech_id,),
            ).fetchall()
        }

        # Reactions in unified
        unified_rxn = {
            row["canonical_key"]
            for row in conn.execute(
                "SELECT canonical_key FROM reactions WHERE mechanism_id=? AND canonical_key IS NOT NULL",
                (unified_mech_id,),
            ).fetchall()
        }

        placeholders = ",".join("?" * len(contributing_mech_ids))

        # All species in contributing mechs
        contrib_sp_rows = conn.execute(
            f"""
            SELECT inchi_key, name, canonical_smiles, mechanism_id
            FROM species
            WHERE mechanism_id IN ({placeholders})
            """,
            contributing_mech_ids,
        ).fetchall()

        gap_species: List[Dict[str, Any]] = []
        seen_ik: Set[str] = set()
        for row in contrib_sp_rows:
            ik   = row["inchi_key"]
            name = row["name"]
            # A species is missing if its inchi_key is absent from unified
            # (or its name, if no inchi_key)
            if ik:
                if ik not in unified_sp and ik not in seen_ik:
                    seen_ik.add(ik)
                    gap_species.append({
                        "inchi_key":        ik,
                        "name":             name,
                        "canonical_smiles": row["canonical_smiles"],
                        "source_mech_id":   row["mechanism_id"],
                    })
            else:
                if name not in unified_sp_names:
                    gap_species.append({
                        "inchi_key":        None,
                        "name":             name,
                        "canonical_smiles": row["canonical_smiles"],
                        "source_mech_id":   row["mechanism_id"],
                    })

        # All reactions in contributing mechs
        contrib_rxn_rows = conn.execute(
            f"""
            SELECT DISTINCT canonical_key
            FROM reactions
            WHERE mechanism_id IN ({placeholders})
              AND canonical_key IS NOT NULL
            """,
            contributing_mech_ids,
        ).fetchall()

        gap_reactions: List[str] = [
            row["canonical_key"]
            for row in contrib_rxn_rows
            if row["canonical_key"] not in unified_rxn
        ]

        return gap_species, gap_reactions

    # -------------------------------------------------------------------------
    # Internal — topological gap detection via SparsityDetector
    # -------------------------------------------------------------------------

    def _find_topological_gaps(
        self,
        conn: sqlite3.Connection,
        unified_mech_id: int,
    ) -> tuple[List[str], List[str]]:
        """
        Run SparsityDetector on unified mechanism; return dead-end and orphan inchi_keys.
        """
        try:
            from rmip.core.graph_builder import GraphBuilder
            from rmip.analysis.sparsity import SparsityDetector
        except ImportError as exc:
            log.warning("SparsityDetector unavailable: %s — skipping topological gaps", exc)
            return [], []

        try:
            gb = GraphBuilder(self._db_path)
            G  = gb.build_graph(unified_mech_id)
            detector = SparsityDetector(self._db_path)
            report   = detector.analyze(mechanism_id=unified_mech_id, G=G)
        except Exception as exc:
            log.warning("SparsityDetector failed: %s", exc)
            return [], []

        # Node IDs are "S_{db_id}"; resolve to inchi_keys
        def node_ids_to_ik(node_ids: List[str]) -> List[str]:
            result: List[str] = []
            for nid in node_ids:
                if nid.startswith("S_"):
                    try:
                        sp_id = int(nid[2:])
                        row = conn.execute(
                            "SELECT inchi_key FROM species WHERE id=?", (sp_id,)
                        ).fetchone()
                        if row and row["inchi_key"]:
                            result.append(row["inchi_key"])
                    except (ValueError, TypeError):
                        pass
            return result

        dead_end_ik = node_ids_to_ik(
            report.dead_end_source_nodes + report.dead_end_sink_nodes
        )

        # Orphan reactants: species that are reactants in orphan reactions
        orphan_ik: List[str] = []
        for rxn_node in report.orphan_reaction_nodes:
            if rxn_node.startswith("R_"):
                try:
                    rxn_id = int(rxn_node[2:])
                    sp_rows = conn.execute(
                        """
                        SELECT sp.inchi_key FROM stoichiometry s
                        JOIN species sp ON sp.id = s.species_id
                        WHERE s.reaction_id = ? AND s.role = 'reactant'
                          AND sp.inchi_key IS NOT NULL
                        """,
                        (rxn_id,),
                    ).fetchall()
                    for row in sp_rows:
                        if row["inchi_key"] not in orphan_ik:
                            orphan_ik.append(row["inchi_key"])
                except (ValueError, TypeError):
                    pass

        return list(set(dead_end_ik)), orphan_ik
