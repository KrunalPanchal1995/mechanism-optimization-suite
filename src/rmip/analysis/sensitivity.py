"""
rmip/analysis/sensitivity.py
============================
Module 8: Sensitivity + Flux Integration

What it does
------------
Reads the `sensitivity_data` and `flux_data` dicts (format defined in
CLAUDE.md §10, written by KP's Chemkin-Pro wrapper) and:

  1. **Attaches sensitivity scores** onto reaction nodes in the bipartite graph.
     Key: `sensitivity_score` = |∂ln(Q)/∂ln(k_i)| normalised to [0, 1] within
     the run set.

  2. **Attaches flux scores** onto reaction nodes.
     Key: `flux_score` = |integrated_flux| / max(|integrated_flux|).

  3. Computes a combined **importance_score**:
       importance = sqrt(sensitivity² + flux²)   [normalised, ∈ [0,1]]
     If only one dataset is provided, that dataset alone drives importance.

  4. Writes all scores to the graph node attributes so M3 (Visualization)
     and M7 (FusionEngine) can use them.

  5. Provides summary DataFrame and CSV export.

Input format (from CLAUDE.md §10)
----------------------------------
sensitivity_data = {
    "observable"  : "IDT",
    "conditions"  : {"T_K": 1100, "P_atm": 10, "phi": 1.0},
    "coefficients": {
        "CH4 + OH <=> CH3 + H2O" : -0.43,
        "H + O2 <=> OH + O"      : +0.85,
    }
}

flux_data = {
    "observable"  : "IDT",
    "conditions"  : {"T_K": 1100, "P_atm": 10, "phi": 1.0, "time_s": 1.2e-3},
    "fluxes"      : {"CH4 + OH <=> CH3 + H2O": 2.34e-4, ...}
}

Matching
--------
The formula_string key in sensitivity_data must match EXACTLY the
`formula_string` stored in the `reactions` DB table for that mechanism.
KP's wrapper should do the index→formula mapping described in CLAUDE.md §10.
If a key does not match any DB reaction, a warning is logged and it is skipped.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class SensitivityFluxIntegrator:
    """
    M8: Sensitivity + Flux Integration.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def attach(
        self,
        mechanism_id:      int,
        G:                 nx.DiGraph,
        sensitivity_data:  Optional[Dict[str, Any]] = None,
        flux_data:         Optional[Dict[str, Any]] = None,
    ) -> nx.DiGraph:
        """
        Attach sensitivity, flux, and importance scores to reaction nodes.

        Parameters
        ----------
        mechanism_id : int
            DB id of the mechanism.  Used to build the formula→node map.
        G : nx.DiGraph
            Bipartite DiGraph from GraphBuilder (mutated in-place).
        sensitivity_data : dict, optional
            Sensitivity dict from KP's Chemkin-Pro wrapper.
        flux_data : dict, optional
            Flux dict from KP's Chemkin-Pro wrapper.

        Returns
        -------
        The same graph G with updated node attributes.
        """
        if sensitivity_data is None and flux_data is None:
            logger.warning("M8: No sensitivity or flux data provided — nothing attached.")
            return G

        # Build formula → reaction node ID map from the graph
        formula_to_node = self._build_formula_map(G)

        # ── Sensitivity ──
        sa_scores: Dict[str, float] = {}
        if sensitivity_data:
            coefficients = sensitivity_data.get("coefficients", {})
            sa_scores    = {k: abs(v) for k, v in coefficients.items()}

        # ── Flux ──
        flux_scores: Dict[str, float] = {}
        if flux_data:
            fluxes      = flux_data.get("fluxes", {})
            flux_scores = {k: abs(v) for k, v in fluxes.items()}

        # Normalise independently
        sa_norm   = self._normalise(sa_scores)
        flux_norm = self._normalise(flux_scores)

        # Attach to nodes
        n_attached_sa   = 0
        n_attached_flux = 0
        n_missing        = 0

        for formula, node_id in formula_to_node.items():
            node = G.nodes[node_id]

            sa_val   = sa_norm.get(formula)
            fl_val   = flux_norm.get(formula)

            if sa_val is not None:
                node["sensitivity_score"] = sa_val
                n_attached_sa += 1
            else:
                node.setdefault("sensitivity_score", None)

            if fl_val is not None:
                node["flux_score"] = fl_val
                n_attached_flux += 1
            else:
                node.setdefault("flux_score", None)

            # Importance score
            s  = node.get("sensitivity_score") or 0.0
            f  = node.get("flux_score")         or 0.0
            if s > 0 or f > 0:
                node["importance_score"] = math.sqrt(s**2 + f**2) / math.sqrt(2)
            else:
                node["importance_score"] = None

        # Warn about unmatched sensitivity entries
        for formula in sa_scores:
            if formula not in formula_to_node:
                logger.debug("M8: sensitivity key not found in graph: %r", formula)
                n_missing += 1

        logger.info(
            "M8 attached: %d sensitivity, %d flux scores "
            "(%d formula keys not matched to graph nodes)",
            n_attached_sa, n_attached_flux, n_missing,
        )
        return G

    def attach_from_files(
        self,
        mechanism_id:     int,
        G:                nx.DiGraph,
        sensitivity_json: Optional[str] = None,
        flux_json:        Optional[str] = None,
    ) -> nx.DiGraph:
        """
        Convenience wrapper: load sensitivity/flux data from JSON files and attach.

        JSON files must conform to the dict format in CLAUDE.md §10.
        """
        sa_data = self._load_json(sensitivity_json) if sensitivity_json else None
        fl_data = self._load_json(flux_json)         if flux_json        else None
        return self.attach(mechanism_id, G, sa_data, fl_data)

    # ------------------------------------------------------------------
    # Query Helpers
    # ------------------------------------------------------------------

    def get_top_reactions(
        self,
        G:       nx.DiGraph,
        n:       int  = 20,
        metric:  str  = "importance_score",
        only_scored: bool = True,
    ) -> "pd.DataFrame":
        """
        Return the top-N reaction nodes sorted by `metric`.

        Parameters
        ----------
        G : nx.DiGraph
        n : int
            Number of top reactions to return.
        metric : str
            Node attribute to sort by.  One of:
            ``'importance_score'`` | ``'sensitivity_score'`` | ``'flux_score'``.
        only_scored : bool
            If True, exclude reactions with None / 0 for the metric.

        Returns
        -------
        pd.DataFrame  (columns: node_id, formula_string, {metric}, ...)
        """
        import pandas as pd

        rows = []
        for node, data in G.nodes(data=True):
            if data.get("type") != "reaction":
                continue
            val = data.get(metric)
            if only_scored and (val is None or val == 0):
                continue
            rows.append({
                "node_id":            node,
                "formula_string":     data.get("formula_string", ""),
                "kinetics_type":      data.get("kinetics_type", ""),
                "sensitivity_score":  data.get("sensitivity_score"),
                "flux_score":         data.get("flux_score"),
                "importance_score":   data.get("importance_score"),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (
            df.sort_values(metric, ascending=False)
              .head(n)
              .reset_index(drop=True)
        )

    def get_score_summary(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Return a summary dict of score statistics."""
        sa_vals   = [d.get("sensitivity_score") for _, d in G.nodes(data=True)
                     if d.get("type") == "reaction" and d.get("sensitivity_score") is not None]
        fl_vals   = [d.get("flux_score") for _, d in G.nodes(data=True)
                     if d.get("type") == "reaction" and d.get("flux_score") is not None]
        imp_vals  = [d.get("importance_score") for _, d in G.nodes(data=True)
                     if d.get("type") == "reaction" and d.get("importance_score") is not None]

        def _stats(vals):
            if not vals:
                return {"n": 0}
            import statistics
            return {
                "n": len(vals),
                "min": min(vals), "max": max(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
            }

        return {
            "sensitivity_score": _stats(sa_vals),
            "flux_score":        _stats(fl_vals),
            "importance_score":  _stats(imp_vals),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_report(
        self,
        G:           nx.DiGraph,
        output_path: str,
        top_n:       int = 50,
    ) -> None:
        """
        Export a CSV report of all scored reactions, sorted by importance_score.
        """
        import pandas as pd

        df = self.get_top_reactions(G, n=top_n, only_scored=False)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("M8 sensitivity report saved: %s (%d rows)", path, len(df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_formula_map(G: nx.DiGraph) -> Dict[str, str]:
        """Return {formula_string: node_id} for all reaction nodes."""
        fmap: Dict[str, str] = {}
        for node, data in G.nodes(data=True):
            if data.get("type") == "reaction":
                fs = data.get("formula_string")
                if fs:
                    fmap[fs] = node
        return fmap

    @staticmethod
    def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalise scores to [0, 1] by max absolute value."""
        if not scores:
            return {}
        max_val = max(scores.values())
        if max_val == 0:
            return {k: 0.0 for k in scores}
        return {k: v / max_val for k, v in scores.items()}

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path) as fh:
            return json.load(fh)
