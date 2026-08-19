"""
graph_sa.sa_bridge
==================

Glue between the MUQ-SAC / MSO sensitivity-analysis pickle and the RMIP
reaction-network graph.

The MSO sensitivity tool writes ``sens_parameters.pkl`` with the structure::

    { "0": { "<reaction formula>": <sensitivity coeff>, ... },   # target 0
      "1": { ... },                                              # target 1
      ...                                                        # one per target
    }

i.e. for every optimization target (T/P/phi condition) a dict of
reaction-formula → sensitivity coefficient for the reactions that were flagged
sensitive.  This module:

1. Loads the pickle and collapses the per-target coefficients into a single
   score per reaction (default: ``max`` of ``|coeff|`` across targets — a
   reaction is "sensitive" if it matters strongly for *any* condition).
2. Matches those reaction formulas to the RMIP database's ``formula_string``
   values using the mechanism-agnostic :func:`rmip.core.graph_builder._canon_formula`
   normaliser (arrow / whitespace / token-order insensitive).
3. Aggregates reaction scores onto their participating species to obtain a
   per-species sensitivity score.
4. Produces a RMIP ``sensitivity_data`` dict (CLAUDE.md §10) so the existing
   :class:`rmip.analysis.sensitivity.SensitivityFluxIntegrator` can decorate the
   full graph, and builds a compact "sensitive subgraph" for a readable figure.

No new sensitivity mathematics is introduced here — it is purely a data bridge.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from rmip.core.graph_builder import _canon_formula


# ---------------------------------------------------------------------------
# target.opt parsing / path resolution
# ---------------------------------------------------------------------------

def resolve_path(path: Optional[str], anchor_dir: str) -> Optional[str]:
    """
    Resolve a path that may have been written on a different machine.

    ``target.opt`` stores absolute paths (e.g. under
    ``/home/krithika/mechanism-optimization-suite/...``) that need not exist on
    the current host.  If *path* exists it is returned as-is; otherwise we look
    for a file with the same basename next to *anchor_dir* (the directory that
    contains ``target.opt``).  Returns ``None`` if nothing is found.
    """
    if not path:
        return None
    if os.path.exists(path):
        return path
    candidate = os.path.join(anchor_dir, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    return path  # return the original so callers can report a helpful error


def load_target_opt(path: str) -> Dict[str, object]:
    """
    Parse a ``target.opt`` YAML config and pull out the fields the graph-SA
    workflow needs.  Paths are resolved relative to the config's directory so
    the workflow runs regardless of the host the config was authored on.
    """
    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    anchor = os.path.dirname(os.path.abspath(path))
    locations = cfg.get("Locations", {}) or {}
    inputs = cfg.get("Inputs", {}) or {}

    mech = resolve_path(locations.get("mechanism"), anchor)
    thermo = resolve_path(locations.get("thermo_file"), anchor)

    # The SA pickle is written next to the run config by run_script.py.
    sa_pickle = os.path.join(anchor, "sens_parameters.pkl")

    return {
        "config_path":   os.path.abspath(path),
        "anchor_dir":    anchor,
        "mechanism":     mech,
        "thermo_file":   thermo,
        "sa_pickle":     sa_pickle,
        "analysis_type": inputs.get("AnalysisType"),
        "fuel":          inputs.get("fuel"),
        "raw":           cfg,
    }


# ---------------------------------------------------------------------------
# Sensitivity pickle → per-reaction scores
# ---------------------------------------------------------------------------

def load_sa_pickle(path: str) -> Dict[str, Dict[str, float]]:
    """Load ``sens_parameters.pkl`` → ``{target_idx: {formula: coeff}}``."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def aggregate_reaction_scores(
    sa: Dict[str, Dict[str, float]],
    method: str = "max",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Collapse per-target sensitivity coefficients into one score per reaction.

    Parameters
    ----------
    sa : dict
        ``{target_idx: {formula: coeff}}`` as loaded from the pickle.
    method : {"max", "mean"}
        How to combine ``|coeff|`` across targets.  Default ``"max"``.

    Returns
    -------
    (norm, raw, per_target)
        ``norm``       : {formula: score in [0, 1]} normalised by the global max.
        ``raw``        : {formula: aggregated |coeff|} before normalisation.
        ``per_target`` : {formula: {target_idx: |coeff|}} for CSV export.
    """
    per_target: Dict[str, Dict[str, float]] = defaultdict(dict)
    for tgt, coeffs in sa.items():
        for formula, coeff in coeffs.items():
            per_target[formula][str(tgt)] = abs(float(coeff))

    raw: Dict[str, float] = {}
    for formula, tvals in per_target.items():
        vals = list(tvals.values())
        if not vals:
            continue
        raw[formula] = max(vals) if method == "max" else sum(vals) / len(vals)

    mx = max(raw.values()) if raw else 1.0
    norm = {f: (v / mx if mx > 0 else 0.0) for f, v in raw.items()}
    return norm, raw, dict(per_target)


# ---------------------------------------------------------------------------
# Matching SA formulas to DB reactions
# ---------------------------------------------------------------------------

def match_scores_to_db(
    conn: sqlite3.Connection,
    mech_id: int,
    scores: Dict[str, float],
) -> Tuple[Dict[str, dict], List[str]]:
    """
    Match SA reaction formulas to database reactions of *mech_id*.

    Matching is done on the canonicalised formula (arrow / whitespace /
    token-order insensitive) so small formatting differences between the SA
    tool and the mechanism file do not prevent a match.

    Returns
    -------
    (matched, unmatched)
        ``matched``   : {db_formula_string: {"score", "db_id", "sa_formula"}}
        ``unmatched`` : list of SA formulas with no DB counterpart.
    """
    rows = conn.execute(
        "SELECT id, formula_string FROM reactions WHERE mechanism_id=?",
        (mech_id,),
    ).fetchall()

    canon_to_db: Dict[str, Tuple[int, str]] = {}
    for r in rows:
        key = _canon_formula(r["formula_string"])
        # First occurrence wins (mechanisms rarely duplicate a canonical form).
        canon_to_db.setdefault(key, (r["id"], r["formula_string"]))

    matched: Dict[str, dict] = {}
    unmatched: List[str] = []
    for sa_formula, score in scores.items():
        key = _canon_formula(sa_formula)
        if key in canon_to_db:
            db_id, db_fs = canon_to_db[key]
            # If several SA formulas map to the same DB reaction, keep the max.
            if db_fs in matched:
                if score > matched[db_fs]["score"]:
                    matched[db_fs].update(score=score, sa_formula=sa_formula)
            else:
                matched[db_fs] = {"score": score, "db_id": db_id,
                                  "sa_formula": sa_formula}
        else:
            unmatched.append(sa_formula)

    return matched, unmatched


# ---------------------------------------------------------------------------
# Species sensitivity aggregation
# ---------------------------------------------------------------------------

def species_sensitivity(
    conn: sqlite3.Connection,
    mech_id: int,
    matched: Dict[str, dict],
) -> List[dict]:
    """
    Aggregate matched reaction scores onto their participating species.

    A species' score is the maximum score over all sensitive reactions it takes
    part in (as reactant or product).  Returns a list of dicts sorted by score
    (descending)::

        {"name", "formula", "carbon_number", "score", "n_reactions"}
    """
    from rmip.core.graph_builder import GraphBuilder  # local: only for _carbon_number

    sp_score: Dict[str, float] = defaultdict(float)
    sp_formula: Dict[str, str] = {}
    sp_count: Dict[str, int] = defaultdict(int)

    for info in matched.values():
        rid = info["db_id"]
        sc = info["score"]
        rows = conn.execute(
            """SELECT sp.name AS name, sp.formula AS formula
                 FROM stoichiometry st
                 JOIN species sp ON sp.id = st.species_id
                WHERE st.reaction_id = ?""",
            (rid,),
        ).fetchall()
        seen = set()
        for r in rows:
            nm = r["name"]
            if nm in seen:
                continue
            seen.add(nm)
            sp_score[nm] = max(sp_score[nm], sc)
            sp_count[nm] += 1
            sp_formula.setdefault(nm, r["formula"] or "")

    out = []
    for nm, sc in sp_score.items():
        formula = sp_formula.get(nm, "")
        out.append({
            "name":          nm,
            "formula":       formula,
            "carbon_number": GraphBuilder._carbon_number(formula),
            "score":         sc,
            "n_reactions":   sp_count[nm],
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Hand-off to RMIP
# ---------------------------------------------------------------------------

def to_sensitivity_data(matched: Dict[str, dict]) -> dict:
    """
    Build a RMIP ``sensitivity_data`` dict (CLAUDE.md §10) keyed by the exact DB
    ``formula_string`` so :meth:`SensitivityFluxIntegrator.attach` matches every
    entry directly.
    """
    return {
        "observable":  "SA",
        "conditions":  {"source": "MUQ-SAC sens_parameters.pkl (max over targets)"},
        "coefficients": {db_fs: info["score"] for db_fs, info in matched.items()},
    }


def build_sensitive_subgraph(
    G: nx.DiGraph,
    matched: Dict[str, dict],
) -> nx.DiGraph:
    """
    Return the induced subgraph of the sensitive reaction nodes plus their
    directly participating species — a compact, readable slice of the full
    mechanism graph.
    """
    rxn_nodes = [f"R_{info['db_id']}" for info in matched.values()
                 if f"R_{info['db_id']}" in G]
    keep = set(rxn_nodes)
    for rn in rxn_nodes:
        keep.update(G.predecessors(rn))
        keep.update(G.successors(rn))
    return G.subgraph(keep).copy()
