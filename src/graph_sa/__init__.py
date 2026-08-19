"""
graph_sa — Sensitivity graph-analysis bridge.

Connects the MUQ-SAC / MSO sensitivity output (``sens_parameters.pkl``) to the
RMIP reaction-network graph so that *sensitive reactions* and *sensitive
species* can be visualised and ranked on the bipartite mechanism graph.

Typical use is through the repo-root driver ``graph_sa_run.py``; the building
blocks live in :mod:`graph_sa.sa_bridge` and the static renderers in
:mod:`graph_sa.plots`.
"""

from .sa_bridge import (
    load_target_opt,
    resolve_path,
    load_sa_pickle,
    aggregate_reaction_scores,
    match_scores_to_db,
    species_sensitivity,
    to_sensitivity_data,
    build_sensitive_subgraph,
)

__all__ = [
    "load_target_opt",
    "resolve_path",
    "load_sa_pickle",
    "aggregate_reaction_scores",
    "match_scores_to_db",
    "species_sensitivity",
    "to_sensitivity_data",
    "build_sensitive_subgraph",
]
