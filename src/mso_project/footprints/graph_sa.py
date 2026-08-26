"""Footprint for graph_sa_run.py. Default --out is outputs/graph_sa
(relative default -> CWD-relative unless the caller passes an absolute
--out); --db defaults to <out>/graph_sa.db unless overridden."""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    FootprintEntry("**/sensitive_reactions.csv", "graph_sa_table"),
    FootprintEntry("**/sensitive_species.csv", "graph_sa_table"),
    FootprintEntry("**/sensitive_subgraph.png", "graph_sa_figure"),
    FootprintEntry("**/full_graph_sensitive.png", "graph_sa_figure"),
    FootprintEntry("**/graph_sa.db", "graph_sa_database"),
]
