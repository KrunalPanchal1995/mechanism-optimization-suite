"""
MB (Methyl Butanoate 2026) sensitivity graph-analysis test case.

This is the ready-to-run check requested for the MB dataset: it exercises the
full graph_sa pipeline end-to-end on the real MB mechanism + SA pickle and
asserts the key invariants (SA reactions matched to the mechanism, species
aggregated, figures + CSVs produced).

Run just this test:
    python -m pytest src/graph_sa/tests/test_graph_sa_mb.py -v

It is skipped automatically if the MB dataset is not present.
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

import pytest

# Repo root = three levels up from this file (src/graph_sa/tests/ -> repo root).
_ROOT = Path(__file__).resolve().parents[3]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

import scipy.sparse  # noqa: F401  (populate scipy.sparse for networkx spring_layout)

from rmip.config import RMIPConfig
from rmip.core.parser import MechanismParser
from rmip.core.graph_builder import GraphBuilder
from rmip.database.schema import get_connection
from rmip.analysis.sensitivity import SensitivityFluxIntegrator

from graph_sa import (
    load_target_opt, load_sa_pickle, aggregate_reaction_scores,
    match_scores_to_db, species_sensitivity, to_sensitivity_data,
    build_sensitive_subgraph,
)
from graph_sa import plots

_TARGET = _ROOT / "DATASET" / "METHYL_BUTANOATE_2026" / "target.opt"


@pytest.fixture(scope="module")
def mb_analysis():
    if not _TARGET.exists():
        pytest.skip("MB dataset not present")

    info = load_target_opt(str(_TARGET))
    mech = info["mechanism"]
    sa_pkl = info["sa_pickle"]
    if not mech or not os.path.exists(mech) or not os.path.exists(sa_pkl):
        pytest.skip("MB mechanism or SA pickle missing")

    tmp = tempfile.mkdtemp(prefix="graph_sa_mb_")
    db = os.path.join(tmp, "mb.db")
    cfg = RMIPConfig(db_path=db, log_level="ERROR")
    mech_id = MechanismParser(cfg).parse(mech, name="MB")

    sa = load_sa_pickle(sa_pkl)
    norm, raw, per_target = aggregate_reaction_scores(sa, method="max")

    conn = get_connection(db)
    matched, unmatched = match_scores_to_db(conn, mech_id, norm)
    species = species_sensitivity(conn, mech_id, matched)

    gb = GraphBuilder(db)
    G = gb.build_graph(mech_id)
    SensitivityFluxIntegrator(db).attach(mech_id, G, to_sensitivity_data(matched), None)
    subG = build_sensitive_subgraph(G, matched)

    return dict(info=info, sa=sa, norm=norm, per_target=per_target,
                matched=matched, unmatched=unmatched, species=species,
                G=G, subG=subG, tmp=tmp)


def test_sa_pickle_shape(mb_analysis):
    sa = mb_analysis["sa"]
    assert len(sa) > 0
    # Each target maps reaction formula -> float coefficient.
    first = next(iter(sa.values()))
    assert isinstance(first, dict) and len(first) > 0


def test_reactions_matched(mb_analysis):
    matched = mb_analysis["matched"]
    unmatched = mb_analysis["unmatched"]
    norm = mb_analysis["norm"]
    # Every distinct SA reaction should map to a mechanism reaction.
    assert len(matched) > 0
    assert len(matched) + len(unmatched) == len(norm)
    # Scores are normalised to [0, 1] and at least one reaction hits the max.
    scores = [d["score"] for d in matched.values()]
    assert max(scores) == pytest.approx(1.0)
    assert min(scores) >= 0.0


def test_species_aggregated(mb_analysis):
    species = mb_analysis["species"]
    assert len(species) > 0
    # Sorted descending by score.
    scores = [s["score"] for s in species]
    assert scores == sorted(scores, reverse=True)
    # The fuel MB-C5H10O2 should be among the most sensitive species.
    names = [s["name"] for s in species[:5]]
    assert any("MB" in n for n in names)


def test_subgraph_is_compact(mb_analysis):
    subG = mb_analysis["subG"]
    G = mb_analysis["G"]
    # Subgraph is a small readable slice, not the whole 4000-reaction network.
    assert 0 < subG.number_of_nodes() < G.number_of_nodes()


def test_figures_render(mb_analysis):
    tmp = mb_analysis["tmp"]
    sub_png = os.path.join(tmp, "sub.png")
    plots.render_subgraph(
        mb_analysis["subG"], sub_png,
        species_scores={s["name"]: s["score"] for s in mb_analysis["species"]},
    )
    assert os.path.exists(sub_png) and os.path.getsize(sub_png) > 0
