"""
graph_sa.plots
=============

Static (headless) matplotlib renderers for the sensitivity graph analysis.

Two figures:

* :func:`render_subgraph` — the compact sensitive subgraph (sensitive reactions
  + their species).  Spring layout, fully labelled, colour/width graded by
  sensitivity.  This is the publication-quality "which chemistry drives the
  targets" figure.
* :func:`render_full_highlighted` — the entire mechanism drawn with the
  precomputed carbon-ring layout, everything faint grey, with the sensitive
  reactions and species overlaid in colour.  Shows where the sensitive
  chemistry sits inside the full 4000-reaction network.

Colour bands mirror RMIP's interactive stylesheet thresholds
(``_default_stylesheet``): < 0.1 grey, ≥ 0.1 green, ≥ 0.5 orange, ≥ 0.8 red.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# Sensitivity colour bands (score in [0, 1])
_GREY = "#BDBDBD"
_GREEN = "#4CAF50"
_ORANGE = "#FF9800"
_RED = "#F44336"


def _band_color(score: Optional[float], threshold: float) -> str:
    if score is None or score < threshold:
        return _GREY
    if score >= 0.8:
        return _RED
    if score >= 0.5:
        return _ORANGE
    return _GREEN


def _band_width(score: Optional[float], base: float = 1.0, span: float = 4.0) -> float:
    s = 0.0 if score is None else max(0.0, min(1.0, score))
    return base + span * s


def _rxn_score(G: nx.DiGraph, node: str) -> float:
    """Reaction sensitivity score from attached attributes (0 if unscored)."""
    data = G.nodes[node]
    for key in ("importance_score", "sensitivity_score"):
        v = data.get(key)
        if v is not None:
            return float(v)
    return 0.0


_LEGEND = [
    mpatches.Patch(color=_RED,    label="sensitivity ≥ 0.8 (top)"),
    mpatches.Patch(color=_ORANGE, label="sensitivity ≥ 0.5"),
    mpatches.Patch(color=_GREEN,  label="sensitivity ≥ threshold"),
    mpatches.Patch(color=_GREY,   label="below threshold / unscored"),
]


# ---------------------------------------------------------------------------
# Sensitive subgraph
# ---------------------------------------------------------------------------

def render_subgraph(
    subG: nx.DiGraph,
    out_png: str,
    species_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.1,
    label_top_reactions: int = 12,
    title: str = "Sensitive reactions & species",
    dpi: int = 200,
) -> str:
    """
    Render the sensitive subgraph to *out_png* and return the path.

    Reaction nodes are squares (colour/size ∝ sensitivity), species nodes are
    circles (colour/size ∝ aggregated species score).  All species are labelled;
    the ``label_top_reactions`` most sensitive reactions are labelled with their
    formula.
    """
    species_scores = species_scores or {}

    species_nodes = [n for n, d in subG.nodes(data=True) if d.get("type") == "species"]
    reaction_nodes = [n for n, d in subG.nodes(data=True) if d.get("type") == "reaction"]

    if subG.number_of_nodes() == 0:
        # Degenerate — still emit a figure so downstream never crashes.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No sensitive reactions matched to the mechanism.",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_png

    # Larger k spreads the dense hub (a fuel species may touch >10 reactions).
    pos = nx.spring_layout(subG, seed=42, k=3.2 / max(1, subG.number_of_nodes() ** 0.5),
                           iterations=400)

    n = subG.number_of_nodes()
    fig_side = max(13, min(30, 0.65 * n))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.8), facecolor="white")

    # --- edges (coloured by adjacent reaction score) ---
    for u, v in subG.edges():
        rxn = u if subG.nodes[u].get("type") == "reaction" else v
        sc = _rxn_score(subG, rxn)
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=_band_color(sc, threshold),
            linewidth=_band_width(sc, base=0.6, span=3.0),
            alpha=0.55 if sc < threshold else 0.9, zorder=1,
        )

    # --- species nodes ---
    sp_colors, sp_sizes = [], []
    for nd in species_nodes:
        name = subG.nodes[nd].get("name", "")
        sc = species_scores.get(name, 0.0)
        sp_colors.append(_band_color(sc, threshold) if sc >= threshold else "#2196F3")
        sp_sizes.append(300 + 1500 * min(1.0, sc))
    nx.draw_networkx_nodes(subG, pos, nodelist=species_nodes, node_color=sp_colors,
                           node_shape="o", node_size=sp_sizes, edgecolors="#37474F",
                           linewidths=0.8, ax=ax)

    # --- reaction nodes ---
    rx_colors, rx_sizes = [], []
    for nd in reaction_nodes:
        sc = _rxn_score(subG, nd)
        rx_colors.append(_band_color(sc, threshold))
        rx_sizes.append(120 + 700 * min(1.0, sc))
    nx.draw_networkx_nodes(subG, pos, nodelist=reaction_nodes, node_color=rx_colors,
                           node_shape="s", node_size=rx_sizes, edgecolors="#212121",
                           linewidths=0.8, ax=ax)

    # --- labels ---
    sp_labels = {nd: subG.nodes[nd].get("name", "") for nd in species_nodes}
    nx.draw_networkx_labels(subG, pos, labels=sp_labels, font_size=8,
                            font_color="#0D47A1", ax=ax)

    top_rxn = sorted(reaction_nodes, key=lambda nd: _rxn_score(subG, nd),
                     reverse=True)[:label_top_reactions]
    rx_labels = {nd: subG.nodes[nd].get("formula_string", "") for nd in top_rxn}
    nx.draw_networkx_labels(
        subG, pos, labels=rx_labels, font_size=7, font_color="#212121", ax=ax,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
    )

    ax.set_title(f"{title}\n{len(reaction_nodes)} sensitive reactions, "
                 f"{len(species_nodes)} species", fontsize=13)
    ax.legend(handles=_LEGEND, loc="upper right", fontsize=9, framealpha=0.9)
    ax.axis("off")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Full graph with sensitive parts highlighted
# ---------------------------------------------------------------------------

def render_full_highlighted(
    G: nx.DiGraph,
    out_png: str,
    threshold: float = 0.1,
    title: str = "Full mechanism — sensitive chemistry highlighted",
    dpi: int = 200,
) -> str:
    """
    Render the whole mechanism using the precomputed carbon-ring layout
    (``pos_x_cring`` / ``pos_y_cring`` — no expensive layout at render time).
    Non-sensitive chemistry is faint grey; sensitive reactions, their edges and
    their species are drawn in colour on top.
    """
    def _pos(nd):
        d = G.nodes[nd]
        return (d.get("pos_x_cring", d.get("pos_x", 0.0)),
                d.get("pos_y_cring", d.get("pos_y", 0.0)))

    pos = {nd: _pos(nd) for nd in G.nodes()}

    sensitive_rxn = {n for n, d in G.nodes(data=True)
                     if d.get("type") == "reaction" and _rxn_score(G, n) >= threshold}
    sensitive_sp = set()
    for rn in sensitive_rxn:
        sensitive_sp.update(G.predecessors(rn))
        sensitive_sp.update(G.successors(rn))

    fig, ax = plt.subplots(figsize=(20, 20), facecolor="white")

    # --- background: all edges, faint grey ---
    for u, v in G.edges():
        if u in sensitive_rxn or v in sensitive_rxn:
            continue
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#E0E0E0", linewidth=0.25, alpha=0.35, zorder=0)

    # --- background nodes ---
    bg_species = [n for n, d in G.nodes(data=True)
                  if d.get("type") == "species" and n not in sensitive_sp]
    bg_rxn = [n for n, d in G.nodes(data=True)
              if d.get("type") == "reaction" and n not in sensitive_rxn]
    nx.draw_networkx_nodes(G, pos, nodelist=bg_species, node_color="#CFD8DC",
                           node_shape="o", node_size=8, linewidths=0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=bg_rxn, node_color="#ECEFF1",
                           node_shape="s", node_size=4, linewidths=0, ax=ax)

    # --- highlighted sensitive edges ---
    for rn in sensitive_rxn:
        sc = _rxn_score(G, rn)
        for sp in set(G.predecessors(rn)) | set(G.successors(rn)):
            ax.plot([pos[rn][0], pos[sp][0]], [pos[rn][1], pos[sp][1]],
                    color=_band_color(sc, threshold),
                    linewidth=_band_width(sc, base=0.8, span=2.5),
                    alpha=0.9, zorder=2)

    # --- highlighted sensitive species + reactions ---
    nx.draw_networkx_nodes(G, pos, nodelist=list(sensitive_sp), node_color="#2196F3",
                           node_shape="o", node_size=90, edgecolors="#0D47A1",
                           linewidths=0.6, ax=ax, label="sensitive species")
    rx_colors = [_band_color(_rxn_score(G, n), threshold) for n in sensitive_rxn]
    rxn_collection = nx.draw_networkx_nodes(
        G, pos, nodelist=list(sensitive_rxn), node_color=rx_colors,
        node_shape="s", node_size=80, edgecolors="#212121", linewidths=0.6, ax=ax)
    if rxn_collection is not None:
        rxn_collection.set_zorder(3)

    # label sensitive species only (keeps the hairball legible)
    sp_labels = {nd: G.nodes[nd].get("name", "") for nd in sensitive_sp}
    nx.draw_networkx_labels(G, pos, labels=sp_labels, font_size=7,
                            font_color="#0D47A1", ax=ax)

    ax.set_title(f"{title}\n{G.number_of_nodes()} nodes; "
                 f"{len(sensitive_rxn)} sensitive reactions, "
                 f"{len(sensitive_sp)} sensitive species highlighted", fontsize=14)
    ax.legend(handles=_LEGEND, loc="upper right", fontsize=10, framealpha=0.9)
    ax.axis("off")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# =============================================================================
# postprocess() -- "process the results" flag entry point
# =============================================================================
# Same shared CLI contract as src/MUQ-SAC/VisualAid.py's postprocess() (see
# that file's docstring for the full rationale) via
# mso_io.postprocess_base. graph_sa_run.py already persists everything this
# needs: ``graph_sa.db`` (the RMIP-format SQLite mechanism graph),
# ``sensitive_reactions.csv`` / ``sensitive_species.csv`` (the matched-score
# tables), and the two PNGs render_subgraph/render_full_highlighted already
# produced. postprocess() relocates the CSVs (always tracked) and, when the
# DB is present, re-derives ``matched`` from the CSV + re-matches it against
# the DB via the same ``graph_sa.sa_bridge`` functions graph_sa_run.py itself
# calls, then re-renders both figures -- no reimplementation of the
# formula-matching or graph-building logic.

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
import sqlite3 as _sqlite3  # noqa: E402
import pandas as _pd  # noqa: E402

_this_dir_pp = _os.path.dirname(_os.path.abspath(__file__))
_src_dir_pp = _os.path.normpath(_os.path.join(_this_dir_pp, ".."))
if _src_dir_pp not in _sys.path:
    _sys.path.append(_src_dir_pp)
try:
    from mso_io.postprocess_base import (  # noqa: E402
        PostProcessResult, add_postprocess_args, default_out_dir, discover,
        formats_from_args, write_manifest, write_table,
    )
except ImportError:  # pragma: no cover
    PostProcessResult = None

_GRAPH_SA_PATTERNS = {
    "db": "**/graph_sa.db",
    "reactions_csv": "**/sensitive_reactions.csv",
    "species_csv": "**/sensitive_species.csv",
}


def postprocess(run_dir, out_dir=None, journal=False, csv_only=False,
                 formats=("png",), intermediate=False):
    """Regenerate graph-SA plots/tables from an existing (or still-running)
    ``graph_sa_run.py`` run directory. Never raises on missing files.
    Returns a :class:`mso_io.postprocess_base.PostProcessResult`."""
    if PostProcessResult is None:
        raise ImportError("mso_io.postprocess_base not importable -- is src/ on sys.path?")

    run_dir = _os.path.abspath(run_dir)
    out_dir = str(default_out_dir(run_dir, out_dir))
    found = discover(run_dir, _GRAPH_SA_PATTERNS)
    result = PostProcessResult(run_dir=run_dir, out_dir=out_dir, partial=bool(intermediate))
    figures_dir = _os.path.join(out_dir, "figures")
    tables_dir = _os.path.join(out_dir, "tables")

    rxn_df = None
    for key, label in (("reactions_csv", "sensitive_reactions"), ("species_csv", "sensitive_species")):
        for src_path in found[key]:
            df = _pd.read_csv(src_path)
            result.tables.append(write_table(df, label, tables_dir))
            if key == "reactions_csv":
                rxn_df = df

    if found["db"] and rxn_df is not None and not csv_only:
        try:
            from graph_sa.sa_bridge import build_sensitive_subgraph, match_scores_to_db
            from rmip.core.graph_builder import GraphBuilder

            db_path = str(found["db"][0])
            conn = _sqlite3.connect(db_path)
            conn.row_factory = _sqlite3.Row
            mech_row = conn.execute("SELECT id, name FROM mechanisms LIMIT 1").fetchone()
            if mech_row is None:
                raise ValueError("graph_sa.db has no mechanisms row")
            mech_id = mech_row["id"]

            score_col = "score_norm" if "score_norm" in rxn_df.columns else "score"
            formula_col = "formula" if "formula" in rxn_df.columns else rxn_df.columns[1]
            scores = dict(zip(rxn_df[formula_col].astype(str), rxn_df[score_col].astype(float)))
            matched, _unmatched = match_scores_to_db(conn, mech_id, scores)

            gb = GraphBuilder(db_path)
            G = gb.build_graph(mech_id)
            subG = build_sensitive_subgraph(G, matched)

            sub_png = _os.path.join(figures_dir, "sensitive_subgraph.png")
            _os.makedirs(figures_dir, exist_ok=True)
            render_subgraph(subG, sub_png, title=f"{mech_row['name']} — sensitive reactions & species")
            result.figures.append(sub_png)

            full_png = _os.path.join(figures_dir, "full_graph_sensitive.png")
            render_full_highlighted(G, full_png, title=f"{mech_row['name']} — full mechanism")
            result.figures.append(full_png)
        except Exception as exc:
            result.skipped.append(f"figure regeneration: {type(exc).__name__}: {exc}")
    elif not found["db"]:
        result.skipped.append(
            f"figure regeneration: no graph_sa.db found under {run_dir}"
            + (" (graph build not yet complete)" if intermediate else ""))
    elif rxn_df is None:
        result.skipped.append(f"figure regeneration: no sensitive_reactions.csv found under {run_dir}")

    if journal and rxn_df is not None:
        journal_dir = _os.path.join(out_dir, "journal_data")
        result.journal_tables.append(
            write_table(rxn_df.sort_values(rxn_df.columns[-1], ascending=False)
                        if len(rxn_df) else rxn_df, "journal_sensitive_reactions", journal_dir))

    write_manifest(result, out_dir)
    return result


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="graph_sa plots: static renderers for the sensitivity "
                     "graph analysis, plus --postprocess to regenerate "
                     "plots/tables from an existing graph_sa_run.py run directory.")
    add_postprocess_args(parser)
    args = parser.parse_args()
    if not args.postprocess:
        parser.error("nothing to do -- pass --postprocess RUN_DIR")
    result = postprocess(
        args.postprocess, out_dir=args.out, journal=args.journal,
        csv_only=args.csv_only, formats=formats_from_args(args),
        intermediate=args.intermediate,
    )
    print(f"[graph_sa.plots.postprocess] {len(result.figures)} figure(s), "
          f"{len(result.tables)} table(s), {len(result.journal_tables)} journal table(s) "
          f"-> {result.out_dir}")
    for note in result.skipped:
        print(f"  skipped: {note}")


if __name__ == "__main__":
    _main()
