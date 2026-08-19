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
