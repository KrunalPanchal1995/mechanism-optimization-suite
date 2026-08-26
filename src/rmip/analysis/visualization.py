"""
rmip/analysis/visualization.py
===============================
Module 3: Interactive Network Visualizer (Phase 2).

Renders a bipartite reaction network graph as an interactive Dash application
using dash-cytoscape.  Supports single-mechanism view and side-by-side
two-mechanism comparison.

Features
--------
* Side-by-side comparison of two mechanisms.
* Union mode: ghost nodes for species/reactions absent from a mechanism.
* Fixed shared positions (preset layout) in union mode so nodes appear at the
  same location in both panels — enabling direct visual pathway comparison.
* Label toggle: full reaction formula  ↔  compact R-ID (R1, R2, …).
* Legend panel in the UI.
* Per-panel high-quality PNG download (full graph at 4× resolution).

Usage::

    from rmip.analysis.visualization import NetworkVisualizer

    # Single mechanism
    viz = NetworkVisualizer(G)
    viz.run(port=8050)

    # Side-by-side comparison with fixed shared positions (from build_union_graphs)
    viz = NetworkVisualizer([G1, G2], title=["Mech A", "Mech B"])
    viz.run(port=8050, label_mode="id")

    # Static export
    elements = viz.to_cytoscape_elements()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import networkx as nx

log = logging.getLogger("rmip.analysis.visualization")

# ── Colour palette ──────────────────────────────────────────────────────────
_COLOURS = {
    "species_universal": "#4CAF50",   # green   — UNIVERSAL across all mechs
    "species_partial":   "#FFC107",   # amber   — PARTIAL (some mechs)
    "species_unique":    "#F44336",   # red     — UNIQUE to one mech
    "species_default":   "#2196F3",   # blue    — default species
    "reaction":          "#9C27B0",   # purple  — reaction node
}

_LAYOUT_OPTIONS = ["breadthfirst", "cose", "dagre", "klay", "circle", "concentric"]

# Preset layout modes (fixed union positions, synchronized across panels)
_PRESET_MODES = {
    "preset_spring":      ("pos_x",           "pos_y"),
    "preset_circle":      ("pos_x_circle",    "pos_y_circle"),
    "preset_bipartite":   ("pos_x_bipartite", "pos_y_bipartite"),
    "preset_carbon_ring": ("pos_x_cring",     "pos_y_cring"),   # Phase 5C
}

# Score attribute name → node attribute key (for SA/flux edge coloring)
_SCORE_ATTR = {
    "importance": "importance_score",
    "sa":         "sensitivity_score",
    "flux":       "flux_score",
}

# Colour for high-importance ghost nodes (missing key species / reactions)
_GHOST_IMPORTANT_BORDER = "#E53935"   # red border — draws attention to major gaps

# Legend definition: (colour_hex, is_ghost, is_reaction, label_text)
_LEGEND = [
    ("#2196F3", False, False, "Species"),
    ("#9C27B0", False, True,  "Reaction"),
    ("#4CAF50", False, False, "Species in ALL mechanisms (comparison mode)"),
    ("#FFC107", False, False, "Species in SOME mechanisms (comparison mode)"),
    ("#F44336", False, False, "Species unique to this mechanism"),
    ("#2196F3", True,  False, "Ghost species (absent from this mechanism)"),
    ("#9C27B0", True,  True,  "Ghost reaction (absent from this mechanism)"),
    # SA/flux edge importance bands (visible only when --sa-json / --flux-json supplied)
    ("#4CAF50", False, False, "Edge — importance ≥ threshold (green, low-medium)"),
    ("#FF9800", False, False, "Edge — importance ≥ 0.5 (orange, medium-high)"),
    ("#F44336", False, False, "Edge — importance ≥ 0.8 (red, top pathway)"),
    ("#CCCCCC", False, False, "Edge — below threshold (grey, minor pathway)"),
]


class NetworkVisualizer:
    """
    Interactive graph visualizer (Module 3).

    Parameters
    ----------
    G_or_list : nx.DiGraph or list[nx.DiGraph]
        Single graph or list of up to 2 graphs for side-by-side view.
    title : str or list[str]
        Title(s) shown in the Dash app header.
    """

    def __init__(
        self,
        G_or_list: Union[nx.DiGraph, List[nx.DiGraph]],
        title: Union[str, List[str]] = "RMIP Network Viewer",
    ) -> None:
        if isinstance(G_or_list, list):
            if len(G_or_list) > 2:
                raise ValueError(
                    "NetworkVisualizer supports at most 2 graphs for side-by-side view."
                )
            self._graphs = G_or_list
            self._titles = (
                title if isinstance(title, list)
                else [title] * len(G_or_list)
            )
        else:
            self._graphs = [G_or_list]
            self._titles = [title] if isinstance(title, str) else title

        # backward-compat aliases
        self._G     = self._graphs[0]
        self._title = self._titles[0]

    # ── Public API ──────────────────────────────────────────────────────────

    def to_cytoscape_elements(
        self,
        G: Optional[nx.DiGraph] = None,
        label_mode: str = "formula",
        pos_mode: str = "spring",
        score: str = "importance",
    ) -> List[Dict[str, Any]]:
        """
        Convert a NetworkX graph to Dash Cytoscape element format.

        Parameters
        ----------
        G : nx.DiGraph, optional
            Graph to convert (defaults to the first stored graph).
        label_mode : str
            ``'formula'`` — show full reaction equation on reaction nodes.
            ``'id'``      — show compact R-label (R1, R2, …) instead.
        pos_mode : str
            Position coordinate set to embed:
            ``'spring'`` (default), ``'circle'``, ``'bipartite'``, or
            ``'carbon_ring'``.
        score : str
            Which score drives edge coloring when SA/flux data is attached:
            ``'importance'`` (default), ``'sa'``, or ``'flux'``.

        Returns
        -------
        list[dict]
            Cytoscape elements with ``"position"`` key when pre-computed
            positions are available (enables ``"preset"`` layout).
        """
        G = G if G is not None else self._G
        px_key, py_key = _PRESET_MODES.get(
            f"preset_{pos_mode}",
            _PRESET_MODES.get("preset_spring", ("pos_x", "pos_y")),
        )
        score_attr = _SCORE_ATTR.get(score, "importance_score")
        elements: List[Dict[str, Any]] = []

        # Pre-build a map from node_id → node_data for fast edge lookups
        node_data_map = dict(G.nodes(data=True))

        for node_id, data in G.nodes(data=True):
            is_ghost  = bool(data.get("ghost"))
            imp       = int(data.get("ghost_importance", 0))

            # Labels: ghost species show name; ghost reactions show formula string.
            # R-IDs are skipped for ghost reactions because both mechanisms use
            # sequential numbering starting at R1, causing collisions (GRI-R1 and
            # Davis-R1 are different reactions — showing both as "R1" in one panel
            # is misleading). Formula strings are mechanism-agnostic.
            if is_ghost:
                if data.get("type") == "reaction" and label_mode == "id":
                    # Global R-ID was assigned by _assign_global_reaction_labels —
                    # same ID as the own node in the other panel for this reaction.
                    label = data.get("reaction_label") or data.get("formula_string", "")
                elif data.get("type") == "reaction":
                    label = data.get("formula_string", "")
                else:
                    label = data.get("name", "")
            elif data.get("type") == "reaction" and label_mode == "id":
                label = data.get("reaction_label") or node_id
            else:
                label = data.get("name") or data.get("formula_string", node_id)

            colour = _node_colour(data)
            node_elem_data: Dict[str, Any] = {
                "id":               node_id,
                "label":            label,
                "type":             data.get("type", "unknown"),
                "colour":           colour,
                "ghost":            str(is_ghost),
                "ghost_importance": imp,
                **{k: str(v) for k, v in data.items() if _serialisable(v)},
            }
            # Variant tooltip (Phase 5D)
            if data.get("variant_tag"):
                node_elem_data["title"] = (
                    f"Variant of {data.get('canonical_name', '?')} "
                    f"({data.get('variant_tag', '')})"
                )
            elem: Dict[str, Any] = {"data": node_elem_data}
            # Pre-computed position → enables "preset" Cytoscape layout
            px = data.get(px_key)
            py = data.get(py_key)
            if px is not None and py is not None:
                elem["position"] = {"x": float(px), "y": float(py)}
            elements.append(elem)

        for src, tgt, edata in G.edges(data=True):
            edge_d: Dict[str, Any] = {
                "source": src,
                "target": tgt,
                "role":   edata.get("role", ""),
                "stoich": str(edata.get("stoich", "")),
            }
            # Embed importance score on edge so stylesheet selectors can fire.
            # The score lives on the adjacent reaction node (not on the edge itself).
            src_data = node_data_map.get(src, {})
            tgt_data = node_data_map.get(tgt, {})
            rxn_data = src_data if src_data.get("type") == "reaction" else tgt_data
            raw_score = rxn_data.get(score_attr)
            if raw_score is not None:
                # Ghost reaction edges always dimmed (importance = 0)
                edge_d["importance"] = 0.0 if rxn_data.get("ghost") else float(raw_score)
            elements.append({"data": edge_d})

        return elements

    def run(
        self,
        port: int = 8050,
        debug: bool = False,
        label_mode: str = "formula",
        threshold: float = 0.1,
        score: str = "importance",
        reduction_result=None,
        reduction_eps: float = 0.0,
    ) -> None:
        """
        Launch the Dash application.

        Parameters
        ----------
        port : int
            Port for the Dash dev server.
        debug : bool
            Enable Dash debug mode.
        label_mode : str
            Initial label mode: ``'formula'`` or ``'id'``.
        threshold : float
            Importance value below which edges are rendered grey (default 0.1).
            Only takes effect when SA/flux data has been attached to the graph.
        score : str
            Which score drives edge coloring: ``'importance'`` (default),
            ``'sa'``, or ``'flux'``.
        reduction_result : ReductionResult, optional
            Phase-5B reduction steps — adds a slider that dims removed
            nodes/edges at each ε threshold.
        reduction_eps : float
            Initial ε for the reduction slider (default 0.0 = full mechanism).

        Requires: ``dash``, ``dash-cytoscape``.
        """
        try:
            import dash
            import dash_cytoscape as cyto
            from dash import html, dcc
        except ImportError:
            raise ImportError(
                "NetworkVisualizer requires 'dash' and 'dash-cytoscape'. "
                "Install: pip install dash dash-cytoscape"
            )

        cyto.load_extra_layouts()
        app = dash.Dash(__name__)

        has_positions = self._has_preset_positions()

        if len(self._graphs) == 1:
            app.layout = self._single_layout(
                html, dcc, cyto, label_mode, has_positions, threshold, score,
                reduction_result=reduction_result, reduction_eps=reduction_eps,
            )
            self._register_single_callbacks(
                app, dash, score,
                reduction_result=reduction_result,
            )
        else:
            app.layout = self._dual_layout(
                html, dcc, cyto, label_mode, has_positions, threshold, score,
                reduction_result=reduction_result, reduction_eps=reduction_eps,
            )
            self._register_dual_callbacks(
                app, dash, score,
                reduction_result=reduction_result,
            )

        log.info("Starting NetworkVisualizer on http://localhost:%d", port)
        app.run(debug=debug, port=port)

    def export_html(self, path: str) -> None:
        """Export a self-contained static HTML representation (requires pyvis)."""
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("export_html requires 'pyvis': pip install pyvis")

        net = Network(height="900px", width="100%", directed=True)
        for node_id, data in self._G.nodes(data=True):
            label  = data.get("name") or data.get("formula_string", node_id)
            colour = _node_colour(data)
            net.add_node(node_id, label=label, color=colour,
                         title=str(data.get("type", "")))
        for src, tgt in self._G.edges():
            net.add_edge(src, tgt)
        net.save_graph(path)
        log.info("Exported static HTML: %s", path)

    def export_publication_figure(
        self,
        path: str,
        label_mode: str = "id",
        dpi: int = 300,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        Export a publication-quality comparison figure (PDF / SVG / PNG).

        Uses matplotlib + networkx with the pre-computed fixed positions so
        both panels are structurally identical.  Vector formats (PDF, SVG) are
        recommended for journal submissions; PNG at 300 DPI is the fallback.

        Parameters
        ----------
        path : str
            Output file.  Extension determines format: ``.pdf``, ``.svg``, or
            ``.png``.
        label_mode : str
            ``'id'``      — compact R-labels on reaction nodes (recommended).
            ``'formula'`` — full reaction formula (very cluttered for large mechs).
        dpi : int
            Resolution for raster formats (PNG).  Ignored for PDF/SVG.
        figsize : tuple, optional
            ``(width_inches, height_inches)``.  Defaults to
            ``(8, 5)`` for a single graph or ``(16, 5)`` for two panels.

        Requires: ``matplotlib``.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")       # headless backend — safe on server
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "export_publication_figure requires 'matplotlib': "
                "pip install matplotlib"
            )

        n = len(self._graphs)
        if figsize is None:
            figsize = (8 * n, 5)

        fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
        if n == 1:
            axes = [axes]

        for ax, G, title in zip(axes, self._graphs, self._titles):
            _draw_on_ax(ax, G, title, label_mode)

        # ── legend (shared across panels, placed below the figure) ──────────────
        handles = [
            mpatches.Patch(color="#2196F3", label="Species (present)"),
            mpatches.Patch(color="#9C27B0", label="Reaction (present)"),
            mpatches.Patch(color="#4CAF50", label="Species (all mechs)"),
            mpatches.Patch(color="#FFC107", label="Species (some mechs)"),
            mpatches.Patch(color="#F44336", label="Species (unique)"),
            mpatches.Patch(color="#2196F3", alpha=0.18, label="Ghost species (absent)"),
            mpatches.Patch(color="#9C27B0", alpha=0.18, label="Ghost reaction (absent)"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Exported publication figure: %s", path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _has_preset_positions(self) -> bool:
        """Return True if the first graph's nodes carry pre-computed positions."""
        for _, data in self._graphs[0].nodes(data=True):
            return data.get("pos_x") is not None
        return False

    def _layout_options(self, has_positions: bool) -> list:
        opts = []
        if has_positions:
            opts += [
                {"label": "Preset — Spring (topology, fixed)",
                 "value": "preset_spring"},
                {"label": "Preset — Circle (alphabetical, fixed)",
                 "value": "preset_circle"},
                {"label": "Preset — Bipartite (species↑ reactions↓, fixed)",
                 "value": "preset_bipartite"},
                {"label": "Preset — Carbon rings (by Cn, fixed)",
                 "value": "preset_carbon_ring"},
            ]
        opts += [{"label": l, "value": l} for l in _LAYOUT_OPTIONS]
        return opts

    def _default_layout_value(self, has_positions: bool) -> str:
        return "preset_spring" if has_positions else "breadthfirst"

    def _resolve_layout(self, layout_val: str, G: nx.DiGraph, label_mode: str,
                        score: str = "importance"):
        """Return (cytoscape_layout_dict, elements_list) for a dropdown value."""
        if layout_val in _PRESET_MODES:
            pos_mode = layout_val[len("preset_"):]   # "spring"/"circle"/"bipartite"/"carbon_ring"
            elems = self.to_cytoscape_elements(G, label_mode, pos_mode, score)
            return {"name": "preset"}, elems
        else:
            elems = self.to_cytoscape_elements(G, label_mode, "spring", score)
            return {"name": layout_val or "breadthfirst"}, elems

    # ── Shared sub-components ────────────────────────────────────────────────

    def _label_radio(self, dcc, label_mode: str):
        return dcc.RadioItems(
            id="label-radio",
            options=[
                {"label": "  Formula  ", "value": "formula"},
                {"label": "  R-ID  ",    "value": "id"},
            ],
            value=label_mode,
            inline=True,
            style={"fontFamily": "sans-serif", "fontSize": "13px"},
        )

    def _legend(self, html):
        """Return an inline legend bar."""
        items = []
        for colour, is_ghost, is_rxn, text in _LEGEND:
            icon_style: Dict[str, Any] = {
                "display":         "inline-block",
                "width":           "12px",
                "height":          "12px",
                "backgroundColor": colour if not is_ghost else "transparent",
                "marginRight":     "4px",
                "verticalAlign":   "middle",
                "borderRadius":    "0" if is_rxn else "50%",
                "border":          "1.5px dashed #999" if is_ghost else "none",
                "opacity":         "0.35" if is_ghost else "1",
            }
            items.append(
                html.Span([
                    html.Span(style=icon_style),
                    html.Span(text, style={"fontSize": "11px", "marginRight": "14px",
                                           "verticalAlign": "middle"}),
                ])
            )
        return html.Div(
            items,
            style={
                "margin":          "4px 10px",
                "padding":         "5px 10px",
                "backgroundColor": "#f8f8f8",
                "border":          "1px solid #ddd",
                "borderRadius":    "4px",
                "fontFamily":      "sans-serif",
                "display":         "flex",
                "flexWrap":        "wrap",
                "alignItems":      "center",
                "gap":             "2px",
            },
        )

    def _reduction_bar(self, html, dcc, reduction_result, reduction_eps: float):
        """Return a slider bar for Phase-5B reduction overlay (or empty Div if None)."""
        if reduction_result is None:
            return html.Div()

        steps = reduction_result.steps
        n     = len(steps)
        if n == 0:
            return html.Div()

        # Find closest step index to requested eps
        init_idx = 0
        for i, s in enumerate(steps):
            if s.threshold <= reduction_eps + 1e-9:
                init_idx = i

        eps_vals = [s.threshold for s in steps]
        marks    = {
            i: f"{eps_vals[i]:.2f}"
            for i in range(0, n, max(1, n // 10))
        }
        marks[n - 1] = f"{eps_vals[-1]:.2f}"

        step0  = steps[0]
        stepI  = steps[init_idx]
        subtitle = (
            f"{stepI.n_species}/{step0.n_species} sp, "
            f"{stepI.n_reactions}/{step0.n_reactions} rxn retained  "
            f"(ε = {stepI.threshold:.3f})"
        )

        return html.Div([
            html.Div([
                html.Span(
                    f"Reduction overlay — {reduction_result.method}  "
                    f"(seed: {', '.join(reduction_result.target_species[:5])}{'…' if len(reduction_result.target_species) > 5 else ''})",
                    style={"fontFamily": "monospace", "fontSize": "11px",
                           "color": "#555", "marginRight": "10px"},
                ),
                html.Span(
                    subtitle,
                    id="reduction-subtitle",
                    style={"fontFamily": "monospace", "fontSize": "11px",
                           "color": "#1565C0", "fontWeight": "600"},
                ),
            ], style={"margin": "2px 10px"}),
            html.Div(
                dcc.Slider(
                    id="reduction-slider",
                    min=0, max=n - 1, step=1,
                    value=init_idx,
                    marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
                style={"margin": "0 20px 4px 20px"},
            ),
        ], style={
            "backgroundColor": "#E8F5E9",
            "border":          "1px solid #A5D6A7",
            "borderRadius":    "4px",
            "margin":          "2px 10px",
            "padding":         "4px 0",
        })

    def _download_btn(self, html, panel_idx: int):
        return html.Button(
            "⬇ Download PNG",
            id=f"download-btn-{panel_idx}",
            style={
                "cursor":       "pointer",
                "padding":      "4px 10px",
                "fontSize":     "12px",
                "border":       "1px solid #aaa",
                "borderRadius": "3px",
                "background":   "#fff",
                "marginLeft":   "8px",
            },
        )

    # ── Diff stats ───────────────────────────────────────────────────────────

    def _diff_stats_bar(self, html, G: nx.DiGraph):
        """Return a compact stats bar showing coverage vs the union."""
        n_own_sp   = sum(1 for _, d in G.nodes(data=True)
                         if not d.get("ghost") and d.get("type") == "species")
        n_own_rx   = sum(1 for _, d in G.nodes(data=True)
                         if not d.get("ghost") and d.get("type") == "reaction")
        n_ghost_sp = sum(1 for _, d in G.nodes(data=True)
                         if d.get("ghost") and d.get("type") == "species")
        n_ghost_rx = sum(1 for _, d in G.nodes(data=True)
                         if d.get("ghost") and d.get("type") == "reaction")
        n_union_sp = n_own_sp + n_ghost_sp
        n_union_rx = n_own_rx + n_ghost_rx
        sp_pct = 100.0 * n_own_sp / max(n_union_sp, 1)
        rx_pct = 100.0 * n_own_rx / max(n_union_rx, 1)

        # Top missing species (highest importance ghost species)
        top_missing = sorted(
            [(d.get("name", "?"), int(d.get("ghost_importance", 0)))
             for _, d in G.nodes(data=True)
             if d.get("ghost") and d.get("type") == "species"],
            key=lambda x: -x[1],
        )[:3]
        missing_str = ", ".join(
            f"{nm}({imp})" for nm, imp in top_missing
        ) if top_missing else "none"

        def _chip(text, bg):
            return html.Span(
                text,
                style={
                    "display":         "inline-block",
                    "padding":         "1px 7px",
                    "marginRight":     "6px",
                    "borderRadius":    "10px",
                    "backgroundColor": bg,
                    "color":           "#fff",
                    "fontSize":        "11px",
                    "fontWeight":      "600",
                    "fontFamily":      "monospace",
                },
            )

        return html.Div([
            _chip(f"✓ {n_own_sp}/{n_union_sp} sp  {sp_pct:.0f}%", "#388E3C"),
            _chip(f"✓ {n_own_rx}/{n_union_rx} rx  {rx_pct:.0f}%", "#1565C0"),
            html.Span(
                f"⚠ ghost: {n_ghost_sp} sp + {n_ghost_rx} rx",
                style={"fontSize": "11px", "color": "#B71C1C",
                       "marginRight": "8px", "fontFamily": "monospace"},
            ),
            html.Span(
                f"Key missing: {missing_str}  (importance = union degree)",
                style={"fontSize": "10px", "color": "#777",
                       "fontStyle": "italic"},
            ),
        ], style={
            "padding":    "3px 8px",
            "background": "#FAFAFA",
            "border":     "1px solid #DDD",
            "borderRadius": "3px",
            "marginBottom": "2px",
            "display":    "flex",
            "alignItems": "center",
            "flexWrap":   "wrap",
            "gap":        "2px",
        })

    # ── Layout builders ──────────────────────────────────────────────────────

    def _single_layout(self, html, dcc, cyto, label_mode: str, has_positions: bool,
                       threshold: float = 0.1, score: str = "importance",
                       reduction_result=None, reduction_eps: float = 0.0):
        default_val = self._default_layout_value(has_positions)
        cyto_layout, elements = self._resolve_layout(
            default_val, self._G, label_mode, score
        )
        graph_height = "72vh" if reduction_result is not None else "82vh"
        return html.Div([
            # ── title + controls row ──
            html.Div([
                html.H2(self._title,
                        style={"fontFamily": "sans-serif", "margin": "0 12px 0 0",
                               "display": "inline-block", "fontSize": "18px"}),
                self._label_radio(dcc, label_mode),
            ], style={"display": "flex", "alignItems": "center", "margin": "8px 10px"}),

            # ── legend ──
            self._legend(html),

            # ── diff stats (union mode only) ──
            self._diff_stats_bar(html, self._G) if has_positions else html.Div(),

            # ── layout dropdown + download button ──
            html.Div([
                html.Span("Layout: ", style={"fontFamily": "sans-serif",
                                              "marginRight": "6px", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="layout-dropdown-0",
                    options=self._layout_options(has_positions),
                    value=default_val,
                    clearable=False,
                    style={"width": "290px", "display": "inline-block",
                           "verticalAlign": "middle"},
                ),
                self._download_btn(html, 0),
            ], style={"margin": "6px 10px", "display": "flex", "alignItems": "center"}),

            # ── Phase 5B reduction slider (only when --reduction-json supplied) ──
            self._reduction_bar(html, dcc, reduction_result, reduction_eps),

            # ── graph ──
            cyto.Cytoscape(
                id="graph-0",
                elements=elements,
                stylesheet=_default_stylesheet(threshold),
                layout=cyto_layout,
                style={"width": "100%", "height": graph_height},
                generateImage={},
            ),
        ])

    def _dual_layout(self, html, dcc, cyto, label_mode: str, has_positions: bool,
                     threshold: float = 0.1, score: str = "importance",
                     reduction_result=None, reduction_eps: float = 0.0):
        default_val = self._default_layout_value(has_positions)
        header = f"{self._titles[0]}  vs  {self._titles[1]}"
        graphs = self._graphs
        graph_height = "68vh" if reduction_result is not None else "78vh"

        def panel(idx):
            cyto_layout, elems = self._resolve_layout(
                default_val, graphs[idx], label_mode, score
            )
            return html.Div([
                # ── panel title + download ──
                html.Div([
                    html.H3(self._titles[idx],
                            style={"fontFamily": "sans-serif", "margin": "4px 0",
                                   "fontSize": "14px", "display": "inline-block",
                                   "flex": "1"}),
                    self._download_btn(html, idx),
                ], style={"display": "flex", "alignItems": "center"}),

                # ── diff stats bar ──
                self._diff_stats_bar(html, graphs[idx]) if has_positions else html.Div(),

                # ── graph ──
                cyto.Cytoscape(
                    id=f"graph-{idx}",
                    elements=elems,
                    stylesheet=_default_stylesheet(threshold),
                    layout=cyto_layout,
                    style={"width": "100%", "height": graph_height},
                    generateImage={},
                ),
            ], style={"width": "50%", "padding": "0 5px", "boxSizing": "border-box"})

        return html.Div([
            # ── title + label radio + SHARED layout dropdown (drives both panels) ──
            html.Div([
                html.H2(header,
                        style={"fontFamily": "sans-serif", "margin": "0 12px 0 0",
                               "display": "inline-block", "fontSize": "14px"}),
                self._label_radio(dcc, label_mode),
                html.Span("Layout: ", style={"fontFamily": "sans-serif", "fontSize": "12px",
                                              "marginLeft": "16px", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="layout-dropdown-shared",
                    options=self._layout_options(has_positions),
                    value=default_val,
                    clearable=False,
                    style={"width": "290px", "display": "inline-block",
                           "verticalAlign": "middle", "fontSize": "11px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                       "margin": "6px 10px", "gap": "4px"}),

            # ── shared legend ──
            self._legend(html),

            # ── Phase 5B reduction slider (shared across both panels) ──
            self._reduction_bar(html, dcc, reduction_result, reduction_eps),

            # ── two panels ──
            html.Div([panel(0), panel(1)],
                     style={"display": "flex", "width": "100%"}),
        ])

    # ── Callback registration ─────────────────────────────────────────────────

    def _register_single_callbacks(
        self, app, dash, score: str = "importance", reduction_result=None
    ) -> None:
        graphs = self._graphs
        viz    = self

        @app.callback(
            dash.Output("graph-0", "layout"),
            dash.Output("graph-0", "elements"),
            dash.Input("layout-dropdown-0", "value"),
            dash.Input("label-radio", "value"),
        )
        def update_panel_0(layout_val, label_mode):
            lo, elems = viz._resolve_layout(layout_val, graphs[0], label_mode, score)
            return lo, elems

        @app.callback(
            dash.Output("graph-0", "generateImage"),
            dash.Input("download-btn-0", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_0(_):
            return {
                "type": "png", "action": "download",
                "filename": "rmip_graph_0",
                "options": {"full": True, "scale": 4},
            }

        # Phase 5B — reduction slider callbacks (only when reduction data supplied)
        if reduction_result is not None:
            steps = reduction_result.steps

            @app.callback(
                dash.Output("graph-0", "stylesheet"),
                dash.Output("reduction-subtitle", "children"),
                dash.Input("reduction-slider", "value"),
            )
            def update_reduction_single(slider_idx):
                idx    = max(0, min(int(slider_idx or 0), len(steps) - 1))
                step   = steps[idx]
                step0  = steps[0]
                sheet  = _default_stylesheet_with_reduction(
                    retained_species   = set(step.retained_species),
                    retained_reactions = set(step.retained_reactions),
                )
                subtitle = (
                    f"{step.n_species}/{step0.n_species} sp, "
                    f"{step.n_reactions}/{step0.n_reactions} rxn retained  "
                    f"(ε = {step.threshold:.3f})"
                )
                return sheet, subtitle

    def _register_dual_callbacks(
        self, app, dash, score: str = "importance", reduction_result=None
    ) -> None:
        graphs = self._graphs
        viz    = self

        @app.callback(
            dash.Output("graph-0", "layout"),
            dash.Output("graph-0", "elements"),
            dash.Output("graph-1", "layout"),
            dash.Output("graph-1", "elements"),
            dash.Input("layout-dropdown-shared", "value"),
            dash.Input("label-radio", "value"),
        )
        def update_both_panels(layout_val, label_mode):
            lo0, e0 = viz._resolve_layout(layout_val, graphs[0], label_mode, score)
            lo1, e1 = viz._resolve_layout(layout_val, graphs[1], label_mode, score)
            return lo0, e0, lo1, e1

        @app.callback(
            dash.Output("graph-0", "generateImage"),
            dash.Input("download-btn-0", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_0(_):
            return {
                "type": "png", "action": "download",
                "filename": "rmip_graph_0",
                "options": {"full": True, "scale": 4},
            }

        @app.callback(
            dash.Output("graph-1", "generateImage"),
            dash.Input("download-btn-1", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_1(_):
            return {
                "type": "png", "action": "download",
                "filename": "rmip_graph_1",
                "options": {"full": True, "scale": 4},
            }

        # Phase 5B — reduction slider callbacks (dual panel: updates both stylesheets)
        if reduction_result is not None:
            steps = reduction_result.steps

            @app.callback(
                dash.Output("graph-0", "stylesheet"),
                dash.Output("graph-1", "stylesheet"),
                dash.Output("reduction-subtitle", "children"),
                dash.Input("reduction-slider", "value"),
            )
            def update_reduction_dual(slider_idx):
                idx    = max(0, min(int(slider_idx or 0), len(steps) - 1))
                step   = steps[idx]
                step0  = steps[0]
                sheet  = _default_stylesheet_with_reduction(
                    retained_species   = set(step.retained_species),
                    retained_reactions = set(step.retained_reactions),
                )
                subtitle = (
                    f"{step.n_species}/{step0.n_species} sp, "
                    f"{step.n_reactions}/{step0.n_reactions} rxn retained  "
                    f"(ε = {step.threshold:.3f})"
                )
                return sheet, sheet, subtitle


# ── Module-level helpers ────────────────────────────────────────────────────

def _node_colour(data: Dict[str, Any]) -> str:
    if data.get("type") == "reaction":
        return _COLOURS["reaction"]
    cls = data.get("classification")
    if cls == "UNIVERSAL":
        return _COLOURS["species_universal"]
    if cls == "PARTIAL":
        return _COLOURS["species_partial"]
    if cls == "UNIQUE":
        return _COLOURS["species_unique"]
    return _COLOURS["species_default"]


def _serialisable(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _draw_on_ax(ax, G: nx.DiGraph, title: str, label_mode: str) -> None:
    """Draw one mechanism graph on a matplotlib Axes for publication output."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for publication export")

    # ── Collect positions from node attributes ────────────────────────────────
    pos: Dict[str, tuple] = {}
    for nid, data in G.nodes(data=True):
        if data.get("pos_x") is not None and data.get("pos_y") is not None:
            pos[nid] = (data["pos_x"], -data["pos_y"])   # flip y: screen → math
    if not pos:
        # Fallback: compute spring layout locally
        pos = nx.spring_layout(G, seed=42, k=1.8, iterations=80, scale=1.0)

    # ── Partition nodes ───────────────────────────────────────────────────────
    def _bucket(pred):
        return [n for n, d in G.nodes(data=True) if pred(d)]

    sp_own   = _bucket(lambda d: d.get("type") == "species"  and not d.get("ghost"))
    sp_ghost = _bucket(lambda d: d.get("type") == "species"  and d.get("ghost"))
    rx_own   = _bucket(lambda d: d.get("type") == "reaction" and not d.get("ghost"))
    rx_ghost = _bucket(lambda d: d.get("type") == "reaction" and d.get("ghost"))

    node_data = dict(G.nodes(data=True))

    def _sp_colour(nid):
        cls = node_data[nid].get("classification")
        return (
            "#4CAF50" if cls == "UNIVERSAL"
            else "#FFC107" if cls == "PARTIAL"
            else "#F44336" if cls == "UNIQUE"
            else "#2196F3"
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    own_edges = [
        (u, v) for u, v in G.edges()
        if not node_data.get(u, {}).get("ghost") and
           not node_data.get(v, {}).get("ghost")
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=own_edges, ax=ax,
        arrows=True, arrowsize=8, arrowstyle="->",
        width=0.6, edge_color="#888888", alpha=0.55,
        node_size=120,          # used only for edge endpoint offset
        connectionstyle="arc3,rad=0.05",
    )

    # ── Species nodes ─────────────────────────────────────────────────────────
    if sp_own:
        nx.draw_networkx_nodes(
            G, pos, nodelist=sp_own, ax=ax,
            node_color=[_sp_colour(n) for n in sp_own],
            node_size=120, node_shape="o", linewidths=0.5, edgecolors="#000",
        )
    if sp_ghost:
        nx.draw_networkx_nodes(
            G, pos, nodelist=sp_ghost, ax=ax,
            node_color="#2196F3", node_size=80, node_shape="o",
            alpha=0.18, linewidths=0.5, edgecolors="#888",
        )

    # ── Reaction nodes ────────────────────────────────────────────────────────
    if rx_own:
        nx.draw_networkx_nodes(
            G, pos, nodelist=rx_own, ax=ax,
            node_color="#9C27B0", node_size=60, node_shape="s",
            linewidths=0.5, edgecolors="#000",
        )
    if rx_ghost:
        nx.draw_networkx_nodes(
            G, pos, nodelist=rx_ghost, ax=ax,
            node_color="#9C27B0", node_size=40, node_shape="s",
            alpha=0.18, linewidths=0.5, edgecolors="#888",
        )

    # ── Labels ────────────────────────────────────────────────────────────────
    sp_labels = {n: node_data[n].get("name", "") for n in sp_own}
    if label_mode == "id":
        rx_labels = {
            n: (node_data[n].get("reaction_label") or n.split("_")[-1])
            for n in rx_own
        }
    else:
        rx_labels = {}      # formula strings are too long for matplotlib

    nx.draw_networkx_labels(G, pos, labels=sp_labels, ax=ax,
                            font_size=6, font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=rx_labels, ax=ax,
                            font_size=4.5, font_color="#440055")

    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")
    ax.set_facecolor("white")


def _default_stylesheet(threshold: float = 0.1) -> List[Dict[str, Any]]:
    return [
        # ── default node ──
        {"selector": "node",
         "style": {
             "background-color": "data(colour)",
             "label":            "data(label)",
             "font-size":        "10px",
             "text-valign":      "bottom",
             "text-margin-y":    "3px",
             "width":            "20px",
             "height":           "20px",
             "color":            "#333",
         }},
        # ── reaction nodes — rectangles ──
        {"selector": "node[type='reaction']",
         "style": {
             "shape":  "rectangle",
             "width":  "14px",
             "height": "14px",
         }},
        # ── ghost nodes — faint, dashed, but now LABELLED ──
        {"selector": "node[ghost = 'True']",
         "style": {
             "opacity":        0.38,
             "border-width":   1.5,
             "border-style":   "dashed",
             "border-color":   "#999",
             "font-size":      "8px",
             "color":          "#999",          # dim grey label text
             "text-valign":    "bottom",
             "text-margin-y":  "2px",
         }},
        # ── high-importance ghost — more visible, red border ──
        # (ghost_importance > 4 means connected to 5+ reactions/species in union)
        {"selector": "node[ghost = 'True'][ghost_importance > 4]",
         "style": {
             "opacity":        0.60,
             "border-width":   2.0,
             "border-color":   "#E53935",       # red — key missing node
             "color":          "#C62828",
             "font-size":      "9px",
             "font-weight":    "bold",
         }},
        # ── Phase 5D: species variant badge — double orange border ──
        {"selector": "node[variant_tag]",
         "style": {
             "border-style": "double",
             "border-width": "3px",
             "border-color": "#FF6F00",
         }},
        # ── edges (default — no SA data) ──
        {"selector": "edge",
         "style": {
             "curve-style":           "bezier",
             "target-arrow-shape":    "triangle",
             "width":                 1.5,
             "line-color":            "#888",
             "target-arrow-color":    "#888",
             "opacity":               0.7,
         }},
        # ── SA/flux importance bands (fire only when edges carry 'importance' data) ──
        # Cascade order is intentional: >= 0.5 and >= 0.8 override the threshold band.
        {"selector": f"edge[importance >= 0][importance < {threshold}]",
         "style": {
             "line-color":         "#CCCCCC",
             "target-arrow-color": "#CCCCCC",
             "width":              0.8,
             "opacity":            0.2,
         }},
        {"selector": f"edge[importance >= {threshold}]",
         "style": {
             "line-color":         "#4CAF50",
             "target-arrow-color": "#4CAF50",
             "width":              1.5,
             "opacity":            0.9,
         }},
        {"selector": "edge[importance >= 0.5]",
         "style": {
             "line-color":         "#FF9800",
             "target-arrow-color": "#FF9800",
             "width":              2.5,
             "opacity":            1.0,
         }},
        {"selector": "edge[importance >= 0.8]",
         "style": {
             "line-color":         "#F44336",
             "target-arrow-color": "#F44336",
             "width":              4.0,
             "opacity":            1.0,
         }},
        # ── selected node highlight ──
        {"selector": "node:selected",
         "style": {
             "border-width": 3,
             "border-color": "#000",
             "opacity":      1.0,
         }},
    ]


def _default_stylesheet_with_reduction(
    retained_species:   "set[str]",
    retained_reactions: "set[str]",
    threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Return a stylesheet with Phase-5B reduction dimming rules appended.

    Removed species and reactions are rendered at 10% opacity with a red
    dashed border, matching PLAN.md §5B visualization spec.

    The ``retained_species`` / ``retained_reactions`` sets contain species
    *names* and reaction *formula_strings* respectively — these are stored as
    node data attributes so Cytoscape can match them.
    """
    base = _default_stylesheet(threshold)

    # Build comma-separated Cytoscape selectors for removed items.
    # Cytoscape data selectors use exact string matching on data attributes.
    # We inject per-name selectors that are NOT in the retained set.
    # This is done server-side: the callback passes the full stylesheet
    # (base + node-specific rules) so it can target individual nodes.
    #
    # Rather than generating one selector per removed node (which can be very
    # long), we use a simpler approach: mark each node with a data attribute
    # "retained" = "true"/"false" in the callback, then style via that.
    # However, since we regenerate elements only on layout changes (not on
    # slider), we use the stylesheet approach instead: nodes are annotated
    # with their name/formula in data, and we build explicit selectors.
    #
    # For brevity, the stylesheet adds a broad "dim all" rule then overrides
    # the retained nodes to be fully visible — this way only retained nodes
    # are highlighted.

    # These selectors use Cytoscape data attribute matching.
    # species nodes: data(name) / reaction nodes: data(formula_string)
    removed_sp_selectors  = []
    removed_rx_selectors  = []

    # We don't know the full species/reaction sets here, so use the inverse:
    # rather than listing removed nodes, we dim everything by default via a
    # special "reduced" class, then un-dim retained ones.
    # The cleaner approach: extend the node data with a "reduced" flag,
    # but that requires rebuilding elements. Instead we use an opaque approach:
    # - Add "dim-all" rule first
    # - Add per-name un-dim rules for retained nodes
    # This produces a large stylesheet for large mechanisms but works correctly.

    reduction_rules: List[Dict[str, Any]] = [
        # Dim all nodes by default (reduced-mode base)
        {"selector": "node",
         "style": {
             "opacity":      0.08,
             "border-style": "dashed",
             "border-color": "#E53935",
             "border-width": 1.5,
         }},
        # Dim all edges
        {"selector": "edge",
         "style": {"opacity": 0.05, "line-color": "#CCCCCC",
                   "target-arrow-color": "#CCCCCC"}},
    ]

    # Un-dim retained species nodes by name
    for sp_name in sorted(retained_species):
        safe = sp_name.replace("'", "\\'").replace('"', '\\"')
        reduction_rules.append({
            "selector": f'node[name = "{safe}"]',
            "style":    {
                "opacity":      1.0,
                "border-style": "solid",
                "border-color": "#000",
                "border-width": 0,
            },
        })

    # Un-dim retained reaction nodes by formula_string
    for formula in sorted(retained_reactions):
        safe = formula.replace('"', '\\"')
        reduction_rules.append({
            "selector": f'node[formula_string = "{safe}"]',
            "style":    {
                "opacity":      1.0,
                "border-style": "solid",
                "border-color": "#000",
                "border-width": 0,
            },
        })

    # Ghost nodes always dim further (already handled by base stylesheet)
    reduction_rules.append({
        "selector": "node[ghost = 'True']",
        "style":    {"opacity": 0.05},
    })

    return base + reduction_rules
