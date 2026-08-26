"""
rmip/reduction/coefficient_viz.py
===================================
Interactive visualization of DRG/DRGPSA interaction coefficients.

Shows the r_AB matrix three ways, all in one Dash app:
  - Tab 1: Heatmap      — Plotly heatmap of r_AB (species × species matrix)
  - Tab 2: 2D Network   — Spring-layout species graph; edge width/colour = r_AB
  - Tab 3: 3D Network   — Plotly Scatter3d with carbon-ring positions from G

Usage::

    from rmip.reduction.coefficient_viz import CoefficientVisualizer
    from rmip.types import ReductionResult
    import networkx as nx

    result = ReductionResult.load("drg_steps.json")
    viz = CoefficientVisualizer(result, G)
    viz.run(port=8055)          # blocking Dash server
    viz.export_html("coeff_viz.html")
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

log = logging.getLogger(__name__)

# Colour scale for r_AB edges: weak=grey → strong=red
_EDGE_COLORSCALE = [
    [0.0,  "#aaaaaa"],
    [0.3,  "#4da6ff"],
    [0.6,  "#ff9900"],
    [1.0,  "#ff2222"],
]


class CoefficientVisualizer:
    """
    Interactive DRG coefficient graph visualizer.

    Parameters
    ----------
    result : ReductionResult
        The reduction result whose ``r_AB`` matrix is visualised.
    G : nx.DiGraph
        Bipartite mechanism graph from ``GraphBuilder`` — used only to extract
        species node positions (``pos_x``, ``pos_y``, ``pos_x_3d``, etc.).
        If no positions are present, a spring layout is computed on-the-fly.
    threshold_default : float
        Default r_AB threshold below which edges are hidden in the network tabs.
    """

    def __init__(
        self,
        result:            "ReductionResult",
        G:                 Optional[nx.DiGraph] = None,
        threshold_default: float = 0.05,
    ) -> None:
        self._result    = result
        self._G         = G
        self._thr_def   = threshold_default
        self._r_AB      = result.r_AB or {}       # {A: {B: r}} nested dict
        self._species   = sorted(self._r_AB.keys())
        self._seeds     = set(result.target_species)

        # Precompute 2D and 3D positions
        self._pos2d: Dict[str, Tuple[float, float]] = self._extract_pos2d()
        self._pos3d: Dict[str, Tuple[float, float, float]] = self._extract_pos3d()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, port: int = 8055, debug: bool = False) -> None:
        """Launch the Dash app (blocking)."""
        app = self._build_app()
        app.run(host="127.0.0.1", port=port, debug=debug)

    def export_html(self, path: str) -> None:
        """Export a standalone HTML snapshot (three static figures)."""
        try:
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")
        import os
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig_heat   = self._heatmap_figure()
        fig_net2d  = self._network_figure_2d(self._thr_def)
        fig_net3d  = self._network_figure_3d(self._thr_def)

        html_parts = [
            "<html><head><meta charset='utf-8'></head><body>",
            f"<h2>DRG Coefficient Heatmap — {self._result.method}</h2>",
            pio.to_html(fig_heat, full_html=False, include_plotlyjs="cdn"),
            "<h2>2D Coefficient Network</h2>",
            pio.to_html(fig_net2d, full_html=False, include_plotlyjs=False),
            "<h2>3D Coefficient Network</h2>",
            pio.to_html(fig_net3d, full_html=False, include_plotlyjs=False),
            "</body></html>",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(html_parts))
        log.info("Coefficient viz exported to %s", path)

    # ── Dash app ───────────────────────────────────────────────────────────────

    def _build_app(self):
        try:
            import dash
            from dash import dcc, html
            import dash.dependencies as dd
        except ImportError:
            raise ImportError("dash is required: pip install dash")

        app = dash.Dash(__name__, title=f"RMIP Coeff Viz — {self._result.method}")

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#1e1e1e",
                   "color": "#eeeeee", "minHeight": "100vh", "padding": "10px"},
            children=[
                html.H2(
                    f"DRG Interaction Coefficients — {self._result.method} "
                    f"(mech_id={self._result.mechanism_id})",
                    style={"color": "#88ccff", "marginBottom": "4px"},
                ),
                html.Div(
                    f"{len(self._species)} species  |  "
                    f"Seeds: {', '.join(sorted(self._seeds))}",
                    style={"color": "#aaaaaa", "fontSize": "12px", "marginBottom": "12px"},
                ),
                html.Label("Edge threshold (hide r_AB below):"),
                dcc.Slider(
                    id="coeff-threshold",
                    min=0.0, max=1.0, step=0.01,
                    value=self._thr_def,
                    marks={v: str(v) for v in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dcc.Tabs(
                    id="coeff-tabs",
                    value="heatmap",
                    style={"marginTop": "8px"},
                    children=[
                        dcc.Tab(label="Heatmap",     value="heatmap",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                        dcc.Tab(label="2D Network",  value="net2d",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                        dcc.Tab(label="3D Network",  value="net3d",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                    ],
                ),
                dcc.Graph(
                    id="coeff-graph",
                    style={"height": "75vh"},
                    config={"displayModeBar": True},
                ),
            ],
        )

        @app.callback(
            dd.Output("coeff-graph", "figure"),
            [dd.Input("coeff-tabs", "value"),
             dd.Input("coeff-threshold", "value")],
        )
        def update_figure(tab, threshold):
            if tab == "heatmap":
                return self._heatmap_figure()
            elif tab == "net2d":
                return self._network_figure_2d(threshold)
            else:
                return self._network_figure_3d(threshold)

        return app

    # ── Figure builders ────────────────────────────────────────────────────────

    def _heatmap_figure(self):
        import plotly.graph_objects as go

        species = self._species
        n = len(species)
        sp_idx = {sp: i for i, sp in enumerate(species)}

        z = [[0.0] * n for _ in range(n)]
        for A, neighbours in self._r_AB.items():
            i = sp_idx.get(A)
            if i is None:
                continue
            for B, r in neighbours.items():
                j = sp_idx.get(B)
                if j is not None:
                    z[i][j] = r

        # Mark seeds with asterisk
        tick_labels = [f"* {sp}" if sp in self._seeds else sp for sp in species]

        fig = go.Figure(go.Heatmap(
            z=z, x=tick_labels, y=tick_labels,
            colorscale="Viridis",
            zmin=0.0, zmax=1.0,
            colorbar=dict(title="r_AB", tickfont=dict(color="#ccc")),
            hovertemplate="r(%{y} → %{x}) = %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=f"r_AB Interaction Matrix — {self._result.method}",
                font=dict(color="#88ccff"),
            ),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#eeeeee"),
            xaxis=dict(tickfont=dict(size=8, color="#ccc"), tickangle=-45),
            yaxis=dict(tickfont=dict(size=8, color="#ccc"), autorange="reversed"),
            margin=dict(l=100, r=60, t=60, b=120),
        )
        return fig

    def _network_figure_2d(self, threshold: float):
        import plotly.graph_objects as go

        pos = self._pos2d
        species = self._species

        # Node trace
        node_x = [pos[sp][0] for sp in species]
        node_y = [pos[sp][1] for sp in species]
        node_colors = [
            "#ffcc00" if sp in self._seeds else "#4488cc" for sp in species
        ]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=species,
            textposition="top center",
            textfont=dict(size=8, color="#eeeeee"),
            marker=dict(size=10, color=node_colors,
                        line=dict(width=1, color="#ffffff")),
            hovertemplate="%{text}<extra></extra>",
            name="Species",
        )

        # Edge traces (one per strength band to support colorscale)
        edge_traces = self._build_edge_traces_2d(pos, threshold)

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=dict(
                text=f"2D r_AB Coefficient Network (threshold={threshold:.2f})",
                font=dict(color="#88ccff"),
            ),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#eeeeee"),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    def _network_figure_3d(self, threshold: float):
        import plotly.graph_objects as go

        pos = self._pos3d
        species = self._species

        node_x = [pos[sp][0] for sp in species]
        node_y = [pos[sp][1] for sp in species]
        node_z = [pos[sp][2] for sp in species]
        node_colors = [
            "#ffcc00" if sp in self._seeds else "#4488cc" for sp in species
        ]
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            text=species,
            textfont=dict(size=7, color="#eeeeee"),
            marker=dict(
                size=6,
                color=node_colors,
                line=dict(width=1, color="#ffffff"),
            ),
            hovertemplate="%{text}<extra></extra>",
            name="Species",
        )

        edge_traces = self._build_edge_traces_3d(pos, threshold)

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=dict(
                text=f"3D r_AB Coefficient Network (threshold={threshold:.2f})",
                font=dict(color="#88ccff"),
            ),
            paper_bgcolor="#111111",
            scene=dict(
                bgcolor="#111111",
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                zaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            ),
            font=dict(color="#eeeeee"),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ── Edge helpers ───────────────────────────────────────────────────────────

    def _build_edge_traces_2d(self, pos, threshold):
        import plotly.graph_objects as go

        edges_by_band: Dict[str, list] = {"weak": [], "med": [], "strong": [], "vstrong": []}
        for A, neighbours in self._r_AB.items():
            for B, r in neighbours.items():
                if r <= threshold or A not in pos or B not in pos:
                    continue
                band = "weak" if r < 0.3 else ("med" if r < 0.6 else ("strong" if r < 0.85 else "vstrong"))
                x0, y0 = pos[A]
                x1, y1 = pos[B]
                edges_by_band[band].append((x0, y0, x1, y1, r))

        colours = {"weak": "#aaaaaa", "med": "#4da6ff", "strong": "#ff9900", "vstrong": "#ff2222"}
        widths  = {"weak": 0.5, "med": 1.5, "strong": 2.5, "vstrong": 4.0}
        traces  = []
        for band, edges in edges_by_band.items():
            if not edges:
                continue
            xs, ys = [], []
            for x0, y0, x1, y1, _ in edges:
                xs += [x0, x1, None]
                ys += [y0, y1, None]
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=colours[band], width=widths[band]),
                hoverinfo="skip",
                name=f"r {band}",
            ))
        return traces

    def _build_edge_traces_3d(self, pos, threshold):
        import plotly.graph_objects as go

        edges_by_band: Dict[str, list] = {"weak": [], "med": [], "strong": [], "vstrong": []}
        for A, neighbours in self._r_AB.items():
            for B, r in neighbours.items():
                if r <= threshold or A not in pos or B not in pos:
                    continue
                band = "weak" if r < 0.3 else ("med" if r < 0.6 else ("strong" if r < 0.85 else "vstrong"))
                x0, y0, z0 = pos[A]
                x1, y1, z1 = pos[B]
                edges_by_band[band].append((x0, y0, z0, x1, y1, z1))

        colours = {"weak": "#888888", "med": "#4da6ff", "strong": "#ff9900", "vstrong": "#ff2222"}
        widths  = {"weak": 1, "med": 2, "strong": 3, "vstrong": 5}
        traces  = []
        for band, edges in edges_by_band.items():
            if not edges:
                continue
            xs, ys, zs = [], [], []
            for x0, y0, z0, x1, y1, z1 in edges:
                xs += [x0, x1, None]
                ys += [y0, y1, None]
                zs += [z0, z1, None]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=colours[band], width=widths[band]),
                hoverinfo="skip",
                name=f"r {band}",
            ))
        return traces

    # ── Position helpers ───────────────────────────────────────────────────────

    def _extract_pos2d(self) -> Dict[str, Tuple[float, float]]:
        """Get 2D positions from G node attributes, or compute spring layout."""
        pos: Dict[str, Tuple[float, float]] = {}
        if self._G is not None:
            for node, data in self._G.nodes(data=True):
                sp = data.get("name", "")
                if data.get("type") != "species" or sp not in self._species:
                    continue
                x = data.get("pos_x")
                y = data.get("pos_y")
                if x is not None and y is not None:
                    pos[sp] = (float(x), float(y))

        missing = [sp for sp in self._species if sp not in pos]
        if missing:
            log.info("Computing spring layout for %d species without 2D positions", len(missing))
            sg = self._build_species_graph(threshold=0.0)
            sub = sg.subgraph(missing)
            spring = nx.spring_layout(sub, seed=42, k=0.5)
            for sp, (x, y) in spring.items():
                pos[sp] = (float(x), float(y))
        return pos

    def _extract_pos3d(self) -> Dict[str, Tuple[float, float, float]]:
        """Get 3D positions from G, or fall back to 2D positions with z=0."""
        pos: Dict[str, Tuple[float, float, float]] = {}
        if self._G is not None:
            for node, data in self._G.nodes(data=True):
                sp = data.get("name", "")
                if data.get("type") != "species" or sp not in self._species:
                    continue
                x = data.get("pos_x_3d")
                y = data.get("pos_y_3d")
                z = data.get("pos_z_3d")
                if x is not None and y is not None and z is not None:
                    pos[sp] = (float(x), float(y), float(z))
        # Fallback: use 2D positions at z=0
        for sp in self._species:
            if sp not in pos:
                xy = self._pos2d.get(sp, (0.0, 0.0))
                pos[sp] = (xy[0], xy[1], 0.0)
        return pos

    def _build_species_graph(self, threshold: float = 0.0) -> nx.DiGraph:
        """Build a directed species-only graph from r_AB above threshold."""
        sg = nx.DiGraph()
        sg.add_nodes_from(self._species)
        for A, neighbours in self._r_AB.items():
            for B, r in neighbours.items():
                if r > threshold:
                    sg.add_edge(A, B, weight=r)
        return sg
