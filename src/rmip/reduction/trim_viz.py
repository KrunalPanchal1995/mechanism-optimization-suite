"""
rmip/reduction/trim_viz.py
===========================
Step-by-step mechanism trim visualization.

Shows how a DRG/DRGEP/PFA/DRGPSA reduction algorithm progressively removes
species and reactions as the ε threshold increases, and how the BFS
propagation wave front spreads from seed species.

Dash app layout:
  - ε slider at top (or BFS level slider in BFS mode)
  - Tab 1: BFS Propagation — species coloured by BFS level at selected ε;
            shows the wave front reaching each species
  - Tab 2: 2D Trim Animation — bipartite mechanism graph with
            nodes coloured by retained (blue/seed green) vs removed (red, faded)
  - Tab 3: 3D Trim Animation — same as Tab 2 in Plotly Scatter3d

Usage::

    from rmip.reduction.trim_viz import TrimVisualizer
    from rmip.types import ReductionResult
    import networkx as nx

    result = ReductionResult.load("drg_steps.json")
    G = GraphBuilder(db_path).build_graph(mech_id)
    viz = TrimVisualizer(result, G)
    viz.run(port=8056)          # blocking Dash server
    viz.export_html("trim_viz.html")
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

log = logging.getLogger(__name__)

# Node colours
_COL_SEED    = "#00cc66"   # green — seed species
_COL_KEPT    = "#4499ff"   # blue  — retained non-seed
_COL_REMOVED = "#cc3333"   # red   — removed
_COL_RXN_OK  = "#9966cc"   # purple — retained reaction
_COL_RXN_RM  = "#553333"   # dark red — removed reaction

# BFS level colour ramp (level 0 = seed green → higher levels = cooler)
_BFS_COLOURS = [
    "#00cc66",   # 0 — seed
    "#22aaee",   # 1
    "#4488ff",   # 2
    "#7755cc",   # 3
    "#aa4499",   # 4
    "#dd3355",   # 5
    "#ff5511",   # 6+
]


class TrimVisualizer:
    """
    Step-by-step mechanism trim visualizer.

    Parameters
    ----------
    result : ReductionResult
        Output from any *Reducer.  Must contain at least one step.
    G : nx.DiGraph
        Bipartite mechanism graph (species + reaction nodes) from GraphBuilder.
        Node data is used for positions and type/name attributes.
    """

    def __init__(self, result: "ReductionResult", G: nx.DiGraph) -> None:
        self._result  = result
        self._G       = G
        self._steps   = result.steps
        self._seeds   = set(result.target_species)
        # Epsilon values for the slider
        self._eps_vals = [s.threshold for s in self._steps]

        # Extract species and reaction node data from G
        self._sp_data:  Dict[str, dict] = {}  # name → node attributes
        self._rxn_data: Dict[str, dict] = {}  # formula_string → node attributes
        for node, data in G.nodes(data=True):
            t = data.get("type", "")
            if t == "species":
                name = data.get("name", "")
                if name:
                    self._sp_data[name] = dict(data)
                    self._sp_data[name]["_node_id"] = node
            elif t == "reaction":
                fs = data.get("formula_string", "")
                if fs:
                    self._rxn_data[fs] = dict(data)
                    self._rxn_data[fs]["_node_id"] = node

        # Precompute 2D and 3D positions for all nodes
        self._pos2d: Dict[str, Tuple[float, float]] = self._collect_pos2d()
        self._pos3d: Dict[str, Tuple[float, float, float]] = self._collect_pos3d()

        # Edge list: list of (node_a, node_b) where both are node IDs
        self._edges: List[Tuple[str, str]] = list(G.edges())

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, port: int = 8056, debug: bool = False) -> None:
        """Launch the Dash app (blocking)."""
        app = self._build_app()
        app.run(host="127.0.0.1", port=port, debug=debug)

    def export_html(self, path: str, eps_index: int = 0) -> None:
        """Export three static figures (BFS, 2D, 3D) at the given step index."""
        try:
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")
        from pathlib import Path

        step = self._steps[eps_index]
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fig_bfs  = self._bfs_figure_2d(step)
        fig_2d   = self._trim_figure_2d(step)
        fig_3d   = self._trim_figure_3d(step)

        html_parts = [
            "<html><head><meta charset='utf-8'></head><body>",
            f"<h2>BFS Propagation — {self._result.method} "
            f"ε={step.threshold:.3f}</h2>",
            pio.to_html(fig_bfs, full_html=False, include_plotlyjs="cdn"),
            "<h2>2D Trim Snapshot</h2>",
            pio.to_html(fig_2d,  full_html=False, include_plotlyjs=False),
            "<h2>3D Trim Snapshot</h2>",
            pio.to_html(fig_3d,  full_html=False, include_plotlyjs=False),
            "</body></html>",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(html_parts))
        log.info("Trim viz exported to %s", path)

    # ── Dash app ───────────────────────────────────────────────────────────────

    def _build_app(self):
        try:
            import dash
            from dash import dcc, html
            import dash.dependencies as dd
        except ImportError:
            raise ImportError("dash is required: pip install dash")

        n_steps = len(self._steps)
        eps_marks = {}
        for i, s in enumerate(self._steps):
            if i % max(1, n_steps // 10) == 0:
                eps_marks[i] = f"{s.threshold:.2f}"

        app = dash.Dash(__name__, title=f"RMIP Trim Viz — {self._result.method}")

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#1e1e1e",
                   "color": "#eeeeee", "minHeight": "100vh", "padding": "10px"},
            children=[
                html.H2(
                    f"Mechanism Trim Animation — {self._result.method} "
                    f"(mech_id={self._result.mechanism_id})",
                    style={"color": "#88ccff", "marginBottom": "4px"},
                ),
                html.Div(id="trim-stats", style={"color": "#aaaaaa", "fontSize": "12px",
                                                  "marginBottom": "8px"}),
                html.Label("Threshold ε:"),
                dcc.Slider(
                    id="trim-eps-slider",
                    min=0, max=n_steps - 1, step=1,
                    value=0,
                    marks=eps_marks,
                    tooltip={"placement": "bottom", "always_visible": True,
                             "transform": "epxToEps"},
                ),
                dcc.Tabs(
                    id="trim-tabs",
                    value="bfs",
                    style={"marginTop": "8px"},
                    children=[
                        dcc.Tab(label="BFS Propagation", value="bfs",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                        dcc.Tab(label="2D Trim",         value="trim2d",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                        dcc.Tab(label="3D Trim",         value="trim3d",
                                style={"backgroundColor": "#2a2a2a", "color": "#ccc"},
                                selected_style={"backgroundColor": "#3a3a3a", "color": "#fff"}),
                    ],
                ),
                dcc.Graph(
                    id="trim-graph",
                    style={"height": "72vh"},
                    config={"displayModeBar": True},
                ),
            ],
        )

        @app.callback(
            [dd.Output("trim-graph", "figure"),
             dd.Output("trim-stats", "children")],
            [dd.Input("trim-tabs",       "value"),
             dd.Input("trim-eps-slider", "value")],
        )
        def update(tab, step_idx):
            step   = self._steps[int(step_idx)]
            n_sp   = len(self._sp_data)
            n_rx   = len(self._rxn_data)
            stats  = (
                f"ε = {step.threshold:.3f} | "
                f"{step.n_species}/{n_sp} species retained | "
                f"{step.n_reactions}/{n_rx} reactions retained | "
                f"Seeds: {', '.join(sorted(self._seeds))}"
            )
            if tab == "bfs":
                fig = self._bfs_figure_2d(step)
            elif tab == "trim2d":
                fig = self._trim_figure_2d(step)
            else:
                fig = self._trim_figure_3d(step)
            return fig, stats

        return app

    # ── Figure builders ────────────────────────────────────────────────────────

    def _bfs_figure_2d(self, step: "ReductionStep"):
        """Species-only graph coloured by BFS level."""
        import plotly.graph_objects as go

        bfs = step.bfs_levels or {}
        species = list(self._sp_data.keys())
        pos = self._pos2d

        # Group species by BFS level (removed = -1)
        by_level: Dict[int, List[str]] = {}
        for sp in species:
            if sp in bfs:
                lvl = bfs[sp]
            else:
                lvl = -1
            by_level.setdefault(lvl, []).append(sp)

        traces = []
        # Edge traces (species–species neighbourhood from r_AB if available)
        if self._result.r_AB:
            xs, ys = [], []
            for A, neighbours in self._result.r_AB.items():
                for B, r in neighbours.items():
                    if r > 0.01 and A in pos and B in pos:
                        xs += [pos[A][0], pos[B][0], None]
                        ys += [pos[A][1], pos[B][1], None]
            if xs:
                traces.append(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="#444444", width=0.5),
                    hoverinfo="skip", name="r_AB edges",
                ))

        max_lvl = max((v for v in bfs.values()), default=0)
        for lvl in sorted(by_level.keys()):
            sps = by_level[lvl]
            if lvl < 0:
                col, sym, size, opac = _COL_REMOVED, "x", 6, 0.25
                label = "Removed"
            else:
                col = _BFS_COLOURS[min(lvl, len(_BFS_COLOURS) - 1)]
                sym, size, opac = "circle", 10, 1.0
                label = f"Seeds (level 0)" if lvl == 0 else f"BFS level {lvl}"

            node_x = [pos[sp][0] for sp in sps if sp in pos]
            node_y = [pos[sp][1] for sp in sps if sp in pos]
            texts  = [sp for sp in sps if sp in pos]
            if not node_x:
                continue
            traces.append(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=texts,
                textposition="top center",
                textfont=dict(size=7, color="#dddddd"),
                marker=dict(
                    size=size, color=col, opacity=opac,
                    symbol=sym, line=dict(width=1, color="#ffffff"),
                ),
                name=label,
                hovertemplate="%{text}<extra></extra>",
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"BFS Propagation — ε={step.threshold:.3f}  "
                     f"(max level {max_lvl})",
                font=dict(color="#88ccff"),
            ),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#eeeeee"),
            legend=dict(
                bgcolor="#2a2a2a", bordercolor="#444",
                font=dict(color="#ccc"),
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    def _trim_figure_2d(self, step: "ReductionStep"):
        """Full bipartite mechanism graph with retained/removed colouring."""
        import plotly.graph_objects as go

        retained_sp = set(step.retained_species)
        retained_rx = set(step.retained_reactions)
        pos = self._pos2d

        # Build edges
        xs_kept, ys_kept = [], []
        xs_rm,   ys_rm   = [], []
        for a, b in self._edges:
            # Determine edge category from node types
            a_data = self._G.nodes.get(a, {})
            b_data = self._G.nodes.get(b, {})
            a_sp   = a_data.get("name", "")
            b_sp   = b_data.get("name", "")
            a_fs   = a_data.get("formula_string", "")
            b_fs   = b_data.get("formula_string", "")

            a_pos = pos.get(a_sp) or pos.get(a_fs)
            b_pos = pos.get(b_sp) or pos.get(b_fs)
            if a_pos is None or b_pos is None:
                continue

            # Edge is "kept" if both endpoints are retained
            a_kept = (a_sp and a_sp in retained_sp) or (a_fs and a_fs in retained_rx)
            b_kept = (b_sp and b_sp in retained_sp) or (b_fs and b_fs in retained_rx)
            if a_kept and b_kept:
                xs_kept += [a_pos[0], b_pos[0], None]
                ys_kept += [a_pos[1], b_pos[1], None]
            else:
                xs_rm += [a_pos[0], b_pos[0], None]
                ys_rm += [a_pos[1], b_pos[1], None]

        traces = []
        if xs_rm:
            traces.append(go.Scatter(
                x=xs_rm, y=ys_rm, mode="lines",
                line=dict(color="#553333", width=0.5),
                hoverinfo="skip", name="Removed edges", opacity=0.3,
            ))
        if xs_kept:
            traces.append(go.Scatter(
                x=xs_kept, y=ys_kept, mode="lines",
                line=dict(color="#445566", width=1.0),
                hoverinfo="skip", name="Retained edges",
            ))

        # Species nodes
        for sp, data in self._sp_data.items():
            if sp not in pos:
                continue
            x, y = pos[sp]
            if sp in self._seeds:
                col, size, opac = _COL_SEED, 12, 1.0
            elif sp in retained_sp:
                col, size, opac = _COL_KEPT, 9, 1.0
            else:
                col, size, opac = _COL_REMOVED, 7, 0.2
            traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[sp],
                textposition="top center",
                textfont=dict(size=7, color="#ccc"),
                marker=dict(size=size, color=col, opacity=opac,
                            line=dict(width=1, color="#fff")),
                hovertemplate=f"Species: {sp}<extra></extra>",
                showlegend=False,
            ))

        # Reaction nodes (diamonds)
        for fs, data in self._rxn_data.items():
            if fs not in pos:
                continue
            x, y = pos[fs]
            if fs in retained_rx:
                col, size, opac = _COL_RXN_OK, 6, 0.8
            else:
                col, size, opac = _COL_RXN_RM, 4, 0.15
            traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=size, color=col, symbol="diamond",
                            opacity=opac, line=dict(width=0.5, color="#888")),
                hovertemplate=f"Reaction: {fs}<extra></extra>",
                showlegend=False,
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"2D Trim — {self._result.method}  ε={step.threshold:.3f}  "
                     f"({step.n_species} sp / {step.n_reactions} rxn retained)",
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

    def _trim_figure_3d(self, step: "ReductionStep"):
        """3D trim figure with Plotly Scatter3d."""
        import plotly.graph_objects as go

        retained_sp = set(step.retained_species)
        retained_rx = set(step.retained_reactions)
        pos3d = self._pos3d

        # Edge lines
        xs_kept, ys_kept, zs_kept = [], [], []
        xs_rm,   ys_rm,   zs_rm   = [], [], []
        for a, b in self._edges:
            a_data = self._G.nodes.get(a, {})
            b_data = self._G.nodes.get(b, {})
            a_name = a_data.get("name") or a_data.get("formula_string", "")
            b_name = b_data.get("name") or b_data.get("formula_string", "")
            a_pos  = pos3d.get(a_name)
            b_pos  = pos3d.get(b_name)
            if a_pos is None or b_pos is None:
                continue
            a_kept = (a_data.get("name", "") in retained_sp or
                      a_data.get("formula_string", "") in retained_rx)
            b_kept = (b_data.get("name", "") in retained_sp or
                      b_data.get("formula_string", "") in retained_rx)
            if a_kept and b_kept:
                xs_kept += [a_pos[0], b_pos[0], None]
                ys_kept += [a_pos[1], b_pos[1], None]
                zs_kept += [a_pos[2], b_pos[2], None]
            else:
                xs_rm += [a_pos[0], b_pos[0], None]
                ys_rm += [a_pos[1], b_pos[1], None]
                zs_rm += [a_pos[2], b_pos[2], None]

        traces = []
        if xs_rm:
            traces.append(go.Scatter3d(
                x=xs_rm, y=ys_rm, z=zs_rm, mode="lines",
                line=dict(color="#553333", width=1),
                hoverinfo="skip", name="Removed", opacity=0.2,
            ))
        if xs_kept:
            traces.append(go.Scatter3d(
                x=xs_kept, y=ys_kept, z=zs_kept, mode="lines",
                line=dict(color="#446688", width=2),
                hoverinfo="skip", name="Retained",
            ))

        # Species
        for group, col, size, opac, sym in [
            ("seed",   _COL_SEED,    8, 1.0, "circle"),
            ("kept",   _COL_KEPT,    6, 1.0, "circle"),
            ("removed",_COL_REMOVED, 4, 0.2, "x"),
        ]:
            if group == "seed":
                members = sorted(sp for sp in self._seeds if sp in pos3d)
            elif group == "kept":
                members = sorted(
                    sp for sp in retained_sp if sp not in self._seeds and sp in pos3d
                )
            else:
                members = sorted(
                    sp for sp in self._sp_data if sp not in retained_sp and sp in pos3d
                )
            if not members:
                continue
            traces.append(go.Scatter3d(
                x=[pos3d[sp][0] for sp in members],
                y=[pos3d[sp][1] for sp in members],
                z=[pos3d[sp][2] for sp in members],
                mode="markers+text",
                text=members,
                textfont=dict(size=6, color="#ccc"),
                marker=dict(size=size, color=col, opacity=opac, symbol=sym,
                            line=dict(width=0.5, color="#fff")),
                name=group.capitalize(),
                hovertemplate="%{text}<extra></extra>",
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"3D Trim — {self._result.method}  ε={step.threshold:.3f}  "
                     f"({step.n_species} sp / {step.n_reactions} rxn retained)",
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
            legend=dict(bgcolor="#1e1e1e", bordercolor="#444", font=dict(color="#ccc")),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ── Position helpers ───────────────────────────────────────────────────────

    def _collect_pos2d(self) -> Dict[str, Tuple[float, float]]:
        pos: Dict[str, Tuple[float, float]] = {}
        for node, data in self._G.nodes(data=True):
            key = data.get("name") or data.get("formula_string", "")
            if not key:
                continue
            x = data.get("pos_x")
            y = data.get("pos_y")
            if x is not None and y is not None:
                pos[key] = (float(x), float(y))

        # For any node without position, place at origin (spring layout not run here
        # to keep startup fast for large mechs)
        for name in list(self._sp_data) + list(self._rxn_data):
            if name not in pos:
                pos[name] = (0.0, 0.0)
        return pos

    def _collect_pos3d(self) -> Dict[str, Tuple[float, float, float]]:
        pos: Dict[str, Tuple[float, float, float]] = {}
        for node, data in self._G.nodes(data=True):
            key = data.get("name") or data.get("formula_string", "")
            if not key:
                continue
            x = data.get("pos_x_3d")
            y = data.get("pos_y_3d")
            z = data.get("pos_z_3d")
            if x is not None and y is not None and z is not None:
                pos[key] = (float(x), float(y), float(z))

        # Fallback to 2D with z=0
        for name in list(self._sp_data) + list(self._rxn_data):
            if name not in pos:
                xy = self._pos2d.get(name, (0.0, 0.0))
                pos[name] = (xy[0], xy[1], 0.0)
        return pos
