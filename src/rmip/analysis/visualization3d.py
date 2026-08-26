"""
rmip/analysis/visualization3d.py
=================================
Module 3b: Interactive 3D "planetary" network visualizer (Phase 5C).

Renders a bipartite reaction network as an interactive Plotly 3D scatter
inside a Dash application.  Species sit on flat stacked rings keyed by
carbon number — C0 (H, O, OH, H2O, …) on the innermost bottom ring,
higher-Cn species on progressively larger rings in higher z-planes, like
galaxy layers.  Reaction nodes float between the layers at the centroid of
their reactants and products.

Two angular orderings are precomputed by ``GraphBuilder`` and switchable
live in the Dash app: alphabetical (``pos_*_3d``) and chemical-family arcs
(``pos_*_3d_fam``).

Positions come from ``GraphBuilder`` node attributes ``pos_x_3d`` /
``pos_y_3d`` / ``pos_z_3d`` (written by ``_assign_union_positions``), so in
union mode the same species/reaction occupies the same 3D coordinates in
every comparison panel.

Usage::

    from rmip.analysis.visualization3d import NetworkVisualizer3D

    viz = NetworkVisualizer3D([G1, G2], title=["Mech A", "Mech B"])
    viz.run(port=8050, label_mode="id")

    # Static export (self-contained HTML, no server needed)
    viz.export_html("mb_3d.html")
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Union

import networkx as nx

log = logging.getLogger("rmip.analysis.visualization3d")

_REACTION_COLOUR = "#9C27B0"   # purple — same as 2D visualizer
_GHOST_COLOUR    = "#BBBBBB"
_EDGE_COLOUR     = "rgba(120,120,120,0.28)"
_ORBIT_COLOUR    = "rgba(100,140,200,0.30)"
_SPECIES_COLOURSCALE = "Turbo"  # perceptually ordered — maps carbon number


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "NetworkVisualizer3D requires 'plotly'. Install: pip install plotly"
        )
    return go


class NetworkVisualizer3D:
    """
    Interactive 3D planetary graph visualizer (Phase 5C).

    Parameters
    ----------
    G_or_list : nx.DiGraph or list[nx.DiGraph]
        Single graph or list of up to 2 graphs for side-by-side view.
        Nodes must carry ``pos_x_3d / pos_y_3d / pos_z_3d`` attributes
        (assigned by ``GraphBuilder``).
    title : str or list[str]
        Title(s) shown in the Dash app header.
    """

    def __init__(
        self,
        G_or_list: Union[nx.DiGraph, List[nx.DiGraph]],
        title: Union[str, List[str]] = "RMIP 3D Network Viewer",
    ) -> None:
        if isinstance(G_or_list, list):
            if len(G_or_list) > 2:
                raise ValueError(
                    "NetworkVisualizer3D supports at most 2 graphs for side-by-side view."
                )
            self._graphs = G_or_list
            self._titles = (
                title if isinstance(title, list) else [title] * len(G_or_list)
            )
        else:
            self._graphs = [G_or_list]
            self._titles = [title] if isinstance(title, str) else title

        for G in self._graphs:
            for _, data in G.nodes(data=True):
                if data.get("pos_x_3d") is None:
                    raise ValueError(
                        "Graph nodes are missing 3D positions (pos_x_3d). "
                        "Build graphs with GraphBuilder.build_graph() / "
                        "build_union_graphs() so positions are assigned."
                    )
                break

    # ── Figure construction ──────────────────────────────────────────────────

    def build_figure(
        self,
        G: Optional[nx.DiGraph] = None,
        label_mode: str = "formula",
        show_edges: bool = True,
        show_orbits: bool = True,
        show_ghosts: bool = True,
        title: Optional[str] = None,
        order_mode: str = "alpha",
    ):
        """
        Build a Plotly 3D figure for one graph.

        Parameters
        ----------
        G : nx.DiGraph, optional
            Graph to render (defaults to the first stored graph).
        label_mode : str
            ``'formula'`` — reaction hover shows the full equation.
            ``'id'``      — reaction hover shows the compact R-label.
        show_edges / show_orbits / show_ghosts : bool
            Toggle the corresponding trace groups.
        title : str, optional
            Figure title.
        order_mode : str
            ``'alpha'``  — species ordered alphabetically around each ring.
            ``'family'`` — species grouped into chemical-family arcs
            (hydrocarbons, then oxygenates by O count, then N-containing).
            Falls back to alphabetical if family positions are absent
            (graphs built before this feature).
        """
        go = _require_plotly()
        G = G if G is not None else self._graphs[0]

        use_family = order_mode == "family"

        def _p(d):
            if use_family and d.get("pos_x_3d_fam") is not None:
                return (float(d["pos_x_3d_fam"]),
                        float(d["pos_y_3d_fam"]),
                        float(d["pos_z_3d_fam"]))
            return (float(d.get("pos_x_3d", 0.0)),
                    float(d.get("pos_y_3d", 0.0)),
                    float(d.get("pos_z_3d", 0.0)))

        sp_x, sp_y, sp_z, sp_cn, sp_text = [], [], [], [], []
        rx_x, rx_y, rx_z, rx_text = [], [], [], []
        gsp_x, gsp_y, gsp_z, gsp_text = [], [], [], []
        grx_x, grx_y, grx_z, grx_text = [], [], [], []

        for nid, d in G.nodes(data=True):
            x, y, z = _p(d)
            if d.get("type") == "species":
                cn = int(d.get("carbon_n", 0) or 0)
                txt = (f"<b>{d.get('name', nid)}</b>"
                       f"<br>formula: {d.get('formula') or '?'}"
                       f"<br>carbon shell: C{cn}")
                if d.get("ghost"):
                    gsp_x.append(x); gsp_y.append(y); gsp_z.append(z)
                    gsp_text.append(txt + "<br><i>ghost — absent from this mechanism</i>")
                else:
                    sp_x.append(x); sp_y.append(y); sp_z.append(z)
                    sp_cn.append(cn); sp_text.append(txt)
            elif d.get("type") == "reaction":
                if label_mode == "id" and d.get("reaction_label"):
                    lbl = f"<b>{d['reaction_label']}</b><br>{d.get('formula_string', '')}"
                else:
                    lbl = f"<b>{d.get('formula_string', nid)}</b>"
                if d.get("ghost"):
                    grx_x.append(x); grx_y.append(y); grx_z.append(z)
                    grx_text.append(lbl + "<br><i>ghost — absent from this mechanism</i>")
                else:
                    rx_x.append(x); rx_y.append(y); rx_z.append(z)
                    rx_text.append(lbl)

        traces = []

        # ── orbit rings (one horizontal circle per carbon layer) ─────────────
        if show_orbits:
            rings = sorted({
                (round(z, 3), round(math.hypot(x, y), 3))
                for x, y, z in zip(sp_x + gsp_x, sp_y + gsp_y, sp_z + gsp_z)
            })
            ox, oy, oz = [], [], []
            ts = [2 * math.pi * i / 72 for i in range(73)]
            for z, r in rings:
                if r <= 0:
                    continue
                for t in ts:
                    ox.append(r * math.cos(t))
                    oy.append(r * math.sin(t))
                    oz.append(z)
                ox.append(None); oy.append(None); oz.append(None)
            traces.append(go.Scatter3d(
                x=ox, y=oy, z=oz, mode="lines",
                line=dict(color=_ORBIT_COLOUR, width=1.5),
                hoverinfo="skip", name="Carbon orbits", showlegend=True,
            ))

        # ── edges (own chemistry only — ghosts have no edges) ────────────────
        if show_edges:
            node_pos = {nid: _p(d) for nid, d in G.nodes(data=True)}
            ex, ey, ez = [], [], []
            seen = set()
            for u, v in G.edges():
                key = (u, v) if u <= v else (v, u)
                if key in seen:            # collapse forward/reverse duplicates
                    continue
                seen.add(key)
                x0, y0, z0 = node_pos[u]
                x1, y1, z1 = node_pos[v]
                ex += [x0, x1, None]; ey += [y0, y1, None]; ez += [z0, z1, None]
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez, mode="lines",
                line=dict(color=_EDGE_COLOUR, width=1),
                hoverinfo="skip", name="Edges", showlegend=True,
            ))

        # ── ghost nodes ───────────────────────────────────────────────────────
        if show_ghosts and gsp_x:
            traces.append(go.Scatter3d(
                x=gsp_x, y=gsp_y, z=gsp_z, mode="markers",
                marker=dict(size=3, color=_GHOST_COLOUR, opacity=0.35,
                            symbol="circle"),
                text=gsp_text, hoverinfo="text",
                name="Ghost species (absent)", showlegend=True,
            ))
        if show_ghosts and grx_x:
            traces.append(go.Scatter3d(
                x=grx_x, y=grx_y, z=grx_z, mode="markers",
                marker=dict(size=2.5, color=_GHOST_COLOUR, opacity=0.3,
                            symbol="diamond"),
                text=grx_text, hoverinfo="text",
                name="Ghost reactions (absent)", showlegend=True,
            ))

        # ── reaction nodes ─────────────────────────────────────────────────────
        if rx_x:
            traces.append(go.Scatter3d(
                x=rx_x, y=rx_y, z=rx_z, mode="markers",
                marker=dict(size=2.5, color=_REACTION_COLOUR, opacity=0.55,
                            symbol="diamond"),
                text=rx_text, hoverinfo="text",
                name="Reactions", showlegend=True,
            ))

        # ── species nodes (coloured by carbon number) ─────────────────────────
        if sp_x:
            traces.append(go.Scatter3d(
                x=sp_x, y=sp_y, z=sp_z, mode="markers",
                marker=dict(
                    size=5, color=sp_cn, colorscale=_SPECIES_COLOURSCALE,
                    cmin=0, cmax=max(sp_cn) if sp_cn else 1,
                    opacity=0.95, symbol="circle",
                    colorbar=dict(title="Cn", thickness=12, len=0.55, x=1.0),
                ),
                text=sp_text, hoverinfo="text",
                name="Species (colour = carbon number)", showlegend=True,
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title or "", font=dict(size=14)),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                bgcolor="rgb(10,10,25)",     # deep-space background
            ),
            paper_bgcolor="rgb(10,10,25)",
            font=dict(color="#DDD"),
            legend=dict(x=0.0, y=1.0, bgcolor="rgba(0,0,0,0.3)",
                        font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=0),
            uirevision="keep-camera",         # preserve camera on toggle updates
        )
        return fig

    # ── Static export ──────────────────────────────────────────────────────────

    def export_html(self, path: str, label_mode: str = "formula",
                    order_mode: str = "alpha") -> None:
        """Export self-contained interactive HTML (first graph only)."""
        fig = self.build_figure(self._graphs[0], label_mode=label_mode,
                                order_mode=order_mode, title=self._titles[0])
        fig.write_html(path, include_plotlyjs="cdn")
        log.info("Exported 3D HTML: %s", path)

    # ── Dash app ──────────────────────────────────────────────────────────────

    def run(
        self,
        port: int = 8050,
        debug: bool = False,
        label_mode: str = "formula",
        order_mode: str = "alpha",
    ) -> None:
        """
        Launch the Dash application (single or side-by-side dual panel).

        Requires: ``dash``, ``plotly``.
        """
        _require_plotly()
        try:
            import dash
            from dash import html, dcc
        except ImportError:
            raise ImportError(
                "NetworkVisualizer3D requires 'dash'. Install: pip install dash"
            )

        app = dash.Dash(__name__)
        n = len(self._graphs)
        viz = self

        controls = html.Div([
            html.Span("Reaction hover: ",
                      style={"fontSize": "12px", "marginRight": "6px"}),
            dcc.RadioItems(
                id="label-radio",
                options=[{"label": " Formula ", "value": "formula"},
                         {"label": " R-ID ",    "value": "id"}],
                value=label_mode, inline=True,
                style={"fontSize": "12px", "display": "inline-block"},
            ),
            html.Span("Ring order: ",
                      style={"fontSize": "12px", "margin": "0 6px 0 18px"}),
            dcc.RadioItems(
                id="order-radio",
                options=[{"label": " Alphabetical ",    "value": "alpha"},
                         {"label": " Chemical family ", "value": "family"}],
                value=order_mode, inline=True,
                style={"fontSize": "12px", "display": "inline-block"},
            ),
            dcc.Checklist(
                id="toggles",
                options=[{"label": " Edges ",         "value": "edges"},
                         {"label": " Carbon orbits ", "value": "orbits"},
                         {"label": " Ghost nodes ",   "value": "ghosts"}],
                value=["edges", "orbits", "ghosts"], inline=True,
                style={"fontSize": "12px", "display": "inline-block",
                       "marginLeft": "18px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "margin": "4px 10px", "fontFamily": "sans-serif",
                  "color": "#DDD"})

        def _panel(idx):
            return html.Div([
                dcc.Graph(
                    id=f"graph3d-{idx}",
                    figure=self.build_figure(
                        self._graphs[idx], label_mode=label_mode,
                        order_mode=order_mode, title=self._titles[idx],
                    ),
                    style={"height": "86vh"},
                    config={"displaylogo": False},
                ),
            ], style={"width": f"{100 // n}%", "display": "inline-block",
                      "verticalAlign": "top"})

        header = "  vs  ".join(self._titles) if n > 1 else self._titles[0]
        app.layout = html.Div([
            html.H2(f"RMIP 3D planetary view — {header}",
                    style={"fontFamily": "sans-serif", "fontSize": "16px",
                           "margin": "6px 10px", "color": "#EEE"}),
            controls,
            html.Div([_panel(i) for i in range(n)]),
        ], style={"backgroundColor": "rgb(10,10,25)"})

        outputs = [dash.Output(f"graph3d-{i}", "figure") for i in range(n)]

        @app.callback(
            *outputs,
            dash.Input("label-radio", "value"),
            dash.Input("order-radio", "value"),
            dash.Input("toggles", "value"),
        )
        def _update(lbl, order, toggles):
            figs = [
                viz.build_figure(
                    viz._graphs[i], label_mode=lbl, order_mode=order,
                    show_edges="edges" in toggles,
                    show_orbits="orbits" in toggles,
                    show_ghosts="ghosts" in toggles,
                    title=viz._titles[i],
                )
                for i in range(n)
            ]
            return figs[0] if n == 1 else tuple(figs)

        log.info("Starting NetworkVisualizer3D on http://localhost:%d", port)
        app.run(debug=debug, port=port)
