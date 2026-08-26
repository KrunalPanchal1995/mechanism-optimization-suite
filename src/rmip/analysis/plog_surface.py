"""
rmip/analysis/plog_surface.py
===============================
Future.md Part 2, Feature 5 — PLOG Rate Surface Visualizer.

For a PLOG (pressure-dependent, log-linear-in-P-interpolated) reaction,
build a log10 k(T,P) surface over a user-specified T/P grid and flag:
  - non-monotonic pressure dependence: at some T, k(P) evaluated at the
    mechanism's own explicit pressure levels is not monotonically
    non-decreasing (a known PLOG-fitting pathology).
  - ill-defined high-pressure limit: fewer than 2 explicit pressure levels
    are given, so there is nothing to extrapolate the high-P limit from
    (the interpolator in ``uncertainty_bands.py`` simply holds the last
    level flat above P_max — a silent extrapolation this flags explicitly).

Reuses the log-linear PLOG interpolation from
``analysis.uncertainty_bands.UncertaintyBandEstimator`` rather than
re-implementing it.

Usage::

    from rmip.analysis.plog_surface import PlogSurfaceBuilder

    b = PlogSurfaceBuilder("project.db")
    surf = b.build(mechanism_id=1, reaction_id=42)
    b.export_html(surf, "plog_surface.html")
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
from rmip.database.schema import init_db
from rmip.types import PlogSurface

log = logging.getLogger("rmip.analysis.plog_surface")

_DEFAULT_T_GRID = [T for T in range(500, 2600, 100)]
_DEFAULT_P_GRID_ATM = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


class PlogSurfaceBuilder:
    """Build and visualise a log k(T,P) surface for one PLOG reaction."""

    def __init__(
        self,
        db_path: str,
        T_grid_K: Optional[List[float]] = None,
        P_grid_atm: Optional[List[float]] = None,
    ) -> None:
        self._db = db_path
        self._conn = init_db(db_path)
        self._T_grid = T_grid_K or list(_DEFAULT_T_GRID)
        self._P_grid = P_grid_atm or list(_DEFAULT_P_GRID_ATM)
        self._estimator = UncertaintyBandEstimator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, mechanism_id: int, reaction_id: int) -> PlogSurface:
        row = self._conn.execute(
            """SELECT id, canonical_key, formula_string, kinetics_type,
                      kinetics_data_json
               FROM reactions WHERE id=? AND mechanism_id=?""",
            (reaction_id, mechanism_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"reaction_id={reaction_id} not found in mechanism_id={mechanism_id}")
        if (row["kinetics_type"] or "").lower() != "plog":
            raise ValueError(
                f"reaction_id={reaction_id} is kinetics_type="
                f"'{row['kinetics_type']}', not 'plog'"
            )

        entries = _parse_plog_entries(row["kinetics_data_json"])
        warnings: List[str] = []
        if len(entries) < 2:
            warnings.append(
                "Fewer than 2 explicit PLOG pressure levels — high-P limit is "
                "extrapolated flat from the single level given."
            )

        P_vals = [e[0] for e in entries]

        # log10 k(T,P) matrix, shape [len(T_grid)][len(P_grid)]
        matrix: List[List[float]] = []
        for T in self._T_grid:
            row_vals = []
            for P_atm in self._P_grid:
                P_Pa = P_atm * 101_325.0
                k = self._estimator._plog_at_T_P(entries, P_vals, P_Pa, T) if entries else None
                row_vals.append(math.log10(k) if (k is not None and k > 0) else float("nan"))
            matrix.append(row_vals)

        non_monotonic_T = _detect_non_monotonic(entries, self._T_grid)
        if non_monotonic_T:
            warnings.append(
                f"Non-monotonic k(P) at {len(non_monotonic_T)} T point(s) — "
                "log-linear PLOG interpolation may misbehave there."
            )

        return PlogSurface(
            canonical_key=row["canonical_key"] or "",
            formula_string=row["formula_string"],
            mechanism_id=mechanism_id,
            T_grid_K=list(self._T_grid),
            P_grid_atm=list(self._P_grid),
            log10_k_matrix=matrix,
            non_monotonic_T=non_monotonic_T,
            high_p_limit_ill_defined=len(entries) < 2,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def figure(self, surface: PlogSurface):
        """Interactive plotly Surface + contour (two-panel) figure."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "xy"}]],
            subplot_titles=("log10 k(T,P) surface", "log10 k(T,P) contour"),
        )
        fig.add_trace(
            go.Surface(
                x=surface.P_grid_atm, y=surface.T_grid_K, z=surface.log10_k_matrix,
                colorscale="Viridis", showscale=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Contour(
                x=surface.P_grid_atm, y=surface.T_grid_K, z=surface.log10_k_matrix,
                colorscale="Viridis", showscale=False,
                contours=dict(showlabels=True),
            ),
            row=1, col=2,
        )
        fig.update_scenes(
            xaxis_title="P [atm]", yaxis_title="T [K]", zaxis_title="log10 k",
        )
        fig.update_xaxes(title_text="P [atm]", type="log", row=1, col=2)
        fig.update_yaxes(title_text="T [K]", row=1, col=2)
        fig.update_layout(
            title=f"PLOG surface — {surface.formula_string}",
            template="plotly_white",
        )
        return fig

    def export_html(self, surface: PlogSurface, path: str) -> None:
        try:
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig = self.figure(surface)
        parts = [
            "<html><head><meta charset='utf-8'></head><body>",
            f"<h2>PLOG rate surface — {surface.formula_string}</h2>",
        ]
        if surface.warnings:
            parts.append("<ul>" + "".join(f"<li>{w}</li>" for w in surface.warnings) + "</ul>")
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
        parts.append("</body></html>")
        with open(path, "w") as fh:
            fh.write("\n".join(parts))
        log.info("PLOG surface exported to %s", path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plog_entries(kinetics_data_json: Optional[str]) -> List[Tuple[float, float, float, float]]:
    """
    Normalise stored PLOG data to a sorted list of (P_Pa, A_si, n, Ea_si).
    Mirrors UncertaintyBandEstimator._eval_plog's format handling.
    """
    if not kinetics_data_json:
        return []
    try:
        kdata = json.loads(kinetics_data_json)
    except (json.JSONDecodeError, TypeError):
        return []

    entries: List[Tuple[float, float, float, float]] = []
    if "pressures" in kdata and "arrhenius" in kdata:
        for P_Pa, arr in zip(kdata["pressures"], kdata["arrhenius"]):
            A = arr.get("A_si") or arr.get("A", 0.0)
            n = arr.get("n", 0.0)
            Ea = arr.get("Ea_si") or arr.get("Ea", 0.0)
            if A > 0:
                entries.append((float(P_Pa), float(A), float(n), float(Ea)))
    elif "plog_rates" in kdata:
        for rate in kdata["plog_rates"]:
            P_atm = rate.get("P_atm", 1.0)
            A_cgs = rate.get("A", 0.0)
            n_val = rate.get("n", 0.0)
            Ea_kcal = rate.get("Ea", 0.0)
            A_si = A_cgs / 1e6
            Ea_si = Ea_kcal * 4184.0
            if A_si > 0:
                entries.append((P_atm * 101_325.0, A_si, n_val, Ea_si))

    entries.sort(key=lambda e: e[0])
    return entries


def _detect_non_monotonic(
    entries: List[Tuple[float, float, float, float]],
    T_grid: List[float],
) -> List[float]:
    """
    Flag T values where k evaluated at the mechanism's own explicit PLOG
    pressure levels (not the interpolation grid) is not monotonically
    non-decreasing in P.
    """
    if len(entries) < 3:
        return []
    R = 8.314_462_618
    flagged: List[float] = []
    for T in T_grid:
        ks = []
        for _, A, n, Ea in entries:
            try:
                ks.append(A * (T ** n) * math.exp(-Ea / (R * T)))
            except (OverflowError, ValueError, ZeroDivisionError):
                ks.append(float("nan"))
        if any(math.isnan(k) for k in ks):
            continue
        diffs = [ks[i + 1] - ks[i] for i in range(len(ks) - 1)]
        if any(d < 0 for d in diffs) and any(d > 0 for d in diffs):
            flagged.append(T)
    return flagged
