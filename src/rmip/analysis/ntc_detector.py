"""
rmip/analysis/ntc_detector.py
===============================
Future.md Part 2, Feature 7 (extended per KP) — Ignition Delay NTC Detection
+ HTT/LTT target classification.

From a φ-fixed IDT sweep over a temperature range, detect whether the
mechanism reproduces the negative-temperature-coefficient (NTC) behaviour
typical of n-alkanes / gasoline surrogates, and — per KP's extension —
use the detected onset to split ALL downstream optimization targets into:

  HTT (High-Temperature Target)  — T > T_NTC_onset
  LTT (Low-Temperature Target)   — T < T_NTC_onset

on the 1/T (Arrhenius) scale. This HTT/LTT split REPLACES the fixed
850 K / 1000 K LTC/NTC/HTC thresholds wherever a mechanism has a detected
NTC onset (``analysis/benchmarking.py`` falls back to the fixed thresholds
only when no ``ntc_runs`` row exists for the mechanism).

Physics
-------
Sweeping T from low to high at fixed φ, P:
  - LTC branch : τ_ign decreases as T increases (normal Arrhenius behaviour)
  - NTC onset  : local MINIMUM of τ_ign(T) — τ starts increasing with T
  - NTC end    : local MAXIMUM of τ_ign(T) — τ resumes decreasing with T
  - HTC branch : τ_ign decreases again (normal Arrhenius behaviour)

Detection is via sign changes in the discrete derivative of ln(τ) vs T,
with a dead-zone (``rel_tol``) to reject noise-level fluctuations.

Usage::

    from rmip.analysis.ntc_detector import NTCDetector

    det = NTCDetector(mechanism_id=1, db_path="project.db")
    report = det.detect(fuel="NC7H16:1.0", phi=1.0, P_atm=13.5,
                         T_min=650, T_max=1250, n_points=25)
    det.store(report)
    det.tag_exp_dataset(dataset_id=3, report=report)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

from rmip.types import NTCReport, SimCondition, TempTargetClass

log = logging.getLogger("rmip.analysis.ntc_detector")


class NTCDetector:
    """
    Detect the NTC region of an IDT(T) sweep and split targets HTT/LTT.

    Parameters
    ----------
    mechanism_id : int
    db_path : str
    rel_tol : float
        Minimum relative change in ln(IDT) between consecutive sweep points
        to be considered a real slope (vs. noise). Default 1%.
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path: str,
        rel_tol: float = 0.01,
    ) -> None:
        self.mechanism_id = mechanism_id
        self._db = db_path
        self._rel_tol = rel_tol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        fuel: str,
        phi: float = 1.0,
        P_atm: float = 1.0,
        oxidizer: str = "O2:1.0, N2:3.76",
        T_min: float = 650.0,
        T_max: float = 1300.0,
        n_points: int = 25,
        n_workers: int = 1,
    ) -> NTCReport:
        """Run a φ-fixed IDT sweep and detect the NTC region."""
        from rmip.simulation.idt_runner import IDTRunner

        T_grid = _linspace(T_min, T_max, n_points)
        conditions = [
            SimCondition(T_K=T, P_atm=P_atm, phi=phi, fuel=fuel, oxidizer=oxidizer)
            for T in T_grid
        ]

        runner = IDTRunner(mechanism_id=self.mechanism_id, db_path=self._db)
        df = runner.run_batch(conditions, n_workers=n_workers)
        idt_s: List[Optional[float]] = [
            (float(v) if v is not None and not _isnan(v) else None)
            for v in df["idt_s"].tolist()
        ]

        onset, end, ratio, has_ntc = self._find_ntc(T_grid, idt_s)

        htt = [T for T in T_grid if onset is not None and T > onset]
        ltt = [T for T in T_grid if onset is not None and T < onset]

        return NTCReport(
            mechanism_id=self.mechanism_id,
            fuel=fuel, phi=phi, P_atm=P_atm,
            T_grid_K=T_grid, idt_s=idt_s,
            has_ntc=has_ntc,
            T_NTC_onset=onset, T_NTC_end=end, IDT_ratio_NTC=ratio,
            htt_points_K=htt, ltt_points_K=ltt,
        )

    def store(self, report: NTCReport) -> int:
        """Persist the report to the ``ntc_runs`` table."""
        import sqlite3
        from rmip.database.schema import init_db
        from rmip.database.schema_phase6 import insert_ntc_run

        conn = init_db(self._db)
        return insert_ntc_run(
            conn, report.mechanism_id, report.fuel, report.phi, report.P_atm,
            report.T_grid_K, report.idt_s, report.has_ntc,
            report.T_NTC_onset, report.T_NTC_end, report.IDT_ratio_NTC,
            len(report.htt_points_K), len(report.ltt_points_K),
        )

    def tag_exp_dataset(self, dataset_id: int, report: NTCReport) -> int:
        """
        Classify every point in an ``exp_data_points`` dataset as HTT/LTT
        using ``report.T_NTC_onset``, for downstream MSO target selection.
        Returns the number of rows updated.
        """
        from rmip.database.schema import init_db

        conn = init_db(self._db)
        rows = conn.execute(
            "SELECT id, T_K FROM exp_data_points WHERE dataset_id=?", (dataset_id,)
        ).fetchall()

        n_updated = 0
        for r in rows:
            if r["T_K"] is None:
                continue
            cls = classify_temperature(r["T_K"], report)
            if cls is None:
                continue
            conn.execute(
                "UPDATE exp_data_points SET temp_target_class=? WHERE id=?",
                (cls, r["id"]),
            )
            n_updated += 1
        conn.commit()
        log.info("Tagged %d/%d points in dataset_id=%d as HTT/LTT (onset=%s K)",
                  n_updated, len(rows), dataset_id, report.T_NTC_onset)
        return n_updated

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def figure(self, report: NTCReport):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")

        inv_T = [1000.0 / T for T in report.T_grid_K]
        idt_ms = [v * 1000.0 if v is not None else None for v in report.idt_s]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=inv_T, y=idt_ms, mode="lines+markers", name="IDT",
            line=dict(color="#4f8ef7"),
        ))
        if report.T_NTC_onset is not None:
            fig.add_vline(x=1000.0 / report.T_NTC_onset, line_dash="dash",
                          line_color="#22c55e",
                          annotation_text=f"onset {report.T_NTC_onset:.0f} K")
        if report.T_NTC_end is not None:
            fig.add_vline(x=1000.0 / report.T_NTC_end, line_dash="dash",
                          line_color="#ef4444",
                          annotation_text=f"end {report.T_NTC_end:.0f} K")
        fig.update_yaxes(type="log", title="Ignition delay [ms]")
        fig.update_xaxes(title="1000/T [1/K]")
        fig.update_layout(
            title=f"IDT sweep — {report.fuel}, φ={report.phi}, P={report.P_atm} atm "
                  f"(NTC {'detected' if report.has_ntc else 'not detected'})",
            template="plotly_white",
        )
        return fig

    def export_html(self, report: NTCReport, path: str) -> None:
        try:
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig = self.figure(report)
        with open(path, "w") as fh:
            fh.write("<html><head><meta charset='utf-8'></head><body>")
            fh.write(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
            fh.write("</body></html>")
        log.info("NTC report exported to %s", path)

    # ------------------------------------------------------------------
    # Internal detection algorithm
    # ------------------------------------------------------------------

    def _find_ntc(
        self,
        T_grid: List[float],
        idt_s: List[Optional[float]],
    ) -> tuple:
        """
        Return (T_NTC_onset, T_NTC_end, IDT_ratio_NTC, has_ntc).
        Onset = local minimum of ln(IDT) vs T (τ starts rising with T).
        End   = subsequent local maximum (τ resumes falling with T).
        """
        # Build sign-of-slope sequence over valid consecutive pairs, using a
        # dead-zone so noise-level wiggles don't register as a slope flip.
        signs: List[int] = []
        idx_map: List[int] = []   # signs[k] corresponds to interval (idx_map[k], idx_map[k]+1)
        for i in range(len(T_grid) - 1):
            a, b = idt_s[i], idt_s[i + 1]
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            rel = (math.log(b) - math.log(a))
            if abs(rel) < self._rel_tol:
                continue
            signs.append(1 if rel > 0 else -1)
            idx_map.append(i)

        onset_i: Optional[int] = None
        end_i: Optional[int] = None
        for k in range(1, len(signs)):
            if onset_i is None and signs[k - 1] < 0 and signs[k] > 0:
                onset_i = idx_map[k]
                continue
            if onset_i is not None and end_i is None and signs[k - 1] > 0 and signs[k] < 0:
                end_i = idx_map[k]
                break

        if onset_i is None:
            return None, None, None, False

        T_onset = T_grid[onset_i]
        T_end = T_grid[end_i] if end_i is not None else None
        ratio = None
        if end_i is not None and idt_s[onset_i] and idt_s[end_i]:
            ratio = idt_s[end_i] / idt_s[onset_i]

        return T_onset, T_end, ratio, True


def classify_temperature(T_K: float, report: NTCReport) -> Optional[str]:
    """Classify a single temperature as HTT/LTT relative to an NTCReport's onset."""
    return report.classify(T_K)


def split_targets_by_ntc(
    conditions: List[SimCondition],
    report: NTCReport,
) -> tuple:
    """
    Split a list of SimCondition targets into (htt, ltt) lists using the
    detected T_NTC_onset. Targets are left unclassified (dropped from both
    lists) only if ``report.T_NTC_onset`` is None (no NTC detected — callers
    should fall back to the fixed LTC/NTC/HTC regime thresholds instead).
    """
    if report.T_NTC_onset is None:
        return [], []
    htt = [c for c in conditions if c.T_K > report.T_NTC_onset]
    ltt = [c for c in conditions if c.T_K < report.T_NTC_onset]
    return htt, ltt


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
