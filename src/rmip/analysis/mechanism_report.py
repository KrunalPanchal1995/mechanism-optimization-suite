"""
rmip/analysis/mechanism_report.py
===================================
Future.md Part 2, Features 2 + 4 (merged per KP) — Mechanism Quality Score
and Mechanism Diff Report, combined into a single "mechanism report".

Given ONE mechanism id, ``MechanismReport.report()`` returns a composite
``QualityScore`` ("trust index"). Given TWO, it returns a `QualityScore` for
each PLUS a `MechanismDiffResult` (reactions/species added-removed, log-A
ratios for shared reactions, thermo-coefficient deltas) — reusing
``ComparisonEngine`` (M6, analysis/comparison.py) rather than re-implementing
the species/reaction cross-referencing.

Quality score components (weights renormalised over the components that are
actually available for a given mechanism):
  - smiles_resolution_pct    (species.inchi_key resolved / total)      weight 0.20
  - thermo_consistency_score (NASA-7 Cp continuity at Tmid, <5% jump)  weight 0.25
  - rate_bounds_score        (1 − flagged / evaluated, ReverseRateScanner) weight 0.25
  - hf_agreement_score       (vs optional local ATcT/JANAF reference)  weight 0.15  [optional]
  - ntc_coverage_score       (1.0 if an NTCReport shows has_ntc)       weight 0.15  [optional]

composite_score is on a 0–100 scale.

Usage::

    from rmip.analysis.mechanism_report import MechanismReport

    rep = MechanismReport("project.db")
    result = rep.report([1])          # quality only
    result = rep.report([1, 2])       # quality-each + diff
    rep.export_html(result, "report.html")
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from rmip.analysis.comparison import ComparisonEngine
from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
from rmip.database.schema import init_db
from rmip.types import MechanismDiffResult, NTCReport, QualityScore

log = logging.getLogger("rmip.analysis.mechanism_report")

_WEIGHTS = {
    "smiles_resolution": 0.20,
    "thermo_consistency": 0.25,
    "rate_bounds":       0.25,
    "hf_agreement":      0.15,
    "ntc_coverage":      0.15,
}
_TMID_DISCONTINUITY_TOL = 0.05     # 5%, matches FusionEngine convention
_HF_AGREEMENT_TOL_KJ = 5.0         # kJ/mol, matches Future.md feature 11


class QualityScorer:
    """Compute a composite 'trust index' for a single mechanism."""

    def __init__(
        self,
        db_path: str,
        atct_reference_path: Optional[str] = None,
    ) -> None:
        self._db = db_path
        self._conn = init_db(db_path)
        self._atct_ref = _load_atct_reference(atct_reference_path)

    def score(
        self,
        mechanism_id: int,
        ntc_report: Optional[NTCReport] = None,
    ) -> QualityScore:
        name_row = self._conn.execute(
            "SELECT name FROM mechanisms WHERE id=?", (mechanism_id,)
        ).fetchone()
        name = name_row["name"] if name_row else f"mech_{mechanism_id}"

        species_rows = self._conn.execute(
            "SELECT * FROM species WHERE mechanism_id=?", (mechanism_id,)
        ).fetchall()
        n_species = len(species_rows)
        n_reactions = self._conn.execute(
            "SELECT COUNT(*) AS c FROM reactions WHERE mechanism_id=?", (mechanism_id,)
        ).fetchone()["c"]

        smiles_pct = self._smiles_resolution_pct(species_rows)
        thermo_score = self._thermo_consistency_score(species_rows)
        hf_score = self._hf_agreement_score(species_rows)

        scanner = ReverseRateScanner(self._db)
        flags = scanner.scan(mechanism_id)
        rate_score = self._rate_bounds_score(flags)
        n_flagged = sum(1 for f in flags if f.flagged)

        ntc_score: Optional[float] = None
        if ntc_report is not None:
            ntc_score = 1.0 if ntc_report.has_ntc else 0.0

        composite = self._composite(
            smiles_pct / 100.0, thermo_score, rate_score, hf_score, ntc_score
        )

        return QualityScore(
            mechanism_id=mechanism_id,
            mechanism_name=name,
            smiles_resolution_pct=smiles_pct,
            thermo_consistency_score=thermo_score,
            hf_agreement_score=hf_score,
            ntc_coverage_score=ntc_score,
            rate_bounds_score=rate_score,
            composite_score=composite,
            n_reactions=n_reactions,
            n_species=n_species,
            n_reverse_rate_flags=n_flagged,
            details={
                "n_rate_pairs_evaluated": len(flags),
                "hf_reference_loaded": self._atct_ref is not None,
            },
        )

    def store(self, score: QualityScore) -> None:
        from rmip.database.schema_phase6 import insert_quality_score
        d = score.to_dict()
        insert_quality_score(
            self._conn, score.mechanism_id,
            score.smiles_resolution_pct, score.thermo_consistency_score,
            score.hf_agreement_score, score.ntc_coverage_score,
            score.rate_bounds_score, score.composite_score,
            score.n_reactions, score.n_species, score.n_reverse_rate_flags,
            details=score.details,
        )

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    @staticmethod
    def _smiles_resolution_pct(species_rows) -> float:
        if not species_rows:
            return 0.0
        n_resolved = sum(1 for r in species_rows if r["inchi_key"])
        return 100.0 * n_resolved / len(species_rows)

    @staticmethod
    def _thermo_consistency_score(species_rows) -> float:
        n_ok = 0
        n_checked = 0
        for r in species_rows:
            lo = _parse_json(r["coeffs_low_json"])
            hi = _parse_json(r["coeffs_high_json"])
            tmid = r["tmid_K"] or 1000.0
            if lo is None or hi is None:
                continue
            n_checked += 1
            cp_lo = _nasa7_cp(lo, tmid)
            cp_hi = _nasa7_cp(hi, tmid)
            denom = (abs(cp_lo) + abs(cp_hi)) / 2.0
            if denom <= 0:
                continue
            rel_jump = abs(cp_hi - cp_lo) / denom
            if rel_jump <= _TMID_DISCONTINUITY_TOL:
                n_ok += 1
        return (n_ok / n_checked) if n_checked else 1.0

    def _hf_agreement_score(self, species_rows) -> Optional[float]:
        if self._atct_ref is None:
            return None
        n_ok = 0
        n_matched = 0
        for r in species_rows:
            ref = self._atct_ref.get(r["inchi_key"]) or self._atct_ref.get(r["name"])
            if ref is None:
                continue
            n_matched += 1
            hf_mech = r["h298_kJ_mol"]
            if hf_mech is None:
                lo = _parse_json(r["coeffs_low_json"])
                if lo is None:
                    continue
                hf_mech = _nasa7_H(lo, 298.15) / 1000.0   # J/mol -> kJ/mol
            if abs(hf_mech - ref) <= _HF_AGREEMENT_TOL_KJ:
                n_ok += 1
        return (n_ok / n_matched) if n_matched else None

    @staticmethod
    def _rate_bounds_score(flags) -> float:
        if not flags:
            return 1.0
        n_flagged = sum(1 for f in flags if f.flagged)
        return 1.0 - (n_flagged / len(flags))

    @staticmethod
    def _composite(
        smiles: float, thermo: float, rate: float,
        hf: Optional[float], ntc: Optional[float],
    ) -> float:
        components = {
            "smiles_resolution": smiles,
            "thermo_consistency": thermo,
            "rate_bounds": rate,
        }
        if hf is not None:
            components["hf_agreement"] = hf
        if ntc is not None:
            components["ntc_coverage"] = ntc

        total_weight = sum(_WEIGHTS[k] for k in components)
        weighted = sum(_WEIGHTS[k] * v for k, v in components.items())
        return 100.0 * weighted / total_weight if total_weight > 0 else 0.0


class MechanismDiffEngine:
    """Two-mechanism diff report, built atop ComparisonEngine (M6)."""

    def __init__(self, db_path: str) -> None:
        self._db = db_path
        self._cmp = ComparisonEngine(db_path)

    def diff(self, mech_id_a: int, mech_id_b: int) -> MechanismDiffResult:
        report = self._cmp.compare([mech_id_a, mech_id_b])
        names = report["mechanism_names"]
        sp = report["species_comparison"]
        rx = report["reaction_comparison"]

        species_added = sp["unique"].get(str(mech_id_b), [])
        species_removed = sp["unique"].get(str(mech_id_a), [])
        reactions_added = rx["unique"].get(str(mech_id_b), [])
        reactions_removed = rx["unique"].get(str(mech_id_a), [])

        log_a_ratio: Dict[str, float] = {}
        for entry in report["kinetic_diffs"]:
            params = entry["params"]
            pa = params.get(str(mech_id_a))
            pb = params.get(str(mech_id_b))
            if pa and pb and pa.get("A_si") and pb.get("A_si") and pa["A_si"] > 0 and pb["A_si"] > 0:
                log_a_ratio[entry["canonical_key"]] = math.log10(pb["A_si"] / pa["A_si"])

        thermo_deltas: List[Dict[str, Any]] = []
        for entry in report["thermo_diffs"]:
            for cmp in entry["comparisons"]:
                thermo_deltas.append({
                    "inchi_key": entry["inchi_key"],
                    "species_name": entry["species_name"],
                    "max_abs_cpR_diff": max(
                        (d for d in cmp["abs_diff"] if d is not None), default=None
                    ),
                })

        return MechanismDiffResult(
            mechanism_id_a=mech_id_a,
            mechanism_id_b=mech_id_b,
            mechanism_name_a=names[0],
            mechanism_name_b=names[1],
            species_added=species_added,
            species_removed=species_removed,
            reactions_added=reactions_added,
            reactions_removed=reactions_removed,
            shared_reaction_log_a_ratio=log_a_ratio,
            thermo_deltas=thermo_deltas,
            n_species_shared=sp["n_universal"],
            n_reactions_shared=rx["n_shared"],
        )


class MechanismReport:
    """
    Top-level entry point: 1 mechanism id -> quality only,
    2 mechanism ids -> quality-each + diff.
    """

    def __init__(
        self,
        db_path: str,
        atct_reference_path: Optional[str] = None,
    ) -> None:
        self._db = db_path
        self._scorer = QualityScorer(db_path, atct_reference_path)
        self._diff_engine = MechanismDiffEngine(db_path)

    def report(
        self,
        mechanism_ids: List[int],
        ntc_reports: Optional[Dict[int, NTCReport]] = None,
    ) -> Dict[str, Any]:
        if len(mechanism_ids) not in (1, 2):
            raise ValueError("MechanismReport.report() accepts 1 or 2 mechanism ids")

        ntc_reports = ntc_reports or {}
        quality = {
            mid: self._scorer.score(mid, ntc_report=ntc_reports.get(mid))
            for mid in mechanism_ids
        }

        diff = None
        if len(mechanism_ids) == 2:
            diff = self._diff_engine.diff(mechanism_ids[0], mechanism_ids[1])

        return {"quality": quality, "diff": diff}

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def export_html(self, result: Dict[str, Any], path: str) -> None:
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except ImportError:
            raise ImportError("plotly is required: pip install plotly")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        parts = ["<html><head><meta charset='utf-8'></head><body>",
                 "<h1>RMIP Mechanism Report</h1>"]

        quality: Dict[int, QualityScore] = result["quality"]
        fig = self._quality_bar_figure(quality)
        parts.append("<h2>Quality Score</h2>")
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

        parts.append("<table border='1' cellpadding='6' style='border-collapse:collapse'>")
        parts.append(
            "<tr><th>Mechanism</th><th>Composite</th><th>SMILES %</th>"
            "<th>Thermo consistency</th><th>Rate bounds</th>"
            "<th>Hf agreement</th><th>NTC coverage</th>"
            "<th>Reverse-rate flags</th></tr>"
        )
        for mid, q in quality.items():
            parts.append(
                f"<tr><td>{q.mechanism_name} (id={mid})</td>"
                f"<td>{q.composite_score:.1f}</td>"
                f"<td>{q.smiles_resolution_pct:.1f}</td>"
                f"<td>{q.thermo_consistency_score:.3f}</td>"
                f"<td>{q.rate_bounds_score:.3f}</td>"
                f"<td>{'' if q.hf_agreement_score is None else f'{q.hf_agreement_score:.3f}'}</td>"
                f"<td>{'' if q.ntc_coverage_score is None else f'{q.ntc_coverage_score:.3f}'}</td>"
                f"<td>{q.n_reverse_rate_flags}</td></tr>"
            )
        parts.append("</table>")

        diff: Optional[MechanismDiffResult] = result.get("diff")
        if diff is not None:
            parts.append(
                f"<h2>Diff: {diff.mechanism_name_a} (id={diff.mechanism_id_a}) "
                f"→ {diff.mechanism_name_b} (id={diff.mechanism_id_b})</h2>"
            )
            parts.append(
                f"<p>Species added: {len(diff.species_added)} · removed: "
                f"{len(diff.species_removed)} · shared: {diff.n_species_shared}<br>"
                f"Reactions added: {len(diff.reactions_added)} · removed: "
                f"{len(diff.reactions_removed)} · shared: {diff.n_reactions_shared}</p>"
            )
            if diff.shared_reaction_log_a_ratio:
                fig2 = self._log_a_ratio_figure(diff.shared_reaction_log_a_ratio)
                parts.append(pio.to_html(fig2, full_html=False, include_plotlyjs=False))

        parts.append("</body></html>")
        with open(path, "w") as fh:
            fh.write("\n".join(parts))
        log.info("Mechanism report exported to %s", path)

    @staticmethod
    def _quality_bar_figure(quality: Dict[int, QualityScore]):
        import plotly.graph_objects as go
        names = [f"{q.mechanism_name} (id={mid})" for mid, q in quality.items()]
        scores = [q.composite_score for q in quality.values()]
        fig = go.Figure(go.Bar(x=names, y=scores, marker_color="#4f8ef7"))
        fig.update_layout(title="Composite Quality Score (0-100)",
                           yaxis_range=[0, 100], template="plotly_white")
        return fig

    @staticmethod
    def _log_a_ratio_figure(log_a_ratio: Dict[str, float]):
        import plotly.graph_objects as go
        keys = list(log_a_ratio.keys())
        vals = [log_a_ratio[k] for k in keys]
        fig = go.Figure(go.Bar(x=keys, y=vals, marker_color="#a855f7"))
        fig.update_layout(
            title="log10(A_B / A_A) for shared reactions",
            xaxis_title="canonical_key", yaxis_title="log10 ratio",
            template="plotly_white", xaxis={"visible": False},
        )
        return fig


# ---------------------------------------------------------------------------
# NASA-7 helpers
# ---------------------------------------------------------------------------

def _nasa7_cp(coeffs: List[float], T: float) -> float:
    a = coeffs
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4


def _nasa7_H(coeffs: List[float], T: float) -> float:
    """H(T) in J/mol."""
    a = coeffs
    R = 8.314_462_618
    H_RT = a[0] + a[1]*T/2.0 + a[2]*T**2/3.0 + a[3]*T**3/4.0 + a[4]*T**4/5.0 + a[5]/T
    return R * T * H_RT


def _parse_json(raw) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return [float(x) for x in val] if val else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _load_atct_reference(path: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Load an optional local ATcT/JANAF reference JSON: {inchi_key_or_name: Hf298_kJ_mol}.
    Returns None if no path given (hf_agreement_score is skipped, not zeroed).
    """
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load ATcT reference '%s': %s", path, exc)
        return None
