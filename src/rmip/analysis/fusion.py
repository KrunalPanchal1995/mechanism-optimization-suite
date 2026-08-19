"""
rmip/analysis/fusion.py
=======================
Module 7: Mechanism Fusion Engine

What it does
------------
Given a ``GraphComparisonResult`` (from M6 GraphComparator) and the originating
DB, the FusionEngine assembles a new *fused* mechanism by:

  1. VERBATIM  reactions  — present in exactly one mechanism → copy unchanged.
  2. COPY      reactions  — present in ≥2 mechanisms, A/n/Ea identical within
                            tolerance → copy from the first (alphabetically by
                            mechanism name) unchanged.
  3. FUSE      reactions  — present in ≥2 mechanisms with differing kinetics →
                            replace A/n/Ea with ParameterEstimator consensus
                            (geometric mean A, arithmetic mean n/Ea) and store
                            min/max as the MSO optimisation bounds.

Species treatment
-----------------
  * UNIVERSAL / PARTIAL / UNIQUE species: ALL are included in the fused mechanism.
  * Thermo: arithmetic mean of NASA-7 coefficients per-coefficient.
    If Tmid differs across sources, the most common Tmid wins (warning logged).
  * Transport: NOT handled (design decision D6).

DB storage
----------
The fused mechanism is written to a new ``mechanisms`` row with
  ``format = "fused"``
  ``metadata_json`` carries the fusion provenance (source IDs, treatment counts,
  tolerances used).

The ``reactions`` rows for the fused mechanism carry two extra columns (added by
Phase-3 migrations):
  ``fusion_class`` TEXT  — "VERBATIM" | "COPY" | "FUSE"
  ``source_mechanism_ids_json`` TEXT — JSON list of source IDs for that reaction

All thermo for the fused mechanism is stored in ``species`` rows as usual.

Dependencies
------------
  rmip.analysis.parameter_estimator  (ArrheniusEstimate, ThermoEstimate)
  rmip.types                          (GraphComparisonResult, ReactionClass,
                                       SpeciesClass, FusionTreatment, ...)
  rmip.database.schema                (plain sqlite3 CRUD)

Design decisions inherited
--------------------------
D1: species identity by inchi_key
D2: reaction identity by canonical_key
D3: SQLite only, no ORM
D6: transport data out of scope

Usage
-----
>>> from rmip.analysis.fusion import FusionEngine
>>> engine = FusionEngine("project.db")
>>> result = engine.fuse(
...     comparison_result,       # GraphComparisonResult from M6
...     name="Fused_n-C7H16",
...     description="Union of Curran2000 + JetSurF2",
... )
>>> print(result.fused_mechanism_id, result.n_verbatim, result.n_copy, result.n_fused)
>>> engine.export_union_spec(result, "union_spec.json")
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rmip.analysis.parameter_estimator import ParameterEstimator, ArrheniusEstimate
from rmip.types import (
    GraphComparisonResult,
    ReactionClass,
    SpeciesClass,
    FusionTreatment,
    ReactionClassification,
    SpeciesClassification,
)

log = logging.getLogger("rmip.analysis.fusion")

R_GAS: float = 8.314_462_618  # J mol⁻¹ K⁻¹


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionRecord:
    """Outcome of fusing a single reaction."""
    canonical_key:   str
    formula_string:  str
    treatment:       FusionTreatment
    source_mech_ids: List[int]

    # Chosen parameters (always in SI)
    A_si:   float
    n:      float
    Ea_si:  float
    kinetics_type:      str
    kinetics_data_json: Optional[str]  # raw JSON for PLOG/Troe etc. (VERBATIM/COPY only)

    # For FUSE reactions: ParameterEstimator bounds (→ MSO search bounds)
    A_min_si:  Optional[float] = None
    A_max_si:  Optional[float] = None
    n_min:     Optional[float] = None
    n_max:     Optional[float] = None
    Ea_min_si: Optional[float] = None
    Ea_max_si: Optional[float] = None

    # Diagnostic
    uncertainty_factor: Optional[float] = None   # A_max / A_min

    def to_dict(self) -> dict:
        return {
            "canonical_key":       self.canonical_key,
            "formula_string":      self.formula_string,
            "treatment":           self.treatment.value,
            "source_mech_ids":     self.source_mech_ids,
            "A_si":                self.A_si,
            "n":                   self.n,
            "Ea_si":               self.Ea_si,
            "kinetics_type":       self.kinetics_type,
            "A_min_si":            self.A_min_si,
            "A_max_si":            self.A_max_si,
            "n_min":               self.n_min,
            "n_max":               self.n_max,
            "Ea_min_si":           self.Ea_min_si,
            "Ea_max_si":           self.Ea_max_si,
            "uncertainty_factor":  self.uncertainty_factor,
        }


@dataclass
class SpeciesFusionRecord:
    """Outcome of fusing a single species' thermo data."""
    inchi_key:        str
    canonical_smiles: str
    name:             str           # name chosen for the fused mechanism
    source_mech_ids:  List[int]

    # Consensus thermo (arithmetic mean per coefficient)
    tmin_K:   float
    tmid_K:   float
    tmax_K:   float
    coeffs_low:  List[float]    # 7 floats
    coeffs_high: List[float]    # 7 floats

    thermo_consistent: bool     # False → Tmid discontinuity > 0.5%

    def to_dict(self) -> dict:
        return {
            "inchi_key":         self.inchi_key,
            "canonical_smiles":  self.canonical_smiles,
            "name":              self.name,
            "source_mech_ids":   self.source_mech_ids,
            "tmin_K":            self.tmin_K,
            "tmid_K":            self.tmid_K,
            "tmax_K":            self.tmax_K,
            "coeffs_low":        self.coeffs_low,
            "coeffs_high":       self.coeffs_high,
            "thermo_consistent": self.thermo_consistent,
        }


@dataclass
class FusionResult:
    """Complete output of FusionEngine.fuse()."""
    fused_mechanism_id: int
    source_mechanism_ids: List[int]
    name: str

    reaction_records: List[FusionRecord]
    species_records:  List[SpeciesFusionRecord]

    n_verbatim: int
    n_copy:     int
    n_fused:    int

    n_species_universal: int
    n_species_partial:   int
    n_species_unique:    int

    # Reactions with high uncertainty (UF > threshold) — should be prioritised in MSO
    high_uncertainty_reactions: List[str]   # canonical_keys

    provenance: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"FusionResult: '{self.name}'",
            f"  Fused mechanism DB id : {self.fused_mechanism_id}",
            f"  Source mechanisms     : {self.source_mechanism_ids}",
            f"  Reactions             : {len(self.reaction_records)} total",
            f"    VERBATIM  : {self.n_verbatim}",
            f"    COPY      : {self.n_copy}",
            f"    FUSE      : {self.n_fused}",
            f"  Species               : {len(self.species_records)} total",
            f"    UNIVERSAL : {self.n_species_universal}",
            f"    PARTIAL   : {self.n_species_partial}",
            f"    UNIQUE    : {self.n_species_unique}",
            f"  High-UF reactions     : {len(self.high_uncertainty_reactions)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "fused_mechanism_id":      self.fused_mechanism_id,
            "source_mechanism_ids":    self.source_mechanism_ids,
            "name":                    self.name,
            "n_verbatim":              self.n_verbatim,
            "n_copy":                  self.n_copy,
            "n_fused":                 self.n_fused,
            "n_species_universal":     self.n_species_universal,
            "n_species_partial":       self.n_species_partial,
            "n_species_unique":        self.n_species_unique,
            "high_uncertainty_reactions": self.high_uncertainty_reactions,
            "provenance":              self.provenance,
            "reaction_records":        [r.to_dict() for r in self.reaction_records],
            "species_records":         [s.to_dict() for s in self.species_records],
        }


# ─────────────────────────────────────────────────────────────────────────────
# FusionEngine
# ─────────────────────────────────────────────────────────────────────────────

class FusionEngine:
    """
    M7 — Mechanism Fusion Engine.

    Parameters
    ----------
    db_path : str | Path
        Path to the RMIP SQLite database.
    uf_warn_threshold : float
        Uncertainty factor (A_max/A_min) above which a reaction is flagged as
        high-uncertainty and logged.  Default: 10.0.

    Notes
    -----
    The FusionEngine does NOT call the MSO tool directly.  It assembles the
    fused mechanism in the DB, generates MSO-ready bounds for FUSE reactions,
    and writes them to ``fusion_params.json``.  MSOInterface (rmip/optimization/)
    reads that file and calls KP's OptimizationTool_v4.py as a subprocess.
    """

    # Default kinetics tolerances for SHARED_IDENTICAL classification.
    # These should match the values used in M6 GraphComparator.
    DEFAULT_REL_TOL_A  = 0.01   # 1% on A
    DEFAULT_ABS_TOL_N  = 0.01   # absolute on n
    DEFAULT_REL_TOL_EA = 0.01   # 1% on Ea_si

    def __init__(
        self,
        db_path: str | Path,
        uf_warn_threshold: float = 10.0,
    ) -> None:
        self._db_path = str(db_path)
        self._uf_warn = uf_warn_threshold
        self._estimator = ParameterEstimator()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        comparison_result: GraphComparisonResult,
        name: str,
        description: str = "",
    ) -> FusionResult:
        """
        Build a fused mechanism from the GraphComparisonResult.

        Returns a FusionResult containing the new mechanism_id and all records.
        The fused mechanism is written to the DB.
        """
        mech_ids = comparison_result.mechanism_ids
        log.info("FusionEngine.fuse(): merging mechanisms %s → '%s'", mech_ids, name)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # ── 1. Load raw DB rows for all source mechanisms ─────────────
            source_rows = self._load_source_rows(conn, mech_ids)

            # ── 2. Fuse species ───────────────────────────────────────────
            species_records = self._fuse_species(
                conn, comparison_result, source_rows
            )

            # ── 3. Fuse reactions ─────────────────────────────────────────
            reaction_records = self._fuse_reactions(
                conn, comparison_result, source_rows
            )

            n_verbatim = sum(1 for r in reaction_records if r.treatment == FusionTreatment.VERBATIM)
            n_copy     = sum(1 for r in reaction_records if r.treatment == FusionTreatment.COPY)
            n_fused    = sum(1 for r in reaction_records if r.treatment == FusionTreatment.FUSE)

            high_uf = [
                r.canonical_key for r in reaction_records
                if r.uncertainty_factor is not None and r.uncertainty_factor > self._uf_warn
            ]
            if high_uf:
                log.warning(
                    "%d reactions have uncertainty factor > %.1f: %s",
                    len(high_uf), self._uf_warn, high_uf[:5],
                )

            # ── 4. Insert fused mechanism row ──────────────────────────────
            provenance = {
                "source_mechanism_ids": mech_ids,
                "comparison_session_id": comparison_result.session_id,
                "n_verbatim": n_verbatim,
                "n_copy": n_copy,
                "n_fused": n_fused,
                "n_species": len(species_records),
                "uf_warn_threshold": self._uf_warn,
                "fused_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_json = json.dumps(provenance)

            cur = conn.execute(
                """
                INSERT INTO mechanisms
                    (name, format, filepath, description,
                     n_species, n_reactions, loaded_at, metadata_json)
                VALUES (?, 'fused', NULL, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    description,
                    len(species_records),
                    len(reaction_records),
                    datetime.now(timezone.utc).isoformat(),
                    meta_json,
                ),
            )
            fused_mech_id = cur.lastrowid
            log.info("Created fused mechanism DB id=%d", fused_mech_id)

            # ── 5. Insert species rows ─────────────────────────────────────
            self._insert_species(conn, fused_mech_id, species_records,
                                 comparison_result, source_rows)

            # ── 6. Insert reaction rows ────────────────────────────────────
            self._insert_reactions(conn, fused_mech_id, reaction_records,
                                   comparison_result, source_rows)

            conn.commit()

        n_universal = sum(
            1 for s in comparison_result.species_classifications
            if s.classification == SpeciesClass.UNIVERSAL
        )
        n_partial = sum(
            1 for s in comparison_result.species_classifications
            if s.classification == SpeciesClass.PARTIAL
        )
        n_unique = sum(
            1 for s in comparison_result.species_classifications
            if s.classification == SpeciesClass.UNIQUE
        )

        result = FusionResult(
            fused_mechanism_id=fused_mech_id,
            source_mechanism_ids=mech_ids,
            name=name,
            reaction_records=reaction_records,
            species_records=species_records,
            n_verbatim=n_verbatim,
            n_copy=n_copy,
            n_fused=n_fused,
            n_species_universal=n_universal,
            n_species_partial=n_partial,
            n_species_unique=n_unique,
            high_uncertainty_reactions=high_uf,
            provenance=provenance,
        )
        log.info("Fusion complete.\n%s", result.summary())
        return result

    def export_union_spec(self, result: FusionResult, path: str) -> None:
        """
        Write the full FusionResult to JSON so MSOInterface can consume it
        without touching the DB.
        """
        with open(path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        log.info("Union spec written to %s", path)

    def export_mso_bounds(self, result: FusionResult, path: str) -> None:
        """
        Write a compact JSON of FUSE-reaction bounds ready for MSOInterface.

        Schema per entry:
        {
          "canonical_key":  str,
          "formula_string": str,
          "params_mean":    [A_si, n, Ea_si],
          "params_lb":      [A_min, n_min, Ea_min],
          "params_ub":      [A_max, n_max, Ea_max],
          "param_names":    ["A_si", "n", "Ea_si"],
          "units":          ["m^3/mol/s (SI)", "1", "J/mol"],
        }
        """
        blocks = []
        for rec in result.reaction_records:
            if rec.treatment != FusionTreatment.FUSE:
                continue
            blocks.append({
                "canonical_key":   rec.canonical_key,
                "formula_string":  rec.formula_string,
                "source_mech_ids": rec.source_mech_ids,
                "params_mean":     [rec.A_si, rec.n, rec.Ea_si],
                "params_lb":       [rec.A_min_si, rec.n_min, rec.Ea_min_si],
                "params_ub":       [rec.A_max_si, rec.n_max, rec.Ea_max_si],
                "param_names":     ["A_si", "n", "Ea_si"],
                "units":           ["m^3/mol/s", "1", "J/mol"],
            })
        out = {
            "fused_mechanism_id": result.fused_mechanism_id,
            "source_mechanism_ids": result.source_mechanism_ids,
            "n_fuse_reactions": len(blocks),
            "fuse_blocks": blocks,
        }
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        log.info("MSO bounds written to %s  (%d FUSE reactions)", path, len(blocks))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers — data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_source_rows(
        self,
        conn: sqlite3.Connection,
        mech_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Load raw DB rows for species, reactions, and stoichiometry for all
        source mechanisms.  Returns {mech_id: {"species": [...], "reactions": [...]}}.
        """
        placeholders = ",".join("?" * len(mech_ids))
        rows: Dict[int, Dict[str, Any]] = {mid: {"species": [], "reactions": []} for mid in mech_ids}

        sp_rows = conn.execute(
            f"SELECT * FROM species WHERE mechanism_id IN ({placeholders})",
            mech_ids,
        ).fetchall()
        for row in sp_rows:
            rows[row["mechanism_id"]]["species"].append(dict(row))

        rxn_rows = conn.execute(
            f"SELECT * FROM reactions WHERE mechanism_id IN ({placeholders})",
            mech_ids,
        ).fetchall()
        for row in rxn_rows:
            rows[row["mechanism_id"]]["reactions"].append(dict(row))

        return rows

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers — species fusion
    # ─────────────────────────────────────────────────────────────────────────

    def _fuse_species(
        self,
        conn: sqlite3.Connection,
        cmp: GraphComparisonResult,
        source_rows: Dict[int, Dict[str, Any]],
    ) -> List[SpeciesFusionRecord]:
        """Compute consensus thermo for every species in the union."""
        # Build lookup: inchi_key → {mech_id: species_row}
        sp_lookup: Dict[str, Dict[int, dict]] = {}
        for mid, data in source_rows.items():
            for sp in data["species"]:
                ik = sp.get("inchi_key") or ""
                if not ik:
                    continue
                sp_lookup.setdefault(ik, {})[mid] = sp

        records: List[SpeciesFusionRecord] = []
        for sc in cmp.species_classifications:
            ik = sc.inchi_key
            source_map = sp_lookup.get(ik, {})
            if not source_map:
                log.warning("No DB rows found for inchi_key=%s", ik)
                continue

            rec = self._compute_species_consensus(ik, sc.canonical_smiles, source_map)
            records.append(rec)

        return records

    def _compute_species_consensus(
        self,
        inchi_key: str,
        canonical_smiles: str,
        source_map: Dict[int, dict],
    ) -> SpeciesFusionRecord:
        """Arithmetic mean of NASA-7 coefficients across all sources."""
        source_ids = list(source_map.keys())

        # Collect thermo arrays
        all_low:  List[List[float]] = []
        all_high: List[List[float]] = []
        tmids:    List[float]       = []
        tmins:    List[float]       = []
        tmaxs:    List[float]       = []
        names:    List[str]         = []

        for sp in source_map.values():
            names.append(sp.get("name", ""))
            try:
                low  = json.loads(sp.get("coeffs_low_json")  or "null")
                high = json.loads(sp.get("coeffs_high_json") or "null")
                if low and high:
                    all_low.append(low)
                    all_high.append(high)
                tmids.append(float(sp.get("tmid_K") or 1000.0))
                tmins.append(float(sp.get("tmin_K") or 200.0))
                tmaxs.append(float(sp.get("tmax_K") or 5000.0))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.debug("Thermo parse error for %s: %s", inchi_key, exc)

        # Choose preferred Tmid (most common)
        tmid = self._modal_value(tmids) if tmids else 1000.0
        tmin = min(tmins) if tmins else 200.0
        tmax = max(tmaxs) if tmaxs else 5000.0

        # Mean coefficients
        consistent = True
        if all_low and all_high:
            n = len(all_low)
            mean_low  = [sum(c[i] for c in all_low)  / n for i in range(7)]
            mean_high = [sum(c[i] for c in all_high) / n for i in range(7)]

            # Continuity check at Tmid
            Cp_from_low  = self._nasa7_Cp(mean_low,  tmid)
            Cp_from_high = self._nasa7_Cp(mean_high, tmid)
            if abs(Cp_from_low) > 1e-10:
                discont = abs(Cp_from_low - Cp_from_high) / abs(Cp_from_low)
                if discont > 0.005:  # 0.5%
                    consistent = False
                    log.warning(
                        "Species %s: mean thermo discontinuity %.2f%% at Tmid=%.0f K",
                        inchi_key, discont * 100, tmid,
                    )
        else:
            mean_low  = [0.0] * 7
            mean_high = [0.0] * 7
            consistent = False

        # Name: prefer the most common non-empty name
        name = self._modal_value([n for n in names if n]) or inchi_key

        return SpeciesFusionRecord(
            inchi_key=inchi_key,
            canonical_smiles=canonical_smiles,
            name=name,
            source_mech_ids=source_ids,
            tmin_K=tmin,
            tmid_K=tmid,
            tmax_K=tmax,
            coeffs_low=mean_low,
            coeffs_high=mean_high,
            thermo_consistent=consistent,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers — reaction fusion
    # ─────────────────────────────────────────────────────────────────────────

    def _fuse_reactions(
        self,
        conn: sqlite3.Connection,
        cmp: GraphComparisonResult,
        source_rows: Dict[int, Dict[str, Any]],
    ) -> List[FusionRecord]:
        """Build FusionRecords for all reactions in the union."""
        # Build lookup: canonical_key → {mech_id: reaction_row}
        rxn_lookup: Dict[str, Dict[int, dict]] = {}
        for mid, data in source_rows.items():
            for rxn in data["reactions"]:
                ck = rxn.get("canonical_key") or rxn.get("formula_string", "")
                rxn_lookup.setdefault(ck, {})[mid] = rxn

        records: List[FusionRecord] = []
        for rc in cmp.reaction_classifications:
            ck = rc.canonical_key
            rxn_map = rxn_lookup.get(ck, {})
            if not rxn_map:
                log.warning("No DB rows for canonical_key=%s", ck)
                continue

            treatment = rc.fusion_treatment()
            rec = self._build_reaction_record(rc, rxn_map, treatment)
            records.append(rec)

        return records

    def _build_reaction_record(
        self,
        rc: ReactionClassification,
        rxn_map: Dict[int, dict],
        treatment: FusionTreatment,
    ) -> FusionRecord:
        """Apply VERBATIM / COPY / FUSE logic to produce a FusionRecord."""
        source_ids = list(rxn_map.keys())

        if treatment in (FusionTreatment.VERBATIM, FusionTreatment.COPY):
            # Pick kinetics from the first (lowest) mechanism ID
            first_row = rxn_map[min(rxn_map.keys())]
            return FusionRecord(
                canonical_key=rc.canonical_key,
                formula_string=rc.formula_string,
                treatment=treatment,
                source_mech_ids=source_ids,
                A_si=first_row.get("A_si") or 0.0,
                n=first_row.get("n") or 0.0,
                Ea_si=first_row.get("Ea_si") or 0.0,
                kinetics_type=first_row.get("kinetics_type") or "arrhenius",
                kinetics_data_json=first_row.get("kinetics_data_json"),
            )

        # FUSE treatment — use ParameterEstimator
        A_vals  = [rxn.get("A_si")  for rxn in rxn_map.values() if rxn.get("A_si")  is not None]
        n_vals  = [rxn.get("n")     for rxn in rxn_map.values() if rxn.get("n")     is not None]
        Ea_vals = [rxn.get("Ea_si") for rxn in rxn_map.values() if rxn.get("Ea_si") is not None]

        if not A_vals:
            # Non-Arrhenius (PLOG/Troe etc.) — VERBATIM from first source
            first_row = rxn_map[min(rxn_map.keys())]
            log.warning(
                "FUSE reaction %s has no A_si values (likely PLOG/Troe)."
                " Falling back to VERBATIM from mech %d.",
                rc.canonical_key, min(rxn_map.keys()),
            )
            return FusionRecord(
                canonical_key=rc.canonical_key,
                formula_string=rc.formula_string,
                treatment=FusionTreatment.VERBATIM,
                source_mech_ids=source_ids,
                A_si=first_row.get("A_si") or 0.0,
                n=first_row.get("n") or 0.0,
                Ea_si=first_row.get("Ea_si") or 0.0,
                kinetics_type=first_row.get("kinetics_type") or "unknown",
                kinetics_data_json=first_row.get("kinetics_data_json"),
            )

        est: ArrheniusEstimate = self._estimator.estimate_arrhenius(
            A_vals=A_vals, n_vals=n_vals, Ea_vals=Ea_vals
        )

        if est.uncertainty_factor > self._uf_warn:
            log.warning(
                "Reaction '%s': uncertainty factor = %.1f× (> %.1f×)",
                rc.formula_string, est.uncertainty_factor, self._uf_warn,
            )

        # Pick kinetics_type from majority vote
        ktypes = [rxn.get("kinetics_type", "arrhenius") for rxn in rxn_map.values()]
        ktype = self._modal_value(ktypes) or "arrhenius"

        return FusionRecord(
            canonical_key=rc.canonical_key,
            formula_string=rc.formula_string,
            treatment=FusionTreatment.FUSE,
            source_mech_ids=source_ids,
            A_si=est.A_mean_si,
            n=est.n_mean,
            Ea_si=est.Ea_mean_si,
            kinetics_type=ktype,
            kinetics_data_json=None,    # mean Arrhenius only; no PLOG fusing
            A_min_si=est.A_min_si,
            A_max_si=est.A_max_si,
            n_min=est.n_min,
            n_max=est.n_max,
            Ea_min_si=est.Ea_min_si,
            Ea_max_si=est.Ea_max_si,
            uncertainty_factor=est.uncertainty_factor,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers — DB insertion
    # ─────────────────────────────────────────────────────────────────────────

    def _insert_species(
        self,
        conn: sqlite3.Connection,
        fused_mech_id: int,
        species_records: List[SpeciesFusionRecord],
        cmp: GraphComparisonResult,
        source_rows: Dict[int, Dict[str, Any]],
    ) -> None:
        """Write SpeciesFusionRecords into the ``species`` table."""
        # Build helper lookup: inchi_key → representative source row
        sp_lookup: Dict[str, dict] = {}
        for mid, data in source_rows.items():
            for sp in data["species"]:
                ik = sp.get("inchi_key") or ""
                if ik and ik not in sp_lookup:
                    sp_lookup[ik] = sp  # first-seen wins for metadata

        for sfr in species_records:
            src_row = sp_lookup.get(sfr.inchi_key, {})
            conn.execute(
                """
                INSERT INTO species (
                    mechanism_id, name, canonical_smiles, source_smiles,
                    inchi, inchi_key, formula, mw_kg_mol, charge,
                    glossary_matched, thermo_type,
                    tmin_K, tmid_K, tmax_K,
                    coeffs_low_json, coeffs_high_json,
                    h298_kJ_mol, s298_J_molK, thermo_comment
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fused_mech_id,
                    sfr.name,
                    sfr.canonical_smiles,
                    src_row.get("source_smiles"),
                    src_row.get("inchi"),
                    sfr.inchi_key,
                    src_row.get("formula"),
                    src_row.get("mw_kg_mol"),
                    src_row.get("charge", 0),
                    src_row.get("glossary_matched", 0),
                    "nasa7",
                    sfr.tmin_K,
                    sfr.tmid_K,
                    sfr.tmax_K,
                    json.dumps(sfr.coeffs_low),
                    json.dumps(sfr.coeffs_high),
                    src_row.get("h298_kJ_mol"),
                    src_row.get("s298_J_molK"),
                    f"Fused from {sfr.source_mech_ids}",
                ),
            )

    def _insert_reactions(
        self,
        conn: sqlite3.Connection,
        fused_mech_id: int,
        reaction_records: List[FusionRecord],
        cmp: GraphComparisonResult,
        source_rows: Dict[int, Dict[str, Any]],
    ) -> None:
        """Write FusionRecords into the ``reactions`` table."""
        # Build lookup: (mech_id, canonical_key) → reaction row
        rxn_lookup: Dict[Tuple[int, str], dict] = {}
        for mid, data in source_rows.items():
            for rxn in data["reactions"]:
                ck = rxn.get("canonical_key") or ""
                rxn_lookup[(mid, ck)] = rxn

        # Collect fused species ids for stoichiometry mapping
        fused_sp = {
            row["inchi_key"]: row["id"]
            for row in conn.execute(
                "SELECT id, inchi_key FROM species WHERE mechanism_id=?",
                (fused_mech_id,),
            ).fetchall()
        }

        for fr in reaction_records:
            # Pick source row for metadata not covered by fusion
            src_mid = min(fr.source_mech_ids)
            src_row = rxn_lookup.get((src_mid, fr.canonical_key), {})

            # Check if fusion_class column exists (added by Phase-3 migration)
            has_fusion_cols = self._has_fusion_cols(conn)

            if has_fusion_cols:
                cur = conn.execute(
                    """
                    INSERT INTO reactions (
                        mechanism_id, formula_string, canonical_key,
                        reversible, family, duplicate,
                        kinetics_type, pressure_dependent, has_third_body,
                        A_si, A_orig, A_units,
                        n,
                        Ea_si, Ea_orig, Ea_units,
                        kinetics_data_json, rmg_comment,
                        fusion_class, source_mechanism_ids_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        fused_mech_id,
                        fr.formula_string,
                        fr.canonical_key,
                        src_row.get("reversible", 1),
                        src_row.get("family"),
                        src_row.get("duplicate", 0),
                        fr.kinetics_type,
                        src_row.get("pressure_dependent", 0),
                        src_row.get("has_third_body", 0),
                        fr.A_si,
                        fr.A_si,
                        "SI",
                        fr.n,
                        fr.Ea_si,
                        fr.Ea_si,
                        "J/mol",
                        fr.kinetics_data_json,
                        f"Fused ({fr.treatment.value}) from {fr.source_mech_ids}",
                        fr.treatment.value,
                        json.dumps(fr.source_mech_ids),
                    ),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO reactions (
                        mechanism_id, formula_string, canonical_key,
                        reversible, family, duplicate,
                        kinetics_type, pressure_dependent, has_third_body,
                        A_si, A_orig, A_units,
                        n,
                        Ea_si, Ea_orig, Ea_units,
                        kinetics_data_json, rmg_comment
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        fused_mech_id,
                        fr.formula_string,
                        fr.canonical_key,
                        src_row.get("reversible", 1),
                        src_row.get("family"),
                        src_row.get("duplicate", 0),
                        fr.kinetics_type,
                        src_row.get("pressure_dependent", 0),
                        src_row.get("has_third_body", 0),
                        fr.A_si,
                        fr.A_si,
                        "SI",
                        fr.n,
                        fr.Ea_si,
                        fr.Ea_si,
                        "J/mol",
                        fr.kinetics_data_json,
                        f"Fused ({fr.treatment.value}) from {fr.source_mech_ids}",
                    ),
                )
            rxn_id = cur.lastrowid

            # ── Stoichiometry: copy from first-available source mechanism ──
            self._copy_stoichiometry(
                conn, rxn_id, fr.canonical_key,
                src_mid, rxn_lookup, fused_sp, source_rows,
            )

    def _copy_stoichiometry(
        self,
        conn: sqlite3.Connection,
        new_rxn_id: int,
        canonical_key: str,
        src_mid: int,
        rxn_lookup: Dict[Tuple[int, str], dict],
        fused_sp: Dict[str, int],
        source_rows: Dict[int, Dict[str, Any]],
    ) -> None:
        """Copy stoichiometry rows from source mechanism into the fused reaction."""
        src_row = rxn_lookup.get((src_mid, canonical_key))
        if not src_row:
            return
        src_rxn_id = src_row["id"]

        stoich_rows = conn.execute(
            """
            SELECT s.stoich_coeff, s.role, sp.inchi_key
              FROM stoichiometry s
              JOIN species sp ON sp.id = s.species_id
             WHERE s.reaction_id = ?
            """,
            (src_rxn_id,),
        ).fetchall()

        for srow in stoich_rows:
            ik = srow["inchi_key"]
            fused_sp_id = fused_sp.get(ik)
            if fused_sp_id is None:
                log.debug("Stoich: species inchi_key=%s not found in fused mechanism", ik)
                continue
            conn.execute(
                "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) VALUES (?,?,?,?)",
                (new_rxn_id, fused_sp_id, srow["stoich_coeff"], srow["role"]),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _modal_value(values: list) -> Any:
        """Return the most-common element in ``values``, or None if empty."""
        if not values:
            return None
        return max(set(values), key=values.count)

    @staticmethod
    def _nasa7_Cp(coeffs: List[float], T: float) -> float:
        """Evaluate Cp/R from a 7-coefficient NASA polynomial at temperature T."""
        a = coeffs
        return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4

    @staticmethod
    def _has_fusion_cols(conn: sqlite3.Connection) -> bool:
        """Check if Phase-3 fusion columns are present in the reactions table."""
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reactions)").fetchall()
        }
        return "fusion_class" in cols and "source_mechanism_ids_json" in cols
