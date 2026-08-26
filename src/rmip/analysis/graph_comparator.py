"""
rmip/analysis/graph_comparator.py
==================================
Multi-Mechanism Graph Comparison Engine

This is the algorithmic heart of Phase 3.  Given N parsed mechanisms in the
DB, it produces a complete classification of every species and reaction across
all mechanisms and constructs the `UnionGraphSpec` that drives the FusionEngine
(M7, Phase 4).

Classification scheme
---------------------
**Species**
  UNIVERSAL  — present in ALL N mechanisms (by InChI key)
  PARTIAL    — present in ≥2 but not all mechanisms
  UNIQUE     — present in exactly one mechanism

**Reactions**  (matched by canonical_key — design decision D2)
  SHARED_IDENTICAL  — same canonical_key in ≥2 mechanisms AND kinetic
                       parameters agree within tolerances
  SHARED_DIVERGENT  — same canonical_key in ≥2 mechanisms BUT kinetic
                       parameters differ beyond tolerance
  UNIQUE            — canonical_key present in exactly one mechanism

Kinetic tolerance criteria (configurable, defaults empirically chosen)
  |ΔA_rel|  ≤ tol_A   (relative, fraction of max)
  |Δn|      ≤ tol_n   (absolute)
  |ΔEa|     ≤ tol_Ea  [J/mol] (absolute)
  A reaction is IDENTICAL only if ALL three criteria hold simultaneously.

Mean kinetics for SHARED_DIVERGENT
------------------------------------
  A   : geometric mean   →  exp( mean(log(Aᵢ)) )   (log-space arithmetic mean)
  n   : arithmetic mean
  Ea  : arithmetic mean  [J/mol]
  Uncertainty bounds: (min, max) for each parameter across all mechanisms.

This is exactly the parameter mean + min/max uncertainty specification
described in KP's Phase-3 workflow, ready for M9 DoE / MSO optimisation.

Thermo data for shared species with divergent NASA-7 polynomials
----------------------------------------------------------------
  Coefficient-wise arithmetic mean for each of the 14 NASA-7 coefficients.
  Per-coefficient (min, max) stored in ThermoSet for M7 to use.

Usage
-----
>>> from rmip.analysis.graph_comparator import GraphComparator
>>> cmp = GraphComparator("project.db")
>>> result = cmp.compare([1, 2, 3])
>>> print(result.summary_dict())
>>> spec = cmp.build_union_spec(result)
>>> spec.save_json("union_graph_spec.json")
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from rmip.database.schema_phase3 import (
    insert_comparison_session,
    insert_reaction_classification,
    insert_species_classification,
)
from rmip.types import (
    FusionTreatment,
    GraphComparisonResult,
    ReactionClass,
    ReactionClassification,
    ReactionKineticsSet,
    SpeciesClass,
    SpeciesClassification,
    ThermoSet,
    UnionGraphSpec,
)

logger = logging.getLogger(__name__)


class GraphComparator:
    """
    Multi-mechanism reaction-graph comparison engine.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    tol_A : float
        Relative tolerance for A comparison  [fraction of max A].
    tol_n : float
        Absolute tolerance for n comparison.
    tol_Ea : float
        Absolute tolerance for Ea [J/mol].
    """

    def __init__(
        self,
        db_path:  str,
        tol_A:    float = 0.01,    # 1% relative
        tol_n:    float = 0.001,   # absolute
        tol_Ea:   float = 100.0,   # J/mol
    ) -> None:
        self._db_path = db_path
        self._tol_A   = tol_A
        self._tol_n   = tol_n
        self._tol_Ea  = tol_Ea

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compare(
        self,
        mechanism_ids: List[int],
        notes:         Optional[str] = None,
    ) -> GraphComparisonResult:
        """
        Compare N mechanisms and classify all species and reactions.

        Parameters
        ----------
        mechanism_ids : list[int]
            DB ids of the mechanisms to compare (≥ 2).
        notes : str, optional
            Annotation stored with the comparison session in the DB.

        Returns
        -------
        GraphComparisonResult
        """
        if len(mechanism_ids) < 2:
            raise ValueError("At least two mechanism IDs required for comparison.")

        conn = self._connect()
        try:
            # 1. Register the session
            session_id = insert_comparison_session(
                conn,
                mechanism_ids = mechanism_ids,
                kinetic_tol_A  = self._tol_A,
                kinetic_tol_n  = self._tol_n,
                kinetic_tol_Ea = self._tol_Ea,
                notes          = notes,
            )

            # 2. Load raw data from DB
            mech_species   = self._load_all_species(conn, mechanism_ids)
            mech_reactions = self._load_all_reactions(conn, mechanism_ids)

            # 3. Classify species
            species_cls = self._classify_species(
                conn, session_id, mechanism_ids, mech_species
            )

            # 4. Classify reactions
            reaction_cls = self._classify_reactions(
                conn, session_id, mechanism_ids, mech_reactions
            )

            conn.commit()

        finally:
            conn.close()

        result = GraphComparisonResult(
            session_id               = session_id,
            mechanism_ids            = mechanism_ids,
            species_classifications  = species_cls,
            reaction_classifications = reaction_cls,
        )

        logger.info(
            "Comparison session %d: %d mechanisms → "
            "%d universal / %d partial / %d unique species | "
            "%d identical / %d divergent / %d unique reactions",
            session_id, len(mechanism_ids),
            len(result.universal_species),
            len(result.partial_species),
            len(result.unique_species),
            len(result.shared_identical_reactions),
            len(result.shared_divergent_reactions),
            len(result.unique_reactions),
        )
        return result

    # ------------------------------------------------------------------
    # Union graph specification builder
    # ------------------------------------------------------------------

    def build_union_spec(
        self,
        result: GraphComparisonResult,
    ) -> UnionGraphSpec:
        """
        Build the `UnionGraphSpec` that the FusionEngine (M7) will consume
        to construct the merged mechanism.

        The spec contains:
          - species_map       : all unique species (by InChI key)
          - verbatim_reactions: unique → copy from single source unchanged
          - copy_reactions    : shared_identical → copy as-is
          - fuse_reactions    : shared_divergent → use mean ± uncertainty
          - fuse_params       : computed mean / (min, max) for fuse reactions
        """
        spec = UnionGraphSpec(
            mechanism_ids         = result.mechanism_ids,
            comparison_session_id = result.session_id,
        )

        # Species map
        for sc in result.species_classifications:
            spec.species_map[sc.inchi_key] = sc

        # Reaction buckets (M9: key on (canonical_key, occurrence_index) so
        # CHEMKIN DUPLICATE reactions don't collide and silently drop one).
        for rc in result.unique_reactions:
            spec.verbatim_reactions[(rc.canonical_key, rc.occurrence_index)] = rc

        for rc in result.shared_identical_reactions:
            spec.copy_reactions[(rc.canonical_key, rc.occurrence_index)] = rc

        for rc in result.shared_divergent_reactions:
            spec.fuse_reactions[(rc.canonical_key, rc.occurrence_index)] = rc

            # Build per-reaction fuse params from stored mean/min/max
            spec.fuse_params[(rc.canonical_key, rc.occurrence_index)] = {
                "mean_A_si":  rc.mean_A_si,
                "mean_n":     rc.mean_n,
                "mean_Ea_si": rc.mean_Ea_si,
                "unc_A_si":   (rc.min_A_si, rc.max_A_si),
                "unc_n":      (rc.min_n,    rc.max_n),
                "unc_Ea_si":  (rc.min_Ea_si, rc.max_Ea_si),
            }

        logger.info(
            "UnionGraphSpec built: %d species | %d verbatim + %d copy + %d fuse reactions",
            len(spec.species_map),
            len(spec.verbatim_reactions),
            len(spec.copy_reactions),
            len(spec.fuse_reactions),
        )
        return spec

    # ------------------------------------------------------------------
    # Species classification
    # ------------------------------------------------------------------

    def _classify_species(
        self,
        conn:          sqlite3.Connection,
        session_id:    int,
        mechanism_ids: List[int],
        mech_species:  Dict[int, Dict[str, Dict]],
    ) -> List[SpeciesClassification]:
        """
        Group species by InChI key and assign UNIVERSAL / PARTIAL / UNIQUE.
        Also loads per-mechanism thermo for divergence reporting.
        """
        # inchi_key → {mech_id: row_dict}
        ik_to_mechs: Dict[str, Dict[int, Dict]] = defaultdict(dict)
        for mid, sp_dict in mech_species.items():
            for ik, row in sp_dict.items():
                ik_to_mechs[ik][mid] = row

        n_mechs = len(mechanism_ids)
        result: List[SpeciesClassification] = []

        for ik, mech_data in ik_to_mechs.items():
            present = sorted(mech_data.keys())
            n_pres  = len(present)

            if n_pres == n_mechs:
                cls = SpeciesClass.UNIVERSAL
            elif n_pres == 1:
                cls = SpeciesClass.UNIQUE
            else:
                cls = SpeciesClass.PARTIAL

            # Pick representative row for metadata
            rep = next(iter(mech_data.values()))

            thermo_set = None
            if cls in (SpeciesClass.UNIVERSAL, SpeciesClass.PARTIAL):
                thermo_data = {
                    mid: {
                        "coeffs_low":  self._parse_json(row.get("coeffs_low_json")),
                        "coeffs_high": self._parse_json(row.get("coeffs_high_json")),
                        "tmin_K":      row.get("tmin_K"),
                        "tmid_K":      row.get("tmid_K"),
                        "tmax_K":      row.get("tmax_K"),
                    }
                    for mid, row in mech_data.items()
                }
                thermo_set = ThermoSet(
                    inchi_key       = ik,
                    canonical_smiles = rep.get("canonical_smiles", ""),
                    mech_data       = thermo_data,
                )

            sc = SpeciesClassification(
                inchi_key          = ik,
                canonical_smiles   = rep.get("canonical_smiles", ""),
                formula            = rep.get("formula", ""),
                classification     = cls,
                present_in_mech_ids = present,
                thermo_set         = thermo_set,
            )
            result.append(sc)

            # Persist to DB
            insert_species_classification(
                conn,
                session_id     = session_id,
                inchi_key      = ik,
                canonical_smiles = rep.get("canonical_smiles"),
                formula         = rep.get("formula"),
                classification  = cls.value,
                present_in_mechs = present,
                thermo_data     = {str(k): v for k, v in thermo_set.mech_data.items()}
                                   if thermo_set else None,
            )

        return result

    # ------------------------------------------------------------------
    # Reaction classification
    # ------------------------------------------------------------------

    def _classify_reactions(
        self,
        conn:           sqlite3.Connection,
        session_id:     int,
        mechanism_ids:  List[int],
        mech_reactions: Dict[int, Dict[Tuple[str, int], Dict]],
    ) -> List[ReactionClassification]:
        """
        Group reactions by (canonical_key, occurrence_index) and assign
        classification (M9: occurrence_index disambiguates CHEMKIN
        DUPLICATE reactions -- see _load_all_reactions). Cross-mechanism
        pairing matches occurrence 0 in one mechanism to occurrence 0 in
        another, occurrence 1 to occurrence 1, etc.
        """
        # (canonical_key, occurrence_index) → {mech_id: row_dict}
        ck_to_mechs: Dict[Tuple[str, int], Dict[int, Dict]] = defaultdict(dict)
        for mid, rxn_dict in mech_reactions.items():
            for key, row in rxn_dict.items():
                ck_to_mechs[key][mid] = row

        n_mechs = len(mechanism_ids)
        result: List[ReactionClassification] = []

        for (ck, occ), mech_data in ck_to_mechs.items():
            present     = sorted(mech_data.keys())
            n_pres      = len(present)
            rep         = next(iter(mech_data.values()))
            formula_str = rep.get("formula_string", "")

            ks = ReactionKineticsSet(
                canonical_key  = ck,
                formula_string = formula_str,
                mech_data      = {
                    mid: {
                        "A_si":           row.get("A_si"),
                        "n":              row.get("n"),
                        "Ea_si":          row.get("Ea_si"),
                        "kinetics_type":  row.get("kinetics_type"),
                        "kinetics_data_json": row.get("kinetics_data_json"),
                    }
                    for mid, row in mech_data.items()
                }
            )

            if n_pres == 1:
                cls = ReactionClass.UNIQUE
                rc  = ReactionClassification(
                    canonical_key       = ck,
                    formula_string      = formula_str,
                    classification      = cls,
                    present_in_mech_ids = present,
                    occurrence_index    = occ,
                    kinetics_set        = ks,
                )
            else:
                # Compute aggregate kinetics
                agg = self._aggregate_kinetics(ks)
                identical = self._kinetics_are_identical(ks)
                cls = ReactionClass.SHARED_IDENTICAL if identical else ReactionClass.SHARED_DIVERGENT

                rc = ReactionClassification(
                    canonical_key       = ck,
                    formula_string      = formula_str,
                    classification      = cls,
                    present_in_mech_ids = present,
                    occurrence_index    = occ,
                    kinetics_set        = ks,
                    mean_A_si           = agg["mean_A"],
                    mean_n              = agg["mean_n"],
                    mean_Ea_si          = agg["mean_Ea"],
                    min_A_si            = agg["min_A"],
                    max_A_si            = agg["max_A"],
                    min_n               = agg["min_n"],
                    max_n               = agg["max_n"],
                    min_Ea_si           = agg["min_Ea"],
                    max_Ea_si           = agg["max_Ea"],
                )

            result.append(rc)

            # Persist to DB
            insert_reaction_classification(
                conn,
                session_id      = session_id,
                canonical_key   = ck,
                formula_string  = formula_str,
                classification  = cls.value,
                present_in_mechs = present,
                kinetics_data   = {str(k): v for k, v in ks.mech_data.items()},
                mean_A  = rc.mean_A_si,  mean_n = rc.mean_n,  mean_Ea = rc.mean_Ea_si,
                min_A   = rc.min_A_si,   max_A  = rc.max_A_si,
                min_n   = rc.min_n,      max_n  = rc.max_n,
                min_Ea  = rc.min_Ea_si,  max_Ea = rc.max_Ea_si,
            )

        return result

    # ------------------------------------------------------------------
    # Kinetics helpers
    # ------------------------------------------------------------------

    def _kinetics_are_identical(self, ks: ReactionKineticsSet) -> bool:
        """
        Return True if all A/n/Ea values agree within tolerances.
        Handles None values conservatively (→ not identical).

        M10 (review.md): this used to compare ONLY the single representative
        A/n/Ea triple (the high-pressure limit for Troe/Lindemann, the
        ~1atm entry for PLOG) -- two mechanisms could share that one number
        while having completely different LOW-pressure parameters, Troe
        alpha/T1/T2/T3, or other PLOG pressure points, and still get
        classified SHARED_IDENTICAL. Now also requires (a) every mechanism
        report the same kinetics_type for this reaction, and (b) for
        pressure-dependent types, the full kinetics_data_json payload
        (falloff/PLOG/Chebyshev table) to match within the same tolerances
        used for the representative A/n/Ea triple above.
        """
        A_vals  = ks.A_values
        n_vals  = ks.n_values
        Ea_vals = ks.Ea_values

        if not A_vals or len(A_vals) < 2:
            return False

        # Relative tolerance for A: (max-min)/max ≤ tol_A
        A_max = max(A_vals)
        if A_max == 0:
            return False
        A_rel_spread = (A_max - min(A_vals)) / A_max
        if A_rel_spread > self._tol_A:
            return False

        if n_vals:
            n_spread = max(n_vals) - min(n_vals)
            if n_spread > self._tol_n:
                return False

        if Ea_vals:
            Ea_spread = max(Ea_vals) - min(Ea_vals)
            if Ea_spread > self._tol_Ea:
                return False

        ktypes = {d.get("kinetics_type") for d in ks.mech_data.values()
                  if d.get("kinetics_type") is not None}
        if len(ktypes) > 1:
            return False

        ktype = next(iter(ktypes), None)
        if ktype in ("troe", "lindemann", "sri", "plog", "chebyshev"):
            payloads = []
            for d in ks.mech_data.values():
                raw = d.get("kinetics_data_json")
                if not raw:
                    return False  # pressure-dependent type must have aux data
                try:
                    payloads.append(json.loads(raw) if isinstance(raw, str) else raw)
                except (TypeError, ValueError):
                    return False
            if len(payloads) >= 2 and not self._kinetics_data_matches(payloads):
                return False

        return True

    def _kinetics_data_matches(self, payloads: List[Any]) -> bool:
        """Pairwise-compare parsed kinetics_data_json payloads (falloff/PLOG/
        Chebyshev tables) within this comparator's existing A/n/Ea
        tolerances. Numeric leaves compared with a relative tolerance
        (tol_A-scale) for magnitude-varying values, everything else
        (list lengths, non-numeric leaves) must match exactly -- a
        differing number of PLOG pressure points, for example, is never
        "identical" regardless of tolerance."""
        first = payloads[0]
        return all(self._values_match(first, other) for other in payloads[1:])

    def _values_match(self, a: Any, b: Any) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._values_match(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(self._values_match(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a is None or b is None:
                return a == b
            mag = max(abs(a), abs(b))
            if mag == 0:
                return True
            return abs(a - b) / mag <= max(self._tol_A, 1e-6)
        return a == b

    @staticmethod
    def _aggregate_kinetics(ks: ReactionKineticsSet) -> Dict[str, Optional[float]]:
        """
        Compute mean (geometric for A, arithmetic for n/Ea) and min/max bounds.
        """
        A_vals  = [v for v in ks.A_values  if v is not None and v > 0]
        n_vals  = [v for v in ks.n_values  if v is not None]
        Ea_vals = [v for v in ks.Ea_values if v is not None]

        # Geometric mean of A (mean in log-space)
        mean_A = math.exp(sum(math.log(v) for v in A_vals) / len(A_vals)) if A_vals else None
        min_A  = min(A_vals) if A_vals else None
        max_A  = max(A_vals) if A_vals else None

        mean_n = sum(n_vals) / len(n_vals) if n_vals else None
        min_n  = min(n_vals) if n_vals else None
        max_n  = max(n_vals) if n_vals else None

        mean_Ea = sum(Ea_vals) / len(Ea_vals) if Ea_vals else None
        min_Ea  = min(Ea_vals) if Ea_vals else None
        max_Ea  = max(Ea_vals) if Ea_vals else None

        return dict(
            mean_A=mean_A, min_A=min_A, max_A=max_A,
            mean_n=mean_n, min_n=min_n, max_n=max_n,
            mean_Ea=mean_Ea, min_Ea=min_Ea, max_Ea=max_Ea,
        )

    # ------------------------------------------------------------------
    # DB loading helpers
    # ------------------------------------------------------------------

    def _load_all_species(
        self,
        conn:          sqlite3.Connection,
        mechanism_ids: List[int],
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Load species from DB for all mechanisms.
        Returns: {mech_id: {inchi_key: row_dict}}

        B4 (review.md): species with no inchi_key (glossary didn't resolve
        them) used to be filtered out of the SQL query entirely, so they
        never appeared in any classification and FusionEngine never saw
        them -- fusing two identical FFCM1 copies (28.9% glossary hit rate)
        dropped to 11/38 species. Unresolved species now fall back to a
        per-mechanism name-based key, so they classify as UNIQUE (never
        cross-match another mechanism's real inchi_key, which is the correct
        conservative behaviour for an identifier we don't trust) instead of
        disappearing.
        """
        placeholders = ",".join("?" * len(mechanism_ids))
        rows = conn.execute(
            f"""SELECT mechanism_id, inchi_key, canonical_smiles, formula, name,
                       coeffs_low_json, coeffs_high_json,
                       tmin_K, tmid_K, tmax_K
                FROM species
                WHERE mechanism_id IN ({placeholders})""",
            mechanism_ids,
        ).fetchall()

        result: Dict[int, Dict[str, Dict]] = {mid: {} for mid in mechanism_ids}
        for row in rows:
            d   = dict(row)
            mid = d["mechanism_id"]
            ik  = d["inchi_key"] or f"\0unresolved\0{mid}\0{d.get('name')}"
            result[mid][ik] = d
        return result

    def _load_all_reactions(
        self,
        conn:          sqlite3.Connection,
        mechanism_ids: List[int],
    ) -> Dict[int, Dict[Tuple[str, int], Dict]]:
        """
        Load reactions from DB for all mechanisms.
        Returns: {mech_id: {(canonical_key, occurrence_index): row_dict}}
        Only Arrhenius-type reactions (with A_si / n / Ea_si populated) are
        included in kinetic comparison.  PLOG/Troe reactions are included but
        flagged via kinetics_type for special handling in M7.

        B4: same fix as _load_all_species -- reactions with no canonical_key
        (depends on species inchi_key resolution) used to be filtered out
        entirely (20/291 reactions survived fusing two identical FFCM1
        copies). They now fall back to a per-mechanism formula-string key,
        classifying as UNIQUE so FusionEngine passes them through VERBATIM
        instead of dropping them.

        M9 (review.md): CHEMKIN DUPLICATE reactions (two rows sharing one
        canonical_key -- e.g. two Arrhenius expressions summed for one
        reaction) used to collide in the `{canonical_key: row}` shape this
        used to return, silently dropping every duplicate but the last row
        read. Keyed by (canonical_key, occurrence_index) instead --
        occurrence_index counts repeats of the same canonical_key within
        one mechanism, in `id` (insertion/file) order.
        """
        placeholders = ",".join("?" * len(mechanism_ids))
        rows = conn.execute(
            f"""SELECT id, mechanism_id, canonical_key, formula_string,
                       kinetics_type, A_si, n, Ea_si, kinetics_data_json
                FROM reactions
                WHERE mechanism_id IN ({placeholders})
                ORDER BY mechanism_id, id""",
            mechanism_ids,
        ).fetchall()

        result: Dict[int, Dict[Tuple[str, int], Dict]] = {mid: {} for mid in mechanism_ids}
        occurrence_counts: Dict[Tuple[int, str], int] = {}
        for row in rows:
            d   = dict(row)
            mid = d["mechanism_id"]
            ck  = d["canonical_key"] or f"\0unresolved\0{mid}\0{d.get('formula_string')}"
            occ = occurrence_counts.get((mid, ck), 0)
            occurrence_counts[(mid, ck)] = occ + 1
            result[mid][(ck, occ)] = d
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _parse_json(s: Optional[str]) -> Optional[list]:
        if s is None:
            return None
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return None
