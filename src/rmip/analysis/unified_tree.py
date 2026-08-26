"""
rmip/analysis/unified_tree.py
==============================
Phase-4 Orchestrator: Unified Mechanism Tree Builder

Two modes
---------
anchored
    One mechanism is the scaffold ("anchor").  All anchor reactions are kept
    verbatim (KEEP_ANCHOR).  Reactions present in only one non-anchor source
    are imported verbatim (IMPORT_SINGLE).  Reactions present in ≥2 non-anchor
    sources are user-directed: default OPTIMIZE, or PICK_SOURCE after editing
    the manifest.

optimize_all
    No anchor.  Reactions in ≥2 sources → OPTIMIZE.  Unique reactions →
    IMPORT_SINGLE.

Workflow
--------
1. plan()  → classify all reactions → write decisions.csv → UnifiedTreePlan
2. (user edits decisions.csv for PATCH_MULTI rows if desired)
3. build() → reads manifest → writes unified mechanism to DB + CSV/YAML/JSON

Decision values
---------------
KEEP_ANCHOR     copy verbatim from anchor
IMPORT_SINGLE   copy verbatim from the single (or chosen) source
OPTIMIZE        compute mean Arrhenius + T-dependent uncertainty band
PICK_SOURCE     copy verbatim from chosen_source_mech_id
DROP            exclude from unified mechanism

Dependencies
------------
  rmip.analysis.graph_comparator  (GraphComparator.compare)
  rmip.analysis.parameter_estimator (ParameterEstimator.estimate_arrhenius)
  rmip.analysis.uncertainty_bands   (UncertaintyBandEstimator)
  rmip.types                        (Phase 4 dataclasses)
  rmip.database.schema              (CRUD helpers)
  rmip.database.schema_phase4       (Phase-4 CRUD helpers)
  rmip.database.migrations          (run_phase4_migrations)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rmip.analysis.parameter_estimator import ParameterEstimator
from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
from rmip.config import RMIPConfig
from rmip.database.schema import (
    get_mechanism, insert_mechanism, get_reactions_for_mechanism,
    get_species_for_mechanism,
)
from rmip.database.schema_phase4 import (
    insert_unified_tree_run, insert_reaction_decision,
    insert_uncertainty_band_rows, upgrade_db_phase4,
)
from rmip.database.migrations import run_phase4_migrations
from rmip.types import (
    ReactionDecision, ReactionKinetics, TemperatureGrid,
    UncertaintyBand, UnifiedTreePlan, UnifiedTreeResult,
)

log = logging.getLogger("rmip.analysis.unified_tree")

# Valid decision values
_VALID_DECISIONS = frozenset({
    "KEEP_ANCHOR", "IMPORT_SINGLE", "OPTIMIZE", "PICK_SOURCE", "DROP"
})

_CSV_FIELDNAMES = [
    "canonical_key", "formula_string", "status", "n_sources",
    "source_mech_ids", "default_decision", "decision",
    "chosen_source_mech_id", "notes", "occurrence_index",
]

_BANDS_CSV_HEADER = [
    "canonical_key", "formula_string", "T_K",
    "k_nom", "k_min", "k_max",
    "f_uncertainty", "sigma_lnk", "f_to_nom",
    "n_contributors", "contributor_mech_ids",
]

_SUMMARY_CSV_HEADER = [
    "canonical_key", "formula_string",
    "f_worst", "f_mean", "sigma_lnk_1000K",
    "A_nom_si", "n_nom", "Ea_nom_si",
    "n_contributors", "contributor_mech_ids", "method", "n_skipped",
]


class UnifiedTreeBuilder:
    """
    Phase-4 orchestrator for building a unified mechanism from N sources.

    Parameters
    ----------
    db_path : str | Path
    config : RMIPConfig | None
    """

    def __init__(
        self,
        db_path: str | Path,
        config: Optional[RMIPConfig] = None,
    ) -> None:
        self._db_path  = str(db_path)
        self._config   = config or RMIPConfig(db_path=str(db_path))
        self._estimator = ParameterEstimator()

    # -------------------------------------------------------------------------
    # Step 1: plan() — classify reactions → decisions.csv
    # -------------------------------------------------------------------------

    def plan(
        self,
        mechanism_ids: List[int],
        mode: str,
        anchor_mech_id: Optional[int] = None,
        manifest_path: str = "decisions.csv",
    ) -> UnifiedTreePlan:
        """
        Classify all reactions across the given mechanisms and write decisions.csv.

        Parameters
        ----------
        mechanism_ids : List[int]
        mode : str
            "anchored" | "optimize_all"
        anchor_mech_id : int | None
            Required when mode == "anchored".
        manifest_path : str
            Path for the CSV manifest.

        Returns
        -------
        UnifiedTreePlan
        """
        if mode not in ("anchored", "optimize_all"):
            raise ValueError(f"mode must be 'anchored' or 'optimize_all', got {mode!r}")
        if mode == "anchored" and anchor_mech_id is None:
            raise ValueError("anchor_mech_id is required for mode='anchored'")
        if anchor_mech_id is not None and anchor_mech_id not in mechanism_ids:
            raise ValueError(
                f"anchor_mech_id={anchor_mech_id} not in mechanism_ids={mechanism_ids}"
            )

        log.info("UnifiedTreeBuilder.plan(): mode=%s, mechs=%s, anchor=%s",
                 mode, mechanism_ids, anchor_mech_id)

        from rmip.analysis.graph_comparator import GraphComparator
        cmp = GraphComparator(self._db_path)
        comparison_result = cmp.compare(mechanism_ids)

        decisions: List[ReactionDecision] = []

        for rc in comparison_result.reaction_classifications:
            present = rc.present_in_mech_ids
            n_src   = len(present)

            if mode == "anchored":
                assert anchor_mech_id is not None
                if anchor_mech_id in present:
                    status           = "IN_ANCHOR"
                    default_decision = "KEEP_ANCHOR"
                else:
                    non_anchor = [m for m in present if m != anchor_mech_id]
                    if len(non_anchor) == 1:
                        status           = "PATCH_SINGLE"
                        default_decision = "IMPORT_SINGLE"
                    else:
                        status           = "PATCH_MULTI"
                        default_decision = "OPTIMIZE"
            else:  # optimize_all
                if n_src >= 2:
                    status           = "SHARED"
                    default_decision = "OPTIMIZE"
                else:
                    status           = "UNIQUE"
                    default_decision = "IMPORT_SINGLE"

            decisions.append(ReactionDecision(
                canonical_key         = rc.canonical_key,
                formula_string        = rc.formula_string,
                status                = status,
                n_sources             = n_src,
                source_mech_ids       = list(present),
                default_decision      = default_decision,
                decision              = default_decision,
                chosen_source_mech_id = None,
                notes                 = "",
                occurrence_index      = rc.occurrence_index,
            ))

        self._write_manifest(decisions, manifest_path)
        self._write_manifest_meta(mode, anchor_mech_id, manifest_path)
        log.info("Manifest written → %s  (%d reactions)", manifest_path, len(decisions))

        plan = UnifiedTreePlan(
            mode              = mode,
            anchor_mech_id    = anchor_mech_id,
            anchor_provenance = None,
            decisions         = decisions,
            manifest_path     = manifest_path,
        )
        return plan

    # -------------------------------------------------------------------------
    # Step 2: build() — assemble unified mechanism from manifest
    # -------------------------------------------------------------------------

    def build(
        self,
        plan: Optional[UnifiedTreePlan] = None,
        manifest_path: str = "decisions.csv",
        name: str = "unified",
        description: str = "",
        tgrid: Optional[TemperatureGrid] = None,
        out_dir: str = ".",
        P_ref_atm: float = 1.0,
        use_cantera_for_falloff: bool = False,
    ) -> UnifiedTreeResult:
        """
        Build the unified mechanism from an (optionally edited) manifest.

        Parameters
        ----------
        plan : UnifiedTreePlan | None
            If provided, decisions are taken from plan.decisions (manifest not
            re-read).  If None, decisions.csv at manifest_path is read.
        manifest_path : str
            Path to read the manifest CSV (used only when plan is None).
        name : str
            Name for the new unified mechanism row.
        description : str
        tgrid : TemperatureGrid | None
            Temperature grid for uncertainty bands.  Default: 500–2500 K, dT=20 K.
        out_dir : str
            Directory for output files.
        P_ref_atm : float
            Reference pressure for PLOG/falloff band evaluation.
        use_cantera_for_falloff : bool
            If True, use Cantera to evaluate Troe/Chebyshev contributions.

        Returns
        -------
        UnifiedTreeResult
        """
        if tgrid is None:
            tgrid = TemperatureGrid()

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Load decisions
        if plan is not None:
            decisions       = plan.decisions
            mode            = plan.mode
            anchor_mech_id  = plan.anchor_mech_id
        else:
            decisions = self._read_manifest(manifest_path)
            mode, anchor_mech_id = self._read_manifest_meta(manifest_path)

        # Validate all decisions
        self._validate_decisions(decisions)

        log.info("UnifiedTreeBuilder.build(): %d decisions", len(decisions))

        band_estimator = UncertaintyBandEstimator(
            P_ref_atm=P_ref_atm,
            use_cantera_for_falloff=use_cantera_for_falloff,
        )

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Ensure Phase-4 tables exist
            upgrade_db_phase4(conn)

            # Collect all mechanism IDs referenced
            all_mech_ids: Set[int] = set()
            for d in decisions:
                all_mech_ids.update(d.source_mech_ids)
            all_mech_ids_list = sorted(all_mech_ids)

            # Load source DB rows
            source_rows = self._load_source_rows(conn, all_mech_ids_list)

            # ---------- Build reaction records ----------
            rxn_records: List[Dict[str, Any]]       = []
            bands:       List[UncertaintyBand]       = []
            # M13 (review.md): `bands` used to be re-paired with decisions
            # after the fact via `zip([d for d in decisions if d.decision ==
            # "OPTIMIZE"], bands)` -- but an OPTIMIZE decision whose record
            # build fails (e.g. _build_optimize_record's "no contributions
            # found" case) is silently dropped from `bands` while still
            # appearing in that re-derived list, shifting every subsequent
            # pairing by one and writing the WRONG uncertainty band against
            # the wrong reaction in the audit DB. Track the decision
            # alongside its band in lockstep instead, so there is no
            # separate list to fall out of sync.
            band_decisions: List[ReactionDecision] = []
            n_keep = n_import = n_opt = n_pick = n_drop = 0

            for d in decisions:
                if d.decision == "DROP":
                    n_drop += 1
                    continue

                rec = self._build_rxn_record(d, source_rows, band_estimator, tgrid,
                                              anchor_mech_id)
                if rec is None:
                    log.warning("Could not build record for %s — treating as DROP",
                                d.canonical_key)
                    n_drop += 1
                    continue

                rxn_records.append(rec)

                if d.decision == "KEEP_ANCHOR":    n_keep   += 1
                elif d.decision == "IMPORT_SINGLE": n_import += 1
                elif d.decision == "OPTIMIZE":      n_opt    += 1
                elif d.decision == "PICK_SOURCE":   n_pick   += 1

                if rec.get("band") is not None:
                    b = rec["band"]
                    b.canonical_key  = d.canonical_key
                    b.formula_string = d.formula_string
                    bands.append(b)
                    band_decisions.append(d)

            # ---------- Collect needed species ----------
            needed_names = self._collect_needed_species(
                conn, rxn_records, source_rows
            )

            # ---------- Fuse species ----------
            sp_records = self._build_species_records(
                conn, needed_names, source_rows
            )

            # ---------- Insert unified mechanism ----------
            ts = datetime.now(timezone.utc).isoformat()
            meta = {
                "mode":      mode,
                "built_at":  ts,
                "P_ref_atm": P_ref_atm,
                "T_grid":    tgrid.to_dict(),
            }
            cur = conn.execute(
                """
                INSERT INTO mechanisms
                    (name, format, filepath, description,
                     n_species, n_reactions, loaded_at, metadata_json)
                VALUES (?, 'unified', NULL, ?, ?, ?, ?, ?)
                """,
                (name, description,
                 len(sp_records), len(rxn_records),
                 ts, json.dumps(meta)),
            )
            unified_id = cur.lastrowid
            log.info("Created unified mechanism DB id=%d", unified_id)

            # ---------- Insert species rows ----------
            sp_name_to_id = self._insert_species(conn, unified_id, sp_records)

            # ---------- Insert reaction rows and stoichiometry ----------
            self._insert_reactions(conn, unified_id, rxn_records,
                                   sp_name_to_id, source_rows)

            # ---------- Write uncertainty bands to DB ----------
            run_id = insert_unified_tree_run(
                conn,
                unified_mechanism_id  = unified_id,
                mode                  = mode,
                contributing_mech_ids = all_mech_ids_list,
                anchor_mech_id        = anchor_mech_id,
                manifest_path         = manifest_path,
                n_keep_anchor         = n_keep,
                n_import_single       = n_import,
                n_optimize            = n_opt,
                n_pick_source         = n_pick,
                n_dropped             = n_drop,
                n_species             = len(sp_records),
            )

            for d, band in zip(band_decisions, bands):
                insert_reaction_decision(
                    conn, run_id,
                    canonical_key         = d.canonical_key,
                    formula_string        = d.formula_string,
                    status                = d.status,
                    decision              = d.decision,
                    source_mech_ids       = d.source_mech_ids,
                    chosen_source_mech_id = d.chosen_source_mech_id,
                    notes                 = d.notes,
                )
                insert_uncertainty_band_rows(
                    conn, run_id,
                    canonical_key  = band.canonical_key,
                    T_K_list       = band.T_K,
                    k_nom_list     = band.k_nom,
                    k_min_list     = band.k_min,
                    k_max_list     = band.k_max,
                    f_list         = band.f_uncertainty,
                    sigma_list     = band.sigma_lnk,
                    f_to_nom_list  = band.f_to_nom,
                    method         = band.method,
                )

            conn.commit()

        # ---------- Write output files ----------
        bands_csv_path   = str(out / "uncertainty_bands.csv")
        summary_csv_path = str(out / "uncertainty_summary.csv")
        yaml_path        = str(out / f"{name}.yaml")
        mso_bounds_path  = str(out / "mso_bounds.json")
        provenance_path  = str(out / "provenance.json")

        self._write_bands_csv(bands, bands_csv_path)
        self._write_summary_csv(bands, rxn_records, summary_csv_path)
        self._write_mso_bounds(rxn_records, unified_id, all_mech_ids_list,
                               mso_bounds_path)
        self._write_provenance(decisions, unified_id, name, meta, provenance_path)

        # Write YAML via CanteraYAMLWriter
        try:
            from rmip.io.cantera_writer import CanteraYAMLWriter
            writer = CanteraYAMLWriter(self._db_path)
            writer.write(
                mechanism_id       = unified_id,
                output_yaml        = yaml_path,
                uncertainty_bands  = bands,
                embed_uncertainty  = True,
            )
        except Exception as exc:
            log.warning("CanteraYAMLWriter failed: %s — YAML not written", exc)
            yaml_path = ""

        result = UnifiedTreeResult(
            unified_mechanism_id = unified_id,
            n_keep_anchor        = n_keep,
            n_import_single      = n_import,
            n_optimize           = n_opt,
            n_pick_source        = n_pick,
            n_dropped            = n_drop,
            n_species            = len(sp_records),
            bands_csv_path       = bands_csv_path,
            summary_csv_path     = summary_csv_path,
            yaml_path            = yaml_path,
            mso_bounds_path      = mso_bounds_path,
            provenance_path      = provenance_path,
        )
        log.info("Build complete.\n%s", result.summary())
        return result

    # -------------------------------------------------------------------------
    # Internal — reaction record building
    # -------------------------------------------------------------------------

    def _build_rxn_record(
        self,
        d: ReactionDecision,
        source_rows: Dict[int, Dict[str, Any]],
        band_estimator: UncertaintyBandEstimator,
        tgrid: TemperatureGrid,
        anchor_mech_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a reaction DB record dict for one decision."""
        decision = d.decision

        if decision in ("KEEP_ANCHOR", "IMPORT_SINGLE", "PICK_SOURCE"):
            # Copy verbatim from the chosen source
            src_mid = self._pick_verbatim_source(d, anchor_mech_id)
            if src_mid is None:
                log.warning("No valid source for decision %s on %s",
                            decision, d.canonical_key)
                return None
            rxn_map = source_rows.get(src_mid, {}).get("reactions_by_key", {})
            src_rxn = rxn_map.get((d.canonical_key, d.occurrence_index))
            if src_rxn is None:
                log.warning("Source mech %d missing reaction %s (occurrence %d)",
                            src_mid, d.canonical_key, d.occurrence_index)
                return None
            return {
                "canonical_key":          d.canonical_key,
                "occurrence_index":       d.occurrence_index,
                "formula_string":         d.formula_string or src_rxn.get("formula_string", ""),
                "decision":               decision,
                "source_mech_id":         src_mid,
                "source_mech_ids":        d.source_mech_ids,
                "A_si":                   src_rxn.get("A_si"),
                "A_orig":                 src_rxn.get("A_orig"),
                "A_units":                src_rxn.get("A_units"),
                "n":                      src_rxn.get("n", 0.0),
                "Ea_si":                  src_rxn.get("Ea_si"),
                "Ea_orig":                src_rxn.get("Ea_orig"),
                "Ea_units":               src_rxn.get("Ea_units"),
                "kinetics_type":          src_rxn.get("kinetics_type", "arrhenius"),
                "kinetics_data_json":     src_rxn.get("kinetics_data_json"),
                "reversible":             src_rxn.get("reversible", 1),
                "duplicate":              src_rxn.get("duplicate", 0),
                "has_third_body":         src_rxn.get("has_third_body", 0),
                "pressure_dependent":     src_rxn.get("pressure_dependent", 0),
                "fusion_class":           "VERBATIM",
                "band":                   None,
            }

        if decision == "OPTIMIZE":
            return self._build_optimize_record(d, source_rows, band_estimator, tgrid)

        return None  # DROP handled by caller

    def _pick_verbatim_source(
        self, d: ReactionDecision, anchor_mech_id: Optional[int] = None,
    ) -> Optional[int]:
        """Return the mechanism ID to copy from for VERBATIM/COPY decisions."""
        if d.decision == "PICK_SOURCE" and d.chosen_source_mech_id is not None:
            return d.chosen_source_mech_id
        if d.decision == "KEEP_ANCHOR":
            if anchor_mech_id is not None and anchor_mech_id in d.source_mech_ids:
                return anchor_mech_id
            log.warning(
                "KEEP_ANCHOR decision on %s but anchor_mech_id (%s) is unknown or "
                "not among sources %s — falling back to lowest source id",
                d.canonical_key, anchor_mech_id, d.source_mech_ids,
            )
        # For IMPORT_SINGLE: there's exactly one source.
        # Fallback (anchor unresolved, or any other case): lowest ID.
        if d.source_mech_ids:
            return min(d.source_mech_ids)
        return None

    def _build_optimize_record(
        self,
        d: ReactionDecision,
        source_rows: Dict[int, Dict[str, Any]],
        band_estimator: UncertaintyBandEstimator,
        tgrid: TemperatureGrid,
    ) -> Optional[Dict[str, Any]]:
        """Build record for OPTIMIZE decision: mean Arrhenius + uncertainty band."""
        # Gather kinetics from each contributing mechanism
        contributions: List[ReactionKinetics] = []
        A_vals: List[float] = []
        n_vals: List[float] = []
        Ea_vals: List[float] = []

        for mid in d.source_mech_ids:
            rxn_map = source_rows.get(mid, {}).get("reactions_by_key", {})
            rxn = rxn_map.get((d.canonical_key, d.occurrence_index))
            if rxn is None:
                continue
            contributions.append(ReactionKinetics(
                mech_id            = mid,
                kinetics_type      = rxn.get("kinetics_type", "arrhenius"),
                A_si               = rxn.get("A_si"),
                n                  = rxn.get("n", 0.0),
                Ea_si              = rxn.get("Ea_si"),
                kinetics_data_json = rxn.get("kinetics_data_json"),
            ))
            if rxn.get("A_si") is not None and rxn.get("Ea_si") is not None:
                A_vals.append(float(rxn["A_si"]))
                n_vals.append(float(rxn.get("n", 0.0)))
                Ea_vals.append(float(rxn["Ea_si"]))

        if not contributions:
            log.warning("OPTIMIZE: no contributions found for %s", d.canonical_key)
            return None

        # Compute mean Arrhenius
        if A_vals:
            est = self._estimator.estimate_arrhenius(A_vals, n_vals, Ea_vals)
            A_nom  = est.A_mean_si
            n_nom  = est.n_mean
            Ea_nom = est.Ea_mean_si
        else:
            # Non-Arrhenius (all PLOG/Troe) — fall back to first source verbatim
            first = contributions[0]
            A_nom  = first.A_si or 0.0
            n_nom  = first.n or 0.0
            Ea_nom = first.Ea_si or 0.0
            log.warning(
                "OPTIMIZE reaction %s has no A_si values — using first source "
                "(mech %d) as nominal Arrhenius.",
                d.canonical_key, contributions[0].mech_id,
            )

        # Compute uncertainty band
        band = band_estimator.band_for_reaction(contributions, tgrid)

        # Pick representative source row for metadata
        src_mid = min(d.source_mech_ids)
        rxn_map = source_rows.get(src_mid, {}).get("reactions_by_key", {})
        src_rxn = rxn_map.get((d.canonical_key, d.occurrence_index), {})

        return {
            "canonical_key":       d.canonical_key,
            "occurrence_index":    d.occurrence_index,
            "formula_string":      d.formula_string or src_rxn.get("formula_string", ""),
            "decision":            "OPTIMIZE",
            "source_mech_id":      src_mid,
            "source_mech_ids":     d.source_mech_ids,
            "A_si":                A_nom,
            "A_orig":              A_nom,
            "A_units":             "SI",
            "n":                   n_nom,
            "Ea_si":               Ea_nom,
            "Ea_orig":             Ea_nom,
            "Ea_units":            "J/mol",
            "kinetics_type":       src_rxn.get("kinetics_type", "arrhenius"),
            "kinetics_data_json":  None,   # mean Arrhenius; PLOG not averaged
            "reversible":          src_rxn.get("reversible", 1),
            "duplicate":           src_rxn.get("duplicate", 0),
            "has_third_body":      src_rxn.get("has_third_body", 0),
            "pressure_dependent":  src_rxn.get("pressure_dependent", 0),
            "fusion_class":        "FUSE",
            "band":                band,
            "A_min_si":            min(A_vals) if A_vals else None,
            "A_max_si":            max(A_vals) if A_vals else None,
            "n_min":               min(n_vals)  if n_vals  else None,
            "n_max":               max(n_vals)  if n_vals  else None,
            "Ea_min_si":           min(Ea_vals) if Ea_vals else None,
            "Ea_max_si":           max(Ea_vals) if Ea_vals else None,
        }

    # -------------------------------------------------------------------------
    # Internal — species handling
    # -------------------------------------------------------------------------

    def _collect_needed_species(
        self,
        conn: sqlite3.Connection,
        rxn_records: List[Dict[str, Any]],
        source_rows: Dict[int, Dict[str, Any]],
    ) -> Set[str]:
        """Return set of species NAMES needed by retained reactions."""
        needed: Set[str] = set()
        for rec in rxn_records:
            src_mid = rec["source_mech_id"]
            rxn_map = source_rows.get(src_mid, {}).get("reactions_by_key", {})
            src_rxn = rxn_map.get((rec["canonical_key"], rec.get("occurrence_index", 0)))
            if src_rxn is None:
                continue
            rxn_id = src_rxn.get("id")
            if rxn_id is None:
                continue
            stoich = conn.execute(
                """
                SELECT sp.name FROM stoichiometry s
                JOIN species sp ON sp.id = s.species_id
                WHERE s.reaction_id = ?
                """,
                (rxn_id,),
            ).fetchall()
            for row in stoich:
                needed.add(row["name"])
        return needed

    def _build_species_records(
        self,
        conn: sqlite3.Connection,
        needed_names: Set[str],
        source_rows: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each needed species name, collect rows from all source mechanisms
        and compute mean thermo (arithmetic mean of NASA-7 coefficients).
        """
        # Build lookup: name → list of source rows
        name_to_rows: Dict[str, List[dict]] = {}
        for mid, data in source_rows.items():
            for sp in data.get("species", []):
                name = sp.get("name", "")
                if name in needed_names:
                    name_to_rows.setdefault(name, []).append(sp)

        records: List[Dict[str, Any]] = []
        for name in sorted(needed_names):
            rows = name_to_rows.get(name, [])
            if not rows:
                log.warning("Species '%s' needed but not found in any source", name)
                continue
            records.append(self._compute_species_record(name, rows))
        return records

    @staticmethod
    def _compute_species_record(
        name: str,
        rows: List[dict],
    ) -> Dict[str, Any]:
        """Arithmetic mean of NASA-7 coefficients across all source rows."""
        import json as _json
        ref = rows[0]   # reference row for metadata

        all_low:  List[List[float]] = []
        all_high: List[List[float]] = []
        tmids, tmins, tmaxs = [], [], []

        for sp in rows:
            try:
                low  = _json.loads(sp.get("coeffs_low_json")  or "null")
                high = _json.loads(sp.get("coeffs_high_json") or "null")
                if low and high and len(low) == 7 and len(high) == 7:
                    all_low.append(low)
                    all_high.append(high)
            except (TypeError, ValueError, _json.JSONDecodeError):
                pass
            tmids.append(float(sp.get("tmid_K") or 1000.0))
            tmins.append(float(sp.get("tmin_K") or 200.0))
            tmaxs.append(float(sp.get("tmax_K") or 5000.0))

        n = max(len(all_low), 1)
        if all_low:
            mean_low  = [sum(c[i] for c in all_low)  / n for i in range(7)]
            mean_high = [sum(c[i] for c in all_high) / n for i in range(7)]
            coeffs_low_json  = _json.dumps(mean_low)
            coeffs_high_json = _json.dumps(mean_high)
        else:
            coeffs_low_json  = None
            coeffs_high_json = None

        # Modal Tmid (most common)
        from collections import Counter as _Counter
        tmid_counts = _Counter(tmids)
        tmid = tmid_counts.most_common(1)[0][0] if tmids else 1000.0

        return {
            "name":              name,
            "inchi_key":         ref.get("inchi_key"),
            "canonical_smiles":  ref.get("canonical_smiles"),
            "source_smiles":     ref.get("source_smiles"),
            "inchi":             ref.get("inchi"),
            "formula":           ref.get("formula"),
            "mw_kg_mol":         ref.get("mw_kg_mol"),
            "charge":            ref.get("charge", 0),
            "glossary_matched":  ref.get("glossary_matched", 0),
            "thermo_type":       "nasa7",
            "tmin_K":            min(tmins) if tmins else 200.0,
            "tmid_K":            tmid,
            "tmax_K":            max(tmaxs) if tmaxs else 5000.0,
            "coeffs_low_json":   coeffs_low_json,
            "coeffs_high_json":  coeffs_high_json,
            "h298_kJ_mol":       ref.get("h298_kJ_mol"),
            "s298_J_molK":       ref.get("s298_J_molK"),
            "thermo_comment":    f"Unified from {len(rows)} source(s)",
        }

    # -------------------------------------------------------------------------
    # Internal — DB insertion
    # -------------------------------------------------------------------------

    def _insert_species(
        self,
        conn: sqlite3.Connection,
        unified_id: int,
        sp_records: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Insert species and return {name: id} mapping."""
        name_to_id: Dict[str, int] = {}
        for sp in sp_records:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO species (
                    mechanism_id, name, canonical_smiles, source_smiles,
                    inchi, inchi_key, formula, mw_kg_mol, charge,
                    glossary_matched, thermo_type,
                    tmin_K, tmid_K, tmax_K,
                    coeffs_low_json, coeffs_high_json,
                    h298_kJ_mol, s298_J_molK, thermo_comment
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    unified_id,
                    sp["name"],
                    sp.get("canonical_smiles"),
                    sp.get("source_smiles"),
                    sp.get("inchi"),
                    sp.get("inchi_key"),
                    sp.get("formula"),
                    sp.get("mw_kg_mol"),
                    sp.get("charge", 0),
                    sp.get("glossary_matched", 0),
                    sp.get("thermo_type", "nasa7"),
                    sp.get("tmin_K"),
                    sp.get("tmid_K"),
                    sp.get("tmax_K"),
                    sp.get("coeffs_low_json"),
                    sp.get("coeffs_high_json"),
                    sp.get("h298_kJ_mol"),
                    sp.get("s298_J_molK"),
                    sp.get("thermo_comment"),
                ),
            )
            if cur.lastrowid and cur.lastrowid > 0:
                name_to_id[sp["name"]] = cur.lastrowid
            else:
                row = conn.execute(
                    "SELECT id FROM species WHERE mechanism_id=? AND name=?",
                    (unified_id, sp["name"]),
                ).fetchone()
                if row:
                    name_to_id[sp["name"]] = row["id"]
        return name_to_id

    def _insert_reactions(
        self,
        conn: sqlite3.Connection,
        unified_id: int,
        rxn_records: List[Dict[str, Any]],
        sp_name_to_id: Dict[str, int],
        source_rows: Dict[int, Dict[str, Any]],
    ) -> None:
        """Insert reaction rows and copy stoichiometry."""
        # Check if fusion columns exist
        cols = {row[1] for row in conn.execute("PRAGMA table_info(reactions)").fetchall()}
        has_fusion_cols = "fusion_class" in cols

        for rec in rxn_records:
            src_mid = rec["source_mech_id"]
            ck      = rec["canonical_key"]

            if has_fusion_cols:
                cur = conn.execute(
                    """
                    INSERT INTO reactions (
                        mechanism_id, formula_string, canonical_key,
                        reversible, duplicate,
                        kinetics_type, pressure_dependent, has_third_body,
                        A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units,
                        kinetics_data_json, rmg_comment,
                        fusion_class, source_mechanism_ids_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        unified_id,
                        rec["formula_string"],
                        ck,
                        rec.get("reversible", 1),
                        rec.get("duplicate", 0),
                        rec.get("kinetics_type", "arrhenius"),
                        rec.get("pressure_dependent", 0),
                        rec.get("has_third_body", 0),
                        rec.get("A_si"),
                        rec.get("A_orig"),
                        rec.get("A_units"),
                        rec.get("n", 0.0),
                        rec.get("Ea_si"),
                        rec.get("Ea_orig"),
                        rec.get("Ea_units"),
                        rec.get("kinetics_data_json"),
                        f"Unified ({rec['decision']}) from {rec['source_mech_ids']}",
                        rec.get("fusion_class", "VERBATIM"),
                        json.dumps(rec.get("source_mech_ids", [])),
                    ),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO reactions (
                        mechanism_id, formula_string, canonical_key,
                        reversible, duplicate,
                        kinetics_type, pressure_dependent, has_third_body,
                        A_si, A_orig, A_units, n, Ea_si, Ea_orig, Ea_units,
                        kinetics_data_json, rmg_comment
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        unified_id,
                        rec["formula_string"],
                        ck,
                        rec.get("reversible", 1),
                        rec.get("duplicate", 0),
                        rec.get("kinetics_type", "arrhenius"),
                        rec.get("pressure_dependent", 0),
                        rec.get("has_third_body", 0),
                        rec.get("A_si"),
                        rec.get("A_orig"),
                        rec.get("A_units"),
                        rec.get("n", 0.0),
                        rec.get("Ea_si"),
                        rec.get("Ea_orig"),
                        rec.get("Ea_units"),
                        rec.get("kinetics_data_json"),
                        f"Unified ({rec['decision']}) from {rec['source_mech_ids']}",
                    ),
                )
            rxn_id = cur.lastrowid

            # Copy stoichiometry from the reference source reaction
            rxn_map = source_rows.get(src_mid, {}).get("reactions_by_key", {})
            src_rxn = rxn_map.get((ck, rec.get("occurrence_index", 0)))
            if src_rxn is None:
                continue
            self._copy_stoichiometry(conn, rxn_id, src_rxn["id"], sp_name_to_id)

    @staticmethod
    def _copy_stoichiometry(
        conn: sqlite3.Connection,
        new_rxn_id: int,
        src_rxn_id: int,
        sp_name_to_id: Dict[str, int],
    ) -> None:
        """Copy stoichiometry from source reaction into unified mechanism."""
        stoich_rows = conn.execute(
            """
            SELECT s.stoich_coeff, s.role, sp.name
            FROM stoichiometry s
            JOIN species sp ON sp.id = s.species_id
            WHERE s.reaction_id = ?
            """,
            (src_rxn_id,),
        ).fetchall()
        for row in stoich_rows:
            sp_id = sp_name_to_id.get(row["name"])
            if sp_id is None:
                log.debug("Stoich: species '%s' not in unified sp_map", row["name"])
                continue
            conn.execute(
                "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
                "VALUES (?,?,?,?)",
                (new_rxn_id, sp_id, row["stoich_coeff"], row["role"]),
            )

    # -------------------------------------------------------------------------
    # Internal — source data loading
    # -------------------------------------------------------------------------

    def _load_source_rows(
        self,
        conn: sqlite3.Connection,
        mech_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Load raw DB rows for all source mechanisms.

        M9 (review.md): `reactions_by_key` used to be keyed on plain
        canonical_key, so two CHEMKIN DUPLICATE rows sharing one
        canonical_key within a mechanism collided -- every ReactionDecision
        referencing that canonical_key (whether it should have resolved to
        occurrence 0 or 1) silently picked up the same last-read row's
        kinetics. Keyed on (canonical_key, occurrence_index) instead, in
        `id` (insertion/file) order -- must match GraphComparator's own
        occurrence counting (`_load_all_reactions`) since plan() sources
        `ReactionDecision.occurrence_index` from there.
        """
        placeholders = ",".join("?" * len(mech_ids))
        data: Dict[int, Dict[str, Any]] = {
            mid: {"species": [], "reactions_by_key": {}}
            for mid in mech_ids
        }

        sp_rows = conn.execute(
            f"SELECT * FROM species WHERE mechanism_id IN ({placeholders})",
            mech_ids,
        ).fetchall()
        for row in sp_rows:
            mid = row["mechanism_id"]
            data[mid]["species"].append(dict(row))

        rxn_rows = conn.execute(
            f"SELECT * FROM reactions WHERE mechanism_id IN ({placeholders}) "
            f"ORDER BY mechanism_id, id",
            mech_ids,
        ).fetchall()
        occurrence_counts: Dict[Tuple[int, str], int] = {}
        for row in rxn_rows:
            mid = row["mechanism_id"]
            ck  = row["canonical_key"] or row["formula_string"]
            occ = occurrence_counts.get((mid, ck), 0)
            occurrence_counts[(mid, ck)] = occ + 1
            data[mid]["reactions_by_key"][(ck, occ)] = dict(row)

        return data

    # -------------------------------------------------------------------------
    # Internal — manifest IO
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_manifest(decisions: List[ReactionDecision], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for d in decisions:
                writer.writerow(d.to_dict())

    @staticmethod
    def _read_manifest(path: str) -> List[ReactionDecision]:
        decisions: List[ReactionDecision] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                decisions.append(ReactionDecision.from_dict(row))
        return decisions

    @staticmethod
    def _manifest_meta_path(manifest_path: str) -> str:
        return manifest_path + ".meta.json"

    @classmethod
    def _write_manifest_meta(
        cls, mode: str, anchor_mech_id: Optional[int], manifest_path: str,
    ) -> None:
        """
        Persist plan()-level info (mode, anchor_mech_id) that decisions.csv's
        per-reaction rows can't carry, so a later build() invoked from a
        separate process (the real `rmip unify-build` CLI path — manifest_path
        only, no in-memory UnifiedTreePlan) can still resolve KEEP_ANCHOR
        decisions to the correct anchor mechanism instead of guessing.
        """
        with open(cls._manifest_meta_path(manifest_path), "w", encoding="utf-8") as fh:
            json.dump({"mode": mode, "anchor_mech_id": anchor_mech_id}, fh)

    @classmethod
    def _read_manifest_meta(
        cls, manifest_path: str,
    ) -> Tuple[str, Optional[int]]:
        """Read the (mode, anchor_mech_id) sidecar written by plan(). Missing
        sidecar (e.g. a manifest from before this fix) degrades gracefully to
        ("unknown", None) — the same fallback behaviour as before this fix."""
        meta_path = cls._manifest_meta_path(manifest_path)
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            return meta.get("mode", "unknown"), meta.get("anchor_mech_id")
        except (FileNotFoundError, json.JSONDecodeError):
            log.warning(
                "No manifest metadata sidecar found at %s — anchor_mech_id "
                "unknown, KEEP_ANCHOR resolution will fall back to lowest "
                "source id",
                meta_path,
            )
            return "unknown", None

    # -------------------------------------------------------------------------
    # Internal — decision validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_decisions(decisions: List[ReactionDecision]) -> None:
        """Raise ValueError on invalid decisions (mirrors MSOInterface.apply() strictness)."""
        errors: List[str] = []
        for d in decisions:
            if d.decision not in _VALID_DECISIONS:
                errors.append(
                    f"Unknown decision '{d.decision}' for {d.canonical_key}"
                )
            if d.decision == "PICK_SOURCE":
                if d.chosen_source_mech_id is None:
                    errors.append(
                        f"PICK_SOURCE requires chosen_source_mech_id for {d.canonical_key}"
                    )
                elif d.chosen_source_mech_id not in d.source_mech_ids:
                    errors.append(
                        f"chosen_source_mech_id={d.chosen_source_mech_id} not in "
                        f"source_mech_ids={d.source_mech_ids} for {d.canonical_key}"
                    )
            if d.decision == "OPTIMIZE" and d.n_sources < 2:
                errors.append(
                    f"OPTIMIZE on single-source reaction {d.canonical_key} — "
                    "use IMPORT_SINGLE or PICK_SOURCE instead"
                )
        if errors:
            raise ValueError(
                f"Manifest validation failed ({len(errors)} errors):\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    # -------------------------------------------------------------------------
    # Internal — output files
    # -------------------------------------------------------------------------

    def _write_bands_csv(
        self,
        bands: List[UncertaintyBand],
        path: str,
    ) -> None:
        """Write long-format uncertainty_bands.csv (one row per reaction × T)."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_BANDS_CSV_HEADER)
            writer.writeheader()
            for band in bands:
                contrib_str = json.dumps(band.contributor_mech_ids)
                n_contrib   = len(band.contributor_mech_ids)
                for i, T in enumerate(band.T_K):
                    writer.writerow({
                        "canonical_key":       band.canonical_key,
                        "formula_string":      band.formula_string,
                        "T_K":                 T,
                        "k_nom":               band.k_nom[i],
                        "k_min":               band.k_min[i],
                        "k_max":               band.k_max[i],
                        "f_uncertainty":       band.f_uncertainty[i],
                        "sigma_lnk":           band.sigma_lnk[i],
                        "f_to_nom":            band.f_to_nom[i],
                        "n_contributors":      n_contrib,
                        "contributor_mech_ids": contrib_str,
                    })

    def _write_summary_csv(
        self,
        bands: List[UncertaintyBand],
        rxn_records: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """Write wide-format uncertainty_summary.csv (one row per OPTIMIZE reaction)."""
        # Build lookup: canonical_key → rxn_record for A/n/Ea
        rec_map = {r["canonical_key"]: r for r in rxn_records}

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SUMMARY_CSV_HEADER)
            writer.writeheader()
            for band in bands:
                rec = rec_map.get(band.canonical_key, {})
                sigma_1000 = band.sigma_lnk_at(1000.0)
                writer.writerow({
                    "canonical_key":       band.canonical_key,
                    "formula_string":      band.formula_string,
                    "f_worst":             band.f_worst(),
                    "f_mean":              band.f_mean(),
                    "sigma_lnk_1000K":     sigma_1000,
                    "A_nom_si":            rec.get("A_si"),
                    "n_nom":               rec.get("n"),
                    "Ea_nom_si":           rec.get("Ea_si"),
                    "n_contributors":      len(band.contributor_mech_ids),
                    "contributor_mech_ids": json.dumps(band.contributor_mech_ids),
                    "method":              band.method,
                    "n_skipped":           band.n_skipped,
                })

    @staticmethod
    def _write_mso_bounds(
        rxn_records: List[Dict[str, Any]],
        unified_id: int,
        source_ids: List[int],
        path: str,
    ) -> None:
        """Write mso_bounds.json (same schema as FusionEngine.export_mso_bounds)."""
        blocks = []
        for rec in rxn_records:
            if rec.get("decision") != "OPTIMIZE":
                continue
            blocks.append({
                "canonical_key":   rec["canonical_key"],
                "formula_string":  rec["formula_string"],
                "source_mech_ids": rec.get("source_mech_ids", []),
                "params_mean":     [rec.get("A_si"), rec.get("n", 0.0), rec.get("Ea_si")],
                "params_lb":       [rec.get("A_min_si"), rec.get("n_min"), rec.get("Ea_min_si")],
                "params_ub":       [rec.get("A_max_si"), rec.get("n_max"), rec.get("Ea_max_si")],
                "param_names":     ["A_si", "n", "Ea_si"],
                "units":           ["m^3/mol/s", "1", "J/mol"],
            })
        out = {
            "unified_mechanism_id": unified_id,
            "source_mechanism_ids": source_ids,
            "n_optimize_reactions": len(blocks),
            "fuse_blocks":          blocks,
        }
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    @staticmethod
    def _write_provenance(
        decisions: List[ReactionDecision],
        unified_id: int,
        name: str,
        meta: dict,
        path: str,
    ) -> None:
        """Write provenance.json with per-reaction decision audit trail."""
        prov = {
            "unified_mechanism_id": unified_id,
            "name":                 name,
            "metadata":             meta,
            "decisions": [
                {
                    "canonical_key":         d.canonical_key,
                    "formula_string":        d.formula_string,
                    "status":                d.status,
                    "decision":              d.decision,
                    "source_mech_ids":       d.source_mech_ids,
                    "chosen_source_mech_id": d.chosen_source_mech_id,
                    "notes":                 d.notes,
                }
                for d in decisions
            ],
        }
        with open(path, "w") as fh:
            json.dump(prov, fh, indent=2)
