"""
rmip/types.py
=============
Shared dataclasses, enumerations, and typed contracts used across all RMIP phases.
All cross-module data transfer objects (DTOs) live here to avoid circular imports.

Design decisions:
  - All dataclasses use __slots__ where safe (no inheritance) for memory efficiency
  - JSON serialisability is provided via .to_dict() / from_dict() on every class
  - numpy arrays never appear here; only plain Python lists/dicts cross module boundaries
  - SI units throughout; original units stored separately only in the DB
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SimType(str, Enum):
    IDT = "IDT"   # Ignition Delay Time
    LFS = "LFS"   # Laminar Flame Speed


class ReactionClass(str, Enum):
    """Classification of a reaction when comparing N mechanisms."""
    UNIQUE          = "UNIQUE"           # present in exactly one mechanism
    SHARED_IDENTICAL = "SHARED_IDENTICAL" # present in ≥2, A/n/Ea agree within tol
    SHARED_DIVERGENT = "SHARED_DIVERGENT" # present in ≥2, A/n/Ea differ


class SpeciesClass(str, Enum):
    """Classification of a species when comparing N mechanisms."""
    UNIVERSAL = "UNIVERSAL"  # present in ALL mechanisms
    PARTIAL   = "PARTIAL"    # present in some (≥2) but not all
    UNIQUE    = "UNIQUE"     # present in exactly one


class FusionTreatment(str, Enum):
    """How the FusionEngine (M7) should handle a reaction."""
    VERBATIM = "VERBATIM"  # copy from the single source mechanism unchanged
    COPY     = "COPY"      # identical across all → copy as-is
    FUSE     = "FUSE"      # divergent across mechanisms → use computed mean/unc


class Regime(str, Enum):
    """Temperature regime for IDT classification."""
    LTC = "LTC"   # Low Temperature Chemistry  T < 850 K
    NTC = "NTC"   # Negative Temperature Coefficient  850 ≤ T < 1000 K
    HTC = "HTC"   # High Temperature Chemistry  T ≥ 1000 K
    ALL = "all"   # across all regimes


# ---------------------------------------------------------------------------
# Simulation Primitives
# ---------------------------------------------------------------------------

@dataclass
class SimCondition:
    """
    A single simulation condition (T, P, phi) for IDT or LFS.
    fuel and oxidizer are Cantera-style composition strings.

    Examples
    --------
    SimCondition(T_K=1200, P_atm=13.5, phi=1.0,
                 fuel="NC7H16:1.0", oxidizer="O2:1.0, N2:3.76")
    """
    T_K:      float
    P_atm:    float
    phi:      float
    fuel:     str                            # Cantera composition string, e.g. "NC7H16:1"
    oxidizer: str = "O2:1.0, N2:3.76"       # standard air

    def regime(self) -> Regime:
        """Classify temperature into combustion regime."""
        if self.T_K < 850.0:
            return Regime.LTC
        elif self.T_K < 1000.0:
            return Regime.NTC
        else:
            return Regime.HTC

    def to_dict(self) -> dict:
        return {
            "T_K": self.T_K, "P_atm": self.P_atm, "phi": self.phi,
            "fuel": self.fuel, "oxidizer": self.oxidizer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimCondition":
        return cls(**d)


@dataclass
class IDTResult:
    """Result of a single constant-volume adiabatic IDT simulation."""
    condition:    SimCondition
    mechanism_id: int
    idt_s:        Optional[float]   # None → no ignition detected within t_end
    converged:    bool
    cpu_time_s:   float
    error_msg:    Optional[str] = None

    def to_dict(self) -> dict:
        return {
            **self.condition.to_dict(),
            "mechanism_id": self.mechanism_id,
            "idt_s":        self.idt_s,
            "converged":    self.converged,
            "cpu_time_s":   self.cpu_time_s,
            "error_msg":    self.error_msg,
        }


@dataclass
class LFSResult:
    """Result of a single freely-propagating premixed flame simulation."""
    condition:    SimCondition
    mechanism_id: int
    lfs_m_s:      Optional[float]   # None → solver did not converge
    converged:    bool
    cpu_time_s:   float
    error_msg:    Optional[str] = None

    def to_dict(self) -> dict:
        return {
            **self.condition.to_dict(),
            "mechanism_id": self.mechanism_id,
            "lfs_m_s":      self.lfs_m_s,
            "converged":    self.converged,
            "cpu_time_s":   self.cpu_time_s,
            "error_msg":    self.error_msg,
        }


# ---------------------------------------------------------------------------
# Sparsity (M4)
# ---------------------------------------------------------------------------

@dataclass
class SparsityReport:
    """
    Output of SparsityDetector.analyze() for a single mechanism.
    Node IDs use the GraphBuilder convention: "S_{db_id}" / "R_{db_id}".
    """
    mechanism_id:          int
    n_species:             int
    n_reactions:           int
    dead_end_source_nodes: List[str]    # in-degree 0 in bipartite graph (never produced)
    dead_end_sink_nodes:   List[str]    # out-degree 0 (never consumed)
    isolated_nodes:        List[str]    # completely disconnected (no edges at all)
    orphan_reaction_nodes: List[str]    # all reactants are dead-end sources
    connected_components:  List[List[str]]  # weakly connected components
    coverage_fraction:     float        # fraction of species that can reach CO2/H2O/N2

    @property
    def n_dead_end_sources(self) -> int: return len(self.dead_end_source_nodes)

    @property
    def n_dead_end_sinks(self) -> int: return len(self.dead_end_sink_nodes)

    @property
    def n_isolated(self) -> int: return len(self.isolated_nodes)

    @property
    def n_components(self) -> int: return len(self.connected_components)

    def to_dict(self) -> dict:
        return {
            "mechanism_id":           self.mechanism_id,
            "n_species":              self.n_species,
            "n_reactions":            self.n_reactions,
            "n_dead_end_sources":     self.n_dead_end_sources,
            "n_dead_end_sinks":       self.n_dead_end_sinks,
            "n_isolated":             self.n_isolated,
            "n_components":           self.n_components,
            "coverage_fraction":      self.coverage_fraction,
            "dead_end_source_nodes":  self.dead_end_source_nodes,
            "dead_end_sink_nodes":    self.dead_end_sink_nodes,
            "isolated_nodes":         self.isolated_nodes,
            "orphan_reaction_nodes":  self.orphan_reaction_nodes,
            "connected_components":   self.connected_components,
        }


@dataclass
class SparsityComparison:
    """
    Cross-mechanism sparsity comparison: what is present in A but not B and vice versa.
    Produced by SparsityDetector.compare_pair() and .compare_all().
    """
    mechanism_ids:         List[int]
    species_exclusive:     Dict[int, List[str]]   # {mech_id: [inchi_keys not in others]}
    reactions_exclusive:   Dict[int, List[str]]   # {mech_id: [canonical_keys not in others]}
    species_common:        List[str]              # inchi_keys present in ALL mechs
    reactions_common:      List[str]              # canonical_keys present in ALL mechs

    def to_dict(self) -> dict:
        return {
            "mechanism_ids":       self.mechanism_ids,
            "species_exclusive":   {str(k): v for k, v in self.species_exclusive.items()},
            "reactions_exclusive": {str(k): v for k, v in self.reactions_exclusive.items()},
            "n_species_common":    len(self.species_common),
            "n_reactions_common":  len(self.reactions_common),
        }


# ---------------------------------------------------------------------------
# Graph Comparison (GraphComparator)
# ---------------------------------------------------------------------------

@dataclass
class ReactionKineticsSet:
    """
    Kinetics values for a single reaction across multiple mechanisms.
    mech_data keys are mechanism DB IDs.
    """
    canonical_key:  str
    formula_string: str       # representative (from first mechanism encountered)
    mech_data: Dict[int, Dict[str, Any]]
    # mech_data schema: {mech_id: {"A_si": float, "n": float, "Ea_si": float,
    #                               "kinetics_type": str, "kinetics_data_json": str}}

    @property
    def mechanism_ids(self) -> List[int]:
        return list(self.mech_data.keys())

    @property
    def A_values(self) -> List[float]:
        return [d["A_si"] for d in self.mech_data.values()
                if d.get("A_si") is not None]

    @property
    def n_values(self) -> List[float]:
        return [d["n"] for d in self.mech_data.values()
                if d.get("n") is not None]

    @property
    def Ea_values(self) -> List[float]:
        return [d["Ea_si"] for d in self.mech_data.values()
                if d.get("Ea_si") is not None]


@dataclass
class ThermoSet:
    """NASA-7 thermo coefficients for a single species across multiple mechanisms."""
    inchi_key:      str
    canonical_smiles: str
    mech_data: Dict[int, Dict[str, Any]]
    # schema: {mech_id: {"coeffs_low": [7], "coeffs_high": [7],
    #                     "tmin_K": float, "tmid_K": float, "tmax_K": float}}


@dataclass
class ReactionClassification:
    """Per-reaction result from GraphComparator."""
    canonical_key:      str
    formula_string:     str
    classification:     ReactionClass
    present_in_mech_ids: List[int]
    kinetics_set:       Optional[ReactionKineticsSet] = None
    # Populated only for SHARED_DIVERGENT
    mean_A_si:   Optional[float] = None   # geometric mean
    mean_n:      Optional[float] = None   # arithmetic mean
    mean_Ea_si:  Optional[float] = None   # arithmetic mean
    min_A_si:    Optional[float] = None
    max_A_si:    Optional[float] = None
    min_n:       Optional[float] = None
    max_n:       Optional[float] = None
    min_Ea_si:   Optional[float] = None
    max_Ea_si:   Optional[float] = None

    def fusion_treatment(self) -> FusionTreatment:
        return {
            ReactionClass.UNIQUE:           FusionTreatment.VERBATIM,
            ReactionClass.SHARED_IDENTICAL: FusionTreatment.COPY,
            ReactionClass.SHARED_DIVERGENT: FusionTreatment.FUSE,
        }[self.classification]

    def to_dict(self) -> dict:
        return {
            "canonical_key":       self.canonical_key,
            "formula_string":      self.formula_string,
            "classification":      self.classification.value,
            "present_in_mech_ids": self.present_in_mech_ids,
            "fusion_treatment":    self.fusion_treatment().value,
            "mean_A_si":           self.mean_A_si,
            "mean_n":              self.mean_n,
            "mean_Ea_si":          self.mean_Ea_si,
            "min_A_si":            self.min_A_si,
            "max_A_si":            self.max_A_si,
            "min_n":               self.min_n,
            "max_n":               self.max_n,
            "min_Ea_si":           self.min_Ea_si,
            "max_Ea_si":           self.max_Ea_si,
        }


@dataclass
class SpeciesClassification:
    """Per-species result from GraphComparator."""
    inchi_key:          str
    canonical_smiles:   str
    formula:            str
    classification:     SpeciesClass
    present_in_mech_ids: List[int]
    thermo_set:         Optional[ThermoSet] = None

    def to_dict(self) -> dict:
        return {
            "inchi_key":           self.inchi_key,
            "canonical_smiles":    self.canonical_smiles,
            "formula":             self.formula,
            "classification":      self.classification.value,
            "present_in_mech_ids": self.present_in_mech_ids,
        }


@dataclass
class GraphComparisonResult:
    """
    Full result of GraphComparator.compare(mechanism_ids).
    This is the central data structure driving M7 (FusionEngine) in Phase 4.
    """
    session_id:               int
    mechanism_ids:            List[int]
    species_classifications:  List[SpeciesClassification]
    reaction_classifications: List[ReactionClassification]

    # ---- convenience filters ----

    @property
    def universal_species(self) -> List[SpeciesClassification]:
        return [s for s in self.species_classifications
                if s.classification == SpeciesClass.UNIVERSAL]

    @property
    def partial_species(self) -> List[SpeciesClassification]:
        return [s for s in self.species_classifications
                if s.classification == SpeciesClass.PARTIAL]

    @property
    def unique_species(self) -> List[SpeciesClassification]:
        return [s for s in self.species_classifications
                if s.classification == SpeciesClass.UNIQUE]

    @property
    def shared_identical_reactions(self) -> List[ReactionClassification]:
        return [r for r in self.reaction_classifications
                if r.classification == ReactionClass.SHARED_IDENTICAL]

    @property
    def shared_divergent_reactions(self) -> List[ReactionClassification]:
        return [r for r in self.reaction_classifications
                if r.classification == ReactionClass.SHARED_DIVERGENT]

    @property
    def unique_reactions(self) -> List[ReactionClassification]:
        return [r for r in self.reaction_classifications
                if r.classification == ReactionClass.UNIQUE]

    def summary_dict(self) -> dict:
        return {
            "n_mechanisms":            len(self.mechanism_ids),
            "mechanism_ids":           self.mechanism_ids,
            "session_id":              self.session_id,
            "species": {
                "total":               len(self.species_classifications),
                "universal":           len(self.universal_species),
                "partial":             len(self.partial_species),
                "unique":              len(self.unique_species),
            },
            "reactions": {
                "total":               len(self.reaction_classifications),
                "shared_identical":    len(self.shared_identical_reactions),
                "shared_divergent":    len(self.shared_divergent_reactions),
                "unique":              len(self.unique_reactions),
            },
        }

    def to_dict(self) -> dict:
        return {
            **self.summary_dict(),
            "species_classifications":  [s.to_dict() for s in self.species_classifications],
            "reaction_classifications": [r.to_dict() for r in self.reaction_classifications],
        }


# ---------------------------------------------------------------------------
# Union Graph Specification (input to M7 FusionEngine, Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class UnionGraphSpec:
    """
    Specification for building the merged (union) graph.

    The union graph contains ALL species and ALL reactions found across
    ANY of the input mechanisms.  Reactions are split into three treatment
    buckets to guide the FusionEngine (M7):

      VERBATIM reactions → copy from the single-source mechanism unchanged
      COPY     reactions → identical across all present mechanisms → copy as-is
      FUSE     reactions → divergent → use (mean_params, unc_params) from DoE

    Transport data is NOT included (design decision D6) but a hook is provided
    via `transport_verbatim` for future use without breaking the core contract.

    Consumed by:
      M7  FusionEngine   → builds the merged CHEMKIN/YAML mechanism
      M9  DoEFramework   → uses fuse_params to set up optimisation bounds
    """
    mechanism_ids:          List[int]
    comparison_session_id:  int

    # All unique species: {inchi_key: SpeciesClassification}
    species_map: Dict[str, SpeciesClassification] = field(default_factory=dict)

    # Reaction buckets: {canonical_key: ReactionClassification}
    verbatim_reactions: Dict[str, ReactionClassification] = field(default_factory=dict)
    copy_reactions:     Dict[str, ReactionClassification] = field(default_factory=dict)
    fuse_reactions:     Dict[str, ReactionClassification] = field(default_factory=dict)

    # Mean/uncertainty params for FUSE reactions (set by GraphComparator)
    # {canonical_key: {
    #     "mean_A_si": float, "mean_n": float, "mean_Ea_si": float,
    #     "unc_A_si":  (min, max),
    #     "unc_n":     (min, max),
    #     "unc_Ea_si": (min, max),
    # }}
    fuse_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Placeholder for transport data (D6: currently always empty)
    transport_verbatim: Dict[int, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mechanism_ids":            self.mechanism_ids,
            "comparison_session_id":    self.comparison_session_id,
            "n_species_total":          len(self.species_map),
            "n_verbatim_reactions":     len(self.verbatim_reactions),
            "n_copy_reactions":         len(self.copy_reactions),
            "n_fuse_reactions":         len(self.fuse_reactions),
        }

    def save_json(self, path: str) -> None:
        """Serialise the spec to JSON for hand-off to M7/M9."""
        data = {
            "mechanism_ids":         self.mechanism_ids,
            "comparison_session_id": self.comparison_session_id,
            "species_map":    {k: v.to_dict() for k, v in self.species_map.items()},
            "verbatim_reactions": {k: v.to_dict() for k, v in self.verbatim_reactions.items()},
            "copy_reactions":     {k: v.to_dict() for k, v in self.copy_reactions.items()},
            "fuse_reactions":     {k: v.to_dict() for k, v in self.fuse_reactions.items()},
            "fuse_params":        self.fuse_params,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# MSO Interface (M9 output / MSO input)
# ---------------------------------------------------------------------------

@dataclass
class MSOParameters:
    """
    Parameter set ready for hand-off to KP's existing MSO framework.
    Each entry in param_blocks corresponds to one FUSE reaction.

    This is the output of M9 DoEFramework.prepare_mso_input() and the
    input to KP's OptimizationTool_v4.py (or MSO wrapper).
    """
    session_id:   int
    fuel_name:    str
    param_blocks: List[Dict[str, Any]]
    # Each block:
    # {
    #   "canonical_key": str,
    #   "formula_string": str,
    #   "params_mean": [A_si, n, Ea_si],
    #   "params_lb":   [A_min, n_min, Ea_min],   # lower bounds
    #   "params_ub":   [A_max, n_max, Ea_max],   # upper bounds
    #   "param_names": ["A_si", "n", "Ea_si"],
    #   "units":       ["m^3/mol/s", "1", "J/mol"],
    # }
    target_conditions: List[Dict[str, Any]]   # list of SimCondition.to_dict()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({
                "session_id":        self.session_id,
                "fuel_name":         self.fuel_name,
                "param_blocks":      self.param_blocks,
                "target_conditions": self.target_conditions,
                "metadata":          self.metadata,
            }, fh, indent=2)


# ---------------------------------------------------------------------------
# Phase 4 — Unified Tree / Uncertainty Bands / Missing Species
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemperatureGrid:
    """Temperature grid for uncertainty band evaluation (every dT K)."""
    T_min: float = 500.0
    T_max: float = 2500.0
    dT:    float = 20.0

    def points(self) -> List[float]:
        pts: List[float] = []
        T = self.T_min
        while T <= self.T_max + 1e-9:
            pts.append(T)
            T += self.dT
        return pts

    def to_dict(self) -> dict:
        return {"T_min": self.T_min, "T_max": self.T_max, "dT": self.dT}


@dataclass
class ReactionKinetics:
    """Kinetics for a single reaction from a single contributing mechanism."""
    mech_id:            int
    kinetics_type:      str          # 'arrhenius'|'plog'|'troe'|'lindemann'|'chebyshev'|...
    A_si:               Optional[float]
    n:                  float
    Ea_si:              Optional[float]
    kinetics_data_json: Optional[str] = None   # raw PLOG/Troe/Cheb blob

    def to_dict(self) -> dict:
        return {
            "mech_id":            self.mech_id,
            "kinetics_type":      self.kinetics_type,
            "A_si":               self.A_si,
            "n":                  self.n,
            "Ea_si":              self.Ea_si,
        }


@dataclass
class UncertaintyBand:
    """T-dependent uncertainty band for one OPTIMIZE reaction (minmax method)."""
    canonical_key:        str
    formula_string:       str
    T_K:                  List[float]
    k_nom:                List[float]   # geometric mean of contributions at each T
    k_min:                List[float]
    k_max:                List[float]
    f_uncertainty:        List[float]   # sqrt(k_max / k_min)
    sigma_lnk:            List[float]   # ln(f) / 3  (3-sigma convention)
    f_to_nom:             List[float]   # k_max / k_nom (diagnostic)
    contributor_mech_ids: List[int]
    method:               str = "minmax"
    n_skipped:            int = 0       # falloff contributions skipped (no Cantera)

    def f_worst(self) -> float:
        return max(self.f_uncertainty) if self.f_uncertainty else 1.0

    def f_mean(self) -> float:
        return sum(self.f_uncertainty) / len(self.f_uncertainty) if self.f_uncertainty else 1.0

    def sigma_lnk_at(self, T_target: float) -> Optional[float]:
        """Return sigma_lnk at the closest grid point to T_target."""
        if not self.T_K:
            return None
        idx = min(range(len(self.T_K)), key=lambda i: abs(self.T_K[i] - T_target))
        return self.sigma_lnk[idx]

    def to_dict(self) -> dict:
        return {
            "canonical_key":        self.canonical_key,
            "formula_string":       self.formula_string,
            "method":               self.method,
            "f_worst":              self.f_worst(),
            "f_mean":               self.f_mean(),
            "n_T_points":           len(self.T_K),
            "contributor_mech_ids": self.contributor_mech_ids,
        }


@dataclass
class ReactionDecision:
    """One row in the decisions.csv manifest."""
    canonical_key:         str
    formula_string:        str
    status:                str    # IN_ANCHOR|PATCH_SINGLE|PATCH_MULTI|SHARED|UNIQUE
    n_sources:             int
    source_mech_ids:       List[int]
    default_decision:      str
    decision:              str    # KEEP_ANCHOR|IMPORT_SINGLE|OPTIMIZE|PICK_SOURCE|DROP
    chosen_source_mech_id: Optional[int] = None
    notes:                 str = ""

    def to_dict(self) -> dict:
        return {
            "canonical_key":         self.canonical_key,
            "formula_string":        self.formula_string,
            "status":                self.status,
            "n_sources":             self.n_sources,
            "source_mech_ids":       json.dumps(self.source_mech_ids),
            "default_decision":      self.default_decision,
            "decision":              self.decision,
            "chosen_source_mech_id": self.chosen_source_mech_id,
            "notes":                 self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReactionDecision":
        src = d.get("source_mech_ids", "[]")
        if isinstance(src, str):
            try:
                src = json.loads(src)
            except (json.JSONDecodeError, ValueError):
                src = []
        csm = d.get("chosen_source_mech_id")
        if csm is not None and csm != "":
            try:
                csm = int(csm)
            except (ValueError, TypeError):
                csm = None
        else:
            csm = None
        return cls(
            canonical_key         = str(d.get("canonical_key", "")),
            formula_string        = str(d.get("formula_string", "")),
            status                = str(d.get("status", "")),
            n_sources             = int(d.get("n_sources", 0)),
            source_mech_ids       = [int(x) for x in src] if src else [],
            default_decision      = str(d.get("default_decision", "")),
            decision              = str(d.get("decision", "")),
            chosen_source_mech_id = csm,
            notes                 = str(d.get("notes", "")),
        )


@dataclass
class UnifiedTreePlan:
    """Output of UnifiedTreeBuilder.plan()."""
    mode:               str          # "anchored" | "optimize_all"
    anchor_mech_id:     Optional[int]
    anchor_provenance:  Optional[dict]
    decisions:          List[ReactionDecision]
    manifest_path:      str

    def to_dict(self) -> dict:
        return {
            "mode":              self.mode,
            "anchor_mech_id":    self.anchor_mech_id,
            "anchor_provenance": self.anchor_provenance,
            "n_decisions":       len(self.decisions),
            "manifest_path":     self.manifest_path,
        }


@dataclass
class UnifiedTreeResult:
    """Output of UnifiedTreeBuilder.build()."""
    unified_mechanism_id: int
    n_keep_anchor:        int
    n_import_single:      int
    n_optimize:           int
    n_pick_source:        int
    n_dropped:            int
    n_species:            int
    bands_csv_path:       str
    summary_csv_path:     str
    yaml_path:            str
    mso_bounds_path:      str
    provenance_path:      str

    @property
    def n_total(self) -> int:
        return (self.n_keep_anchor + self.n_import_single
                + self.n_optimize + self.n_pick_source + self.n_dropped)

    def summary(self) -> str:
        lines = [
            f"UnifiedTreeResult: mech_id={self.unified_mechanism_id}",
            f"  KEEP_ANCHOR   : {self.n_keep_anchor}",
            f"  IMPORT_SINGLE : {self.n_import_single}",
            f"  OPTIMIZE      : {self.n_optimize}",
            f"  PICK_SOURCE   : {self.n_pick_source}",
            f"  DROP          : {self.n_dropped}",
            f"  Species       : {self.n_species}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "unified_mechanism_id": self.unified_mechanism_id,
            "n_keep_anchor":        self.n_keep_anchor,
            "n_import_single":      self.n_import_single,
            "n_optimize":           self.n_optimize,
            "n_pick_source":        self.n_pick_source,
            "n_dropped":            self.n_dropped,
            "n_species":            self.n_species,
            "bands_csv_path":       self.bands_csv_path,
            "summary_csv_path":     self.summary_csv_path,
            "yaml_path":            self.yaml_path,
            "mso_bounds_path":      self.mso_bounds_path,
            "provenance_path":      self.provenance_path,
        }


@dataclass
class MissingSpeciesReport:
    """Output of MissingSpeciesPredictor.predict()."""
    unified_mech_id:      int
    contributing_mech_ids: List[int]

    # Cross-mechanism gaps (high-confidence: chemistry IS in some source mech)
    gap_species:   List[Dict[str, Any]]   # [{inchi_key, name, canonical_smiles, source_mech_id}]
    gap_reactions: List[str]              # canonical_keys

    # Topological gaps (heuristic from SparsityDetector)
    dead_end_species: List[str]   # inchi_keys (dead-end sources/sinks)
    orphan_species:   List[str]   # inchi_keys for orphan reactions' reactants

    # RMG seed lists
    seed_smiles:         List[str]   # expansion species
    must_include_smiles: List[str]   # inclusion list

    evidence: Dict[str, str]   # {inchi_key or canonical_key: explanation}

    def to_dict(self) -> dict:
        return {
            "unified_mech_id":       self.unified_mech_id,
            "contributing_mech_ids": self.contributing_mech_ids,
            "n_gap_species":         len(self.gap_species),
            "n_gap_reactions":       len(self.gap_reactions),
            "n_dead_end_species":    len(self.dead_end_species),
            "n_orphan_species":      len(self.orphan_species),
            "n_seed_smiles":         len(self.seed_smiles),
            "n_must_include":        len(self.must_include_smiles),
            "gap_species":           self.gap_species,
            "gap_reactions":         self.gap_reactions,
            "dead_end_species":      self.dead_end_species,
            "orphan_species":        self.orphan_species,
            "seed_smiles":           self.seed_smiles,
            "must_include_smiles":   self.must_include_smiles,
            "evidence":              self.evidence,
        }


@dataclass
class RMGSeedSpec:
    """Seed specification consumed by RMGInterface.generate()."""
    seed_smiles:           List[str]
    must_include_smiles:   List[str]
    source_unified_mech_id: int
    evidence:              Dict[str, str]

    def to_dict(self) -> dict:
        return {
            "seed_smiles":            self.seed_smiles,
            "must_include_smiles":    self.must_include_smiles,
            "source_unified_mech_id": self.source_unified_mech_id,
            "evidence":               self.evidence,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
