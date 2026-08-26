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
    PRIORITY = "PRIORITY"  # priority-fusion mode: verbatim from the highest-
                            # priority present source (2026-07-14 fusion session)


class Regime(str, Enum):
    """
    Temperature regime for IDT classification.

    LTC/NTC/HTC use the fixed 850/1000 K thresholds and remain the fallback
    when no NTC sweep has been run for a mechanism. HTT/LTT (Future.md pt. 7)
    use the mechanism-specific detected NTC onset instead — see
    ``analysis.ntc_detector.NTCDetector`` and ``analysis.benchmarking``,
    which prefers HTT/LTT over LTC/NTC/HTC whenever an ``ntc_runs`` row
    exists for the mechanism being scored.
    """
    LTC = "LTC"   # Low Temperature Chemistry  T < 850 K  (fixed-threshold fallback)
    NTC = "NTC"   # Negative Temperature Coefficient  850 ≤ T < 1000 K  (fixed-threshold fallback)
    HTC = "HTC"   # High Temperature Chemistry  T ≥ 1000 K  (fixed-threshold fallback)
    HTT = "HTT"   # High-Temperature Target  T > T_NTC_onset  (detected, per-mechanism)
    LTT = "LTT"   # Low-Temperature Target   T < T_NTC_onset  (detected, per-mechanism)
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
    """Per-reaction result from GraphComparator.

    occurrence_index (M9, review.md): CHEMKIN DUPLICATE reactions (two rows
    sharing one canonical_key -- a legitimate, common pattern, e.g. two
    Arrhenius expressions summed for one reaction) used to collide in every
    `{canonical_key: row}` map this pipeline builds, silently dropping all
    but the last-inserted duplicate. 0 for the first occurrence of a
    canonical_key within a mechanism, 1 for the second, etc. -- callers
    building lookup dicts must key on `(canonical_key, occurrence_index)`,
    not canonical_key alone. Cross-mechanism pairing for classification
    matches occurrence 0 in one mechanism to occurrence 0 in another (order
    of appearance in each mechanism's own REACTIONS block) -- not
    scientifically guaranteed to pair the "correct" duplicate term across
    sources, but strictly better than silently discarding one.
    """
    canonical_key:      str
    formula_string:     str
    classification:     ReactionClass
    present_in_mech_ids: List[int]
    occurrence_index:   int = 0
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
            "occurrence_index":    self.occurrence_index,
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

    # Reaction buckets: {(canonical_key, occurrence_index): ReactionClassification}
    # (M9, review.md: tuple key so CHEMKIN DUPLICATE reactions -- two rows
    # sharing one canonical_key -- don't collide and silently drop one.)
    verbatim_reactions: Dict[Tuple[str, int], ReactionClassification] = field(default_factory=dict)
    copy_reactions:     Dict[Tuple[str, int], ReactionClassification] = field(default_factory=dict)
    fuse_reactions:     Dict[Tuple[str, int], ReactionClassification] = field(default_factory=dict)

    # Mean/uncertainty params for FUSE reactions (set by GraphComparator)
    # {(canonical_key, occurrence_index): {
    #     "mean_A_si": float, "mean_n": float, "mean_Ea_si": float,
    #     "unc_A_si":  (min, max),
    #     "unc_n":     (min, max),
    #     "unc_Ea_si": (min, max),
    # }}
    fuse_params: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)

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
        # Reaction buckets are keyed by (canonical_key, occurrence_index) tuples
        # (M9 fix, see the field comments above) -- json.dump rejects non-str/
        # int/float/bool/None dict keys, so tuple keys are stringified here as
        # "canonical_key||occurrence_index" for this JSON export only. Nothing
        # reads union_graph_spec.json back in (it's a hand-off/inspection
        # artifact, confirmed via repo-wide grep), so this is a one-way format.
        def _str_key(k):
            return f"{k[0]}||{k[1]}" if isinstance(k, tuple) else k

        data = {
            "mechanism_ids":         self.mechanism_ids,
            "comparison_session_id": self.comparison_session_id,
            "species_map":    {k: v.to_dict() for k, v in self.species_map.items()},
            "verbatim_reactions": {_str_key(k): v.to_dict() for k, v in self.verbatim_reactions.items()},
            "copy_reactions":     {_str_key(k): v.to_dict() for k, v in self.copy_reactions.items()},
            "fuse_reactions":     {_str_key(k): v.to_dict() for k, v in self.fuse_reactions.items()},
            "fuse_params":        {_str_key(k): v for k, v in self.fuse_params.items()},
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
    """One row in the decisions.csv manifest.

    occurrence_index (M9, review.md): same convention as
    ReactionClassification.occurrence_index -- CHEMKIN DUPLICATE reactions
    (two rows sharing one canonical_key within a mechanism) used to collide
    in unified_tree.py's `{canonical_key: row}` source-row lookups, silently
    picking the same (last-read) row's kinetics for every decision that
    shares that canonical_key. Copied straight from the originating
    ReactionClassification at plan() time; callers building lookup dicts
    keyed on decisions must key on (canonical_key, occurrence_index), not
    canonical_key alone.
    """
    canonical_key:         str
    formula_string:        str
    status:                str    # IN_ANCHOR|PATCH_SINGLE|PATCH_MULTI|SHARED|UNIQUE
    n_sources:             int
    source_mech_ids:       List[int]
    default_decision:      str
    decision:              str    # KEEP_ANCHOR|IMPORT_SINGLE|OPTIMIZE|PICK_SOURCE|DROP
    chosen_source_mech_id: Optional[int] = None
    notes:                 str = ""
    occurrence_index:      int = 0

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
            "occurrence_index":      self.occurrence_index,
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
            # Absent in manifests written before this fix -- default to 0
            # (first/only occurrence), the correct value for the common
            # non-duplicated case.
            occurrence_index      = int(d.get("occurrence_index") or 0),
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


# ---------------------------------------------------------------------------
# Mechanism Reduction (Phase 5B)
# ---------------------------------------------------------------------------

@dataclass
class ReductionStep:
    """
    Retained/removed species and reactions at one ε threshold.

    Produced by DRGReducer, DRGEPReducer, and PFAReducer for each ε in the
    sweep range.  ``retained_*`` lists use species names / reaction
    formula_strings respectively — both are stored in the DB.

    ``bfs_levels`` maps species name → BFS propagation level (0 = seed,
    1 = first neighbours above ε, 2 = second neighbours, …).  Species
    not retained at this ε are absent from the dict.
    """
    threshold:           float
    retained_species:    List[str]   # species names in retained set
    retained_reactions:  List[str]   # formula_strings of retained reactions
    removed_species:     List[str]   # species names removed (= all - retained)
    removed_reactions:   List[str]   # formula_strings of removed reactions
    n_species:           int
    n_reactions:         int
    bfs_levels:          Optional[Dict[str, int]] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "threshold":          self.threshold,
            "retained_species":   self.retained_species,
            "retained_reactions": self.retained_reactions,
            "removed_species":    self.removed_species,
            "removed_reactions":  self.removed_reactions,
            "n_species":          self.n_species,
            "n_reactions":        self.n_reactions,
            "bfs_levels":         self.bfs_levels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReductionStep":
        return cls(
            threshold          = d["threshold"],
            retained_species   = d["retained_species"],
            retained_reactions = d["retained_reactions"],
            removed_species    = d["removed_species"],
            removed_reactions  = d["removed_reactions"],
            n_species          = d["n_species"],
            n_reactions        = d["n_reactions"],
            bfs_levels         = d.get("bfs_levels"),
        )


@dataclass
class ReductionResult:
    """
    Full multi-threshold reduction produced by a DRG/DRGEP/PFA/DRGPSA reducer.

    ``steps`` is ordered by ascending threshold; step 0 (ε = 0.0) retains
    everything; later steps progressively remove more species.

    Optional fields populated when ``--save-coefficients`` is used or when
    a ``DRGPSAReducer`` is run:
    - ``r_AB``: interaction coefficient matrix as {A: {B: r_AB}} (all pairs).
    - ``sa_coefficients``: SA coefficients {reaction_formula: |s_i|} used for
      SA-weighting in DRGPSA.  Absent for plain DRG/DRGEP/PFA.
    """
    mechanism_id:     int
    method:           str            # 'DRG', 'DRGEP', 'PFA', 'DRGPSA', 'DRGSA'
    target_species:   List[str]      # seed species specified by the user
    steps:            List[ReductionStep]
    run_id:           Optional[int]                         = None
    r_AB:             Optional[Dict[str, Dict[str, float]]] = field(default=None)
    sa_coefficients:  Optional[Dict[str, float]]            = field(default=None)

    def to_dict(self) -> dict:
        return {
            "mechanism_id":    self.mechanism_id,
            "method":          self.method,
            "target_species":  self.target_species,
            "steps":           [s.to_dict() for s in self.steps],
            "run_id":          self.run_id,
            "r_AB":            self.r_AB,
            "sa_coefficients": self.sa_coefficients,
        }

    def save(self, path: str) -> None:
        """Serialise to JSON file."""
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ReductionResult":
        return cls(
            mechanism_id    = d["mechanism_id"],
            method          = d["method"],
            target_species  = d["target_species"],
            steps           = [ReductionStep.from_dict(s) for s in d["steps"]],
            run_id          = d.get("run_id"),
            r_AB            = d.get("r_AB"),
            sa_coefficients = d.get("sa_coefficients"),
        )

    @classmethod
    def load(cls, path: str) -> "ReductionResult":
        """Deserialise from a JSON file written by :meth:`save`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Zero-RK GSA (ZERORK_GSA_PLAN.md Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class GSARunConfig:
    """
    Configuration for one ZerorkGSARunner screening run.

    ``reaction_ids`` is None for "every reaction in the mechanism" (the
    common case); a subset restricts the SALib parameter space to just
    those reactions' A-factors. ``factor_bound`` is the symmetric
    log-uniform A-factor multiplier bound (2.0 -> each reaction's A-factor
    sampled log-uniformly in [A/2, 2A]) -- matches Zero-RK's own
    perturbAFactor default 2x multiplier convention (see
    ZERORK_GSA_PLAN.md section 1.1.3).
    """
    mechanism_id:        int
    method:              str                    # 'sobol' | 'morris'
    n_samples:           int
    reaction_ids:        Optional[List[int]] = field(default=None)
    factor_bound:        float = 2.0
    T_K:                 float = 1000.0
    P_atm:               float = 1.0
    phi:                 float = 1.0
    fuel:                str = ""
    oxidizer:            str = ""
    idt_temp_delta:      float = 400.0
    n_mpi_ranks:         int = 4
    seed:                Optional[int] = None
    morris_num_levels:   int = 4

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "method":            self.method,
            "n_samples":         self.n_samples,
            "reaction_ids":      self.reaction_ids,
            "factor_bound":      self.factor_bound,
            "T_K":               self.T_K,
            "P_atm":             self.P_atm,
            "phi":               self.phi,
            "fuel":              self.fuel,
            "oxidizer":          self.oxidizer,
            "idt_temp_delta":    self.idt_temp_delta,
            "n_mpi_ranks":       self.n_mpi_ranks,
            "seed":              self.seed,
            "morris_num_levels": self.morris_num_levels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GSARunConfig":
        return cls(
            mechanism_id      = d["mechanism_id"],
            method            = d["method"],
            n_samples         = d["n_samples"],
            reaction_ids      = d.get("reaction_ids"),
            factor_bound      = d.get("factor_bound", 2.0),
            T_K               = d.get("T_K", 1000.0),
            P_atm             = d.get("P_atm", 1.0),
            phi               = d.get("phi", 1.0),
            fuel              = d.get("fuel", ""),
            oxidizer          = d.get("oxidizer", ""),
            idt_temp_delta    = d.get("idt_temp_delta", 400.0),
            n_mpi_ranks       = d.get("n_mpi_ranks", 4),
            seed              = d.get("seed"),
            morris_num_levels = d.get("morris_num_levels", 4),
        )


@dataclass
class SobolResult:
    """
    Output of ZerorkGSARunner's Sobol/Morris screening (Phase 3).

    Keyed by reaction ``parameter_names`` (canonical_key/reaction_label).
    For ``method='sobol'``: S1/S1_conf/ST/ST_conf are populated,
    mu_star/mu_star_conf/sigma are None. For ``method='morris'``: the
    reverse. This mirrors ``DoEFramework.run_screening()``'s documented
    return contract (``{canonical_key: {"S1": float, "S1_conf": float,
    ...}}``) -- see :meth:`to_screening_dict`.
    """
    mechanism_id:      int
    method:            str
    parameter_names:   List[str]
    n_samples:         int
    n_failed_samples:  int = 0
    S1:                Optional[Dict[str, float]] = field(default=None)
    S1_conf:           Optional[Dict[str, float]] = field(default=None)
    ST:                Optional[Dict[str, float]] = field(default=None)
    ST_conf:           Optional[Dict[str, float]] = field(default=None)
    mu_star:           Optional[Dict[str, float]] = field(default=None)
    mu_star_conf:      Optional[Dict[str, float]] = field(default=None)
    sigma:             Optional[Dict[str, float]] = field(default=None)
    run_id:            Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mechanism_id":     self.mechanism_id,
            "method":           self.method,
            "parameter_names":  self.parameter_names,
            "n_samples":        self.n_samples,
            "n_failed_samples": self.n_failed_samples,
            "S1":               self.S1,
            "S1_conf":          self.S1_conf,
            "ST":               self.ST,
            "ST_conf":          self.ST_conf,
            "mu_star":          self.mu_star,
            "mu_star_conf":     self.mu_star_conf,
            "sigma":            self.sigma,
            "run_id":           self.run_id,
        }

    def to_screening_dict(self) -> Dict[str, Dict[str, float]]:
        """Match DoEFramework.run_screening()'s per-parameter dict shape."""
        out: Dict[str, Dict[str, float]] = {}
        for name in self.parameter_names:
            entry: Dict[str, float] = {}
            for field_name in ("S1", "S1_conf", "ST", "ST_conf",
                               "mu_star", "mu_star_conf", "sigma"):
                values = getattr(self, field_name)
                if values is not None and name in values:
                    entry[field_name] = values[name]
            out[name] = entry
        return out

    def save(self, path: str) -> None:
        """Serialise to JSON file."""
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SobolResult":
        return cls(
            mechanism_id     = d["mechanism_id"],
            method           = d["method"],
            parameter_names  = d["parameter_names"],
            n_samples        = d["n_samples"],
            n_failed_samples = d.get("n_failed_samples", 0),
            S1               = d.get("S1"),
            S1_conf          = d.get("S1_conf"),
            ST               = d.get("ST"),
            ST_conf          = d.get("ST_conf"),
            mu_star          = d.get("mu_star"),
            mu_star_conf     = d.get("mu_star_conf"),
            sigma            = d.get("sigma"),
            run_id           = d.get("run_id"),
        )

    @classmethod
    def load(cls, path: str) -> "SobolResult":
        """Deserialise from a JSON file written by :meth:`save`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class SurrogateResult:
    """
    Output of PCESurrogateBuilder (Phase 5, Future.md pt. 7): a generalized
    Polynomial-Chaos (gPC) surrogate of a scalar QoI over uncertain reaction
    A-factors, plus the variance decomposition and gradient-based active
    subspace derived from it.

    ``parameter_names`` are the perturbed reactions (Cantera equation strings).
    ``sobol_main`` / ``sobol_total`` are the analytic first-order / total Sobol
    indices read off the PCE coefficients (not Monte-Carlo estimated). The
    active subspace is the eigendecomposition of C = E[∇f ∇fᵀ] in standardised
    input space: ``as_eigenvalues`` (descending), ``as_eigenvectors`` (columns
    are directions, row i = parameter i), ``as_activity_scores`` (per-parameter
    Σ λ_j W_ij²), and ``as_n_active`` (directions before the largest spectral
    gap).
    """
    mechanism_id:        int
    qoi:                 str                 # 'idt' | 'lfs' | 'custom'
    order:               int
    parameter_names:     List[str]
    n_samples:           int
    n_failed_samples:    int = 0
    n_terms:             int = 0
    r2_insample:         float = 0.0
    qoi_mean:            Optional[float] = None
    qoi_variance:        Optional[float] = None
    sobol_main:          Dict[str, float] = field(default_factory=dict)
    sobol_total:         Dict[str, float] = field(default_factory=dict)
    as_eigenvalues:      List[float] = field(default_factory=list)
    as_eigenvectors:     List[List[float]] = field(default_factory=list)
    as_activity_scores:  Dict[str, float] = field(default_factory=dict)
    as_n_active:         int = 0
    condition:           Optional[SimCondition] = None
    log_transformed:     bool = True         # QoI fitted in log-space (IDT/LFS)
    # Portable monomial form of the PCE (evaluable in pure numpy outside the
    # chaospy venv): f(x) = Σ_k pce_coefficients[k] · Π_i x_i^pce_exponents[k][i].
    # Feeds Bayesian/MUM-PCE calibration as the cheap forward model (pt. 6).
    pce_exponents:       List[List[int]] = field(default_factory=list)
    pce_coefficients:    List[float] = field(default_factory=list)
    run_id:              Optional[int] = None

    def ranked_parameters(self) -> List[tuple]:
        """``[(name, total_sobol), ...]`` sorted by total Sobol index desc."""
        return sorted(self.sobol_total.items(), key=lambda kv: -kv[1])

    def to_dict(self) -> dict:
        return {
            "mechanism_id":       self.mechanism_id,
            "qoi":                self.qoi,
            "order":              self.order,
            "parameter_names":    self.parameter_names,
            "n_samples":          self.n_samples,
            "n_failed_samples":   self.n_failed_samples,
            "n_terms":            self.n_terms,
            "r2_insample":        self.r2_insample,
            "qoi_mean":           self.qoi_mean,
            "qoi_variance":       self.qoi_variance,
            "sobol_main":         self.sobol_main,
            "sobol_total":        self.sobol_total,
            "as_eigenvalues":     self.as_eigenvalues,
            "as_eigenvectors":    self.as_eigenvectors,
            "as_activity_scores": self.as_activity_scores,
            "as_n_active":        self.as_n_active,
            "condition":          self.condition.to_dict() if self.condition else None,
            "log_transformed":    self.log_transformed,
            "pce_exponents":      self.pce_exponents,
            "pce_coefficients":   self.pce_coefficients,
            "run_id":             self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class CalibrationResult:
    """
    Output of the MUM-PCE / Bayesian calibrator (Phase 6, Future.md pt. 6).

    A posterior over the uncertain parameters (log-A multipliers, same space as
    the PCE surrogates that serve as the cheap forward model) obtained by fitting
    experimental targets. ``posterior_mean`` is the MAP estimate; ``posterior_sigma``
    / ``posterior_cov`` are the Laplace (Gauss-Newton) posterior spread. Prior vs
    posterior widths give the per-parameter ``uncertainty_reduction``. The
    ``mcmc_*`` fields are populated only when the optional emcee sampler is run.
    """
    parameter_names:      List[str]
    n_targets:            int
    prior_sigma:          List[float]
    posterior_mean:       List[float]         # MAP, in log-multiplier space
    posterior_sigma:      List[float]
    posterior_cov:        List[List[float]]
    prior_factor:         List[float]         # exp(2σ_prior): ±2σ uncertainty factor
    posterior_factor:     List[float]
    uncertainty_reduction: List[float]        # 1 − σ_post/σ_prior, per parameter
    chi2:                 float
    reduced_chi2:         float
    target_names:         List[str] = field(default_factory=list)
    observed:             List[float] = field(default_factory=list)
    predicted_prior:      List[float] = field(default_factory=list)   # at θ=0
    predicted_map:        List[float] = field(default_factory=list)   # at MAP
    obs_sigma:            List[float] = field(default_factory=list)
    converged:            bool = True
    mechanism_id:         Optional[int] = None
    # Optional full-Bayes (emcee) posterior, when run
    mcmc_mean:            Optional[List[float]] = None
    mcmc_sigma:           Optional[List[float]] = None
    mcmc_n_samples:       Optional[int] = None
    run_id:               Optional[int] = None

    def ranked_uncertainty_reduction(self) -> List[tuple]:
        return sorted(zip(self.parameter_names, self.uncertainty_reduction),
                      key=lambda kv: -kv[1])

    def to_dict(self) -> dict:
        return {
            "parameter_names":       self.parameter_names,
            "n_targets":             self.n_targets,
            "prior_sigma":           self.prior_sigma,
            "posterior_mean":        self.posterior_mean,
            "posterior_sigma":       self.posterior_sigma,
            "posterior_cov":         self.posterior_cov,
            "prior_factor":          self.prior_factor,
            "posterior_factor":      self.posterior_factor,
            "uncertainty_reduction": self.uncertainty_reduction,
            "chi2":                  self.chi2,
            "reduced_chi2":          self.reduced_chi2,
            "target_names":          self.target_names,
            "observed":              self.observed,
            "predicted_prior":       self.predicted_prior,
            "predicted_map":         self.predicted_map,
            "obs_sigma":             self.obs_sigma,
            "converged":             self.converged,
            "mechanism_id":          self.mechanism_id,
            "mcmc_mean":             self.mcmc_mean,
            "mcmc_sigma":            self.mcmc_sigma,
            "mcmc_n_samples":        self.mcmc_n_samples,
            "run_id":                self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class B2BDCResult:
    """
    Output of Bound-to-Bound Data Collaboration feasible-set analysis
    (Phase 7, Future.md pt. 5).

    The dataset couples PCE surrogates ``M_e(x)`` with experimental bounds
    ``[L_e, U_e]`` over the prior parameter box ``H``. ``consistency_measure`` is
    the scalar C = max_{x∈H} min_e (1 − |r_e(x)|), r_e the normalised model-minus-
    data residual: C > 0 ⇒ the collection of experiments+model is self-consistent
    (feasible set non-empty, with margin C); C < 0 ⇒ inconsistent by |C|.

    All bounds here are **inner** (local-optimisation) bounds: a positive
    ``consistency_measure`` with a feasible point is a rigorous certificate of
    consistency, and every reported parameter/prediction interval lies *inside*
    the true feasible-set range. Rigorous SDP **outer** bounds (quadratic-
    surrogate S-procedure) are a documented follow-on.
    """
    n_constraints:        int
    parameter_names:      List[str]
    consistency_measure:  float
    consistent:           bool
    feasible_point:       List[float]
    constraint_names:     List[str]
    constraint_margins:   List[float]                    # 1−|r_e| at feasible_point
    prior_bounds:         Dict[str, List[float]]         # name → [lo, hi] of box H
    parameter_bounds:     Dict[str, List[float]] = field(default_factory=dict)
    parameter_reduction:  Dict[str, float] = field(default_factory=dict)  # 1−post/prior width
    predictions:          List[Dict[str, Any]] = field(default_factory=list)  # [{name,lower,upper}]
    n_starts:             int = 1
    bound_type:           str = "inner"
    mechanism_id:         Optional[int] = None
    notes:                List[str] = field(default_factory=list)
    run_id:               Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "n_constraints":       self.n_constraints,
            "parameter_names":     self.parameter_names,
            "consistency_measure": self.consistency_measure,
            "consistent":          self.consistent,
            "feasible_point":      self.feasible_point,
            "constraint_names":    self.constraint_names,
            "constraint_margins":  self.constraint_margins,
            "prior_bounds":        self.prior_bounds,
            "parameter_bounds":    self.parameter_bounds,
            "parameter_reduction": self.parameter_reduction,
            "predictions":         self.predictions,
            "n_starts":            self.n_starts,
            "bound_type":          self.bound_type,
            "mechanism_id":        self.mechanism_id,
            "notes":               self.notes,
            "run_id":              self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class CSPResult:
    """
    Output of Computational Singular Perturbation timescale analysis
    (Phase 8, FEATURE_ROADMAP #10).

    A condition-specific diagnostic: at the evaluated state the isothermal
    chemical Jacobian is eigendecomposed into modes with timescales
    ``mode_timescales`` [s]; modes faster than ``tau_threshold`` are *exhausted*
    (``n_exhausted_modes``).  ``qss_candidates`` lists the quasi-steady-state
    species (chemical lifetime below the threshold), each annotated with its
    lifetime, QSS defect (creation/destruction imbalance) and the exhausted CSP
    mode that points to it — the principled input for QSSA elimination / lumping.
    """
    mechanism_id:      Optional[int]
    condition:         Optional["SimCondition"]
    eval_state:        str
    state_T_K:         float
    n_species:         int
    n_reactions:       int
    jacobian_analytic: bool
    tau_threshold:     float
    fastest_tau:       Optional[float]
    slowest_tau:       Optional[float]
    n_exhausted_modes: int
    mode_timescales:   List[float] = field(default_factory=list)
    qss_candidates:    List[Dict[str, Any]] = field(default_factory=list)
    n_qss:             int = 0
    excluded_species:  List[str] = field(default_factory=list)
    notes:             List[str] = field(default_factory=list)
    cpu_time_s:        float = 0.0
    run_id:            Optional[int] = None

    def qss_species(self) -> List[str]:
        return [q["species"] for q in self.qss_candidates]

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "condition":         self.condition.to_dict() if self.condition else None,
            "eval_state":        self.eval_state,
            "state_T_K":         self.state_T_K,
            "n_species":         self.n_species,
            "n_reactions":       self.n_reactions,
            "jacobian_analytic": self.jacobian_analytic,
            "tau_threshold":     self.tau_threshold,
            "fastest_tau":       self.fastest_tau,
            "slowest_tau":       self.slowest_tau,
            "n_exhausted_modes": self.n_exhausted_modes,
            "mode_timescales":   self.mode_timescales,
            "qss_candidates":    self.qss_candidates,
            "n_qss":             self.n_qss,
            "excluded_species":  self.excluded_species,
            "notes":             self.notes,
            "cpu_time_s":        self.cpu_time_s,
            "run_id":            self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class LumpingResult:
    """
    Output of a-priori thermodynamic chemical/isomer lumping
    (Phase 8, FEATURE_ROADMAP #10).

    ``groups`` are proposed lumps of thermochemically-similar isomers (same
    elemental composition, ΔH_f° / S° within the given tolerances), each with its
    members, most-stable representative and thermochemical spread.  Lumping them
    removes ``n_species_removed`` species (``n_species_full`` → ``n_species_lumped``).
    """
    mechanism_id:      Optional[int]
    method:            str
    dHf_tol_kJ:        float
    dS_tol:            float
    min_atoms:         int
    n_species_full:    int
    n_lumps:           int
    n_species_removed: int
    n_species_lumped:  int
    groups:            List[Dict[str, Any]] = field(default_factory=list)
    excluded_species:  List[str] = field(default_factory=list)
    notes:             List[str] = field(default_factory=list)
    cpu_time_s:        float = 0.0
    run_id:            Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "method":            self.method,
            "dHf_tol_kJ":        self.dHf_tol_kJ,
            "dS_tol":            self.dS_tol,
            "min_atoms":         self.min_atoms,
            "n_species_full":    self.n_species_full,
            "n_lumps":           self.n_lumps,
            "n_species_removed": self.n_species_removed,
            "n_species_lumped":  self.n_species_lumped,
            "groups":            self.groups,
            "excluded_species":  self.excluded_species,
            "notes":             self.notes,
            "cpu_time_s":        self.cpu_time_s,
            "run_id":            self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class ErrorMapResult:
    """
    Skeletal-vs-detailed error atlas over a (T, P, φ) grid (FEATURE_ROADMAP #11a).

    Each cell compares a reduced/skeletal mechanism's QoI (IDT or LFS) against the
    detailed mechanism (the reference).  ``cells`` carries the per-condition
    relative error ``|skeletal − detailed|/detailed`` and log error
    ``|log10(skeletal/detailed)|``; the summary flags the worst cell and the
    fraction of the design space within ``tolerance`` — i.e. *where the skeletal
    model is trustworthy and where it breaks*.
    """
    detailed_mech_id: int
    skeletal_mech_id: int
    sim_type:         str
    fuel:             str
    oxidizer:         str
    tolerance:        float
    n_conditions:     int
    n_evaluated:      int                                  # both mechs converged
    cells:            List[Dict[str, Any]] = field(default_factory=list)
    max_rel_err:      Optional[float] = None
    mean_rel_err:     Optional[float] = None
    max_log_err:      Optional[float] = None
    frac_within_tol:  Optional[float] = None
    worst_cell:       Optional[Dict[str, Any]] = None
    notes:            List[str] = field(default_factory=list)
    run_id:           Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "detailed_mech_id": self.detailed_mech_id,
            "skeletal_mech_id": self.skeletal_mech_id,
            "sim_type":         self.sim_type,
            "fuel":             self.fuel,
            "oxidizer":         self.oxidizer,
            "tolerance":        self.tolerance,
            "n_conditions":     self.n_conditions,
            "n_evaluated":      self.n_evaluated,
            "cells":            self.cells,
            "max_rel_err":      self.max_rel_err,
            "mean_rel_err":     self.mean_rel_err,
            "max_log_err":      self.max_log_err,
            "frac_within_tol":  self.frac_within_tol,
            "worst_cell":       self.worst_cell,
            "notes":            self.notes,
            "run_id":           self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class ParetoResult:
    """
    Output of multi-objective (NSGA-II) mechanism optimization (FEATURE_ROADMAP #9).

    ``front`` is the set of non-dominated parameter sets — each a dict of
    ``params`` (reaction log-A multipliers) and its ``objectives`` vector — i.e.
    the trade-off surface between the objectives (no single point is better on all
    of them). ``hypervolume`` (2-objective) summarises the front's quality.
    """
    mechanism_id:      Optional[int]
    parameter_names:   List[str]
    objective_names:   List[str]
    objective_run_ids: List[int]
    objective_targets: List[float]
    n_generations:     int
    pop_size:          int
    n_front:           int
    front:             List[Dict[str, Any]] = field(default_factory=list)
    hypervolume:       Optional[float] = None
    notes:             List[str] = field(default_factory=list)
    run_id:            Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "parameter_names":   self.parameter_names,
            "objective_names":   self.objective_names,
            "objective_run_ids": self.objective_run_ids,
            "objective_targets": self.objective_targets,
            "n_generations":     self.n_generations,
            "pop_size":          self.pop_size,
            "n_front":           self.n_front,
            "front":             self.front,
            "hypervolume":       self.hypervolume,
            "notes":             self.notes,
            "run_id":            self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class DoEResult:
    """
    Output of optimal experiment design (FEATURE_ROADMAP #8).

    Ranks/selects candidate experiments (each a stored gPC surrogate of its QoI)
    by the Fisher information they add to the calibration. ``selected`` is the
    greedy A-/D-optimal set in selection order with per-step ``gain``;
    ``ranked_candidates`` is every candidate's first-step marginal gain. The
    prior→posterior precision (``logdet_*``) and covariance (``trace_cov_*``)
    quantify the joint uncertainty shrinkage the selection buys.
    """
    mechanism_id:      Optional[int]
    parameter_names:   List[str]
    criterion:         str
    prior_sigma:       List[float]
    n_candidates:      int
    n_selected:        int
    selected:          List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    posterior_sigma:   List[float] = field(default_factory=list)
    uncertainty_reduction: List[float] = field(default_factory=list)
    logdet_prior:      float = 0.0
    logdet_post:       float = 0.0
    trace_cov_prior:   float = 0.0
    trace_cov_post:    float = 0.0
    candidate_run_ids: List[int] = field(default_factory=list)
    notes:             List[str] = field(default_factory=list)
    run_id:            Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "parameter_names":   self.parameter_names,
            "criterion":         self.criterion,
            "prior_sigma":       self.prior_sigma,
            "n_candidates":      self.n_candidates,
            "n_selected":        self.n_selected,
            "selected":          self.selected,
            "ranked_candidates": self.ranked_candidates,
            "posterior_sigma":   self.posterior_sigma,
            "uncertainty_reduction": self.uncertainty_reduction,
            "logdet_prior":      self.logdet_prior,
            "logdet_post":       self.logdet_post,
            "trace_cov_prior":   self.trace_cov_prior,
            "trace_cov_post":    self.trace_cov_post,
            "candidate_run_ids": self.candidate_run_ids,
            "notes":             self.notes,
            "run_id":            self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


@dataclass
class TDACResult:
    """
    Output of a tabulated dynamic adaptive chemistry batch integration
    (FEATURE_ROADMAP #11b).

    A constant-UV reactor is integrated while, at a set cadence, the locally
    active species set is recomputed by instantaneous DRGEP around the target
    species and the reduced chemistry is used for the next step.  Recurring
    thermochemical states reuse a tabulated active-set (the "T" in TDAC), so the
    result reports both accuracy (``idt_s`` vs the full mechanism) and efficiency
    (``mean_active_species`` below the full count, ``hit_rate`` > 0).
    """
    mechanism_id:        Optional[int]
    condition:           Optional["SimCondition"]
    eps:                 float
    update_every:        int
    bin_T:               float
    n_species_full:      int
    n_reactions_full:    int
    idt_s:               Optional[float]
    converged:           bool
    n_updates:           int = 0
    n_cache_hits:        int = 0
    n_cache_builds:      int = 0
    mean_active_species: Optional[float] = None
    min_active_species:  Optional[int] = None
    max_active_species:  Optional[int] = None
    active_trace:        List[Dict[str, Any]] = field(default_factory=list)
    target_species:      List[str] = field(default_factory=list)
    notes:               List[str] = field(default_factory=list)
    cpu_time_s:          float = 0.0
    run_id:              Optional[int] = None

    @property
    def hit_rate(self) -> Optional[float]:
        tot = self.n_cache_hits + self.n_cache_builds
        return (self.n_cache_hits / tot) if tot else None

    @property
    def mean_active_fraction(self) -> Optional[float]:
        if self.mean_active_species is None or not self.n_species_full:
            return None
        return self.mean_active_species / self.n_species_full

    def to_dict(self) -> dict:
        return {
            "mechanism_id":        self.mechanism_id,
            "condition":           self.condition.to_dict() if self.condition else None,
            "eps":                 self.eps,
            "update_every":        self.update_every,
            "bin_T":               self.bin_T,
            "n_species_full":      self.n_species_full,
            "n_reactions_full":    self.n_reactions_full,
            "idt_s":               self.idt_s,
            "converged":           self.converged,
            "n_updates":           self.n_updates,
            "n_cache_hits":        self.n_cache_hits,
            "n_cache_builds":      self.n_cache_builds,
            "hit_rate":            self.hit_rate,
            "mean_active_species": self.mean_active_species,
            "mean_active_fraction": self.mean_active_fraction,
            "min_active_species":  self.min_active_species,
            "max_active_species":  self.max_active_species,
            "active_trace":        self.active_trace,
            "target_species":      self.target_species,
            "notes":               self.notes,
            "cpu_time_s":          self.cpu_time_s,
            "run_id":              self.run_id,
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# ---------------------------------------------------------------------------
# Phase 6 — Future.md features 1–7
# ---------------------------------------------------------------------------

class SolverName(str, Enum):
    """The three solver backends usable for JSR/PFR/flux targets (Future.md pt. 3/6)."""
    CANTERA     = "cantera"
    CHEMKIN     = "CHEMKIN_PRO"
    FLAMEMASTER = "FlameMaster"


class TempTargetClass(str, Enum):
    """1/T-scale classification relative to the detected NTC onset (Future.md pt. 7)."""
    HTT = "HTT"   # T > T_NTC_onset — high-temperature target
    LTT = "LTT"   # T < T_NTC_onset — low-temperature target


@dataclass
class JSRResult:
    """Result of a jet-stirred (perfectly-stirred) reactor simulation — Future.md pt. 6."""
    condition:              SimCondition
    mechanism_id:           int
    solver:                 str                       # 'cantera'|'CHEMKIN_PRO'|'FlameMaster'
    residence_time_s:       float
    species_mole_fractions: Dict[str, float]          # steady-state exit composition
    converged:              bool
    cpu_time_s:             Optional[float] = None
    error_msg:              Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            **self.condition.to_dict(),
            "mechanism_id":           self.mechanism_id,
            "solver":                 self.solver,
            "residence_time_s":       self.residence_time_s,
            "species_mole_fractions": self.species_mole_fractions,
            "converged":              self.converged,
            "cpu_time_s":             self.cpu_time_s,
            "error_msg":              self.error_msg,
        }


@dataclass
class PFRResult:
    """Result of a plug-flow reactor simulation — Future.md pt. 6."""
    condition:              SimCondition
    mechanism_id:           int
    solver:                 str
    length_m:               float
    residence_time_s:       Optional[float]
    species_mole_fractions: Dict[str, float]          # exit composition
    axial_profile:          Optional[Dict[str, List[float]]] = None  # 'position_m','T_K',<species>
    converged:              bool = True
    cpu_time_s:             Optional[float] = None
    error_msg:              Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            **self.condition.to_dict(),
            "mechanism_id":           self.mechanism_id,
            "solver":                 self.solver,
            "length_m":               self.length_m,
            "residence_time_s":       self.residence_time_s,
            "species_mole_fractions": self.species_mole_fractions,
            "has_axial_profile":      self.axial_profile is not None,
            "converged":              self.converged,
            "cpu_time_s":             self.cpu_time_s,
            "error_msg":              self.error_msg,
        }


@dataclass
class FluxResult:
    """
    Reaction-pathway flux analysis result (Future.md pt. 3).

    ``rop_by_reaction`` maps canonical_key → integrated (or instantaneous)
    rate of progress [mol/cm3/s or solver-native units].  ``element_flux``
    holds the per-element (C/H/O) Sankey-ready edges: each entry is
    ``{"source": species, "target": species, "reaction": canonical_key,
    "value": flux_magnitude}``.
    """
    mechanism_id:    int
    condition:       SimCondition
    solver:          str
    element:         str                         # 'C' | 'H' | 'O'
    rop_by_reaction: Dict[str, float]
    element_flux:    List[Dict[str, Any]]
    converged:       bool
    cpu_time_s:      Optional[float] = None
    error_msg:       Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            "mechanism_id":    self.mechanism_id,
            **self.condition.to_dict(),
            "solver":          self.solver,
            "element":         self.element,
            "n_reactions":     len(self.rop_by_reaction),
            "n_flux_edges":    len(self.element_flux),
            "converged":       self.converged,
            "cpu_time_s":      self.cpu_time_s,
            "error_msg":       self.error_msg,
        }


@dataclass
class ReverseRateFlag:
    """
    A forward-only reaction whose Keq-derived reverse rate exceeds the
    collision limit at some temperature (Future.md pt. 1).
    """
    canonical_key:      str
    formula_string:     str
    mechanism_id:       int
    T_K:                float
    k_fwd:              float
    k_rev:              float
    collision_limit:    float
    exceedance_factor:  float     # k_rev / collision_limit
    flagged:            bool

    def to_dict(self) -> dict:
        return {
            "canonical_key":     self.canonical_key,
            "formula_string":    self.formula_string,
            "mechanism_id":      self.mechanism_id,
            "T_K":               self.T_K,
            "k_fwd":             self.k_fwd,
            "k_rev":             self.k_rev,
            "collision_limit":   self.collision_limit,
            "exceedance_factor": self.exceedance_factor,
            "flagged":           self.flagged,
        }


@dataclass
class QualityScore:
    """Composite mechanism 'trust index' (Future.md pt. 2, merged with pt. 4)."""
    mechanism_id:               int
    mechanism_name:             str
    smiles_resolution_pct:      float
    thermo_consistency_score:   float             # 0-1, Tmid continuity based
    hf_agreement_score:         Optional[float]   # 0-1 vs ATcT/JANAF reference; None if no reference
    ntc_coverage_score:         Optional[float]   # 0-1; None if NTC sweep not run
    rate_bounds_score:          float             # 0-1, fraction of reactions within collision limit
    composite_score:            float             # 0-100 weighted overall
    n_reactions:                int
    n_species:                  int
    n_reverse_rate_flags:       int
    details:                    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mechanism_id":             self.mechanism_id,
            "mechanism_name":           self.mechanism_name,
            "smiles_resolution_pct":    self.smiles_resolution_pct,
            "thermo_consistency_score": self.thermo_consistency_score,
            "hf_agreement_score":       self.hf_agreement_score,
            "ntc_coverage_score":       self.ntc_coverage_score,
            "rate_bounds_score":        self.rate_bounds_score,
            "composite_score":          self.composite_score,
            "n_reactions":              self.n_reactions,
            "n_species":                self.n_species,
            "n_reverse_rate_flags":     self.n_reverse_rate_flags,
            "details":                  self.details,
        }


@dataclass
class MechanismDiffResult:
    """Two-mechanism diff report (Future.md pt. 4, merged into pt. 2)."""
    mechanism_id_a:            int
    mechanism_id_b:            int
    mechanism_name_a:          str
    mechanism_name_b:          str
    species_added:             List[str]             # inchi_keys in B not in A
    species_removed:           List[str]             # inchi_keys in A not in B
    reactions_added:           List[str]             # canonical_keys in B not in A
    reactions_removed:         List[str]             # canonical_keys in A not in B
    shared_reaction_log_a_ratio: Dict[str, float]     # canonical_key -> log10(A_B/A_A)
    thermo_deltas:             List[Dict[str, Any]]   # per shared species Cp/H/S deltas
    n_species_shared:          int
    n_reactions_shared:        int

    def to_dict(self) -> dict:
        return {
            "mechanism_id_a":              self.mechanism_id_a,
            "mechanism_id_b":              self.mechanism_id_b,
            "mechanism_name_a":            self.mechanism_name_a,
            "mechanism_name_b":            self.mechanism_name_b,
            "n_species_added":             len(self.species_added),
            "n_species_removed":           len(self.species_removed),
            "n_reactions_added":           len(self.reactions_added),
            "n_reactions_removed":         len(self.reactions_removed),
            "species_added":               self.species_added,
            "species_removed":             self.species_removed,
            "reactions_added":             self.reactions_added,
            "reactions_removed":           self.reactions_removed,
            "shared_reaction_log_a_ratio": self.shared_reaction_log_a_ratio,
            "thermo_deltas":               self.thermo_deltas,
            "n_species_shared":            self.n_species_shared,
            "n_reactions_shared":          self.n_reactions_shared,
        }


@dataclass
class PlogSurface:
    """log k(T,P) surface for a single PLOG reaction (Future.md pt. 5)."""
    canonical_key:            str
    formula_string:           str
    mechanism_id:             int
    T_grid_K:                 List[float]
    P_grid_atm:               List[float]
    log10_k_matrix:           List[List[float]]     # shape [len(T_grid)][len(P_grid)]
    non_monotonic_T:          List[float]            # T values where k(P) is non-monotonic
    high_p_limit_ill_defined: bool
    warnings:                 List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canonical_key":            self.canonical_key,
            "formula_string":           self.formula_string,
            "mechanism_id":             self.mechanism_id,
            "n_T":                      len(self.T_grid_K),
            "n_P":                      len(self.P_grid_atm),
            "non_monotonic_T":          self.non_monotonic_T,
            "high_p_limit_ill_defined": self.high_p_limit_ill_defined,
            "warnings":                 self.warnings,
        }


@dataclass
class NTCReport:
    """
    NTC (negative temperature coefficient) detection + HTT/LTT split
    (Future.md pt. 7, extended per KP: replaces fixed LTC/NTC/HTC thresholds).
    """
    mechanism_id:     int
    fuel:             str
    phi:              float
    P_atm:            float
    T_grid_K:         List[float]
    idt_s:            List[Optional[float]]
    has_ntc:          bool
    T_NTC_onset:      Optional[float]      # K; start of the NTC region (1/T scale)
    T_NTC_end:        Optional[float]      # K; end of the NTC region
    IDT_ratio_NTC:    Optional[float]      # IDT(T_end) / IDT(T_onset)
    htt_points_K:     List[float] = field(default_factory=list)   # T > T_NTC_onset
    ltt_points_K:     List[float] = field(default_factory=list)   # T < T_NTC_onset

    def classify(self, T_K: float) -> Optional[str]:
        """Classify a single temperature as HTT/LTT relative to T_NTC_onset."""
        if self.T_NTC_onset is None:
            return None
        return TempTargetClass.HTT.value if T_K > self.T_NTC_onset else TempTargetClass.LTT.value

    def to_dict(self) -> dict:
        return {
            "mechanism_id":  self.mechanism_id,
            "fuel":          self.fuel,
            "phi":           self.phi,
            "P_atm":         self.P_atm,
            "T_grid_K":      self.T_grid_K,
            "idt_s":         self.idt_s,
            "has_ntc":       self.has_ntc,
            "T_NTC_onset":   self.T_NTC_onset,
            "T_NTC_end":     self.T_NTC_end,
            "IDT_ratio_NTC": self.IDT_ratio_NTC,
            "n_htt_points":  len(self.htt_points_K),
            "n_ltt_points":  len(self.ltt_points_K),
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _Path
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
