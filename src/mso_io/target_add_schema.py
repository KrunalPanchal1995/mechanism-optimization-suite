"""target_add_schema.py -- option-surface inventory for the MUQ-SAC
``target.opt`` / ``.target`` / ``.add`` config trio.

Single source of truth for both the Tkinter (brute_force_sa_gui.py,
idt_sensitivity_gui.py, nominal_sim_gui.py) and PySide6
(src/mso_studio/gui/modules/sensitivity.py) GUIs, so the option surface a
form exposes can't drift between the two stacks.

Two engine copies of ``make_input_file.py``/``combustion_target_class.py``
exist in this repo: ``src/MUQ-SAC/`` (fuller -- has PFR, all 4 solvers incl.
zerork, flux_analysis) and ``src/Sensitivity_Analysis/`` (the one
``sens_3_params.py`` actually imports and runs, missing those additions).
Per KRUNAL_JULY_23.md point 4, the GUI exposes the *full* MUQ-SAC surface
for authoring ``.target``/``.add`` files, but gates the Run action for any
solver/target-type combo the live engine can't execute -- see
``is_runnable()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Enumerated option surface (src/MUQ-SAC/make_input_file.py, full inventory)
# ---------------------------------------------------------------------------

SOLVERS: List[str] = ["cantera", "CHEMKIN_PRO", "zerork", "FlameMaster"]

TARGET_TYPES: List[str] = ["Tig", "JSR", "PFR", "Flf", "Flf_a", "RCM", "Fls", "Flw"]

# What src/Sensitivity_Analysis/{make_input_file,combustion_target_class}.py
# -- the copies sens_3_params.py / simulation_manager2_0.py actually import
# -- can run today (bug.md / KRUNAL_JULY_23.md point 4 investigation).
# FlameMaster is the original/default solver (combustion_target_class sets
# it as the fallback when no .add entry exists at all) -- fully supported by
# the live engine. Only "zerork" is the MUQ-SAC-only addition among solvers.
LIVE_ENGINE_SOLVERS = {"cantera", "CHEMKIN_PRO", "FlameMaster"}
LIVE_ENGINE_TARGET_TYPES = {"Tig", "JSR", "Flf", "Flf_a", "RCM", "Fls", "Flw"}  # no PFR

IGN_COND_CHOICES = ["max", "onset", "dt-max", "specific"]
IGN_DELAY_DEF_EXAMPLES = ["OH", "OH*", "NH3", "p", "CHV"]
RCM_TYPE_CHOICES = ["reactive", "non_reactive"]
FUEL_TYPE_CHOICES = ["Mono", "Multi"]
FLS_TYPE_CHOICES = ["N/A", "phi"]
TRANSPORT_MODEL_CHOICES = ["Multi", "Mix"]
VOLUME_PROFILE_TYPE_CHOICES = ["csv_file"]
FLW_METHOD_CHOICES = ["slope"]
FLW_REACTOR_CHOICES = ["JSR", "PSR", "FlowReactor"]


def is_runnable(solver: str, target_type: str, flux_analysis: bool = False) -> Tuple[bool, str]:
    """Whether (solver, target_type) is executable by the
    ``src/Sensitivity_Analysis/`` engine specifically (the one
    ``sens.py``/``sens_3_params.py`` instantiate). Returns (ok, reason).

    This does NOT reflect the nominal-simulation pipeline
    (``src/mso_compare/run_nominal_sim.py``), which imports the fuller
    ``src/MUQ-SAC/`` copy of ``make_input_file.py`` and *does* support PFR
    and zerork -- confirmed empirically (H2/review/flag.md). A "not
    runnable" result here is scoped to the Sensitivity-Analysis engine only;
    callers previewing targets destined for the nominal-sim pipeline should
    not treat it as a blanket verdict.
    """
    if solver not in LIVE_ENGINE_SOLVERS:
        return False, (
            f"solver={solver!r} is not runnable via the Sensitivity-Analysis "
            "engine (src/Sensitivity_Analysis/'s copy of make_input_file.py has "
            "no branch for it yet -- bug.md, KRUNAL_JULY_23.md point 4). It IS "
            "supported by the nominal-simulation pipeline's src/MUQ-SAC/ engine "
            "copy if that's the run path this target file is destined for."
        )
    if target_type not in LIVE_ENGINE_TARGET_TYPES:
        return False, (
            f"target type={target_type!r} (PFR) has no branch at all in the "
            "Sensitivity-Analysis engine's make_input_file.py copy. It IS "
            "supported by the nominal-simulation pipeline's src/MUQ-SAC/ engine "
            "copy if that's the run path this target file is destined for."
        )
    if flux_analysis:
        return False, (
            "flux_analysis is a src/MUQ-SAC/-only addition; the "
            "Sensitivity-Analysis engine's make_input_file.py has no "
            "flux_analysis/flux_element handling."
        )
    return True, "runnable"


# ---------------------------------------------------------------------------
# ``target.add`` per-dataset-ID option defaults
# (src/MUQ-SAC/combustion_target_class.py:341-527 default cascade)
# ---------------------------------------------------------------------------

ADD_DEFAULTS: Dict[str, object] = {
    "solver": "FlameMaster",
    "Thermodiffusion": True,
    "ComputeWithRadiation": True,
    "saveAll": False,
    "BoundaryLayer": False,
    "ign_delay_def": "OH",
    "ign_cond": "max",
    "specific_cond": "None;None",
    "width": 0.014,
    "ratio": 3,
    "slope": 0.015,
    "curve": 0.015,
    "volume_profile_type": "csv_file",
    "dt": 1e-5,
    "loglevel": 1,
    "auto": "True",
    "width_bsf": 0.5,
    "slope_bsf": 0.05,
    "phi": "N/A",
    "type": "N/A",
    "flw_length": 1,
    "flow_velocity": 0.06,
    "crossSectionalArea": 0.000314,
    "reactorVolume": 0.000314,
    "residenceTime": 1,
    "time_step": 2000,
    "flw_method": "slope",
    "flw_species": "H2",
    "flw_limits": (40, 60),
    "anchor": 50,
    "limit_units": "percentage",
    "flow_rate": "0.06",
    "flf_target": "H",
    "flf_cond": "max",
    "T_ambient": 298,
    "transport_model": "Multi",
    "MaxSimulationTime": 0.4,
    "species": "CO2",
    "pfr_length": 0.1,
    "pfr_diameter": 0.02,
    "pfr_mdot": 1e-3,
    "pfr_n_points": 200,
    "flux_analysis": False,
    "flux_element": "C",
    "zerork_install_dir": "/home/krithika/zero-rk/build2/inst_dir",
    "zerork_idt_temp_delta": 400.0,
    "zerork_rtol": 1e-8,
    "zerork_atol": 1e-20,
    "zerork_eps_lin": 0.05,
    "zerork_nonlinear_convergence_coefficient": 0.05,
    "zerork_preconditioner_threshold": 2.048e-3,
    "zerork_jsr_preconditioner_threshold": 1.0e-6,
    "zerork_n_mpi_ranks": 1,
    "zerork_rcm_volume_mult": 1.0e-6,
    "zerork_rcm_stroke_mult": 1.0,
    "zerork_rcm_max_dt_internal": 1.0e-3,
    "zerork_rcm_max_steps": 1000000,
    "zerork_rcm_ref_temp": 1000.0,
    "zerork_rcm_ref_moles": 0.25,
    "zerork_rcm_max_order": 5,
    "zerork_rcm_max_num_nonlin_iters": 3,
    "ExpTempFile": "",
}

# Keys with no default -- required only for specific branches, but raise
# KeyError/AssertionError in create_input_file if missing when that branch
# is hit. Surfaced so a form can prompt for them conditionally.
CONDITIONALLY_REQUIRED = {
    "RCM_type": "required when target type is RCM (reactive branch)",
    "volume_profile": "required for RCM and Tig VTIM (volume-time-history) branches",
    "dpdt": "required for dp/dt-driven Tig branches (CHEMKIN_PRO/FlameMaster)",
    "mol_wt": "required when ign_cond=='specific' and its unit is 'molecule'",
    "file_header": "required when ExpTempFile is set (Flf burner-temp profile)",
    "fuel": "required for Fls target type=='phi' branch (phi-mode fuel override)",
}

# ---------------------------------------------------------------------------
# Field groups -- which target.add keys are relevant per target type, for
# conditional show/hide in a form driven by the "target type" selector.
# ---------------------------------------------------------------------------

COMMON_FIELDS = ["solver", "ign_delay_def", "ign_cond", "specific_cond",
                  "mol_wt", "exp_conc", "saveAll", "flux_analysis", "flux_element"]

TYPE_FIELDS: Dict[str, List[str]] = {
    "JSR": ["residenceTime", "MaxSimulationTime", "species"],
    "PFR": ["pfr_length", "pfr_diameter", "pfr_mdot", "pfr_n_points", "species"],
    "Flf": ["residenceTime", "MaxSimulationTime", "species", "ExpTempFile",
            "file_header", "flf_target", "flf_cond"],
    "Flf_a": ["residenceTime", "MaxSimulationTime", "species"],
    "RCM": ["RCM_type", "volume_profile", "volume_profile_type", "exp_conc",
            "saveAll"],
    "Tig": ["BoundaryLayer", "volume_profile", "volumeProfile",
            "volume_profile_type", "dpdt", "dt", "MaxSimulationTime"],
    "Fls": ["width", "ratio", "slope", "curve", "loglevel", "auto", "type",
            "fuel", "ComputeWithRadiation", "Thermodiffusion",
            "StartProfilesFile"],
    "Flw": ["flw_length", "flow_velocity", "crossSectionalArea",
            "reactorVolume", "residenceTime", "species", "time_step",
            "flw_method", "flw_species", "flw_limits", "anchor",
            "anchor_time", "limit_units"],
}

ZERORK_FIELDS = [
    "zerork_install_dir", "zerork_idt_temp_delta", "zerork_stop_time",
    "zerork_rtol", "zerork_atol", "zerork_eps_lin",
    "zerork_nonlinear_convergence_coefficient",
    "zerork_preconditioner_threshold", "zerork_jsr_preconditioner_threshold",
    "zerork_rcm_volume_mult", "zerork_rcm_stroke_mult",
    "zerork_rcm_max_dt_internal", "zerork_rcm_max_steps",
    "zerork_rcm_ref_temp", "zerork_rcm_ref_moles", "zerork_rcm_max_order",
    "zerork_rcm_max_num_nonlin_iters",
]


def fields_for(target_type: str, solver: str) -> List[str]:
    """Ordered list of target.add fields relevant to this (type, solver)
    combo -- common fields, then type-specific, then zerork if selected."""
    out = list(COMMON_FIELDS)
    out += [f for f in TYPE_FIELDS.get(target_type, []) if f not in out]
    if solver == "zerork":
        out += [f for f in ZERORK_FIELDS if f not in out]
    return out


@dataclass
class NewCaseFields:
    """Minimal field set for the GUI's "manual entry" mode -- building a
    brand-new .target line + .add entry for one experimental case, without
    hand-typing the pipe-delimited target-file syntax.

    Deliberately covers the common Tig/RCM/JSR single-condition case (the
    realistic manual-entry scenario); Fls/Flf/Flw/PFR manual authoring is
    left to hand-editing the .target file directly (feature B's embedded
    editor covers that).
    """
    dataset_id: str
    target_type: str = "Tig"
    simulation: str = "Isochor Homo Reactor"
    solver: str = "cantera"
    temperature_K: float = 1000.0
    pressure_atm: float = 1.0
    fuel: Dict[str, float] = field(default_factory=lambda: {"H2": 1.0})
    oxidizer: str = "O2"
    oxidizer_x: float = 1.0
    bath_gas: Dict[str, float] = field(default_factory=lambda: {"Ar": 1.0})
    phi: str = "N/A"
    observed: float = 0.0
    deviation: float = 1.0
    data_weight: float = 1.0
    species: str = "OH"
    add_overrides: Dict[str, object] = field(default_factory=dict)
