"""
gui_common/target_types.py
============================
KP's existing target-type vocabulary, extracted verbatim from
generate_target_opt.py. This is the only place this vocabulary is defined
in the codebase (confirmed during the GUI-separation planning research --
no separate markdown doc defines it; it lives here and in
src/MUQ-SAC/combustion_target_class.py's parsing of the `target--` field,
which the two already agree with).

Every split-out GUI app that touches `.target`/`.add` files must import
this, never re-invent target-type strings.
"""
from __future__ import annotations

# Experimental target/simulation types actually dispatched on in
# src/MUQ-SAC/make_input_file.py (create_input_file()). Used everywhere a
# "target_type" combobox is shown (Target Builder, Addendum Builder) so they
# can't drift out of sync with each other or with the backend again.
TARGET_TYPES = {
    "Tig":    "Ignition delay (shock tube reflected-shock, or RCM via ign_mode). "
              "Solvers: cantera, CHEMKIN_PRO, FlameMaster.",
    "Fls":    "Freely-propagating laminar flame speed. Solver: cantera only.",
    "Flf":    "Residence-time-controlled reactor calc (JSR-like, despite the name). "
              "Solver: cantera only.",
    "Flf_a":  "Heat-flux / burner-stabilized flame (imposed T profile). "
              "Solvers: cantera, FlameMaster.",
    "JSR":    "Jet/perfectly-stirred reactor (residence-time controlled). "
              "Solvers: cantera, CHEMKIN_PRO, FlameMaster. Supports flux_analysis.",
    "PFR":    "Plug-flow reactor (length/diameter/mdot marched). "
              "Solvers: cantera, CHEMKIN_PRO, FlameMaster (approximate). Supports flux_analysis.",
    "RCM":    "Rapid compression machine (reactive or non_reactive volume profile). "
              "Solvers: cantera, CHEMKIN_PRO.",
    "Flw":    "Flow-reactor species-decay/rate experiment. Solvers: cantera, FlameMaster.",
}

# Fields specific to each target_type, shown only when that type is selected
# in the Addendum Builder row editor (mirrors combustion_target_class.py).
TARGET_TYPE_FIELDS = {
    "Tig":   ["ign_delay_def", "ign_cond", "specific_cond", "BoundaryLayer"],
    "Fls":   ["fls_type", "fls_fuel"],
    "Flf":   ["residenceTime", "flf_species"],
    "Flf_a": ["ExpTempFile", "flow_rate"],
    "JSR":   ["residenceTime", "jsr_species", "flux_analysis", "flux_element"],
    "PFR":   ["pfr_length", "pfr_diameter", "pfr_mdot", "pfr_n_points", "flux_analysis", "flux_element"],
    "RCM":   ["RCM_type", "ign_delay_def", "ign_cond"],
    "Flw":   ["flw_length", "flw_species", "flw_method"],
}

# Columns backing the Experimental Data tab's target-row DataFrame.
TARGET_ROW_COLUMNS = [
    "dataset_id", "target_type", "fuel_species", "fuel_xf",
    "oxidizer_species", "ox_xf", "bath_species", "bath_xf",
    "T", "P", "P_unit", "phi", "observed", "deviation_pct", "data_weight",
    "max_simulation_time", "temp_target_class",
]
