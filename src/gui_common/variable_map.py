"""
gui_common/variable_map.py
============================
Maps GUI field names to codebase variable names + explanatory comments,
extracted verbatim from generate_target_opt.py's VariableNameMapper class.
Reference/documentation data (tooltips), not deeply wired into any tab's
logic -- low coupling, safe to centralize as-is.
"""
from __future__ import annotations


class VariableNameMapper:
    """
    Map GUI field names to codebase variable names with comments.
    Ensures aesthetic GUI names translate cleanly to MUQ-SAC/RMIP codebase.
    """
    # GUI name → (codebase name, comment explaining the mapping)
    MAPPINGS = {
        # Project tab
        "proj_name": ("project.name", "Project identifier (no spaces)"),
        "proj_fuel": ("project.fuel_name", "Human-readable fuel name shown in reports"),
        "proj_formula": ("Inputs.fuel.formula", "Cantera/CHEMKIN species ID (e.g. NC7H16)"),
        "proj_outdir": ("project.output_base_dir", "Root for outputs: NOMINAL/, Opt/, etc."),

        # Type tab
        "opt_type": ("Type.optimization_type", "PRS | SAC | direct"),
        "alg_type": ("Type.Algorithm_type", "gradient-based | gradient-based-zeta | evolutionary | custom"),
        "algorithm": ("Type.Algorithm", "SLSQP | L-BFGS-B | trust-constr | SLSQP-zeta | "
                      "L-BFGS-B-zeta | DE | GA-pygad | GA-deap | Powell | <custom>"),

        # Counts
        "n_rxn": ("Counts.total_reactions", "Number of reactions optimized"),
        "n_thr": ("Counts.parallel_threads", "Parallel threads (capped at CPU-5)"),
        "n_tar": ("Counts.targets_count", "Number of experimental target points"),
        "carbon_n": ("SA.carbon_number", "Carbon number for SA grouping (0=H2/CO)"),

        # Locations
        "loc_mech": ("Locations.mechanism", ".inp or .yaml mechanism file"),
        "loc_thermo": ("Locations.thermo_file", "Separate CHEMKIN thermo (.dat)"),
        "loc_unsrt": ("Locations.uncertainty_data", "Uncertainty XML (.xml)"),
        "loc_targets": ("Locations.targets", "Experimental targets (.target file)"),
        "loc_add": ("Locations.addendum", "Solver addendum (.add YAML file)"),

        # Target row fields (dynamic)
        "target_T": ("T", "Temperature (K)"),
        "target_P": ("P", "Pressure (atm or bar, see units)"),
        "target_phi": ("Phi", "Equivalence ratio (unitless)"),
        "target_observed": ("observed", "Measured ignition delay or flame speed"),
        "target_deviation": ("deviation", "Measurement uncertainty (same units as observed)"),

        # Fuel in target
        "fuel_species": ("Fuel", "Fuel species name in .target format"),
        "fuel_xf": ("fuel_x", "Fuel mole fraction in mixture"),

        # Oxidizer in target
        "ox_species": ("Oxidizer", "Oxidizer species (usually O2)"),
        "ox_xf": ("ox_x", "Oxidizer mole fraction"),

        # Bath gas in target
        "bath_species": ("Bath_gas", "Diluent species (N2, AR, etc.)"),
        "bath_xf": ("bath_x", "Diluent mole fraction"),
    }

    @staticmethod
    def get_codebase_name(gui_name: str) -> str:
        """Get codebase variable name from GUI name."""
        return VariableNameMapper.MAPPINGS.get(gui_name, (gui_name, ""))[0]

    @staticmethod
    def get_comment(gui_name: str) -> str:
        """Get explanatory comment for a GUI field."""
        return VariableNameMapper.MAPPINGS.get(gui_name, ("", ""))[1]
