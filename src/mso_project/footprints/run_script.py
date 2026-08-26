"""
mso_project.footprints.run_script
===================================
Full artifact footprint for ``src/MUQ-SAC/run_script.py``, patterns relative
to the run's working_dir (the directory ``run_script.py`` is invoked from --
it does ``os.chdir("Opt")``/``os.chdir("..")`` internally, but every path it
touches is CWD-relative to where it started, per the file-by-file audit this
footprint is built from).

Entries are checked in order; more specific patterns are listed before
broader ones. See ``schema.py``'s ``artifacts`` table for what ``role``
feeds into (used for report routing and for ``restore.resume()``'s
copy-vs-link safety decisions -- roles noted "rewritten in place" below are
never safe to symlink back for a subsequent run).
"""
from __future__ import annotations

from . import FootprintEntry

FOOTPRINT = [
    # -- pre-DesignMatrix bookkeeping --------------------------------------
    FootprintEntry("unsrt.pkl", "uncertainty_cache"),
    FootprintEntry("targets.csv", "target_summary"),
    FootprintEntry("T-P-phi_space.png", "t_p_phi_plot"),
    FootprintEntry("ACTIVE_RXN_INDEX.csv", "reaction_index"),
    FootprintEntry("ACTIVE_RXN_LIST.csv", "reaction_index"),
    FootprintEntry("rxn_list_a_fact.csv", "reaction_index"),
    FootprintEntry("Uncertainty_Matrices/*/L.csv", "uncertainty_matrix"),
    FootprintEntry("Uncertainty_Matrices/*/Sigma.csv", "uncertainty_matrix"),
    FootprintEntry("Uncertainty_Matrices/*/nominal.csv", "uncertainty_matrix"),

    # -- sensitivity analysis (Partial PRS only) ---------------------------
    FootprintEntry("sens_parameters.pkl", "sensitivity_cache"),
    FootprintEntry("sens_3p_parameters.pkl", "sensitivity_cache"),
    FootprintEntry("SA_tool.out", "sensitivity_log"),
    FootprintEntry("SA_tool_3p.out", "sensitivity_log"),
    FootprintEntry("rxn_list.csv", "reaction_index"),

    # -- design matrix + perturbed mechanisms --------------------------------
    FootprintEntry("DesignMatrix.csv", "design_matrix"),
    FootprintEntry("Perturbed_Mech/mechanism_*.yaml", "perturbed_mechanism"),
    FootprintEntry("DM_FOR_PARTIAL_PRS/*/*", "partial_prs_cache"),
    FootprintEntry("YAML_FILES_FOR_PARTIAL_PRS/*/*", "partial_prs_yaml"),

    # -- Opt/ run tree --------------------------------------------------------
    FootprintEntry("Opt/progress", "progress_marker"),
    FootprintEntry("Opt/locations", "progress_marker"),
    FootprintEntry("Opt/Data/Simulations/sim_data_case-*.lst", "simulation_log"),
    FootprintEntry("Opt/Data/Simulations/failed_sim_data_case-*.lst", "simulation_log"),
    FootprintEntry("Opt/Data/ResponseSurface/responsecoef_case-*.csv", "response_surface_coeff"),
    FootprintEntry("Opt/Data/Outlier_Simulations/sim_data_case-*_pass-*.lst", "outlier_resample_data"),
    FootprintEntry("Opt/Data/Outlier_Simulations/*.meta.json", "outlier_resample_data"),
    FootprintEntry("Opt/Data/Outlier_Simulations/failed_sim_case-*.lst", "outlier_resample_data"),
    FootprintEntry("Opt/Data/Outlier_Simulations/Outlier_DM_case-*.csv", "outlier_resample_data"),
    FootprintEntry("Opt/Outlier_Resample/**/mechanism_*.yaml", "outlier_perturbed_mechanism"),
    FootprintEntry("Opt/Plots/Posterior/**", "posterior_plot"),

    # -- run root: fit quality, archives, solution ---------------------------
    FootprintEntry("Prediction_errors.csv", "fit_quality_summary"),
    FootprintEntry("Prediction_errors_detailed.csv", "fit_quality_summary"),
    FootprintEntry("Plots/**", "diagnostic_figure"),
    FootprintEntry(".study_manifest.json", "study_marker"),
    FootprintEntry("CASE_STUDIES/**/manifest.json", "study_marker"),
    FootprintEntry("CASE_STUDIES/**", "archived_study"),
    FootprintEntry("solution_zeta.save", "solution"),
    FootprintEntry("posterior_cov.pkl", "posterior_covariance"),
    FootprintEntry("Optimized_mech.yaml", "optimized_mechanism"),
    FootprintEntry("optimization_cost.csv", "cost_summary"),
]
