# CHANGELOG — Mechanism Optimization Suite

## 2026-08-26 — Merge t2 (MULTI-STAGE_OPTIMIZATION) + t3 (FINALIZED_MECH_OPT_SUITE) into `src/`, plus empty-output-folder and Ansys-license fixes

Branch: `three-param-pprs-visualaid`. Rollback point: tag `pre-merge-20260826` (+ tarball `/home/krithika/src-backup-20260826.tgz`).

### Pre-merge snapshot (commit `a3afe1c` "snapshot S0")

One line per dirty `src/` file, committed before any merge:
- `MUQ-SAC/{AdaptiveResampler, DesignMatrix, Uncertainty, partial_PRS_system, simulation_manager2_0}.py` — uncommitted latest correlation-aware resampling / band-validity work.
- `src/HYBRID_PRS.py` deleted (module lives in `MUQ-SAC_EXTENDED/` in all trees).
- `src/aim.md`, `src/_ORIGINALS_20260819/` — untracked; now committed for provenance.
- Also: all tracked `*.pyc` untracked (commit `203cbe5`); `_ORIGINALS_*/` added to `.gitignore` (commit `9902e3b`).

### Merged from t2 (`/data/KRUNAL/MULTI-STAGE_OPTIMIZATION/src`)

- `src/` file-equivalence verified: t2's tree is byte-identical to the repo's except one stray line in `MUQ-SAC/make_input_file.py` — `echo "10.24.0.171   cclicserver" | sudo tee -a /etc/hosts` — **deliberately NOT merged** (see license fix below: the mapping already exists in `/etc/hosts`, and a non-interactive `sudo tee` would hang). No file changes from t2.
- t2's `.gitignore` patterns are already covered by the repo root `.gitignore`.
- t2's root-level study artifacts (`report_10fold_validation/`, `report_resampling/`, `TEST_THREE_PARAM_pPRS/`, `TEST_CORRELATION_AWARE/`, `TEST_CORRELATIONS_CORRECTED/`, `THREE-PARAM-MSO-LLNL-N_HEPTANE_159*`, `PLAN.md`, `Goal.md`) are **data-side, not merged** — they remain under `/data/KRUNAL/MULTI-STAGE_OPTIMIZATION/`. The 10-fold validation *code* lives in `src/MUQ-SAC/validation/` (present in all three trees; kept from the repo).

### Merged from t3 (`/data/FINALIZED_MECH_OPT_SUITE/src`) — copied wholesale

t3 is a strict superset for these (repo had no divergent content):
- `mso_io/` — **renames `io/`** (stdlib-name shadowing) and adds: `latex_engine.py`, `mpl_theme.py`, `plot_style.py`, `native_extensions.py`, `notebook_gen.py`, `run_status.py`, `target_add_io.py`, `target_add_schema.py`, `postprocess_base.py`, `mechanism_check_fix_examples.py`. Importer `run_report.py` updated (`io.*` → `mso_io.*`).
- `rmip/` — big expansion: `reduction/` (11), `config/` **directory replaces the old `config.py` file** (its `__init__.py` is byte-identical to the old file), `postprocess.py`, ~25 new analysis modules (`bayesian_calibration`, `pareto`, `surrogate`, `thermo_fitter`, `transport_estimator`, `experiment_design`, `yaml_validator`, `mechanism_check/fixer`, `version_regression`, …), ~20 new simulation modules (`jsr_runner`, `pfr_runner`, `rop_runner`, `tdac_runner`, `zerork_bridge`, `zerork_gsa`, `muqsac_bridge`, `idt_sensitivity`, `gsa_prs_crosscheck`, `gsa_reaction_prefilter`, `finite_solid_reactor`, `_salib/_chaospy/_cvxpy_backend`, …), ~80 new tests, expanded `cli.py` (4718-line subcommand surface) and `types.py`.
- New packages: `automation/` (PRS DOE/fit/optimize/preflight/report pipeline), `mso_studio/`, `mso_project/`, `mso_workbench/`, `gui_common/`, `notebook_gemini/`.
- New top-level modules: `latex_project.py`, `pychemkin_reader.py`, `target_sampling.py`, `target_tables.py`.
- `MUQ-SAC/ResponseSurface.py` — M23 monomial-sum fix (order ≥ 3 PRS evaluation).
- `MUQ-SAC/combustion_target_class.py` — chem_data MW/stoich + fuel dict-shape dispatch (Phase-6 migration: `MaxSimulationTime` etc.).
- `MUQ-SAC/mech_diagnostics.py`, `mso_compare/run_h2_plots.py`, `forward_uq/uq_plots.py`, `graph_sa/plots.py`, `Sensitivity_Analysis/*` (14 files newer), `MUQ-SAC_EXTENDED/` (plus `run_script_search_iters.py`).

### Merged from t3 — cherry-picked hunks into repo-newer files

Repo kept as canonical; t3-only improvements grafted in:
- `MUQ-SAC/run_script.py` — `_maybe_apply_gsa_prefilter()` (opt-in `GSA_Prefilter` block, lazy `rmip` import) + Bug-4 `Perturbed_Mech` YAML-completion check (count `mechanism_*.yaml` vs `design_matrix` before trusting a restart).
- `MUQ-SAC/Uncertainty.py` — `_canon_rxn_equation()` grafted in as **tier 3** of `_find_rxn_index` (exact → arrow-agnostic → order/stoichiometry/falloff-notation-independent). Repo's arrow-agnostic tier 2 untouched.
- `MUQ-SAC/partial_PRS_system.py` — `glob`/`shutil` imports, Bug-3 fix (`enumerate(p_design_matrix)` — files are generated from the partial matrix), Bug-4 YAML-completion check for `YAML_FILES_FOR_PARTIAL_PRS/<case>`, corrected correlation-rescue docstring.
- `MUQ-SAC/simulation_manager2_0.py` — `_thread_cap()` clamp on `parallel_threads` (never pins all cores); `Worker.progress_file` explicit absolute path (pool callbacks no longer write `../progress` from arbitrary CWDs); **removed the never-used `pool1` ProcessPoolExecutor** (leaked `workers` OS processes per `Worker()`, e.g. 80 per instantiation); `locations`-file dedupe in `make_dir_in_parallel`/`run_nominal_case`; try/finally CWD restore in the resumed-case branch. Repo's RESUMED-CASE fix (2026-08-19) kept as the resume behavior.
- `MUQ-SAC/{DesignMatrix, MUQ-SAC_EXTENDED/DesignMatrix}.py` — t3's `self.rxn_len` fix (was a bare `AttributeError` in `getSA_samples`/`getNominal_samples`) + `n_reactions_in_design` introspection hook. All repo-only machinery (pool provenance, `refresh_class_pools`, safeguard validation, worker-error logging) kept.

### Hand-merged (both trees had real work)

- `MUQ-SAC/VisualAid.py` — **union**: repo's 14 PPRS/correlation/band/SVD plotters + t3's target-set workbench (`TargetRow/TargetSet/TargetSetPlotter/TargetFinding/TargetChecker/DummyPointGenerator/TargetPruner/NominalParityAnalyzer`), `H2SweepPlotter`, 3D-cube helpers, and the `postprocess()` CLI (+ `mso_io` path bootstrap). No name collisions. 4055 lines, compiles.
- `mso_compare/sim_runner.py` — t3 base (PFR dataset-CSV branch, `preview_targets_for_run`, extraction None-guards, Fls double-index fix, `import data_management` — fixing the latent `data_management_nominal` ImportError) **plus** the repo's `_sweep_missing_simulations` cross-case re-run sweep re-attached.
- `modules/plotting.py` — repo kept (its `create_temperature_comparison_plot` fontsize/legend/thickness API is strictly newer; t3's version was older).
- `MUQ-SAC/OptimizationTool.py`, `MUQ-SAC/IdentifiablePRS.py`, `MUQ-SAC/AdaptiveResampler.py` — repo kept unchanged; diff review showed t3's versions were strictly older (repo added RCM branches, `FITNESS_PRS_BASED_OPT_A_FACTO`, `fprior_mode` full-band scoring, pool-provenance gate, `oversample_boost` stall recovery, `selected_signature`).
- `MUQ-SAC/make_input_file.py` — **all 44 t3 hunks applied** (= t3's file): `estimateTIG`/`maxSimulationTime`/`total_time` → `MaxSimulationTime` key migration (completes the Phase-6 migration the repo's target class already had), `unitless` ignition-condition branches (8 UnboundLocalError fixes), new zerork RCM/Tig branches + solver dispatch, chemkin-solver JSR engine (`CKReactorGenericPSR` — carries "KP to verify" markers; no JSR rows in the n-heptane study, so latent), Fls `fuel=repr()` fix, `makedirs(exist_ok=True)` + `bash ./run_profile` robustness. CHEMKIN_PRO `s_run` chain kept byte-identical except the `&&` gating below.
- `MUQ-SAC/data_management.py` — t3 working tree as base (PFR branch, `_log_missing_solver_output`, Bug-18 solver-honoring retry, `file_path=None` guards, `_run_script_needs_soln2ck`) **plus the new empty-folder fix below**.
- `run_report.py` — t3's version (LaTeX engine ladder, `--config` non-interactive path) + `io`→`mso_io` import rewrite.

### New fixes (this merge)

**Empty-output-folder failures** (root cause: a failed simulation leaves `output/` EMPTY — license failure before `tau.out`, interrupted `./run`, or the unconditional `find -delete` cleanup — and empty folders were treated as "complete" by the resume path):
- `data_management.extract_output` — no longer silently `os.makedirs` a missing folder; missing/empty folders report explicit `N/A (output folder missing|empty: ...)`.
- `data_management.generate_target_value_tables` — one-shot retry replaced by a bounded 3-attempt detect→re-run→re-extract loop.
- `data_management.run_sim` — idempotent: never `rmtree`s a folder that already holds a valid result (parses and reuses it).
- `run_script.py` — cached `sim_data_case-N.lst` trusted only when it has ≥1 row; an empty `.lst` falls through to regeneration.
- `simulation_manager2_0.getDirectoryList` — an existing-but-EMPTY `output/` is treated as never-simulated (wipe + re-enqueue) so resumed runs repair them.
- `data_management2_0_A_factor.py` — twin bug fixed: N/A branches returned `string = path` (folder) instead of `path + "<outfile>"`, which made `run_sim` look for a file named `output`; same bounded retry + missing/empty detection mirrored.
- `make_input_file.py` — CHEMKIN_PRO / zerork / FlameMaster `s_run` chains now `&&`-gate the destructive `find ./output … -delete` so a failed solve can never wipe its own output.

**Ansys Chemkin license failures** (user's top priority — `~/Desktop/PyChemkin-master/chemkin.py`, the live copy on `PYTHONPATH`):
- `preprocess()` and `postprocess()` now run through `_run_with_license_retry` (exponential backoff, `CHEMKIN_LICENSE_RETRY_MAX/DELAY/BACKOFF` env-tunable) — previously only `run()` retried, so the dominant "ANSYSLI exited or could not read server port ansyscl" transient failed samples outright on first checkout.
- `_verify_license_hosts_entry()` — read-only, idempotent `/etc/hosts` check, logged once per `ChemkinJob`; never writes (no passwordless sudo — a blind `sudo tee -a` would hang). Mapping `10.24.0.171 cclicserver` verified present on this host.
- Recommended launch env: `CHEMKIN_LICENSE_RETRY_MAX=6 CHEMKIN_LICENSE_RETRY_DELAY=2 CHEMKIN_LICENSE_RETRY_BACKOFF=2`.
- Note: `/home/krithika/MUQ-SAC/PyChemkin-master/chemkin.py` is an older, unused copy (not on `PYTHONPATH`) — left untouched.

### Deliberately not merged

- t2 root study artifacts (above) — data-side, remain in `/data`.
- t3 root content (`launcher_*.py`, `*_gui.py`, `paper.md/bib`, `SIM_RUNS/`, `outputs/`, `assets/`, `MarkDownFiles/`, `docs/`) — tooling/prose outside `src/`; `MarkDownFiles/HISTORY.md` referenced as the project timeline.
- The stray `sudo tee` line from t2's `make_input_file.py` (replaced by the retry + verification fix above).
- `src/rmip/` brace-named junk files (`{database,…}` etc.) — left tracked as-is; separate cleanup only with user approval.

### Operational notes

- `MUQ-SAC/validation/` requires `MUQSAC_REPO_ROOT` env for its config (see `validation/README.md`).
- Live study `target.opt` `bin:` points at `/home/krithika/mechanism-optimization-suite/src/MUQ-SAC` — the repo tree is the executing code.
- Dataset addendums still keyed `estimateTIG`/`maxSimulationTime` (e.g. butadiene, methyl-butanoate `DATASET/` entries) must be migrated to `MaxSimulationTime` before those studies next run — value-neutral for n-heptane (defaults coincide), silently wrong otherwise.
- `/home/krithika/MUQ-SAC/V3.0/` holds an older runtime copy of `make_input_file.py`/`combustion_target_class.py` — sync both together from this repo before using that tree.
