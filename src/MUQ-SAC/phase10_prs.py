"""
phase10_prs.py — Phase 10 (per-case p-PRS fitting) extracted from run_script.py.

Extracted 2026-08-27 (branch three-param-pprs-visualaid) so the entire Phase 10
pipeline lives in one debuggable, re-usable class instead of ~880 inline lines
of run_script.py. Behaviour is line-for-line identical to the extracted block;
the only additions are the structured STEP traces (`verbose=True`, default ON)
and an optional per-combo callback into the hyperparameter search
(`trace_search=True`, passed through to IdentifiablePRS.fit_prs_with_cv_search /
fit_prs_with_search via the optional `trace_cb` argument).

Usage from run_script.py::

    from phase10_prs import Phase10PRS
    phase10 = Phase10PRS({...all names Phase 10 used as module globals...})
    ResponseSurfaces = phase10.run()

The offline single-case reproducer (Phase10CaseDiagnostics) re-runs one case's
fit from a study directory on disk (unsrt.pkl + DM_FOR_PARTIAL_PRS/<case>/ +
Opt/Data/Simulations/sim_data_case-<case>.lst) — no Chemkin, no full pipeline.
It was validated against the pilot_5case_case0 run: it reproduces the logged
numbers (CV winner tol=1.0/k=20/order=2/ridge=0.3 @ 17.444% worst-fold; final
train 15.423% / holdout 14.817%) and the TEST_CORRELATIONS_CORRECTED case-0
numbers (train 0.894% / holdout 0.644%) exactly.
"""

import os
import json
import hashlib
import traceback

import numpy as np

# ── No-ignition sentinel (2026-08-19) ────────────────────────────────────
# data_management.extract_output substitutes this EXACT value
# (ln(100 * 10000) = ln(1e6)) whenever a solver's output file exists but
# doesn't parse into a real ignition-delay reading -- a fabricated "no
# ignition found" placeholder, not a real measurement. Zero occurrences in the
# pilot study's data, but screen_case_samples checks for it explicitly so a
# future batch containing one can never silently train/score against it.
NO_IGNITION_SENTINEL = np.log(100 * 10000)


def _mini_dm_fingerprint(prs_rows):
    """SHA1 fingerprint of a mini-DM's rows (rounded to tame float noise),
    used to validate a cached simulation result actually matches the CURRENT
    pass's mini-DM before reusing it -- see Phase10PRS._simulate_new_samples."""
    arr = np.round(np.asarray(prs_rows, dtype=float), 8)
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ── Experiment-arm configuration (2026-08-28) ────────────────────────────
# The four arms: original (today's behavior, byte-identical) / transform
# (response encoding-decoding) / resampling (correlation-aware augmentation)
# / resampling_transform (both). Every key is optional; missing keys resolve
# to `original`. See Readme.md for the full schema.

EXPERIMENT_DEFAULTS = {
    "mode": "original",
    "transform": {
        "variant": "power",           # "power" | "rank" | "holder_linear"
        "P": 0.1,                     # power exponent in (0, 1] -> 0+
        "P_grid": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02],   # offline sweep
        "p_lo": 0.02,                 # holder_linear: per-sample p lower bound
        "standardize": False,
        "rank_fit_on": "all",         # "all" | "train_val"
    },
    "resampling": {
        "n_new": 2000,                # candidate volume per case
        "empirical_weight": 1.0,      # lambda: 0 = physical corr, 1 = empirical
        "band_tolerance": 1.0,        # candidate validity gate
        "qmc": True,                  # scrambled-Sobol -> Normal draws
        "seed": 20260828,
        "center_mode": "pool_center", # "pool_center" | "kmeans"
        "dedup_eps": 1e-6,
        "dry_run": True,              # live run_script: NO simulation unless False
    },
    "stage3_with_transform": False,   # Stage 3 disabled in non-original arms
}

# Plot-folder per arm (user-specified names, verbatim -- note "transfrom").
ARM_FOLDER = {
    "original": "Plots",
    "transform": "Plots_transfrom",
    "resampling": "Plots_re-sampling",
    "resampling_transform": "Plots_re-sampling_transform",
}


def _merge_experiment_cfg(user_cfg):
    """Deep-merge a user experiment_cfg over EXPERIMENT_DEFAULTS."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in EXPERIMENT_DEFAULTS.items()}
    if not user_cfg:
        return cfg
    for k, v in user_cfg.items():
        if k in ("transform", "resampling") and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    if cfg["mode"] not in ARM_FOLDER:
        raise ValueError(f"prs_experiment_mode must be one of "
                         f"{sorted(ARM_FOLDER)}, got {cfg['mode']!r}")
    return cfg


# ── Decode-patch helpers (transform arms report in y-space) ────────────────

def _decode_patch_train(Response, y_raw_train, encoder):
    """After create_response_surface(), rewrite the training-side diagnostics
    (self.Y / resFramWrk / error / relative_error / MaxError / MeanError /
    RMS) from encoded z-space to DECODED y-space, so every downstream report
    and plot reads physical ln(tau) values. Idempotent per fit."""
    y_raw_train = np.asarray(y_raw_train, dtype=float).ravel()
    pred_y = encoder.decode(np.asarray(Response.resFramWrk, dtype=float))
    err = pred_y - y_raw_train
    denom = np.where(np.abs(y_raw_train) > 1e-12, y_raw_train, 1e-12)
    rel = np.abs(err / denom) * 100.0
    Response.Y = y_raw_train
    Response.resFramWrk = list(pred_y)
    Response.error = list(err)
    Response.relative_error = list(rel)
    Response.RMS_error = list(err ** 2)
    Response.MaxError = float(np.max(rel))
    Response.MeanError = float(np.mean(rel))
    Response.RMS = float(np.sqrt(np.mean(err ** 2)))


def _score_decoded(Response, xTest, y_raw_test, encoder):
    """Response.test() in encoded z-space, then rewrite the test-side
    diagnostics to DECODED y-space. Returns (MaxError, ytestMaxError,
    criteria) with both MaxError (training) and ytestMaxError in y-space.
    Leaves error_testing_relative in y-space so the band-valid gating and
    the Stage-3 trigger downstream read physical errors unchanged."""
    y_raw_test = np.asarray(y_raw_test, dtype=float).ravel()
    Response.test(xTest, encoder.encode(y_raw_test))
    pred_y = encoder.decode(np.asarray(Response.y_Test_Predict, dtype=float))
    err = pred_y - y_raw_test
    denom = np.where(np.abs(y_raw_test) > 1e-12, y_raw_test, 1e-12)
    rel = err / denom * 100.0
    Response.y_Test_Predict = list(pred_y)
    Response.y_Test_simulation = y_raw_test
    Response.error_testing = list(err)
    Response.error_testing_relative = list(rel)
    Response.ytestMaxError = float(np.max(np.abs(rel)))
    Response.yTestMeanError = float(np.mean(rel))
    return Response.MaxError, Response.ytestMaxError, Response.criteria


class Phase10PRS(object):
    """The whole Phase 10 pipeline: per-case screening -> 3-way split ->
    (tolerance, k, order, ridge) search -> identifiable-subspace PLS + PRS
    fit -> optional Stage-3 D-optimal outlier retraining -> one-time
    untouched hold-out verdict, with a structured per-step trace.

    Parameters
    ----------
    ctx : dict
        Everything the extracted block referenced as run_script module
        globals. Keys (all required)::

            target_list, unsrt_data, optInputs, design_matrix_dict,
            kept_row_index, selected_params_dict, temp_sim_opt, no_of_sim,
            cost_estimate_dict, optDir, home_dir, fuel, PRS_type, design_type,
            MRE_criteria, prs_fit_cfg, outlier_cfg,
            modules = {"IPRS":..., "PRS":..., "AR":..., "simulator":...,
                       "data_management":...},
            csv_files = {"PRS_ERROR_FILE":..., "PRS_DETAIL_FILE":...,
                         "SCREENING_SUMMARY_FILE":..., "COST_FILE":...},
            log = {"step":..., "kv":..., "table":..., "result":...,
                   "warn":..., "error":..., "append_csv_row":...}

    verbose : bool
        Print the structured STEP-by-STEP trace (default True). The trace
        shows, for every case: the inputs each step receives, the numbers it
        computes, and what it hands to the next step.

    trace_search : bool
        Also print every (tolerance, k, order, ridge) combo's worst-fold CV
        score as the search evaluates it (default False -- the full grid is
        hundreds of fits; the winning combo is always printed regardless).
    """

    def __init__(self, ctx, verbose=True, trace_search=False):
        self.target_list        = ctx["target_list"]
        self.unsrt_data         = ctx["unsrt_data"]
        self.optInputs          = ctx["optInputs"]
        self.design_matrix_dict = ctx["design_matrix_dict"]
        self.kept_row_index     = ctx["kept_row_index"]
        self.selected_params_dict = ctx["selected_params_dict"]
        self.temp_sim_opt       = ctx["temp_sim_opt"]
        self.no_of_sim          = ctx["no_of_sim"]
        self.cost_estimate_dict = ctx["cost_estimate_dict"]
        self.optDir             = ctx["optDir"]
        self.home_dir           = ctx["home_dir"]
        self.fuel               = ctx["fuel"]
        self.PRS_type           = ctx["PRS_type"]
        self.design_type        = ctx["design_type"]
        self.MRE_criteria       = ctx["MRE_criteria"]
        self.prs_fit_cfg        = ctx["prs_fit_cfg"]
        self.outlier_cfg        = ctx["outlier_cfg"]

        # experiment arms (2026-08-28): transform / correlation-aware
        # resampling. Missing ctx key => defaults => mode "original" =>
        # zero behavior change (backward compatible).
        self.experiment_cfg = _merge_experiment_cfg(ctx.get("experiment_cfg"))
        self.mode = self.experiment_cfg["mode"]
        self._encoders = {}      # case -> ResponseEncoder (transform arms)
        self._exp_rows = []      # prs_experiment_results.csv rows (run())

        m = ctx["modules"]
        self.IPRS          = m["IPRS"]
        self.PRS           = m["PRS"]
        self.AR            = m["AR"]
        self.simulator     = m["simulator"]
        self.data_management = m["data_management"]

        c = ctx["csv_files"]
        self.PRS_ERROR_FILE  = c["PRS_ERROR_FILE"]
        self.PRS_DETAIL_FILE = c["PRS_DETAIL_FILE"]
        self.SCREENING_SUMMARY_FILE = c["SCREENING_SUMMARY_FILE"]
        self.COST_FILE       = c["COST_FILE"]

        lg = ctx["log"]
        self._step           = lg["step"]
        self._kv             = lg["kv"]
        self._table          = lg["table"]
        self._result         = lg["result"]
        self._warn           = lg["warn"]
        self._error          = lg["error"]
        self._append_csv_row = lg["append_csv_row"]

        self.verbose      = verbose
        self.trace_search = trace_search
        self.ResponseSurfaces = {}   # filled by run(); consumed by Phase 11

    # ────────────────────────────────────────────────────────────────────
    # Step-by-step trace printer
    # ────────────────────────────────────────────────────────────────────

    def _t(self, case, step, total, title, **kv):
        """One numbered STEP box in the per-case trace. The exact numbers each
        step receives/computes are printed as key = value lines -- this is the
        "what exactly is Phase 10 doing" instrumentation."""
        if not self.verbose:
            return
        print(f"\n  ┌─ [Case-{case}] STEP {step}/{total} · {title}")
        for k, v in kv.items():
            print(f"  │   {k:<30s} = {v}")
        print("  └" + "─" * 58)

    def _search_trace_cb(self, tol, k, order, ridge, fold_scores, worst, kept_info):
        """Per-combo callback handed to IdentifiablePRS's search (only wired
        when trace_search=True). fold_scores: list of the CV folds' max(train,
        val) scores for this combo."""
        if not self.trace_search:
            return
        scores = "  ".join(f"{s:6.2f}" for s in fold_scores)
        print(f"      [search] tol={tol} k={k:2d} order={order} ridge={ridge:.0e} "
              f"worst-fold={worst:6.2f}%   folds=[{scores}]  kept={kept_info}")

    # ────────────────────────────────────────────────────────────────────
    # Mini-DM simulation machinery (moved verbatim from run_script.py)
    # ────────────────────────────────────────────────────────────────────

    def _simulate_new_samples(self, case, prs_rows, full_rows, sel_mask, pass_id,
                              force_resim=False):
        """Simulate a mini-DM of new samples, reusing the same SM /
        data_management machinery as the main flow. Results are cached under
        Opt/Data/Outlier_Simulations/ with a fingerprint manifest, so a cached
        result is only reused when it matches THIS pass's mini-DM. Returns the
        ETA (target) values aligned with the input rows.
        cwd is assumed to be optDir on entry and is restored on exit."""
        case_int = int(case)
        sim_store = os.path.join(self.optDir, "Data", "Outlier_Simulations")
        os.makedirs(sim_store, exist_ok=True)
        sim_data_file = os.path.join(sim_store, f"sim_data_case-{case_int}_pass-{pass_id}.lst")
        meta_file = os.path.join(sim_store, f"sim_data_case-{case_int}_pass-{pass_id}.meta.json")
        fingerprint = _mini_dm_fingerprint(prs_rows)

        if not force_resim and os.path.isfile(sim_data_file) and os.path.isfile(meta_file):
            try:
                meta = json.load(open(meta_file))
            except (ValueError, OSError):
                meta = {}
            if meta.get("n_rows") == len(prs_rows) and meta.get("fingerprint") == fingerprint:
                print(f"\t[Case-{case}] pass {pass_id}: reusing cached simulation data "
                      f"(fingerprint verified)")
                ETA = [float(l.split("\t")[1]) for l in open(sim_data_file).readlines()]
                return list(np.asarray(ETA, dtype=float).ravel())
            print(f"\t[Case-{case}] pass {pass_id}: cached simulation data found but its "
                  f"fingerprint doesn't match this pass's mini-DM -- resimulating.")

        work_dir = os.path.join(self.optDir, "Outlier_Resample",
                                f"case-{case_int}", f"pass-{pass_id}")
        mech_dir = os.path.join(work_dir, "Perturbed_Mech")
        os.makedirs(mech_dir, exist_ok=True)

        OSM = self.simulator.SM(self.target_list, self.optInputs, self.unsrt_data,
                                {case_int: [list(r) for r in full_rows]}, tag="Outlier")

        yaml_loc = [os.path.join(mech_dir, f"mechanism_{i}.yaml")
                    for i in range(len(prs_rows))]
        if all(os.path.isfile(p) for p in yaml_loc) and len(yaml_loc) > 0:
            print(f"\t[Case-{case}] pass {pass_id}: mechanism YAMLs already generated")
        else:
            if self.PRS_type == "Full":
                yaml_list = OSM.getYAML_List([list(r) for r in full_rows])
            else:
                sel_rows = [list(sel_mask) for _ in range(len(prs_rows))]
                yaml_list = OSM.getYAML_List([list(r) for r in prs_rows], selection=sel_rows)
            location_mech = [mech_dir for _ in yaml_list]
            index_list = [str(i) for i in range(len(yaml_list))]
            OSM.getPerturbedMechLocation(yaml_list, location_mech, index_list)

        yaml_loc_dict = {case_int: yaml_loc}
        cwd0 = os.getcwd()
        os.chdir(work_dir)
        try:
            exec_locs = OSM.make_dir_in_parallel(yaml_loc_dict, cases=[case_int])
            os.chdir(f"case-{case_int}")
            data_sheet, failed_sim, ETA = self.data_management.generate_target_value_tables(
                exec_locs, self.target_list, case_int, self.fuel, input_=self.optInputs)
        finally:
            os.chdir(cwd0)

        open(sim_data_file, "w").write(data_sheet)
        open(os.path.join(sim_store, f"failed_sim_case-{case_int}_pass-{pass_id}.lst"),
             "w").write(failed_sim)
        json.dump({"n_rows": len(prs_rows), "fingerprint": fingerprint},
                  open(meta_file, "w"))
        return list(np.asarray(ETA, dtype=float).ravel())

    def _write_coef_cache(self, coef_path, coeff):
        """(Re)write the coefficient cache file from an explicit coeff array --
        guarantees the on-disk cache matches `best`, not a rejected trial that
        happened to write over it (see retrain_around_outlier)."""
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        with open(coef_path, "w") as f:
            f.write("Coefficients\n")
            for c in coeff:
                f.write(f"{float(c)}\n")

    # ────────────────────────────────────────────────────────────────────
    # Step 3: bad-sample screening
    # ────────────────────────────────────────────────────────────────────

    def screen_case_samples(self, case, case_index, xData_raw, yData_raw,
                            orig_row_idx, blocks, dm_case_dir, band_tolerance_eval):
        """Split one case's (design matrix, response) pair into a GOOD submatrix
        (what the PRS is fit and scored on) and a QUARANTINED remainder, and
        write the full audit trail to disk under dm_case_dir.

        Every row is re-scored against blocks -- the CURRENT band-validity test
        -- regardless of when or by which sampler version it was generated, so
        a stale DesignMatrix.csv can never silently corrupt a fit.

        Rejection reasons, checked in order (first hit wins):
          1. band_invalid  -- IPRS.violation_detail(x, blocks) exceeds
                               band_tolerance_eval.
          2. sim_sentinel  -- response equals NO_IGNITION_SENTINEL.
          3. non_finite    -- NaN or +-inf response.
          4. non_physical  -- exp(response) outside (0, 1e6] microseconds.

        Returns (xGood, yGood, stats) with the full per-reason breakdown.
        """
        xData_raw = np.asarray(xData_raw, dtype=float)
        yData_raw = np.asarray(yData_raw, dtype=float).ravel()
        orig_row_idx = np.asarray(orig_row_idx, dtype=int)
        n = len(yData_raw)
        reasons = np.array([""] * n, dtype=object)

        ratio, worst_rxn = self.IPRS.violation_detail(xData_raw, blocks)
        tol = band_tolerance_eval if band_tolerance_eval is not None else float("inf")
        band_bad = ratio > tol
        reasons[band_bad] = "band_invalid"

        is_sentinel = (reasons == "") & np.isclose(yData_raw, NO_IGNITION_SENTINEL, atol=1e-6)
        reasons[is_sentinel] = "sim_sentinel"

        non_finite = (reasons == "") & (~np.isfinite(yData_raw))
        reasons[non_finite] = "non_finite"

        with np.errstate(over="ignore"):
            tau_us = np.exp(np.clip(yData_raw, -700, 700))
        non_physical = (reasons == "") & ((tau_us <= 0) | (tau_us > 1e6 + 1e-6))
        reasons[non_physical] = "non_physical"

        good_mask = (reasons == "")
        os.makedirs(dm_case_dir, exist_ok=True)

        bad_idx = np.where(~good_mask)[0]
        with open(os.path.join(dm_case_dir, "failed_DM.csv"), "w") as fh:
            for i in bad_idx:
                fh.write(f"{orig_row_idx[i]}," + ",".join(f"{v}" for v in xData_raw[i]) + "\n")

        with open(os.path.join(dm_case_dir, "failed_DM_reasons.csv"), "w") as fh:
            fh.write("row_index,reason,violation_ratio,worst_reaction,response_value\n")
            for i in bad_idx:
                fh.write(f"{orig_row_idx[i]},{reasons[i]},{ratio[i]:.6f},"
                         f"{worst_rxn[i]},{yData_raw[i]:.6f}\n")

        good_idx = np.where(good_mask)[0]
        with open(os.path.join(dm_case_dir, "good_DM_index.csv"), "w") as fh:
            fh.write("row_index\n")
            for i in good_idx:
                fh.write(f"{orig_row_idx[i]}\n")

        stats = {
            "n_generated_kept_for_sim": n,
            "n_band_invalid": int(band_bad.sum()),
            "n_sentinel": int(is_sentinel.sum()),
            "n_non_finite": int(non_finite.sum()),
            "n_non_physical": int(non_physical.sum()),
            "n_good": int(good_mask.sum()),
            "good_fraction_pct": float(100.0 * good_mask.sum() / n) if n else 0.0,
            "worst_violation_ratio": float(ratio.max()) if n else 0.0,
            "median_violation_ratio": float(np.median(ratio)) if n else 0.0,
        }
        with open(os.path.join(dm_case_dir, "screening_summary.json"), "w") as fh:
            json.dump(stats, fh, indent=2)

        return xData_raw[good_mask], yData_raw[good_mask], stats

    # ────────────────────────────────────────────────────────────────────
    # Step 3b (experiment arms): correlation-aware augmentation
    # ────────────────────────────────────────────────────────────────────

    def _correlation_augment(self, case, case_index, xData, yData, blocks,
                             selectedParams):
        """Correlation-aware resampling (the "resampling" / "resampling_transform"
        arms): generate band-valid, deduped candidate rows whose per-reaction
        perturbation covariance follows the case's band-valid correlation
        structure (CorrelationResampler), save the mini-DM + verification
        stats + correlation-comparison plots, and -- in live mode
        (dry_run=False) -- simulate them via the cached _simulate_new_samples
        machinery and augment the training set.

        Returns (xData_aug, yData_aug, aug_stats). Dry-run returns the input
        data unchanged (zero simulation calls)."""
        rc = self.experiment_cfg["resampling"]
        arm = ARM_FOLDER[self.mode]
        resampler = CorrelationResampler(
            self.unsrt_data, np.asarray(selectedParams, dtype=int),
            self.PRS_type, blocks, self.IPRS,
            empirical_weight=rc["empirical_weight"],
            band_tolerance=rc["band_tolerance"], qmc=rc["qmc"],
            seed=rc["seed"], dedup_eps=rc["dedup_eps"],
            center_mode=rc["center_mode"])
        prs_rows, full_rows, sel_mask, gen_stats = resampler.generate(
            rc["n_new"], np.asarray(xData, dtype=float))

        sim_store = os.path.join(self.optDir, "Data", "Outlier_Simulations")
        os.makedirs(sim_store, exist_ok=True)
        np.savetxt(os.path.join(sim_store, f"CorrelationAug_DM_case-{case_index}.csv"),
                   prs_rows, delimiter=",")
        with open(os.path.join(sim_store, f"CorrelationAug_stats_case-{case_index}.json"),
                  "w") as fh:
            json.dump(gen_stats, fh, indent=2)
        try:
            resampler.plot_correlation_comparison(
                np.asarray(xData, dtype=float), prs_rows,
                os.path.join(self.home_dir, arm), case_index)
        except Exception as exc:
            self._warn(f"[Case-{case}] correlation-comparison plots failed: {exc!r}")

        self._step(f"Correlation-aware augmentation: {gen_stats['n_accepted']} band-valid "
                   f"candidates accepted from {gen_stats['n_generated']} generated "
                   f"(reject_rate {100 * gen_stats['reject_rate']:.1f}%, "
                   f"median violation {gen_stats['median_violation_ratio']:.3f})")

        aug_stats = {"candidates_generated": gen_stats["n_generated"],
                     "candidates_accepted": gen_stats["n_accepted"],
                     "reject_rate": gen_stats["reject_rate"],
                     "sims_run": 0, "sims_kept": 0,
                     "corr_dist_sigma_draw": gen_stats["corr_dist_sigma_draw"],
                     "corr_dist_original": gen_stats["corr_dist_original"],
                     "coverage_ratio_mean": gen_stats["coverage_ratio_mean"]}
        if rc["dry_run"]:
            print(f"\t[Case-{case}] dry_run=True -- candidates generated and verified, "
                  f"NO simulation (set 'do simulation: true' in the Stats block "
                  f"to simulate).")
            return xData, yData, aug_stats
        if len(prs_rows) == 0:
            self._warn(f"[Case-{case}] no band-valid candidates -- augmentation skipped.")
            return xData, yData, aug_stats

        y_new = self._simulate_new_samples(case, prs_rows, full_rows, sel_mask, "aug")
        aug_stats["sims_run"] = len(prs_rows)
        # screen the simulated y with the same physical predicates as
        # screen_case_samples (Stage 3 historically skipped this -- noted in
        # Readme.md; the augmentation step does NOT).
        y_new = np.asarray(y_new, dtype=float).ravel()
        if len(y_new) != len(prs_rows):
            self._warn(f"[Case-{case}] simulated {len(y_new)} of {len(prs_rows)} "
                       f"augmentation points; truncating to matched rows.")
            m = min(len(y_new), len(prs_rows))
            prs_rows, y_new = prs_rows[:m], y_new[:m]
        keep = (~np.isclose(y_new, NO_IGNITION_SENTINEL, atol=1e-6)) & \
               np.isfinite(y_new)
        with np.errstate(over="ignore"):
            tau_us = np.exp(np.clip(y_new, -700, 700))
        keep &= (tau_us > 0) & (tau_us <= 1e6 + 1e-6)
        aug_stats["sims_kept"] = int(keep.sum())
        print(f"\t[Case-{case}] augmentation: simulated {len(prs_rows)}, kept "
              f"{int(keep.sum())} after sentinel/non-finite/non-physical screening.")
        if keep.any():
            xData = np.vstack([np.asarray(xData, dtype=float),
                               np.asarray(prs_rows, dtype=float)[keep]])
            yData = np.concatenate([np.asarray(yData, dtype=float).ravel(),
                                    y_new[keep]])
        return xData, yData, aug_stats

    # ────────────────────────────────────────────────────────────────────
    # Step 7: Stage-3 adaptive outlier retraining
    # ────────────────────────────────────────────────────────────────────

    def retrain_around_outlier(self, case, case_index, Response, xTrain, yTrain,
                               xVal, yVal, xData, selectedParams, blocks, search,
                               resampler):
        """Adaptive densification around PRS VALIDATION outliers, then refit.

        Outlier DETECTION and the accept/reject decision use ONLY xVal/yVal --
        the caller's final untouched hold-out is evaluated exactly once, after
        this function returns, purely for reporting.

        Each pass: build a focused mini-DM around the current worst VALIDATION
        outliers (via `resampler`), simulate the new points, augment the
        training set, refit a fresh PLS feature map (k/order/ridge stay FIXED
        at whatever the initial fit_prs_with_*search chose) + PRS, and score
        on xVal/yVal.

        Accept rule (two bug fixes stacked, see CHANGELOG): a pass is kept ONLY
        if max(train, val) does not worsen -- the original unconditionally did
        `best = refit`, and a validation-only rule let TRAIN error drift upward
        across passes. xVal is filtered to band-valid rows first (a band-invalid
        row is not a fair signal for either detection or accept/reject).

        Returns (best_response, resampler, stage3_stats).
        """
        if self.design_type != "A1+B1+C1":
            print(f"[Case-{case}] retrain_around_outlier called under "
                  f"design_type={self.design_type!r} -- returning the caller's Response "
                  f"unchanged (Stage 3 is supported only for 'A1+B1+C1').")
            return Response, resampler, {"passes_attempted": 0, "passes_accepted": 0,
                                         "sims_run": 0, "sims_kept": 0, "stalled_passes": 0,
                                         "aborted_reason": f"design={self.design_type!r}"}
        if self.prs_fit_cfg["band_tolerance_eval"] is not None:
            _vmask = self.IPRS.band_valid_mask(xVal, blocks,
                                               self.prs_fit_cfg["band_tolerance_eval"])
            if _vmask.any():
                xVal, yVal = xVal[_vmask], yVal[_vmask]
        target = self.outlier_cfg["target_fraction"] * self.MRE_criteria
        # stage3_with_transform escape hatch: when this case runs through an
        # encoder, Stage 3 operates fully in z-space -- the target becomes
        # the z-equivalent threshold and every simulated y is encoded before
        # it joins the (encoded) training stack.
        enc_stage3 = self._encoders.get(case)
        if enc_stage3 is not None and enc_stage3.mre_equivalent_z(target) is not None:
            target = enc_stage3.mre_equivalent_z(target)
        n_train0 = len(xTrain)
        X_aug = np.asarray(xTrain, dtype=float)
        y_aug = np.asarray(yTrain, dtype=float).ravel()
        xVal = np.asarray(xVal, dtype=float)
        yVal = np.asarray(yVal, dtype=float).ravel()
        best = Response
        best_train_score, best_val_score, _ = Response.test(xVal, yVal)
        best_score = max(best_train_score, best_val_score)
        coef_path = os.path.join(self.optDir, "Data", "ResponseSurface",
                                 f"responsecoef_case-{case_index}.csv")
        mode = self.outlier_cfg["sample_mode"]

        min_outliers = max(int(self.outlier_cfg.get("min_outlier_count", 5)),
                           int(np.ceil(self.outlier_cfg.get("min_outlier_frac", 0.005)
                                       * len(yVal))))
        if self.verbose:
            print(f"\t[Case-{case}] Stage-3 entry: val points={len(yVal)}, "
                  f"best max(train,val)={best_score:.3f}%, target={target:.3f}%, "
                  f"min_outliers={min_outliers}, mode={mode}")

        stage3_stats = {"passes_attempted": 0, "passes_accepted": 0,
                        "sims_run": 0, "sims_kept": 0, "stalled_passes": 0,
                        "aborted_reason": None}
        stall_count = 0
        prev_signature = None
        oversample_boost = 1

        it = 0
        while it < self.outlier_cfg["max_iter"]:
            rel = np.asarray(best.error_testing_relative, dtype=float)
            worst_abs = float(np.abs(rel).max()) if len(rel) else 0.0
            still_failing = worst_abs > self.MRE_criteria
            below_min_passes = stage3_stats["passes_attempted"] < self.outlier_cfg["min_passes"]
            oidx = self.AR.find_outliers(rel, target, force_worst=False)
            if oidx.size == 0:
                if still_failing and below_min_passes:
                    oidx = self.AR.find_outliers(rel, target, force_worst=True)
                else:
                    print(f"\t[Case-{case}] worst VALIDATION error {worst_abs:.3f}% already "
                          f"<= target {target:.3f}% -- no re-training needed.")
                    break
            if it == 0 and 0 < oidx.size < min_outliers and not still_failing:
                print(f"\t[Case-{case}] only {oidx.size} point(s) above target "
                      f"(< min_outlier threshold {min_outliers}) and case already "
                      f"clears MRE_criteria -- likely validation-split noise; "
                      f"skipping further retraining.")
                break
            if self.verbose:
                print(f"\t[Case-{case}] pass {it + 1}: {oidx.size} outliers above "
                      f"{target:.3f}% on validation (worst |err|={worst_abs:.3f}%)")

            resampler.rng = np.random.default_rng(self.outlier_cfg["seed"] + it)
            try:
                outlier_rows = xVal[oidx]
                weights = np.abs(rel[oidx])
                dopt_info = None
                if mode == "local":
                    prs_rows, full_rows, sel_mask, row_class = resampler.resample(outlier_rows, xData)
                elif mode == "dm_global":
                    prs_rows, full_rows, sel_mask, row_class = resampler.resample_global(outlier_rows, xData)
                elif mode == "gap_fill":
                    ref_rows = (np.vstack([xData, X_aug[n_train0:]])
                                if len(X_aug) > n_train0 else xData)
                    prs_rows, full_rows, sel_mask, row_class = resampler.resample_gap_fill(
                        outlier_rows, ref_rows)
                elif mode == "hybrid":
                    prs_rows, full_rows, sel_mask, row_class = resampler.resample_hybrid(
                        outlier_rows, xData, local_fraction=self.outlier_cfg["local_fraction"])
                else:  # "d_optimal" (default) -- band-valid candidates, greedy D-optimal pick
                    prs_rows, full_rows, sel_mask, row_class, dopt_info = resampler.resample_d_optimal(
                        outlier_rows, xData, X_aug, best.build_basis,
                        target_weights=weights, local_fraction=self.outlier_cfg["local_fraction"],
                        oversample_boost=oversample_boost)

                if mode == "d_optimal" and dopt_info is not None:
                    sig = dopt_info.get("selected_signature", ())
                    stalled = (dopt_info["n_selected"] == 0
                               or (sig and prev_signature is not None and sig == prev_signature))
                    if stalled:
                        stall_count += 1
                        stage3_stats["stalled_passes"] += 1
                        print(f"\t[Case-{case}] pass stalled ({stall_count}/{self.outlier_cfg['stall_patience']}): "
                              f"D-optimal selected {dopt_info['n_selected']} candidate(s) from "
                              f"{dopt_info['n_candidates_band_valid']}/{dopt_info['n_candidates_generated']} "
                              f"band-valid (reject_rate {100*dopt_info['reject_rate']:.1f}%)"
                              + (", identical to the previous pass's selection"
                                 if sig and sig == prev_signature else "")
                              + " -- no simulation run for this pass.")
                        if stall_count >= self.outlier_cfg["stall_patience"]:
                            stage3_stats["aborted_reason"] = (
                                f"stalled {stall_count} consecutive passes (candidate pool "
                                f"exhausted -- check class-pool provenance for this case)")
                            print(f"\t[Case-{case}] aborting Stage 3: {stage3_stats['aborted_reason']}.")
                            break
                        oversample_boost = min(oversample_boost * 4, 64)
                        it += 1
                        continue
                    else:
                        stall_count = 0
                        oversample_boost = 1
                        if sig:
                            prev_signature = sig

                print(f"\n[Case-{case}] adaptive re-training pass "
                      f"{stage3_stats['passes_attempted']+1}/{self.outlier_cfg['max_iter']} "
                      f"(attempt {it+1}) [{mode}]: {oidx.size} point(s) above target on "
                      f"VALIDATION, worst |err|={worst_abs:.3f}% "
                      f"(target {target:.3f}% = {self.outlier_cfg['target_fraction']}x MRE "
                      f"{self.MRE_criteria}%)")
                if dopt_info is not None:
                    print(f"\t[Case-{case}] D-optimal candidates: band-valid "
                          f"{dopt_info['n_candidates_band_valid']}/{dopt_info['n_candidates_generated']} "
                          f"(reject_rate {100*dopt_info['reject_rate']:.1f}%), "
                          f"selected {dopt_info['n_selected']}/{dopt_info['n_budget']}")

                stage3_stats["passes_attempted"] += 1

                sim_store = os.path.join(self.optDir, "Data", "Outlier_Simulations")
                os.makedirs(sim_store, exist_ok=True)
                row_class = np.asarray(row_class)
                np.savetxt(os.path.join(sim_store, f"Outlier_DM_case-{case}_pass-{it+1}.csv"),
                           prs_rows, delimiter=",")
                for lab in sorted(set(row_class)):
                    np.savetxt(os.path.join(sim_store,
                                            f"Outlier_DM_case-{case}_pass-{it+1}_{lab}type.csv"),
                               prs_rows[row_class == lab], delimiter=",")
                print(f"\t[Case-{case}] mini-DM by class: " +
                      ", ".join(f"{lab}={int((row_class==lab).sum())}"
                                for lab in sorted(set(row_class))))
                y_new = self._simulate_new_samples(case, prs_rows, full_rows, sel_mask,
                                                   it + 1, force_resim=self.outlier_cfg["force_resim"])
                stage3_stats["sims_run"] += len(prs_rows)
                if len(y_new) != len(prs_rows):
                    self._warn(f"[Case-{case}] simulated {len(y_new)} of {len(prs_rows)} new "
                               f"points; truncating to matched rows.")
                    m = min(len(y_new), len(prs_rows))
                    prs_rows, y_new = prs_rows[:m], y_new[:m]

                if self.prs_fit_cfg["band_tolerance_eval"] is not None:
                    _new_valid = self.IPRS.band_valid_mask(
                        prs_rows, blocks, self.prs_fit_cfg["band_tolerance_eval"])
                    stage3_stats["sims_kept"] += int(_new_valid.sum())
                else:
                    stage3_stats["sims_kept"] += len(prs_rows)

                if self._encoders.get(case) is not None:
                    y_new = self._encoders[case].encode(y_new)
                X_trial = np.vstack([X_aug, prs_rows])
                y_trial = np.concatenate([y_aug, np.asarray(y_new, dtype=float).ravel()])

                fmap_trial = self.IPRS.fit_feature_map(
                    X_trial, y_trial, blocks, xData.shape[1], k=search["k"])
                if os.path.isfile(coef_path):
                    os.remove(coef_path)
                trial = self.PRS.ResponseSurface(
                    X_trial, y_trial, case, case_index,
                    responseOrder=search["order"], prs_type=self.PRS_type,
                    selected_params=selectedParams, criteria=self.MRE_criteria,
                    feature_map=fmap_trial, ridge_alpha=search["ridge_alpha"])
                trial.create_response_surface()
                mtr, mva, crit = trial.test(xVal, yVal)
                trial_score = max(mtr, mva)
                print(f"\t[Case-{case}] pass {it+1}: VALIDATION max error "
                      f"{best_val_score:.3f}% -> {mva:.3f}% (train {mtr:.3f}%, "
                      f"max(train,val) {best_score:.3f}% -> {trial_score:.3f}%, "
                      f"target {target:.3f}%, +{len(prs_rows)} pts, {len(X_trial)} total)")

                if trial_score <= best_score:
                    X_aug, y_aug = X_trial, y_trial
                    best = trial
                    best_val_score = mva
                    best_train_score = mtr
                    best_score = trial_score
                    stage3_stats["passes_accepted"] += 1
                    self._append_csv_row(self.PRS_ERROR_FILE, [
                        f"Case-{case}-retrained-p{it+1}", "validation", it + 1,
                        mtr, mva, crit, "", len(yVal), 0, len(yVal),
                        "PASS" if mva <= self.MRE_criteria else "FAIL"])
                    if mva <= target:
                        print(f"\t[Case-{case}] reached target ({mva:.3f}% <= {target:.3f}%) "
                              f"after {stage3_stats['passes_attempted']} pass(es) -- stopping.")
                        break
                    if stage3_stats["passes_attempted"] >= self.outlier_cfg["max_iter"]:
                        print(f"\t[Case-{case}] max_iter ({self.outlier_cfg['max_iter']}) reached; "
                              f"best VALIDATION error {mva:.3f}% still above target {target:.3f}%.")
                else:
                    print(f"\t[Case-{case}] pass {it+1} made max(train,val) WORSE "
                          f"({trial_score:.3f}% > {best_score:.3f}%, train {mtr:.3f}% val {mva:.3f}%) "
                          f"-- discarding this pass's draw, keeping the last accepted surface, "
                          f"and retrying with a new draw" +
                          (f" (max_iter reached)."
                           if stage3_stats["passes_attempted"] >= self.outlier_cfg["max_iter"]
                           else "."))
                    if stage3_stats["passes_attempted"] >= self.outlier_cfg["max_iter"]:
                        break
            except Exception as exc:
                print(f"\t[Case-{case}] adaptive re-training FAILED: "
                      f"{type(exc).__name__}: {exc}. Keeping the last good surface.")
                traceback.print_exc()
                stage3_stats["aborted_reason"] = f"exception: {type(exc).__name__}: {exc}"
                break
            it += 1

        self._write_coef_cache(coef_path, best.coeff)
        return best, resampler, stage3_stats

    # ────────────────────────────────────────────────────────────────────
    # Per-case pipeline
    # ────────────────────────────────────────────────────────────────────

    def _run_case(self, case):
        """One case through the whole pipeline (steps 1-8 of the trace)."""
        case_index = int(case)
        total = 8

        # ── STEP 1: load this case's (design matrix, response) pair ─────────
        yData_sim = np.asarray(self.temp_sim_opt[case]).flatten()
        full_xData = np.asarray(self.design_matrix_dict[case_index])
        orig_row_idx = self.kept_row_index[case]
        if len(orig_row_idx) != len(yData_sim):
            self._warn(f"[Case-{case}] kept_row_index length ({len(orig_row_idx)}) != "
                       f"response count ({len(yData_sim)}) -- falling back to positional "
                       f"0..N-1 indexing for this case.")
            orig_row_idx = np.arange(len(yData_sim))
        xData_sim = full_xData[orig_row_idx]
        selectedParams = np.asarray(self.selected_params_dict[case_index])
        self._t(case, 1, total, "LOAD case data",
                n_response=len(yData_sim),
                n_dm_rows=len(full_xData),
                n_selected_params=int(selectedParams.sum()),
                n_full_params=len(selectedParams))

        # ── STEP 2: per-reaction band geometry (identifiable subspace) ─────
        Tc = self.target_list[case_index].temperature
        twf = self.prs_fit_cfg["temp_window_frac"]
        T_local = np.array([Tc * (1.0 - twf), Tc, Tc * (1.0 + twf)])
        resampler = self.AR.DOptimalResampler(
            self.unsrt_data, selectedParams, self.design_type, self.PRS_type,
            self.home_dir, case,
            N_new=self.outlier_cfg["N_new"],
            delta_fraction=self.outlier_cfg["delta_fraction"],
            seed=self.outlier_cfg["seed"], uns_cap=self.outlier_cfg["uns_cap"],
            shuffle_factor=self.outlier_cfg["shuffle_factor"],
            method=self.outlier_cfg["method"],
            gapfill_oversample=self.outlier_cfg["gapfill_oversample"],
            gapfill_knn_radius_frac=self.outlier_cfg["gapfill_knn_radius_frac"],
            gapfill_knn_k=self.outlier_cfg["gapfill_knn_k"],
            gapfill_shuffle_factor=self.outlier_cfg["gapfill_shuffle_factor"],
            band_tolerance=self.outlier_cfg["band_tolerance"],
            temps_for_validity={"__default__": np.linspace(300., 2500.,
                                self.outlier_cfg["validity_n_temps"])},
            acquisition_alpha=self.outlier_cfg["acquisition_alpha"],
            candidate_oversample=self.outlier_cfg["candidate_oversample"])
        layout_triples = [(lay["rxn"],
                           slice(lay["comp_offset"], lay["comp_offset"] + lay["n_cols"]),
                           lay["prs_local"])
                          for lay in resampler.layout if lay["n_cols"] > 0]

        def _sigma_of(rxn):
            L = self.unsrt_data[rxn].cholskyDeCorrelateMat
            return self.IPRS.sigma_from_cholesky(
                L, n_active=len(self.unsrt_data[rxn].activeParameters))

        blocks = self.IPRS.build_layout_blocks(
            layout_triples, _sigma_of, T_local,
            svd_tol=self.prs_fit_cfg["svd_tol"],
            fprior_mode=self.prs_fit_cfg["band_fprior"])
        self._t(case, 2, total, "BUILD band blocks",
                T_nominal=Tc, T_window_frac=twf,
                n_reaction_blocks=len(blocks),
                fprior_mode=self.prs_fit_cfg["band_fprior"],
                svd_tol=self.prs_fit_cfg["svd_tol"])

        # ── STEP 3: screen bad samples out (audit to DM_FOR_PARTIAL_PRS) ───
        dm_case_dir = os.path.join(self.home_dir, "DM_FOR_PARTIAL_PRS", str(case_index))
        xData, yData, screen_stats = self.screen_case_samples(
            case, case_index, xData_sim, yData_sim, orig_row_idx, blocks,
            dm_case_dir, self.prs_fit_cfg["band_tolerance_eval"])
        self._append_csv_row(self.SCREENING_SUMMARY_FILE, [
            f"Case-{case}", screen_stats["n_generated_kept_for_sim"],
            screen_stats["n_band_invalid"],
            self.no_of_sim[case_index] - screen_stats["n_generated_kept_for_sim"],
            screen_stats["n_sentinel"], screen_stats["n_non_finite"],
            screen_stats["n_non_physical"],
            screen_stats["n_good"], f"{screen_stats['good_fraction_pct']:.2f}",
            f"{screen_stats['worst_violation_ratio']:.3f}",
            f"{screen_stats['median_violation_ratio']:.3f}"])
        self._kv("Samples simulated", screen_stats["n_generated_kept_for_sim"])
        self._kv("Band-invalid (quarantined)", f"{screen_stats['n_band_invalid']} "
                 f"({100 - screen_stats['good_fraction_pct']:.1f}%)")
        self._kv("Good (fit + scored on)", f"{screen_stats['n_good']} "
                 f"({screen_stats['good_fraction_pct']:.1f}%)")
        self._t(case, 3, total, "SCREEN samples",
                n_in=screen_stats["n_generated_kept_for_sim"],
                n_good=screen_stats["n_good"],
                n_band_invalid=screen_stats["n_band_invalid"],
                n_sentinel=screen_stats["n_sentinel"],
                worst_violation_ratio=f"{screen_stats['worst_violation_ratio']:.3f}",
                median_violation_ratio=f"{screen_stats['median_violation_ratio']:.3f}",
                band_tolerance_eval=self.prs_fit_cfg["band_tolerance_eval"])
        if screen_stats["n_good"] < 50:
            self._warn(f"[Case-{case}] only {screen_stats['n_good']} good sample(s) -- "
                       f"PRS capacity (k) will be tightly capped by prs_max_terms_frac.")

        # ── STEP 3b: correlation-aware augmentation (resampling arms) ────────
        aug_stats = {"candidates_generated": 0, "candidates_accepted": 0,
                     "reject_rate": None, "sims_run": 0, "sims_kept": 0,
                     "corr_dist_sigma_draw": None, "corr_dist_original": None,
                     "coverage_ratio_mean": None}
        if self.mode in ("resampling", "resampling_transform"):
            xData, yData, aug_stats = self._correlation_augment(
                case, case_index, xData, yData, blocks, selectedParams)
            self._t(case, "3b", total, "CORRELATION-AWARE AUGMENT",
                    candidates_accepted=aug_stats["candidates_accepted"],
                    reject_rate=(f"{100 * aug_stats['reject_rate']:.1f}%"
                                 if aug_stats["reject_rate"] is not None else "n/a"),
                    sims_run=aug_stats["sims_run"],
                    sims_kept=aug_stats["sims_kept"],
                    dry_run=self.experiment_cfg["resampling"]["dry_run"])

        # ── STEP 4: 3-way split of GOOD rows only ────────────────────────────
        train_frac = 1.0 - self.prs_fit_cfg["val_frac"] - self.prs_fit_cfg["holdout_frac"]
        itr, ival, iho = self.IPRS.three_way_split(
            len(yData), seed_outer=self.prs_fit_cfg["split_seed"],
            seed_inner=self.prs_fit_cfg["split_seed_inner"],
            train_frac=train_frac, val_frac=self.prs_fit_cfg["val_frac"])
        xTrain, yTrainR = xData[itr], yData[itr]     # yTrainR etc. keep RAW y
        xVal,   yValR   = xData[ival], yData[ival]
        xHold,  yHoldR  = xData[iho], yData[iho]
        self._t(case, 4, total, "SPLIT train/val/holdout",
                n_train=len(xTrain), n_val=len(xVal), n_holdout=len(xHold),
                seed_outer=self.prs_fit_cfg["split_seed"],
                seed_inner=self.prs_fit_cfg["split_seed_inner"])

        # ── STEP 4b: encode the response (transform arms) ────────────────────
        # Screening (STEP 3) and augmentation (STEP 3b) always operate on RAW
        # y = ln(tau); the encoder is fit and applied only after the split.
        # Everything downstream (search / fit / tests) then sees z, and every
        # y-space report is produced by the decode-patch helpers.
        encoder = None
        yTrain, yVal, yHold = yTrainR, yValR, yHoldR
        if self.mode in ("transform", "resampling_transform"):
            tc = self.experiment_cfg["transform"]
            encoder = ResponseEncoder(variant=tc["variant"], P=tc["P"],
                                      p_lo=tc.get("p_lo", 0.02),
                                      standardize=tc["standardize"],
                                      rank_fit_on=tc["rank_fit_on"],
                                      criteria_y=self.MRE_criteria)
            fit_idx = (np.concatenate([itr, ival])
                       if tc["rank_fit_on"] == "train_val" else None)
            encoder.fit(yData, fit_idx=fit_idx)
            enc_dir = os.path.join(self.optDir, "Data", "Encoding_Decoding")
            encoder.save(enc_dir, case_index)
            self._encoders[case] = encoder
            zData = encoder.encode(yData)
            yTrain, yVal, yHold = zData[itr], zData[ival], zData[iho]
            self._t(case, "4b", total, "ENCODE response",
                    variant=encoder.variant, P=encoder.P,
                    p_lo=(encoder.p_lo
                          if encoder.variant == "holder_linear" else "n/a"),
                    z_range=f"[{zData.min():.4f}, {zData.max():.4f}]",
                    criteria_z_equiv=(f"{encoder.criteria_z_equiv:.4f}%"
                                      if encoder.criteria_z_equiv is not None
                                      else "n/a"),
                    enc_dir=os.path.join("Opt", "Data", "Encoding_Decoding"))

        # ── STEP 5: hyperparameter search (Stage 1+2) ────────────────────────
        if self.prs_fit_cfg["search_method"] == "cv_kfold":
            X_search_pool = np.vstack([xTrain, xVal])
            y_search_pool = np.concatenate([yTrain, yVal])
            search = self.IPRS.fit_prs_with_cv_search(
                self.PRS.ResponseSurface, X_search_pool, y_search_pool, blocks,
                xData.shape[1], case, case_index, self.PRS_type, selectedParams,
                self.MRE_criteria, cache_dir=self.optDir,
                tol_grid=self.prs_fit_cfg["screening_tolerances"],
                k_grid=self.prs_fit_cfg["k_grid"],
                order_grid=self.prs_fit_cfg["order_grid"],
                ridge_grid=self.prs_fit_cfg["ridge_grid"],
                max_terms_frac=self.prs_fit_cfg["max_terms_frac"],
                cv_folds=self.prs_fit_cfg["cv_folds"],
                cv_seed=self.prs_fit_cfg["cv_seed"],
                band_tolerance_eval=self.prs_fit_cfg["band_tolerance_eval"],
                trace_cb=self._search_trace_cb if self.trace_search else None)
            self._kv("Stage1+2 search", f"cv_kfold/{self.prs_fit_cfg['cv_folds']}-fold")
            self._kv("  tolerance / k / order / ridge", f"{search['tolerance']} / "
                     f"{search['k']} / {search['order']} / {search['ridge_alpha']:.0e}")
            self._kv("  rows kept / pool", f"{search['n_kept']}/{len(X_search_pool)}")
            self._kv("  worst-fold score", f"{search['cv_score']:.3f}%")
            _skipped = sum(search["skip_counts"].values())
            if _skipped:
                self._kv("  combos skipped by guard",
                         f"{_skipped} ({search['skip_counts']})")
            search["train_pct"] = search["val_pct"] = search["cv_score"]
        else:
            search = self.IPRS.fit_prs_with_search(
                self.PRS.ResponseSurface, xTrain, yTrain, xVal, yVal, blocks,
                xData.shape[1], case, case_index, self.PRS_type, selectedParams,
                self.MRE_criteria, cache_dir=self.optDir,
                tol_grid=self.prs_fit_cfg["screening_tolerances"],
                k_grid=self.prs_fit_cfg["k_grid"],
                order_grid=self.prs_fit_cfg["order_grid"],
                ridge_grid=self.prs_fit_cfg["ridge_grid"],
                max_terms_frac=self.prs_fit_cfg["max_terms_frac"],
                band_tolerance_eval=self.prs_fit_cfg["band_tolerance_eval"],
                trace_cb=self._search_trace_cb if self.trace_search else None)
            self._kv("Stage1+2 search", "single_split")
            self._kv("  tolerance / k / order / ridge", f"{search['tolerance']} / "
                     f"{search['k']} / {search['order']} / {search['ridge_alpha']:.0e}")
            self._kv("  rows kept / train", f"{search['n_kept']}/{len(xTrain)}")
            self._kv("  train / val score", f"{search['train_pct']:.3f}% / "
                     f"{search['val_pct']:.3f}%")
            _skipped = sum(search["skip_counts"].values())
            if _skipped:
                self._kv("  combos skipped by guard",
                         f"{_skipped} ({search['skip_counts']})")
        self._t(case, 5, total, "SEARCH hyperparameters",
                method=self.prs_fit_cfg["search_method"],
                tolerance=search["tolerance"], k=search["k"],
                order=search["order"], ridge_alpha=f"{search['ridge_alpha']:.0e}",
                score=f"{search.get('cv_score', max(search['train_pct'], search['val_pct'])):.3f}%")

        # ── STEP 6: final fit on the winning combo ───────────────────────────
        if self.prs_fit_cfg["search_method"] == "cv_kfold":
            keep_mask, _ = self.IPRS.screen_rows(xTrain, blocks, search["tolerance"])
        else:
            keep_mask = search["keep_mask"]
        xTrainK, yTrainK = xTrain[keep_mask], yTrain[keep_mask]
        feature_map = self.IPRS.fit_feature_map(xTrainK, yTrainK, blocks,
                                                xData.shape[1], k=search["k"])
        coef_path = os.path.join(self.optDir, "Data", "ResponseSurface",
                                 f"responsecoef_case-{case_index}.csv")
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        Response = self.PRS.ResponseSurface(
            xTrainK, yTrainK, case, case_index,
            responseOrder=search["order"], prs_type=self.PRS_type,
            selected_params=selectedParams, criteria=self.MRE_criteria,
            feature_map=feature_map, ridge_alpha=search["ridge_alpha"])
        Response.create_response_surface()

        # transform arms: capture the z-space numbers (reported in
        # prs_experiment_results.csv), then patch the train diagnostics to
        # y-space and score every later test through the decoder.
        z_train_mre = z_val_mre = z_hold_mre = None
        if encoder is not None:
            if encoder.variant == "rank":
                # rank z in (0,1): relative error diverges near z->0, so the
                # rank z-metric is RMSE (ResponseEncoder.z_metric).
                Response.test(xVal, yVal)
                z_val_mre = encoder.z_metric(yValR, np.asarray(Response.y_Test_Predict))
                z_train_mre = encoder.z_metric(yTrainR[keep_mask],
                                               np.asarray(Response.resFramWrk))
            else:
                _, z_val_mre, _ = Response.test(xVal, yVal)  # z-space score
                z_train_mre = float(Response.MaxError)       # z-space training
            _decode_patch_train(Response, yTrainR[keep_mask], encoder)

        def _test(xt, ytr):
            if encoder is not None:
                return _score_decoded(Response, xt, ytr, encoder)
            return Response.test(xt, ytr)

        Max_Train_Error, Max_Val_Error, Criterion = _test(xVal, yValR)
        self._t(case, 6, total, "FIT final ResponseSurface",
                n_train_rows_kept=len(xTrainK),
                feature_map_k=feature_map.k,
                order=search["order"], ridge_alpha=f"{search['ridge_alpha']:.0e}",
                train_error=f"{Max_Train_Error:.3f}%",
                val_error=f"{Max_Val_Error:.3f}%",
                **({"z_train_mre": f"{z_train_mre:.3f}%",
                    "z_val_mre": f"{z_val_mre:.3f}%"} if encoder is not None else {}))

        # ── STEP 7: Stage-3 trigger + adaptive retraining ────────────────────
        if self.prs_fit_cfg["band_tolerance_eval"] is not None:
            _trigger_vmask = self.IPRS.band_valid_mask(
                xVal, blocks, self.prs_fit_cfg["band_tolerance_eval"])
            _trigger_rel = np.asarray(Response.error_testing_relative, dtype=float)
            _stage3_val_error = (float(np.max(np.abs(_trigger_rel[_trigger_vmask])))
                                 if _trigger_vmask.any() else Max_Val_Error)
        else:
            _stage3_val_error = Max_Val_Error
        _stage3_needed = max(Max_Train_Error, _stage3_val_error) > self.MRE_criteria

        stage3_stats = {"passes_attempted": 0, "passes_accepted": 0,
                        "sims_run": 0, "sims_kept": 0, "stalled_passes": 0,
                        "aborted_reason": None}
        if (self.outlier_cfg["enable"] and self.design_type == "A1+B1+C1"
                and _stage3_needed
                and (self.mode == "original"
                     or self.experiment_cfg["stage3_with_transform"])):
            self._step(f"Stage 3: adaptive resampling (min {self.outlier_cfg['min_passes']} "
                       f"passes, max {self.outlier_cfg['max_iter']})")
            self._t(case, 7, total, "STAGE 3 retrain (triggered)",
                    band_valid_val_error=f"{_stage3_val_error:.3f}%",
                    mre_criteria=self.MRE_criteria,
                    mode=self.outlier_cfg["sample_mode"],
                    max_iter=self.outlier_cfg["max_iter"])
            try:
                Response, resampler, stage3_stats = self.retrain_around_outlier(
                    case, case_index, Response, xTrainK, yTrainK, xVal, yVal,
                    xData, selectedParams, blocks, search, resampler)
            except Exception as e:
                self._warn(f"[Case-{case}] Stage 3 raised {e!r} -- keeping the Stage 1+2 "
                           f"surface and reporting zero Stage-3 passes for this case.")
                stage3_stats["aborted_reason"] = f"exception: {e!r}"
            self._kv("Stage 3 passes attempted/accepted",
                     f"{stage3_stats['passes_attempted']}/{stage3_stats['passes_accepted']}" +
                     (f" (aborted: {stage3_stats['aborted_reason']})"
                      if stage3_stats["aborted_reason"] else ""))
            self._kv("Stage 3 new sims run/kept",
                     f"{stage3_stats['sims_run']}/{stage3_stats['sims_kept']}")
        elif (self.outlier_cfg["enable"] and self.design_type == "A1+B1+C1"
                and _stage3_needed):
            # experiment mode without the escape hatch: Stage-3 outlier
            # thresholds and accept/reject scores would mix z- and y-space.
            self._warn(f"[Case-{case}] Stage 3 skipped: experiment mode {self.mode!r} "
                       f"with stage3_with_transform=false (see Readme.md).")
            self._t(case, 7, total, "STAGE 3 retrain (skipped)",
                    reason=f"mode={self.mode!r}, stage3_with_transform=false")
        elif self.outlier_cfg["enable"] and self.design_type != "A1+B1+C1" and _stage3_needed:
            self._warn(f"[Case-{case}] Stage 3 D-optimal resampling SKIPPED: "
                       f"design_type={self.design_type!r} (A-factor-only optimization). "
                       f"Resampling is supported only for the 'A1+B1+C1' design.")
            self._t(case, 7, total, "STAGE 3 retrain (skipped)",
                    reason=f"design={self.design_type!r}")
        else:
            self._t(case, 7, total, "STAGE 3 retrain (not needed)",
                    band_valid_val_error=f"{_stage3_val_error:.3f}%",
                    mre_criteria=self.MRE_criteria)

        # ── STEP 8: one-time untouched hold-out verdict + ledgers ────────────
        if encoder is not None:
            # z-space holdout number for the experiment CSV, then y-space
            Response.test(xHold, yHold)
            z_hold_mre = (encoder.z_metric(yHoldR, np.asarray(Response.y_Test_Predict))
                          if encoder.variant == "rank"
                          else float(Response.ytestMaxError))
        Max_Train_Final, Max_Holdout_Final, Criterion = _test(xHold, yHoldR)

        # Plotting must never cost a case its verdict -- a Matplotlib/IO
        # failure here is cosmetic, not a fit failure, so it is caught and
        # warned rather than left to propagate into the case-level exception
        # handler (which would report the whole case as "FAIL (fit error)"
        # even though the response surface itself fit fine).
        try:
            if self.mode == "original":
                Response.plot(case_index)
            else:
                arm_dir = os.path.join(self.home_dir, ARM_FOLDER[self.mode])
                os.makedirs(arm_dir, exist_ok=True)
                ratio_hold, _ = self.IPRS.violation_detail(xHold, blocks)
                Response.plot_parity_scaled(
                    ratio=ratio_hold,
                    y_global=(float(np.min(yHoldR)), float(np.max(yHoldR))),
                    location=arm_dir, tag=f"_arm-{self.mode}")
                if encoder is not None:
                    plot_transform_profile(
                        encoder, yHoldR, encoder.encode(yHoldR),
                        os.path.join(arm_dir, f"transform_profile_case-{case_index}.pdf"),
                        case_index)
        except Exception as _plot_exc:
            self._warn(f"[Case-{case}] plotting FAILED: "
                       f"{type(_plot_exc).__name__}: {_plot_exc} -- verdict "
                       f"and ledgers are unaffected, only this case's plot is missing.")

        if self.prs_fit_cfg["band_tolerance_eval"] is not None:
            hold_valid_mask = self.IPRS.band_valid_mask(
                xHold, blocks, self.prs_fit_cfg["band_tolerance_eval"])
            rel = np.asarray(Response.error_testing_relative, dtype=float)
            Max_Holdout_BandValid = (float(np.max(np.abs(rel[hold_valid_mask])))
                                     if hold_valid_mask.any() else Max_Holdout_Final)
            gating_holdout = Max_Holdout_BandValid
            _mtr_v, _mva_v, _ = _test(xVal, yValR)
            val_valid_mask = self.IPRS.band_valid_mask(
                xVal, blocks, self.prs_fit_cfg["band_tolerance_eval"])
            relv = np.asarray(Response.error_testing_relative, dtype=float)
            Max_Val_BandValid = (float(np.max(np.abs(relv[val_valid_mask])))
                                 if val_valid_mask.any() else _mva_v)
            _test(xHold, yHoldR)
        else:
            Max_Holdout_BandValid = Max_Holdout_Final
            gating_holdout = Max_Holdout_Final
            Max_Val_BandValid = Max_Val_Error

        verdict = "PASS" if max(Max_Train_Final, gating_holdout) <= self.MRE_criteria else "FAIL"

        self._append_csv_row(self.PRS_ERROR_FILE, [
            f"Case-{case}", "holdout", "final",
            Max_Train_Final, Max_Holdout_Final, Criterion, Max_Holdout_BandValid,
            screen_stats["n_good"],
            screen_stats["n_band_invalid"] + screen_stats["n_sentinel"]
            + screen_stats["n_non_finite"] + screen_stats["n_non_physical"],
            search["n_kept"], verdict])
        self._append_csv_row(self.PRS_DETAIL_FILE, [
            case, search["tolerance"], search["k"], search["order"],
            search["ridge_alpha"], self.no_of_sim[case_index],
            screen_stats["n_good"], len(xTrain), search["n_kept"],
            Max_Train_Final, Max_Val_Error, Max_Holdout_Final,
            Max_Val_BandValid, Max_Holdout_BandValid])

        total_sims_run = self.no_of_sim[case_index] + stage3_stats["sims_run"]
        total_sims_useful = screen_stats["n_good"] + stage3_stats["sims_kept"]
        self._append_csv_row(self.COST_FILE, [
            f"Case-{case}",
            self.cost_estimate_dict[case_index]["Total_Rxns"],
            self.cost_estimate_dict[case_index]["Active_Rxns"],
            self.cost_estimate_dict[case_index]["Cost_Estimate_pPRS"],
            self.cost_estimate_dict[case_index]["Cost_Estimate_FullPRS"],
            self.no_of_sim[case_index], screen_stats["n_generated_kept_for_sim"],
            self.no_of_sim[case_index] - screen_stats["n_generated_kept_for_sim"],
            screen_stats["n_sentinel"], screen_stats["n_non_finite"],
            screen_stats["n_non_physical"],
            screen_stats["n_band_invalid"], screen_stats["n_good"],
            stage3_stats["passes_attempted"], stage3_stats["passes_accepted"],
            stage3_stats["sims_run"], stage3_stats["sims_kept"],
            total_sims_run, total_sims_useful,
            f"{100.0 * total_sims_useful / total_sims_run:.2f}" if total_sims_run else "0.00"])

        self._result(verdict == "PASS",
                     f"[Case-{case}] FINAL: train={Max_Train_Final:.3f}%  "
                     f"holdout(band-valid)={Max_Holdout_BandValid:.3f}%  "
                     f"holdout(all)={Max_Holdout_Final:.3f}%  {verdict}  "
                     f"(criteria {self.MRE_criteria}%, {screen_stats['n_good']} good / "
                     f"{self.no_of_sim[case_index]} generated samples)")
        self._t(case, 8, total, "HOLD-OUT verdict",
                train=f"{Max_Train_Final:.3f}%",
                holdout_bandvalid=f"{Max_Holdout_BandValid:.3f}%",
                holdout_all=f"{Max_Holdout_Final:.3f}%",
                verdict=verdict, criteria=f"{self.MRE_criteria}%")

        # experiment ledger (z-space numbers + arm metadata; y-space verdicts
        # stay in the existing ledgers only)
        self._exp_rows.append({
            "case": case, "mode": self.mode,
            "variant": (encoder.variant if encoder is not None else ""),
            "P": ((encoder.p_lo if encoder.variant == "holder_linear"
                   else encoder.P) if encoder is not None else ""),
            "n_train": len(xTrainK),
            "n_aug": aug_stats["candidates_accepted"],
            "cv_score_z": (search.get("cv_score")
                           if search.get("cv_score") is not None
                           else max(search["train_pct"], search["val_pct"])),
            "train_mre_z": z_train_mre, "val_mre_z": z_val_mre,
            "holdout_mre_z": z_hold_mre,
            "train_mre_y": Max_Train_Final,
            "holdout_mre_y": Max_Holdout_BandValid,
            "holdout_mre_y_all": Max_Holdout_Final,
            "verdict_y": verdict,
            "aug_sims_run": aug_stats["sims_run"],
            "aug_sims_kept": aug_stats["sims_kept"],
            "encoder_file": (f"encoder_case-{case_index}_{encoder.variant}.json"
                             if encoder is not None else ""),
        })
        return (case, Max_Train_Final, Max_Holdout_Final, Max_Holdout_BandValid,
                verdict, Response)

    # ────────────────────────────────────────────────────────────────────
    # run(): the per-case loop + closing summary (the whole Phase 10)
    # ────────────────────────────────────────────────────────────────────

    def run(self):
        """Execute Phase 10 for every case in temp_sim_opt. Fills
        self.ResponseSurfaces (consumed by Phase 11's optimizer) and returns
        it. Ends with the per-case summary table. On exit, cwd is restored to
        the study root (os.chdir("..") from optDir), exactly like the original
        inline block."""
        _n_cases_total = len(list(self.temp_sim_opt))
        _n_case_done = 0
        _case_verdicts = []

        for case in self.temp_sim_opt:
            case_index = int(case)
            _n_case_done += 1
            self._step(f"Case {case} ({_n_case_done}/{_n_cases_total})")
            try:
                out = self._run_case(case)
                _case_verdicts.append(out[:5])
                self.ResponseSurfaces[int(out[0])] = out[5]
                del out
            except Exception as _case_exc:
                # CASE-LEVEL ROBUSTNESS NET: a single case's fit genuinely
                # failing must not abort the entire run -- every case gets a
                # verdict, including a case that could not be fit at all.
                self._error(f"[Case-{case}] PRS fitting FAILED: "
                            f"{type(_case_exc).__name__}: {_case_exc}")
                traceback.print_exc()
                _screen_n_good = ('screen_stats' in dir()
                                  and screen_stats.get('n_good', 0)) or 0
                self._append_csv_row(self.PRS_ERROR_FILE, [
                    f"Case-{case}", "holdout", "final", "", "", self.MRE_criteria, "",
                    _screen_n_good or '', '', '', 'FAIL (fit error)'])
                self._append_csv_row(self.COST_FILE, [
                    f"Case-{case}",
                    self.cost_estimate_dict[case_index]['Total_Rxns'],
                    self.cost_estimate_dict[case_index]['Active_Rxns'],
                    self.cost_estimate_dict[case_index]['Cost_Estimate_pPRS'],
                    self.cost_estimate_dict[case_index]['Cost_Estimate_FullPRS'],
                    self.no_of_sim[case_index], _screen_n_good or '', '', '', '',
                    '', '', _screen_n_good or '', 0, 0, 0, 0,
                    self.no_of_sim[case_index], 0, '0.00'])
                _case_verdicts.append((case, float('nan'), float('nan'),
                                       float('nan'), 'FAIL (fit error)'))

        os.chdir("..")

        # experiment ledger: z-space + arm metadata (y-space verdicts live in
        # the existing Prediction_errors.csv ledgers only)
        if self._exp_rows:
            import time
            exp_path = os.path.join(self.home_dir, "prs_experiment_results.csv")
            cols = ["run_id", "case", "mode", "variant", "P", "n_train", "n_aug",
                    "cv_score_z", "train_mre_z", "val_mre_z", "holdout_mre_z",
                    "train_mre_y", "holdout_mre_y", "holdout_mre_y_all",
                    "verdict_y", "aug_sims_run", "aug_sims_kept", "encoder_file"]
            run_id = time.strftime("%Y%m%d-%H%M%S")
            new_file = not os.path.isfile(exp_path)
            with open(exp_path, "a") as fh:
                if new_file:
                    fh.write(",".join(cols) + "\n")
                for row in self._exp_rows:
                    fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
            self._kv("Experiment ledger", exp_path)

        _n_pass = sum(1 for *_, v in _case_verdicts if v == "PASS")
        _n_fail = len(_case_verdicts) - _n_pass
        self._step("Per-case PRS summary")
        self._table(["Case", "Train%", "Holdout(BV)%", "Holdout(all)%", "Verdict"],
                    [[c, f"{tr:.3f}", f"{hv:.3f}", f"{ha:.3f}", v]
                     for c, tr, ha, hv, v in _case_verdicts])
        self._kv("Cases passing", f"{_n_pass}/{len(_case_verdicts)}")
        if _n_fail:
            self._warn(f"Failing case(s): {', '.join(c for c, *_ , v in _case_verdicts if v == 'FAIL')}")
        return self.ResponseSurfaces


# ═══════════════════════════════════════════════════════════════════════════
# Response coordinate transforms in R space (encoding / decoding) — 2026-08-28
# ═══════════════════════════════════════════════════════════════════════════
#
# Two per-sample scalar reparameterizations of the case response y = ln(tau_us)
# so the PRS is trained on an easier-to-learn coordinate z, with every
# parameter needed for the exact inverse persisted under
# Opt/Data/Encoding_Decoding/ (the user applies the inverse during Phase 11).
#
#   power : z = (y^P - 1) / P,  P in (0, 1]          (Holder/power-mean family;
#           P -> 0+ tends to ln y). Exact inverse: y = (1 + P z)^(1/P).
#           y must be > 0, so y is floored at y_floor = max(1e-6, min(y_fit)).
#   rank  : z = interp(y; sorted_y, z_grid) with z_grid = (rank-0.5)/N --
#           the sorted profile becomes a linear rise from ~0 to ~1 ("gradual
#           rise that is easy to learn based on their response value").
#           Decode is the inverse lookup; out-of-table values clamp.
#   holder_linear : PER-SAMPLE p-Holder scalarization (one p_i per sample).
#           Samples are sorted by response magnitude; each p_i in [p_lo, 1]
#           is solved so z_i = (y_i^p_i - 1)/p_i lands on a straight-line
#           "gradual rise" vs the sorted rank (the target line endpoints are
#           the least-squares fit through the per-sample feasible bands
#           [g(p_lo; y), y-1], clipped into the band). Where the raw profile
#           is too concave to linearize exactly, p_i clips to p_lo or 1 and
#           the residual is recorded. ALL per-sample p_i are persisted
#           (p_values_case-<N>.npy) together with the monotone (z_i, y_i)
#           table; decode is 1-D monotone interpolation on that table --
#           the exact inverse at every fit point, smooth for new points.
#
# Honesty contract (see Readme.md): the PRS is fit and hyperparameter-searched
# in z-space, but EVERY pass/fail verdict and the existing ledgers are reported
# in DECODED y-space (the patch helpers in Phase10PRS). z-space numbers are
# logged separately (prs_experiment_results.csv) because the z metric is
# not comparable to the y metric: |dz/z| = |dy/y| * P*y^P/(y^P - 1), so z-space
# relative errors shrink by ~1.2-1.9x on this data -- never a free lunch, just
# a different loss geometry.


class ResponseEncoder(object):
    """Per-case scalar response transform y = ln(tau_us) -> z, with the exact
    inverse persisted to disk.

    Parameters
    ----------
    variant : "power" | "rank" | "holder_linear"
    P : float, power exponent in (0, 1] (power variant only; P->0+ ~ ln).
    p_lo : float
        holder_linear only: lower bound of the per-sample p_i in (0, 1) --
        "almost towards zero" per the user's spec (default 0.02).
    standardize : bool
        Optional post-map (z - mu)/sigma using the FIT subset (default False).
        Off by default for rank: standardization sends z through 0 where the
        relative-error metric diverges.
    rank_fit_on : "all" | "train_val"
        Which subset the rank table is fit on (see fit()); "all" includes the
        hold-out y values in the table (mild leakage, same as any response
        standardization), "train_val" removes it at the cost of clamped
        out-of-table decoding.
    criteria_y : float
        The case's y-space MRE criterion (%) -- stored with the encoder and
        used to compute criteria_z_equiv for Stage-3-in-z-space use.
    """

    def __init__(self, variant="power", P=0.1, p_lo=0.02, standardize=False,
                 rank_fit_on="all", criteria_y=2.0):
        if variant not in ("power", "rank", "holder_linear"):
            raise ValueError(f"variant must be 'power', 'rank' or "
                             f"'holder_linear', got {variant!r}")
        if not (0.0 < P <= 1.0):
            raise ValueError(f"P must be in (0, 1], got {P}")
        if not (0.0 < p_lo < 1.0):
            raise ValueError(f"p_lo must be in (0, 1), got {p_lo}")
        self.variant = str(variant)
        self.P = float(P)
        self.p_lo = float(p_lo)
        self.standardize = bool(standardize)
        self.rank_fit_on = str(rank_fit_on)
        self.criteria_y = float(criteria_y)

        self._fitted = False
        self.y_floor = None          # power: min allowed y for y^P
        self.z_shift = 0.0           # standardize post-map
        self.z_scale = 1.0
        self.table_u = None          # rank / holder_linear: unique sorted y
        self.table_z = None          # rank / holder_linear: their mapped z
        self.table_file = None       # rank / holder_linear: saved .npy filename
        self.p_values = None         # holder_linear: per-sample p_i (n,)
        self.p_file = None           # holder_linear: saved .npy filename
        self.stats = None            # holder_linear: fit diagnostics dict
        self.z_min = self.z_max = None
        self.n_fit_samples = None
        self.y_min = self.y_max = None
        self.criteria_z_equiv = None
        self.fitted_on = None

    # ── fit / encode / decode ──────────────────────────────────────────────

    def fit(self, y, fit_idx=None):
        """Compute the transform parameters from y (or y[fit_idx]).
        No-op beyond bookkeeping for 'power' (the map has no data-dependent
        constants except the y>0 floor); 'rank' builds the sorted-y table;
        'holder_linear' solves the per-sample p_i (see _fit_holder_linear)."""
        y = np.asarray(y, dtype=float).ravel()
        if fit_idx is not None:
            y_fit = y[np.asarray(fit_idx, dtype=int)]
            self.fitted_on = "subset"
        else:
            y_fit = y
            self.fitted_on = "all"
        if len(y_fit) == 0:
            raise ValueError("ResponseEncoder.fit got an empty fit set")
        self.n_fit_samples = int(len(y_fit))
        self.y_min, self.y_max = float(y.min()), float(y.max())

        if self.variant == "power":
            self.y_floor = max(1e-6, float(np.min(y_fit)))
            # criteria_z_equiv: z-space relative-% threshold equivalent to
            # criteria_y % y-space error at the MEDIAN y. Jacobian relation
            # (dy/y) = (dz/z) * (y^P - 1)/(P y^P), P<1; (y-1)/y at P=1.
            y_med = float(np.median(y_fit))
            if self.P < 1.0:
                num = y_med ** self.P - 1.0
                denom = self.P * y_med ** self.P
            else:
                num, denom = y_med - 1.0, y_med
            if abs(num) > 1e-12 and abs(denom) > 1e-12:
                self.criteria_z_equiv = float(self.criteria_y * denom / num)
            else:
                self.criteria_z_equiv = None
            if self.standardize:
                z_fit = self._encode_raw(y_fit)
                self.z_shift = float(np.mean(z_fit))
                self.z_scale = max(float(np.std(z_fit)), 1e-12)
        elif self.variant == "rank":
            # Table on UNIQUE y values (ties must never reach np.interp's
            # duplicate-xp path, which yields NaN from 0/0 slopes). Each
            # unique value gets the AVERAGE rank of its tied block, so
            # decode(encode(y)) == y exactly at every fit value.
            sorted_y = np.sort(y_fit)
            uy = np.unique(sorted_y)
            n = float(len(sorted_y))
            zu = np.empty(len(uy))
            block_start = np.searchsorted(sorted_y, uy, side="left")
            block_end = np.searchsorted(sorted_y, uy, side="right")
            avg_pos = 0.5 * (block_start + block_end - 1)   # 0-based avg rank
            zu = (avg_pos + 0.5) / n                        # (rank-0.5)/N
            self.table_u = uy
            self.table_z = zu
            self.criteria_z_equiv = None   # rank uses RMSE in z, no % threshold
        else:  # holder_linear
            self._fit_holder_linear(y_fit)
        self._fitted = True
        return self

    def _fit_holder_linear(self, y_fit):
        """Per-sample p-Holder scalarization onto a LINEAR rank profile.

        For y = ln(tau_us) > 1 the map g(p; y) = (y^p - 1)/p is strictly
        increasing in p on [p_lo, 1] (g'(p) = (y^p (p ln y - 1) + 1)/p^2 >= 0),
        so each sample's feasible z-interval is
            [L_i, U_i] = [g(p_lo; y_i), y_i - 1]   (g(p_lo) ~ ln y, g(1) = y-1).
        The target "gradual rise" is the least-squares line through the band
        midpoints vs sorted rank, with its endpoints clipped INTO the feasible
        band (so z_min >= L_0 and z_max <= U_{n-1} always hold). Each p_i is
        then solved by bisection so z_i = g(p_i; y_i) = z_t(rank_i); where the
        raw profile is too concave to linearize exactly, z_i clips to the band
        edge (p_i = p_lo or 1) and the residual is recorded in stats.
        """
        p_lo = self.p_lo
        if float(np.min(y_fit)) <= 1.0:
            raise ValueError(
                "holder_linear requires all y > 1 (y = ln(tau_us); the map "
                "g(p;y) = (y^p-1)/p is monotone in p only for y > 1). "
                f"Got min(y) = {float(np.min(y_fit)):.4f}")

        order = np.argsort(y_fit)
        y_s = y_fit[order]
        n = float(len(y_s))
        L = (np.power(y_s, p_lo) - 1.0) / p_lo          # g(p_lo; y) ~ ln y
        U = y_s - 1.0                                    # g(1; y)
        r = np.arange(len(y_s), dtype=float)

        # least-squares line through the band midpoints vs rank
        mid = 0.5 * (L + U)
        if n > 1:
            slope, intercept = np.polyfit(r, mid, 1)
        else:
            slope, intercept = 0.0, float(mid[0])
        z_min = float(np.clip(intercept, L[0], U[0]))
        z_max = float(np.clip(intercept + slope * (n - 1.0), L[-1], U[-1]))
        if z_max - z_min <= 1e-6:                        # near-constant y
            z_min, z_max = float(L[0]), max(float(U[-1]), float(L[0]) + 1e-6)
        self.z_min, self.z_max = z_min, z_max

        z_t = z_min + (z_max - z_min) * r / max(n - 1.0, 1.0)
        z_i = np.clip(z_t, L, U)                         # per-sample projection
        p_i = np.empty(len(y_s), dtype=float)
        tol = 1e-12
        lo_clip = (z_t <= L + tol) | (z_i <= L + tol)
        hi_clip = (z_t >= U - tol) | (z_i >= U - tol)
        p_i[lo_clip] = p_lo
        p_i[hi_clip] = 1.0

        # bisection on g(p; y) = z for the unclipped samples (g monotone in p)
        mid_mask = ~(lo_clip | hi_clip)
        if mid_mask.any():
            a = np.full(mid_mask.sum(), p_lo)
            b = np.ones(mid_mask.sum())
            y_mid, z_mid = y_s[mid_mask], z_i[mid_mask]
            for _ in range(60):
                m = 0.5 * (a + b)
                gm = (np.power(y_mid, m) - 1.0) / m
                up = gm < z_mid
                a = np.where(up, m, a)
                b = np.where(up, b, m)
            p_i[mid_mask] = 0.5 * (a + b)

        # per-sample p back in the UNSORTED order; decode table on UNIQUE y
        # (ties get the tied block's mean z -- same discipline as rank, so
        # np.interp never sees duplicate xp).
        p_unsorted = np.empty(len(y_fit), dtype=float)
        p_unsorted[order] = p_i
        z_unsorted = np.empty(len(y_fit), dtype=float)
        z_unsorted[order] = z_i
        uy, inv = np.unique(y_s, return_inverse=True)
        zsum = np.zeros(len(uy))
        cnt = np.zeros(len(uy))
        np.add.at(zsum, inv, z_i)
        np.add.at(cnt, inv, 1.0)
        self.table_u = uy
        self.table_z = zsum / cnt
        self.table_file = None        # set by save()
        self.p_values = p_unsorted
        self.p_file = None            # set by save()

        resid = z_i - z_t
        self.stats = {
            "n_fit_samples": int(len(y_fit)),
            "p_lo": p_lo,
            "p_min": float(np.min(p_i)), "p_max": float(np.max(p_i)),
            "z_min": z_min, "z_max": z_max,
            "n_clipped_plo": int(lo_clip.sum()),
            "n_clipped_p1": int(hi_clip.sum()),
            "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
            "linear_r2": float(1.0 - np.sum(resid ** 2) /
                               max(np.sum((z_i - z_i.mean()) ** 2), 1e-30)),
        }
        # criteria_z_equiv at the median-y sample with its solved p
        y_med = float(np.median(y_fit))
        p_med = float(p_unsorted[int(np.argmin(np.abs(y_fit - y_med)))])
        num = y_med ** p_med - 1.0
        denom = p_med * y_med ** p_med
        if abs(num) > 1e-12 and abs(denom) > 1e-12:
            self.criteria_z_equiv = float(self.criteria_y * denom / num)
        else:
            self.criteria_z_equiv = None

    def _encode_raw(self, y):
        """The primary map without the optional standardization post-map."""
        if self.variant == "power":
            yc = np.maximum(np.asarray(y, dtype=float), self.y_floor)
            return (np.power(yc, self.P) - 1.0) / self.P
        # rank / holder_linear: lookup through the fit table; out-of-table
        # values clamp to the endpoints (np.interp default) -- decode is its
        # exact inverse.
        return np.interp(np.asarray(y, dtype=float), self.table_u, self.table_z)

    def encode(self, y):
        """(n,) raw y -> (n,) z. New samples are mapped THROUGH the fitted
        transform (never re-ranked on their own), so train/val/holdout stay
        on one comparable scale."""
        if not self._fitted:
            raise RuntimeError("ResponseEncoder must be fit() before encode()")
        z = self._encode_raw(np.asarray(y, dtype=float).ravel())
        if self.standardize:
            z = (z - self.z_shift) / self.z_scale
        return z

    def decode(self, z):
        """(n,) z -> (n,) raw y -- the exact inverse (power) or the monotone
        table inverse lookup (rank / holder_linear)."""
        if not self._fitted:
            raise RuntimeError("ResponseEncoder must be fit() before decode()")
        z = np.asarray(z, dtype=float).ravel()
        if self.standardize:
            z = z * self.z_scale + self.z_shift
        if self.variant == "power":
            base = 1.0 + self.P * z
            base = np.maximum(base, 1e-12)              # keep (·)^(1/P) real
            return np.power(base, 1.0 / self.P)
        return np.interp(z, self.table_z, self.table_u)

    # ── metrics ────────────────────────────────────────────────────────────

    def z_metric(self, y_true, z_pred):
        """Accuracy of predictions z_pred against ground truth y_true,
        measured in z-space: max |relative %| (power / holder_linear) or
        RMSE (rank)."""
        z_true = self.encode(np.asarray(y_true, dtype=float).ravel())
        z_pred = np.asarray(z_pred, dtype=float).ravel()
        if self.variant in ("power", "holder_linear"):
            denom = np.where(np.abs(z_true) > 1e-6, z_true, 1e-6)
            return float(np.max(np.abs((z_pred - z_true) / denom) * 100.0))
        return float(np.sqrt(np.mean((z_pred - z_true) ** 2)))

    def mre_equivalent_z(self, criteria_y=None):
        """z-space relative-% threshold equivalent to `criteria_y`% y-space
        error at the fit's median y (power / holder_linear). None for rank."""
        if criteria_y is None or self.variant not in ("power", "holder_linear"):
            return self.criteria_z_equiv if criteria_y is None else None
        return self.criteria_z_equiv * (criteria_y / self.criteria_y)

    # ── persistence (Opt/Data/Encoding_Decoding/) ─────────────────────────

    def save(self, enc_dir, case_index):
        """Persist every parameter needed for the inverse transform:
        encoder_case-<N>_<variant>.json (+ sorted_y_case-<N>.npy for rank;
        + p_values_case-<N>.npy and holder_table_case-<N>.npy for
        holder_linear), and refresh encoder_meta.json (the 'active' variant
        per case)."""
        os.makedirs(enc_dir, exist_ok=True)
        base = f"encoder_case-{int(case_index)}_{self.variant}"
        meta = {
            "variant": self.variant, "P": self.P, "p_lo": self.p_lo,
            "standardize": self.standardize,
            "rank_fit_on": self.rank_fit_on, "criteria_y": self.criteria_y,
            "criteria_z_equiv": self.criteria_z_equiv, "y_floor": self.y_floor,
            "z_shift": self.z_shift, "z_scale": self.z_scale,
            "n_fit_samples": self.n_fit_samples, "y_min": self.y_min,
            "y_max": self.y_max, "fitted_on": self.fitted_on,
            "table_file": None, "p_file": None,
            "z_min": self.z_min, "z_max": self.z_max,
            "stats": self.stats,
        }
        if self.variant == "rank":
            table_file = f"sorted_y_case-{int(case_index)}.npy"
            np.save(os.path.join(enc_dir, table_file),
                    np.column_stack([self.table_u, self.table_z]))
            meta["table_file"] = table_file
            self.table_file = table_file
        elif self.variant == "holder_linear":
            # distinct filenames so power + rank + holder_linear can coexist
            table_file = f"holder_table_case-{int(case_index)}.npy"
            p_file = f"p_values_case-{int(case_index)}.npy"
            np.save(os.path.join(enc_dir, table_file),
                    np.column_stack([self.table_u, self.table_z]))
            np.save(os.path.join(enc_dir, p_file),
                    np.asarray(self.p_values, dtype=float))
            meta["table_file"] = table_file
            meta["p_file"] = p_file
            self.table_file, self.p_file = table_file, p_file
        json_path = os.path.join(enc_dir, f"{base}.json")
        with open(json_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        meta_path = os.path.join(enc_dir, "encoder_meta.json")
        all_meta = {}
        if os.path.isfile(meta_path):
            try:
                all_meta = json.load(open(meta_path))
            except (ValueError, OSError):
                all_meta = {}
        all_meta[str(int(case_index))] = {"variant": self.variant,
                                          "encoder_file": os.path.basename(json_path)}
        with open(meta_path, "w") as fh:
            json.dump(all_meta, fh, indent=2)
        return json_path

    @classmethod
    def load(cls, enc_dir, case_index, variant=None):
        """Reload a saved encoder (self-contained -- needs no study data).
        variant=None reads encoder_meta.json for the case's active variant."""
        if variant is None:
            meta_path = os.path.join(enc_dir, "encoder_meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"{meta_path} missing -- no encoder saved for "
                                        f"case {case_index}")
            all_meta = json.load(open(meta_path))
            entry = all_meta.get(str(int(case_index)))
            if entry is None:
                raise KeyError(f"case {case_index} has no entry in encoder_meta.json")
            variant = entry["variant"]
        json_path = os.path.join(enc_dir, f"encoder_case-{int(case_index)}_{variant}.json")
        meta = json.load(open(json_path))
        enc = cls(variant=meta["variant"], P=meta["P"],
                  p_lo=meta.get("p_lo", 0.02),
                  standardize=meta.get("standardize", False),
                  rank_fit_on=meta.get("rank_fit_on", "all"),
                  criteria_y=meta.get("criteria_y", 2.0))
        enc.y_floor = meta.get("y_floor")
        enc.z_shift = meta.get("z_shift", 0.0)
        enc.z_scale = meta.get("z_scale", 1.0)
        enc.n_fit_samples = meta.get("n_fit_samples")
        enc.y_min, enc.y_max = meta.get("y_min"), meta.get("y_max")
        enc.criteria_z_equiv = meta.get("criteria_z_equiv")
        enc.fitted_on = meta.get("fitted_on")
        enc.z_min, enc.z_max = meta.get("z_min"), meta.get("z_max")
        enc.stats = meta.get("stats")
        enc.table_file = meta.get("table_file")
        enc.p_file = meta.get("p_file")
        if meta.get("table_file"):
            arr = np.load(os.path.join(enc_dir, meta["table_file"]))
            enc.table_u, enc.table_z = arr[:, 0], arr[:, 1]
        if meta.get("p_file"):
            enc.p_values = np.load(os.path.join(enc_dir, meta["p_file"]))
        enc._fitted = True
        return enc


def default_enc_dir(study_dir):
    """Opt/Data/Encoding_Decoding/ under a study root (live-pipeline layout)."""
    return os.path.join(study_dir, "Opt", "Data", "Encoding_Decoding")


def _resolve_enc_dir(enc_dir, study_dir):
    if enc_dir:
        return enc_dir
    if study_dir:
        return default_enc_dir(study_dir)
    raise ValueError("either enc_dir or study_dir must be given")


def load_encoder(case_index, variant=None, enc_dir=None, study_dir=None):
    """Phase-11 helper: reload a case's saved encoder from disk."""
    return ResponseEncoder.load(_resolve_enc_dir(enc_dir, study_dir),
                                int(case_index), variant=variant)


def encode_response(y, case_index, variant=None, enc_dir=None, study_dir=None):
    """Phase-11 helper: map raw y -> z with the case's saved encoder."""
    return load_encoder(case_index, variant, enc_dir, study_dir).encode(y)


def decode_response(z, case_index, variant=None, enc_dir=None, study_dir=None):
    """Phase-11 helper: map z -> raw y with the case's saved encoder.
    This is the inverse transformation the optimizer applies to a PRS
    prediction made in z-space (see Readme.md, 'Inverting during Phase 11')."""
    return load_encoder(case_index, variant, enc_dir, study_dir).decode(z)


# ── plot helper (dataviz palette: no dual axes; small multiples) ───────────
# Categorical slots (validated adjacent-order, light mode): blue/orange/aqua.
_CAT1, _CAT2, _CAT3 = "#2a78d6", "#eb6834", "#1baf7a"
_SEQ_LOW, _SEQ_HIGH = "#9ec5f4", "#184f95"     # sequential blue ramp endpoints
_DIV_LO, _DIV_MID, _DIV_HI = "#2a78d6", "#f0efec", "#e34948"  # diverging corr
_INK, _INK2, _SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def plot_transform_profile(encoder, y_raw, z, out_path, case_index):
    """2x2 small-multiples visual proof of the encoding: (a) sorted raw y
    profile ("the chaotic response"), (b) sorted z profile ("the gradual
    rise"), (c) z vs y monotone map, (d) decode(encode(y)) - y round-trip
    residual. Each panel carries its own scale -- never a dual-axis chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y_raw, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    order = np.argsort(y)
    rt = encoder.decode(encoder.encode(y)) - y

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.6))
    fig.patch.set_facecolor(_SURF)
    for ax in axs.ravel():
        ax.set_facecolor(_SURF)
        ax.tick_params(colors=_INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(_INK2)
            ax.spines[sp].set_linewidth(0.7)
        ax.grid(True, color=_INK2, alpha=0.15, linewidth=0.5)

    axs[0, 0].plot(np.arange(len(y)), y[order], color=_CAT1, linewidth=2)
    axs[0, 0].set_title(f"Case {case_index} · raw response y = ln(τ) (sorted)", fontsize=10)
    axs[0, 0].set_xlabel("sample order"); axs[0, 0].set_ylabel("y")

    axs[0, 1].plot(np.arange(len(z)), z[order], color=_CAT2, linewidth=2)
    axs[0, 1].set_title(f"Encoded z · {encoder.variant}"
                        + (f", P={encoder.P}" if encoder.variant == "power" else "")
                        + " (sorted)", fontsize=10)
    axs[0, 1].set_xlabel("sample order"); axs[0, 1].set_ylabel("z")

    axs[1, 0].scatter(y, z, s=18, facecolors="none", edgecolors=_CAT3,
                      linewidths=0.6, alpha=0.55, zorder=3)
    axs[1, 0].set_title("Monotone map y → z (one-to-one)", fontsize=10)
    axs[1, 0].set_xlabel("y"); axs[1, 0].set_ylabel("z")

    axs[1, 1].plot(np.arange(len(y)), rt, color=_CAT1, linewidth=1.2)
    axs[1, 1].axhline(0, color=_INK2, linewidth=0.7)
    axs[1, 1].set_title(f"Round-trip residual decode(encode(y)) − y "
                        f"(max |·| = {np.abs(rt).max():.2e})", fontsize=10)
    axs[1, 1].set_xlabel("sample index"); axs[1, 1].set_ylabel("residual")

    fig.suptitle(f"Case {case_index}: response coordinate transform "
                 f"({encoder.variant}{', P=' + str(encoder.P) if encoder.variant == 'power' else ''})",
                 fontsize=12, color=_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Correlation-aware resampling (candidate generation) — 2026-08-28
# ═══════════════════════════════════════════════════════════════════════════
#
# Generates NEW design-matrix candidate rows whose per-reaction perturbation
# covariance respects the CORRELATION structure of the case's band-valid
# data, optionally blended with the mechanism's stated physical correlation
# (Sigma_r = L_r L_r^T from unsrt[rxn].cholskyDeCorrelateMat -- the same
# covariance the band blocks were built from). This is the "correlation is
# taken into account" part of the request.
#
#    C_draw       = (1 - lambda) corr(Sigma_r) + lambda corr(S_emp)
#    Sigma_draw_r = diag(sqrt(diag(S_emp))) @ C_draw @ diag(sqrt(diag(S_emp)))
#    delta_r      = L_draw_r @ u,   u ~ N(0, I)   (one joint Sobol block)
#    row          = mu_r + delta_r  (mu_r = good-row mean, or a cluster mean)
#
# SHAPE (correlation) from the blend; SCALE (variance) from the band-valid
# observations only -- taking scale from Sigma_r would recreate the
# near-singularity blowup PAAR fixed. Default lambda = 1.0 (pure empirical):
# the physical correlation is near-rank-1 and band-hostile (measured, see
# CorrelationResampler.empirical_weight's docstring).
#
# Four safeguards:
#   1. The empirical covariance is WINSORIZED (median +- q95-q05) before
#      estimation -- the good rows carry heavy-tailed pre-cap rows that
#      would dominate a raw covariance.
#   2. The empirical covariance is diagonally shrinkage-regularized
#      ((1-eta) S + eta diag(S)) -- never tr(S)/m, which is scale-naive
#      across zeta_Ea (~2050) vs zeta_A (~1.6) columns.
#   3. Every candidate column is CLIPPED to +-1.2x the good rows' observed
#      (q95 - q05) spread -- clipping preserves the correlation structure,
#      unlike per-column rescaling.
#   4. Every candidate is gated by IdentifiablePRS.band_valid_mask (the exact
#      screening test) and deduped against the reference rows before it can
#      ever reach a simulation.
#
# Full step-by-step in Readme.md ("How the resampling takes place").


_SHRINK_ETA = 0.01      # empirical-covariance shrinkage fraction
_SCALE_CAP = 1.2        # per-column clip multiplier on observed q95-q05 spread
_MAX_BATCH_ROUNDS = 3   # candidate generation rounds before accepting shortfall
_CAND_OVERSAMPLE = 3    # candidates generated per target row, per round


def _build_prs_layout(unsrt_data, selected_params, prs_type):
    """Mirror of AdaptiveResampler.OutlierResampler._build_layout (same
    full_offset/comp_offset/prs_local conventions) so CorrelationResampler's
    prs_rows/full_rows feed SM.getYAML_List unchanged."""
    layout = []
    full_offset = 0
    comp_offset = 0
    for rxn in unsrt_data:
        alen = len(unsrt_data[rxn].activeParameters)
        chunk = np.asarray(selected_params, dtype=int)[full_offset: full_offset + alen]
        sel_local = [i for i, s in enumerate(chunk) if s == 1]
        if prs_type == "Full":
            n_cols, prs_local = alen, list(range(alen))
        else:
            n_cols, prs_local = len(sel_local), list(sel_local)
        layout.append({"rxn": rxn, "active_len": alen,
                       "full_offset": full_offset, "sel_local": sel_local,
                       "comp_offset": comp_offset, "n_cols": n_cols,
                       "prs_local": prs_local})
        full_offset += alen
        comp_offset += n_cols
    return layout


class CorrelationResampler(object):
    """Correlation-aware candidate generation in compressed-zeta (PRS) space.

    Parameters
    ----------
    unsrt_data : dict
        {rxn: UncertaintyExtractor} -- supplies cholskyDeCorrelateMat (L with
        L L^T = Sigma) per reaction.
    selected_params : (n_full,) int array
        Global 0/1 selection mask (same object Phase 10 gets).
    prs_type : "Full" | "Partial"
    blocks : list of IdentifiablePRS.ReactionBlock
        The case's band blocks -- used ONLY for the validity gate.
    iprs : module
        IdentifiablePRS (band_valid_mask / violation_detail / reduced_cholesky).
    empirical_weight : float in [0,1]
        lambda blend: 0 = pure physical Sigma_r correlation, 1 = pure
        empirical. DEFAULT 1.0 -- measured on the pilot (n-heptane, 1293 K):
        the physical correlation is near-rank-1 (corr(n,Ea) = -0.996) and even
        a 5% blend (lambda=0.95) collapses the band-valid yield from 33% to
        6% while a 10% blend (lambda=0.9) collapses it to 1%. The band-valid
        data lives in a shell of the physical prior with its own correlation
        ((A,n) = -0.93); that REALIZED correlation is the correct proposal
        shape. Lower lambda values remain selectable and reproduce this
        negative result (see Readme.md).
    band_tolerance : float
        Gate = band_valid_mask(X, blocks, band_tolerance).
    qmc : bool
        Draw u via scrambled-Sobol -> Normal CDF (low-discrepancy Gaussian),
        else i.i.d. standard normal.
    seed : int
    dedup_eps : float
        Candidates closer than this (Euclidean, compressed space) to any
        reference row are dropped.
    center_mode : "pool_center" | "kmeans"
        Draw centers: the good-row mean, or k-means cluster means drawn
        proportional to cluster size (localized correlation-aware draws).
    """

    def __init__(self, unsrt_data, selected_params, prs_type, blocks, iprs,
                 empirical_weight=1.0, band_tolerance=1.0, qmc=True,
                 seed=20260828, dedup_eps=1e-6, center_mode="pool_center"):
        if not (0.0 <= empirical_weight <= 1.0):
            raise ValueError(f"empirical_weight must be in [0,1], got {empirical_weight}")
        if center_mode not in ("pool_center", "kmeans"):
            raise ValueError(f"center_mode must be 'pool_center' or 'kmeans', "
                             f"got {center_mode!r}")
        self.unsrt = unsrt_data
        self.selected_params = np.asarray(selected_params, dtype=int)
        self.prs_type = prs_type
        self.blocks = blocks
        self.IPRS = iprs
        self.empirical_weight = float(empirical_weight)
        self.band_tolerance = float(band_tolerance)
        self.qmc = bool(qmc)
        self.seed = int(seed)
        self.dedup_eps = float(dedup_eps)
        self.center_mode = center_mode

        self.layout = _build_prs_layout(unsrt_data, selected_params, prs_type)
        self.n_full = self.layout[-1]["full_offset"] + self.layout[-1]["active_len"]
        self.n_prs = self.layout[-1]["comp_offset"] + self.layout[-1]["n_cols"]
        self._sigma_draw_full = None   # n_prs x n_prs block-diag (for plotting)
        self._centers_full = None      # draw centers used (for stats)

    # ── per-reaction draw covariance ───────────────────────────────────────

    def _per_reaction_covs(self, X_good):
        """For each reaction with n_cols>0: (C_draw_r, Sigma_draw_r,
        L_draw_r, spread_r).

        Correlation-aware = SHAPE from a blend of the mechanism's physical
        correlation (Sigma_r = L L^T restricted to the selected params) and
        the good rows' empirical correlation, but SCALE from the empirical
        variances only::

            C_draw    = (1-lambda) corr(Sigma_r) + lambda corr(S_emp)
            Sigma_draw = diag(sqrt(diag(S_emp))) @ C_draw @ diag(sqrt(diag(S_emp)))

        Taking the variance scale from Sigma_r directly would reproduce the
        near-singularity problem PAAR fixed (its dominant eigenvector is the
        band-restricted direction -- draws along it are guaranteed band
        violations). spread_r = observed q95-q05 per column (the clip bound).
        """
        covs = []
        for lay in self.layout:
            if lay["n_cols"] == 0:
                continue
            cols = slice(lay["comp_offset"], lay["comp_offset"] + lay["n_cols"])
            Xr = X_good[:, cols]
            m = Xr.shape[1]
            # physical covariance of the SELECTED params, from the mechanism
            L = self.unsrt[lay["rxn"]].cholskyDeCorrelateMat
            Sigma_full = self.IPRS.sigma_from_cholesky(
                L, n_active=len(self.unsrt[lay["rxn"]].activeParameters))
            Sigma_r, _ = self.IPRS.reduced_cholesky(Sigma_full, lay["sel_local"])
            std_r = np.sqrt(np.clip(np.diag(Sigma_r), 1e-30, None))
            C_r = Sigma_r / np.outer(std_r, std_r)
            # ROBUST empirical covariance: the good rows contain heavy-tailed
            # rows (the pre-cap zeta_Ea era) that pass the band only through
            # full-combination cancellation. A raw covariance is dominated by
            # them and its Gaussian draws land far outside the band (observed:
            # 0/4500 candidates band-valid). Winsorize each column to the
            # median +- (q95-q05) window (which contains ~90% of the mass by
            # construction) so the draw scale follows the BULK of the data.
            q50 = np.quantile(Xr, 0.50, axis=0)
            q05 = np.quantile(Xr, 0.05, axis=0)
            q95 = np.quantile(Xr, 0.95, axis=0)
            spread = np.maximum(q95 - q05, 1e-9)
            Xw = np.clip(Xr, q50 - spread, q50 + spread)
            Xc = Xw - Xw.mean(axis=0, keepdims=True)
            S = (Xc.T @ Xc) / max(len(Xr) - 1, 1)
            # DIAGONAL shrinkage (Ledoit-Wolf style): each column shrinks
            # toward its OWN variance. Never shrink toward tr(S)/m * I --
            # zeta_Ea columns (~2050 std) and zeta_A columns (~1.6 std) share
            # a reaction, and the trace-average target inflates the small
            # columns by ~1e4 (observed: std 1.6 -> 119) while flattening the
            # empirical correlation structure.
            eta = _SHRINK_ETA
            S_reg = (1.0 - eta) * S + eta * np.diag(np.diag(S))
            std_e = np.sqrt(np.clip(np.diag(S_reg), 1e-30, None))
            C_e = S_reg / np.outer(std_e, std_e)
            # shape blend + empirical scale
            C_draw = ((1.0 - self.empirical_weight) * C_r
                      + self.empirical_weight * C_e)
            Sigma_draw = np.diag(std_e) @ C_draw @ np.diag(std_e)
            try:
                L_draw = np.linalg.cholesky(Sigma_draw)
            except np.linalg.LinAlgError:
                eps = 1e-10 * max(float(np.trace(Sigma_draw)), 1e-12)
                L_draw = np.linalg.cholesky(Sigma_draw + eps * np.eye(m))
            covs.append((lay, C_draw, Sigma_draw, L_draw, spread))
        return covs

    def _draw_units(self, n, d, seed_part):
        """u ~ N(0, I_d): scrambled-Sobol mapped through the Normal CDF
        (low-discrepancy Gaussian) or plain standard normal."""
        import warnings
        if self.qmc:
            from scipy.stats import qmc as _qmc
            from scipy.stats import norm as _norm
            sobol = _qmc.Sobol(d, scramble=True, seed=self.seed + seed_part)
            with warnings.catch_warnings():
                # Sobol prefers power-of-2 n; our batch sizes rarely are and
                # that is fine for this use -- the balance warning is noise.
                warnings.simplefilter("ignore", UserWarning)
                pts = sobol.random(n)
            u = _norm.ppf(np.clip(pts, 1e-12, 1.0 - 1e-12))
        else:
            u = np.random.default_rng(self.seed + seed_part).standard_normal((n, d))
        return u

    # ── main entry point ───────────────────────────────────────────────────

    def generate(self, n_target, X_good):
        """Generate n_target band-valid, deduped, correlation-aware candidate
        rows in compressed space. Returns (prs_rows, full_rows, sel_mask,
        stats). Never raises -- a shortfall is reported in stats."""
        from scipy.spatial import cKDTree

        X_good = np.asarray(X_good, dtype=float)
        if X_good.ndim != 2 or X_good.shape[1] != self.n_prs:
            raise ValueError(f"X_good shape {X_good.shape} != (n, {self.n_prs})")
        n_target = int(n_target)
        stats = {"n_target": n_target, "n_generated": 0, "n_band_valid": 0,
                 "n_dedup": 0, "n_accepted": 0, "reject_rate": None,
                 "worst_violation_ratio": None, "median_violation_ratio": None,
                 "corr_dist_sigma_draw": None, "corr_dist_original": None,
                 "coverage_ratio_mean": None, "rounds": 0, "center_mode": self.center_mode}

        # draw centers: pool mean, or k-means cluster means (sized-proportional)
        if self.center_mode == "pool_center":
            centers = [X_good.mean(axis=0)]
            weights = [1.0]
        else:
            from sklearn.cluster import KMeans
            k = max(1, min(8, len(X_good) // 50))
            km = KMeans(n_clusters=k, n_init=2, random_state=self.seed).fit(X_good)
            counts = np.bincount(km.labels_, minlength=k)
            centers = [X_good[km.labels_ == c].mean(axis=0)
                       for c in range(k) if counts[c] > 0]
            weights = [counts[c] / len(X_good)
                       for c in range(k) if counts[c] > 0]

        ref_tree = cKDTree(X_good)
        accepted = []
        batch_n = max(int(n_target * _CAND_OVERSAMPLE / max(len(centers), 1)), 4)

        for rnd in range(_MAX_BATCH_ROUNDS):
            stats["rounds"] += 1
            # per-center budgets (weights sum to 1)
            budgets = [max(1, int(round(batch_n * w))) for w in weights]
            # ONE global low-discrepancy draw for the whole row, then slice
            # per-reaction columns from it. Consecutive-seed per-reaction
            # Sobol blocks stay coordinate-wise correlated (~0.4-0.7) across
            # blocks and would fabricate cross-reaction correlation the draws
            # must NOT have; a single joint draw is the same pattern the
            # original DM builders use (one joint shadow block).
            U = self._draw_units(sum(budgets), self.n_prs, 997 * (rnd + 1))
            cand = []
            row0 = 0
            for ci, (mu, n_c) in enumerate(zip(centers, budgets)):
                rows = np.zeros((n_c, self.n_prs))
                for lay, C_draw, Sigma_draw, L_draw, spread in \
                        self._per_reaction_covs(X_good):
                    cols = slice(lay["comp_offset"], lay["comp_offset"] + lay["n_cols"])
                    u = U[row0:row0 + n_c, cols]
                    delta = u @ L_draw.T
                    # per-column clip to 1.2x observed spread (keeps the
                    # correlation structure; guards Sigma_r's null directions)
                    delta = np.clip(delta, -_SCALE_CAP * spread, _SCALE_CAP * spread)
                    rows[:, cols] = mu[cols] + delta
                cand.append(rows)
                row0 += n_c
            cand = np.vstack(cand)
            stats["n_generated"] += len(cand)

            # gate: band-validity (the exact screening test), then dedup
            valid = self.IPRS.band_valid_mask(cand, self.blocks, self.band_tolerance)
            ratio, _ = self.IPRS.violation_detail(cand, self.blocks)
            if len(cand):
                stats["worst_violation_ratio"] = float(np.max(ratio))
                stats["median_violation_ratio"] = float(np.median(ratio))
            cand_valid = cand[valid]
            if len(cand_valid) > 1:
                d, _ = ref_tree.query(cand_valid, k=1)
                cand_valid = cand_valid[d > self.dedup_eps]
            stats["n_band_valid"] += int(valid.sum())
            stats["n_dedup"] += int(valid.sum()) - len(cand_valid)
            accepted.append(cand_valid)
            if sum(len(a) for a in accepted) >= n_target:
                break
        prs_rows = np.vstack(accepted) if accepted else np.zeros((0, self.n_prs))
        prs_rows = prs_rows[:n_target]
        stats["n_accepted"] = len(prs_rows)
        stats["reject_rate"] = (1.0 - len(prs_rows) / max(stats["n_generated"], 1))

        # full-space rows + selection mask (contract of _simulate_new_samples)
        full_rows = np.zeros((len(prs_rows), self.n_full))
        for lay in self.layout:
            for k, full_local in enumerate(lay["prs_local"]):
                full_rows[:, lay["full_offset"] + full_local] = \
                    prs_rows[:, lay["comp_offset"] + k]
        sel_mask = self.selected_params.astype(float)

        # offline-verifiable correlation metrics
        self._sigma_draw_full = self._assemble_sigma_draw(X_good)
        if len(prs_rows) >= 2:
            C_good = np.corrcoef(X_good.T)
            C_cand = np.corrcoef(prs_rows.T)
            d = self.n_prs
            stats["corr_dist_sigma_draw"] = float(
                np.linalg.norm(C_cand - self._sigma_draw_full, ord="fro") / d)
            stats["corr_dist_original"] = float(
                np.linalg.norm(C_cand - C_good, ord="fro") / d)
            stats["coverage_ratio_mean"] = self._coverage_ratio(X_good, prs_rows)
        self._centers_full = np.asarray(centers)
        return prs_rows, full_rows, sel_mask, stats

    def _assemble_sigma_draw(self, X_good):
        """n_prs x n_prs BLOCK-DIAGONAL correlation matrix of the per-reaction
        Sigma_draw blends (correlation, not covariance -- comparable with the
        empirical corr matrices in the comparison plots)."""
        C = np.eye(self.n_prs)
        for lay, C_draw, Sigma_draw, L_draw, spread in self._per_reaction_covs(X_good):
            cols = slice(lay["comp_offset"], lay["comp_offset"] + lay["n_cols"])
            C[cols, cols] = C_draw
        return C

    def _coverage_ratio(self, X_good, prs_rows):
        """Mean over columns of (candidate q95-q05) / (good q95-q05). ~1.0
        means the candidates cover the observed range without blowout."""
        g = np.quantile(X_good, 0.95, axis=0) - np.quantile(X_good, 0.05, axis=0)
        c = np.quantile(prs_rows, 0.95, axis=0) - np.quantile(prs_rows, 0.05, axis=0)
        return float(np.mean(c / np.maximum(g, 1e-9)))

    # ── offline verification plots ─────────────────────────────────────────

    def plot_correlation_comparison(self, X_good, prs_cand, out_dir, case_index):
        """Correlation-structure verification (offline dry-run evidence):
        two figures -- a 2x2 correlation-matrix panel (good / candidates /
        Sigma_draw target / |candidates - target|) and a per-column coverage
        bar chart -- plus the stats JSON. Returns the stats dict."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        X_good = np.asarray(X_good, dtype=float)
        prs_cand = np.asarray(prs_cand, dtype=float)
        C_good = np.corrcoef(X_good.T) if len(X_good) > 1 else np.eye(self.n_prs)
        C_cand = np.corrcoef(prs_cand.T) if len(prs_cand) > 1 else np.eye(self.n_prs)
        C_tgt = self._sigma_draw_full if self._sigma_draw_full is not None \
            else self._assemble_sigma_draw(X_good)
        d = self.n_prs
        stats = {
            "case": int(case_index),
            "corr_dist_candidates_vs_sigma_draw": float(np.linalg.norm(C_cand - C_tgt, ord="fro") / d),
            "corr_dist_candidates_vs_good": float(np.linalg.norm(C_cand - C_good, ord="fro") / d),
            "coverage_ratio_mean": self._coverage_ratio(X_good, prs_cand)
            if len(prs_cand) > 1 else None,
        }
        with open(os.path.join(out_dir, f"corr_aug_stats_case-{case_index}.json"), "w") as fh:
            json.dump(stats, fh, indent=2)

        mats = [("good rows", C_good), ("candidates", C_cand),
                ("Σ_draw target", C_tgt), ("|cand − target|", np.abs(C_cand - C_tgt))]
        fig, axs = plt.subplots(2, 2, figsize=(12.0, 10.0))
        fig.patch.set_facecolor(_SURF)
        for (title, M), ax in zip(mats, axs.ravel()):
            vmin, vmax = (-1.0, 1.0) if "−" not in title else (0.0, max(0.5, float(np.max(M))))
            cmap = self._diverging_cmap() if "−" not in title else self._sequential_cmap()
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=10, color=_INK)
            ax.tick_params(colors=_INK2, labelsize=6)
            ax.set_facecolor(_SURF)
            for sp in ax.spines.values():
                sp.set_color(_INK2)
                sp.set_linewidth(0.7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Case {case_index}: correlation structure — candidates vs "
                     f"Σ_draw (dist {stats['corr_dist_candidates_vs_sigma_draw']:.3f}) "
                     f"vs good rows (dist {stats['corr_dist_candidates_vs_good']:.3f})",
                     fontsize=11, color=_INK)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(out_dir, f"corr_comparison_case-{case_index}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

        # per-column coverage: candidate spread / good spread, reference at 1
        g = np.quantile(X_good, 0.95, axis=0) - np.quantile(X_good, 0.05, axis=0)
        c = np.quantile(prs_cand, 0.95, axis=0) - np.quantile(prs_cand, 0.05, axis=0)
        ratio = c / np.maximum(g, 1e-9)
        fig2, ax = plt.subplots(figsize=(11.0, 3.6))
        fig2.patch.set_facecolor(_SURF)
        ax.set_facecolor(_SURF)
        ax.bar(np.arange(self.n_prs), ratio, color=_CAT1, width=0.8)
        ax.axhline(1.0, color=_CAT2, linewidth=1.5)
        ax.set_xlabel("compressed parameter column")
        ax.set_ylabel("candidate q95−q05 / good q95−q05")
        ax.set_title(f"Case {case_index}: per-column coverage ratio "
                     f"(mean {stats['coverage_ratio_mean']:.2f})", fontsize=10, color=_INK)
        ax.tick_params(colors=_INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(_INK2)
            ax.spines[sp].set_linewidth(0.7)
        ax.grid(axis="y", color=_INK2, alpha=0.15, linewidth=0.5)
        out2 = os.path.join(out_dir, f"coverage_case-{case_index}.pdf")
        fig2.savefig(out2, bbox_inches="tight", dpi=150)
        fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig2)
        return stats

    @staticmethod
    def _diverging_cmap():
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            "corr_div", [_DIV_LO, _DIV_MID, _DIV_HI], N=256)

    @staticmethod
    def _sequential_cmap():
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            "corr_seq", [_SURF, _SEQ_LOW, _SEQ_HIGH], N=256)


def _load_lst(path):
    """sim_data_case-N.lst: tab-separated `row_idx \\t ln_tau_us`."""
    return np.asarray([float(l.rstrip("\n").split("\t")[1]) for l in open(path)])


def _load_dm(path):
    """pDesignMatrix.csv: comma-separated compressed-zeta rows (trailing
    empty columns tolerated)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip("\n").strip(",").strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split(",") if v != ""])
    return np.asarray(rows)


def load_case_data(study_dir, case):
    """Shared loader of Phase10CaseDiagnostics / Phase10ExperimentRunner:
    (unsrt, selected_params, X, y, active_params) for one case of a study
    directory (unsrt.pkl + DM_FOR_PARTIAL_PRS/<case>/selection_details.csv +
    pDesignMatrix.csv + Opt/Data/Simulations/sim_data_case-<case>.lst)."""
    import pickle as _pickle
    import csv as _csv
    with open(os.path.join(study_dir, "unsrt.pkl"), "rb") as fh:
        unsrt = _pickle.load(fh)
    sel_rows = list(_csv.DictReader(open(
        os.path.join(study_dir, "DM_FOR_PARTIAL_PRS", str(case),
                     "selection_details.csv"))))
    active_params = [r["param"] for r in sel_rows]
    selected_params = np.asarray(
        [1 if r["final_in_PRS"] == "True" else 0 for r in sel_rows], dtype=int)
    X = _load_dm(os.path.join(study_dir, "DM_FOR_PARTIAL_PRS", str(case),
                              "pDesignMatrix.csv"))
    y = _load_lst(os.path.join(study_dir, "Opt", "Data", "Simulations",
                               f"sim_data_case-{case}.lst"))
    if len(X) != len(y):
        raise ValueError(f"DM rows ({len(X)}) != response rows ({len(y)})")
    return unsrt, selected_params, X, y, active_params


class Phase10CaseDiagnostics(object):
    """Offline single-case reproducer of Phase 10's fit -- loads one case's
    data from a study directory (unsrt.pkl + DM_FOR_PARTIAL_PRS/<case>/ +
    Opt/Data/Simulations/sim_data_case-<case>.lst) and runs the SAME screening
    -> split -> CV-search -> fit -> hold-out chain, with the full STEP trace.

    No Chemkin, no full pipeline: this is the test bench for "does a changed
    grid/tolerance/split reach <2% on this case's data". Validated to
    reproduce the pilot_5case_case0 (15.423/14.817%) and
    TEST_CORRELATIONS_CORRECTED case-0 (0.894/0.644%) published numbers
    (the TCC numbers require its study's own grids: tol [1.0, inf],
    ridge [0, 1e-7] -- pass them via tol_grid/ridge_grid).

    BUG FIX (2026-08-28): the final fit used to call
    create_response_surface() with the launch cwd, so a stale
    cwd/Data/ResponseSurface/responsecoef_case-N.csv silently short-circuited
    the fit (and the first run littered the source tree). The final fit now
    chdirs into cache_dir and removes the stale cache file first, mirroring
    IdentifiablePRS._fit_once and the live pipeline's STEP 6.

    Parameters
    ----------
    study_dir : str
        Study root containing unsrt.pkl, DM_FOR_PARTIAL_PRS/, targets.csv and
        Opt/Data/Simulations/.
    case : int
        Case index.
    temperature : float
        Nominal target temperature (K) for this case -- read from targets.csv.
    criteria : float
        MRE criterion (%) used for scoring (does not affect the fit itself).
    tol_grid / k_grid / order_grid / ridge_grid : lists
        Search grid overrides (defaults mirror run_script's prs_fit_cfg).
    fprior_mode : "full" | "reduced"
    band_tolerance_eval : float or None
    cache_dir : str
        Scratch dir for the transient coefficient-cache files the real
        ResponseSurface writes (default ./prs_diag_cache).
    verbose : bool
    """

    def __init__(self, study_dir, case, temperature, criteria=2.0,
                 tol_grid=(1.0,), k_grid=(8, 10, 12, 14, 16, 20),
                 order_grid=(2, 3), ridge_grid=(0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5),
                 fprior_mode="full", band_tolerance_eval=1.0,
                 cache_dir="./prs_diag_cache", verbose=True):
        import sys as _sys

        # resolve the MUQ-SAC modules the same way run_script.py does
        _src = os.path.dirname(os.path.abspath(__file__))
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        import IdentifiablePRS as _IPRS
        from ResponseSurface import ResponseSurface as _RS

        self.IPRS = _IPRS
        self.RS = _RS
        self.study_dir = study_dir
        self.case = case
        self.Tc = temperature
        self.criteria = criteria
        self.tol_grid = list(tol_grid)
        self.k_grid = list(k_grid)
        self.order_grid = list(order_grid)
        self.ridge_grid = list(ridge_grid)
        self.fprior_mode = fprior_mode
        self.band_tolerance_eval = band_tolerance_eval
        self.cache_dir = cache_dir
        self.verbose = verbose
        os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)

        (self.unsrt, self.selected_params, self.X, self.y,
         self.active_params) = load_case_data(study_dir, case)

    def _build_blocks(self):
        """Reconstruct the layout + band blocks exactly as Phase 10's STEP 2."""
        layout = []
        full_offset = comp_offset = 0
        for rxn in self.unsrt:
            alen = len(self.unsrt[rxn].activeParameters)
            chunk = self.selected_params[full_offset: full_offset + alen]
            sel_local = [i for i, s in enumerate(chunk) if s == 1]
            layout.append({"rxn": rxn, "comp_offset": comp_offset,
                           "n_cols": len(sel_local), "prs_local": list(sel_local)})
            full_offset += alen
            comp_offset += len(sel_local)
        triples = [(l["rxn"], slice(l["comp_offset"], l["comp_offset"] + l["n_cols"]),
                    l["prs_local"]) for l in layout if l["n_cols"] > 0]

        def _sigma_of(rxn):
            L = self.unsrt[rxn].cholskyDeCorrelateMat
            return self.IPRS.sigma_from_cholesky(
                L, n_active=len(self.unsrt[rxn].activeParameters))

        T_local = np.array([self.Tc * 0.75, self.Tc, self.Tc * 1.25])
        return self.IPRS.build_layout_blocks(triples, _sigma_of, T_local,
                                             svd_tol=1e-6, fprior_mode=self.fprior_mode)

    # ── step methods (reused by Phase10ExperimentRunner) ───────────────────

    def _screen(self, blocks):
        """STEP 3: band-validity screen. Returns (ratio, keep_mask)."""
        ratio, worst = self.IPRS.violation_detail(self.X, blocks)
        keep = ratio <= (self.band_tolerance_eval
                         if self.band_tolerance_eval is not None else float("inf"))
        return ratio, keep

    def _split(self):
        """STEP 4: the pipeline's deterministic 70/15/15 split."""
        return self.IPRS.three_way_split(len(self.y), seed_outer=104,
                                         seed_inner=7, train_frac=0.70,
                                         val_frac=0.15)

    def _search(self, blocks, xTrain, yTrain, xVal, yVal):
        """STEP 5: worst-of-10-fold CV hyperparameter search over the
        configured grids, on the pooled train+val rows."""
        return self.IPRS.fit_prs_with_cv_search(
            self.RS, np.vstack([xTrain, xVal]), np.concatenate([yTrain, yVal]),
            blocks, self.X.shape[1], str(self.case), self.case, "Partial",
            self.selected_params, self.criteria, cache_dir=self.cache_dir,
            tol_grid=self.tol_grid, k_grid=self.k_grid, order_grid=self.order_grid,
            ridge_grid=self.ridge_grid, max_terms_frac=0.4, cv_folds=10,
            cv_seed=13, band_tolerance_eval=self.band_tolerance_eval,
            trace_cb=(None if not self.verbose else
                      lambda tol, k, o, r, fs, w, ki: print(
                          f"      [search] tol={tol} k={k:2d} order={o} ridge={r:.0e} "
                          f"worst-fold={w:6.2f}%  kept={ki}")))

    def _final_fit(self, blocks, search, xTrain, yTrain):
        """STEP 6: screen train by the winning tolerance, fit the feature map,
        fit the ResponseSurface. Returns (rs, fmap, xTrainK, yTrainK,
        keep_mask) -- keep_mask maps back to the caller's xTrain/yTrain rows
        (needed by the decode-patch for the raw-y training references)."""
        keep_mask, _ = self.IPRS.screen_rows(xTrain, blocks, search["tolerance"])
        xTrainK, yTrainK = xTrain[keep_mask], yTrain[keep_mask]
        fmap = self.IPRS.fit_feature_map(xTrainK, yTrainK, blocks,
                                         self.X.shape[1], k=search["k"])
        rs = self.RS(xTrainK, yTrainK, str(self.case), self.case,
                     responseOrder=search["order"], prs_type="Partial",
                     selected_params=self.selected_params, criteria=self.criteria,
                     feature_map=fmap, ridge_alpha=search["ridge_alpha"])
        # BUG FIX: create_response_surface() reads/writes cwd-relative
        # Data/ResponseSurface/responsecoef_case-N.csv -- run it inside
        # cache_dir with any stale cache file removed first (the live
        # pipeline's STEP 6 does exactly this; the old code left a stale
        # cache able to silently short-circuit the final fit).
        coef_path = os.path.join(self.cache_dir, "Data", "ResponseSurface",
                                 f"responsecoef_case-{self.case}.csv")
        if os.path.isfile(coef_path):
            os.remove(coef_path)
        cwd0 = os.getcwd()
        os.chdir(self.cache_dir)
        try:
            rs.create_response_surface()
        finally:
            os.chdir(cwd0)
        return rs, fmap, xTrainK, yTrainK, keep_mask

    def run(self):
        """Run the full fit chain on the loaded case and print the STEP trace.
        Returns the final ResponseSurface."""
        def t(step, total, title, **kv):
            if not self.verbose:
                return
            print(f"\n  ┌─ [case-{self.case}] STEP {step}/{total} · {title}")
            for k, v in kv.items():
                print(f"  │   {k:<26s} = {v}")
            print("  └" + "─" * 58)

        t(1, 6, "LOAD data", n_rows=len(self.X), n_cols=self.X.shape[1],
          n_selected=int(self.selected_params.sum()),
          T_nominal=self.Tc, y_range=f"[{self.y.min():.3f}, {self.y.max():.3f}]",
          y_std=f"{self.y.std():.4f}")

        blocks = self._build_blocks()
        t(2, 6, "BUILD band blocks", n_blocks=len(blocks),
          fprior_mode=self.fprior_mode)

        ratio, keep = self._screen(blocks)
        t(3, 6, "SCREEN samples", n_in=len(self.X), n_good=int(keep.sum()),
          n_band_invalid=int((~keep).sum()),
          worst_violation_ratio=f"{ratio.max():.3f}",
          median_violation_ratio=f"{np.median(ratio):.3f}")

        itr, ival, iho = self._split()
        t(4, 6, "SPLIT train/val/holdout", n_train=len(itr), n_val=len(ival),
          n_holdout=len(iho))

        xTrain, yTrain = self.X[itr], self.y[itr]
        xVal,   yVal   = self.X[ival], self.y[ival]
        xHold,  yHold  = self.X[iho], self.y[iho]
        search = self._search(blocks, xTrain, yTrain, xVal, yVal)
        t(5, 6, "SEARCH hyperparameters (10-fold CV)",
          tolerance=search["tolerance"], k=search["k"], order=search["order"],
          ridge_alpha=f"{search['ridge_alpha']:.0e}",
          worst_fold_score=f"{search['cv_score']:.3f}%",
          n_kept=search["n_kept"])

        rs, fmap, xTrainK, yTrainK, _keep_mask = self._final_fit(
            blocks, search, xTrain, yTrain)
        mtr, mva, _ = rs.test(xVal, yVal)
        mtf, mhf, _ = rs.test(xHold, yHold)
        rel = np.asarray(rs.error_testing_relative, dtype=float)
        hold_valid = self.IPRS.band_valid_mask(
            xHold, blocks,
            self.band_tolerance_eval if self.band_tolerance_eval is not None else 1.0)
        t(6, 6, "FIT + one-time hold-out verdict",
          train=f"{mtf:.3f}%", val=f"{mva:.3f}%", holdout_all=f"{mhf:.3f}%",
          holdout_bandvalid=(f"{np.abs(rel[hold_valid]).max():.3f}%"
                             if hold_valid.any() else "n/a"),
          feature_map_k=fmap.k, n_train_rows=len(xTrainK))
        return rs


class Phase10ExperimentRunner(Phase10CaseDiagnostics):
    """Offline (no Chemkin) multi-arm experiment on an existing study.

    Arms:
      original             : the diagnostics chain (repro gate) + parity plot
      transform            : ResponseEncoder candidates (power P-grid and/or
                             rank) -- per-candidate CV search + fit on the
                             encoded response, P selected by DECODED y-space
                             max(train, val); the untouched hold-out is scored
                             exactly once per candidate for the report.
      resampling           : CorrelationResampler DRY-RUN -- candidates are
                             generated, band-gated, deduped, and saved with
                             correlation-comparison plots + stats. NO
                             simulation, and deliberately NO fake parity plots.
      resampling_transform : dry-run candidates + the transform chain.

    Writes ONLY under out_dir/<study_tag>/ (arms never touch the study dir):
      Plots/  Plots_re-sampling/  Plots_re-sampling_transform/  Plots_transfrom/
      Encoding_Decoding/  prs_experiment_results.csv  _cache_<arm>/ (coefficient
      cache, per-arm isolated -- the cache key is case-only, so arms for the
      same case must never share a cache dir).
    """

    def __init__(self, study_dir, case, temperature, mode="transform",
                 criteria=2.0, tol_grid=None, k_grid=None, order_grid=None,
                 ridge_grid=None, p_grid=None, rank_variant=False,
                 holder_linear=False, p_lo=0.02, n_new=2000,
                 empirical_weight=1.0, resample_band_tolerance=1.0,
                 resample_qmc=True, resample_seed=20260828,
                 resample_center_mode="pool_center", out_dir="./prs_experiment_out",
                 study_tag=None, verbose=True):
        if mode not in ARM_FOLDER:
            raise ValueError(f"mode must be one of {sorted(ARM_FOLDER)}, got {mode!r}")
        self.mode = mode
        self.out_dir = out_dir
        self.study_tag = study_tag or os.path.basename(os.path.abspath(study_dir))
        self.p_grid = list(p_grid) if p_grid is not None else \
            list(EXPERIMENT_DEFAULTS["transform"]["P_grid"])
        self.rank_variant = bool(rank_variant)
        self.holder_linear = bool(holder_linear)
        self.p_lo = float(p_lo)
        self.n_new = int(n_new)
        self.rc = {"empirical_weight": float(empirical_weight),
                   "band_tolerance": float(resample_band_tolerance),
                   "qmc": bool(resample_qmc), "seed": int(resample_seed),
                   "center_mode": str(resample_center_mode)}
        self.arm_root = os.path.join(out_dir, self.study_tag)
        os.makedirs(self.arm_root, exist_ok=True)
        super().__init__(study_dir, case, temperature, criteria=criteria,
                         tol_grid=tol_grid or (1.0,),
                         k_grid=k_grid or (8, 10, 12, 14, 16, 20),
                         order_grid=order_grid or (2, 3),
                         ridge_grid=ridge_grid or (0.0, 1e-7, 1e-5, 1e-3, 1e-2,
                                                   0.1, 0.3, 0.5),
                         fprior_mode="full", band_tolerance_eval=1.0,
                         cache_dir=os.path.join(self.arm_root, f"_cache_{mode}"),
                         verbose=verbose)

    def _arm_plots(self, arm_name):
        d = os.path.join(self.arm_root, arm_name)
        os.makedirs(d, exist_ok=True)
        return d

    def _write_results_row(self, row):
        exp_path = os.path.join(self.arm_root, "prs_experiment_results.csv")
        cols = ["run_id", "case", "mode", "variant", "P", "n_train", "n_aug",
                "cv_score_z", "train_mre_z", "val_mre_z", "holdout_mre_z",
                "train_mre_y", "holdout_mre_y", "holdout_mre_y_all",
                "verdict_y", "aug_sims_run", "aug_sims_kept", "encoder_file"]
        new_file = not os.path.isfile(exp_path)
        with open(exp_path, "a") as fh:
            if new_file:
                fh.write(",".join(cols) + "\n")
            fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
        return exp_path

    # ── arms ───────────────────────────────────────────────────────────────

    def run(self):
        if self.mode == "original":
            return self.run_original()
        if self.mode == "transform":
            return self.run_transform(self.p_grid, self.rank_variant)
        if self.mode == "resampling":
            return self.run_resampling()
        if self.mode == "resampling_transform":
            return self.run_resampling_transform()
        raise ValueError(f"unknown mode {self.mode!r}")

    def run_original(self):
        """The diagnostics chain + an arm-folder parity plot + ledger row."""
        rs = super().run()
        blocks = self._build_blocks()
        itr, ival, iho = self._split()
        xHold = self.X[iho]
        ratio_h, _ = self.IPRS.violation_detail(xHold, blocks)
        plots = self._arm_plots(ARM_FOLDER["original"])
        rs.plot_parity_scaled(ratio=ratio_h,
                              y_global=(float(self.y.min()), float(self.y.max())),
                              location=plots, tag="")
        hold_valid = self.IPRS.band_valid_mask(xHold, blocks, 1.0)
        rel = np.asarray(rs.error_testing_relative, dtype=float)
        hold_bv = float(np.max(np.abs(rel[hold_valid]))) if hold_valid.any() else rs.ytestMaxError
        verdict = "PASS" if max(rs.MaxError, hold_bv) <= self.criteria else "FAIL"
        self._write_results_row({
            "case": self.case, "mode": "original", "variant": "", "P": "",
            "n_train": len(self.y), "n_aug": 0,
            "cv_score_z": "", "train_mre_z": "", "val_mre_z": "", "holdout_mre_z": "",
            "train_mre_y": rs.MaxError, "holdout_mre_y": hold_bv,
            "holdout_mre_y_all": rs.ytestMaxError, "verdict_y": verdict,
            "aug_sims_run": 0, "aug_sims_kept": 0, "encoder_file": ""})
        return rs

    def run_transform(self, p_grid, rank_variant):
        """P-grid (power) + optional rank + optional holder_linear candidates.
        Per candidate: CV search and fit in z-space, scores decoded to
        y-space. Winner = min decoded max(train, val) -- VALIDATION only;
        hold-out is scored once per candidate purely for the report (never
        used for selection)."""
        blocks = self._build_blocks()
        _, keep = self._screen(blocks)
        itr, ival, iho = self._split()
        xTrain, yTrainR = self.X[itr], self.y[itr]
        xVal, yValR = self.X[ival], self.y[ival]
        xHold, yHoldR = self.X[iho], self.y[iho]
        ratio_h, _ = self.IPRS.violation_detail(xHold, blocks)

        candidates = []
        if rank_variant:
            candidates.append(ResponseEncoder(variant="rank", criteria_y=self.criteria))
        for P in p_grid:
            candidates.append(ResponseEncoder(variant="power", P=P,
                                              criteria_y=self.criteria))
        if self.holder_linear:
            candidates.append(ResponseEncoder(variant="holder_linear",
                                              p_lo=self.p_lo,
                                              criteria_y=self.criteria))
        rows = []
        for enc in candidates:
            fit_idx = (np.concatenate([itr, ival])
                       if enc.rank_fit_on == "train_val" else None)
            enc.fit(self.y, fit_idx=fit_idx)
            z = enc.encode(self.y)
            search = self._search(blocks, xTrain, z[itr], xVal, z[ival])
            rs, fmap, xTrainK, yTrainK, keep_mask = self._final_fit(
                blocks, search, xTrain, z[itr])
            if enc.variant == "rank":
                # rank z lives in (0,1): RELATIVE error diverges near z->0,
                # so the rank z-metric is RMSE (ResponseEncoder.z_metric).
                rs.test(xVal, z[ival])
                z_val = float(enc.z_metric(yValR, np.asarray(rs.y_Test_Predict)))
                z_train = float(enc.z_metric(yTrainR[keep_mask],
                                             np.asarray(rs.resFramWrk)))
            else:
                _, z_val, _ = rs.test(xVal, z[ival])
                z_train = float(rs.MaxError)
            _decode_patch_train(rs, yTrainR[keep_mask], enc)
            mtr_y, mva_y, _ = _score_decoded(rs, xVal, yValR, enc)
            if enc.variant == "rank":
                rs.test(xHold, z[iho])
                z_hold = float(enc.z_metric(yHoldR, np.asarray(rs.y_Test_Predict)))
            else:
                rs.test(xHold, z[iho])
                z_hold = float(rs.ytestMaxError)
            mtf_y, mhf_y, _ = _score_decoded(rs, xHold, yHoldR, enc)
            hold_valid = self.IPRS.band_valid_mask(xHold, blocks, 1.0)
            rel = np.asarray(rs.error_testing_relative, dtype=float)
            hold_bv = (float(np.max(np.abs(rel[hold_valid])))
                       if hold_valid.any() else mhf_y)
            verdict = "PASS" if max(mtf_y, hold_bv) <= self.criteria else "FAIL"
            if enc.variant == "power":
                key = f"power-P{enc.P:g}"
            elif enc.variant == "rank":
                key = "rank"
            else:
                key = f"holder_linear-p_lo{enc.p_lo:g}"
            enc.save(os.path.join(self.arm_root, "Encoding_Decoding"), self.case)
            row = {"case": self.case, "mode": self.mode, "variant": enc.variant,
                   "P": (enc.p_lo if enc.variant == "holder_linear"
                         else enc.P if enc.variant == "power" else ""),
                   "n_train": len(xTrainK), "n_aug": 0,
                   "cv_score_z": (search.get("cv_score")
                                  if search.get("cv_score") is not None
                                  else max(search["train_pct"], search["val_pct"])),
                   "train_mre_z": z_train, "val_mre_z": z_val,
                   "holdout_mre_z": z_hold,
                   "train_mre_y": mtf_y, "val_mre_y": mva_y,
                   "holdout_mre_y": hold_bv, "holdout_mre_y_all": mhf_y,
                   "verdict_y": verdict, "aug_sims_run": 0, "aug_sims_kept": 0,
                   "encoder_file": f"encoder_case-{self.case}_{enc.variant}.json",
                   "_key": key, "_rs": rs, "_enc": enc}
            rows.append(row)
            print(f"[{key}] CV(z)={row['cv_score_z']:.3f}%  train={mtf_y:.3f}% "
                  f"val={mva_y:.3f}%  holdout(y,band-valid)={hold_bv:.3f}%  "
                  f"holdout(z)={z_hold:.3f}%  {verdict}")

        winner = min(rows, key=lambda r: max(r["train_mre_y"], r["val_mre_y"]))
        print(f"winner: {winner['_key']} (decoded val criterion)")

        arm = ARM_FOLDER[self.mode]
        plots = self._arm_plots(arm)
        winner["_rs"].plot_parity_scaled(
            ratio=ratio_h, y_global=(float(self.y.min()), float(self.y.max())),
            location=plots, tag=f"_{winner['_key']}")
        plot_transform_profile(
            winner["_enc"], self.y, winner["_enc"].encode(self.y),
            os.path.join(plots, f"transform_profile_case-{self.case}_{winner['_key']}.pdf"),
            self.case)
        self._plot_p_sweep(rows, plots)
        for r in rows:
            self._write_results_row({k: v for k, v in r.items()
                                     if not k.startswith("_")})
        return rows, winner

    def run_resampling(self):
        """Correlation-aware candidate generation (dry-run): mini-DM + stats +
        correlation-comparison plots. Zero simulations, zero parity plots."""
        blocks = self._build_blocks()
        _, keep = self._screen(blocks)
        X_good = self.X[keep]
        cr = CorrelationResampler(
            self.unsrt, self.selected_params, "Partial", blocks, self.IPRS,
            empirical_weight=self.rc["empirical_weight"],
            band_tolerance=self.rc["band_tolerance"], qmc=self.rc["qmc"],
            seed=self.rc["seed"], dedup_eps=1e-6,
            center_mode=self.rc["center_mode"])
        prs_rows, full_rows, sel_mask, stats = cr.generate(self.n_new, X_good)
        plots = self._arm_plots(ARM_FOLDER[self.mode])
        np.savetxt(os.path.join(plots, f"CorrelationAug_DM_case-{self.case}.csv"),
                   prs_rows, delimiter=",")
        with open(os.path.join(plots, f"CorrelationAug_stats_case-{self.case}.json"),
                  "w") as fh:
            json.dump(stats, fh, indent=2)
        cr.plot_correlation_comparison(X_good, prs_rows, plots, self.case)
        self._write_results_row({
            "case": self.case, "mode": self.mode, "variant": "", "P": "",
            "n_train": len(X_good), "n_aug": stats["n_accepted"],
            "cv_score_z": "", "train_mre_z": "", "val_mre_z": "",
            "holdout_mre_z": "", "train_mre_y": "", "holdout_mre_y": "",
            "holdout_mre_y_all": "",
            "verdict_y": "DRY-RUN (no simulation)",
            "aug_sims_run": 0, "aug_sims_kept": 0, "encoder_file": ""})
        print(f"[resampling dry-run] accepted={stats['n_accepted']}/"
              f"{stats['n_generated']}  reject_rate={stats['reject_rate']:.2f}  "
              f"corr_dist_sigma_draw={stats['corr_dist_sigma_draw']:.3f}  "
              f"corr_dist_original={stats['corr_dist_original']:.3f}  "
              f"coverage={stats['coverage_ratio_mean']:.2f}")
        return stats

    def run_resampling_transform(self):
        self.run_resampling()
        return self.run_transform(self.p_grid, self.rank_variant)

    # ── sweep visualization ────────────────────────────────────────────────

    def _plot_p_sweep(self, rows, plots):
        """P vs MRE: val(y) / holdout(y) / holdout(z), log-x. One y-axis.
        Picked candidate marked. (Small multiples would be overkill -- these
        three share the MRE-% unit.)"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        power_rows = [r for r in rows if r["variant"] == "power"]
        if len(power_rows) < 2:
            return
        Pvals = np.array([r["P"] for r in power_rows], dtype=float)
        val_y = [r["val_mre_y"] for r in power_rows]
        hold_y = [r["holdout_mre_y"] for r in power_rows]
        hold_z = [r["holdout_mre_z"] for r in power_rows]
        winner = min(rows, key=lambda r: max(r["train_mre_y"], r["val_mre_y"]))

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        fig.patch.set_facecolor(_SURF)
        ax.set_facecolor(_SURF)
        ax.set_xscale("log")
        ax.plot(Pvals, val_y, "o-", color=_CAT1, linewidth=2, markersize=6,
                label="validation MRE (decoded y)")
        ax.plot(Pvals, hold_y, "s-", color=_CAT2, linewidth=2, markersize=6,
                label="hold-out MRE (decoded y)")
        ax.plot(Pvals, hold_z, "^-", color=_CAT3, linewidth=1.5, markersize=6,
                label="hold-out MRE (z-space)")
        ax.axhline(self.criteria, color=_INK2, linewidth=0.8, linestyle="--")
        ax.text(Pvals[0], self.criteria, f" criteria {self.criteria}% ",
                fontsize=8, va="bottom", ha="left", color=_INK2)
        if winner["variant"] == "power":
            ax.axvline(winner["P"], color=_INK2, linewidth=0.8, linestyle=":")
            ax.text(winner["P"], ax.get_ylim()[1] * 0.96, f" winner P={winner['P']:g} ",
                    fontsize=8, ha="right", color=_INK2)
        ax.set_xlabel("power exponent P (log scale)")
        ax.set_ylabel("MRE %")
        ax.set_title(f"Case {self.case} ({self.study_tag}): P-sweep, "
                     f"{self.mode} arm\n"
                     + (f"rank variant: train={next((r['train_mre_y'] for r in rows if r['variant']=='rank'), 'n/a')}% "
                        if any(r["variant"] == "rank" for r in rows) else ""),
                     fontsize=10, color=_INK)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.tick_params(colors=_INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(_INK2)
            ax.spines[sp].set_linewidth(0.7)
        ax.grid(True, color=_INK2, alpha=0.15, linewidth=0.5)
        fig.tight_layout()
        out = os.path.join(plots, f"p_sweep_case-{self.case}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Offline Phase-10 fit reproducer / experiment runner "
                    "(no Chemkin). Reproduces the pipeline's fit on one case's "
                    "existing DM + simulation data; --experiment adds the "
                    "transform / resampling arms.")
    ap.add_argument("study_dir")
    ap.add_argument("case", type=int)
    ap.add_argument("temperature", type=float, help="nominal target T (K)")
    ap.add_argument("--criteria", type=float, default=2.0)
    ap.add_argument("--tol-grid", nargs="+", default=None,
                    help="screening-tolerance grid, floats or 'inf'")
    ap.add_argument("--k-grid", nargs="+", type=int, default=None)
    ap.add_argument("--order-grid", nargs="+", type=int, default=None)
    ap.add_argument("--ridge-grid", nargs="+", type=float, default=None)
    ap.add_argument("--experiment", default=None,
                    choices=["original", "transform", "resampling",
                             "resampling_transform"],
                    help="experiment arm (Phase10ExperimentRunner)")
    ap.add_argument("--p-grid", nargs="+", type=float, default=None,
                    help="power-exponent sweep, e.g. 1.0 0.5 0.2 0.1 0.05 0.02")
    ap.add_argument("--rank", action="store_true",
                    help="include the rank (gradual-rise) variant")
    ap.add_argument("--holder-linear", action="store_true",
                    help="include the per-sample p-Holder holder_linear variant")
    ap.add_argument("--p-lo", type=float, default=0.02,
                    help="holder_linear: per-sample p lower bound (default 0.02)")
    ap.add_argument("--n-new", type=int, default=2000,
                    help="resampling arm: candidate volume")
    ap.add_argument("--out-dir", default="./prs_experiment_out")
    ap.add_argument("--study-tag", default=None)
    ap.add_argument("--cache-dir", default="./prs_diag_cache")
    args = ap.parse_args()
    kwargs = {"criteria": args.criteria}
    if args.tol_grid is not None:
        kwargs["tol_grid"] = [float("inf") if str(v) == "inf" else float(v)
                              for v in args.tol_grid]
    if args.k_grid is not None:
        kwargs["k_grid"] = list(args.k_grid)
    if args.order_grid is not None:
        kwargs["order_grid"] = list(args.order_grid)
    if args.ridge_grid is not None:
        kwargs["ridge_grid"] = list(args.ridge_grid)
    if args.experiment:
        runner = Phase10ExperimentRunner(
            args.study_dir, args.case, args.temperature, mode=args.experiment,
            tol_grid=kwargs.get("tol_grid"), k_grid=kwargs.get("k_grid"),
            order_grid=kwargs.get("order_grid"),
            ridge_grid=kwargs.get("ridge_grid"),
            p_grid=args.p_grid, rank_variant=args.rank,
            holder_linear=args.holder_linear, p_lo=args.p_lo,
            n_new=args.n_new,
            out_dir=args.out_dir, study_tag=args.study_tag)
        runner.run()
    else:
        d = Phase10CaseDiagnostics(args.study_dir, args.case, args.temperature,
                                   cache_dir=args.cache_dir, **kwargs)
        d.run()
