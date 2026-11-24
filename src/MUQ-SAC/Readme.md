# Phase 10 p-PRS → MRE < 2%: response transforms + correlation-aware resampling

Companion to `phase10_prs.py` (2026-08-28). This documents the two new
experiment levers added to Phase 10 — **response coordinate transforms in R
space (encoding/decoding)** and **correlation-aware resampling** — how they
work step by step, how to run the four experiment arms, and how to invert the
transforms during Phase 11 (optimization).

All new code lives in `phase10_prs.py`:

| Class / helper | Role |
|---|---|
| `ResponseEncoder` | per-case scalar response transform y = ln(τ) → z, with the exact inverse persisted to disk |
| `CorrelationResampler` | correlation-aware candidate generation in compressed-ζ space (band-gated, deduped) |
| `Phase10ExperimentRunner` | offline (no Chemkin) multi-arm experiment on an existing study |
| `Phase10PRS` (extended) | live pipeline wiring: `prs_experiment_mode` arms, augmentation step, decode-patched y-space verdicts |
| `load_encoder` / `encode_response` / `decode_response` | Phase-11 helpers to invert a saved transform |

`run_script.py` only gained config plumbing (parses optional `target.opt`
Stats keys into `experiment_cfg`, passes it in the Phase-10 ctx, and archives
the new output folders). `IdentifiablePRS.py`, `ResponseSurface.py`, and
`AdaptiveResampler.py` are **unchanged** — the transform is applied to the
response arrays before the search/fit and decoded back for reporting.

---

## 1. The four experiment arms

Selectable per run via `prs_experiment_mode` (default `original`):

| mode | what happens | plot folder |
|---|---|---|
| `original` | today's Phase 10 pipeline, byte-identical | `Plots` |
| `transform` | response encoding/decoding only | `Plots_transfrom` |
| `resampling` | correlation-aware augmentation only | `Plots_re-sampling` |
| `resampling_transform` | both | `Plots_re-sampling_transform` |

(The folder names are verbatim from the requirement — `Plots_transfrom` is
spelled that way on purpose.)

**Ordering inside a case** (transform arms): screening (band validity,
sentinel, non-finite, non-physical — all on **raw** y) → optional
correlation-aware augmentation (raw simulated y) → 3-way split → **encode** →
CV hyperparameter search + fit on z → decode-patch → y-space verdict. The
encoding happens *after* screening because the screening predicates are
physical and defined on raw ln(τ).

---

## 2. How the response transform works (encoding / decoding)

Each case's response is one scalar per sample: y = ln(τ_us). Three variants,
all monotone and exactly invertible:

### 2a. Power (Hölder-mean family) transform

    z = (y^P − 1) / P,   P ∈ (0, 1]        (P → 0⁺ tends to ln y)
    y = (1 + P·z)^(1/P)                    (exact inverse)

With P → 0⁺ the transform approaches the log map and compresses the
multiplicative dynamic range of τ toward a linear scale (the "scalarize each
sample's response with a power almost towards zero" idea — the P→0 limit of
the power mean is the geometric mean). Since y must be > 0, y is floored at
`y_floor = max(1e-6, min(y_fit))`; decode clamps `1 + P·z > 0`.

### 2b. Rank (gradual-rise) transform

    z_i = interp(y_i ; sorted_y, z_grid),  z_grid = (rank − 0.5)/N

The sorted response profile becomes a perfectly linear rise from ~0 to ~1
("a gradual rise or linear rise that is easy to learn, based on their
response value"). Encoding new samples goes **through the fitted table**
(never re-ranked on their own batch), so train/val/hold-out stay on one
comparable scale; out-of-table values clamp to the endpoints. The table is
stored on unique y values with their tied-block average ranks (ties must
never reach `np.interp`'s duplicate-xp path). Decode is the inverse lookup.

**Caveat (measured):** rank z lives in (0, 1), so the *relative*-error
metric used by `ResponseSurface.test` diverges near z→0 and misleads the CV
hyperparameter search (the z-space numbers printed/CSV'd for rank are
RMSE via `ResponseEncoder.z_metric`, but the search itself still selects on
relative error). On both validation studies the rank variant therefore
picked worse hyperparameters than the raw fit — pilot hold-out 19.7% vs
14.8%, TCC 6.2% vs 0.56%. Use the rank variant for diagnostic/profile
visualization; the power variant is the operative one.

### 2b′. Holder-linear transform (PER-SAMPLE p-Hölder scalarization)

This is the "manipulate the response-surface structure" transform: instead
of one global power P, **every sample gets its own p_i**, solved so the
transformed response surface becomes an almost-linear "gradual rise" when
the samples are **arranged by response magnitude**.

The map is the same Hölder/power-mean family per sample,

    z_i = (y_i^{p_i} − 1) / p_i,     p_i ∈ [p_lo, 1],  p_lo = 0.02 (default)

with `g(p; y) = (y^p − 1)/p` **strictly increasing in p on [p_lo, 1] for
y > 1** (`g'(p) = (y^p·(p·ln y − 1) + 1)/p² ≥ 0`; `g(p_lo) ≈ ln y`,
`g(1) = y − 1`). Each sample's feasible z-interval is therefore
`[L_i, U_i] = [g(p_lo; y_i), y_i − 1]`.

**Fit algorithm** (`ResponseEncoder._fit_holder_linear`):

1. Require `min(y_fit) > 1` (y = ln(τ_us) always satisfies this — the map is
   monotone in p only for y > 1).
2. Sort samples by response magnitude; rank `r = 0…n−1`.
3. Per-sample feasible band `[L_i, U_i]` from `p_lo` and `p = 1`.
4. **Target line**: least-squares fit through the band *midpoints* vs rank;
   its endpoints are clipped into the feasible band
   (`z_min = clip(intercept, L_0, U_0)`,
   `z_max = clip(intercept + slope·(n−1), L_{n−1}, U_{n−1})`), so the
   target is feasible by construction. Degenerate (near-constant y) guard
   falls back to the band endpoints.
5. Target profile: `z_t(r) = z_min + (z_max − z_min)·r/(n−1)` — the gradual
   rise.
6. Per sample: `z_i = clip(z_t(r_i), L_i, U_i)` and solve `p_i` — closed
   form at the clip edges (`p_lo` or 1.0), **bisection on
   `g(p; y_i) = z_i`** elsewhere (60 iterations, tol 1e-12; g is monotone).
   Where the raw profile is too concave to linearize exactly, the sample
   clips to its band edge and the residual is recorded.
7. Ties (duplicate y) collapse into the decode table with their block's mean
   z — same discipline as rank, so `np.interp` never sees duplicate-xp.

**Encode/decode**: both are 1-D monotone interpolation through the fitted
sorted table (`encode: np.interp(y, table_u, table_z)`,
`decode: np.interp(z, table_z, table_u)`) — exact at every fit point, smooth
and out-of-range-clamping for new points. The user's requirement to
**remember all ~6000 p values** is honored: every per-sample p_i is saved
(see §2d), usable by Phase 11 to invert any z back to y.

Fit diagnostics are recorded in the encoder JSON (`stats`): `n_clipped_plo`,
`n_clipped_p1`, `rms_residual` (of z vs the target line), `linear_r2`,
`p_min/p_max`. `criteria_z_equiv` uses the median-y sample's solved p:
`(dy/y) = (dz/z)·(y^p − 1)/(p·y^p)`.

**Metric caveat (same honesty contract as §2c)**: holder-linear z lives in a
compact range (≈ [2.9, 4.4] on the pilot vs y ∈ [5.1, 7.7]), so z-space
relative errors compress ~1.5–2×. Decoded y-space verdicts remain the only
pass/fail metric.

### 2c. Honesty contract (dual-space reporting)

The PRS is searched and fit in z-space, but **every pass/fail verdict and the
existing ledgers (`Prediction_errors.csv`, `Prediction_errors_detailed.csv`,
`optimization_cost.csv`) are reported in DECODED y-space**. The z metric is
not comparable to the y metric:

    (dy/y) = (dz/z) · (y^P − 1)/(P·y^P)      →  |dz/z| ≈ |dy/y| / 1.1–1.9

so z-space relative errors shrink by ~1.1–1.9× on this data without any real
improvement. All z-space numbers (CV score, z-MRE, P) go to
`prs_experiment_results.csv` instead. A PASS verdict requires the **decoded
y-space** hold-out to clear `MRE_criteria`.

### 2d. Encoder persistence (what Phase 11 needs to invert)

`Opt/Data/Encoding_Decoding/`:

```
encoder_meta.json                # {case: {variant, encoder_file}} — active variant
encoder_case-<N>_power.json      # {variant, P, y_floor, z_shift, z_scale,
                                 #  criteria_y, criteria_z_equiv, n_fit_samples,
                                 #  y_min, y_max, fitted_on, table_file}
encoder_case-<N>_rank.json       # + table_file → sorted_y_case-<N>.npy
sorted_y_case-<N>.npy            # rank table (unique y, average-rank z), 2 cols
encoder_case-<N>_holder_linear.json
                                 # {variant, p_lo, z_min, z_max, stats, ...}
p_values_case-<N>.npy            # holder_linear: ALL per-sample p_i (n × 1) —
                                 # the "remembered" ~6000 p values
holder_table_case-<N>.npy        # holder_linear: unique (y, z) decode table, 2 cols
```

These files are self-contained — Phase 11 needs no study data, only the
folder. Invert in the optimization phase with:

```python
from phase10_prs import decode_response, encode_response, load_encoder
y_pred = decode_response(z_pred, case_index=0, variant="power",
                         enc_dir="<study>/Opt/Data/Encoding_Decoding")
y_pred = decode_response(z_pred, case_index=0, variant="holder_linear",
                         enc_dir="<study>/Opt/Data/Encoding_Decoding")
# or resolve the study dir automatically: decode_response(z_pred, 0, "power", study_dir=<study>)

# the raw per-sample p values (e.g. for an explicit g-inverse y=(1+p·z)^(1/p)):
import numpy as np
p_vals = np.load("<study>/Opt/Data/Encoding_Decoding/p_values_case-0.npy")
```

`variant=None` (default) reads `encoder_meta.json` for the case's active
variant. The PRS object returned to Phase 11 predicts **z** in transform
arms; wrap every call with `decode_response` before using the value (and
`encode_response` if a y-space quantity must enter a z-space computation).

---

## 3. How the resampling takes place (correlation-aware, step by step)

Goal: new design-matrix candidate rows whose per-reaction perturbation
covariance **respects the correlation structure** of the case's band-valid
data, optionally blended with the mechanism's stated physical correlation —
"correlation is taken into account".

Inputs: the case's good (band-valid, screened) rows `X_good` (n × n_prs,
compressed ζ), the band blocks from Phase 10 STEP 2, and per-reaction
`unsrt[rxn].cholskyDeCorrelateMat` (L with L Lᵀ = Σ_r, the same covariance
the band blocks were built from).

1. **Per-reaction physical correlation.** For reaction r with selected
   parameters `sel_local`: Σ_r = L Lᵀ restricted to `sel_local`
   (`IdentifiablePRS.reduced_cholesky`); `C_r = corr(Σ_r)`.
2. **Robust empirical covariance.** The good rows contain heavy-tailed rows
   (the pre-cap ζ_Ea era) that pass the band only through full-combination
   cancellation. Each column is **winsorized** to median ± (q95−q05) before
   estimating the covariance, so the draw scale follows the *bulk* of the
   data (a raw covariance gave 0/4500 band-valid candidates).
3. **Diagonal shrinkage** (Ledoit–Wolf style): `S_reg = (1−η)·S +
   η·diag(diag(S))` with η = 0.01. Never `tr(S)/m·I` — the ζ_Ea columns
   (~2050 std) and ζ_A columns (~1.6 std) share a reaction, and the
   trace-average target inflates the small columns by ~10⁴.
4. **Shape blend + empirical scale**:

       C_draw     = (1−λ)·C_r + λ·corr(S_reg)          (λ = empirical_weight)
       Σ_draw     = diag(std(S_reg)) @ C_draw @ diag(std(S_reg))

   The *correlation shape* is the blend; the *variance scale* comes from the
   band-valid observations only. Taking the scale from Σ_r directly
   recreates the near-singularity blowup PAAR fixed (its dominant
   eigenvector is the band-restricted direction).
5. **Default λ = 1.0 (pure empirical)** — measured on the pilot (n-heptane,
   1293 K): the physical correlation is near-rank-1 (corr(n, Ea) = −0.996)
   and even a 5% blend (λ=0.95) collapses the band-valid yield from 33% to
   6%; λ=0.9 collapses it to 1%. The band-valid data lives in a *shell* of
   the physical prior with its own correlation ((A, n) = −0.93). Lower λ
   remains selectable and reproduces this negative result.
6. **Draws.** One **global** low-discrepancy block per batch (scrambled
   Sobol → Normal CDF, dim = n_prs) is sliced per reaction:
   `δ_r = u_r @ L_draw_r.T`, `row = μ + δ` (μ = good-row mean for
   `center_mode="pool_center"`, or k-means cluster means drawn
   proportionally to cluster size for `"kmeans"`). One joint draw is
   mandatory: consecutive-seed per-reaction Sobol blocks stay
   coordinate-wise correlated (~0.4–0.7) across blocks and would fabricate
   cross-reaction correlation.
7. **Per-column clip.** Every candidate column is clipped to
   ±1.2×(q95−q05 of the good rows) — clipping preserves the correlation
   structure, unlike per-column rescaling.
8. **Band gate + dedup.** Every candidate must pass
   `IdentifiablePRS.band_valid_mask(X, blocks, band_tolerance)` — the exact
   screening test — and must be farther than `dedup_eps` (Euclidean,
   compressed space) from every reference row. Generation rounds repeat
   until `n_new` candidates are accepted or the round budget is spent
   (shortfall is reported, never silently padded).
9. **Live mode** (`do simulation: true` in the `Stats:` block — see §4):
   accepted rows are simulated via the existing `_simulate_new_samples`
   machinery (fingerprint-cached under
   `Opt/Data/Outlier_Simulations/sim_data_case-<N>_pass-aug.lst` + a
   `.meta.json` holding the SHA1 of the rounded mini-DM rows), the new y is
   screened (sentinel / non-finite / non-physical — a check Stage 3
   historically skipped; this step does not), and the surviving rows augment
   xData/yData before the split. Dry-run (default) generates, gates, saves,
   and plots — zero simulation calls.

   **Cross-arm cache reuse**: the cache key is the mini-DM fingerprint
   alone, so `resampling` and `resampling_transform` runs with the same
   `resample_seed`/`resample_n_new`/screened X feed the identical candidate
   set and the second run **reuses** the first run's augmentation `.lst`
   (zero duplicate simulations — the cache hit returns before any
   YAML/directory/solver work). `run_script.py`'s `archive_previous_study`
   therefore *copies* (not moves) `Opt/Data/Outlier_Simulations/` when a
   live resampling run starts, so the cache survives across arms.

### Offline verification metrics (no simulation needed)

Saved to `CorrelationAug_stats_case-<N>.json` + plotted in
`corr_comparison_case-<N>.pdf` (4-panel: corr(good) / corr(candidates) /
corr(Σ_draw target) / |cand − target|) and `coverage_case-<N>.pdf`
(per-column candidate-spread/good-spread ratio):

- `corr_dist_sigma_draw` — Frobenius distance / n_prs between the candidates'
  correlation matrix and the block-diagonal Σ_draw target (sampling-noise
  floor ≈ 0.03–0.05 at n=500; observed 0.034).
- `corr_dist_original` — same against the good rows' empirical correlation.
- `coverage_ratio_mean` — mean per-column (q95−q05) ratio; ≈1.0 means the
  candidates cover the observed range without blowout (observed 1.0).
- `reject_rate`, `worst/median_violation_ratio` — band-gate accounting.

---

## 4. Config schema (target.opt → `Stats:` block, every key optional)

```yaml
Stats:
  # ── experiment arms ──
  prs_experiment_mode: original        # original | transform | resampling | resampling_transform
  # live simulation switch (user's exact wording)
  do simulation: true                  # true  = resampling arms launch live chemkin
                                       # false = dry-run (candidates + plots only)
  # aliases accepted for the same switch: do_simulation / resample_dry_run (legacy)
  # transform arm
  transform_variant: power             # power | rank | holder_linear
  transform_P: 0.1                     # fixed P for live runs (0 < P <= 1)
  transform_P_grid: [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]   # offline sweep
  transform_p_lo: 0.02                 # holder_linear: per-sample p lower bound
  transform_standardize: false         # optional z-shift/z-scale post-map
  transform_rank_fit_on: all           # all | train_val  (rank table fit subset)
  # resampling arm
  resample_n_new: 2000                 # candidate volume per case
  resample_empirical_weight: 1.0       # lambda: 0 = physical corr, 1 = empirical
  resample_band_tolerance: 1.0         # candidate validity gate
  resample_qmc: true                   # scrambled-Sobol -> Normal draws
  resample_seed: 20260828
  resample_center_mode: pool_center    # pool_center | kmeans
  resample_dedup_eps: 0.000001
  # Stage 3 x transform
  stage3_with_transform: false         # Stage 3 disabled in non-original arms
```

Missing keys → defaults → `original` → zero behavior change. Switch
semantics: `do simulation` / `do_simulation` are *positive* keys —
`do simulation: true` → live sims, `false` → dry-run. If neither positive
key is present, the legacy *negative* key `resample_dry_run` is used
verbatim (`true` = dry-run, its historical meaning).

---

## 5. Offline usage (no Chemkin — runs on existing study data)

```bash
cd src/MUQ-SAC

# A. Repro gate (must reproduce the published numbers)
python3 phase10_prs.py /data/KRUNAL/pilot_5case_case0 0 1293.3097 --criteria 2.0
#    expect train 15.423% / holdout(band-valid) 14.817%
python3 phase10_prs.py /data/KRUNAL/MULTI-STAGE_OPTIMIZATION/TEST_CORRELATIONS_CORRECTED \
    0 1127.2686 --criteria 1.2 --tol-grid 1.0 inf --ridge-grid 0 1e-7
#    expect train 0.894% / holdout 0.644%

# B. Transform arm (P-grid model selection on VALIDATION, one hold-out report)
python3 phase10_prs.py /data/KRUNAL/pilot_5case_case0 0 1293.3097 \
    --experiment transform --p-grid 1.0 0.5 0.2 0.1 0.05 0.02 --rank \
    --criteria 2.0 --out-dir ./prs_experiment_out

# C. Resampling dry-run (candidates + correlation plots, no simulation)
python3 phase10_prs.py /data/KRUNAL/pilot_5case_case0 0 1293.3097 \
    --experiment resampling --n-new 2000 --out-dir ./prs_experiment_out

# D. Both
python3 phase10_prs.py /data/KRUNAL/pilot_5case_case0 0 1293.3097 \
    --experiment resampling_transform --n-new 2000 --p-grid 1.0 0.2 \
    --out-dir ./prs_experiment_out

# E. holder_linear transform arm (per-sample p-Hölder, TCC demo)
python3 phase10_prs.py /data/KRUNAL/MULTI-STAGE_OPTIMIZATION/TEST_CORRELATIONS_CORRECTED \
    0 1127.2686 --criteria 2.0 --experiment transform --p-grid 1.0 0.2 0.1 \
    --holder-linear --p-lo 0.02 \
    --out-dir /data/KRUNAL/phase10_demo_tcc --study-tag TCC_case0_holder_linear
```

### Live runs (the four arms, from the study dir)

```bash
cd /data/KRUNAL/pilot_5case_case0
# per run: set prs_experiment_mode (+ transform_variant for transform arms)
# in target.opt's Stats block, keep `do simulation: true`, then:
nohup python3.9 run_script.py target.opt > pilot.log 2>&1 &
# run order: original (outlier_retraining: false) → transform (power) →
# transform (holder_linear) → resampling (~2000 new sims) →
# resampling_transform (0 new sims — fingerprint cache reuse)
```

Each run archives the previous run's `Plots*` folders / encoders /
`prs_experiment_results.csv` into `CASE_STUDIES/<tag>/`; the four plot
folders are consolidated back after the last run.

Everything lands under `out_dir/<study_tag>/` — the four `Plots*` folders,
`Encoding_Decoding/`, `prs_experiment_results.csv`, per-arm coefficient
caches (`_cache_<arm>/`, isolated because the cache key is case-only). The
offline runner **never writes into the study directory**.

Runtime note: the pilot's CV search is ~96 combos × 10 folds; a 6-P sweep
multiplies that. For a sweep, use reduced grids (`--k-grid 8 12 20
--ridge-grid 0 0.3`), then confirm the winner once with the full grids.

---

## 6. Stage 3 × transform interaction, and ledger policy

- **Stage 3 (adaptive outlier retraining) is disabled by default in every
  non-original arm.** Its outlier thresholds and accept/reject scores would
  mix z- and y-space. Escape hatch `stage3_with_transform: true` runs Stage 3
  fully in z-space (the target becomes `criteria_z_equiv` via
  `ResponseEncoder.mre_equivalent_z`, and every simulated y is encoded with
  the case's encoder before joining the training stack). Use only after the
  transform arm is proven without it.
- **Ledgers**: `Prediction_errors.csv` / `Prediction_errors_detailed.csv` /
  `optimization_cost.csv` keep their schemas and hold y-space numbers in
  every arm. z-space numbers and arm metadata go to
  `prs_experiment_results.csv` (live: study root; offline: `out_dir/
  <study_tag>/`). Augmentation simulations are tracked only there
  (`aug_sims_run`/`aug_sims_kept`) — the COST_FILE schema is frozen.
- **Rank-table mild leakage**: with `rank_fit_on: all` (default) the table
  includes hold-out y values — the same information any response
  standardization leaks; `train_val` removes it at the cost of clamped
  out-of-table decoding.

## 7. Bugs fixed while debugging Phase 10 (2026-08-28)

1. **Stale-coefficient-cache hazard** in `Phase10CaseDiagnostics`: the final
   `create_response_surface()` read/wrote `cwd/Data/ResponseSurface/
   responsecoef_case-N.csv` — the launch cwd, not the cache dir — so a stale
   file silently short-circuited the final fit and a first run littered the
   source tree (observed as a hard crash from a clean tmp cwd). Fixed by
   chdir-ing into `cache_dir` + removing the stale file first (mirrors the
   live pipeline's STEP 6).
2. Hardcoded `1.0` band tolerance in the diagnostics hold-out print now uses
   the configured `band_tolerance_eval`.
3. `run_script.py` `archive_previous_study` now archives
   `Opt/Data/Encoding_Decoding`, the three new `Plots_*` folders, and
   `prs_experiment_results.csv`, so successive runs never overwrite a
   previous run's artifacts.
4. Documented (not changed): Stage 3 never screens simulated y for
   sentinel/non-finite/non-physical — the new augmentation step does.
5. **`holder_linear` encoder variant** (2026-08-28): per-sample p-Hölder
   scalarization onto a linear rank profile (§2b′) — every sample's p_i is
   solved by bisection on `g(p;y)=(y^p−1)/p`, all p_i persisted to
   `p_values_case-<N>.npy`, decode = monotone table interpolation.
6. **`do simulation: true` flag** (2026-08-28): the user-facing live-sim
   switch for the resampling arms; `do_simulation` and `resample_dry_run`
   remain as aliases. `archive_previous_study` now *copies* the
   `Outlier_Simulations` cache for live resampling runs so the
   resampling_transform arm reuses the resampling arm's simulations via the
   fingerprint cache.
7. **`self.design` → `self.design_type`** in two Stage-3 messages (latent
   `AttributeError` if ever reached under a non-`A1+B1+C1` design), and a
   dead no-op removed from `CorrelationResampler.generate`.
