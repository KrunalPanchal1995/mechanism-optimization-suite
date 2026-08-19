# Adaptive outlier re-training: how it works, what went wrong, and the fix

This documents the adaptive outlier re-training feature (`AdaptiveResampler.py` +
the wiring in `run_script.py`), why the previous default made a real fit worse,
and the new default (`hybrid`) that replaces it.

> **2026-07 update — see `report/` for the full PAAR (Physics-Aware Adaptive
> Resampling) writeup.** Everything below (the `local`/`dm_global`/`hybrid`/
> `gap_fill` diagnosis and mechanics) remains accurate and those modes are
> unchanged, but it turned out **not** to be the deepest cause of the failures
> this feature was built to fix — `dm_global`'s ~4x volume blowup and lack of
> outlier-concentration made a bad situation *worse*, but the *underlying*
> reason the PRS needed rescuing at all (up to 59.14% hold-out error on Case 6)
> is that the PRS is fit directly on decorrelated Arrhenius coordinates ζ whose
> covariance is near-singular (eigenvalues `[7e-7, 1.07, 45.9]`), so a
> near-unidentifiable coordinate gets sampled up to ±4900 and the quadratic
> design matrix ends up with condition number ~1e11. See
> `IdentifiablePRS.py`'s module docstring and `report/main.tex §2` for the full
> derivation, and §2.5 below for the new `d_optimal` mode (now the default)
> this diagnosis motivated.

## 1. What the loop does end-to-end

After a case's PRS (Polynomial Response Surface, order 2 by default —
`ResponseSurface.py`) is fit and tested against a held-out split
(`train_test_split`, 10% test), `run_script.py`'s main PRS loop checks
`Response.selection` (set by `ResponseSurface.test`, based on `MRE_criteria`).
If it's `0` (i.e. some hold-out point exceeded the error criterion), it calls
`retrain_around_outlier(...)`:

1. Find every hold-out point whose `|relative error %|` exceeds a **tighter**
   target than the pass/fail threshold: `target = target_fraction * MRE_criteria`
   (default `target_fraction=0.5`, so if `MRE_criteria=2%`, the loop keeps
   pushing until the worst error is `<= 1%`, not just `<= 2%`).
   (`AdaptiveResampler.find_outliers`)
2. Build a small "mini design matrix" of new rows around those outlier(s)
   (`OutlierResampler.resample_hybrid` / `.resample` / `.resample_global`,
   selected via `outlier_cfg["sample_mode"]`).
3. Simulate just those new rows for just this one case
   (`_simulate_new_samples`, isolated under
   `Opt/Outlier_Resample/case-<case>/pass-<n>/`, results also copied to
   `Opt/Data/Outlier_Simulations/` for inspection — already-generated
   YAMLs/results are reused, not regenerated).
4. Append the new rows + simulated targets to the training set, bust the
   cached PRS coefficient CSV (`ResponseSurface.create_response_surface`
   silently reuses `Data/ResponseSurface/responsecoef_case-<case_index>.csv`
   if present — it MUST be deleted before a real refit), and refit.
5. Repeat (each pass reseeds the resampler `seed + it` so passes don't just
   resample the same points) until the worst hold-out error is `<= target` or
   `outlier_cfg["max_iter"]` passes are used up (default 5). The best-so-far
   surface is always kept; any exception aborts the loop and keeps the last
   good surface (never crashes the pipeline).

The whole thing is agnostic to **Full vs Partial PRS** and to
**A-facto vs A1+B1+C1 design** — see `AdaptiveResampler.py`'s module docstring
for the FULL-space vs PRS-space coordinate mapping.

## 2. The four sampling modes

`outlier_cfg["sample_mode"]` selects one of:

### `local` (original attempt — clusters around the outlier)
`OutlierResampler.resample()`. For each outlier, and for each available curve
class (A/B/C, from the cached `*_type_samples*.pkl` written by
`DesignMatrix.py`), it finds the `knn` nearest same-class cached curves and
draws new points as random convex combinations of those neighbours, scaled by
`delta_fraction` — i.e. small perturbations that stay on the class manifold,
very close to the outlier. **Problem observed**: piling ~200 points into one
tight neighbourhood distorts the *global* fit — visible in
`only generating samples in outlier local region.pdf` as vertical "tower"
clusters of training points that sit off the parity line, even though the
reported error looked decent (3.055% train / 0.708% test) — the global fit
was not what the numbers implied.

### `dm_global` (previous default — recombines cached curves globally)
`OutlierResampler.resample_global()`. Instead of clustering near the outlier,
it mirrors how `DesignMatrix.getSample_partial` builds the *original* design
matrix: an "unshuffled" block that jointly recombines cached per-reaction
class curves into new cross-reaction row combinations (a Sobol/LHS/numpy draw
picks, per reaction, WHICH cached curve to reuse), plus a "shuffled" block
(`shuffle_factor`x independent re-permutations of that same unshuffled data).

**This is why the fit got worse** (`Parity_plot_case_3_TESTING.pdf`: 11.6%
train / 10.3% test, badly scattered). Two compounding problems, found by
reading the code:

1. **The configured budget was never actually respected.** The unshuffled
   block size is `ceil_pow2(min(uns_cap, N_new))`. With the defaults
   (`uns_cap=256`, `N_new=200`), that's `ceil_pow2(200) = 256` — Sobol's
   power-of-2 requirement rounds *up*, not down. The shuffled block then adds
   `shuffle_factor (2) * 256 = 512` more. Total new rows per pass ≈
   `256 + 512 + k(outliers) ≈ 769`, **not** the configured 200 — roughly **4x**
   the intended volume, silently. Over `max_iter=5` passes that's up to
   **~3,845** extra rows added to a per-case training set that is very likely
   of comparable or smaller size, before any of this was surfaced to the user.
2. **The new rows are pure recombinations of already-cached values** — for
   each reaction, a value is picked from a fixed pool of curves *already used*
   to build the original training data; only the cross-reaction *combination*
   is new. This is exactly the "resampling using the already generated
   samples" behaviour recalled from the last run. Because no reaction's
   per-column marginal distribution actually changes, and the draw is uniform
   over the whole cached pool (not concentrated anywhere near the outlier),
   this mode does nothing to specifically fix the region that was failing —
   it just adds a large volume of combinatorially "new" but informationally
   redundant rows spread evenly over the whole domain. At ~4x the intended
   volume, a low-order (quadratic) least-squares fit ends up dominated by
   these rows rather than by genuine local structure, which is consistent
   with the fit moving further from the parity line rather than closer to it.

### `hybrid` (new default — capped local + sparse global)
`OutlierResampler.resample_hybrid()`. Directly addresses both problems above:

- **Hard cap, actually enforced in code**: `n_total = min(N_new, HARD_CAP_DEFAULT)`
  where `HARD_CAP_DEFAULT = 500` (a module-level constant in
  `AdaptiveResampler.py`) — regardless of `N_new`, `uns_cap`, or
  `shuffle_factor`, a single pass can never add more than 500 rows. Five passes
  (the default `max_iter`) ⇒ at most 2,500 additional rows per case, matching
  the requested budget exactly.
- **The outlier row(s) are always included verbatim** (`row_class == 'OUT'`),
  counted against the cap.
- The remaining budget splits `local_fraction` (default `0.4`) /
  `1 - local_fraction`:
  - **`LOC` block** — reuses the exact per-outlier, per-class local
    interpolation `resample()` already does (`_build_block` /
    `_sample_block_classaware` / `_sample_block_box` fallback), just
    re-budgeted to `n_local` instead of the old `N_new`. This is the part
    that concentrates new information where the error actually is.
  - **`GLB` block** — a single low-discrepancy draw of `n_global` anchor rows
    spread across the *whole* observed design-matrix range (reusing
    `_build_global_unshuffled`'s joint per-reaction pool recombination, or
    `_sample_block_box_global` for the no-class-pool / A-facto case) — but
    **with no shuffled/duplicate block**. This is what makes it "sparse":
    `n_global` real rows, not `n_global * (1 + shuffle_factor)`. It exists so
    regions of the domain *other* than the outlier's neighbourhood still get
    a light, evenly-spread top-up, instead of all-in-one-place clustering
    (the `local` mode's failure) or a dense blowout (the `dm_global` mode's
    failure).
- Row order is shuffled at the end ("just in case", matching `dm_global`'s
  existing convention).

`local` and `dm_global` remain fully intact and selectable (unchanged code,
verified by a regression test) for comparison/debugging — see `outlier_cfg`
below.

### `gap_fill` — the only mode that generates genuinely NEW curves

`OutlierResampler.resample_gap_fill()`. `local`, `dm_global`, and `hybrid` all
share one limitation: they only **recombine or locally perturb curves that
are already cached** from the original DM build (`*_type_samples*.pkl`). No
reaction's per-column marginal distribution ever actually changes — only
which *combination* of already-seen curves lands in a given row.

`gap_fill` instead calls `DesignMatrix.py`'s **real** curve generators —
`getClassA_Curves`/`getClassB_Curves`/`getClassC_Curves` for Full PRS, their
`getClass*_Curves_partial` counterparts (driven by
`DesignMatrix._get_rxn_param_indices`) for Partial PRS — to synthesize
brand-new Arrhenius curves that were never part of the original DM. This
needs a live `DesignMatrix` instance, lazily constructed inside
`OutlierResampler._get_dm_instance()` (imported lazily, not at module top, so
`local`/`dm_global`/`hybrid` keep working in environments where
`DesignMatrix.py`'s own hard dependencies — `qmc_sampling.py`, `/shuffle.so`
— aren't importable; `gap_fill` raises a clear `RuntimeError` only if it's
actually selected and those imports fail).

Structure, mirroring `DesignMatrix.getSamples()`/`getSample_partial()`'s own
real 20:80 unshuffled:shuffled split and 10:45:45 A:B:C ratio (for the
`'A1+B1+C1'` design):

1. For each class (A, B, C), draw `gapfill_oversample`x the needed count via
   a joint Sobol/LHS shadow block and call the real generator.
2. **Isolation filter** (`_isolation_filter`, module-level, reusable/testable
   in isolation): normalize both the candidates and a reference set by the
   reference set's own per-column range, build a `scipy.spatial.cKDTree` on
   the reference, and score each candidate by its distance to its **k-th**
   nearest reference neighbour (`gapfill_knn_k`, default 3). Reject
   candidates scoring below `gapfill_knn_radius_frac` (default 0.05) of a
   hard reject *and* a ranking key simultaneously; rank survivors by score
   (most-isolated/biggest-gap first). If a tight radius/k would reject
   everything, the filter falls back to ranking without hard-rejecting
   (never starves a pass).
3. Keep the most-isolated `n_a_req`/`n_b_req`/`n_c_req` candidates per class
   — these become the **unshuffled** block, and (like the real DM) are
   row-homogeneous by class across every reaction, since the joint shadow
   block feeds every reaction's generator from the same underlying draw.
   The running reference set grows after each class (A, then B, then C) so
   later classes don't re-fill the same gap an earlier class in this pass
   just used.
4. **Shuffled** block: reuses `_build_global_shuffled` verbatim on the
   chosen unshuffled per-reaction curves (`gapfill_shuffle_factor`x,
   default 4 — matching `DesignMatrix`'s own real shuffle factor, unlike the
   legacy `shuffle_factor` config key which stays default 2 and is
   `dm_global`-only), oversampled and isolation-filtered the same way.
5. Reactions where a real class-B/C draw is infeasible (Partial PRS, < 2
   selected params) are padded with extra class-A draws — mirroring
   `DesignMatrix.getSample_partial`'s own `V_` backfill exactly.
6. `'A-facto'` has no class or shuffle structure in the real DM either
   (`DesignMatrix.getSamples()`'s A-facto branch is one flat box draw), so
   `gap_fill` mirrors that asymmetry: a single isolation-filtered
   low-discrepancy block spanning the full budget, no shuffled block,
   `row_class='GAP'`.
7. Outlier row(s) prepended verbatim (`'OUT'`); same `HARD_CAP_DEFAULT=500`
   ceiling as `hybrid`; final row order shuffled, matching every other mode.

**The reference set is the one thing `run_script.py` grows across passes**,
unlike the other three modes (which reuse the static original DM every
pass): `retrain_around_outlier` passes `xData` unioned with every row added
by prior passes this run, so pass 2 doesn't re-fill a gap pass 1 already
used.

**A subtlety worth knowing**: in Partial-PRS mode, the `GLB` block's
class-pool recombination fills every reaction's full-space slot (selected or
not) for joint-coverage purposes — the same as `dm_global` already does. This
does **not** matter functionally: for Partial PRS, `_simulate_new_samples`
builds YAMLs from `prs_rows` (the compressed representation) + the selection
mask, never from `full_rows`' raw values; `full_rows` is only used elsewhere
for its row *count* (`simulation_manager2_0.getDirectoryList` indexes
`range(len(design_matrix_dict[case]))`). The real invariant —
`prs_rows == full_to_prs(full_rows)` for every row — always holds, and is
covered by a test. The `LOC` block additionally keeps unselected full-space
columns at exact zero (it round-trips through `prs_to_full`), which is a
stronger but non-essential guarantee.

### `d_optimal` — physics-aware D-optimal adaptive resampling (current default)

`AdaptiveResampler.DOptimalResampler` (subclasses `OutlierResampler`, reusing
its `_isolation_filter`, `find_outliers`, `_build_layout`/`prs_to_full`/
`full_to_prs`, `_load_class_pool`, `_get_dm_instance`). Replaces `hybrid` as
the default now that the PRS is fit in the identifiable subspace
(`IdentifiablePRS.py`): once the surface is linear in its fitted coefficients,
classical D-optimal design gives an *exact*, minimum-simulation-count answer
to "which candidate point reduces coefficient-estimate variance the most" —
for a candidate's basis-feature row `b` and current Gram matrix
`G = B_train^T B_train`, adding it increases `det(G)` by `(1 + b^T G^-1 b)`.

Two-part contract, both new relative to every mode above:

1. **Every candidate is band-validity-gated before it is even eligible.**
   `DOptimalResampler.band_valid_mask` calls `Uncertainty.is_band_valid` /
   `batch_violation_ratio` (or `IdentifiablePRS`'s dependency-light
   reimplementation, cross-validated to agree bit-for-bit — see
   `AdaptiveResampler.py`'s top-of-file comment) over the temperatures in
   `temps_for_validity`, which should span the reaction's *global*
   uncertainty-validity window (e.g. 300–2500K, matching what the real
   class-A/B/C samplers themselves check) — **not** just the one case's local
   T, since a resampled point feeds a full perturbed `mechanism.yaml` that
   must be physically plausible everywhere it might be evaluated. This is the
   only mode that can never spend a simulation on an implausible mechanism.
2. **Selection is greedy D-optimal, not geometric.** Candidates are drawn from
   the same local (near-outlier, class-aware) / global (sparse,
   low-discrepancy) machinery `resample_hybrid` uses
   (`generate_candidates`), scored by
   `score = acquisition_alpha * (D-optimality, normalized) + (1 -
   acquisition_alpha) * (RBF proximity to the failing point(s), residual-
   weighted)`, and picked one at a time with a Sherman-Morrison rank-1 update
   of `G^-1` after each pick (`select_d_optimal`) — the augmentation-only
   special case of Fedorov exchange (no swap/removal step, since existing
   training rows are never discarded here).

`resample_d_optimal(outlier_prs_rows, X_ref_prs, X_train_prs, build_basis,
target_weights=None, local_fraction=0.5, ridge_alpha=0.0)` is the entry point
`run_script.retrain_around_outlier` calls; `build_basis` should be the
*current* fit's `ResponseSurface.build_basis` (bound to its `feature_map`/
`order`) so candidates are scored in the exact basis being refit on. Returns
`(prs_rows, full_rows, sel_mask, row_class, info)` — one extra element,
`info`, vs. every other mode: a diagnostics dict (`n_candidates_generated`,
`n_candidates_band_valid`, `n_selected`, `n_budget`, `reject_rate`).

Verified by synthetic unit tests (`test_doptimal.py`, mock uncertainty
extractors implementing the real reduced-Cholesky formula so the physics is
genuine, not faked — `DesignMatrix.py`/`Uncertainty.py` aren't importable in
every environment this repo runs in): band-validity gating rejects a
20,000σ-equivalent perturbation with zero false accepts while accepting
well-scaled rows; greedy selection prefers isolated/novel candidates over
near-duplicates of the training set (>80% of picks in a controlled test);
`det(G)` increases monotonically with diminishing marginal gain; and
`acquisition_alpha = 0`/`1` correctly interpolate pure targeting vs. pure
D-optimality.

## 3. Config reference (`optInputs["Stats"]` keys, all optional)

| `outlier_cfg` key | Stats override key | Default | Meaning |
|---|---|---|---|
| `enable` | `outlier_retraining` | `True` | Master on/off switch |
| `N_new` | `outlier_N_new` | `500` | Requested new rows per pass (hard-clamped to 500 by `hybrid`; not clamped by legacy `local`/`dm_global`) |
| `delta_fraction` | `outlier_delta_fraction` | `0.15` | Local neighbourhood size (fraction of local knn step / box range) used by `LOC`/`local` |
| `seed` | `outlier_seed` | `42` | Base RNG seed (reseeded `seed + pass_index` each pass) |
| `max_iter` | `outlier_max_iter` | `5` | Hard cap on re-training passes per case |
| `target_fraction` | `outlier_target_fraction` | `0.5` | Loop target = `target_fraction * MRE_criteria` |
| `sample_mode` | `outlier_sample_mode` | `"d_optimal"` | `"d_optimal"` (default, see §2.5) / `"hybrid"` / `"local"` / `"dm_global"` / `"gap_fill"` |
| `band_tolerance` | `outlier_band_tolerance` | `1.0` | `d_optimal`-only: max allowed `max_{reaction,T}\|Δκ\|/f_prior` for a NEW point to be eligible at all |
| `validity_n_temps` | `outlier_validity_n_temps` | `12` | `d_optimal`-only: number of temperatures spanning [300,2500]K checked for candidate validity |
| `acquisition_alpha` | `outlier_acquisition_alpha` | `0.6` | `d_optimal`-only: D-optimality vs. targeting mix (1.0 = pure D-optimal, 0.0 = pure proximity-to-outlier) |
| `candidate_oversample` | `outlier_candidate_oversample` | `6` | `d_optimal`-only: candidates generated (pre band-validity/selection) per selected row |
| `method` | `outlier_method` | `"sobol"` | Low-discrepancy generator: `"sobol"` / `"lhs"` / `"numpy"` |
| `uns_cap` | `outlier_uns_cap` | `256` | Unshuffled block size cap — only used by legacy `dm_global` |
| `shuffle_factor` | `outlier_shuffle_factor` | `2` | Shuffled-block multiplier — only used by legacy `dm_global` |
| `local_fraction` | `outlier_local_fraction` | `0.4` | Share of the (post-outlier) hybrid budget spent on the `LOC` block; rest goes to `GLB` |
| `gapfill_oversample` | `outlier_gapfill_oversample` | `2` | `gap_fill` candidate-pool multiplier before isolation filtering |
| `gapfill_knn_radius_frac` | `outlier_gapfill_knn_radius_frac` | `0.05` | `gap_fill` isolation-filter reject radius (fraction of reference-set column range) |
| `gapfill_knn_k` | `outlier_gapfill_knn_k` | `3` | `gap_fill` isolation-filter neighbour count |
| `gapfill_shuffle_factor` | `outlier_gapfill_shuffle_factor` | `4` | `gap_fill` shuffled-block multiplier (matches DesignMatrix's real factor) |
| `force_resim` | `outlier_force_resim` | `False` | Skip the simulation-result cache entirely and always resimulate every pass |

## 4. Where things land

- `Opt/Outlier_Resample/case-<case>/pass-<n>/` — isolated simulation work dirs
  (never clobbers the main run's `case-*` dirs).
- `Opt/Data/Outlier_Simulations/`:
  - `sim_data_case-<case>_pass-<n>.lst`, `failed_sim_case-<case>_pass-<n>.lst`
  - `sim_data_case-<case>_pass-<n>.meta.json` — fingerprint manifest
    (`n_rows` + a SHA1 of the mini-DM's rows) written right after a real
    simulation completes. `_simulate_new_samples` only reuses a cached
    `.lst` if this manifest exists AND its fingerprint matches the CURRENT
    pass's mini-DM — a missing or mismatched manifest (e.g. a stale `.lst`
    left over from an earlier/different attempt, or one predating this
    check) always triggers a real resimulation instead of silently reusing
    old response values. `outlier_cfg["force_resim"]` (`outlier_force_resim`)
    skips the cache entirely when you want to be certain.
  - `Outlier_DM_case-<case>_pass-<n>.csv` — the full mini-DM for that pass
  - `Outlier_DM_case-<case>_pass-<n>_<label>type.csv` — split by `row_class`
    (`OUT`/`LOC`/`GLB` for hybrid; `A`/`B`/`C`/`MIX` for `local`;
    `OUT`/`UNS`/`SHF` for `dm_global`; `OUT`/`A`/`B`/`C`/`SHF` for `gap_fill`
    on `'A1+B1+C1'`, `OUT`/`GAP` on `'A-facto'`)
- `Prediction_errors.csv` — each pass logged as `Case-<case>-retrained-p<n>`.

## 5. Verification

No external simulator/`StatisticalAnalysis`/`data_management` stack is
importable in this working copy, so the full pipeline can't be executed here.
`resample_hybrid` (plus a regression check of `resample`/`resample_global`) was
verified with synthetic unit tests covering: the hard cap holds for any
`N_new` (including deliberately oversized requests); the `local_fraction`
split; `LOC` rows land closer to the outlier than `GLB` rows on average; `GLB`
rows exactly match cached pool entries (proving recombination, not
synthesis) and span most of the pool's coordinate range (proving spread, not
clustering); the A-facto/no-pool box-fallback path behaves the same way;
Partial-PRS `prs_rows`/`full_rows` round-trip consistency; and that `local` /
`dm_global` are byte-for-byte unchanged (including reproducing the exact
~4x row-count blowup formula for `dm_global`, confirming the diagnosis above
is accurate, not speculative).

Whether the *real* refit error actually drops with `hybrid` can only be
confirmed by re-running the actual pipeline (this requires the external
dependencies not present in this sandbox).

`gap_fill` is verified the same way, with one extra gap: it calls
`DesignMatrix.py`'s real curve generators, and `DesignMatrix.py` itself
imports `qmc_sampling` (confirmed absent from this working copy) and
`/shuffle.so` at module load time — so even `DesignMatrix` can't be imported
here, let alone exercised. Unit tests stub the generator calls (isolation
filter math, unshuffled/shuffled budget arithmetic, `A`/`B`/`C`/`SHF`/`GAP`
row-class labeling, Partial-PRS round-trip, the reference set actually
growing pass-over-pass) rather than calling the real `DesignMatrix`. The
fingerprint cache-hardening fix (Part 3) is covered directly, since it's pure
`run_script.py` logic with no external deps. Both `gap_fill`'s real
refit-error outcome and the cache fix's real-world behavior need a pipeline
run by the user to confirm.

## Addendum (2026-07-30): two more `retrain_around_outlier` accept/reject bugs

Found while pushing `MRE_criteria` from 2% to 1.2% for real (see
`report/main.tex`, Section "Pushing to a stricter criterion" for the full
narrative and evidence).

1. **Accept/reject compared validation only, not train.** Under a tight
   enough internal target, a pass can genuinely improve validation while
   letting training error drift upward pass over pass — observed on Case 3:
   5 accepted passes walked validation from 2.06% to 1.17% while training
   climbed to 2.04%, silently regressing above an already-achieved 2%
   criterion. Fixed: the accept test now compares `max(train, val)` before
   vs. after each trial (identical to the criterion `fit_prs_with_search`
   already uses for its own hyperparameter selection), so a pass is only
   kept if the *worse* of the two does not get worse.
2. **No guard against chasing a handful of noisy validation outliers.** A
   pass that legitimately improves `max(train, val)` can still make the
   *untouched hold-out* worse, because train/val (by design, to avoid the
   leakage this project already fixed once) can never see hold-out — this
   showed up concretely on Case 1: a single accepted pass chasing 3
   borderline points out of ~1,135 validation rows improved validation
   1.362%→0.966% but made hold-out 1.847%→2.407%, matching the same
   small-validation-set noise signature already diagnosed for
   hyperparameter *search* (this file's history) but not previously
   guarded for retraining *accept/reject*. Fixed with a minimum-outlier-
   count guard (`outlier_cfg["min_outlier_count"]`/`["min_outlier_frac"]`,
   default 5 / 0.5% of the validation set) applied **only on a case's first
   pass** — skip retraining entirely if fewer than that many points exceed
   target. Applying the same guard to *every* pass was tried first and is
   wrong: it also blocked Case 3's genuinely-beneficial second pass (only 3
   outliers, but continuing an already-validated improving trend), which
   would have left Case 3 regressed to 2.148% train. A case already
   mid-trend is trusted; a fresh case with almost no outliers is not.

Both fixes are in `retrain_around_outlier` in `run_script.py` and are
general decision-rule changes, not per-case constants.

## Addendum 2 (2026-07-30): 10-fold CV robustness audit before production

Full writeup: `report_10fold_validation/main.tex`. Short version: a single
70/15/15 split (everything above) can badly hide true out-of-sample risk.
Cross-validating the last production hyperparameters with genuine 10-fold
CV found only 3/8 cases kept every fold <=2% MRE — Case 4's single-split
hold-out was 1.45%, but the SAME hyperparameters had a 10-fold worst-fold
of 20.3%. Root cause: every worst-fold failure traced to a single test
row with a band-validity violation ratio 7x-30x the physical band (the
same class-B m=3 sampler bug as Addendum 1's diagnosis) — ~30% of every
case's design matrix is band-invalid. Fix, now default in `src/MUQ-SAC`:

1. `IdentifiablePRS.band_valid_mask` (new) — MRE is evaluated only on
   band-valid points (training is never filtered by this).
2. `run_script.py`'s Stage-1+2 search defaults to `fit_prs_with_cv_search`
   (10-fold, `prs_fit_cfg["search_method"]="cv_kfold"`), replacing the
   single-split `fit_prs_with_search` — override to `"single_split"` via
   `target.opt`'s `prs_search_method` for the old behaviour.
3. `fit_prs_with_cv_search` itself had the SAME screen-both-sides leakage
   bug as Addendum 1 warns about, but in production code, not a test
   script — fixed to screen only each fold's training side.
4. `ridge_grid` widened to include up to 0.5 (Cases 4/7 needed 0.3-0.5;
   ridge alone could not fix the band-invalid-row problem, only the
   band-valid-only scoring could — regularization and evaluation fairness
   are different problems that looked similar from the symptom alone).

Result: all 8 cases keep every one of 10 independently-folded partitions
<=2% MRE (worst observed 1.85%), using zero new simulations.
