# CLAUDE.md — Combustion Kinetic Model Optimization Context

## Project Overview

This project is part of a PhD thesis on combustion kinetic model optimization
(`Development of a Fast, Robust and Kinetics-based Optimization Framework for
Combustion Kinetic Models`, IIT Madras, supervised by Dr. Krithika Narayanaswamy).

The primary codebase is `OptimizationTool.py`, part of the **MUQ-SAC/MSO
(Mechanism Optimization Suite)** framework. The current focus is the
`run_optimization_with_selected_PRS` method, specifically the **A-factor
(A-facto) branch** that optimizes the pre-exponential Arrhenius parameter.

---

## The Problem Being Solved

### Context

Multi-dataset combustion kinetic optimization using **GA (PyGAD) + L-BFGS-B**
over a polynomial response surface (PRS) in zeta-space. Target observables
include Ignition Delay Time (`Tig`/`RCM`), Laminar Burning Velocity (`Fls`),
and Flame Speed (`Flw`) across multiple conditions (φ, P, T).

### Core Problem: Dataset Dominance

One or more datasets have per-dataset squared error `S_d` orders of magnitude
larger than others. For example:

| Dataset          | Nominal σ (error) |
|------------------|-------------------|
| φ=1, P=1 atm     | ~4.9              |
| φ=1, P=4 atm     | ~5.4              |
| φ=0.25, P=1 atm  | ~16.4             |
| **φ=1, P=30 atm**| **~948.5** ← dominant |

Any sum-based objective `obj = Σ_d S_d` is 99% governed by the dominant
dataset's gradient. The optimizer reduces the large error but **worsens all
smaller datasets**. This was confirmed across multiple objective functions
(WMRE, logRMSE, logDatasetWMRE) in the uploaded PDF plots:

- `combined_all_stages_MB_new_HTT.pdf` — WMRE / HTT results
- `combined_nominal_vs_s1_logDatasetWMRE.pdf` — logDatasetWMRE results
- `combined_all_stages_LOGrsme.pdf` — logRMSE results

In all cases, Stage-1 optimization improved the dominant dataset but worsened
all others. Stage-2 reverted entirely to nominal (σ = nominal for all datasets).

---

## Existing Objective Functions (Already Implemented)

All live in `_compute_objective_A_facto(x, obj_type, dataset_weights)`:

| `obj_type`        | Description                                              | Status  |
|-------------------|----------------------------------------------------------|---------|
| `"WMRE"`          | Weighted Mean Relative Error squared (default)           | Exists  |
| `"logRMSE"`       | Log-scale RMSE                                           | Exists  |
| `"logDatasetWMRE"`| log(1 + S_d) compression at dataset level                | Exists  |
| `"unbiasedWMRE"`  | Frozen per-dataset fairness weights calibrated at ζ=0    | Exists  |
| `"dynamicWMRE"`   | PER-inspired EMA-updated adaptive weights                | Exists  |

None of these fully solved the dominance problem in practice.

---

## New Methods Added in This Session

Two new objective functions were implemented in `OptimizationTool.py`
(+309 lines). The modified file is `OptimizationTool.py` (output).

### Method 1: `softChebysheWMRE` — Smooth Augmented Chebyshev Minimax

**Core idea:** Replace the sum `Σ_d Ŝ_d` with a smooth approximation of
`max_d Ŝ_d` (Chebyshev scalarization from MOEA/D literature):

```
obj = (1/β) · log(Σ_d exp(β · Ŝ_d))  +  ρ · Σ_d Ŝ_d

where  Ŝ_d = W_k_d · S_d(x) / (S_d(0) + ε)
```

- `β` (temperature): as β→∞ this approaches `max_d Ŝ_d` (hard minimax).
- `ρ` (augmentation): small term (≈0.01) for Pareto efficiency — breaks
  degeneracy on the max-iso surface.
- **β-annealing**: β starts at `β_final/10` (smooth landscape for GA
  exploration) and grows exponentially to `β_final` over the GA run.
  Updated every generation in `_on_gen`.

**Why it works:** The softmax weight `exp(β·Ŝ_d)/Σexp(...)` concentrates near 1
for the WORST dataset and near 0 for all others. The optimizer pushes hardest
on the currently-worst dataset, then cycles to the next worst as each improves.
At convergence, **all Ŝ_d are equal** — the most equitable Pareto-optimal result.

**Input tags:**
```yaml
Stats:
  Objective: "softChebysheWMRE"
  softmax_beta:    20.0      # β_final. Try range 10–50.
  softmax_anneal:  true      # Exponential β ramp (recommended)
  softmax_rho:     0.01      # Augmentation weight (0.001–0.05)
  num_generations: 3200      # Must match GA num_generations for annealing
```

---

### Method 2: `penWMRE` — Penalty-Augmented Normalised WMRE with Error Floor

**Core idea:** Normalised sum (like unbiasedWMRE) PLUS a quadratic barrier
penalty for any dataset that worsens beyond threshold α:

```
obj = Σ_d W_k_d · Ŝ_d  +  K · Σ_d max(Ŝ_d − α, 0)²

where  Ŝ_d = S_d(x) / (S_d(0) + ε)
```

- At ζ=0: base = Σ W_k_d ≈ n_datasets, penalty = 0 (since all Ŝ_d = 1 ≤ α=1.0).
- Any move that worsens a dataset above α immediately activates a penalty
  that grows **quadratically** with the violation.
- α=1.0 (default): hard floor at nominal — no dataset is allowed to get worse.
- α>1.0: permissive floor for cases where some degradation is unavoidable.
- K: penalty weight. Start at 10; increase to 50–100 if floor is violated.

**Why it works:** This is the **only method that creates a hard barrier** against
dataset degradation. The path that improves D while worsening A will have
`penalty = K·(Ŝ_A − 1)²` added to its objective, making it less attractive than
a path that improves everything modestly.

**Input tags:**
```yaml
Stats:
  Objective: "penWMRE"
  pen_alpha:  1.0            # Floor threshold (1.0 = not worse than nominal)
  pen_K:      10.0           # Penalty weight
```

---

## New Supporting Method

### `_calibrate_s0_dataset_sums(n_params)`

Shared calibration backbone for both new methods. Evaluates every dataset at
ζ=0 and stores raw `S_d(0)` on `self._s0_dataset_sums`. Called once before
the GA, writes `s0_dataset_sums.log` for inspection.

**Key difference from `_calibrate_unbiased_dataset_weights`:**
- Stores raw `S_d(0)`, not inverted weights `1/S_d(0)`.
- No capping/clipping needed — division by S_d(0) cannot explode
  the way multiplication by `1/S_d(0)` can for near-zero S_d(0).

---

## Code Architecture Summary

```
OptimizationTool (class)
│
├── _compute_objective_A_facto(x, obj_type, dataset_weights)
│     ├── WMRE / logRMSE / logDatasetWMRE / unbiasedWMRE / dynamicWMRE  [existing]
│     ├── softChebysheWMRE    ← NEW (log-sum-exp smooth minimax)
│     └── penWMRE             ← NEW (normalised sum + quadratic penalty)
│
├── _calibrate_s0_dataset_sums(n_params)   ← NEW (S_d(0) calibration)
├── _calibrate_unbiased_dataset_weights()  [existing]
├── _init_dynamic_weights()                [existing]
├── _update_dynamic_weights()              [existing]
│
├── FITNESS_PRS_BASED_OPT_A_FACTO()
│     └── fitness_func_A_facto_PRS(ga_instance, x, solution_idx)
│           └── calls _compute_objective_A_facto(...)
│
└── run_optimization_with_selected_PRS(Unsrt_data, ResponseSurfaces, Input_data, ...)
      └── A-facto branch:
            ├── [existing] obj_type / dataset_weights extraction
            ├── [existing] unbiasedWMRE calibration
            ├── [existing] dynamicWMRE bootstrap
            ├── [NEW]      softChebysheWMRE/penWMRE → _calibrate_s0_dataset_sums()
            │               + beta-annealing state init
            ├── [existing] multi-start loop (n_ms runs, each GA → L-BFGS-B)
            │     └── _on_gen(ga):
            │           ├── [NEW]      softChebysheWMRE β-annealing update
            │           ├── [existing] dynamicWMRE EMA weight update
            │           └── [existing] diversity injection
            └── returns (optimal_parameters, optimal_parameters_zeta, cov)
```

---

## Key Variable Names & State

| Attribute                        | Set by                            | Used by              |
|----------------------------------|-----------------------------------|----------------------|
| `self._s0_dataset_sums`          | `_calibrate_s0_dataset_sums`      | softChebysheWMRE, penWMRE |
| `self._softmax_beta_current`     | init block + `_on_gen` annealing  | softChebysheWMRE     |
| `self._unbiased_dataset_weights` | `_calibrate_unbiased_dataset_weights` | unbiasedWMRE     |
| `self._dynamic_dataset_weights`  | `_init_dynamic_weights` / `_update_dynamic_weights` | dynamicWMRE |
| `self._ds_sums`                  | loop in `_compute_objective_A_facto` | logDatasetWMRE, softChebysheWMRE, penWMRE |

---

## Recommendation for Next Run

1. **Try `penWMRE` first** (pen_alpha=1.0, pen_K=10).
   - Check `s0_dataset_sums.log` to see S_d(0) per dataset.
   - If the dominant dataset (P=30 atm) cannot reach Ŝ_d ≤ 1 at all
     (model limitation), raise `pen_alpha` to 1.5 or 2.0.
   - If the floor is still being violated, raise `pen_K` to 50 or 100.

2. **If penWMRE stalls, switch to `softChebysheWMRE`** (beta=20, anneal=true).
   - This will find the most equitable balance the model *can* achieve,
     even if the dominant dataset cannot be brought below its nominal error.

3. Both methods still honour `Weights: {Tig: ..., Fls: ...}` for type-level
   scaling. `dataset_weights` is applied via `W_k_d` inside both normalisation
   and penalty terms.

---

## Log Files Produced

| File                          | Content                                       |
|-------------------------------|-----------------------------------------------|
| `s0_dataset_sums.log`         | S_d(0) per dataset at ζ=0 (new methods)       |
| `unbiased_dataset_weights.txt`| w_ds calibration table (unbiasedWMRE)         |
| `dynamic_dataset_weights.log` | Weight evolution per generation (dynamicWMRE) |
| `Objective.txt`               | obj value at every function evaluation        |
| `Dataset_based_obj`           | Per-case relative errors {uniqueID: err}      |
| `samplefile.txt`              | gen, count, obj, fitness per evaluation       |
| `guess_values.txt`            | x (zeta) at every evaluation                  |
| `response_values.txt`         | (y_exp, y_calc) pairs per evaluation          |

---

## Session Notes

- The Stage-2 complete reversion to nominal (σ_Stage2 = σ_nominal exactly) seen
  in all three PDFs is a **separate bug** from the objective dominance problem —
  likely Stage-2 is not receiving Stage-1's zeta as `initial_solution`, or the
  Stage-2 PRS is evaluated with zeta=0 by default. This was noted but not fixed
  in this session; the session focused only on the objective function.
- All changes are confined to `OptimizationTool.py`. No other framework files
  were modified.
- The modified file passes `ast.parse()` (syntax verified).
