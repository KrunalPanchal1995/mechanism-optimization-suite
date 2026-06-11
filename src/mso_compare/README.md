# mso_compare

A Python framework for running, comparing, and visualising Multi-Stage Optimisation (MSO) combustion kinetic mechanism results against experimental ignition delay time (IDT) data.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation & Requirements](#installation--requirements)
4. [Quick Start — Full MSO Pipeline](#quick-start--full-mso-pipeline)
5. [Quick Start — Nominal-Only Run](#quick-start--nominal-only-run)
6. [Configuration File Reference](#configuration-file-reference)
7. [Individual Scripts](#individual-scripts)
8. [Output Structure](#output-structure)
9. [Error Metric Definition](#error-metric-definition)
10. [Dataset ID Format](#dataset-id-format)
11. [Common Workflows](#common-workflows)
12. [Bug Fixes Over Original Scripts](#bug-fixes-over-original-scripts)
13. [Troubleshooting](#troubleshooting)

---

## Overview

`mso_compare` automates the four-phase workflow that follows a two-stage combustion mechanism optimisation:

```
Phase 1 — Simulate    Run up to 6 nominal simulations
                       (Nominal / Stage-1 / Stage-2) × (HTC / LTC)

Phase 2 — Merge       Align per-run CSV outputs; organise by φ and P
                       → all/ nominal_s1/ nominal_s2/

Phase 3 — Errors      Compute χ² error metrics for each condition

Phase 4 — Plot        Per-φ PDFs + combined multi-panel figures
```

All four phases are configurable and individually skippable, so the framework can be used in full-pipeline mode or as individual tools.

---

## Project Structure

```
mso_compare/
├── src/
│   └── mso_compare/          ← Python package
│       ├── __init__.py
│       ├── config.py          — YAML config loader + dataclasses
│       ├── utils.py           — Logging, path helpers, dataset-ID parser
│       ├── sim_runner.py      — Nominal simulation runner (Phase 1)
│       ├── merge.py           — CSV merge logic (Phase 2)
│       ├── error_metrics.py   — χ² error computation (Phase 3)
│       ├── plotter.py         — IDT comparison plots (Phase 4)
│       └── pipeline.py        — End-to-end orchestrator
│
├── scripts/
│   ├── run_mso_pipeline.py    ← Full pipeline (recommended entry point)
│   ├── run_nominal_sim.py     ← Run one simulation only
│   ├── run_merge.py           ← Merge only
│   └── run_plots.py           ← Plots only (+ two-threshold mode)
│
├── configs/
│   ├── template_mso_pipeline.yaml   ← Fully annotated config template
│   └── template_nominal_only.yaml   ← Minimal nominal-only template
│
└── README.md
```

---

## Installation & Requirements

### Python version

Python 3.8 or higher is required.

### Dependencies

```bash
pip install numpy pandas matplotlib pyyaml
```

The framework also requires your existing **MUQ-SAC** modules (not included):
- `simulation_manager2_0`
- `combustion_target_class`
- `data_management`
- `MechManipulator2_0`

Set `muqsac_src_dir` in your config to point to the directory containing these modules.

### No installation needed

The package is importable directly from the repository. All scripts add `src/` to `sys.path` automatically.

---

## Quick Start — Full MSO Pipeline

### Step 1: Copy and fill in the config template

```bash
cp configs/template_mso_pipeline.yaml my_project/mso_config.yaml
```

Open `mso_config.yaml` and fill in every `/PATH/TO/...` placeholder.
At minimum you need:

| Key | Description |
|-----|-------------|
| `base_opt_file` | Path to your `.opt` YAML template |
| `muqsac_src_dir` | Path to `src/MUQ-SAC` |
| `simulations.*mechanism` | One path per run |
| `simulations.*targets` | One targets file per run |
| `simulations.*targets_count` | Integer |
| `simulations.*thermo_file` | Thermodynamics data |
| `simulations.*trans_file` | Transport data |
| `simulations.*output_dir` | Where to write results |

### Step 2: Verify the plan (dry run)

```bash
cd my_project
python ../scripts/run_mso_pipeline.py --config mso_config.yaml --dry-run
```

This prints every path and setting without executing anything.

### Step 3: Run the full pipeline

```bash
python ../scripts/run_mso_pipeline.py --config mso_config.yaml
```

Output is written to the directories specified in the config (defaults: `./SIM_RUNS/`, `./MERGE_OUTPUT/`, `./ERROR_METRICS/`, `./PLOTS/`).

---

## Quick Start — Nominal-Only Run

If you only want to run the nominal mechanism and compare against data:

```bash
cp configs/template_nominal_only.yaml my_project/nominal_config.yaml
# fill in placeholders ...

python scripts/run_mso_pipeline.py --config my_project/nominal_config.yaml
```

Or interactively (no config file needed):

```bash
python scripts/run_nominal_sim.py --interactive
```

---

## Configuration File Reference

The config file is a YAML document that lives in your working directory. See `configs/template_mso_pipeline.yaml` for a fully annotated example.

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `project.name` | string | Project label used in logs and titles |
| `project.working_dir` | string | Absolute path to the project root |
| `project.fuel` | string | Fuel label (informational) |
| `base_opt_file` | string | Path to the `.opt` YAML template |
| `muqsac_src_dir` | string | Path to the `src/MUQ-SAC` directory |
| `simulations` | dict | One entry per simulation run (see below) |
| `merge` | dict | Output directories for Phase 2 |
| `error_metrics` | dict | Output directories + σ fraction for Phase 3 |
| `plots` | dict | Output directories + style options for Phase 4 |

### Per-simulation keys (under `simulations.<label>`)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `mechanism` | string | ✓ | Path to mechanism file |
| `targets` | string | ✓ | Path to `.target` file |
| `targets_count` | int | ✓ | Number of target lines to read |
| `thermo_file` | string | ✓ | Path to thermodynamics data |
| `trans_file` | string | ✓ | Path to transport data |
| `uncertainty_data` | string | | Path to uncertainty XML (dummy OK for nominal) |
| `output_dir` | string | ✓ | Where to write NOMINAL/, Plot/, and log file |
| `extra_overrides` | dict | | Additional `.opt` overrides in `{Section: {key: val}}` form |

### Plot mode options

| Mode | Description |
|------|-------------|
| `auto` | Infer from CSV columns (recommended) |
| `nominal_only` | Experimental data + Nominal |
| `nominal_vs_s1` | Experimental data + Nominal + Stage-1 |
| `nominal_vs_s2` | Experimental data + Nominal + Stage-2 |
| `all_stages` | Experimental data + Nominal + Stage-1 + Stage-2 |

---

## Individual Scripts

All scripts accept `--help` for a full argument reference.

### `run_mso_pipeline.py` — Full pipeline

```bash
# Full run
python scripts/run_mso_pipeline.py --config mso_config.yaml

# Dry run
python scripts/run_mso_pipeline.py --config mso_config.yaml --dry-run

# Skip simulations (re-run merge onwards)
python scripts/run_mso_pipeline.py --config mso_config.yaml --skip-simulations

# Re-run plots only
python scripts/run_mso_pipeline.py --config mso_config.yaml \
    --skip-simulations --skip-merge --skip-error-metrics

# HTC regime only
python scripts/run_mso_pipeline.py --config mso_config.yaml --regime htc
```

### `run_nominal_sim.py` — Single simulation

```bash
# From config file
python scripts/run_nominal_sim.py --config mso_config.yaml --run nominal_htc

# Dry run
python scripts/run_nominal_sim.py --config mso_config.yaml --run nominal_htc --dry-run

# List all run labels in your config
python scripts/run_nominal_sim.py --config mso_config.yaml --list-runs

# Interactive mode (no config file needed)
python scripts/run_nominal_sim.py --interactive
```

### `run_merge.py` — Merge only

```bash
# Both regimes
python scripts/run_merge.py --config mso_config.yaml

# LTC only
python scripts/run_merge.py --config mso_config.yaml --regime ltc

# Dry run (show paths without writing)
python scripts/run_merge.py --config mso_config.yaml --dry-run
```

### `run_plots.py` — Plots only

```bash
# Standard plots (all regimes, auto mode)
python scripts/run_plots.py --config mso_config.yaml

# HTC only, force all_stages mode
python scripts/run_plots.py --config mso_config.yaml --regime htc --mode all_stages

# Combined figure only (skip per-φ PDFs)
python scripts/run_plots.py --config mso_config.yaml --no-per-phi

# Per-φ PDFs only (skip combined figure)
python scripts/run_plots.py --config mso_config.yaml --no-combined

# Two-threshold comparison
python scripts/run_plots.py \
    --two-threshold \
    --data-dir-set1 MERGE_OUTPUT/HTC/all \
    --data-dir-set2 MERGE_OUTPUT_delta005/HTC/all \
    --set1-label "delta=0.01" \
    --set2-label "delta=0.05" \
    --plot-dir PLOTS/HTC/threshold_comparison
```

---

## Output Structure

After a full pipeline run the working directory contains:

```
SIM_RUNS/
├── nominal_htc/
│   ├── NOMINAL/                     ← MUQ-SAC simulation case directories
│   │   ├── case-0/  case-1/  ...
│   │   ├── Data/Simulations/        ← Cached sim_data_case-*.lst files
│   │   ├── progress                 ← Written by SM() after completion
│   │   └── locations                ← Execution locations list
│   ├── Plot/
│   │   └── Dataset/
│   │       ├── Tig/   ← per-dataset CSVs: DS_ID, T, Obs(us), Nominal
│   │       ├── RCM/
│   │       ├── JSR/
│   │       └── Fls/
│   └── mso_nominal_htc.log
├── nominal_ltc/
├── stage1_htc/
│   └── ...
└── ...

MERGE_OUTPUT/
├── HTC/
│   ├── all/                         ← Nominal + Stage-1 + Stage-2
│   │   ├── phi_0_5/
│   │   │   ├── Phi_0_5_P_10.csv    ← DS_ID, T, Obs(us), Nominal, Stage-1, Stage-2
│   │   │   └── Phi_0_5_P_20.csv
│   │   ├── phi_1_0/
│   │   └── phi_2_0/
│   ├── nominal_s1/                  ← Nominal + Stage-1 only
│   │   └── phi_*/...
│   └── nominal_s2/                  ← Nominal + Stage-2 only
│       └── phi_*/...
└── LTC/
    └── ...  (same structure)

ERROR_METRICS/
├── HTC/
│   ├── all/phi_0_5/Phi_0_5_P_10.csv   ← Stage, Error_Function
│   └── ...
└── LTC/

PLOTS/
├── HTC/
│   ├── all/
│   │   ├── phi_0_5/
│   │   │   └── IDT_phi_0_5_all_stages.pdf
│   │   ├── phi_1_0/
│   │   ├── phi_2_0/
│   │   └── combined_all_stages.pdf     ← multi-panel figure
│   ├── nominal_s1/
│   └── nominal_s2/
└── LTC/
```

---

## Error Metric Definition

The framework uses a **weighted sum of squares** (chi-squared) error:

```
ε = Σᵢ [ (sim_i − obs_i) / σᵢ ]²

where  σᵢ = uncertainty_fraction × obs_i
```

Default uncertainty fraction: **10 %** (`uncertainty_fraction: 0.10` in config).

Each error CSV file contains one row per simulation stage:

```csv
Stage,Error_Function
Nominal,97.5621
Stage-1,82.9634
Stage-2,77.6412
```

A lower `Error_Function` value indicates better agreement with experimental data.

---

## Dataset ID Format

The framework recognises dataset IDs of the form:

```
<prefix>_phi<phi_value>_p<pressure_value>
```

Examples:

| Dataset ID | φ | P |
|------------|---|---|
| `ing_BUTADIENE_HTC_phi0_5_p10` | 0.5 | 10 |
| `ing_BUTADIENE_HTC_phi1_0_p20` | 1.0 | 20 |
| `ing_BUTADIENE_HTC_phi2_p40` | 2.0 | 40 |

The prefix (everything before `phi`) is ignored during parsing. Underscores within the φ value are interpreted as decimal separators: `phi0_5` → φ = 0.5.

These dataset IDs are converted to standardised filenames:

| Dataset ID | Standardised filename |
|------------|----------------------|
| `ing_BUTADIENE_HTC_phi0_5_p10.csv` | `Phi_0_5_P_10.csv` |
| `ing_BUTADIENE_HTC_phi1_0_p20.csv` | `Phi_1_0_P_20.csv` |

---

## Common Workflows

### Re-run plots after a style change

```bash
python scripts/run_plots.py --config mso_config.yaml
```

### Run only the HTC comparison after Stage-2 optimisation is done

```bash
# Add stage2_htc to your config, then:
python scripts/run_mso_pipeline.py --config mso_config.yaml \
    --skip-simulations   \   # nominal already done
    --regime htc
```

Actually, if only Stage-2 HTC is new:

```bash
# Run just the Stage-2 HTC simulation
python scripts/run_nominal_sim.py --config mso_config.yaml --run stage2_htc

# Then re-run merge + errors + plots for HTC
python scripts/run_mso_pipeline.py --config mso_config.yaml \
    --skip-simulations --regime htc
```

### Compare two optimisation runs with different thresholds

```bash
python scripts/run_plots.py \
    --two-threshold \
    --data-dir-set1 MERGE_OUTPUT/HTC/all \
    --data-dir-set2 PATH_TO_RUN2/MERGE_OUTPUT/HTC/all \
    --set1-label "threshold=0.01" \
    --set2-label "threshold=0.05" \
    --plot-dir PLOTS/HTC/comparison
```

### Use the framework as a Python module

```python
from mso_compare import MSO_Pipeline, merge_simulation_results, plot_comparison

# Full pipeline
pipe = MSO_Pipeline.from_yaml("mso_config.yaml")
pipe.run(skip_simulations=True)   # use existing sim results

# Or call individual functions
from pathlib import Path
merge_simulation_results(
    nominal_folder = Path("SIM_RUNS/nominal_htc/Plot/Dataset/Tig"),
    stage1_folder  = Path("SIM_RUNS/stage1_htc/Plot/Dataset/Tig"),
    stage2_folder  = Path("SIM_RUNS/stage2_htc/Plot/Dataset/Tig"),
    output_root    = Path("MERGE_OUTPUT/HTC"),
)
plot_comparison(
    data_dir  = Path("MERGE_OUTPUT/HTC/all"),
    plot_dir  = Path("PLOTS/HTC/all"),
    mode      = "all_stages",
)
```

---

## Bug Fixes Over Original Scripts

| Original script | Bug | Fix in mso_compare |
|----------------|-----|--------------------|
| `run_nominal_sim.py` | `progress` file written to parent dir | `os.chdir(nominal_dir)` called **before** `SM()` |
| `run_nominal_sim.py` | `Plot/Dataset/Tig/` not created → crash | All subdirs created with `mkdir(parents=True, exist_ok=True)` upfront |
| `run_nominal_sim.py` | `import yaml` silently overwrote `ruamel_yaml` | Only PyYAML used; no dual import |
| `run_nominal_sim.py` | `os.chdir()` in loops not wrapped in try/finally | All `chdir` calls have `try/finally` to restore original CWD |
| `rms_HTC_ALL.py` | All PDFs except the last overwritten (phi variable from last loop) | Output filename derived from the φ sub-folder name, not a loop variable |
| `rms_4_LTT.py` | `KeyError` for φ values not in hardcoded marker dict | Per-folder marker pool with `_DEFAULT_MARKERS` fallback |
| `merge_idt_results.py` | Fragile `split("_phi")` / `split("_p")` filename parser | Robust regex: `phi(\d+(?:_\d+)?)_p(\d+)` |

---

## Troubleshooting

**`ImportError: No module named 'simulation_manager2_0'`**
→ Set `muqsac_src_dir` in your config to the correct path.

**`FileNotFoundError: No CSV files found in: ...`**
→ The nominal simulation has not been run yet, or `output_dir` in the config does not match the actual run directory.

**`ValueError: Cannot parse φ/P from '...'`**
→ Check that your dataset IDs follow the format `phi<value>_p<value>`.

**All PDFs look the same / are overwritten**
→ This was a bug in the original scripts, fixed in `plotter.py`.

**`KeyError` in plotter for an unusual φ value**
→ The plotter now uses `_DEFAULT_MARKERS` as a fallback; no action required.

**Want to add more comparison stages in future?**
→ Define additional run labels in the `simulations` section of the config. The merge and plot modules automatically detect which stage columns are present.
