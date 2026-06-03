# Active Parameter Framework

Screen sensitivity-analysis (SA) results to identify active reactions, then
generate MUQ uncertainty XML files, HTC/LTC sensitivity plots, publication-ready
LaTeX tables, and reduced mechanisms — all from a single driver.

---

## Table of Contents

- [Installation](#installation)
- [Project Layout](#project-layout)
- [Input Files](#input-files)
- [Quick Start](#quick-start)
- [Command-Line Reference](#command-line-reference)
- [Usage by Task](#usage-by-task)
- [Output Structure](#output-structure)
- [Notes & Conventions](#notes--conventions)

---

## Installation

Requires Python 3.8+.

```bash
git clone <your-repo-url> ActiveParameterFramework
cd ActiveParameterFramework
pip install numpy pandas matplotlib pyyaml tqdm
```

`tqdm` is optional (progress bars); the code falls back gracefully if it is
not installed.

---

## Project Layout

```
ActiveParameterFramework/
├── ActiveParameterScreen.py        # main driver (run this)
├── README.md
└── modules/
    ├── __init__.py
    ├── interfaces.py               # canonical schemas + shared helpers
    ├── sensitivity_parser.py       # reads SA files
    ├── mechanism_utils.py          # reads Cantera YAML mechanism
    ├── screening.py                # threshold + carbon screening
    ├── xml_generator.py            # MUQ uncertainty XML
    ├── plotting.py                 # HTC/LTC sensitivity plots
    ├── latex_tables.py             # LaTeX reaction tables
    └── filtered_mechanism.py       # reduced mechanism generation
```

---

## Input Files

Three inputs. Standard filenames used throughout this README:

### 1. Sensitivity files — directory `sa_files/`

Tab-separated, one file per (case, temperature). Temperature is parsed from
the filename, so it **must** contain a `T` token.

Accepted filename patterns (case-insensitive):

```
sa_files/case_1_T_700.txt
sa_files/case1_T700.txt
sa_files/ignition_case_3_T_900.txt
```

File contents — first line is a header (skipped), then
`Sensitivity <TAB> RxnID <TAB> Reaction`:

```
Sensitivity	RxnID	Reaction
0.201	1	H + O2 <=> O + OH
0.083	2	C7H16 + OH <=> C7H15-1 + H2O
0.152	3	NC4H9 <=> C4H8-1 + H
```

### 2. Mechanism — `mechanism.yaml`

A standard Cantera YAML mechanism. Reaction IDs are **1-based** and map to the
YAML reaction list as `Reaction_ID = yaml_index + 1`.

### 3. Classification file — `classification.csv`

**User-supplied.** This is how you assign each reaction to a temperature regime
(HTC/LTC) and an uncertainty factor `f`. It is required only for XML, plots, and
LaTeX generation — not for plain screening.

Minimum columns:

```csv
Reaction_ID,Reaction,Group,f
1,H + O2 <=> O + OH,HTC,2
2,C7H16 + OH <=> C7H15-1 + H2O,LTC,4
3,NC4H9 <=> C4H8-1 + H,LTC,4
```

Optional columns:

| Column          | Purpose                                            | Default              |
|-----------------|----------------------------------------------------|----------------------|
| `data_type`     | Uncertainty data type                              | `constant;end_points`|
| `Reaction_Type` | `Elementary` / `Duplicate` / `PLOG` / `PLOG-Duplicate` / `ThirdBody` | `Elementary` |
| `RxnCount`      | Interconnected-row count for PLOG blocks            | `2`                  |

Accepted aliases:
- `Group` values: `HTC`/`High`/`high`/`high temperature` → **HTC**; `LTC`/`Low`/`low`/`low temperature` → **LTC**.
- `Unsrt` is accepted as an alias for the `f` column.
- A leading `R` on `Reaction_ID` (e.g. `R707`) is stripped automatically.

The uncertainty written to XML is `ln(f)` — e.g. `f=2 → 0.6931`, `f=4 → 1.3863`.

---

## Quick Start

Plain screening (no classification file needed):

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --threshold 0.05
```

Everything at once:

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --threshold 0.05 \
    --carbon 7 \
    --classification-file classification.csv \
    --fuel nHeptane \
    --generate-xml \
    --generate-plots \
    --generate-latex \
    --compare-temperatures 700 900 1200 \
    --filtered-mechanism
```

---

## Command-Line Reference

| Flag                       | Type    | Default          | Description                                                        |
|----------------------------|---------|------------------|--------------------------------------------------------------------|
| `--sens-dir`               | path    | `.`              | Directory containing SA `.txt` files.                              |
| `--mechanism`              | path    | `mechanism.yaml` | Cantera YAML mechanism.                                            |
| `--threshold`              | float   | `0.05`           | Absolute-sensitivity cutoff for screening.                        |
| `--carbon`                 | int     | `0`              | Minimum carbon number; keep reactions with a species C ≥ this. `0` disables. |
| `--classification-file`    | path    | `None`           | User classification CSV. Required for XML/plots/LaTeX.            |
| `--fuel`                   | str     | `Fuel`           | Fuel name used in LaTeX captions/labels.                          |
| `--results-dir`            | path    | `results`        | Output directory.                                                 |
| `--generate-xml`           | flag    | off              | Generate MUQ uncertainty XML.                                     |
| `--generate-plots`         | flag    | off              | Generate HTC/LTC sensitivity plots.                               |
| `--generate-latex`         | flag    | off              | Generate LaTeX reaction tables.                                   |
| `--longtable`              | flag    | off              | Use `longtable` instead of `tabular` in LaTeX output.            |
| `--filtered-mechanism`     | flag    | off              | Write a reduced mechanism from the active reaction list.         |
| `--compare-temperatures`   | floats  | `None`           | Temperatures for the multi-temperature comparison plot.          |

---

## Usage by Task

### Screening only

Produces the active-reaction CSVs and the SA/YAML mismatch log.

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --threshold 0.05 \
    --carbon 7
```

### Generate XML uncertainty files

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --classification-file classification.csv \
    --generate-xml
```

Writes `HTC_all.xml`, `HTC_factor.xml`, `LTC_all.xml`, `LTC_factor.xml`
(`all` vs `factor` differ only in `<perturbation_type>`).

### Generate sensitivity plots

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --classification-file classification.csv \
    --generate-plots \
    --compare-temperatures 700 900 1200
```

Per group you get: a low-vs-high paired plot, one plot per temperature under
`All_Temperatures/`, and (if `--compare-temperatures` is given) a grouped-bar
comparison. **Reaction order always follows the classification file** so plots
are directly comparable — sensitivities are never re-sorted.

### Generate LaTeX tables

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --classification-file classification.csv \
    --fuel nHeptane \
    --generate-latex --longtable
```

### Generate a reduced mechanism

```bash
python ActiveParameterScreen.py \
    --sens-dir sa_files \
    --mechanism mechanism.yaml \
    --filtered-mechanism
```

Writes `results/filtered_mechanism.yaml` containing only the active reactions
and the species they involve.

### Using the modules directly (Python API)

```python
from modules import sensitivity_parser as sp, mechanism_utils as mu
from modules.screening import run_screening
from modules.interfaces import load_classification_file
from modules.xml_generator import generate_xml_files

records = sp.build_detailed_records("sa_files")
mech    = mu.load_mechanism("mechanism.yaml")
lookup  = mu.build_reaction_lookup(mech)
cmap    = mu.build_species_carbon_map(mech)

result  = run_screening(records, lookup, cmap, threshold=0.05, carbon_threshold=7)

cdf = load_classification_file("classification.csv")
generate_xml_files(cdf, "results/xml")
```

---

## Output Structure

```
results/
├── active_reactions_detailed.csv      # every passing (case, T, reaction) row
├── master_active_reactions.csv        # unique reactions, max sensitivity, counts
├── active_species.csv                 # species occurrence + carbon number
├── active_reaction_summary.csv        # counts per reaction type
├── reaction_id_mismatch.log           # only if SA equation ≠ YAML equation
├── filtered_mechanism.yaml            # only with --filtered-mechanism
├── xml/
│   ├── HTC_all.xml
│   ├── HTC_factor.xml
│   ├── LTC_all.xml
│   └── LTC_factor.xml
├── plots/
│   ├── HTC/
│   │   ├── HTC_SA.pdf / .png
│   │   ├── HTC_Temperature_Comparison.pdf / .png
│   │   └── All_Temperatures/
│   │       ├── HTC_700K.pdf / .png
│   │       └── ...
│   └── LTC/  (same structure)
└── latex/
    ├── HTC_table.tex
    └── LTC_table.tex
```

---

## Notes & Conventions

- **Reaction IDs are 1-based** and validated against the YAML
  (`Reaction_ID = yaml_index + 1`). Any SA-vs-YAML equation mismatch is
  recorded in `reaction_id_mismatch.log` rather than failing silently.
- **Screening output ≠ classification input.** `master_active_reactions.csv`
  tells you *which* reactions matter; `classification.csv` is where *you* assign
  `Group` and `f`. They are deliberately separate files.
- **Plot ordering** follows the classification file exactly (no re-sorting), so
  the same reaction sits at the same vertical position across all plots.
- **XML uncertainty** is `ln(f)`. Confirm the `RxnCount` semantics and operator
  entity-encoding (`<=>` → `&#60;&#61;&#62;`) against a known-good legacy XML
  before trusting downstream — both are noted inline in `xml_generator.py`.
- Plotting uses the non-interactive `Agg` backend, so it runs headless (no
  display required).
