# MSO Report Generator

Generates a comprehensive LaTeX/PDF report for p-PRS Multi-Stage Optimization
of combustion kinetic mechanisms.

---

## Installation

```bash
pip install -r requirements.txt
```

pdflatex is required for PDF compilation (install TeX Live or MiKTeX).

---

## Quick Start

```bash
python run_report.py
```

The script will ask you step-by-step for all inputs.

---

## Required Input Files

### 1. `metadata.yaml`
Fill in the provided template. Key fields:
- `fuel_name`, `fuel_formula`, `mechanism_name`
- `HTC_T_range`, `LTC_T_range`, `HTC_P_range`, `LTC_P_range`
- `equivalence_ratios`, `pressures_atm`

### 2. Error CSV folders (one folder per stage)
Each folder contains one CSV file per (φ, P) condition.

**Filename format** (case-insensitive):
```
phi_(0_5)_P_(10).csv      ←  φ = 0.5,  P = 10 atm
phi_(1_0)_P_(20).csv      ←  φ = 1.0,  P = 20 atm
PHI_(2_0)_p_(40).csv      ←  φ = 2.0,  P = 40 atm
```
Underscore in the φ value is interpreted as a decimal point (`0_5` → `0.5`).

**File contents** (two columns, header row):
```
Stage,Error_Function
Nominal,97.56
Stage-1,82.96
Stage-2,77.64
```
Column names are flexible — position matters, not name.

### 3. Cost CSV (one per stage per threshold)
```
Case_ID,Total_Rxns,Active_Rxns,p-PRS_Cost,Full_PRS_Cost
Case-0,37,32,2244,2964
Case-1,37,25,1404,2964
...
```

### 4. PRS Statistics CSV (optional, one per stage)
```
Case_ID,Training_MRE,Testing_MRE,Threshold
Case-0,0.52,1.23,5.0
Case-1,0.31,0.98,5.0
...
```
MRE values in percent (%).

### 5. Sensitivity Coefficients CSV (optional, for heatmap)
Wide format — one row per reaction, one column per condition:
```
Reaction_ID,phi05_P10,phi05_P20,...
R1,0.043,0.021,...
R2,0.112,0.087,...
```

### 6. Convergence CSV (optional)
```
Iteration,Best_Objective
1,2208.79
2,1543.21
...
```

### 7. IDT Comparison Plots folder (optional)
Folder containing pre-generated PDF figures of IDT comparison plots.
They will be included via `\includegraphics` in the report.

---

## Output Structure

```
mso_output/
├── mso_report.tex          ← LaTeX source
├── mso_report.pdf          ← compiled PDF (if pdflatex available)
├── metadata_template.yaml  ← auto-generated template
├── plots/
│   ├── cost_comparison.pdf
│   ├── radar_HTC.pdf
│   ├── radar_LTC.pdf
│   ├── sensitivity_heatmap_HTC.pdf   (if sensitivity CSV provided)
│   ├── mre_distribution_HTC.pdf
│   ├── mre_distribution_LTC.pdf
│   └── convergence.pdf               (if convergence CSV provided)
└── plot_data/
    ├── cost_comparison_data.csv
    ├── radar_data_HTC.csv
    ├── radar_data_LTC.csv
    ├── mre_distribution_HTC.csv
    └── mre_distribution_LTC.csv
```

All CSV files in `plot_data/` contain the exact data used for each figure,
so any plot can be reproduced independently.

---

## LaTeX Document Structure

| Section | Content |
|---------|---------|
| A       | Methodology: variables, formulas (PRS, p-PRS, MRE, Δε) |
| B       | Cost analysis table (p-PRS / Full-PRS staged / Conventional) |
| C       | Per-stage error tables + threshold comparison (if applicable) |
|         | Aggregate improvement table |
| D       | PRS statistics table |
| E       | Auto-generated summary paragraphs |
| F       | All figures |

---

## Notes

- **No colours** in LaTeX tables: **bold** = improvement over nominal; *italic* = degradation.
- The conventional Full-PRS cost is computed for `k_union ∈ {max(k_HTC, k_LTC), k_HTC + k_LTC}` (full overlap to no overlap).
- For threshold comparison tables, provide a separate error folder for each threshold (currently the code uses the primary threshold folder; extension is straightforward).
