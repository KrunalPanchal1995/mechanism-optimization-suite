# MUQ-SAC: Method of Uncertainty Quantification and Sampling of ARRHENIUS CURVES
The suit of code is part of project undertaken for kinetic mechanism optimization framework development. The code includes the temperature-dependent Modified Uncertainty Quantification (MUQ) method and Sampling of Arrhenius Curves (SAC) technique.


Step 0
 -  Compile the shared object files (*.so) for your system


Step 1

 - Converting mechanism files into yaml: (write code in terminal)
     
     ck2yaml --input=chem.inp --thermo=therm.dat --transport=tran.dat --permissive
     (eg:ck2yaml --input=MB_MB2D.inp --thermo=MB+MB2D_therm.dat --transport=Mb+MB2d_trans.dat --permissive)
     
Step 2

 - Generating xml file for reactions considered in sensitivity analysis
 
Step 3
 
 - Generating target input file [target file, Addendum file]
 - Collect the mechanism file [mech file, thermo file, transport file]

#For Optimization
Step 4
- run in terminal
	python3.9 /home/krithika/Desktop/MUQ-SAC/MUQ-SAC/run_script.py [target_file]



#For running simple simulations
Step 4
- run in terminal
	python3.9 /home/krithika/Desktop/MUQ-SAC/MUQ-SAC/run_nominal_sim.py [target_file]


#For running sensitivity analysis
Step 4
- run in terminal
	python3.9 /home/krithika/Desktop/MUQ-SAC/SENSITIVITY_ANALYSIS/sens.py [target_file]
 




# Mechanism Optimization Suite

A unified toolkit for combustion **kinetic-mechanism optimization**, **sensitivity
graph analysis**, and **forward uncertainty quantification (UQ)**.

Three capabilities live in one repository, sharing the same `target.opt` config
and dataset layout:

| # | Capability | Entry point | What it does |
|---|------------|-------------|--------------|
| 1 | **Optimization (MUQ-SAC / MSO)** | `src/MUQ-SAC/run_script.py` | Partial/full response-surface optimization of Arrhenius parameters against experimental targets. |
| 2 | **Sensitivity Graph Analysis** | `graph_sa_run.py` | Overlays the sensitivity results (`sens_parameters.pkl`) on the reaction network to reveal **sensitive reactions** and **sensitive species** — graphs + ranked CSVs. |
| 3 | **Forward UQ** | `forward_uq.py` | Propagates Arrhenius uncertainty through trained PRS surrogates (Monte-Carlo + analytical Sobol indices). |

The graph engine underneath capability #2 is **RMIP** (Reaction Mechanism
Intelligence Platform), now vendored at `src/rmip/`.

---

## Repository layout

```
.
├── src/
│   ├── MUQ-SAC/              # MSO / MUQ-SAC optimization pipeline (run_script.py)
│   ├── rmip/                 # RMIP graph engine (parser, graph builder, sensitivity, viz)
│   ├── graph_sa/             # NEW: SA-pickle → RMIP graph bridge (+ static renderers)
│   ├── forward_uq/           # Forward-UQ propagation utilities (MC, Sobol, plots)
│   ├── modules/ io/ …        # MSO support modules
│   └── Sensitivity_Analysis/ # brute-force SA tool used by run_script.py
│
├── graph_sa_run.py           # driver: sensitivity graph analysis
├── run_graph_sa.sh           # one-command runner (defaults to the MB dataset)
├── forward_uq.py             # driver: forward UQ propagation
├── run_forward_uq.sh         # forward-UQ runner with retry/back-off
├── rmip_cli.py               # repo-root launcher for the RMIP CLI
├── conftest.py               # puts src/ on sys.path for pytest (append-only, see note)
│
├── DATASET/
│   └── METHYL_BUTANOATE_2026/   # MB reference dataset
│       ├── target.opt              # run configuration (YAML)
│       ├── sens_parameters.pkl     # sensitivity results (from run_script.py)
│       ├── MB_MB2D_LALIT_2024.yaml # mechanism (623 species, 4050 reactions)
│       └── …
│
├── outputs/                  # generated figures, graphs, CSVs
├── CLAUDE.md                 # RMIP design decisions & full module reference
└── QUICKSTART.md             # copy-paste commands for each workflow
```

---

## Install

```bash
# Python 3.9 (base conda) is the reference interpreter.
pip install networkx numpy scipy pandas pyyaml matplotlib pytest --break-system-packages
pip install cantera --break-system-packages     # mechanism parsing + simulations
# rdkit is optional (only needed for InChIKey/SMILES resolution in RMIP)
```

---

## Quick check (Methyl Butanoate dataset)

```bash
# Sensitivity graph analysis on the bundled MB dataset
./run_graph_sa.sh
#   → outputs/graph_sa/sensitive_subgraph.png
#   → outputs/graph_sa/full_graph_sensitive.png
#   → outputs/graph_sa/sensitive_reactions.csv
#   → outputs/graph_sa/sensitive_species.csv
```

See **[QUICKSTART.md](QUICKSTART.md)** for all three workflows.

---

## Important: the `src/io` / `sys.path` note

`src/io/` is a package literally named `io`. If `src/` is placed at the **front**
of `sys.path` while the interpreter is starting up, it shadows the standard
library `io` and crashes Python. Everything in this repo therefore follows one
rule:

* **Append** `src/` to `sys.path` (drivers, `conftest.py`, `rmip_cli.py`) — safe.
* Do **not** `cd src && python -m rmip.cli`. Use the repo-root launcher
  `python rmip_cli.py …` instead.
* The one exception is `forward_uq.py`, which must front-load `src/` to resolve a
  name clash with the `src/forward_uq` package; that is runtime-safe because
  stdlib `io` is already cached by then (documented in the file).

---

## Tests

```bash
python -m pytest src/rmip/tests/ -v         # 189 RMIP engine tests
python -m pytest src/graph_sa/tests/ -v     # MB end-to-end sensitivity-graph test
```
