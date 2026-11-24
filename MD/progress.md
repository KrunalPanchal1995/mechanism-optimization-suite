# RMIP — Implementation Progress Log
Last updated: 2026-06-29

---

## Current status: Phase 1–5A/5C/5D COMPLETE, Phase 5B PLANNED

**Test suite:** 189/189 passing (`python -m pytest rmip/tests/ -v`)
- 62 parser tests (`test_chemkin_parser.py`)
- 50 Phase-3 tests (`test_phase3.py`)
- 48 Phase-4 tests (`test_phase4.py`)
- 29 Phase-5 tests (`test_phase5.py`)

**Source files:** ~15,700 lines across 38 Python files  
**Working directory:** `/data/OPEN_CODE/PHASE-3/`  
**Python:** `/home/krithika/anaconda3/bin/python` (3.9, base conda)

---

## Completed phases

### Phase 1 — Core parsing & graph (COMPLETE)
- `rmip/io/chemkin_reader.py` — dual-path CHEMKIN reader (RMG-Py or built-in)
- `rmip/io/_chemkin_parser.py` — standalone pure-Python CHEMKIN-II parser
  (NASA-7 thermo, Arrhenius/PLOG/Troe/Lindemann/Chebyshev, third-body, DUPLICATE)
- `rmip/io/cantera_reader.py` — Cantera YAML reader
- `rmip/io/glossary_reader.py` — species SMILES glossary CSV/YAML reader
- `rmip/core/parser.py` — MechanismParser (readers → DB)
- `rmip/core/canonicalize.py` — RDKit InChIKey + SMILES canonicalization
- `rmip/core/graph_builder.py` — bipartite reaction network graph builder
- `rmip/database/schema.py` — core DDL (mechanisms/species/reactions/stoich)
- CLI: `parse`, `graph`, `list`, `summary`

### Phase 2 — Analysis & visualization (COMPLETE)
- `rmip/analysis/visualization.py` — interactive Dash network visualizer (865 lines)
  - Single-panel and side-by-side dual-panel modes
  - Union mode: ghost nodes (38% opacity, dashed border) for absent species/reactions
  - Three fixed shared preset layouts: Spring (topology), Circle (alphabetical), Bipartite
  - **Global union R-IDs** (2026-06-28): same reaction gets same R-label across all
    mechanisms — `_assign_global_reaction_labels()` in `graph_builder.py`
  - Shared layout dropdown: one dropdown drives both panels simultaneously (no desync)
  - Ghost reaction labels: formula string in formula mode, global R-ID in R-ID mode
  - Diff stats bar: species/reaction coverage %, ghost counts, key missing species
  - Per-panel PNG download (4× resolution via Cytoscape `generateImage`)
  - Publication figure export: PDF/SVG/PNG via matplotlib (`--export-figure PATH`)
  - CLI flags: `--union`, `--label {formula|id}`, `--export-figure PATH`,
    `--figure-dpi N`, `--port N`
- `rmip/analysis/comparison.py` — M6 kinetics/thermo numerical comparison
- `rmip/analysis/sparsity.py` — M4 dead-end / orphan / coverage detection
- `rmip/analysis/sensitivity.py` — M8 SA/flux score attachment to reaction nodes
  (attaches `sensitivity_score`, `flux_score`, `importance_score`; edge coloring
  NOT yet in visualizer — that is Phase 5A)
- `rmip/database/schema_phase3.py` — Phase-3 simulation/comparison tables
- CLI: `compare`, `visualize`, `sparsity`, `sensitivity`, `reaction-matrix`, `reactions`

### Phase 3 — Fusion, optimization, simulation (COMPLETE)
- `rmip/analysis/fusion.py` — M7 FusionEngine (VERBATIM/COPY/FUSE kinetics logic)
- `rmip/analysis/graph_comparator.py` — M6b graph-level comparison → union spec
- `rmip/analysis/parameter_estimator.py` — Arrhenius statistics, MSO bounds
- `rmip/analysis/benchmarking.py` — IDT/LFS vs experimental data (MRE/RMSE)
- `rmip/simulation/idt_runner.py` — Cantera constant-volume adiabatic ignition
- `rmip/simulation/lfs_runner.py` — Cantera freely-propagating premixed flame
- `rmip/simulation/performance_ranker.py` — MRE/RMSE ranking by LTC/NTC/HTC regime
- `rmip/optimization/mso_interface.py` — JSON contract bridge to OptimizationTool_v4.py
- `rmip/generation/rmg_interface.py` — RMG-Py subprocess + DB append
- `rmip/io/chemkin_writer.py` — DB → CHEMKIN-II .inp exporter (609 lines)
- `rmip/database/migrations.py` — idempotent schema upgrades (Phase-3 + Phase-4)
- CLI: `fuse`, `optimize`, `rank`, `complete`, `export`, `compare-graph`,
        `exp-load`, `bench`, `reaction-matrix`, `reactions`

### Phase 4 — Unified mechanism & uncertainty (COMPLETE)
- `rmip/analysis/uncertainty_bands.py` — per-T k(T) uncertainty bands (3σ convention)
- `rmip/analysis/unified_tree.py` — UnifiedTreeBuilder: plan (CSV manifest) + build
- `rmip/io/cantera_writer.py` — DB → Cantera YAML with uncertainty embedding
- `rmip/generation/missing_species_predictor.py` — cross-mech + topological gap finder
- `rmip/database/schema_phase4.py` — unified_tree_runs / reaction_decisions / uncertainty_bands
- CLI: `unify-plan`, `unify-build`, `predict-missing`, `export <id> --format yaml`

---

## Visualizer bug fixes (sessions 2026-06-27 to 2026-06-29)

| Date | Fix | File(s) | Description |
|------|-----|---------|-------------|
| 2026-06-27 | Ghost node ID collision | `graph_builder.py` | `ghost_S_{id}` for species, `ghost_R_{id}` for reactions — avoids collision between species and reaction rows that share the same integer db_id |
| 2026-06-27 | Position identity key mismatch | `graph_builder.py` | Rewrote `_assign_union_positions()` to build a true union graph keyed by `ik:{inchi_key}` / `nm:{name}` / `ck:{canonical_key}` / `cf:{canon_formula}`. Ghost nodes copy `**src_data` so have same identity → same (x,y) guaranteed. |
| 2026-06-27 | Formula string mismatch between mechs | `graph_builder.py` | Added `_canon_formula()` — normalizes arrow notation and sorts species within each side. "H + O2 <=> OH + O" and "O2 + H = O + OH" now hash to the same key. Union graph shrank from ~85 to ~66 nodes for GRI+Davis. |
| 2026-06-28 | Independent per-panel layout dropdowns | `visualization.py` | Replaced two `layout-dropdown-0` / `layout-dropdown-1` with single `layout-dropdown-shared` in the header. Single callback `update_both_panels` writes layout+elements for both panels simultaneously — they can never desync. |
| 2026-06-28 | Ghost reactions had no labels | `visualization.py` | Changed `label = ""` to `label = data.get("formula_string", "")` for ghost reactions. They now show formula string in formula mode and global R-ID in R-ID mode. |
| 2026-06-28 | Per-mechanism R-IDs — same reaction had different labels in each panel | `graph_builder.py` | Added `_assign_global_reaction_labels()` called from `build_union_graphs()`. First-appearance order (mech-1 reactions first, then unique reactions from mech-2, etc.) → same reaction gets same R-label everywhere. |

---

## Phase 5A — SA/Flux Edge Highlighting (COMPLETE — 2026-06-29)

- `rmip/analysis/visualization.py` — `to_cytoscape_elements()` embeds `importance` on edges; `_default_stylesheet(threshold)` adds 4-band importance selectors; `run()` accepts `threshold` / `score` params; `_single_layout()` / `_dual_layout()` thread threshold
- `rmip/cli.py` — `visualize` command: `--sa-json`, `--flux-json`, `--threshold`, `--score` args; attach SA/flux scores before launching app
- `rmip/simulation/sensitivity_runner.py` — NEW: `CanteraSensitivityRunner.run_idt()` wraps Cantera built-in SA → `sensitivity_data` dict (CLAUDE.md §10 format)
- `rmip/cli.py` — NEW `sa-run` subcommand: run Cantera IDT SA and save JSON

**Usage:**
```bash
# Generate SA JSON from Cantera
python -m rmip.cli sa-run FFCM1.yaml --T 1200 --P 1.0 --phi 1.0 --fuel H2 --out sa_ffcm1.json

# Visualize with edge coloring
python -m rmip.cli visualize 1 2 --union --db project.db --sa-json sa_gri.json --threshold 0.1
```

---

## Phase 5C — Carbon-Ring 2D Layout (COMPLETE — 2026-06-29)

- `rmip/core/graph_builder.py` — `_carbon_number(formula)` helper; `_carbon_ring_positions(union_G)` assigns concentric ring positions by carbon number; `build_graph()` now calls `_assign_union_positions([G])` to populate ALL preset positions (spring, circle, bipartite, carbon-ring) in single-panel mode too
- `rmip/analysis/visualization.py` — `_PRESET_MODES` extended with `"preset_carbon_ring"`; `_layout_options()` shows "Carbon rings" in dropdown

**Usage:** Launch `rmip visualize` (any mode) → select "Preset — Carbon rings" from dropdown

---

## Phase 5D — Species SMILES Disambiguation, Tier 1 (COMPLETE — 2026-06-29)

- `rmip/io/glossary_reader.py` — NEW `load_variants(path)` function reads `species_variants.csv`
- `rmip/core/parser.py` — `parse()` accepts `variants_path`; `_insert_species()` applies variant overrides: OHEX shares OH's InChIKey
- `rmip/database/migrations.py` — NEW `run_phase5d_migrations(conn)` adds `species.variant_tag` and `species.canonical_name` columns (idempotent)
- `rmip/cli.py` — `parse` command: `--variants FILE` argument
- `rmip/analysis/visualization.py` — `_default_stylesheet()` adds `node[variant_tag]` selector (double orange border)

**Usage:**
```bash
# Parse with variant overrides
python -m rmip.cli parse mech.inp --db project.db \
    --glossary species_smiles.csv --variants species_variants.csv
```

`species_variants.csv` format:
```csv
variant_name,canonical_name,reason
OHEX,OH,excited_state
HNO2,HONO,tautomer
```

---

## Output folder organization (2026-06-29)

Root-level clutter moved to `outputs/` subdirectories:
- `outputs/graphs/`    — graphml and JSON graph exports
- `outputs/reactions/` — reaction CSV and matrix files
- `outputs/figures/`   — PNG/PDF/SVG visualizations
- `outputs/databases/` — demo and test databases (project.db remains at root)

---

## Known limitations / deferred items

| Item | Status | Target phase |
|------|--------|-------------|
| Cantera SA wrapper: run_lfs() for LFS sensitivity | Not built | Phase 5A follow-up |
| 3D graph for large mechanisms | Not built | Phase 5C |
| Mechanism reduction (DRG/DRGEP/PFA) | Not built | Phase 5B |
| `reaction-matrix` CSV still uses per-mechanism DB labels | Minor inconsistency | Phase 5A cleanup |
| RMG-Py not installed | Expected (source at `/data/OPEN_CODE/RMG_py/`) | N/A |
| Transport data | Out of scope (D6) | N/A |
| NASA-9 polynomials | Not supported | N/A |

---

## Live project database (`project.db`)

| ID | Name | Species | Reactions | Format |
|----|------|---------|-----------|--------|
| 1  | GRI-3.0 | 15 | 48 | chemkin |
| 2  | Davis-2005 | 14 | 38 | chemkin |
| 3  | Keromenes-2013 | 17 | 49 | chemkin |
| 4  | Li-2015 | 14 | 37 | chemkin |
| 5  | FFCM1 | 38 | 291 | cantera_yaml |

---

## Mechanism file paths

| Mechanism | Path |
|-----------|------|
| GRI-3.0 | `/home/krithika/MUQ-SAC/MECHANISMS/GRI3/chem_gri30_wetCO.inp` |
| Davis-2005 | `/home/krithika/MUQ-SAC/MECHANISMS/Davis_2005/chem_Davis2005_mech_wetCO.inp` |
| Keromenes-2013 | `/home/krithika/MUQ-SAC/MECHANISMS/Keromens/chem_Keromnes2013_orig_mech_wetCO.inp` |
| Li-2015 | `/home/krithika/MUQ-SAC/MECHANISMS/Li_2015/chem_Li2015_wetCO_mech.inp` |
| NUIG-2010 | `/home/krithika/MUQ-SAC/MECHANISMS/NUIG-NGM_2010/chem_NUIG2010_wetCO_nc5_49.inp` |
| CRECK-2012 | `/home/krithika/MUQ-SAC/MECHANISMS/CRECK/chem_CRECK2012_mech_wetCO_v1212.inp` |
| FFCM1 | `/home/krithika/MUQ-SAC/MECHANISMS/FFCM1/FFCM1.yaml` |
| AramcoMech 1.3 | `/home/krithika/MUQ-SAC/MECHANISMS/Keromens/aramcomech_1.3_syngas_chem/aramcomech_1.3_syngas_chem.yaml` |
| nHeptane | `/home/krithika/MUQ-SAC/MECHANISMS/nHeptane/heptane.inp` |

---

## How to resume work in a new session

```bash
cd /data/OPEN_CODE/PHASE-3/

# 1. Verify everything is still working
python -m pytest rmip/tests/ -v   # expect 189/189

# 2. Check project DB
python -m rmip.cli list --db project.db

# 3. Launch interactive visualizer (union comparison, GRI vs Davis)
python -m rmip.cli visualize 1 2 --union --db project.db

# 4. SA/flux edge highlighting (Phase 5A):
#    a. Generate SA JSON from Cantera:
python -m rmip.cli sa-run /home/krithika/MUQ-SAC/MECHANISMS/FFCM1/FFCM1.yaml \
    --T 1200 --P 1.0 --phi 1.0 --fuel H2 --out outputs/sa_ffcm1.json
#    b. Visualize with colored edges:
python -m rmip.cli visualize 1 2 --union --db project.db --sa-json outputs/sa_ffcm1.json

# 5. Carbon-ring layout (Phase 5C): just launch visualizer and select dropdown
python -m rmip.cli visualize 1 2 --union --db project.db
# → "Preset — Carbon rings (by Cn, fixed)" is now in the layout dropdown

# 6. Species variants (Phase 5D):
python -m rmip.cli parse mech.inp --db project.db \
    --glossary species_smiles.csv --variants species_variants.csv

# 7. Next implementation target: Phase 5B (mechanism reduction DRG/DRGEP/PFA)
#    See PLAN.md §5B for full design
```

---

## Key design decisions (immutable, from CLAUDE.md §2)

| ID | Decision |
|----|----------|
| D1 | Species identity = `inchi_key` (RDKit InChIKey) |
| D2 | Reaction identity = `canonical_key` (sorted-SMILES frozenset, directional) |
| D3 | SQLite only — no ORM, stdlib `sqlite3`, SI units in DB |
| D4 | IO readers return plain `dict` — no DB writes inside readers |
| D5 | Inter-module transfer uses dataclasses from `rmip/types.py` |
| D6 | Transport data out of scope |
| D7 | MSO interface = JSON file contract only (subprocess, no import) |
| D8 | Ghost node IDs: `ghost_S_{db_id}_{src_mech}` (species) / `ghost_R_{db_id}_{src_mech}` (reaction) |
| D9 | Global union R-IDs: assigned by `_assign_global_reaction_labels()` in union mode; first-appearance order from mechanism 1 |
| D10 | 3σ uncertainty convention: σ_lnk = ln(f)/3, f = sqrt(k_max/k_min) |
