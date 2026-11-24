# RMIP — Feature Roadmap
Last updated: 2026-06-29

Full design detail for Phases 5A–5D. See `progress.md` for current completion status
and `CLAUDE.md §15` for a compact summary.

---

## Phase 5A — SA/Flux Edge Highlighting in Visualizer  (PRIORITY 1)

### What it does
Color and weight edges in the bipartite reaction graph by sensitivity coefficient
or integrated flux. Reactions with importance_score above a user-supplied threshold
get bright colored edges; reactions below threshold are greyed out.
This lets the user see at a glance WHICH pathways drive the observable (IDT, LFS).

### What already exists
- `rmip/analysis/sensitivity.py` — `SensitivityFluxIntegrator.attach()` already
  attaches `sensitivity_score`, `flux_score`, `importance_score` to REACTION NODES.
- `rmip/cli.py` — `sensitivity` CLI command already reads JSON files and calls `attach()`.
- `rmip/analysis/visualization.py` — `to_cytoscape_elements()` serialises node data
  but DOES NOT yet use `importance_score` to color edges.
- `rmip/simulation/idt_runner.py`, `lfs_runner.py` — Cantera simulation runners exist.

### What needs to be built

**Step 1: Run simulations and extract SA/flux data**
- KP runs Chemkin-Pro simulations for target conditions and exports sensitivity
  coefficients and integrated fluxes as JSON files conforming to the CLAUDE.md §10
  dict schema.
- OR: use Cantera's built-in sensitivity analysis (IdealGasConstPressureReactor +
  `gas.add_sensitivity_reaction(i)`) to generate the same dict format in Python.
  Cantera SA output can be wrapped to emit the same `sensitivity_data` dict.
  This wrapper should live in `rmip/simulation/sensitivity_runner.py`.

**Step 2: Attach to graph**
Already works via `rmip sensitivity <mech_id> --db DB --sa-json FILE --flux-json FILE`.
Stores scores on reaction nodes in the in-memory graph.

**Step 3: Visualizer edge coloring**
Modify `to_cytoscape_elements()` in `visualization.py`:
- For each edge `(u, v)`, look up the reaction node (whichever of u, v has
  `type == "reaction"`) and read its `importance_score`.
- Encode importance into edge data: `"importance": score_or_0`.
- Add `--threshold FLOAT` CLI argument (default 0.1) to `visualize` command.
  Pass threshold to `NetworkVisualizer.run()` and then to `to_cytoscape_elements()`.
- In `_default_stylesheet()`, add selectors:
  ```
  edge[importance < 0.1]  → grey, width 1.0, opacity 0.25  (below threshold)
  edge[importance >= 0.1] → green (#4CAF50), width 1.5
  edge[importance >= 0.5] → orange (#FF9800), width 2.5
  edge[importance >= 0.8] → red (#F44336), width 4.0
  ```
  Cytoscape does not support continuous color mapping natively; 4 discrete threshold
  bands implemented as separate CSS selectors covers all practical use cases.

**Step 4: CLI integration**
Add to `visualize` command in `cli.py`:
```
--sa-json FILE        sensitivity JSON file (CLAUDE.md §10 format)
--flux-json FILE      flux JSON file (optional)
--threshold FLOAT     importance cutoff below which edges are greyed (default 0.1)
--score {sa,flux,importance}   which score to drive coloring (default: importance)
```

**Step 5: Dual-panel union mode**
In union comparison mode, each mechanism has its own SA/flux data.
- `--sa-json` accepts a comma-separated pair: `--sa-json sa_gri.json,sa_davis.json`
- Each panel uses its own scores; ghost-node edges always get importance = 0 (grey).

### Design constraint
SA/flux highlighting is a VIEW LAYER only — it writes nothing to the DB.
The scores are attached in-memory on the graph at visualization time.

### Files to modify
| File | Change |
|------|--------|
| `rmip/analysis/visualization.py` | `to_cytoscape_elements()` — embed `importance` in edge data; `_default_stylesheet()` — 4 importance band selectors; `NetworkVisualizer.run()` — accept `sa_data`, `flux_data`, `threshold` params |
| `rmip/cli.py` | `_add_visualize_cmd()` — add `--sa-json`, `--flux-json`, `--threshold`, `--score`; `_cmd_visualize()` — load JSON, call `SensitivityFluxIntegrator.attach()`, pass to visualizer |
| `rmip/simulation/sensitivity_runner.py` | NEW — wrap Cantera SA to emit `sensitivity_data` dict; optional alternative to Chemkin-Pro JSON |

---

## Phase 5B — Mechanism Reduction Visualization  (OPTIONAL)

### What it does
Integrate published reduction algorithms (DRG, DRGEP, PFA) and visualize the
graph at each step of species removal. The user can watch the mechanism shrink
in real time and change the threshold interactively.

### Algorithms to implement
| Method | Reference | Key parameter |
|--------|-----------|---------------|
| DRG (Directed Relation Graph) | Lu & Law 2005 | ε_DRG threshold |
| DRGEP | Pepiot-Desjardins & Pitsch 2008 | ε_DRGEP threshold |
| PFA (Path Flux Analysis) | Sun et al. 2010 | threshold |
| Sensitivity-based | standard | SA coefficient cutoff |

### Graph representation
The bipartite reaction graph already exists. Reduction = iterative node/edge removal:
1. Start with the full mechanism graph.
2. Compute interaction coefficients (r_AB for DRG, EP_AB for DRGEP).
3. Mark species below threshold as "to remove" starting from target species set.
4. Remove marked species + their exclusive reactions from graph and repeat.
5. Record each step as a `ReductionStep`.

DRG interaction coefficient:
```
r_AB = Σ|ν_A,i * ω_i| / Σ|ν_A,i * ω_i|   (where i runs over reactions involving A)
       reactions involving both A and B
```
ω_i = reaction rate (from Cantera evaluation at the target condition).

### Visualization design
- **Slider** in Dash UI: drag ε threshold from 0 → 1 → graph updates live.
- Removed nodes (below threshold): shown at 10% opacity, red dashed border.
- Retained nodes: normal colour.
- Panel subtitle: "N species / M reactions retained (ε = 0.05)"
- Animation button: step through thresholds at 0.01 intervals automatically.
- Download button: export current reduced mechanism as CHEMKIN-II `.inp`.

### New files
```
rmip/reduction/
├── __init__.py
├── drg.py          # DRGReducer — computes interaction coefficients + iterative removal
├── drgep.py        # DRGEPReducer — extended path coefficient version
└── pfa.py          # PFAReducer — path flux analysis
```

`DRGReducer` API:
```python
reducer = DRGReducer(db_path)
steps: List[ReductionStep] = reducer.reduce(
    mech_id,
    G,                    # bipartite DiGraph with flux_score on reaction nodes
    target_species,       # List[str] — "important" species to keep (e.g., fuel, oxidizer, products)
    epsilon_range,        # np.linspace(0.0, 0.5, 51)
)
# ReductionStep dataclass:
#   threshold: float
#   retained_species: List[str]
#   retained_reactions: List[str]
#   removed_species: List[str]
#   removed_reactions: List[str]
#   n_species: int
#   n_reactions: int
```

New CLI command:
```
rmip reduce <mech_id> --db DB --method drg --target "H2,O2,H2O" \
    --eps-max 0.5 --eps-steps 50 --out reduction_steps.json
```

### DB changes (Phase 5B migration)
```sql
CREATE TABLE IF NOT EXISTS reduction_runs (
    id                      INTEGER PRIMARY KEY,
    mechanism_id            INTEGER NOT NULL REFERENCES mechanisms(id),
    method                  TEXT NOT NULL,          -- 'DRG', 'DRGEP', 'PFA'
    target_species_json     TEXT NOT NULL,          -- JSON list of target species names
    epsilon                 REAL NOT NULL,
    n_species_retained      INTEGER,
    n_reactions_retained    INTEGER,
    retained_species_json   TEXT,
    retained_reactions_json TEXT,
    created_at              TEXT NOT NULL
);
```

### Dependency
Phase 5B requires flux scores on reaction nodes — needs Phase 5A (or at minimum
the `rmip sensitivity` command to have been run with flux data).

---

## Phase 5C — 3-D / Concentric-Carbon-Ring Graph Layout

### Problem
For mechanisms with 200–500 species (e.g., nHeptane, AramcoMech), a 2D spring
layout becomes an unreadable hairball. A structured layout by carbon number makes
chemical sense: species at the same carbon count cluster together; inter-shell
edges ARE the key pyrolysis/oxidation pathways.

### Carbon number extraction (from DB, no external tools needed)
```python
import re
def _carbon_number(formula: str) -> int:
    if not formula:
        return 0
    m = re.search(r'C(\d*)', formula)
    if not m:
        return 0                  # no carbon — C0 core (H, O, N, inorganic)
    return int(m.group(1)) if m.group(1) else 1   # C alone = C1
```
`species.formula` is already stored in the DB (populated by the CHEMKIN/YAML readers).

### 2D concentric-ring layout (implement first)

**Shell assignments:**
- Shell 0 (center): species with C_n = 0 (H, O, OH, H2, H2O, CO, CO2, NO, N2, …)
- Shell k: species with C_n = k (k ≥ 1)

**Angular positions:** Within each shell, species sorted alphabetically, spaced
uniformly: `θ_i = 2π × i / n_shell`.  Coordinates:
```python
x = R_k * cos(θ_i),   y = R_k * sin(θ_i),   R_k = k * delta_r   (delta_r = 300 px)
```

**Reaction node positions:** Centroid of all participating species' shell-average radii.
```python
r_rxn = mean([R_k for species in (reactants + products)])
θ_rxn = mean([θ for species])   # average angle
```

**Fixed cross-mechanism positions:**
- Same species name (or InChIKey) → same shell k → same angular index (alphabetical)
- Reactions matched by canonical_key → same (r_rxn, θ_rxn)
- Ghost nodes get same position as their source → panels are structurally identical

**Add to `_PRESET_MODES` in `visualization.py`:**
```python
"preset_carbon_ring": ("pos_x_cring", "pos_y_cring"),
```

**Add to `_assign_union_positions()` in `graph_builder.py`:**
Compute `pos_x_cring` and `pos_y_cring` alongside spring/circle/bipartite.

**Add to `_layout_options()`:**
`{"label": "Preset — Carbon rings (by Cn, fixed)", "value": "preset_carbon_ring"}`

### 3D spherical layout (implement after 2D ring works)

**Concept:** Each carbon shell is a sphere of radius R_k. Species on each sphere
distributed using Fibonacci sphere point sampling (uniform, deterministic, scales
to any number of points).

```python
# Fibonacci sphere for n points on a unit sphere
import math
golden = (1 + math.sqrt(5)) / 2
for i in range(n):
    theta = math.acos(1 - 2*(i+0.5)/n)      # polar angle
    phi   = 2 * math.pi * i / golden          # azimuthal angle
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
```

Species sorted alphabetically within each shell → same species always gets the
same Fibonacci index → deterministic positions across mechanisms.

**New method:** `_assign_union_positions_3d(graphs)` writes `pos_x_3d`, `pos_y_3d`,
`pos_z_3d` to every node in every graph.

**New CLI command:**
```
rmip visualize3d <mech_id> [mech_id2] --db DB [--label formula|id] [--port 8051]
```

**Rendering:** Plotly Dash app with `go.Scatter3d` for nodes and `go.Scatter3d` lines
for edges. Color by species type / carbon number. Rotation via mouse drag.

**Export:** `--export-figure PATH` writes static PNG via `plotly.io.write_image` (requires
`kaleido` package) or matplotlib's 3D projection as fallback.

### New files
```
rmip/analysis/visualization_3d.py    # NetworkVisualizer3D class
```

### Dependencies
`plotly >= 5.0` for 3D interactive rendering.
`kaleido` (optional) for static 3D figure export.
Carbon number extraction uses `species.formula` (already in DB).

---

## Phase 5D — Identical-SMILES Species Disambiguation

### Problem
Some species have the same canonical SMILES but are treated as distinct in the
combustion literature:
- `OHEX` (excited-state OH) vs `OH` — same ground-state SMILES `[OH]`
- `HNO2` vs `HONO` (nitrous acid) — same tautomer SMILES in some force fields
- `CH2(S)` (singlet methylene) vs `CH2` (triplet) — same SMILES `[CH2]`

**Current workaround** (CLAUDE.md §13): OHEX and HNO2 manually omitted from
`species_smiles.csv`. They parse and are stored in the DB correctly, but get
`inchi_key = NULL`. Cross-mechanism matching via InChIKey therefore never fires
for these species, so they never become ghosts — they just appear as isolated
uniquely-named nodes.

### Proposed strategy — Tier 1 (implement now, backward-compatible)

**New file: `species_variants.csv`** (alongside `species_smiles.csv`):
```csv
variant_name,canonical_name,reason
OHEX,OH,excited_state
HNO2,HONO,tautomer
CH2S,CH2,singlet_state
```

**`glossary_reader.py` changes:**
- Accept optional `variants_path` kwarg (default `None`).
- After normal glossary InChIKey lookup, apply variant overrides:
  - Look up `canonical_name` in the already-resolved glossary.
  - If found: set `species.inchi_key = canonical_species.inchi_key`,
    `species.canonical_smiles = canonical_species.canonical_smiles`,
    `species.variant_tag = reason`, `species.canonical_name = canonical_name`.
  - Log: `INFO: OHEX treated as variant of OH (excited_state) — sharing InChIKey`.
- `MechanismParser` passes `variants_path` to `glossary_reader` when provided.

**New CLI flag:**
```
rmip parse mech.inp --db project.db --glossary species_smiles.csv \
    --variants species_variants.csv
```

**Effect on union graph:**
- OHEX in GRI and OH in Davis now share the same InChIKey.
- In `build_union_graphs()`, the identity-key lookup `_sp_key(data)` returns
  `ik:{same_InChIKey}` for both → they are recognized as the SAME species.
- OHEX/OH appear at the same node position in both panels. ✓

**DB migration (Tier 1):**
```sql
-- Idempotent: ALTER TABLE only if column does not exist
ALTER TABLE species ADD COLUMN variant_tag TEXT;      -- e.g. 'excited_state', 'tautomer'
ALTER TABLE species ADD COLUMN canonical_name TEXT;   -- name of the canonical base species
```
Add as `run_phase5d_migrations(conn)` in `rmip/database/migrations.py`.

### Tier 2 — Automatic collision detection (future)
During glossary application, if two species in different mechanisms get the same
InChIKey (from SMILES canonicalization), log a `WARNING: InChIKey collision` and
suggest adding them to `species_variants.csv`. No automatic resolution.

### Tier 3 — RDKit InChI layer pruning (optional, RDKit-dependent)
Use `Chem.MolToInchi(mol, options='-SNon')` to strip stereochemistry from InChI
and detect tautomers more reliably. Deferred; RDKit is available but this adds
complexity to canonicalize.py.

### Visualizer changes (minor)
In `_default_stylesheet()`, add a selector for nodes with `variant_tag` attribute:
```css
node[variant_tag]  →  border-style: double; border-width: 3px; border-color: #FF6F00
```
Tooltip (Cytoscape `title` data field) shows `"Variant of <canonical_name> (<reason>)"`.

### Files to modify
| File | Change |
|------|--------|
| `rmip/io/glossary_reader.py` | Accept `variants_path`; apply overrides after glossary lookup |
| `rmip/core/parser.py` | Pass `variants_path` from CLI args to `GlossaryReader` |
| `rmip/database/migrations.py` | Add `run_phase5d_migrations()` |
| `rmip/cli.py` | Add `--variants FILE` to `parse` command |
| `rmip/analysis/visualization.py` | Add variant badge to stylesheet |

---

## Implementation Order (recommended)

| Priority | Phase | Estimated effort | Key dependency |
|----------|-------|-----------------|----------------|
| 1st | **5A** — SA/flux edge highlighting | 2–3 sessions | Chemkin-Pro JSON or Cantera SA wrapper |
| 2nd | **5D** — SMILES disambiguation | 1 session (Tier 1) | None |
| 3rd | **5C** — Carbon-ring 2D first, then 3D | 3–4 sessions | `species.formula` in DB (already present) |
| 4th | **5B** — Mechanism reduction | 3–4 sessions | Phase 5A (flux scores as weights) |

---

## Open questions / decisions needed from KP

1. **Phase 5A — SA data source**: Use Chemkin-Pro JSON files (KP writes wrapper), or
   wrap Cantera's built-in SA? Cantera SA is available NOW; Chemkin-Pro wrapper needs
   to be written separately.
2. **Phase 5A — threshold default**: 0.1 (10% of max importance) is proposed.
   Is that appropriate for H₂/CO mechanisms?
3. **Phase 5C — 3D backend**: Plotly (interactive, browser) vs matplotlib 3D
   (static export, but no interactivity) vs VTK/Mayavi (heavy, not installed)?
   Recommend Plotly — already used for dash.
4. **Phase 5D — OHEX treatment**: Should OHEX and OH appear as ONE merged node in
   the union graph (positions collapse) or as SEPARATE nodes with a visual link?
   Merging simplifies the graph; keeping separate preserves mechanistic detail.
5. **Phase 5B — target species**: For H₂ mechanisms, target is typically {H₂, O₂, H₂O}.
   How should the user specify this? CLI string `"H2,O2,H2O"` or a JSON file?
