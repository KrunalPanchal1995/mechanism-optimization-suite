# Quick-Start

**Working directory:** repo root (the folder containing this file)
**Reference Python:** `python3` = `/home/krithika/anaconda3/bin/python` (3.9, base conda)

All commands are run from the repo root. Do **not** `cd src` (see the `src/io`
note in [README.md](README.md)).

---

## 1. Sensitivity Graph Analysis  ← the new feature

Turns the sensitivity results already produced by the optimizer
(`sens_parameters.pkl`) into a graph view of the **sensitive reactions** and
**sensitive species**.

### One command (Methyl Butanoate reference dataset)

```bash
./run_graph_sa.sh
```

Equivalent explicit call:

```bash
python graph_sa_run.py DATASET/METHYL_BUTANOATE_2026/target.opt --out outputs/graph_sa
```

### Options

```bash
python graph_sa_run.py <target.opt> \
    --out outputs/graph_sa \      # output directory
    --method max \                # aggregate |coeff| across targets: max (default) | mean
    --threshold 0.1 \             # sensitivity below this is greyed
    --top 25 \                    # rows printed to the terminal
    --no-full-graph               # skip the slow full-mechanism figure
```

### Outputs (in `outputs/graph_sa/`)

| File | Contents |
|------|----------|
| `sensitive_subgraph.png` | Compact, labelled graph of the sensitive reactions + their species. Nodes/edges colour- and size-graded by sensitivity (green ≥ threshold, orange ≥ 0.5, red ≥ 0.8). |
| `full_graph_sensitive.png` | The whole mechanism (carbon-ring layout) with the sensitive chemistry highlighted in colour over a faint background. |
| `sensitive_reactions.csv` | Reactions ranked by max sensitivity, with a per-target breakdown column. |
| `sensitive_species.csv` | Species ranked by aggregated sensitivity (max over the reactions they participate in), with carbon number and reaction count. |

### How it works

1. `target.opt` → mechanism path + `sens_parameters.pkl` path (resolved relative
   to the config so it runs on any host).
2. `sens_parameters.pkl` = `{target_idx: {reaction: coeff}}` is collapsed to one
   score per reaction (**max |coeff| across all targets**, normalised to [0, 1]).
3. Reactions are matched to the parsed mechanism by canonicalised formula
   (arrow/whitespace/token-order insensitive); the match rate is reported.
4. Reaction scores are aggregated onto species, attached to the RMIP reaction
   graph, and rendered.

### Test / verify

```bash
python -m pytest src/graph_sa/tests/ -v      # end-to-end MB check
```

---

## 2. Optimization (MUQ-SAC / MSO)

The original optimization pipeline, unchanged.

```bash
cd src/MUQ-SAC
python3.9 run_script.py <path-to>/target.opt
```

`run_script.py` reads `target.opt`, runs (or reuses) the brute-force sensitivity
analysis to produce `sens_parameters.pkl`, builds the response surfaces, and
optimizes. That same `sens_parameters.pkl` is the input to workflow #1 above.

---

## 3. Forward UQ

Propagates Arrhenius uncertainty through the trained PRS surrogates.

```bash
python forward_uq.py DATASET/METHYL_BUTANOATE_2026/target.opt \
    --n_samples 50000 --method sobol --output_dir outputs/forward_uq

# or, with automatic retry/back-off:
./run_forward_uq.sh
```

**Prerequisite:** forward UQ needs the PRS coefficient files that the
optimization run (`run_script.py`) generates. Run workflow #2 first.

---

## 4. RMIP graph engine (direct CLI)

The full RMIP CLI is available through the repo-root launcher:

```bash
python rmip_cli.py list --db project.db
python rmip_cli.py summary 1 --db project.db
python rmip_cli.py visualize 1 2 --union --db project.db     # interactive Dash app
```

See [CLAUDE.md](CLAUDE.md) §8 for the complete command reference.
