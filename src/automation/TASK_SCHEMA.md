# run_tasks.py — YAML task file reference

One YAML file = one study. Give it to the runner:

```bash
cd /data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/MSO_A_FACTOR/p_PRS/_automation
python3.9 run_tasks.py my_study.yaml --validate   # ALWAYS do this first
python3.9 run_tasks.py my_study.yaml --daemon     # detach, runs for days
python3.9 run_tasks.py my_study.yaml --status     # progress summary
python3.9 run_tasks.py my_study.yaml              # foreground (testing)
```

Runner files (all in `_automation/`, named after the YAML's basename):
`state_<name>.json` (resumable state), `runner_<name>.log` (main log),
`task_logs/` (one log per launched run), `<name>.complete` (written when every
task finished), `ALERTS.log` (shared failure log). Restarting the runner is
safe: launched child processes keep running and are reconciled on the next
scan. A lock file prevents two runners on the same YAML.

## Top level

```yaml
version: 1
settings:   {...}     # optional
locations:  {...}     # named directories used by tasks
tasks:      [...]     # executed by type-specific rules, see below
```

## settings

| key | default | meaning |
|---|---|---|
| `system_thread_cap` | 160 | hard cap; validation refuses any task whose `max_terminals × parallel_threads` exceeds it, and the dispatcher defers launches that would push the live committed total over it |
| `poll_minutes_busy` | 8 | sleep between scans while runs are active |
| `poll_minutes_idle` | 30 | sleep when only waiting (e.g. for new_mech.yaml) |
| `poll_seconds_busy` / `poll_seconds_idle` | — | override in seconds (testing) |
| `python` | anaconda3 python3.9 | interpreter used in default commands |
| `state_file` | auto | `auto` → `_automation/state_<yamlname>.json` |

## locations

```yaml
locations:
  myfuel_opt:
    dir: /abs/path/to/fuel/HTC        # required
    target_opt: target.opt            # optimize tasks: master .opt (default target.opt)
  myfuel_cmp:
    dir: /abs/path/to/MECHANISM_COMPARISON/...
    ref_config: /abs/path/.../MSO_CONFIG.yaml   # compare tasks: reference config to copy
```

## Common task keys

- `id` (required, unique), `type` (required: `edit_params` | `optimize` | `compare` | `report`)
- `depends_on`: task id or list. `compare` depending on an `optimize` task chains
  **per case** (each comparison starts as soon as that case's marker appears);
  all other dependencies wait for the whole dependency task to complete.
  Failed runs never trigger their dependents; everything else continues.

**Parameter keys are dotted YAML paths** exactly as nested in the file:
`Stats.pm_p_start`, `Counts.parallel_threads` (target.opt),
`simulations.stage1_htc.mechanism` (MSO_CONFIG.yaml). Validation refuses keys
that don't exist in the target file — typos are caught before anything runs.
Edits preserve comments and formatting.

## type: edit_params

Edit keys in one file, once.

```yaml
- id: set_baseline
  type: edit_params
  file: myfuel_opt/target.opt      # "<location>/<relative>" or absolute path
  edits:
    Stats.pm_anneal_gens: 1200
    Counts.parallel_threads: 2
```

## type: optimize

Launch a sweep of MUQ-SAC optimizations. Every run gets its own isolated copy
of the master `target.opt` (`target_auto_<taskid>_<folder>.opt`) — concurrent
runs never share a mutable file.

```yaml
- id: opt_sweep
  type: optimize
  locations: [myfuel_opt, otherfuel_opt]
  sweep:                                  # cross-product grid
    Stats.pm_p_start: [1, 3, 5]
    Stats.pm_p: [0.05, 0.15]
  linked:                                 # follow a swept value
    Stats.pm_p_polish: "{Stats.pm_p}"
  fixed:                                  # constants for every run
    Stats.pm_anneal_gens: 1200
  output_folder_template: "powerMeanWMRE_{Stats.pm_p_start}_{Stats.pm_p}_1200_GEN_PM_POLISH_{Stats.pm_p_polish}"
  max_terminals: 45          # REQUIRED — refuses to start without it
  parallel_threads: 2        # REQUIRED — written into each run's .opt copy
  stagger_seconds: 20        # default 20; pause between launches
  skip_if_output_exists: true  # default true; pre-existing case folders are
                               # not relaunched (marked done/skipped_existing)
  done_marker: new_mech.yaml   # default; per-case completion signal
  cleanup_intermediates: true  # default false; see Cleanup below
  # command: "{python} /home/krithika/MUQ-SAC/V3.0/run_script.py {opt_file}"  # default
  # login_shell: true          # default true (bash -i -c 'exec ...')
  # parallel_threads_key: Counts.parallel_threads   # default
```

Template placeholders take the parameter's value with dots→underscores
(`0.05` → `0_05`), matching the MUQ-SAC folder convention. The template must
produce a **unique** folder per grid point (validated).

A run is **done** when `<location dir>/<folder>/<done_marker>` exists,
**failed** when its process is gone without the marker (alert logged, no
retry — relaunch later with a small YAML if wanted; `skip_if_output_exists`
only skips folders that already exist, so a failed run whose folder was
created will not be relaunched automatically).

## type: compare

Run the mechanism-comparison pipeline for each finished optimization case:
create `<to dir>/<case name>` (prefix `_` if the name already exists there),
copy `ref_config` in as `MSO_CONFIG.yaml`, apply `config_edits`, run the
pipeline in that folder.

```yaml
- id: cmp
  type: compare
  depends_on: opt_sweep         # OR standalone with watch:
  # watch:
  #   case_glob: "powerMeanWMRE_*"
  #   expect_total: 15          # keep watching until N cases processed;
  #                             # without it the task completes once every
  #                             # case CURRENTLY matching has been processed
  source_marker: new_mech.yaml  # default; readiness signal in the source case
  pairs:
    - {from: myfuel_opt, to: myfuel_cmp}
  config_edits:
    simulations.stage1_htc.mechanism: "{case_dir}/new_mech.yaml"
  collision_prefix: "_"         # default
  max_terminals: 8              # REQUIRED
  parallel_threads: 19          # REQUIRED — written ONCE per 'to' location into
                                # the base target.opt referenced by ref_config's
                                # base_opt_file (a shared file is never edited
                                # per-case — that's the 2026-07-11 race class)
  done_marker: PLOTS/HTC/nominal_s1/combined_nominal_vs_s1.pdf   # default
  # command: "{python} .../run_mso_pipeline.py --config MSO_CONFIG.yaml"  # default
  # cleanup_intermediates: true # also allowed here (applies to SOURCE cases)
  # cleanup_sim_runs: true      # default false; see Cleanup below
```

Placeholders available in `config_edits` and `command`: `{case_dir}`,
`{case_name}`, `{from_dir}`, `{to_dir}`, `{target_dir}`, `{python}`.

## type: report

One LaTeX report (+ CSV of the same numbers) per location, written to
`output_dir` as `report_<location>.tex/.csv`. Objective values are the sum of
the `Error_Function` column over all `ERROR_METRICS/HTC/nominal_s1/phi_*/*.csv`
files of each case. Table: case-ID index, Nominal row, Stage-1 objective per
case with % vs nominal, sum row; plus generated observations (best/worst case,
regressions vs nominal, nominal-consistency check).

```yaml
- id: reports
  type: report
  depends_on: cmp
  locations: [myfuel_cmp]
  output_dir: /abs/path/MECHANISM_COMPARISON
  case_glob: "*powerMeanWMRE_*"    # default "*"
  # metrics_glob: ERROR_METRICS/HTC/nominal_s1/phi_*/*.csv   # default
  # title: "My study title"
```

## Concurrency model (fully explicit — the runner never picks numbers for you)

1. Every `optimize`/`compare` task **must** state `max_terminals` and
   `parallel_threads`. Missing → validation error, nothing runs.
2. Per task: `max_terminals × parallel_threads ≤ system_thread_cap`, else
   validation refuses.
3. Live: before each launch the runner sums `parallel_threads` over ALL
   currently active runs of ALL tasks in this YAML; a launch that would exceed
   the cap is deferred to a later scan (logged). So an optimize task and a
   compare task overlapping will throttle each other instead of oversubscribing
   the machine.
4. Typical numbers: comparison ≤ 8 terminals × 19 threads = 152; optimization
   40–100 terminals × 1–2 threads.

## Cleanup (pre-authorized scope only)

With `cleanup_intermediates: true`, after a case's `new_mech.yaml` exists the
runner deletes — from that case folder, searched recursively at any depth
(these files live inside `run_1/`, `run_2/`, ... subfolders, not the case
folder itself) — exactly: `Dataset_based_obj`, `guess_values.txt`,
`guess_values_TRANSFORMED.txt`, `response_values.txt`,
`zeta_guess_values.txt`. Never anything else, never from a case still in
progress. Default is **off**.

With `cleanup_sim_runs: true` on a `compare` task, once a case's `done_marker`
(the plot) is confirmed present, the runner also removes every `SIM_RUNS`
directory found anywhere under that comparison's target folder (the raw
per-condition simulation working dirs — safe once `PLOTS`/`ERROR_METRICS`
already hold the derived outputs). Also opt-in, default **off**, and applied
on the runner-restart orphan-reconcile path too.

For ad hoc or retroactive cleanup outside the YAML runner (e.g. reclaiming an
existing backlog), see `cleanup_space.py --help`: it reuses this exact same
deletion scope/logic, defaults to a dry-run preview, and never touches any
`state_*.json` file (safe to run alongside a live daemon).

## Failure policy

Failed run → `status=failed` in the state file + a line in `ALERTS.log`;
other runs continue; dependents of that case don't start. No automatic
retries. `--status` lists every failed run with its note and log path.
