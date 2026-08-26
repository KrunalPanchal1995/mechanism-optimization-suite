#!/usr/bin/env python3
"""
run_tasks.py — generic YAML-driven task runner for the MSO_A_FACTOR automation.

Integrated into the mechanism-optimization-suite repo (2026-07-21) from the
external /data2/.../MSO_A_FACTOR/p_PRS/_automation/ project. DEFAULT_PYTHON /
DEFAULT_OPTIMIZE_CMD / DEFAULT_COMPARE_CMD below now point at THIS repo's
copies of run_script.py / run_mso_pipeline.py by default (both overridable
per-study: settings.python in the YAML, or a task's own `command:` key) --
see MSO_STUDIO_SCIENCE_PYTHON if a different interpreter is needed.

Replaces the one-off scripts (phase_a_driver.py, factorial_dispatch.py, ...):
you describe a study in a YAML file (see TASK_SCHEMA.md and examples/) and this
runner executes it unsupervised for days.

Task types:
  edit_params  — edit dotted-path keys in a YAML-formatted input file
  optimize     — grid/sweep of MUQ-SAC optimizations (run_script.py), one
                 isolated .opt copy per run (never a shared mutable file)
  compare      — mechanism-comparison pipelines (run_mso_pipeline.py), chained
                 per-case off an optimize task or a standalone watch
  report       — LaTeX report per location from ERROR_METRICS CSV sums

Concurrency is FULLY EXPLICIT: every optimize/compare task must state both
max_terminals and parallel_threads; the runner refuses to start if
max_terminals * parallel_threads exceeds settings.system_thread_cap, and at
dispatch time defers launches that would push the live committed-thread total
over the cap. It never adjusts your numbers for you.

Usage:
  run_tasks.py tasks.yaml               run in foreground
  run_tasks.py tasks.yaml --daemon      detach and keep running after logout
  run_tasks.py tasks.yaml --validate    check the YAML and exit (nothing runs)
  run_tasks.py tasks.yaml --status      print progress from the state file
"""
import argparse
import fcntl
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

AUTOMATION_DIR = Path(__file__).resolve().parent
TASK_LOG_DIR = AUTOMATION_DIR / "task_logs"
ALERTS_LOG = AUTOMATION_DIR / "ALERTS.log"

# src/automation/run_tasks.py -> src/automation -> src -> repo root
_REPO_ROOT = AUTOMATION_DIR.parent.parent


def _default_science_python():
    override = os.environ.get("MSO_STUDIO_SCIENCE_PYTHON")
    if override:
        return override
    candidate = Path.home() / "anaconda3" / "bin" / "python3.9"
    return str(candidate) if candidate.exists() else "python3.9"


DEFAULT_PYTHON = _default_science_python()
DEFAULT_OPTIMIZE_CMD = (
    "{python} " + str(_REPO_ROOT / "src" / "MUQ-SAC" / "run_script.py") + " {opt_file}"
)
DEFAULT_COMPARE_CMD = (
    "{python} " + str(_REPO_ROOT / "src" / "run_mso_pipeline.py")
    + " --config MSO_CONFIG.yaml"
)

# Pre-authorized deletion scope (agreed 2026-07-08). Never extend this list.
# Files are only ever deleted from a case folder that already contains that
# task's done_marker (i.e. the optimization producing them has finished), and
# only when the task explicitly sets cleanup_intermediates: true. Matched
# recursively at any depth under the case folder (2026-07-15: these files
# actually live inside run_1/run_2/... subfolders, not the case folder
# itself, so a flat top-level check never matched anything in practice).
CLEANUP_WHITELIST = (
    "Dataset_based_obj",
    "guess_values.txt",
    "guess_values_TRANSFORMED.txt",
    "response_values.txt",
    "zeta_guess_values.txt",
)

# Directory name deleted wholesale (shutil.rmtree) once a compare task's
# done_marker confirms the plot was produced, when that task explicitly sets
# cleanup_sim_runs: true. Opt-in, default off (2026-07-15).
SIM_RUNS_DIRNAME = "SIM_RUNS"


def recursive_delete_whitelisted(root, whitelist, delete=True):
    """Find every file under root (any depth) whose name is exactly one of
    `whitelist`, and unlink it (unless delete=False, for dry-run previews).
    Symlinks are skipped (never followed, never deleted as if they were the
    real file). Returns (freed_bytes, [(path, size), ...]) for files that
    were deleted (or, in dry-run, would have been)."""
    root = Path(root)
    freed = 0
    deleted = []
    for fname in whitelist:
        for fp in root.rglob(fname):
            try:
                if not fp.is_file() or fp.is_symlink():
                    continue
                size = fp.stat().st_size
            except OSError:
                continue
            if delete:
                try:
                    fp.unlink()
                except OSError:
                    continue
            freed += size
            deleted.append((fp, size))
    return freed, deleted


def _dir_size(d):
    total = 0
    for p in d.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def recursive_delete_named_dirs(root, dirname, delete=True):
    """Find every directory named exactly `dirname` anywhere under root (any
    depth) and remove it wholesale (shutil.rmtree), unless delete=False for a
    dry-run preview. Returns (freed_bytes, [(path, size), ...])."""
    if not dirname or dirname in (".", "..") or os.sep in dirname:
        raise ValueError(f"invalid dirname for recursive_delete_named_dirs: {dirname!r}")
    root = Path(root)
    freed = 0
    deleted = []
    for dp in list(root.rglob(dirname)):
        try:
            if not dp.is_dir() or dp.is_symlink():
                continue
            size = _dir_size(dp)
        except OSError:
            continue
        if delete:
            try:
                shutil.rmtree(dp)
            except (FileNotFoundError, OSError):
                continue
        freed += size
        deleted.append((dp, size))
    return freed, deleted

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)

TERMINAL_STATUSES = ("done", "failed", "skipped_existing")


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------------------
# Comment-preserving dotted-path editor for YAML-formatted files
# (target.opt and MSO_CONFIG.yaml are both YAML; we edit line-by-line so that
# comments, ordering and formatting survive untouched.)
# ---------------------------------------------------------------------------
KEYLINE_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_\-]*)(:)(\s*)(.*)$")


def _walk_yaml_lines(lines):
    """Yield (line_index, dotted_path, match) for every 'key: ...' line,
    tracking nesting purely by indentation."""
    stack = []  # list of (indent, key)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        m = KEYLINE_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        yield i, ".".join(k for _, k in stack), m


def _split_value_comment(rest):
    if rest.startswith("#"):
        return "", rest
    idx = rest.find(" #")
    if idx >= 0:
        return rest[:idx], rest[idx:]
    return rest, ""


def yaml_path_edit(file_path, edits, dry_run=False):
    """Apply {dotted.path: new_value} edits to a YAML-formatted file,
    preserving comments/formatting. Returns {path: n_lines_changed}.
    In dry_run mode nothing is written (used by validation)."""
    text = Path(file_path).read_text()
    lines = text.splitlines(keepends=True)
    counts = {p: 0 for p in edits}
    for i, path, m in _walk_yaml_lines(lines):
        if path not in edits:
            continue
        indent, key, colon, space, rest = m.groups()
        eol = "\n" if lines[i].endswith("\n") else ""
        rest = rest.rstrip("\n")
        _, comment = _split_value_comment(rest)
        new_value = str(edits[path])
        sep = space if space else " "
        new_line = f"{indent}{key}{colon}{sep}{new_value}"
        if comment:
            new_line += "  " + comment.strip()
        lines[i] = new_line + eol
        counts[path] += 1
    if not dry_run:
        tmp = Path(str(file_path) + f".tmp.{os.getpid()}")
        tmp.write_text("".join(lines))
        tmp.replace(file_path)
    return counts


def yaml_paths_in_file(file_path):
    lines = Path(file_path).read_text().splitlines()
    return {path for _, path, _ in _walk_yaml_lines(lines)}


# ---------------------------------------------------------------------------
# Config loading / expansion
# ---------------------------------------------------------------------------
def fmt_token(value):
    """Value formatting used in output folder names: dots -> underscores."""
    return str(value).replace(".", "_")


PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")


def fill_template(template, mapping, folder_style=False):
    """Replace {name} placeholders. folder_style formats values with
    dots->underscores (the MUQ-SAC case-folder convention)."""
    def sub(m):
        name = m.group(1)
        if name not in mapping:
            raise ConfigError(f"unknown placeholder {{{name}}} in '{template}'")
        v = mapping[name]
        return fmt_token(v) if folder_style else str(v)
    return PLACEHOLDER_RE.sub(sub, template)


def template_placeholders(template):
    return set(PLACEHOLDER_RE.findall(template or ""))


class Config:
    def __init__(self, yaml_path):
        self.yaml_path = Path(yaml_path).resolve()
        raw = yaml.safe_load(self.yaml_path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError("top level of the task file must be a mapping")
        self.raw = raw
        s = raw.get("settings") or {}
        self.thread_cap = int(s.get("system_thread_cap", 160))
        self.poll_busy = float(s.get("poll_seconds_busy",
                                     60 * float(s.get("poll_minutes_busy", 8))))
        self.poll_idle = float(s.get("poll_seconds_idle",
                                     60 * float(s.get("poll_minutes_idle", 30))))
        self.python = s.get("python", DEFAULT_PYTHON)
        stem = self.yaml_path.stem
        state = s.get("state_file", "auto")
        if state in (None, "auto"):
            self.state_file = AUTOMATION_DIR / f"state_{stem}.json"
        else:
            self.state_file = Path(state)
        self.runner_log = AUTOMATION_DIR / f"runner_{stem}.log"
        self.complete_marker = AUTOMATION_DIR / f"{stem}.complete"

        self.locations = raw.get("locations") or {}
        for name, loc in self.locations.items():
            if not isinstance(loc, dict) or "dir" not in loc:
                raise ConfigError(f"location '{name}' needs at least a 'dir' key")
        self.tasks = raw.get("tasks") or []
        seen = set()
        for t in self.tasks:
            if "id" not in t or "type" not in t:
                raise ConfigError("every task needs 'id' and 'type'")
            if t["id"] in seen:
                raise ConfigError(f"duplicate task id '{t['id']}'")
            seen.add(t["id"])
        self.task_by_id = {t["id"]: t for t in self.tasks}

    def loc(self, name):
        if name not in self.locations:
            raise ConfigError(f"unknown location '{name}'")
        return self.locations[name]

    def loc_dir(self, name):
        return Path(self.loc(name)["dir"])

    def resolve_file(self, spec):
        """'locname/relative/path' or absolute path -> Path."""
        if os.path.isabs(spec):
            return Path(spec)
        head, _, tail = spec.partition("/")
        if head in self.locations and tail:
            return self.loc_dir(head) / tail
        raise ConfigError(
            f"file '{spec}' is neither absolute nor '<location>/<relative>'")

    @staticmethod
    def deps_of(task):
        d = task.get("depends_on")
        if d is None:
            return []
        return d if isinstance(d, list) else [d]


def expand_optimize_runs(cfg, task):
    """Return the full list of runs for an optimize task:
    [{location, folder, params(dict dotted-path -> value)}]."""
    sweep = task.get("sweep") or {}
    linked = task.get("linked") or {}
    fixed = dict(task.get("fixed") or {})
    pt_key = task.get("parallel_threads_key", "Counts.parallel_threads")
    if "parallel_threads" in task:
        fixed[pt_key] = task["parallel_threads"]
    keys = list(sweep.keys())
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*[sweep[k] for k in keys])] or [{}]
    template = task.get("output_folder_template")
    if not template:
        raise ConfigError(f"optimize task '{task['id']}' needs output_folder_template")
    runs = []
    for locname in task.get("locations") or []:
        for combo in combos:
            params = dict(combo)
            for lk, lv in linked.items():
                params[lk] = fill_template(str(lv), params) if isinstance(lv, str) else lv
            params.update(fixed)
            folder = fill_template(template, params, folder_style=True)
            runs.append({"location": locname, "folder": folder, "params": params})
    return runs


# ---------------------------------------------------------------------------
# Validation (also exposed via --validate)
# ---------------------------------------------------------------------------
def validate(cfg):
    errors, warnings = [], []

    def err(msg):
        errors.append(msg)

    for name, loc in cfg.locations.items():
        d = Path(loc["dir"])
        if not d.is_dir():
            err(f"location '{name}': dir does not exist: {d}")
        rc = loc.get("ref_config")
        if rc and not Path(rc).is_file():
            err(f"location '{name}': ref_config does not exist: {rc}")

    ids = set(cfg.task_by_id)
    for t in cfg.tasks:
        for d in Config.deps_of(t):
            if d not in ids:
                err(f"task '{t['id']}': depends_on unknown task '{d}'")

    # dependency cycles (DFS)
    def has_cycle(tid, chain):
        if tid in chain:
            return True
        for d in Config.deps_of(cfg.task_by_id[tid]):
            if d in ids and has_cycle(d, chain | {tid}):
                return True
        return False
    for t in cfg.tasks:
        if has_cycle(t["id"], set()):
            err(f"task '{t['id']}': dependency cycle detected")
            break

    for t in cfg.tasks:
        tid, ttype = t["id"], t["type"]
        if ttype == "edit_params":
            try:
                f = cfg.resolve_file(t.get("file", ""))
                if not f.is_file():
                    err(f"task '{tid}': file not found: {f}")
                else:
                    present = yaml_paths_in_file(f)
                    for p in (t.get("edits") or {}):
                        if p not in present:
                            err(f"task '{tid}': key '{p}' not found in {f}")
            except ConfigError as e:
                err(f"task '{tid}': {e}")
            if not t.get("edits"):
                err(f"task '{tid}': edit_params needs a non-empty 'edits' map")

        elif ttype == "optimize":
            if "max_terminals" not in t or "parallel_threads" not in t:
                err(f"task '{tid}': optimize tasks MUST state both "
                    f"max_terminals and parallel_threads explicitly")
                continue
            product = int(t["max_terminals"]) * int(t["parallel_threads"])
            if product > cfg.thread_cap:
                err(f"task '{tid}': max_terminals x parallel_threads = {product} "
                    f"exceeds system_thread_cap {cfg.thread_cap} — refusing")
            if not t.get("locations"):
                err(f"task '{tid}': optimize needs a 'locations' list")
            try:
                runs = expand_optimize_runs(cfg, t)
                folders = [(r["location"], r["folder"]) for r in runs]
                if len(set(folders)) != len(folders):
                    err(f"task '{tid}': output_folder_template does not produce "
                        f"unique folder names across the sweep — add the missing "
                        f"{{placeholder}}s")
                # every edited key must exist in each location's target.opt
                for locname in t.get("locations") or []:
                    loc = cfg.loc(locname)
                    opt = Path(loc["dir"]) / loc.get("target_opt", "target.opt")
                    if not opt.is_file():
                        err(f"task '{tid}': {locname}: target.opt not found: {opt}")
                        continue
                    present = yaml_paths_in_file(opt)
                    sample = runs[0]["params"] if runs else {}
                    for p in sample:
                        if p not in present:
                            err(f"task '{tid}': key '{p}' not found in {opt}")
                tmpl_ph = template_placeholders(t.get("output_folder_template"))
                sample_keys = set(runs[0]["params"]) if runs else set()
                for ph in tmpl_ph - sample_keys:
                    err(f"task '{tid}': template placeholder '{{{ph}}}' is not a "
                        f"swept/linked/fixed parameter")
            except ConfigError as e:
                err(f"task '{tid}': {e}")

        elif ttype == "compare":
            if "max_terminals" not in t or "parallel_threads" not in t:
                err(f"task '{tid}': compare tasks MUST state both "
                    f"max_terminals and parallel_threads explicitly")
                continue
            product = int(t["max_terminals"]) * int(t["parallel_threads"])
            if product > cfg.thread_cap:
                err(f"task '{tid}': max_terminals x parallel_threads = {product} "
                    f"exceeds system_thread_cap {cfg.thread_cap} — refusing")
            pairs = t.get("pairs") or []
            if not pairs:
                err(f"task '{tid}': compare needs a 'pairs' list of "
                    f"{{from: <loc>, to: <loc>}}")
            for pr in pairs:
                for side in ("from", "to"):
                    if pr.get(side) not in cfg.locations:
                        err(f"task '{tid}': pair {side}: unknown location "
                            f"'{pr.get(side)}'")
                to = pr.get("to")
                if to in cfg.locations:
                    rc = cfg.locations[to].get("ref_config")
                    if not rc:
                        err(f"task '{tid}': location '{to}' needs ref_config "
                            f"(the reference MSO_CONFIG.yaml to copy)")
                    elif Path(rc).is_file():
                        present = yaml_paths_in_file(rc)
                        for p in (t.get("config_edits") or {}):
                            if p not in present:
                                err(f"task '{tid}': key '{p}' not found in {rc}")
            if not Config.deps_of(t) and not t.get("watch"):
                err(f"task '{tid}': compare needs either depends_on (an optimize "
                    f"task) or a watch: block")
            for d in Config.deps_of(t):
                dt = cfg.task_by_id.get(d)
                if dt and dt["type"] not in ("optimize", "edit_params"):
                    err(f"task '{tid}': compare can only depend on optimize/"
                        f"edit_params tasks (got '{d}' of type {dt['type']})")

        elif ttype == "report":
            if not t.get("locations"):
                err(f"task '{tid}': report needs a 'locations' list")
            if not t.get("output_dir"):
                err(f"task '{tid}': report needs an 'output_dir'")
        else:
            err(f"task '{tid}': unknown type '{ttype}'")

    # worst-case overlap warning (informational — per-task caps are the hard
    # rule; the dispatcher additionally defers launches over the live cap)
    total = sum(int(t.get("max_terminals", 0)) * int(t.get("parallel_threads", 0))
                for t in cfg.tasks if t["type"] in ("optimize", "compare"))
    if total > cfg.thread_cap:
        warnings.append(
            f"if ALL optimize/compare tasks ran at full width simultaneously they "
            f"would want {total} threads (> cap {cfg.thread_cap}). The dispatcher "
            f"enforces the cap live by deferring launches, so this is allowed, "
            f"but expect tasks to throttle each other while they overlap.")
    return errors, warnings


# ---------------------------------------------------------------------------
# Process liveness helpers
# ---------------------------------------------------------------------------
def proc_with_cmdline_token(token):
    """PID of any live process whose cmdline contains `token` (used for
    optimize runs: each has a unique .opt filename)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if token.encode() in f.read():
                    return int(pid)
        except OSError:
            continue
    return None


def proc_cmdline_matches(pid, cmd):
    """True if /proc/<pid> exists and its cmdline still contains the launch
    command text (guards against PID recycling by unrelated processes)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode(errors="replace")
        raw = raw.strip()
        if not raw:
            return False  # zombie / kernel thread: empty cmdline
        return cmd in raw or raw in cmd
    except OSError:
        return False


def proc_with_cwd_under(target_dir):
    """PID of any live process whose cwd is target_dir or nested under it.
    Reliable for compare runs (unique target dir per case); see the
    phase_a_driver.py notes for why argv matching is NOT used."""
    target_str = str(target_dir)
    prefix = target_str + os.sep
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if cwd == target_str or cwd.startswith(prefix):
                return int(pid)
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        self.data.setdefault("tasks", {})

    def task(self, tid):
        t = self.data["tasks"].setdefault(tid, {})
        t.setdefault("runs", {})
        return t

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(
                f".tmp.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
            tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = State(cfg.state_file)
        self.stop = False
        self.pools = {}      # compare task id -> ThreadPoolExecutor
        self.futures = {}    # (task id, run key) -> Future
        TASK_LOG_DIR.mkdir(exist_ok=True)

    # -- logging ------------------------------------------------------------
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.cfg.runner_log, "a") as f:
            f.write(line + "\n")

    def alert(self, msg):
        self.log("ALERT: " + msg)
        with open(ALERTS_LOG, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[run_tasks:{self.cfg.yaml_path.name}] {msg}\n")

    # -- thread budget --------------------------------------------------------
    def committed_threads(self):
        total = 0
        for t in self.cfg.tasks:
            if t["type"] not in ("optimize", "compare"):
                continue
            st = self.state.task(t["id"])
            active = sum(1 for r in st["runs"].values()
                         if r.get("status") in ("launched", "running"))
            total += active * int(t.get("parallel_threads", 0))
        return total

    def can_launch(self, task):
        need = int(task["parallel_threads"])
        st = self.state.task(task["id"])
        active = sum(1 for r in st["runs"].values()
                     if r.get("status") in ("launched", "running"))
        if active >= int(task["max_terminals"]):
            return False, "task at max_terminals"
        if self.committed_threads() + need > self.cfg.thread_cap:
            return False, "system thread cap"
        return True, ""

    # -- dependency gating ----------------------------------------------------
    def task_complete(self, tid):
        t = self.cfg.task_by_id[tid]
        st = self.state.task(tid)
        if t["type"] in ("edit_params", "report"):
            return bool(st.get("complete"))
        if t["type"] == "optimize":
            runs = expand_optimize_runs(self.cfg, t)
            if not runs:
                return True
            return all(
                st["runs"].get(self.opt_key(r), {}).get("status") in TERMINAL_STATUSES
                for r in runs)
        if t["type"] == "compare":
            deps = Config.deps_of(t)
            if deps and not all(self.task_complete(d) for d in deps):
                return False
            sources = self.compare_sources(t, only_ready=False)
            expect = (t.get("watch") or {}).get("expect_total")
            if expect is not None and len(sources) < int(expect):
                return False
            if not deps and not sources:
                return False  # standalone watch that has found nothing yet
            return all(
                st["runs"].get(self.cmp_key(s), {}).get("status") in TERMINAL_STATUSES
                for s in sources)
        return False

    def deps_met(self, task):
        return all(self.task_complete(d) for d in Config.deps_of(task))

    # -- edit_params ----------------------------------------------------------
    def do_edit_params(self, task):
        st = self.state.task(task["id"])
        if st.get("complete"):
            return False
        if not self.deps_met(task):
            return False
        f = self.cfg.resolve_file(task["file"])
        counts = yaml_path_edit(f, task["edits"])
        missing = [p for p, n in counts.items() if n == 0]
        if missing:
            st["complete"] = True
            st["status"] = "failed"
            self.alert(f"edit_params '{task['id']}': keys not found in {f}: {missing}")
        else:
            st["complete"] = True
            st["status"] = "done"
            self.log(f"[{task['id']}] edited {f}: "
                     + ", ".join(f"{p} -> {task['edits'][p]}" for p in counts))
        self.state.save()
        return True

    # -- optimize ---------------------------------------------------------------
    @staticmethod
    def opt_key(run):
        return f"{run['location']}::{run['folder']}"

    def opt_token(self, task, run):
        return f"target_auto_{task['id']}_{run['folder']}.opt"

    def do_optimize(self, task):
        """Scan/advance one optimize task. Returns True if anything happened."""
        if not self.deps_met(task):
            return False
        st = self.state.task(task["id"])
        activity = False
        runs = expand_optimize_runs(self.cfg, task)
        done_marker = task.get("done_marker", "new_mech.yaml")

        # 1) status refresh for known runs
        for run in runs:
            k = self.opt_key(run)
            r = st["runs"].get(k)
            if not r or r.get("status") in TERMINAL_STATUSES:
                if (r and r.get("status") in ("done", "skipped_existing")
                        and task.get("cleanup_intermediates")
                        and not r.get("cleanup_done")):
                    self.cleanup_case(
                        self.cfg.loc_dir(run["location"]) / run["folder"],
                        done_marker, task["id"], r)
                continue
            case_dir = self.cfg.loc_dir(run["location"]) / run["folder"]
            marker = case_dir / done_marker
            if marker.exists():
                r["status"] = "done"
                r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.log(f"[{task['id']}/{k}] DONE ({done_marker} found)")
                if task.get("cleanup_intermediates"):
                    self.cleanup_case(case_dir, done_marker, task["id"], r)
                self.state.save()
                activity = True
            elif r.get("status") == "launched":
                if not self.optimize_run_alive(r, case_dir):
                    # re-check marker once: it may have landed between checks
                    if marker.exists():
                        r["status"] = "done"
                        r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        r["status"] = "failed"
                        r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        r["note"] = "process gone, no done_marker"
                        self.alert(f"optimize {task['id']}/{k} FAILED: process "
                                   f"gone without producing {done_marker} "
                                   f"(log: {r.get('log')})")
                    self.state.save()
                    activity = True

        # 2) launch pending runs, round-robin across locations
        per_loc = {}
        for run in runs:
            k = self.opt_key(run)
            r = st["runs"].get(k)
            if r and r.get("status") in ("launched",) + TERMINAL_STATUSES:
                continue
            case_dir = self.cfg.loc_dir(run["location"]) / run["folder"]
            if case_dir.exists() and task.get("skip_if_output_exists", True):
                marker = case_dir / done_marker
                status = "done" if marker.exists() else "skipped_existing"
                st["runs"][k] = {
                    "status": status,
                    "note": "output folder already existed; not launched",
                    "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.log(f"[{task['id']}/{k}] pre-existing output folder -> {status}")
                self.state.save()
                activity = True
                continue
            per_loc.setdefault(run["location"], []).append(run)

        stagger = float(task.get("stagger_seconds", 20))
        batches = itertools.zip_longest(*per_loc.values()) if per_loc else []
        for batch in batches:
            for run in batch:
                if run is None:
                    continue
                ok, why = self.can_launch(task)
                if not ok:
                    if why == "system thread cap":
                        self.log(f"[{task['id']}] deferring launches: live thread "
                                 f"budget would exceed cap {self.cfg.thread_cap}")
                    return activity
                self.launch_optimize(task, run)
                activity = True
                if stagger:
                    time.sleep(stagger)
        return activity

    @staticmethod
    def optimize_run_alive(r, case_dir):
        """Liveness for a launched optimize run, robust to runner restarts and
        custom commands: any of (a) the unique .opt filename appears in a live
        process's cmdline, (b) the recorded pid is alive and its cmdline still
        matches the recorded launch command, (c) a live process has its cwd
        inside the run's case folder."""
        if r.get("token") and proc_with_cmdline_token(r["token"]) is not None:
            return True
        if r.get("pid") and r.get("cmd") and proc_cmdline_matches(r["pid"], r["cmd"]):
            return True
        if case_dir.is_dir() and proc_with_cwd_under(case_dir) is not None:
            return True
        return False

    def launch_optimize(self, task, run):
        st = self.state.task(task["id"])
        k = self.opt_key(run)
        locname = run["location"]
        loc = self.cfg.loc(locname)
        fuel_dir = self.cfg.loc_dir(locname)
        master = fuel_dir / loc.get("target_opt", "target.opt")
        opt_fn = self.opt_token(task, run)
        opt_path = fuel_dir / opt_fn
        # isolated per-run copy: concurrent runs must never share a mutable file
        shutil.copy(master, opt_path)
        counts = yaml_path_edit(opt_path, run["params"])
        missing = [p for p, n in counts.items() if n == 0]
        if missing:
            st["runs"][k] = {"status": "failed",
                             "note": f"keys not found in {opt_path}: {missing}"}
            self.alert(f"optimize {task['id']}/{k}: keys missing in opt file: "
                       f"{missing} — run not launched")
            self.state.save()
            return

        cmd_tmpl = task.get("command", DEFAULT_OPTIMIZE_CMD)
        cmd = fill_template(cmd_tmpl, {
            "python": self.cfg.python, "opt_file": opt_fn,
            "case_dir": str(fuel_dir / run["folder"]),
            "location_dir": str(fuel_dir),
        })
        if task.get("login_shell", True):
            cmd = f"bash -i -c 'exec {cmd}'"
        env = dict(os.environ)
        for var in THREAD_ENV_VARS:
            env[var] = str(task["parallel_threads"])
        log_path = TASK_LOG_DIR / f"{task['id']}__{locname}__{run['folder']}.log"
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(fuel_dir),
                stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        st["runs"][k] = {
            "status": "launched", "pid": proc.pid, "token": opt_fn, "cmd": cmd,
            "params": {p: str(v) for p, v in run["params"].items()},
            "log": str(log_path),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.state.save()
        self.log(f"[{task['id']}/{k}] launched pid={proc.pid} "
                 f"({task['parallel_threads']} threads) log={log_path.name}")

    # -- cleanup (pre-authorized scope only) ------------------------------------
    def cleanup_case(self, case_dir, done_marker, tid, run_entry):
        if not (case_dir / done_marker).exists():
            return
        freed, deleted = recursive_delete_whitelisted(case_dir, CLEANUP_WHITELIST)
        for fp, size in deleted:
            self.log(f"[{tid}] cleanup: deleted {fp} ({size/1e9:.2f} GB)")
        run_entry["cleanup_done"] = True
        run_entry["freed_bytes"] = run_entry.get("freed_bytes", 0) + freed
        self.state.save()

    def cleanup_sim_runs(self, target_dir, tid, run_entry):
        """Opt-in (cleanup_sim_runs: true on a compare task): once a
        comparison's plot is confirmed generated, remove SIM_RUNS/ under its
        target_dir (any depth) — the raw per-condition simulation working
        dirs, superseded by the already-produced PLOTS/ERROR_METRICS."""
        if run_entry.get("sim_runs_cleanup_done"):
            return
        freed, deleted = recursive_delete_named_dirs(target_dir, SIM_RUNS_DIRNAME)
        for dp, size in deleted:
            self.log(f"[{tid}] cleanup: removed {dp} ({size/1e9:.2f} GB)")
        run_entry["sim_runs_cleanup_done"] = True
        run_entry["sim_runs_freed_bytes"] = (
            run_entry.get("sim_runs_freed_bytes", 0) + freed)
        self.state.save()

    # -- compare -----------------------------------------------------------------
    @staticmethod
    def cmp_key(source):
        return f"{source['to']}::{source['case']}"

    def compare_sources(self, task, only_ready=True):
        """Discover source cases for a compare task.
        With depends_on: the dependency optimize task's expected output folders.
        With watch: glob for case folders in each pair's from-location.
        only_ready=True keeps only cases whose source_marker exists."""
        marker_name = task.get("source_marker", "new_mech.yaml")
        sources = []
        pairs = task.get("pairs") or []
        dep_opt_tasks = [self.cfg.task_by_id[d] for d in Config.deps_of(task)
                         if self.cfg.task_by_id[d]["type"] == "optimize"]
        for pr in pairs:
            frm, to = pr["from"], pr["to"]
            from_dir = self.cfg.loc_dir(frm)
            case_names = set()
            if dep_opt_tasks:
                for dt in dep_opt_tasks:
                    dst = self.state.task(dt["id"])
                    for run in expand_optimize_runs(self.cfg, dt):
                        if run["location"] != frm:
                            continue
                        status = dst["runs"].get(self.opt_key(run), {}).get("status")
                        if status == "failed":
                            continue  # failed optimization: no comparison
                        case_names.add(run["folder"])
            watch = task.get("watch")
            if watch or not dep_opt_tasks:
                glob_pat = (watch or {}).get("case_glob", "powerMeanWMRE_*")
                for d in sorted(from_dir.glob(glob_pat)):
                    if d.is_dir():
                        case_names.add(d.name)
            for case in sorted(case_names):
                case_dir = from_dir / case
                ready = (case_dir / marker_name).is_file()
                if only_ready and not ready:
                    continue
                sources.append({"from": frm, "to": to, "case": case,
                                "case_dir": case_dir, "ready": ready})
        return sources

    def ensure_compare_threads_set(self, task):
        """Write parallel_threads ONCE per compare 'to' location, into the
        base target.opt referenced by that location's ref_config. Shared file:
        never edited per-case (2026-07-11 race-class lesson)."""
        st = self.state.task(task["id"])
        done = st.setdefault("threads_set", {})
        for pr in task.get("pairs") or []:
            to = pr["to"]
            if done.get(to):
                continue
            ref = Path(self.cfg.loc(to)["ref_config"])
            base_opt = yaml.safe_load(ref.read_text()).get("base_opt_file")
            key = task.get("parallel_threads_key", "Counts.parallel_threads")
            if base_opt and Path(base_opt).is_file():
                counts = yaml_path_edit(Path(base_opt),
                                        {key: task["parallel_threads"]})
                if counts.get(key):
                    self.log(f"[{task['id']}] set {key}="
                             f"{task['parallel_threads']} in {base_opt}")
                else:
                    self.alert(f"compare '{task['id']}': key '{key}' not found "
                               f"in base_opt_file {base_opt} — thread count NOT "
                               f"set; verify manually")
            else:
                self.alert(f"compare '{task['id']}': base_opt_file missing or "
                           f"not readable from {ref} — thread count NOT set")
            done[to] = True
            self.state.save()

    def do_compare(self, task):
        deps = Config.deps_of(task)
        for d in deps:
            dt = self.cfg.task_by_id[d]
            if dt["type"] == "edit_params" and not self.task_complete(d):
                return False  # hard gate on config edits; optimize deps chain per-case
        st = self.state.task(task["id"])
        self.ensure_compare_threads_set(task)
        pool = self.pools.get(task["id"])
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=int(task["max_terminals"]))
            self.pools[task["id"]] = pool

        self.reconcile_compare_orphans(task)

        activity = False
        for source in self.compare_sources(task, only_ready=True):
            k = self.cmp_key(source)
            r = st["runs"].get(k)
            if r and r.get("status") in ("running",) + TERMINAL_STATUSES:
                continue
            fut = self.futures.get((task["id"], k))
            if fut is not None and not fut.done():
                continue
            ok, why = self.can_launch(task)
            if not ok:
                if why == "system thread cap":
                    self.log(f"[{task['id']}] deferring compare launches: live "
                             f"thread budget would exceed cap {self.cfg.thread_cap}")
                break
            st["runs"][k] = {"status": "running",
                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.state.save()
            self.futures[(task["id"], k)] = pool.submit(
                self.run_compare_case, task, source)
            activity = True
        return activity

    def compare_target_dir(self, task, source):
        st = self.state.task(task["id"])
        k = self.cmp_key(source)
        to_dir = self.cfg.loc_dir(source["to"])
        r = st["runs"].setdefault(k, {})
        if "target_folder" in r:
            return to_dir / r["target_folder"]
        candidate = source["case"]
        if (to_dir / candidate).exists():
            candidate = task.get("collision_prefix", "_") + candidate
        r["target_folder"] = candidate
        return to_dir / candidate

    def run_compare_case(self, task, source):
        st = self.state.task(task["id"])
        k = self.cmp_key(source)
        r = st["runs"][k]
        try:
            target_dir = self.compare_target_dir(task, source)
            target_dir.mkdir(parents=True, exist_ok=True)
            ref = Path(self.cfg.loc(source["to"])["ref_config"])
            dest_cfg = target_dir / "MSO_CONFIG.yaml"
            shutil.copy(ref, dest_cfg)
            mapping = {
                "case_dir": str(source["case_dir"]),
                "case_name": source["case"],
                "from_dir": str(self.cfg.loc_dir(source["from"])),
                "to_dir": str(self.cfg.loc_dir(source["to"])),
                "target_dir": str(target_dir),
            }
            edits = {p: fill_template(str(v), mapping)
                     for p, v in (task.get("config_edits") or {}).items()}
            counts = yaml_path_edit(dest_cfg, edits)
            missing = [p for p, n in counts.items() if n == 0]
            if missing:
                raise RuntimeError(f"keys not found in {dest_cfg}: {missing}")

            cmd = fill_template(task.get("command", DEFAULT_COMPARE_CMD),
                                dict(mapping, python=self.cfg.python))
            if task.get("login_shell", False):
                cmd = f"bash -i -c 'exec {cmd}'"
            log_path = TASK_LOG_DIR / f"{task['id']}__{source['to']}__{source['case']}.log"
            r["log"] = str(log_path)
            self.state.save()
            self.log(f"[{task['id']}/{k}] -> {target_dir.name}: running pipeline")
            with open(log_path, "w") as lf:
                proc = subprocess.run(cmd, shell=True, cwd=str(target_dir),
                                      stdout=lf, stderr=subprocess.STDOUT)
            marker = target_dir / task.get(
                "done_marker", "PLOTS/HTC/nominal_s1/combined_nominal_vs_s1.pdf")
            if proc.returncode == 0 and marker.exists():
                r["status"] = "done"
                self.log(f"[{task['id']}/{k}] pipeline SUCCEEDED")
                if task.get("cleanup_intermediates"):
                    self.cleanup_case(source["case_dir"],
                                      task.get("source_marker", "new_mech.yaml"),
                                      task["id"], r)
                if task.get("cleanup_sim_runs"):
                    self.cleanup_sim_runs(target_dir, task["id"], r)
            else:
                r["status"] = "failed"
                r["note"] = (f"rc={proc.returncode}, "
                             f"marker {'present' if marker.exists() else 'missing'}")
                self.alert(f"compare {task['id']}/{k} FAILED ({r['note']}, "
                           f"log: {log_path})")
            r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.state.save()
        except Exception:
            r["status"] = "failed"
            r["error"] = traceback.format_exc()
            r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.state.save()
            self.alert(f"compare {task['id']}/{k} EXCEPTION:\n{traceback.format_exc()}")

    def reconcile_compare_orphans(self, task):
        """'running' entries with no live future belong to a previous runner
        process. Reconcile via done_marker, else via a live process with cwd
        under the case's target dir; otherwise mark failed."""
        st = self.state.task(task["id"])
        for k, r in list(st["runs"].items()):
            if r.get("status") != "running":
                continue
            fut = self.futures.get((task["id"], k))
            if fut is not None and not fut.done():
                continue
            to, case = k.split("::", 1)
            target_dir = self.cfg.loc_dir(to) / r.get("target_folder", case)
            marker = target_dir / task.get(
                "done_marker", "PLOTS/HTC/nominal_s1/combined_nominal_vs_s1.pdf")
            if marker.exists():
                r["status"] = "done"
                r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.log(f"[{task['id']}/{k}] reconciled as DONE (marker found)")
                if task.get("cleanup_sim_runs"):
                    self.cleanup_sim_runs(target_dir, task["id"], r)
            elif proc_with_cwd_under(target_dir) is None:
                r["status"] = "failed"
                r["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                r["note"] = "orphaned process gone, no done_marker"
                self.alert(f"compare {task['id']}/{k} reconciled as FAILED "
                           f"(orphan gone, no {marker})")
            # else: orphan still working unattended — leave it
        self.state.save()

    # -- report ---------------------------------------------------------------
    def do_report(self, task):
        st = self.state.task(task["id"])
        if st.get("complete"):
            return False
        if not self.deps_met(task):
            return False
        outdir = Path(task["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        for locname in task["locations"]:
            try:
                self.write_report_for_location(task, locname, outdir)
            except Exception:
                self.alert(f"report '{task['id']}' for {locname} EXCEPTION:\n"
                           f"{traceback.format_exc()}")
        st["complete"] = True
        st["status"] = "done"
        self.state.save()
        return True

    def collect_case_objectives(self, case_dir, metrics_glob):
        nominal = s1 = 0.0
        n_files = 0
        for csv_path in sorted(case_dir.glob(metrics_glob)):
            try:
                for line in csv_path.read_text().splitlines():
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    stage, val = parts[0].strip(), parts[1].strip()
                    if stage == "Nominal":
                        nominal += float(val)
                    elif stage in ("Stage-1", "Stage_1", "Stage1"):
                        s1 += float(val)
                n_files += 1
            except (OSError, ValueError):
                continue
        return (nominal, s1, n_files)

    def write_report_for_location(self, task, locname, outdir):
        loc_dir = self.cfg.loc_dir(locname)
        case_glob = task.get("case_glob", "*")
        metrics_glob = task.get(
            "metrics_glob", "ERROR_METRICS/HTC/nominal_s1/phi_*/*.csv")
        rows = []
        for case_dir in sorted(loc_dir.glob(case_glob)):
            if not case_dir.is_dir():
                continue
            nominal, s1, n = self.collect_case_objectives(case_dir, metrics_glob)
            if n == 0:
                continue
            rows.append({"case": case_dir.name, "nominal": nominal,
                         "s1": s1, "n_conditions": n})
        if not rows:
            self.alert(f"report '{task['id']}': no cases with ERROR_METRICS "
                       f"data under {loc_dir} (glob {case_glob})")
            return
        name = re.sub(r"[^A-Za-z0-9_\-]", "_", locname)
        tex_path = outdir / f"report_{name}.tex"
        csv_path = outdir / f"report_{name}.csv"
        with open(csv_path, "w") as f:
            f.write("case_id,folder,nominal_obj,stage1_obj,n_conditions\n")
            for i, row in enumerate(rows, 1):
                f.write(f"C{i},{row['case']},{row['nominal']:.6f},"
                        f"{row['s1']:.6f},{row['n_conditions']}\n")
        tex_path.write_text(self.render_report_tex(task, locname, rows))
        self.log(f"[{task['id']}] wrote {tex_path} and {csv_path} "
                 f"({len(rows)} cases)")

    @staticmethod
    def tex_escape(s):
        return s.replace("\\", r"\textbackslash{}").replace("_", r"\_") \
                .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")

    def render_report_tex(self, task, locname, rows):
        esc = self.tex_escape
        nominals = [r["nominal"] for r in rows]
        nominal_ref = sorted(nominals)[len(nominals) // 2]  # median
        inconsistent = [r["case"] for r in rows
                        if nominal_ref and
                        abs(r["nominal"] - nominal_ref) / nominal_ref > 1e-3]
        best = min(rows, key=lambda r: r["s1"])
        worst = max(rows, key=lambda r: r["s1"])
        total_s1 = sum(r["s1"] for r in rows)
        title = task.get("title", f"Parametric study comparison --- {locname}")
        gen = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        a = lines.append
        a(r"\documentclass[11pt]{article}")
        a(r"\usepackage[margin=2.5cm]{geometry}")
        a(r"\usepackage{booktabs}")
        a(r"\usepackage{longtable}")
        a(r"\begin{document}")
        a(r"\section*{%s}" % esc(title))
        a(r"\noindent Generated %s by run\_tasks.py from "
          r"\texttt{%s}. Objective values are the sum of the "
          r"\texttt{Error\_Function} column over all "
          r"ERROR\_METRICS/HTC/nominal\_s1 condition CSVs of each case." % (gen, esc(self.cfg.yaml_path.name)))
        a("")
        a(r"\subsection*{Case index}")
        a(r"\begin{longtable}{ll}")
        a(r"\toprule")
        a(r"Case ID & Folder \\")
        a(r"\midrule")
        for i, row in enumerate(rows, 1):
            a(r"C%d & \texttt{%s} \\" % (i, esc(row["case"])))
        a(r"\bottomrule")
        a(r"\end{longtable}")
        a("")
        a(r"\subsection*{Objective function values}")
        a(r"\begin{longtable}{lrrr}")
        a(r"\toprule")
        a(r"Case & Objective (sum) & vs.\ nominal & Conditions \\")
        a(r"\midrule")
        a(r"Nominal & %.4f & --- & %d \\" % (nominal_ref, rows[0]["n_conditions"]))
        for i, row in enumerate(rows, 1):
            impr = ((nominal_ref - row["s1"]) / nominal_ref * 100.0
                    if nominal_ref else 0.0)
            a(r"C%d & %.4f & %+.1f\%% & %d \\"
              % (i, row["s1"], impr, row["n_conditions"]))
        a(r"\midrule")
        a(r"Sum over all cases & %.4f & & \\" % total_s1)
        a(r"\bottomrule")
        a(r"\end{longtable}")
        a("")
        a(r"\subsection*{Observations}")
        best_id = "C%d" % (rows.index(best) + 1)
        worst_id = "C%d" % (rows.index(worst) + 1)
        a(r"\begin{itemize}")
        a(r"\item Best case: %s (\texttt{%s}) with objective %.4f "
          r"(%.1f\%% below nominal)."
          % (best_id, esc(best["case"]), best["s1"],
             (nominal_ref - best["s1"]) / nominal_ref * 100.0 if nominal_ref else 0))
        a(r"\item Worst case: %s (\texttt{%s}) with objective %.4f."
          % (worst_id, esc(worst["case"]), worst["s1"]))
        regressions = [r for r in rows if r["s1"] > r["nominal"]]
        if regressions:
            a(r"\item %d case(s) ended WORSE than nominal: %s."
              % (len(regressions),
                 ", ".join(esc(r["case"]) for r in regressions)))
        else:
            a(r"\item Every case improved on the nominal mechanism.")
        if inconsistent:
            a(r"\item WARNING: nominal objective differs by more than 0.1\%% "
              r"across cases (%s) --- check that all cases used the same "
              r"targets/conditions." % ", ".join(esc(c) for c in inconsistent))
        a(r"\end{itemize}")
        a(r"\end{document}")
        return "\n".join(lines) + "\n"

    # -- main loop ---------------------------------------------------------------
    def all_tasks_complete(self):
        return all(self.task_complete(t["id"]) for t in self.cfg.tasks)

    def cycle(self):
        activity = False
        for task in self.cfg.tasks:
            if self.stop:
                break
            try:
                if task["type"] == "edit_params":
                    activity |= self.do_edit_params(task)
                elif task["type"] == "optimize":
                    activity |= self.do_optimize(task)
                elif task["type"] == "compare":
                    activity |= self.do_compare(task)
                elif task["type"] == "report":
                    activity |= self.do_report(task)
            except Exception:
                self.alert(f"task '{task['id']}' cycle EXCEPTION:\n"
                           f"{traceback.format_exc()}")
        return activity

    def anything_active(self):
        if any(not f.done() for f in self.futures.values()):
            return True
        for t in self.cfg.tasks:
            st = self.state.task(t["id"])
            if any(r.get("status") in ("launched", "running")
                   for r in st["runs"].values()):
                return True
        return False

    def run(self):
        self.log(f"=== run_tasks starting: {self.cfg.yaml_path} "
                 f"(state: {self.cfg.state_file.name}) ===")

        def on_signal(signum, frame):
            self.log(f"received signal {signum}; finishing cycle and exiting "
                     f"(launched child processes keep running; restart to resume)")
            self.stop = True
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

        while not self.stop:
            activity = self.cycle()
            if self.all_tasks_complete():
                self.cfg.complete_marker.write_text(
                    time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                self.log(f"ALL TASKS COMPLETE — marker: {self.cfg.complete_marker}")
                break
            sleep_s = self.cfg.poll_busy if (activity or self.anything_active()) \
                else self.cfg.poll_idle
            self.log(f"cycle done (activity={activity}, "
                     f"committed_threads={self.committed_threads()}). "
                     f"Sleeping {sleep_s/60:.1f} min.")
            slept = 0.0
            while slept < sleep_s and not self.stop:
                time.sleep(min(5.0, sleep_s - slept))
                slept += 5.0
        for pool in self.pools.values():
            pool.shutdown(wait=False)
        self.log("=== run_tasks exiting ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_status(cfg):
    if not cfg.state_file.exists():
        print(f"No state file yet ({cfg.state_file}) — nothing has run.")
        return
    data = json.loads(cfg.state_file.read_text())
    print(f"State: {cfg.state_file}")
    for t in cfg.tasks:
        st = data.get("tasks", {}).get(t["id"], {})
        runs = st.get("runs", {})
        if t["type"] in ("edit_params", "report"):
            print(f"  [{t['type']:>11}] {t['id']}: "
                  f"{st.get('status', 'pending')}")
            continue
        counts = {}
        for r in runs.values():
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no runs yet"
        print(f"  [{t['type']:>11}] {t['id']}: {summary}")
        for k, r in sorted(runs.items()):
            if r.get("status") == "failed":
                print(f"        FAILED {k}: {r.get('note', r.get('error', ''))[:120]}")


def daemonize(log_path):
    if os.fork() > 0:
        print(f"runner detached; log: {log_path}")
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDONLY)
    logfd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(devnull, 0)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_file", help="task description YAML")
    ap.add_argument("--validate", action="store_true",
                    help="validate the YAML and exit (nothing is launched)")
    ap.add_argument("--status", action="store_true",
                    help="print progress from the state file and exit")
    ap.add_argument("--daemon", action="store_true",
                    help="detach and keep running after logout")
    args = ap.parse_args()

    try:
        cfg = Config(args.yaml_file)
    except (ConfigError, yaml.YAMLError, OSError) as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2

    if args.status:
        print_status(cfg)
        return 0

    errors, warnings = validate(cfg)
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\nvalidation FAILED with {len(errors)} error(s) — nothing was run.",
              file=sys.stderr)
        return 1
    print("validation OK"
          + (f" ({len(warnings)} warning(s))" if warnings else ""))
    if args.validate:
        for t in cfg.tasks:
            if t["type"] == "optimize":
                runs = expand_optimize_runs(cfg, t)
                print(f"  optimize '{t['id']}': {len(runs)} runs, "
                      f"{t['max_terminals']} terminals x "
                      f"{t['parallel_threads']} threads = "
                      f"{int(t['max_terminals'])*int(t['parallel_threads'])} "
                      f"of {cfg.thread_cap} cap")
            elif t["type"] == "compare":
                print(f"  compare  '{t['id']}': {t['max_terminals']} terminals x "
                      f"{t['parallel_threads']} threads = "
                      f"{int(t['max_terminals'])*int(t['parallel_threads'])} "
                      f"of {cfg.thread_cap} cap")
        return 0

    # single-runner lock per state file
    lock_path = Path(str(cfg.state_file) + ".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another runner already holds {lock_path} — refusing to start "
              f"a second one on the same task file.", file=sys.stderr)
        return 3

    if args.daemon:
        daemonize(cfg.runner_log)
        # re-acquire in the daemon process (fds survive fork, but be explicit)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()

    Runner(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
