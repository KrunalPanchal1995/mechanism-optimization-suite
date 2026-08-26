#!/usr/bin/env python3
"""
prs_preflight.py — read-only checks run before printing the "go ahead and
launch" instructions for a generated PRS follow-up YAML. Never launches or
deletes anything itself.

check_disk: refuses if /data2 free space is below a threshold. Points at
cleanup_space.py (dry-run) rather than deleting anything itself -- no new
deletion scope, per this project's standing rule that the pre-authorized
whitelist (run_tasks.py.CLEANUP_WHITELIST) is never extended.

check_thread_headroom: this project's thread budget is tracked in
parallel_threads units (OMP/MKL env vars), not raw OS thread counts, and
run_tasks.py.Runner.committed_threads() already implements exactly that
accounting per task file. There is no cross-process registry, so this scans
every state_*.json under one or more automation directories (one state file
per ever-run YAML) and sums each one's committed_threads() the same way its
own Runner would, giving an accurate live total without touching that other
process's state.

Integrated into the mechanism-optimization-suite repo (2026-07-21) alongside
the original external _automation/ project, which keeps running its own
live, long-lived studies there independently. A NEW study launched from
this repo's copy of run_tasks.py would, by default, only see state files in
ITS OWN directory -- silently blind to whatever the external project has
already committed -- so this scan now covers multiple automation
directories: this script's own directory, plus every path in
MSO_STUDIO_EXTRA_AUTOMATION_DIRS (os.pathsep-separated), plus (if that env
var is unset and the path exists on this machine) the known external p_PRS
_automation/ directory as a same-machine safety default. Set
MSO_STUDIO_EXTRA_AUTOMATION_DIRS="" explicitly to scan only this directory.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tasks import Config, State  # noqa: E402

AUTOMATION_DIR = Path(__file__).resolve().parent
DATA2_ROOT = Path("/data2")

# Same-machine safety default: the external project this repo's automation
# code was integrated from (2026-07-21) keeps running independently there.
# Only used when MSO_STUDIO_EXTRA_AUTOMATION_DIRS is unset AND this path
# still exists on this machine -- harmless (silently skipped) elsewhere.
_KNOWN_EXTERNAL_AUTOMATION_DIR = Path(
    "/data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/MSO_A_FACTOR/p_PRS/_automation"
)


def automation_dirs():
    """Every automation directory whose state_*.json files should count
    toward the live thread total -- see the module docstring."""
    dirs = [AUTOMATION_DIR]
    override = os.environ.get("MSO_STUDIO_EXTRA_AUTOMATION_DIRS")
    if override is not None:
        dirs += [Path(p) for p in override.split(os.pathsep) if p]
    elif _KNOWN_EXTERNAL_AUTOMATION_DIR.is_dir():
        dirs.append(_KNOWN_EXTERNAL_AUTOMATION_DIR)

    seen = set()
    unique = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            unique.append(d)
    return unique


def check_disk(min_free_gb=200, root=DATA2_ROOT):
    usage = shutil.disk_usage(root)
    free_gb = usage.free / 1e9
    ok = free_gb >= min_free_gb
    return {
        "ok": ok,
        "free_gb": round(free_gb, 1),
        "min_free_gb": min_free_gb,
        "used_pct": round(usage.used / usage.total * 100, 1),
        "message": (
            f"{free_gb:.1f} GB free ({usage.used / usage.total * 100:.1f}% used) "
            f"{'>= ' if ok else '< '}{min_free_gb} GB threshold"
            + ("" if ok else
               "\n  REFUSING: run a dry-run cleanup preview first, e.g.\n"
               "    python3.9 cleanup_space.py files <fuel HTC dirs...> "
               "--require-marker new_mech.yaml\n"
               "  (add --delete only after reviewing the dry-run output; "
               "this never extends the pre-authorized 5-file whitelist)")
        ),
    }


def _find_yaml_for_state(state_path, search_dir):
    stem = state_path.stem
    if stem.startswith("state_"):
        stem = stem[len("state_"):]
    for candidate in search_dir.rglob(f"{stem}.yaml"):
        return candidate
    return None


def live_committed_threads(automation_dirs_=None, exclude_state=None):
    """Sum committed parallel_threads across every state_*.json found in
    :func:`automation_dirs` (or an explicit list passed in *automation_dirs_*).
    """
    dirs = automation_dirs_ if automation_dirs_ is not None else automation_dirs()
    total = 0
    details = []
    for search_dir in dirs:
        for state_path in sorted(search_dir.glob("state_*.json")):
            if exclude_state and state_path.resolve() == Path(exclude_state).resolve():
                continue
            yaml_path = _find_yaml_for_state(state_path, search_dir)
            if yaml_path is None:
                continue  # stale state file, its YAML was moved/removed; skip
            try:
                cfg = Config(yaml_path)
                state = State(cfg.state_file)
                committed = 0
                for t in cfg.tasks:
                    if t["type"] not in ("optimize", "compare"):
                        continue
                    st = state.task(t["id"])
                    active = sum(1 for r in st["runs"].values()
                                 if r.get("status") in ("launched", "running"))
                    committed += active * int(t.get("parallel_threads", 0))
            except Exception as e:  # noqa: BLE001 -- a bad/foreign state file must not abort preflight
                details.append({"yaml": str(yaml_path), "error": str(e)})
                continue
            if committed:
                total += committed
                details.append({"yaml": str(yaml_path), "committed_threads": committed})
    return total, details


def check_thread_headroom(planned_max_threads, cap=160, automation_dirs_=None):
    dirs = automation_dirs_ if automation_dirs_ is not None else automation_dirs()
    live_total, details = live_committed_threads(dirs)
    headroom = cap - live_total
    ok = planned_max_threads <= headroom
    return {
        "ok": ok,
        "live_committed_threads": live_total,
        "cap": cap,
        "headroom": headroom,
        "planned_max_threads": planned_max_threads,
        "scanned_dirs": [str(d) for d in dirs],
        "details": details,
        "message": (
            f"live committed threads across {len(dirs)} automation "
            f"dir(s) ({', '.join(str(d) for d in dirs)}): "
            f"{live_total} of {cap} cap (headroom {headroom}); this plan's "
            f"worst-case want is {planned_max_threads} -> "
            f"{'OK' if ok else 'REFUSING: would exceed the cap'}"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-free-gb", type=int, default=200)
    ap.add_argument("--planned-max-threads", type=int, default=40)
    ap.add_argument("--cap", type=int, default=160)
    args = ap.parse_args()

    disk = check_disk(args.min_free_gb)
    threads = check_thread_headroom(args.planned_max_threads, args.cap)
    print(f"disk:    {disk['message']}")
    print(f"threads: {threads['message']}")
    for d in threads["details"]:
        print(f"    {d}")
    ok = disk["ok"] and threads["ok"]
    print("\nPREFLIGHT " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
