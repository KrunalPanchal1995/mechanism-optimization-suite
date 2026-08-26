#!/usr/bin/env python3
"""
prs_driver.py — thin end-to-end wrapper for the PRS-based post-DoE
hyperparameter refinement (mirrors phase_a_driver.py's role as a
convenience wrapper, not a new engine).

Chain per fuel:
  extract DoE (prs_extract_doe) -> fit PRS + LOOCV gate (prs_fit) ->
  optimize the surrogate (prs_optimize, cheap, no simulation) ->
  plan a bounded set of new runs (prs_plan_runs: verification if trusted,
  local refinement if not, always +gens exploration, +MB outlier repeat) ->
  write a run_tasks.py YAML -> preflight (disk + live thread headroom) ->
  print the exact next command for a human to run.

This script NEVER calls `run_tasks.py --daemon` itself, and never deletes
or launches anything. Per this project's standing convention (TASK_SCHEMA.md:
"ALWAYS run --validate first") and the live-process caution noted for this
study, a human reviews the generated YAML and the preflight result before
any new simulation is launched.

Usage:
  python3.9 prs_driver.py --fuel n_Heptane_159
  python3.9 prs_driver.py --all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prs_extract_doe import FUEL_REPORTS, parse_factorial_report, write_csv  # noqa: E402
from prs_fit import fit_and_save  # noqa: E402
from prs_optimize import optimize_prs, FUEL_BOUNDS, DEFAULT_BOUNDS  # noqa: E402
from prs_plan_runs import build_fuel_plan, write_followup_yaml  # noqa: E402
from prs_preflight import check_disk, check_thread_headroom  # noqa: E402

FUELS = list(FUEL_REPORTS)


def run_fuel(fuel, thread_cap=40, min_free_gb=200, cap=160):
    print(f"\n{'=' * 70}\n{fuel}\n{'=' * 70}")

    tex_path = FUEL_REPORTS[fuel]
    rows = parse_factorial_report(tex_path)
    csv_path = write_csv(fuel, rows)
    print(f"[1/6] extracted {len(rows)} DoE points -> {csv_path}")

    fit_result, fit_json_path = fit_and_save(fuel)
    verdict = "TRUSTED" if fit_result.trusted else "NOT TRUSTED"
    print(f"[2/6] PRS fit: LOOCV max|err|={fit_result.loocv_max_abs_pct:.2f}% "
          f"(threshold {fit_result.threshold_pct}%) -> {verdict}")

    bounds = FUEL_BOUNDS.get(fuel, DEFAULT_BOUNDS)
    optimum = optimize_prs(fit_result, bounds, precision=0.001)
    print(f"[3/6] PRS-predicted optimum (informational if not trusted): "
          f"p_start={optimum.p_start} p_final={optimum.p_final} "
          f"predicted={optimum.predicted_wmre:.4f}")
    import json
    (Path(__file__).resolve().parent / "prs_doe_data"
     / f"{fuel}_optimum.json").write_text(json.dumps(optimum.to_json_dict(), indent=2))

    plan = build_fuel_plan(fuel, thread_cap=thread_cap)
    yaml_path, n_new = write_followup_yaml(fuel, plan, thread_cap=thread_cap)
    print(f"[4/6] plan: branch={plan['branch']}, {n_new} new run(s) -> {yaml_path}")

    disk = check_disk(min_free_gb)
    threads = check_thread_headroom(thread_cap, cap=cap)
    print(f"[5/6] preflight: {disk['message']}")
    print(f"             {threads['message']}")

    ok = disk["ok"] and threads["ok"]
    print(f"[6/6] {'READY' if ok else 'BLOCKED'}")
    if ok:
        print(f"  Next steps (human-run, this script does not do this for you):")
        print(f"    python3.9 run_tasks.py generated/{fuel}_prs_followup.yaml --validate")
        print(f"    python3.9 run_tasks.py generated/{fuel}_prs_followup.yaml --daemon")
        print(f"  Then once done: python3.9 prs_report.py --fuel {fuel}")
    else:
        print(f"  Not printing a launch command: preflight failed, see above. "
              f"Re-run prs_driver.py once resolved.")
    return {"fuel": fuel, "branch": plan["branch"], "trusted": fit_result.trusted,
            "n_new_runs": n_new, "yaml": str(yaml_path), "preflight_ok": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", choices=FUELS, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--thread-cap", type=int, default=40,
                     help="system_thread_cap written into each generated YAML "
                          "(default 40, conservative headroom under any other "
                          "live run_tasks.py process)")
    ap.add_argument("--min-free-gb", type=int, default=200)
    ap.add_argument("--cap", type=int, default=160)
    args = ap.parse_args()

    if not args.fuel and not args.all:
        print("specify --fuel <name> or --all", file=sys.stderr)
        return 2
    fuels = FUELS if args.all else [args.fuel]

    results = [run_fuel(f, thread_cap=args.thread_cap,
                         min_free_gb=args.min_free_gb, cap=args.cap)
               for f in fuels]

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    total_new = sum(r["n_new_runs"] for r in results)
    for r in results:
        print(f"  {r['fuel']}: branch={r['branch']} trusted={r['trusted']} "
              f"new_runs={r['n_new_runs']} preflight={'OK' if r['preflight_ok'] else 'BLOCKED'}")
    print(f"  TOTAL new runs planned across {len(results)} fuel(s): {total_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
