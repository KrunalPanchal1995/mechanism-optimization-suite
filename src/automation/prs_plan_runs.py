#!/usr/bin/env python3
"""
prs_plan_runs.py — bounded new-simulation run-plan generator + run_tasks.py
YAML emitter for the post-DoE PRS hyperparameter refinement.

Integrated into the mechanism-optimization-suite repo (2026-07-21) -- see
prs_extract_doe.py's docstring for the MSO_PPRS_DATA_ROOT override
FUEL_LOCATIONS now uses (the fuel optimization/comparison trees are
external, multi-GB data, not siblings of this script anymore).

Every branch below is capped (single round, fixed budget) -- there is
deliberately no re-invocation loop that keeps proposing new points based on
new results, per the "First Way is unboundedly expensive" judgment this
tool implements. Duplicate points already present in the formal DoE (see
prs_extract_doe.py) are dropped for free -- no value re-running something
whose WMRE we already have. Collisions against any OTHER already-completed
folder (e.g. the live/expanded Butadiene sweep) are handled for free by
run_tasks.py's own skip_if_output_exists at dispatch time; this script does
not need to duplicate that check.

Branches:
  plan_verification_runs  -- PRS trusted: predicted optimum + up to 4 flanks. <=5
  plan_local_refinement   -- PRS not trusted: tight grid around empirical
                              best, one round, capped.                        <=9
  plan_mb_repeat          -- MB only: independent repeat of the (4,0.10)
                              outlier point under a distinct folder name.      1
  plan_gens_exploration   -- all fuels: gens in {600,900,1500,1800} at each
                              fuel's empirical-best (p_start,p_final).         4/fuel

write_followup_yaml() emits one run_tasks.py-schema YAML per fuel to
_automation/generated/<fuel>_prs_followup.yaml. It does NOT launch anything;
per house convention (TASK_SCHEMA.md) that YAML must be run through
`run_tasks.py <file> --validate` by a human before `--daemon`.
"""
import argparse
import json
import os
from pathlib import Path

import yaml

from prs_fit import load_doe_csv, DOE_DIR as _DOE_DIR
from prs_optimize import FUEL_BOUNDS, DEFAULT_BOUNDS

AUTOMATION_DIR = Path(__file__).resolve().parent
DOE_DIR = _DOE_DIR
GENERATED_DIR = AUTOMATION_DIR / "generated"
GENS_FIXED = 1200
GENS_EXPLORE = (600, 900, 1500, 1800)

LOCAL_STEP_P_START = 0.5
LOCAL_STEP_P_FINAL = 0.025

VERIFY_FLANK_P_FINAL_STEP = 0.005

CAP_TRUSTED = 5
CAP_LOCAL_REFINEMENT = 9

# Per-fuel locations of the running optimizer's target.opt and (for compare)
# a known-good reference MSO_CONFIG.yaml, matching examples/full_chain_template*.yaml.
# See prs_extract_doe.py's docstring for MSO_PPRS_DATA_ROOT.
P_PRS = Path(os.environ.get("MSO_PPRS_DATA_ROOT", str(AUTOMATION_DIR.parent)))
FUEL_LOCATIONS = {
    "1_3_Butadiene": {
        "opt_dir": str(P_PRS / "1_3_Butadiene" / "HTC"),
        "target_opt": "target.opt",
        "ref_config": str(P_PRS / "MECHANISM_COMPARISON" / "1_3_Butadiene" / "MRE_2"
                           / "GA" / "powerMeanWMRE_1_0_05_1200_GEN_PM_POLISH_0_05"
                           / "MSO_CONFIG.yaml"),
        "cmp_dir": str(P_PRS / "MECHANISM_COMPARISON" / "1_3_Butadiene" / "MRE_2"
                       / "PRS_VERIFY"),
    },
    "n_Heptane_159": {
        "opt_dir": str(P_PRS / "n_Heptane_159" / "HTC"),
        "target_opt": "target.opt",
        "ref_config": str(P_PRS / "MECHANISM_COMPARISON" / "n-Heptane" / "Threshold_0_05"
                           / "MRE_2" / "GA" / "powerMeanWMRE_1_0_05_1200_GEN_PM_POLISH_0_05"
                           / "MSO_CONFIG.yaml"),
        "cmp_dir": str(P_PRS / "MECHANISM_COMPARISON" / "n-Heptane" / "Threshold_0_05"
                       / "MRE_2" / "PRS_VERIFY"),
    },
    "MB": {
        "opt_dir": str(P_PRS / "MB" / "STAGE_1_HTC_D_0_05_CORRELATION_AWARE_6_CASES"),
        "target_opt": "target.opt",
        "ref_config": str(P_PRS / "MECHANISM_COMPARISON" / "MB" / "THRESHOLD_0_05"
                           / "MRE_6_CORRELATED"
                           / "powerMeanWMRE_1_0_05_1200_GEN_PM_POLISH_0_05"
                           / "MSO_CONFIG.yaml"),
        "cmp_dir": str(P_PRS / "MECHANISM_COMPARISON" / "MB" / "THRESHOLD_0_05"
                       / "PRS_VERIFY"),
    },
}


def _clean(x, ndigits=4):
    return round(float(x), ndigits)


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _dedup_against_doe(candidates, doe_points, gens=GENS_FIXED, eps=1e-9):
    tested = {(round(p["p_start"], 6), round(p["p_final"], 6), p["gens"])
              for p in doe_points}
    out = []
    seen = set()
    for c in candidates:
        key = (round(c["p_start"], 6), round(c["p_final"], 6), c.get("gens", gens))
        if key in tested or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def best_point(doe_points):
    return min(doe_points, key=lambda p: p["wmre"])


def plan_verification_runs(fuel, fit, optimum, doe_points):
    """Trusted branch: predicted optimum + up to 4 flanking points."""
    bounds = FUEL_BOUNDS.get(fuel, DEFAULT_BOUNDS)
    (s_lo, s_hi), (f_lo, f_hi) = bounds
    p_start, p_final = optimum.p_start, optimum.p_final
    candidates = [{"p_start": _clean(p_start), "p_final": _clean(p_final),
                   "gens": GENS_FIXED, "reason": "prs_predicted_optimum"}]
    for df in (-VERIFY_FLANK_P_FINAL_STEP, VERIFY_FLANK_P_FINAL_STEP):
        candidates.append({
            "p_start": _clean(p_start),
            "p_final": _clean(_clip(p_final + df, f_lo, f_hi)),
            "gens": GENS_FIXED, "reason": "flank",
        })
    for corner in sorted(doe_points, key=lambda p: (
            (p["p_start"] - p_start) ** 2 + (p["p_final"] - p_final) ** 2))[:2]:
        candidates.append({"p_start": corner["p_start"], "p_final": corner["p_final"],
                            "gens": GENS_FIXED, "reason": "nearest_doe_corner_recheck"})
    candidates = _dedup_against_doe(candidates, doe_points)
    return candidates[:CAP_TRUSTED]


def plan_local_refinement(fuel, doe_points):
    """Not-trusted branch: one bounded round -- a tight grid around the
    current empirical best point, plus (always) the PRS-predicted optimum
    as a triangulation point even though it isn't fully trusted, since it's
    cheap to also check. n-Heptane's bounds intentionally extend below the
    tested p_final floor (see prs_optimize.FUEL_BOUNDS): that extrapolation
    is justified by the raw DoE trend itself, independent of whether the
    global quadratic happened to pass the strict LOOCV gate."""
    bounds = FUEL_BOUNDS.get(fuel, DEFAULT_BOUNDS)
    (s_lo, s_hi), (f_lo, f_hi) = bounds
    best = best_point(doe_points)

    candidates = []
    fit_path = DOE_DIR / f"{fuel}_optimum.json"
    if fit_path.is_file():
        opt = json.loads(fit_path.read_text())
        candidates.append({
            "p_start": _clean(opt["p_start"]), "p_final": _clean(opt["p_final"]),
            "gens": GENS_FIXED, "reason": "prs_predicted_optimum_untrusted_triangulation",
        })

    s_vals = sorted({_clip(_clean(best["p_start"] + d), s_lo, s_hi)
                      for d in (-LOCAL_STEP_P_START, 0.0, LOCAL_STEP_P_START)})
    f_vals = sorted({_clip(_clean(best["p_final"] + d), f_lo, f_hi)
                      for d in (-LOCAL_STEP_P_FINAL, 0.0, LOCAL_STEP_P_FINAL)})
    grid = [{"p_start": s, "p_final": f, "gens": GENS_FIXED, "reason": "local_grid"}
            for s in s_vals for f in f_vals]
    grid.sort(key=lambda p: (p["p_start"] - best["p_start"]) ** 2
                            + (p["p_final"] - best["p_final"]) ** 2)
    candidates.extend(grid)

    candidates = _dedup_against_doe(candidates, doe_points)
    return candidates[:CAP_LOCAL_REFINEMENT]


def plan_mb_repeat(doe_points):
    """MB only: an independent REPEAT of the (p_start=4, p_final=0.10)
    outlier under a distinct folder name, since telling systematic effect
    from stochastic fluke needs a rerun of the SAME point, not a nearby
    one (see report_MB_factorial.tex's own recommendation)."""
    return [{"p_start": 4.0, "p_final": 0.10, "gens": GENS_FIXED,
             "reason": "outlier_confirmation_repeat", "folder_suffix": "_REPEAT"}]


def plan_gens_exploration(fuel, doe_points):
    best = best_point(doe_points)
    return [{"p_start": _clean(best["p_start"]), "p_final": _clean(best["p_final"]),
             "gens": g, "reason": "gens_1d_exploration"}
            for g in GENS_EXPLORE]


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------
def _opt_task(task_id, fuel_key, point, extra_folder_suffix=""):
    p_start, p_final, gens = point["p_start"], point["p_final"], point["gens"]
    folder_suffix = point.get("folder_suffix", "") + extra_folder_suffix
    return {
        "id": task_id,
        "type": "optimize",
        "locations": [f"{fuel_key}_opt"],
        "sweep": {
            "Stats.pm_p_start": [p_start],
            "Stats.pm_p": [p_final],
            "Stats.pm_anneal_gens": [gens],
        },
        "linked": {"Stats.pm_p_polish": "{Stats.pm_p}"},
        "output_folder_template": (
            "powerMeanWMRE_{Stats.pm_p_start}_{Stats.pm_p}_{Stats.pm_anneal_gens}"
            f"_GEN_PM_POLISH_{{Stats.pm_p_polish}}{folder_suffix}"),
        "max_terminals": 1,
        "parallel_threads": 2,
        "stagger_seconds": 20,
        "skip_if_output_exists": True,
        "done_marker": "new_mech.yaml",
        "cleanup_intermediates": True,
        "_reason": point.get("reason", ""),
    }


def build_fuel_plan(fuel, thread_cap=40):
    doe_points = load_doe_csv(DOE_DIR / f"{fuel}.csv")
    fit_data = json.loads((DOE_DIR / f"{fuel}_fit.json").read_text())
    trusted = fit_data["trusted"]

    plan = {"fuel": fuel, "trusted": trusted, "points": [], "branch": None}
    if trusted:
        optimum = json.loads((DOE_DIR / f"{fuel}_optimum.json").read_text())
        from prs_fit import PRSFitResult
        fit = PRSFitResult(fuel=fuel, coeff=fit_data["coeff"],
                            n_points=fit_data["n_points"],
                            loocv_errors_pct=fit_data["loocv_errors_pct"],
                            trusted=trusted, threshold_pct=fit_data["threshold_pct"])

        class _O:
            pass
        o = _O()
        o.p_start, o.p_final = optimum["p_start"], optimum["p_final"]
        points = plan_verification_runs(fuel, fit, o, doe_points)
        plan["branch"] = "verification"
    else:
        points = plan_local_refinement(fuel, doe_points)
        plan["branch"] = "local_refinement"
    plan["points"] = points

    if fuel == "MB":
        plan["mb_repeat"] = plan_mb_repeat(doe_points)
    plan["gens_points"] = plan_gens_exploration(fuel, doe_points)
    return plan


def write_followup_yaml(fuel, plan, out_dir=GENERATED_DIR, thread_cap=40):
    fuel_loc = FUEL_LOCATIONS[fuel]
    out_dir.mkdir(parents=True, exist_ok=True)
    # The compare destination must exist for run_tasks.py's validate() to
    # accept the location; this only creates an empty directory (mirrors
    # phase_a_driver.py preparing MECHANISM_COMPARISON case folders ahead of
    # time), it does not touch or delete anything.
    Path(fuel_loc["cmp_dir"]).mkdir(parents=True, exist_ok=True)

    tasks = []
    opt_ids = []
    for i, pt in enumerate(plan["points"], 1):
        tid = f"opt_{plan['branch']}_{i}"
        tasks.append(_opt_task(tid, fuel, pt))
        opt_ids.append(tid)

    for pt in plan.get("mb_repeat", []):
        tid = "opt_mb_outlier_repeat"
        tasks.append(_opt_task(tid, fuel, pt))
        opt_ids.append(tid)

    if plan["gens_points"]:
        tid = "opt_gens_exploration"
        best_p_final = plan["gens_points"][0]["p_final"]
        tasks.append({
            "id": tid,
            "type": "optimize",
            "locations": [f"{fuel}_opt"],
            "sweep": {"Stats.pm_anneal_gens": list(GENS_EXPLORE)},
            # pm_p_polish can't be a `linked` ref to pm_p here: linked is
            # resolved against swept keys only, before `fixed` is merged in
            # (run_tasks.py expand_optimize_runs), and pm_p is fixed (not
            # swept) in this task -- so both go in `fixed` directly instead.
            "fixed": {
                "Stats.pm_p_start": plan["gens_points"][0]["p_start"],
                "Stats.pm_p": best_p_final,
                "Stats.pm_p_polish": best_p_final,
            },
            "output_folder_template": (
                "powerMeanWMRE_{Stats.pm_p_start}_{Stats.pm_p}_{Stats.pm_anneal_gens}"
                "_GEN_PM_POLISH_{Stats.pm_p_polish}"),
            "max_terminals": 2,
            "parallel_threads": 2,
            "stagger_seconds": 20,
            "skip_if_output_exists": True,
            "done_marker": "new_mech.yaml",
            "cleanup_intermediates": True,
            "_reason": "gens_1d_exploration",
        })
        opt_ids.append(tid)

    cmp_task = {
        "id": "cmp_prs_verify",
        "type": "compare",
        "depends_on": opt_ids,
        "pairs": [{"from": f"{fuel}_opt", "to": f"{fuel}_prs_verify"}],
        "config_edits": {
            "simulations.stage1_htc.mechanism": "{case_dir}/new_mech.yaml",
        },
        "collision_prefix": "_",
        "max_terminals": 2,
        "parallel_threads": 19,
        "done_marker": "PLOTS/HTC/nominal_s1/combined_nominal_vs_s1.pdf",
    }
    report_task = {
        "id": "report_prs_verify",
        "type": "report",
        "depends_on": "cmp_prs_verify",
        "locations": [f"{fuel}_prs_verify"],
        "output_dir": str(P_PRS / "MECHANISM_COMPARISON"),
        "case_glob": "*powerMeanWMRE_*",
    }

    doc = {
        "version": 1,
        "settings": {
            "system_thread_cap": thread_cap,
            "poll_minutes_busy": 8,
            "poll_minutes_idle": 30,
        },
        "locations": {
            f"{fuel}_opt": {"dir": fuel_loc["opt_dir"], "target_opt": fuel_loc["target_opt"]},
            f"{fuel}_prs_verify": {"dir": fuel_loc["cmp_dir"], "ref_config": fuel_loc["ref_config"]},
        },
        "tasks": tasks + [cmp_task, report_task],
    }

    out_path = out_dir / f"{fuel}_prs_followup.yaml"
    header = (
        f"# Auto-generated by prs_plan_runs.py -- branch={plan['branch']} "
        f"(PRS trusted={plan['trusted']}). DO NOT hand-edit without also\n"
        f"# updating prs_doe_data/{fuel}_fit.json accordingly.\n"
        f"# ALWAYS run with --validate first (see TASK_SCHEMA.md):\n"
        f"#   python3.9 run_tasks.py generated/{fuel}_prs_followup.yaml --validate\n"
        f"#   python3.9 run_tasks.py generated/{fuel}_prs_followup.yaml --daemon\n\n"
    )
    out_path.write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    return out_path, len(plan["points"]) + len(plan.get("mb_repeat", [])) + len(plan["gens_points"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", required=True, choices=list(FUEL_LOCATIONS))
    ap.add_argument("--thread-cap", type=int, default=40)
    args = ap.parse_args()

    plan = build_fuel_plan(args.fuel, thread_cap=args.thread_cap)
    out_path, n_new = write_followup_yaml(args.fuel, plan, thread_cap=args.thread_cap)
    print(f"{args.fuel}: branch={plan['branch']} (PRS trusted={plan['trusted']}), "
          f"{n_new} new run(s) planned -> {out_path}")
    for pt in plan["points"]:
        print(f"    {plan['branch']}: p_start={pt['p_start']} p_final={pt['p_final']} "
              f"gens={pt['gens']}  [{pt['reason']}]")
    for pt in plan.get("mb_repeat", []):
        print(f"    mb_repeat: p_start={pt['p_start']} p_final={pt['p_final']} "
              f"gens={pt['gens']}  [{pt['reason']}]")
    for pt in plan["gens_points"]:
        print(f"    gens: p_start={pt['p_start']} p_final={pt['p_final']} "
              f"gens={pt['gens']}  [{pt['reason']}]")


if __name__ == "__main__":
    main()
