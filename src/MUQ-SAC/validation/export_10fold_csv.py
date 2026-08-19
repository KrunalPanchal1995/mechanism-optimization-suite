"""
Exports the 10-fold CV results (results_dir("tenfold")/01_baseline_10fold/
and .../07_final_10fold/) to tidy CSVs, so report_figures/plot_10fold_
evolution.py never needs to touch the raw JSON -- data and plotting are
fully separated, per the 2026-07-31 request to be able to regenerate/modify
the plots from the CSVs alone.

Writes to results_dir("tenfold")/csv/:
  10fold_per_fold_before.csv  -- case, fold, train_pct, test_pct, worst_pct
                                  (= max(train_pct, test_pct), the actual
                                  pass/fail metric), tolerance, k, order,
                                  ridge_alpha, pass_fail
  10fold_per_fold_after.csv   -- same, band-valid-only test_pct/worst_pct
  10fold_case_summary.csv     -- one row per case: worst_fold_before_pct,
                                  worst_fold_after_pct, n_folds_failed_before,
                                  n_folds_failed_after

Run: python3.9 export_10fold_csv.py [--before-json PATH] [--after-json PATH]
                                     [--output-dir DIR]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # validation/
import config

import csv
import json

CRITERION = config.CRITERION


def fmt_tol(tol):
    return "inf" if tol == float("inf") else f"{tol:g}"


def export_before(path_json, out_dir):
    data = json.load(open(path_json))["raw"]
    rows = []
    for r in sorted(data, key=lambda r: r["case"]):
        hp = r["hp"]
        for f in r["folds"]:
            rows.append({
                "case": r["case"], "fold": f["fold"],
                "train_pct": f["train_pct"], "test_pct": f["test_pct"],
                "worst_pct": f["worst_pct"],  # = max(train_pct, test_pct); the official
                                               # per-fold pass/fail metric (report's own
                                               # "robust" definition), NOT test_pct alone
                "tolerance": fmt_tol(hp["tolerance"]), "k": hp["k"],
                "order": hp["order"], "ridge_alpha": hp["ridge_alpha"],
                "pass_fail": "PASS" if f["worst_pct"] <= CRITERION else "FAIL",
            })
    out = os.path.join(out_dir, "10fold_per_fold_before.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return data


def export_after(path_json, out_dir):
    data = json.load(open(path_json))["raw"]
    rows = []
    for r in sorted(data, key=lambda r: r["case"]):
        hp = r["hp"]
        for f in r["folds"]:
            rows.append({
                "case": r["case"], "fold": f["fold"],
                "train_pct": f["train_pct"], "test_pct": f["test_pct_valid"],
                "test_pct_all_points": f["test_pct_all"],
                "worst_pct": f["worst_pct_valid"],  # = max(train_pct, test_pct_valid),
                                                      # matches tables/final_10fold.tex exactly
                "tolerance": fmt_tol(hp["tolerance"]), "k": hp["k"],
                "order": hp["order"], "ridge_alpha": hp["ridge_alpha"],
                "pass_fail": "PASS" if f["worst_pct_valid"] <= CRITERION else "FAIL",
            })
    out = os.path.join(out_dir, "10fold_per_fold_after.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return data


def export_summary(before_data, after_data, out_dir):
    before_by_case = {r["case"]: r for r in before_data}
    after_by_case = {r["case"]: r for r in after_data}
    rows = []
    for case in sorted(before_by_case):
        b = before_by_case[case]["folds"]
        a = after_by_case[case]["folds"]
        worst_before = max(f["worst_pct"] for f in b)
        worst_after = max(f["worst_pct_valid"] for f in a)
        n_fail_before = sum(1 for f in b if f["worst_pct"] > CRITERION)
        n_fail_after = sum(1 for f in a if f["worst_pct_valid"] > CRITERION)
        rows.append({
            "case": case, "worst_fold_before_pct": round(worst_before, 3),
            "worst_fold_after_pct": round(worst_after, 3),
            "n_folds_failed_before": n_fail_before, "n_folds_failed_after": n_fail_after,
            "n_folds_total": len(b),
        })
    out = os.path.join(out_dir, "10fold_case_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    results_dir = config.results_dir("tenfold")
    ap.add_argument("--before-json", default=os.path.join(results_dir, "01_baseline_10fold", "kfold_cv_results.json"))
    ap.add_argument("--after-json", default=os.path.join(results_dir, "07_final_10fold", "kfold_cv_bandvalid_results.json"))
    ap.add_argument("--output-dir", default=os.path.join(results_dir, "csv"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    before_data = export_before(args.before_json, args.output_dir)
    after_data = export_after(args.after_json, args.output_dir)
    export_summary(before_data, after_data, args.output_dir)


if __name__ == "__main__":
    main()
