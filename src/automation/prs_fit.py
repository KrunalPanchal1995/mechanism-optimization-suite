#!/usr/bin/env python3
"""
prs_fit.py — 2nd-order Polynomial Response Surface (PRS) fit of the existing
DoE data (p_start, p_final) -> WMRE, with a leave-one-out cross-validation
(LOOCV) trust gate.

Mirrors the house methodology already used elsewhere in this pipeline for
the (lower-level) mechanism A-factor response surface
(/home/krithika/MUQ-SAC/V3.0/ResponseSurface.py: quadratic basis
[1, x1, x2, x1^2, x1*x2, x2^2], QR-based least squares, and a hard 6%
max-relative-error reject threshold, ResponseSurface.test()'s
`criterion = 6`) -- reimplemented standalone here (not imported) because
that module is tightly coupled to its own case-index/file-path/
StatisticalAnalysis side effects and isn't a clean fit for a 15-point,
2-variable meta-level fit.

Both parameters are normalized to [0, 1] before fitting (p_start via
(x-1)/4, p_final via (x-0.05)/0.10) -- without this the ~30x scale mismatch
between the two raw ranges ill-conditions the QR solve.

Usage (library):
    from prs_fit import fit_prs, loocv_relative_errors, PRSFitResult
    points = load_doe_csv("prs_doe_data/n_Heptane_159.csv")
    result = fit_prs(points)
    print(result.trusted, result.loocv_max_abs_pct)

Usage (CLI, prints a fit summary per fuel):
    python3.9 prs_fit.py --fuel n_Heptane_159
"""
import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AUTOMATION_DIR = Path(__file__).resolve().parent
DOE_DIR = AUTOMATION_DIR / "prs_doe_data"

TRUST_THRESHOLD_PCT = 6.0  # same numeric convention as ResponseSurface.py

# Normalization ranges: the tested DoE box. Fits/evaluates are done in
# normalized [0,1]^2 coordinates; predict()/evaluate() take raw units.
P_START_RANGE = (1.0, 5.0)
P_FINAL_RANGE = (0.05, 0.15)


def _normalize(p_start, p_final):
    lo1, hi1 = P_START_RANGE
    lo2, hi2 = P_FINAL_RANGE
    return (p_start - lo1) / (hi1 - lo1), (p_final - lo2) / (hi2 - lo2)


def load_doe_csv(path):
    points = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            points.append({
                "p_start": float(row["p_start"]),
                "p_final": float(row["p_final"]),
                "gens": int(float(row["gens"])),
                "wmre": float(row["wmre"]),
                "source": row.get("source", ""),
            })
    return points


def _design_row(x1, x2):
    """Quadratic basis [1, x1, x2, x1^2, x1*x2, x2^2] -- same term set as
    ResponseSurface.py's order=2 MatPolyFitTransform for 2 variables."""
    return [1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2]


def _design_matrix(points):
    rows = []
    for p in points:
        x1, x2 = _normalize(p["p_start"], p["p_final"])
        rows.append(_design_row(x1, x2))
    return np.array(rows)


def _fit_coeffs(points):
    A = _design_matrix(points)
    y = np.array([p["wmre"] for p in points])
    Q, R = np.linalg.qr(A)
    coeff = np.linalg.solve(R, Q.T @ y)
    return coeff


def _predict_normalized(coeff, x1, x2):
    row = np.array(_design_row(x1, x2))
    return float(row @ coeff)


@dataclass
class PRSFitResult:
    fuel: str
    coeff: list
    n_points: int
    loocv_errors_pct: list  # signed relative % error per held-out point
    trusted: bool
    threshold_pct: float = TRUST_THRESHOLD_PCT

    @property
    def loocv_max_abs_pct(self):
        return max(abs(e) for e in self.loocv_errors_pct) if self.loocv_errors_pct else float("nan")

    @property
    def loocv_mean_abs_pct(self):
        return (sum(abs(e) for e in self.loocv_errors_pct) / len(self.loocv_errors_pct)
                if self.loocv_errors_pct else float("nan"))

    def predict(self, p_start, p_final):
        x1, x2 = _normalize(p_start, p_final)
        return _predict_normalized(np.array(self.coeff), x1, x2)

    def to_json_dict(self):
        return {
            "fuel": self.fuel,
            "coeff": list(self.coeff),
            "n_points": self.n_points,
            "loocv_errors_pct": self.loocv_errors_pct,
            "loocv_max_abs_pct": self.loocv_max_abs_pct,
            "loocv_mean_abs_pct": self.loocv_mean_abs_pct,
            "trusted": self.trusted,
            "threshold_pct": self.threshold_pct,
            "p_start_range": list(P_START_RANGE),
            "p_final_range": list(P_FINAL_RANGE),
        }


def loocv_relative_errors(points):
    """Leave-one-out over `points`: for each held-out point, refit on the
    rest and return (predicted - actual) / actual * 100. With 15 points and
    6 coefficients, each fold solves on 14 points -- always overdetermined."""
    errors = []
    n = len(points)
    if n <= 6:
        raise ValueError(f"need > 6 points for a 6-coefficient quadratic LOOCV, got {n}")
    for i in range(n):
        train = points[:i] + points[i + 1:]
        held = points[i]
        coeff = _fit_coeffs(train)
        x1, x2 = _normalize(held["p_start"], held["p_final"])
        pred = _predict_normalized(coeff, x1, x2)
        actual = held["wmre"]
        errors.append((pred - actual) / actual * 100.0 if actual else float("nan"))
    return errors


def fit_prs(points, fuel="unknown"):
    coeff = _fit_coeffs(points)
    loocv_errors = loocv_relative_errors(points)
    trusted = max(abs(e) for e in loocv_errors) <= TRUST_THRESHOLD_PCT
    return PRSFitResult(
        fuel=fuel, coeff=list(coeff), n_points=len(points),
        loocv_errors_pct=loocv_errors, trusted=trusted,
    )


def fit_and_save(fuel, doe_dir=DOE_DIR):
    csv_path = doe_dir / f"{fuel}.csv"
    points = load_doe_csv(csv_path)
    result = fit_prs(points, fuel=fuel)
    out_path = doe_dir / f"{fuel}_fit.json"
    out_path.write_text(json.dumps(result.to_json_dict(), indent=2))
    return result, out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--doe-dir", default=str(DOE_DIR))
    args = ap.parse_args()

    result, out_path = fit_and_save(args.fuel, Path(args.doe_dir))
    verdict = "TRUSTED" if result.trusted else "NOT TRUSTED"
    print(f"{args.fuel}: {result.n_points} points, "
          f"LOOCV max|err|={result.loocv_max_abs_pct:.2f}% "
          f"mean|err|={result.loocv_mean_abs_pct:.2f}% "
          f"(threshold {result.threshold_pct}%) -> {verdict}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
