#!/usr/bin/env python3
"""
prs_optimize.py — cheap (no-simulation) optimization of a fitted PRS surface
(prs_fit.PRSFitResult) to a target decimal precision.

Per-fuel bounds are deliberately NOT symmetric/generic: n-Heptane's DoE data
showed a clean monotonic trend bottoming out at the tested floor
(p_final=0.05), so its p_final bound is allowed to extend slightly below the
tested range (down to 0.02) -- a bounded extrapolation justified by the
trend, not an open-ended one. Every other fuel/dimension is clamped to the
tested DoE box, since nothing in the data justifies extrapolating there
(1,3-Butadiene's optimum is a sharp interior corner; MB is flat-with-outlier).

This module only ever touches the cheap fitted polynomial -- it launches no
simulations and costs no wall-clock time beyond a local optimizer call.

Usage (library):
    from prs_optimize import optimize_prs, FUEL_BOUNDS
    optimum = optimize_prs(fit_result, FUEL_BOUNDS["n_Heptane_159"], precision=0.001)
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

AUTOMATION_DIR = Path(__file__).resolve().parent
DOE_DIR = AUTOMATION_DIR / "prs_doe_data"

# (p_start_bounds, p_final_bounds). Default: the tested DoE box.
DEFAULT_BOUNDS = ((1.0, 5.0), (0.05, 0.15))
FUEL_BOUNDS = {
    "1_3_Butadiene": DEFAULT_BOUNDS,
    "n_Heptane_159": ((1.0, 5.0), (0.02, 0.15)),  # bounded extrapolation below the tested floor
    "MB": DEFAULT_BOUNDS,
}


@dataclass
class PRSOptimum:
    fuel: str
    p_start: float
    p_final: float
    predicted_wmre: float
    precision: float
    grid_agrees: bool
    grid_p_start: float
    grid_p_final: float
    grid_predicted_wmre: float

    def to_json_dict(self):
        return dict(self.__dict__)


def _round_to_precision(x, precision):
    if precision <= 0:
        return x
    return round(round(x / precision) * precision, 10)


def _brute_force_grid(fit, bounds, precision):
    (s_lo, s_hi), (f_lo, f_hi) = bounds
    s_vals = np.arange(s_lo, s_hi + precision / 2, precision)
    f_vals = np.arange(f_lo, f_hi + precision / 2, precision)
    best = None
    for s in s_vals:
        for f in f_vals:
            val = fit.predict(s, f)
            if best is None or val < best[2]:
                best = (s, f, val)
    return best


def optimize_prs(fit, bounds, precision=0.001, n_multistart=9):
    (s_lo, s_hi), (f_lo, f_hi) = bounds
    scipy_bounds = [(s_lo, s_hi), (f_lo, f_hi)]

    rng = np.random.default_rng(42)
    seeds = [
        (s_lo, f_lo), (s_hi, f_lo), (s_lo, f_hi), (s_hi, f_hi),
        ((s_lo + s_hi) / 2, (f_lo + f_hi) / 2),
    ]
    while len(seeds) < n_multistart:
        seeds.append((rng.uniform(s_lo, s_hi), rng.uniform(f_lo, f_hi)))

    def obj(x):
        return fit.predict(x[0], x[1])

    best_grad = None
    for seed in seeds:
        res = minimize(obj, x0=np.array(seed), method="L-BFGS-B", bounds=scipy_bounds)
        if best_grad is None or res.fun < best_grad.fun:
            best_grad = res

    grid_s, grid_f, grid_val = _brute_force_grid(fit, bounds, precision)

    p_start = _round_to_precision(float(best_grad.x[0]), precision)
    p_final = _round_to_precision(float(best_grad.x[1]), precision)
    predicted = fit.predict(p_start, p_final)

    grid_agrees = bool(abs(p_start - grid_s) <= precision * 1.5
                       and abs(p_final - grid_f) <= precision * 1.5)

    return PRSOptimum(
        fuel=fit.fuel, p_start=p_start, p_final=p_final,
        predicted_wmre=predicted, precision=precision,
        grid_agrees=grid_agrees,
        grid_p_start=float(grid_s), grid_p_final=float(grid_f),
        grid_predicted_wmre=float(grid_val),
    )


def main():
    from prs_fit import PRSFitResult

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--precision", type=float, default=0.001)
    ap.add_argument("--doe-dir", default=str(DOE_DIR))
    args = ap.parse_args()

    fit_path = Path(args.doe_dir) / f"{args.fuel}_fit.json"
    data = json.loads(fit_path.read_text())
    fit = PRSFitResult(
        fuel=data["fuel"], coeff=data["coeff"], n_points=data["n_points"],
        loocv_errors_pct=data["loocv_errors_pct"], trusted=data["trusted"],
        threshold_pct=data["threshold_pct"],
    )
    bounds = FUEL_BOUNDS.get(args.fuel, DEFAULT_BOUNDS)
    optimum = optimize_prs(fit, bounds, precision=args.precision)

    verdict = "TRUSTED" if fit.trusted else "NOT TRUSTED (informational only)"
    print(f"{args.fuel} [{verdict}]: gradient optimum "
          f"p_start={optimum.p_start} p_final={optimum.p_final} "
          f"predicted WMRE={optimum.predicted_wmre:.4f}%")
    print(f"  grid cross-check: p_start={optimum.grid_p_start:.4f} "
          f"p_final={optimum.grid_p_final:.4f} "
          f"predicted WMRE={optimum.grid_predicted_wmre:.4f}% "
          f"(agrees={optimum.grid_agrees})")

    out_path = Path(args.doe_dir) / f"{args.fuel}_optimum.json"
    out_path.write_text(json.dumps(optimum.to_json_dict(), indent=2))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
