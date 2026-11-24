#python default modules
import json
import hashlib
import pickle
import yaml 
import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing
import matplotlib as mpl
import concurrent.futures
import parallel_yaml_writer
import scipy.stats as stats
import matplotlib.pyplot as plt
import partial_PRS_system as P_PRS
from scipy.optimize import minimize
from collections import OrderedDict
from scipy.linalg import block_diag
from scipy import optimize as spopt
mpl.rc('figure', max_open_warning = 0)
import os, sys, re, threading, subprocess, time, traceback, shutil, glob, select
from sklearn.model_selection import train_test_split
sys.path.append('/parallel_yaml_writer.so')
from scipy.interpolate import Akima1DInterpolator
####################################
##  Importing the sampling file   ##
##                                ##
####################################
#program specific modules
import data_management
from copy import deepcopy
import DesignMatrix as DM
from mpire import WorkerPool
import ResponseSurface as PRS
import AdaptiveResampler as AR
import IdentifiablePRS as IPRS
import combustion_target_class
import Uncertainty as uncertainty
import correlation_analysis as CA
from MechanismParser import Parser
import simulation_manager2_0 as simulator
from MechManipulator2_0 import Manipulator
from Target2CSV import Exporter as Target2CSV
from VisualAid import TPhiPlotter, ArrheniusPlotter
from OptimizationTool import OptimizationTool as Optimizer


####################################################################################
##  Structured, phase-aware terminal output (2026-08-19)                        ##
##                                                                                ##
##  Replaces ~59 mutually-inconsistent print() idioms (hash banners, bare        ##
##  \n-padded lines, [Case-N]/\t[Case-N]/\t[warn]/[TAG] prefixes, mixed          ##
##  .format()/f-string/concatenation) with one small, dependency-free set of     ##
##  helpers. Falls back to plain ASCII automatically if stdout can't encode      ##
##  box-drawing glyphs (output is normally redirected to run_log.txt, so this    ##
##  must never crash on encoding). Purely presentational -- no numeric or        ##
##  control-flow change lives in this block.                                     ##
####################################################################################
import datetime as _dt

def _supports_unicode():
	enc = (getattr(sys.stdout, "encoding", None) or "").lower()
	try:
		"═─│✓✗⚠→".encode(enc or "utf-8")
		return bool(enc) and "utf" in enc
	except Exception:
		return False

_UNICODE = _supports_unicode()
_GLYPH = {
	"heavy":  "═" if _UNICODE else "=",
	"light":  "─" if _UNICODE else "-",
	"ok":     "✓" if _UNICODE else "[OK]",
	"fail":   "✗" if _UNICODE else "[FAIL]",
	"warn":   "⚠" if _UNICODE else "[WARN]",
	"arrow":  "→" if _UNICODE else "->",
}
_PHASE_T0 = {}

def _ts():
	return _dt.datetime.now().strftime("%H:%M:%S")

def phase(n, total, title):
	"""Numbered heavy-rule banner marking one of run_script.py's 11 pipeline
	phases (input parse, targets, UQ, reaction dict, sensitivity analysis,
	correlation analysis, design matrix, simulation, response extraction,
	per-case PRS loop, optimization)."""
	bar = _GLYPH["heavy"] * 78
	print(f"\n{bar}")
	print(f"  PHASE {n}/{total} {_GLYPH['arrow']} {title.upper()}   [{_ts()}]")
	print(bar)
	_PHASE_T0[n] = time.time()

def phase_end(n, title=""):
	elapsed = time.time() - _PHASE_T0.get(n, time.time())
	print(f"{_GLYPH['light'] * 78}\n  Phase {n} complete{(' ' + title) if title else ''} "
	      f"({elapsed:.1f}s)\n")

def step(title):
	print(f"{_GLYPH['light'] * 4} {title} " + _GLYPH["light"] * max(4, 74 - len(title)))

def item(msg, status=None):
	tag = {"ok": _GLYPH["ok"], "fail": _GLYPH["fail"], "warn": _GLYPH["warn"]}.get(status, " ")
	print(f"  {tag} {msg}")

def kv(label, value, unit=""):
	pad = "." * max(2, 34 - len(label))
	print(f"    {label} {pad} {value}{(' ' + unit) if unit else ''}")

def result(ok, msg):
	tag = _GLYPH["ok"] if ok else _GLYPH["fail"]
	print(f"  {tag} {msg}")

def warn(msg):
	print(f"  {_GLYPH['warn']} WARNING: {msg}")

def error(msg):
	print(f"  {_GLYPH['fail']} ERROR: {msg}")

def case_row(cols):
	"""One aligned row of the fixed-width per-case summary table used by the
	PRS loop (Part 6) -- cols is a list of already-formatted strings."""
	print("  " + "  ".join(cols))

def table(headers, rows):
	widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
	          for i, h in enumerate(headers)]
	line = "  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
	print(line)
	print("  " + _GLYPH["light"] * (len(line) - 2))
	for r in rows:
		print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

def _stream_sa_subprocess(cmd, out_log, tag, heartbeat_s=300):
	"""Run a Phase-4 sensitivity-analysis subprocess (sens.py / sens_3_params.py)
	with LIVE streaming instead of capture_output=True.

	The old subprocess.run(..., capture_output=True) held every stdout/stderr
	line in memory until the child exited, so a multi-hour SA run printed
	nothing: the terminal (or run_log.txt) looked frozen, and SA_tool.out /
	SA_tool_3p.out only got written at the very end. Here every chunk the
	child emits is instead:
	  * echoed to this process's stdout immediately (raw bytes, so tqdm's
	    \\r progress bars pass through untouched), and
	  * appended to `out_log` at the same time (the same SA_tool*.out
	    artifact the old code produced, just populated live instead of at
	    exit), and
	  * if the child stays silent for heartbeat_s seconds (e.g. one long
	    simulation or YAML chunk), a heartbeat line reports the elapsed
	    time so the run never looks stuck.

	Returns the child's exit code (old code never inspected it).
	"""
	print(f"  [{tag}] launching: {' '.join(cmd)}", flush=True)
	# PYTHONUNBUFFERED: the child's stdout is a pipe, so Python would
	# otherwise switch print() to 8 KiB block buffering and progress lines
	# would arrive late in batches; unbuffered makes every print() reach the
	# tee/terminal as soon as it is issued.
	_child_env = dict(os.environ)
	_child_env.setdefault("PYTHONUNBUFFERED", "1")
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
	                        env=_child_env)
	last_out = time.time()
	with open(out_log, "a") as fh:
		while True:
			r, _, _ = select.select([proc.stdout], [], [], 1.0)
			if r:
				chunk = os.read(proc.stdout.fileno(), 65536)
				if chunk:
					data = chunk.decode("utf-8", errors="replace")
					fh.write(data)
					fh.flush()
					sys.stdout.write(data)
					sys.stdout.flush()
					last_out = time.time()
					continue
			if proc.poll() is not None:
				# drain whatever the child buffered right before exiting
				while True:
					chunk = os.read(proc.stdout.fileno(), 65536)
					if not chunk:
						break
					data = chunk.decode("utf-8", errors="replace")
					fh.write(data)
					fh.flush()
					sys.stdout.write(data)
					sys.stdout.flush()
				break
			if time.time() - last_out > heartbeat_s:
				elapsed = time.time() - _PHASE_T0.get(4, time.time())
				print(f"  [{tag}] … still running — elapsed {elapsed/60.0:.1f} min "
				      f"(no new output for {heartbeat_s}s)", flush=True)
				last_out = time.time()
	rc = proc.wait()
	print(f"  [{tag}] subprocess exited with code {rc}", flush=True)
	return rc

####################################################################################

### KEY WORDS #######
optType = "optimization_type"
targets = "targets"
mech = "mechanism"
pre_file = "Initial_pre_file"
count = "Counts"
countTar = "targets_count"
home_dir = os.getcwd()
fuel = "fuel"
fuelClass = "fuelClass"
bin_solve = "solver_bin"
bin_opt = "bin"
globRxn = "global_reaction"
countThreads = "parallel_threads"
unsrt = "uncertainty_data"
thermoF = "thermo_file"
transF = "trans_file"
order = "Order_of_PRS"
startProfile = "StartProfilesData"
design = "Design_of_PRS"
countRxn = "total_reactions"
fT = "fileType"
add = "addendum"

# GSA reaction prefilter (opt-in, default OFF -- merged from
# FINALIZED_MECH_OPT_SUITE). No-op unless target.opt has a truthy
# `GSA_Prefilter` block (and that block isn't itself `enabled: false` --
# see rmip.simulation.gsa_reaction_prefilter.apply_gsa_prefilter). The
# `rmip` import is lazy (only inside the enabled branch) so the cost of
# that import is paid only when a target.opt actually opts in.
def _maybe_apply_gsa_prefilter(optInputs, unsrt_data):
	gsa_cfg = optInputs.get("GSA_Prefilter") or {}
	if not gsa_cfg:
		return unsrt_data
	# Only reached when a target.opt actually opts in -- see note above.
	import sys as _sys
	from pathlib import Path as _Path
	_src_root = str(_Path(__file__).resolve().parent.parent)
	if _src_root not in _sys.path:
		_sys.path.insert(0, _src_root)
	from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter
	return apply_gsa_prefilter(unsrt_data, gsa_cfg)


################################################
# Open the input file and check for arguements #
################################################

TOTAL_PHASES = 12
phase(0, TOTAL_PHASES, "Input parse & configuration")

if len(sys.argv) > 1:
	input_file = open(sys.argv[1],'r')
	optInputs = yaml.safe_load(input_file)
	item(f"Input file found: {sys.argv[1]}", "ok")
else:
	error("Please provide a valid input file name as argument. See the UserManual for details of preparing the input file.")
	exit()

################################################
# Reading the input YAML file                  #
################################################

iFile = str(os.getcwd())+"/"+str(sys.argv[1])
dataCounts = optInputs[count]
binLoc = optInputs["Bin"]
inputs = optInputs["Inputs"]
locations = optInputs["Locations"]
startProfile_location = optInputs[startProfile]
stats_ = optInputs["Stats"]
global A_fact_samples
A_fact_samples = stats_["Sampling_of_PRS"]
# SA-stage (Phase 5 correlation-analysis) reaction-selection cutoff, DECOUPLED
# (2026-08-29) from cut_off_percentage, which remains the p-PRS/training
# parameter-selection cutoff read directly by PartialPRS (partial_PRS_system.py:78).
# A single cut_off_percentage used to drive both stages identically; per-study
# needs diverged (e.g. a stricter 1% SA screen feeding a looser 5% p-PRS/
# training cutoff), so this is now independently configurable. Default 1.0
# (=1%) is this project's SA convention; cut_off_percentage's own default is
# unchanged -- it stays REQUIRED, as before.
sa_cut_off_percentage = float(stats_.get("sa_cut_off_percentage", 1.0))
threshold = sa_cut_off_percentage / 100
# Correlation-based rescue tuning for the SA stage (see correlation_analysis.py):
# corr_cutoff = |r| trigger to rescue a sub-threshold reaction correlated with
# a selected one; threshold_corr = rescue eligibility floor on the candidate's
# own |S| (never rescue pure noise). None for sa_corr_cutoff preserves today's
# behaviour exactly: run_pipeline() auto-derives it from
# suggest_corr_cutoff(n_cases, alpha) when not given; only override (for both
# the diagnostic table and the actual rescued set) when target.opt sets it.
sa_corr_cutoff = stats_.get("sa_corr_cutoff", None)
sa_corr_cutoff = float(sa_corr_cutoff) if sa_corr_cutoff is not None else None
sa_threshold_corr = float(stats_.get("sa_threshold_corr", 0.01))
unsrt_location = locations[unsrt]
mech_file_location = locations[mech]
thermo_file_location = locations[thermoF]
trans_file_location = locations[transF]
fileType = inputs[fT]
samap_executable = optInputs["Bin"]["samap_executable"]
jpdap_executable = optInputs["Bin"]["jpdap_executable"]
if fileType == "chemkin":
	file_specific_input = "-f chemkin"
else:
	file_specific_input = ""
fuel = inputs[fuel]
gr = inputs[globRxn]
global_reaction = gr
design_type = stats_[design]
# Fail hard on an unrecognized Design_of_PRS rather than silently falling
# through a design-mode check somewhere downstream (see Part 0 of the
# 2026-07-31 A-factor/A1+B1+C1 resampling-gate work: Stage 3 D-optimal
# resampling must ONLY run for "A1+B1+C1" -- a typo'd/unexpected value here
# must not silently enable it).
assert design_type in ("A-facto", "A1+B1+C1"), (
	f"Unrecognized {design}: {design_type!r} (expected 'A-facto' or 'A1+B1+C1')")
parallel_threads = dataCounts[countThreads]
targets_count = int(dataCounts["targets_count"])
rps_order = stats_[order]
PRS_type = stats_["PRS_type"]
MRE_criteria = stats_["MRE_criteria"]
# ── Adaptive outlier re-training config (optional; safe defaults) ──────────
# When a PRS hold-out point exceeds MRE_criteria, densify the training set in a
# small local region around that point (reusing the same DesignMatrix curve
# classes and simulation machinery) and refit. See AdaptiveResampler.py.
# The loop keeps adding N_new fresh samples + resimulating + refitting each pass
# until the worst hold-out error drops to target_fraction * MRE_criteria (i.e.
# a tighter target than the original pass/fail threshold), or max_iter passes
# are used up -- both are overridable per-run from the target.opt Stats block.
#
# sample_mode 'hybrid' (default) hard-caps new rows per pass at 500 (5 passes =
# 2500 total) and splits that budget between a block concentrated near the
# outlier(s) (on-manifold, class-aware) and a sparse, low-discrepancy block
# spread evenly across the rest of the design space. This replaces 'dm_global'
# as the default: dm_global silently ignored N_new (Sobol power-of-2 rounding
# + the shuffle_factor multiplier injected ~4x the configured budget per pass,
# ~769 rows instead of 200) and never concentrated points near the actual
# outlier, which measurably made a real fit worse (see
# README_AdaptiveResampler.md for the diagnosis). 'local' (pure outlier
# clustering, can distort the global fit if overused) and 'dm_global' (pure
# whole-DM recombination) remain selectable for comparison/debugging.
# 'gap_fill' is the only mode that calls DesignMatrix's REAL class-A/B/C curve
# generators to synthesize genuinely NEW curves (every other mode only
# recombines/perturbs already-cached ones), filtered by a kNN isolation test
# against a reference set that grows every pass (original DM + everything
# added so far) so it never piles new points on already-simulated ones.
outlier_cfg = {
	"enable":               stats_.get("outlier_retraining", True),
	"N_new":                int(stats_.get("outlier_N_new", 500)),
	"delta_fraction":       float(stats_.get("outlier_delta_fraction", 0.15)),
	"seed":                 int(stats_.get("outlier_seed", 42)),
	"max_iter":             int(stats_.get("outlier_max_iter", 5)),
	"target_fraction":      float(stats_.get("outlier_target_fraction", 0.5)),
	# CHANGED DEFAULT: 'd_optimal' (see AdaptiveResampler.DOptimalResampler) replaces
	# 'hybrid' as the recommended mode once the PRS is fit in the identifiable
	# subspace (IdentifiablePRS.py) -- the surface is then linear in its fitted
	# coefficients, so classical D-optimal design augmentation is the exact,
	# minimum-simulation-count way to choose new points, and every candidate is
	# rejection-sampled against the reaction's own physical uncertainty band
	# before it is even eligible (never spends a simulation on an implausible
	# mechanism). 'local' / 'dm_global' / 'hybrid' / 'gap_fill' remain selectable
	# for comparison -- see report/ for the parametric study across all five.
	"sample_mode":          stats_.get("outlier_sample_mode", "d_optimal"),
	"method":               stats_.get("outlier_method", "sobol"),
	"uns_cap":              int(stats_.get("outlier_uns_cap", 256)),
	"shuffle_factor":       int(stats_.get("outlier_shuffle_factor", 2)),
	"local_fraction":       float(stats_.get("outlier_local_fraction", 0.4)),
	"gapfill_oversample":       int(stats_.get("outlier_gapfill_oversample", 2)),
	"gapfill_knn_radius_frac":  float(stats_.get("outlier_gapfill_knn_radius_frac", 0.05)),
	"gapfill_knn_k":            int(stats_.get("outlier_gapfill_knn_k", 3)),
	"gapfill_shuffle_factor":   int(stats_.get("outlier_gapfill_shuffle_factor", 4)),
	"force_resim":          bool(stats_.get("outlier_force_resim", False)),
	# ── 'd_optimal'-only knobs (AdaptiveResampler.DOptimalResampler) ───────────
	# band_tolerance: max allowed max_{reaction,T} |Delta kappa|/f_prior for a
	#   NEW (simulated) point to even be eligible -- independent of
	#   prs_fit_cfg's screening_tolerances, which governs the HISTORICAL rows
	#   (per instruction: new points are strictly band-gated regardless of what
	#   policy is used for old ones).
	"band_tolerance":       float(stats_.get("outlier_band_tolerance", 1.0)),
	# validity_n_temps: candidate validity is checked over a temperature sweep
	#   spanning the GLOBAL mechanism-uncertainty window (300-2500K, matching
	#   what Uncertainty's own class-A/B/C samplers check), not just the one
	#   case's local T -- a perturbed mechanism.yaml must be physically
	#   plausible everywhere it might be evaluated, not only at this case's
	#   nominal shock-tube condition.
	"validity_n_temps":     int(stats_.get("outlier_validity_n_temps", 12)),
	"acquisition_alpha":    float(stats_.get("outlier_acquisition_alpha", 0.6)),
	"candidate_oversample": int(stats_.get("outlier_candidate_oversample", 6)),
	# ── min_passes (2026-08-19): a case still above MRE_criteria must
	#   ATTEMPT at least this many resampling passes before any early-exit
	#   path (first-pass-noise guard, "no outliers found") is allowed to
	#   stop it -- see retrain_around_outlier's min-pass enforcement. Fixes
	#   the observed gap where D-optimal never resampled a genuinely
	#   failing case at all (a stale/exhausted candidate pool made pass 1
	#   select 0 points and the loop exited immediately, indistinguishable
	#   from "this case doesn't need help"). max_iter is raised to at least
	#   this value automatically.
	"min_passes":           int(stats_.get("outlier_min_passes", 5)),
	# stall_patience: consecutive passes with n_selected==0 or an identical
	#   selection to the previous pass (the "same handful of points
	#   re-picked every time" pathology, observed on Case 56: 34/2748
	#   candidates re-selected on 7 straight passes) before the case is
	#   aborted with a diagnostic instead of burning its full max_iter
	#   budget on simulations that cannot possibly help. A stalled pass does
	#   NOT count toward min_passes and runs NO simulation.
	"stall_patience":       int(stats_.get("outlier_stall_patience", 2)),
	# min_outlier_count / min_outlier_frac: previously read via .get() at the
	# first-pass noise-guard but never actually present in this dict, so
	# they were permanently stuck at their hardcoded defaults (5 /
	# 0.5%) and unsettable from target.opt. Wired in properly now.
	"min_outlier_count":    int(stats_.get("outlier_min_outlier_count", 5)),
	"min_outlier_frac":     float(stats_.get("outlier_min_outlier_frac", 0.005)),
}
outlier_cfg["max_iter"] = max(outlier_cfg["max_iter"], outlier_cfg["min_passes"])

# ── PRS fitting config: identifiable-subspace reparameterization + row
#    screening + hyperparameter search (Stages 1-2 of PAAR; see report/) ────
prs_fit_cfg = {
	# Row-screening tolerance grid searched alongside (k, order, ridge) -- the
	# full grid from the parametric study (report/), spanning "strict" (1.0,
	# every row must stay inside its reaction's stated uncertainty band)
	# through "no screening" (inf). The parametric study found tol=inf wins
	# for MOST cases (once the fit is in the identifiable subspace, most of a
	# band-invalid row's extremity lives in a direction the fit already
	# ignores, so screening the HISTORICAL rows often does not help -- unlike
	# screening NEW points, which is unconditional, see outlier_cfg above) --
	# but an intermediate tolerance (e.g. 10.0) is the best validation score
	# for at least one case, so the full grid is searched, not just the two
	# endpoints, and the winner is always logged, never silently assumed.
	"screening_tolerances": [float(x) for x in stats_.get(
		"prs_screening_tolerances", [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")])],
	# NOTE: capped at 20, not higher, despite k=22-24 occasionally helping one
	# case (see report/ Sec. "Case 1: an honest, diagnosed limitation") --
	# empirically, extending the cap to 24 combined with the tie-break below
	# preferring larger k among near-tied scores made a DIFFERENT case (Case 3)
	# regress from a reliable pass (holdout 1.98%) to a fail (2.38%), because
	# it also pushed that case toward k=24 where it had previously and
	# consistently used k=20. No single static cap threads every case's
	# partition-sensitivity at once; 20 is the value validated most broadly
	# across repeated runs.
	"k_grid":               [int(x) for x in stats_.get("prs_k_grid", [8, 10, 12, 14, 16, 20])],
	"order_grid":           [int(x) for x in stats_.get("prs_order_grid", [2, 3])],
	# WIDENED 2026-07-30 (10-fold CV robustness investigation, report/ sec
	# "10-fold cross-validation"): the original [0.0, 1e-7] grid could not
	# stabilize Case 4's worst-fold error (20.3% at ridge=0) -- a worst-of-
	# 10-fold search over this wider grid found ridge=0.3-0.5 needed for
	# Cases 4 and 7 specifically. Kept wide by default since the search
	# always picks per-case, never forces a large ridge where 0.0 already
	# wins.
	"ridge_grid":           [float(x) for x in stats_.get(
		"prs_ridge_grid", [0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.3, 0.5])],
	"svd_tol":              float(stats_.get("prs_svd_tol", 1e-6)),
	# 3-way split: train_frac = 1 - val_frac - holdout_frac.
	"val_frac":             float(stats_.get("prs_val_frac", 0.15)),
	"holdout_frac":         float(stats_.get("prs_holdout_frac", 0.15)),
	"split_seed":           int(stats_.get("prs_split_seed", 104)),
	"split_seed_inner":     int(stats_.get("prs_split_seed_inner", 7)),
	# Stage-1 identifiable-subspace temperature window: +-temp_window_frac
	# around THIS case's nominal T (deliberately narrower than outlier_cfg's
	# global validity window -- see IdentifiablePRS.py's module docstring for
	# why these are two different, both-correct concepts).
	"temp_window_frac":     float(stats_.get("prs_temp_window_frac", 0.25)),
	# ── 10-fold-CV-robust search (default 2026-07-30) ───────────────────────
	# A single 70/15/15 split can massively underestimate true out-of-sample
	# risk (observed: Case 4 passed a single split at 1.45% hold-out, but a
	# genuine 10-fold sweep at the SAME hyperparameters found mean test error
	# 8.1%, worst fold 20.3%). search_method="cv_kfold" (default) replaces
	# the single-split fit_prs_with_search with fit_prs_with_cv_search
	# (worst-of-cv_folds-fold selection); "single_split" restores the
	# original behaviour for comparison. band_tolerance_eval (default 1.0)
	# scores CV folds and the final hold-out on band-VALID points only --
	# every worst-fold catastrophe traced to a single band-invalid test row
	# (ratio 7x-30x the physical band, the class-B m=3 sampler bug, see
	# sec:upstream item 1); training is never filtered by this, only scoring.
	"search_method":        stats_.get("prs_search_method", "cv_kfold"),
	"cv_folds":             int(stats_.get("prs_cv_folds", 10)),
	"cv_seed":              int(stats_.get("prs_cv_seed", 13)),
	"band_tolerance_eval":  (None if stats_.get("prs_band_tolerance_eval", 1.0) in (None, "none")
	                          else float(stats_.get("prs_band_tolerance_eval", 1.0))),
	# "reduced" (default, unchanged): band-validity ratio uses only the
	# SELECTED Arrhenius parameters' own uncertainty (L_r). "full": uses the
	# sampler's actual acceptance-test band (all 3 Arrhenius parameters,
	# L_full) -- see IdentifiablePRS.reaction_W_fprior's docstring. "reduced"
	# is kept as the default so report/ and report_10fold_validation/ stay
	# bit-for-bit reproducible; a study whose reduced-subspace scoring is
	# throwing away most of its rows (m<3 reactions dominating the layout)
	# should set prs_band_fprior: full in target.opt's Stats: block.
	"band_fprior":          str(stats_.get("prs_band_fprior", "reduced")),
	# max_terms_frac (2026-08-19): previously hardcoded to 0.4 inside
	# IdentifiablePRS.fit_prs_with_search/fit_prs_with_cv_search and never
	# threaded through from here, so it was unsettable from target.opt even
	# though it is exactly the guard that matters once a case's fittable
	# pool shrinks to its band-valid rows only (e.g. Case 0: 368 good rows
	# vs 6025 generated) -- it now directly caps the achievable k for a
	# data-starved case, and should be visible/tunable, not silently fixed.
	"max_terms_frac":       float(stats_.get("prs_max_terms_frac", 0.4)),
}

def _dry_run_from_stats(stats_):
	"""Resolve the resampling live/dry switch from the Stats block.

	'do simulation: true' (user's exact wording) -> False (live chemkin).
	'do_simulation' is the underscore alias; when NEITHER positive key is
	present, the legacy negative key 'resample_dry_run' (true = dry-run)
	is used verbatim. The positive keys invert: they ask FOR simulation,
	resample_dry_run asks to SKIP it.
	"""
	do_sim = stats_.get("do simulation", stats_.get("do_simulation"))
	if do_sim is not None:
		return not bool(do_sim)
	return bool(stats_.get("resample_dry_run", True))


# ── Experiment arms (2026-08-28): response coordinate transforms
#    (encoding/decoding, ResponseEncoder) + correlation-aware resampling
#    (CorrelationResampler), selectable per run. Default `original` == the
#    pre-2026-08-28 behavior exactly (missing keys change nothing). Every
#    value is parsed from the optional target.opt Stats: block; the full
#    schema and semantics live in phase10_prs.py (EXPERIMENT_DEFAULTS) and
#    src/MUQ-SAC/Readme.md.
#      prs_experiment_mode: original | transform | resampling |
#                           resampling_transform
experiment_cfg = {
	"mode":                 str(stats_.get("prs_experiment_mode", "original")),
	"transform": {
		"variant":          str(stats_.get("transform_variant", "power")),
		"P":                float(stats_.get("transform_P", 0.1)),
		"P_grid":           [float(x) for x in stats_.get(
			"transform_P_grid", [1.0, 0.5, 0.2, 0.1, 0.05, 0.02])],
		"standardize":      bool(stats_.get("transform_standardize", False)),
		"rank_fit_on":      str(stats_.get("transform_rank_fit_on", "all")),
		"p_lo":             float(stats_.get("transform_p_lo", 0.02)),
	},
	"resampling": {
		"n_new":            int(stats_.get("resample_n_new", 2000)),
		"empirical_weight": float(stats_.get("resample_empirical_weight", 1.0)),
		"band_tolerance":   float(stats_.get("resample_band_tolerance", 1.0)),
		"qmc":              bool(stats_.get("resample_qmc", True)),
		"seed":             int(stats_.get("resample_seed", 20260828)),
		"center_mode":      str(stats_.get("resample_center_mode", "pool_center")),
		"dedup_eps":        float(stats_.get("resample_dedup_eps", 1e-6)),
		# "do simulation: true" (user's exact wording) is the live-simulation
		# switch for the resampling arms: true -> dry_run=False (live
		# chemkin), false/absent -> dry-run. "do_simulation" is the
		# underscore alias; if neither key exists, the legacy
		# "resample_dry_run" (true = dry-run) is used verbatim.
		# NOTE the inversion: do-simulation keys are positive ("run the
		# sims"), resample_dry_run is negative.
		"dry_run":          _dry_run_from_stats(stats_),
	},
	"stage3_with_transform": bool(stats_.get("stage3_with_transform", False)),
}
item(f"Parallel threads: {parallel_threads}")
targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())

phase_end(0)
phase(1, TOTAL_PHASES, "Target loading & T-P-φ space")
####################################################
##  Unloading the target data	  	           	##
## TARGET CLASS CONTAINING EACH TARGET AS A CASE  ##
####################################################

target_list = []
c_index = 0
string_target = ""

for target in targetLines[:targets_count]:
	if "#" in target:
		target = target[:target.index('#')]	
	add = deepcopy(addendum)
	t = combustion_target_class.combustion_target(target,add,c_index)
	c_index +=1
	target_list.append(t)

case_dir = range(0,len(target_list))
# Optional targeted-rerun override (2026-08-19): restrict the per-case PRS
# loop below to a specific subset of case indices, e.g. to re-run adaptive
# retraining for a handful of cases that failed a previous full pass without
# re-doing the (fast, but not free) Stage1+2 search for every already-passing
# case. Every setup step above (SA, correlation analysis, design-matrix/YAML
# generation) still runs for its own full, natural scope -- this only trims
# the loop at "for case_index,case in enumerate(temp_sim_opt):" below.
# NOTE: with this set, the final Optimizer.run_optimization_with_selected_PRS
# step would only see PRS surfaces for the subset -- harmless here only
# because solution_zeta.save already exists from a prior full run, so that
# step is skipped entirely (see the `if "solution_zeta.save" not in
# os.listdir():` guard near the end of this file); do not use this override
# for a run meant to produce a fresh optimized mechanism.
_case_subset_env = os.environ.get("MUQSAC_CASE_SUBSET")
if _case_subset_env:
	case_dir = [int(c.strip()) for c in _case_subset_env.split(",") if c.strip() != ""]
	print(f"[MUQSAC_CASE_SUBSET] restricting per-case PRS loop to: {case_dir}")
print(f"Case directories: {case_dir}")
print("\n\nOptimization targets identified.\n")

# SAVING THE TARGETS IN A CSV FILE
df = Target2CSV(target_list).to_dataframe()
df.to_csv("targets.csv", index=False)
print("\t Targets exported to targets.csv\n\n")

#Plotting the T-P-phi space
print("\t Plotting the T-P-phi space...")
TPhiPlotter(df).plot_t_p_phi(color="Phi", save_path="T-P-phi_space.png", show=False)
print("T-P-phi space plot saved as T-P-phi_space.png\n\n")
print("\n\nOptimization targets identified.\nStarted the MUQ process.......\n")


phase_end(1)
phase(2, TOTAL_PHASES, "Uncertainty quantification")
############################################
##  Uncertainty Quantification            ##
##  					   				  ##
############################################

if "unsrt.pkl" not in os.listdir():
	UncertDataSet = uncertainty.uncertaintyData(locations,binLoc);
	unsrt_data = UncertDataSet.extract_uncertainty();
	# Save the object to a file
	with open('unsrt.pkl', 'wb') as file_:
		pickle.dump(unsrt_data, file_)
	item("Uncertainty analysis finished (freshly computed)", "ok")

else:
	# Load the object from the file
	with open('unsrt.pkl', 'rb') as file_:
		unsrt_data = pickle.load(file_)
	item("Uncertainty analysis already finished (loaded from unsrt.pkl)", "ok")

# GSA reaction prefilter (opt-in, default OFF -- see _maybe_apply_gsa_prefilter)
unsrt_data = _maybe_apply_gsa_prefilter(optInputs, unsrt_data)

###########################################################
#### Printing the reactions and their index in the file ###
###########################################################

string_Rxn = ""
str_rxn = ""
for i in unsrt_data:
	string_Rxn += f"{i}\t{unsrt_data[i].index}\n"
	str_rxn += f"{i}\n"
file_rxn = open("ACTIVE_RXN_INDEX.csv","w").write(string_Rxn)
file_rxn_list = open("ACTIVE_RXN_LIST.csv","w").write(str_rxn)
kv("Active reactions", len(unsrt_data))
kv("Sampler version", uncertainty.SAMPLER_VERSION)

phase_end(2)
phase(3, TOTAL_PHASES, "Reaction dictionary & covariance")
######################################################################
##  CREATING A DICTIONARY CONTAINING ALL THE DATA FROM UNSRT CLASS  ##
##		                                                    		##
######################################################################

manipulationDict = {}
selection = []
Cholesky_list = []
zeta_list = []
activeParameters = []
nominal_list = []
rxn_list = []
rIndex = []
rxn_list_a_fact = []

for rxn in unsrt_data:
	#print(i)
	data = {}
	data["temperatures"] = unsrt_data[rxn].temperatures
	data["uncertainties"] = unsrt_data[rxn].uncertainties
	data["Arrhenius"] = unsrt_data[rxn].nominal
	#print(rxn,data)
	rxn_list.append(rxn)
	rxn_list_a_fact.append(unsrt_data[rxn].rxn)
	selection.extend(unsrt_data[rxn].selection)
	Cholesky_list.append(unsrt_data[rxn].cholskyDeCorrelateMat)
	zeta_list.append(unsrt_data[rxn].perturb_factor)
	activeParameters.extend(unsrt_data[rxn].activeParameters)
	nominal_list.append(unsrt_data[rxn].nominal)
	rIndex.append(unsrt_data[rxn].rIndex)

rxn_list_a_fact = set(rxn_list_a_fact)
rxn_a_fact = ""
for rxn in rxn_list_a_fact:
	rxn_a_fact+=f"{rxn}\n"
file_rxn_a_fact = open("rxn_list_a_fact.csv","w").write(rxn_a_fact)	
manipulationDict["selection"] = deepcopy(selection)#.deepcopy()
manipulationDict["Cholesky"] = deepcopy(Cholesky_list)#.deepcopy()
manipulationDict["zeta"] = deepcopy(zeta_list)#.deepcopy()
manipulationDict["activeParameters"] = deepcopy(activeParameters)#.deepcopy()
manipulationDict["nominal"] = deepcopy(nominal_list)#.deepcopy()
print(f"\n\n################################################\nThe total reactions selected for this study: {len(manipulationDict['activeParameters'])}\n\tThe list is as follows:\n\t")
print("\t"+f"{manipulationDict['activeParameters']}\n")
print("\n################################################\n\n")

R = 1.987  # cal/mol-K
base_dir = "Uncertainty_Matrices"
os.makedirs(base_dir, exist_ok=True)

for rxn in unsrt_data:

    # Create a safe folder name
    rxn_dir = os.path.join(base_dir, str(rxn).replace("/", "_"))
    os.makedirs(rxn_dir, exist_ok=True)

    # Cholesky matrix
    L = np.asarray(unsrt_data[rxn].cholskyDeCorrelateMat)

    # Covariance matrix (sigma_from_cholesky normalizes the A-factor-only
    # 1-element form -- see IdentifiablePRS.py's docstring -- so this
    # diagnostics artifact matches what the pipeline actually fits with,
    # instead of a bare L @ L.T scalar for those reactions)
    Sigma = IPRS.sigma_from_cholesky(
        L, n_active=len(unsrt_data[rxn].activeParameters))
    
    # Nominal values in uncertainty space
    nominal = np.asarray(unsrt_data[rxn].nominal)

    # Convert back to Arrhenius form
    A_nom = np.exp(nominal[0])
    n_nom = nominal[1]
    Ea_nom = nominal[2] * R
    
    # Save files
    np.savetxt(
        os.path.join(rxn_dir, "L.csv"),
        np.atleast_2d(L),
        delimiter=",",
        fmt="%.10e"
    )

    np.savetxt(
        os.path.join(rxn_dir, "Sigma.csv"),
        np.atleast_2d(Sigma),
        delimiter=",",
        fmt="%.10e"
    )
    
    # Save nominal physical values
    nominal_df = pd.DataFrame({
        "Parameter": ["A", "n", "Ea"],
        "Value": [A_nom, n_nom, Ea_nom]
    })

    nominal_df.to_csv(
        os.path.join(rxn_dir, "nominal.csv"),
        index=False
    )
print(f"\nSaved matrices for {len(unsrt_data)} reactions in '{base_dir}'")
print("\n################################################\n\n")

import csv

# =============================================================================
# CSV FILE
# =============================================================================

# PATH NOTE (2026-08-19): COST_FILE used to be written from the DESIGN-MATRIX
# GENERATION loop, which runs BEFORE os.chdir("Opt") -- so a bare relative
# filename correctly resolved to home_dir/optimization_cost.csv. It is now
# written from the per-case PRS loop instead (finalized as a real ledger
# AFTER screening + Stage 3, see HEADERS' comment), which runs AFTER
# os.chdir("Opt") -- so it needs the SAME "../" prefix PRS_ERROR_FILE/
# PRS_DETAIL_FILE already use, or it would silently land in Opt/ instead of
# the run root.
COST_FILE = "../optimization_cost.csv"
PRS_ERROR_FILE = "../Prediction_errors.csv"

# =============================================================================
# CREATE FILE WITH HEADER (ONLY ONCE)
# =============================================================================

# HEADERS (2026-08-19 rewrite): optimization_cost.csv is now a per-case
# SIMULATION-ACCOUNTING LEDGER, finalized once (after screening + Stage 3)
# per case, not a pre-simulation closed-form estimate appended every run.
# "Cost_Estimate_*" keeps the ORIGINAL two columns' meaning unchanged (a
# theoretical 2nd-order-formula sample count, still useful for sizing), but
# every column after it is an ACTUAL count, so "how many simulations did
# this case really cost, and how many of them were useful" is finally
# answerable from this one file. See screen_case_samples() / the per-case
# loop's cost-ledger write for how each column is populated.
HEADERS = [
    "Case_ID", "Total_Rxns", "Active_Rxns",
    "Cost_Estimate_pPRS", "Cost_Estimate_FullPRS",
    "Samples_Generated", "Samples_Simulated",
    "Samples_Failed_Sim", "Samples_Sentinel", "Samples_NonFinite", "Samples_NonPhysical",
    "Samples_BandInvalid", "Samples_Good",
    "Stage3_Passes_Attempted", "Stage3_Passes_Accepted",
    "Stage3_Sims_Run", "Stage3_Sims_Kept",
    "Total_Sims_Run", "Total_Sims_Useful", "Sample_Efficiency_Pct",
]

PRS_HEADER = [
	"Case_ID", "Split", "Pass",
	"Max_Training_Error",
	"Max_Testing_Error",
	"Selection_Criterion",
	"Max_Testing_Error_BandValid",
	"Rows_Good", "Rows_Quarantined", "Rows_Kept_For_Training",
	"Verdict",
]
# NOTE on semantics (see report/ for the full before/after): "Max_Testing_Error"
# is now the max relative error on an UNTOUCHED hold-out split (never used for
# fitting, hyperparameter search, or outlier-retraining decisions) -- a
# strictly stronger guarantee than the original single 90/10 split, which
# reused the same xTest both to pick hyperparameters/outliers AND to report
# the number. Column name kept unchanged for continuity with historical
# Prediction_errors.csv files.
#
# "Max_Testing_Error_BandValid" (added 2026-08-19): the SAME hold-out split,
# scored only on rows the sampler's own physical-validity test would have
# accepted (see IdentifiablePRS.band_valid_mask / prs_fit_cfg's
# band_tolerance_eval). This is the number the PASS/FAIL line printed at the
# end of the per-case loop actually gates on -- "Max_Testing_Error" alone can
# be wildly inflated by a handful of physically-implausible rows (up to
# >1000% observed pre-fix) and should not be read as the fit quality on its
# own; always read it alongside this column.

PRS_DETAIL_FILE = "../Prediction_errors_detailed.csv"
# "Rows_Available" used to report len(xTrain) (the 70% split, e.g. 4218) --
# not the case's actual row count (6025) -- silently understating how much
# data a case really had. Both are now reported explicitly, alongside the
# good/quarantined breakdown screen_case_samples() produces.
PRS_DETAIL_HEADER = [
	"Case_ID", "Tolerance", "K", "Order", "Ridge_Alpha",
	"Rows_Generated", "Rows_Good", "Rows_Train_Pool", "Rows_Kept",
	"Max_Train_Error", "Max_Val_Error", "Max_Holdout_Error",
	"Max_Val_Error_BandValid", "Max_Holdout_Error_BandValid",
]

def initialize_csv(csv_name, headers):
    """
    Create CSV file with headers if it does not exist.

    Parameters
    ----------
    csv_path : str
        Output CSV path

    headers : list
        List of column names
    """
    csv_path = os.path.join(os.getcwd(),csv_name)
    
    if not os.path.exists(csv_path):

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        print(f"Created CSV: {csv_path}")

def reset_csv(csv_name, headers):
    """Always (re)write csv_name with a fresh header, discarding any prior
    content. Unlike initialize_csv (header-only-if-missing, hence how
    optimization_cost.csv accumulated 697 duplicate Case_ID rows across
    reruns of the same study -- every append_csv_row call landed on the SAME
    never-truncated file), this is what a per-run LEDGER (optimization_cost/
    Prediction_errors/_detailed.csv) needs: one row per case, per run,
    every time. The previous run's file is expected to already be archived
    by archive_previous_study before this is called, so nothing is lost."""
    csv_path = os.path.join(os.getcwd(), csv_name)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# =============================================================================
# APPEND ROW DATA
# =============================================================================

def append_csv_row(csv_path, row_data):
    """
    Append a single row to CSV.

    Parameters
    ----------
    csv_path : str
        CSV file path

    row_data : list or tuple
        Data row to append
    """

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

# =============================================================================
# APPEND DICTIONARY DATA
# =============================================================================

def append_csv_dict(csv_path, data_dict, headers=None):
    """
    Append dictionary data to CSV.

    Parameters
    ----------
    csv_path : str
        CSV file path

    data_dict : dict
        Dictionary containing row data

    headers : list, optional
        Header order for writing dictionary values
    """

    if headers is None:
        headers = list(data_dict.keys())

    row = [data_dict[h] for h in headers]

    append_csv_row(csv_path, row)

def getTotalUnknowns(N):
	n_ = 1 + 2*N + (N*(N-1))/2
	return int(n_)
	
def getSim(n,design):
	n_ = getTotalUnknowns(n)
	if design == "A-facto":
		sim = 4*n_
	else:
		sim = 7*n_	
	return sim

"""
For Partial PRS (p-PRS), we need to do a second level of sensitivity analysis:	
	
	sensitivity_analysis: a dictionary that stores the sensitivity coefficients of all the active
			      parameters (selected in the first set of sensitivity analysis) reaction wise
			      as in unsrt_data dictionary
"""

phase_end(3)
phase(4, TOTAL_PHASES, "Sensitivity analysis")
if PRS_type == "Partial":
	####################################################
	## Sensitivity Analysis of the selected reactions ##
	####################################################

	item("Partial PRS selected as the solution-mapping technique -- running sensitivity analysis (kindly be patient)")
	if "A-facto" in design_type:
		if "sens_parameters.pkl" not in os.listdir():
			status_ = "Pending"
			# Arguments to pass to script2.py
			args = [sys.argv[1], 'rxn_list_a_fact.csv','&>SA.out']

			# Doing A factor sensitivity Analysis
			# Live-streamed (Phase 4 only): the old capture_output=True held
			# every line until the subprocess exited, so the run looked frozen.
			_rc = _stream_sa_subprocess(
				['python3.9', binLoc["SA_tool"]] + args,
				out_log="SA_tool.out", tag="SA", heartbeat_s=300,
			)

			print("\n\nSensitivity Analysis for A-factor is Done!!")
			if _rc != 0:
				error(f"SA subprocess exited with code {_rc} — check SA_tool.out; "
				      f"sens_parameters.pkl may be missing")
			with open('sens_parameters.pkl', 'rb') as file_:
				sensitivity_analysis = pickle.load(file_)
			PICKLE_PATH = 'sens_parameters.pkl'
		else:
			PICKLE_PATH = 'sens_parameters.pkl'
			status_ = "Pending"
			print("\n\tBrute-force Sensitivity analysis is already over!!\n")
			with open('sens_parameters.pkl', 'rb') as file_:
				sensitivity_analysis = pickle.load(file_)
	
	else:
		if "sens_3p_parameters.pkl" not in os.listdir() :
			status_ = "Pending"
			args = [sys.argv[1], 'rxn_list.csv','&>SA_3p.out']
			print("\n\tRunning sens_3_params.py code\n")
			# Live-streamed (Phase 4 only): the old capture_output=True held
			# every line until the subprocess exited, so the run looked frozen.
			_rc = _stream_sa_subprocess(
				['python3.9', binLoc["SA_tool_3p"]] + args,
				out_log="SA_tool_3p.out", tag="SA-3P", heartbeat_s=300,
			)
			print("\n\nSensitivity Analysis for 3 parameters is Done!!")
			if _rc != 0:
				error(f"SA-3P subprocess exited with code {_rc} — check SA_tool_3p.out; "
				      f"sens_3p_parameters.pkl may be missing")
			with open('sens_3p_parameters.pkl', 'rb') as file_:
				sensitivity_analysis = pickle.load(file_)
			PICKLE_PATH = 'sens_3p_parameters.pkl'
		else:
			PICKLE_PATH = 'sens_3p_parameters.pkl'
			status_ = "Pending"
			print("\n\t3-Parameter Sensitivity analysis is already over!!\n")
			with open('sens_3p_parameters.pkl', 'rb') as file_:
				sensitivity_analysis = pickle.load(file_)
	
	
	OUTPUT_DIR = "sa_pipeline_v2_out"
	case_ids = list(sensitivity_analysis.keys())
	first_case = case_ids[0]
	rxn_ids = list(sensitivity_analysis[first_case].keys())
	first_rxn = rxn_ids[0]

	print(f"Found {len(case_ids)} cases, e.g. case={first_case!r}")
	print(f"Found {len(rxn_ids)} reactions in that case, e.g. rxn={first_rxn!r}")
	print(f"Sample value for (case={first_case!r}, rxn={first_rxn!r}): "
		f"{sensitivity_analysis[first_case][first_rxn]!r}")

	phase_end(4)
	phase(5, TOTAL_PHASES, "Correlation analysis")
	# ── Correlation analysis: build per-case active-feature lists ─────────
	if "A-facto" in design_type:
		# Per-parameter (A-only) Pearson pipeline.
		# Returns {case_str -> [rxn_equation, ...]} for the final active set.
		_final_sets_A = CA.run_pipeline(
			PICKLE_PATH, OUTPUT_DIR, threshold=threshold,
			corr_cutoff=sa_corr_cutoff,
			final_set_corr_cutoff=(sa_corr_cutoff if sa_corr_cutoff is not None else 0.9),
			threshold_corr=sa_threshold_corr,
		)
		# Convert reaction-equation strings to activeParameter strings ("R1093_A")
		# that match PartialPRS.active_params in the A-facto branch.
		_eq_to_ap = {
			rxn.split(":")[0]: unsrt_data[rxn].activeParameters[0]
			for rxn in unsrt_data
		}
		correlation_results = {
			case_str: [_eq_to_ap[eq] for eq in rxn_list if eq in _eq_to_ap]
			for case_str, rxn_list in _final_sets_A.items()
		}
		# Ensure every case has an entry (guard against missing cases in pkl)
		for case in case_dir:
			correlation_results.setdefault(str(case), [])
	else:
		# Combined cross-Arrhenius pipeline for 3-param (A, n, Ea).
		# Returns {case_str -> ["R1093_A", "R1093_n", ...]} directly in the
		# same format as PartialPRS.active_params — no conversion needed.
		correlation_results = CA.run_combined_3p_pipeline(
			sensitivity_analysis,
			unsrt_data,
			OUTPUT_DIR,
			threshold=threshold,
			corr_cutoff=(sa_corr_cutoff if sa_corr_cutoff is not None else 0.9),
			threshold_corr=sa_threshold_corr,
		)
		for case in case_dir:
			correlation_results.setdefault(str(case), [])
	phase_end(5)
	phase(6, TOTAL_PHASES, "Design matrix generation")
	partialPRS_Object = []
	selected_params_dict = {}
	design_matrix_dict = {}
	yaml_loc_dict = {}
	no_of_sim = {}
	# cost_estimate_dict: the closed-form (Total_Rxns, Active_Rxns, formula
	# cost) figures per case -- kept aside rather than written to
	# optimization_cost.csv immediately, because that file is now a per-case
	# ledger finalized ONCE, after screening + Stage 3 (see Part 5 / the
	# per-case PRS loop below), not a running append -- the old
	# write-before-any-simulation-or-screening behaviour is what let 697
	# duplicate rows accumulate across reruns and never accounted for
	# Stage-3 simulations at all.
	cost_estimate_dict = {}
	step(f"Generating partial-PRS design matrices for {len(case_dir)} case(s)")
	# stale_report: cases whose freshly-computed selection (current
	# sa_cut_off_percentage/cut_off_percentage + correlation rescue) no longer
	# matches the DesignMatrix.csv/selected_parameters.csv already cached on
	# disk for that case (see PartialPRS.__init__'s stale_selection detection,
	# 2026-08-29). Detected here, across every case, BEFORE anything is built
	# or simulated for any of them -- deliberately detect-and-report, not
	# auto-rebuild: rebuilding a case correctly also requires discarding its
	# real Opt/case-<N>/ simulation output and touching the study-global
	# Opt/locations / Opt/progress files, which this run must never do
	# without the operator's say-so.
	stale_report = []
	for case in case_dir:
		PPRS_system = P_PRS.PartialPRS(sensitivity_analysis[str(case)],correlation_results[str(case)],unsrt_data,optInputs,target_list,str(case),activeParameters,design_type,status=status_)
		if PPRS_system.stale_selection is not None:
			added, removed = PPRS_system.stale_selection
			stale_report.append((case, added, removed))
			continue   # do not build a design matrix or simulate this case this run
		yaml_loc, p_design_matrix, selected_params, Active_Rxns = PPRS_system.partial_DesignMatrix()
		partialPRS_Object.append(PPRS_system)
		no_of_sim[case] = int(PPRS_system.no_of_sim)
		yaml_loc_dict[case] = yaml_loc
		design_matrix_dict[case] = p_design_matrix
		selected_params_dict[case] = selected_params
		cost_estimate_dict[case] = {
			"Total_Rxns": len(activeParameters),
			"Active_Rxns": Active_Rxns,
			"Cost_Estimate_pPRS": getSim(Active_Rxns, design_type),
			"Cost_Estimate_FullPRS": getSim(len(activeParameters), design_type),
		}

	if stale_report:
		error(f"{len(stale_report)} case(s) have a STALE cached selection -- their "
		      f"parameter set changed (new thresholds and/or correlation rescue) since "
		      f"DM_FOR_PARTIAL_PRS/<case>/ was last built, but the cached design matrix "
		      f"and simulations reflect the OLD selection. Stopping before any "
		      f"simulation or PRS fitting runs, rather than silently reusing stale data.")
		for case, added, removed in stale_report:
			print(f"\n  Case-{case}: +{added} -{removed}")
			print(f"    DM_FOR_PARTIAL_PRS/{case}/            (selected_parameters.csv, DesignMatrix.csv, ...)")
			print(f"    YAML_FILES_FOR_PARTIAL_PRS/{case}/    (regen guard is row-count-only -- must go too)")
			print(f"    Opt/case-{case}/                      (raw Chemkin simulation output for this case)")
			print(f"    Opt/Data/Simulations/sim_data_case-{case}.lst")
			print(f"    Opt/Data/Simulations/failed_sim_data_case-{case}.lst")
		print(f"\n  Opt/locations and Opt/progress are STUDY-GLOBAL (shared by every case, "
		      f"not just the ones above):\n"
		      f"    - Opt/progress gates simulation dispatch for the WHOLE study -- while it "
		      f"exists, simulations are never (re)launched for any case, changed or not; "
		      f"remove it so dispatch runs again (unaffected cases are still skipped by "
		      f"their own per-case existence check, so this alone is safe).\n"
		      f"    - Opt/locations accumulates one line per simulated row across every "
		      f"case; after clearing a case's Opt/case-<N>/ above, also drop that case's "
		      f"lines from Opt/locations so its row bookkeeping doesn't reference "
		      f"directories that no longer exist.\n"
		      f"  Clear the paths listed per case above (and re-simulate), then rerun.")
		sys.exit(1)

	item(f"Design matrices generated for {len(case_dir)} case(s) -- "
	     f"{sum(no_of_sim.values())} total rows on disk", "ok")
##################################################################
##  Use the unsrt data to sample the Arrhenius curves           ##
##  MUQ-SAC: Method of Uncertainty Quantification and           ##
##	Sampling of Arrhenius Curves                                ##
##################################################################
	item("Perturbed mechanism files for partial-PRS generated", "ok")
	SSM = simulator.SM(target_list,optInputs,unsrt_data,design_matrix_dict,tag="Partial")
	
else:
	"""
	Function Inputs:
		Unsrt Data:
	Output:
		Desired Number of Samples - n_s for
			- Class A curves
			- Class B curves
			- Class C curves

	"""
		
	def getTotalUnknowns(N):
		n_ = 1 + 2*N + (N*(N-1))/2
		return int(n_)
		
	def getSim(n,design):
		n_ = getTotalUnknowns(n)
		if design == "A-facto":
			sim = int(A_fact_samples)*n_
		else:
			sim = 7*n_	
		return sim
	no_of_sim = {}
	if "DesignMatrix.csv" not in os.listdir():
		print(f"\n\n\tNo. of Simulations required with {len(manipulationDict['activeParameters'])} reactions ({A_fact_samples}*n_): {getSim(len(manipulationDict['activeParameters']),design)}\n")
		no_of_sim_ = getSim(len(manipulationDict["activeParameters"]),design_type)
		design_matrix = DM.DesignMatrix(unsrt_data,design_type,getSim(len(manipulationDict["activeParameters"]),design_type),len(manipulationDict["activeParameters"])).getSamples()
		no_of_sim_ = len(design_matrix)
	else:
		no_of_sim_ = getSim(len(manipulationDict["activeParameters"]),design_type)
		design_matrix_file = open("DesignMatrix.csv").readlines()
		design_matrix = []
		no_of_sim_ = len(design_matrix_file)
		for row in design_matrix_file:
			design_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
	design_matrix_dict = {}
	for case in case_dir:
		design_matrix_dict[case] = design_matrix
		no_of_sim[case] = no_of_sim_
	
	SSM = simulator.SM(target_list,optInputs,unsrt_data,design_matrix,tag="Full")

	# Bug 4 fix (merged from FINALIZED_MECH_OPT_SUITE): directory existing is
	# not proof all YAML files were written -- files are generated in chunks
	# of 500, so an interrupted run can leave a directory with only some
	# mechanism_*.yaml files. Verify file count against design_matrix before
	# trusting the restart path; otherwise wipe and regenerate.
	_perturbed_mech_exists = "Perturbed_Mech" in os.listdir()
	_existing_mech_count = len(glob.glob(os.path.join("Perturbed_Mech", "mechanism_*.yaml"))) if _perturbed_mech_exists else 0
	if _perturbed_mech_exists and _existing_mech_count != len(design_matrix):
		print(f"\nIncomplete Perturbed_Mech directory found "
		      f"({_existing_mech_count}/{len(design_matrix)} files) -- regenerating\n")
		shutil.rmtree("Perturbed_Mech")
		_perturbed_mech_exists = False

	if not _perturbed_mech_exists:
		os.mkdir("Perturbed_Mech")
		print("\nPerturbing the Mechanism files\n")
		
		chunk_size = 500
		params_yaml = [design_matrix[i:i+chunk_size] for i in range(0, len(design_matrix), chunk_size)]
		count = 0
		yaml_loc = []
		for params in params_yaml:
			
			yaml_list = SSM.getYAML_List(params)
			#yaml_loc = []
			location_mech = []
			index_list = []
			for i,dict_ in enumerate(yaml_list):
				index_list.append(str(count+i))
				location_mech.append(os.getcwd()+"/Perturbed_Mech")
				yaml_loc.append(os.getcwd()+"/Perturbed_Mech/mechanism_"+str(count+i)+".yaml")
			count+=len(yaml_list)
			SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
			print(f"\nGenerated {count} files!!\n")
		print("\nGenerated the YAML files required for simulations!!\n")
	else:
		print("\nYAML files already generated!!")
		yaml_loc = []
		location_mech = []
		index_list = []
		for i,sample in enumerate(design_matrix):
			index_list.append(i)
			location_mech.append(os.getcwd()+"/Perturbed_Mech")
			yaml_loc.append(os.getcwd()+"/Perturbed_Mech/mechanism_"+str(i)+".yaml")
	
	selected_params = []
	for params in activeParameters:
		selected_params.append(1)
	
	selected_params_dict = {}
	design_matrix_dict = {}
	yaml_loc_dict = {}
	for case in case_dir:
		yaml_loc_dict[case] = yaml_loc
		design_matrix_dict[case] = design_matrix
		selected_params_dict[case] = selected_params
item("Perturbed mechanism files for full-PRS generated", "ok")
phase_end(6, "design matrix generation")
phase(7, TOTAL_PHASES, "Simulation")
##############################################################
##     Doing simulations using the design matrix            ##
##     The goal is to test the design matrix                ##
##############################################################

if "Opt" not in os.listdir():
	os.mkdir("Opt")
	os.chdir("Opt")
	optDir = os.getcwd()
	os.mkdir("Data")
	os.chdir("Data")
	os.mkdir("Simulations")
	os.mkdir("ResponseSurface")
	os.chdir("..")
else:
	os.chdir("Opt")
	optDir = os.getcwd()

if os.path.isfile("progress") == False:
	FlameMaster_Execution_location = SSM.make_dir_in_parallel(yaml_loc_dict)
	#raise AssertionError("The Target class, Uncertainty class, Design Matrix and Simulations")
else:
	item("Progress file detected -- resuming from existing simulation locations")
	progress = open(optDir+"/progress",'r').readlines()
	FlameMaster_Execution_location = []
	if os.path.isfile(optDir+"/locations"):
		with open(optDir+"/locations") as infile:
			for line in infile:
				FlameMaster_Execution_location.append(line)
	else:
		# 2026-08-27 resume self-heal: a FIRST run interrupted mid-Phase-7
		# leaves `progress` (Worker callback) but never `locations` --
		# make_dir_in_parallel only writes that file in its resumed-case
		# branch, which a first run never takes. Reconstruct the list from
		# the existing case dirs exactly as that branch does
		# (getDirectoryList chdir's into case-N and does not chdir back),
		# then persist it so later resumes take the fast path. Hit live on
		# pilot_5case_case0: restart crashed at this line with
		# FileNotFoundError: Opt/locations.
		item("locations file missing -- reconstructing from existing case dirs")
		for case_index in case_dir:
			try:
				_, _, _, _, run_list, _ = SSM.getDirectoryList(case_index, case_index, yaml_loc_dict)
				for key in sorted(run_list, key=lambda k: int(k) if str(k).isdigit() else k):
					FlameMaster_Execution_location.append(run_list[key] + "/run\n")
			finally:
				os.chdir(optDir)
		with open(optDir+"/locations", "w") as lf:
			lf.writelines(FlameMaster_Execution_location)
	#FlameMaster_Execution_location = open(optDir+"/locations",'r').readlines()
	#missing_location = data_management.find_missing_location(FlameMaster_Execution_location, progress)

phase_end(7)
phase(8, TOTAL_PHASES, "Response extraction")
#############################################
##     Extracting simulation results       ##
##       (The module if validated          ##
#############################################
temp_sim_opt = {}
# kept_row_index: {case_str -> np.ndarray of design-matrix row indices, one
# per entry in temp_sim_opt[case_str], IN ORDER}. X/Y ALIGNMENT FIX
# (2026-08-19): sim_data_case-N.lst's first column (`folderName` in
# data_management.generate_target_value_tables) IS the design-matrix row
# index each response came from -- but the ORIGINAL reader here discarded
# it (`i.split("\t")[1]` only) and assumed positional alignment (yData[k]
# <-> xData[k]) instead. generate_target_value_tables DROPS failed-
# simulation rows from its output entirely (they go to failed_sim, not
# data), so a single failed simulation anywhere in a case would silently
# shift every subsequent row's pairing by one -- benign for THIS run (every
# failed_sim_data_case-*.lst on disk is currently 0 bytes) but a live
# correctness bug the moment any simulation fails. Now parsed and carried
# through explicitly; the per-case PRS loop below indexes xData by
# kept_row_index[case] instead of assuming a 1:1 positional match.
kept_row_index = {}
for case in case_dir:
	os.chdir("Data/Simulations")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		lines = [ln for ln in open("sim_data_case-"+str(case)+".lst").readlines() if ln.strip()]
	else:
		lines = []
	if lines:
		# 2026-08-26 empty-lst gate: trust the cached sim_data_case-N.lst
		# only when it has >=1 non-empty row. An EMPTY .lst left behind by
		# a previous failed run used to short-circuit regeneration
		# permanently (this branch never called
		# generate_target_value_tables again), starving the case of
		# responses. Now it falls through to the regenerate branch below.
		idx = np.array([int(ln.split("\t")[0]) for ln in lines])
		ETA = [float(ln.split("\t")[1]) for ln in lines]
		temp_sim_opt[str(case)] = ETA
		kept_row_index[str(case)] = idx
		os.chdir(optDir)
	else:
		os.chdir(optDir)
		os.chdir("case-"+str(case))
		data_sheet,failed_sim, ETA = data_management.generate_target_value_tables(FlameMaster_Execution_location, target_list, case, fuel,input_=optInputs)
		temp_sim_opt[str(case)] = ETA
		lines = [ln for ln in data_sheet.splitlines() if ln.strip()]
		kept_row_index[str(case)] = np.array([int(ln.split("\t")[0]) for ln in lines])
		if len(kept_row_index[str(case)]) != len(ETA):
			warn(f"[Case-{case}] row-index/ETA length mismatch after simulation "
			     f"({len(kept_row_index[str(case)])} indices vs {len(ETA)} ETA values) -- "
			     f"falling back to positional 0..N-1 indexing for this case; "
			     f"inspect sim_data_case-{case}.lst by hand.")
			kept_row_index[str(case)] = np.arange(len(ETA))
		f = open('../Data/Simulations/sim_data_case-'+str(case)+'.lst','w').write(data_sheet)
		g = open('../Data/Simulations/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim)
		os.chdir(optDir)
item(f"Responses extracted for {len(case_dir)} case(s)", "ok")

#######################################
### Using different Tig definition   ##
###                                  ##
#######################################

'''
case_dict = {}
for case in case_dir:
    case_pattern = f'case-{case}'
    case_dict[case_pattern] = []
    for path in FlameMaster_Execution_location:
        path = path.strip()
        path_list = path.split('/')
        if case_pattern in path_list:
            case_dict[case_pattern].append(path[:-3]+'time_history.csv')


onset_times_dict = {}
for key in case_dict:
    #time_dict = {}
    #Pressure_list = {}
    onset_times = []
    for loc in case_dict[key]:  #Each sample of a case
        data = pd.read_csv(loc, skipinitialspace=True, sep=',')  #Reading Modified.csv
        data.columns = data.columns.str.strip()  # Strip whitespace from column names in modified.csv
        time_column = 'Time  (sec)'
        pressure_column = 'Pressure  (bar)'
        time_ = (data[time_column]).tolist()   #Corresponds to modified.csv
        P_ = data[pressure_column].tolist()     #Corresponds to modified.csv
        #print(time_)
        #print(P_)
        interpolator = Akima1DInterpolator(time_, P_)
        ###########################################
        dt_onset = np.diff(time_)
        dX_onset = np.diff(P_)
        slope_onset = dX_onset/dt_onset
        change_of_slope = np.diff(slope_onset)
        max_of_change_of_slope = change_of_slope.argmax()  # Index of maximum change in slope
        onset_time = time_[max_of_change_of_slope]
        P_at_onset_time = P_[max_of_change_of_slope]
        #print(max_of_change_of_slope,max_of_slope)
        #raise AssertionError("STOP")
        onset_times.append(onset_time)
        
        plt.plot(time_, P_,"b-", linewidth=0.5)  # Line plot with markers
        plt.xlim([0, onset_time*1.50]) 
        plt.scatter(onset_time, P_at_onset_time, color="red", marker = 'x', label="ONSET_RISE", s=100)  # Point as a red dot
        #plt.text(onset_time, P_at_onset_time, f"({dt_max:.2f})", fontsize=8, color="red", ha="left")  # Label the point
       
        plt.title(f"Sample Analysis")
        plt.xlabel("Time (ms)")
        plt.ylabel("X-OH Concentration")
        
        plt.legend()
        plt.grid(True)
        plot_filename = f"plot_sample_1.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # Close the figure to avoid overlapping plots
        raise AssertionError("Onset def check!")
        #time_dict_us = time*1000
        #print(data_dict)
    onset_times_dict[key] = onset_times
    
#raise AssertionError("ALERT!")
'''
###############################################
##      Generating the response surface      ##
##                                           ##
###############################################

_STUDY_MARKER = os.path.join(os.path.dirname(optDir), ".study_manifest.json")


def archive_previous_study(optDir, sample_mode, outlier_cfg,
                           preserve_outlier_sims=False):
	"""Move the PREVIOUS resampling study's artifacts into
	CASE_STUDIES/<method>_<params>_<timestamp>/ before this run starts a new
	one, so each study's simulations/YAMLs/PRS coefficients/plots stay
	available for later review instead of being silently overwritten by the
	next run (per instruction: "run cleanly" means the outlier-resampling
	artifacts move, not that the whole run starts over).

	Moved: Opt/Outlier_Resample/, Opt/Data/Outlier_Simulations/,
	       Opt/Data/ResponseSurface/, Plots/, Prediction_errors.csv,
	       Prediction_errors_detailed.csv.
	Preserved (NOT moved): Opt/Data/Simulations/ (the expensive main-DM
	       simulation .lst files) and DM_FOR_PARTIAL_PRS/ / YAML_FILES_FOR_PARTIAL_PRS/
	       (the main design matrices) -- these reload automatically at the top
	       of this script and must survive across studies.

	The archive is tagged with the PREVIOUS run's OWN sample_mode/config (read
	from ``.study_manifest.json``, written by that run just below), not this
	run's -- otherwise, e.g., archiving a 'hybrid' study right before starting
	a 'd_optimal' one would mislabel the hybrid results as "d_optimal_...".
	Falls back to "unknown" if no marker exists (e.g. very first archive).
	"""
	run_root = os.path.dirname(optDir)
	movable = [
		(os.path.join(optDir, "Outlier_Resample"), os.path.join("Opt", "Outlier_Resample")),
		(os.path.join(optDir, "Data", "Outlier_Simulations"), os.path.join("Opt", "Data", "Outlier_Simulations")),
		(os.path.join(optDir, "Data", "ResponseSurface"), os.path.join("Opt", "Data", "ResponseSurface")),
		(os.path.join(run_root, "Plots"), "Plots"),
		(os.path.join(run_root, "Prediction_errors.csv"), "Prediction_errors.csv"),
		(os.path.join(run_root, "Prediction_errors_detailed.csv"), "Prediction_errors_detailed.csv"),
		# optimization_cost.csv (2026-08-19): now a per-run ledger, reset
		# fresh every run via reset_csv rather than append-forever via
		# initialize_csv (the latter is why it had accumulated 697
		# duplicate rows across reruns) -- archive the previous run's copy
		# here, same as the two Prediction_errors files, so no history is
		# lost by truncating it.
		(os.path.join(run_root, "optimization_cost.csv"), "optimization_cost.csv"),
		(os.path.join(run_root, "screening_summary.csv"), "screening_summary.csv"),
		# experiment-arm artifacts (2026-08-28) -- archived like the rest so
		# successive runs never overwrite a previous run's encoders/plots.
		(os.path.join(optDir, "Data", "Encoding_Decoding"), os.path.join("Opt", "Data", "Encoding_Decoding")),
		(os.path.join(run_root, "Plots_transfrom"), "Plots_transfrom"),
		(os.path.join(run_root, "Plots_re-sampling"), "Plots_re-sampling"),
		(os.path.join(run_root, "Plots_re-sampling_transform"), "Plots_re-sampling_transform"),
		(os.path.join(run_root, "prs_experiment_results.csv"), "prs_experiment_results.csv"),
	]
	if not any(os.path.exists(src) for src, _ in movable):
		return  # nothing from a previous study on disk yet -- nothing to archive

	prev_manifest = {}
	if os.path.isfile(_STUDY_MARKER):
		try:
			prev_manifest = json.load(open(_STUDY_MARKER))
		except (ValueError, OSError):
			prev_manifest = {}
	prev_mode = prev_manifest.get("sample_mode", "unknown")
	prev_cfg = prev_manifest.get("outlier_cfg", {})

	ts = time.strftime("%Y%m%d-%H%M%S")
	tag = (f"{prev_mode}_tol{prev_cfg.get('band_tolerance', '')}"
	       f"_alpha{prev_cfg.get('acquisition_alpha', '')}_{ts}")
	dest_root = os.path.join(run_root, "CASE_STUDIES", tag)
	os.makedirs(dest_root, exist_ok=True)

	for src, rel in movable:
		if os.path.exists(src):
			dest = os.path.join(dest_root, rel)
			os.makedirs(os.path.dirname(dest), exist_ok=True)
			if preserve_outlier_sims and rel == os.path.join("Opt", "Data", "Outlier_Simulations"):
				# The cross-arm fingerprint cache: run N+1 (resampling_transform)
				# must reuse run N's (resampling) augmentation .lst via the
				# SHA1-verified cache in Phase10PRS._simulate_new_samples --
				# moving it into the archive would delete that cache. Copy it
				# (history kept) and leave the live cache in place.
				shutil.copytree(src, dest, dirs_exist_ok=True)
				continue
			shutil.move(src, dest)

	# Opt/Data/ResponseSurface must exist fresh for THIS run's coefficient
	# cache -- archiving it wholesale (correctly clears any stale cache from a
	# different method/study) means it has to be recreated immediately after.
	os.makedirs(os.path.join(optDir, "Data", "ResponseSurface"), exist_ok=True)

	with open(os.path.join(dest_root, "manifest.json"), "w") as f:
		json.dump({"archived_at": ts, "sample_mode": prev_mode,
		           "outlier_cfg": prev_cfg}, f, indent=2, default=str)
	print(f"\nArchived previous resampling study ({prev_mode}) -> {dest_root}\n")


_preserve_sims = (experiment_cfg["mode"] in ("resampling", "resampling_transform")
                  and not experiment_cfg["resampling"]["dry_run"])
archive_previous_study(optDir, outlier_cfg["sample_mode"], outlier_cfg,
                       preserve_outlier_sims=_preserve_sims)
reset_csv(COST_FILE, HEADERS)
reset_csv(PRS_ERROR_FILE, PRS_HEADER)
reset_csv(PRS_DETAIL_FILE, PRS_DETAIL_HEADER)
SCREENING_SUMMARY_FILE = "../screening_summary.csv"  # see COST_FILE's PATH NOTE above -- same reasoning
reset_csv(SCREENING_SUMMARY_FILE, [
	"Case_ID", "Rows_Generated", "Rows_BandInvalid", "Rows_FailedSim",
	"Rows_Sentinel", "Rows_NonFinite", "Rows_NonPhysical", "Rows_Good",
	"Good_Fraction_Pct", "Worst_Violation_Ratio", "Median_Violation_Ratio",
])

# Record THIS run's config so a FUTURE run's archive_previous_study labels
# what it archives correctly (see that function's docstring).
with open(_STUDY_MARKER, "w") as f:
	json.dump({"sample_mode": outlier_cfg["sample_mode"], "outlier_cfg": outlier_cfg,
	           "prs_fit_cfg": prs_fit_cfg, "started_at": time.strftime("%Y%m%d-%H%M%S")},
	          f, indent=2, default=str)

phase_end(8)
phase(9, TOTAL_PHASES, "Class-pool integrity (pre-Stage-3)")

def _refresh_pool_worker(args):
	"""multiprocessing.Pool worker: refresh ONE case's class-A/B/C pool.
	Top-level (picklable) by design. Single-threaded BLAS per worker, per
	this project's established parallel-sweep pattern (see
	feedback-long-running-numerical-runs memory) -- 68 independent,
	CPU-bound, numpy-only tasks is exactly the (case x work) cross-product
	that pattern targets, not a naive per-case-only parallelization.
	"""
	case_index, sim_est, n_active, method = args
	os.environ["OMP_NUM_THREADS"] = "1"
	os.environ["OPENBLAS_NUM_THREADS"] = "1"
	os.environ["MKL_NUM_THREADS"] = "1"
	try:
		d = DM.DesignMatrix(unsrt_data, design_type, sim_est, n_active)
		prov = d.refresh_class_pools(
			case_index, selected_params_dict[case_index], method=method,
			base_dir=home_dir)
		return (case_index, True, prov.get("verified_max_ratio", None), None)
	except Exception as exc:
		return (case_index, False, None, f"{type(exc).__name__}: {exc}")

if design_type == "A1+B1+C1" and outlier_cfg["enable"]:
	# Only cases whose pool is NOT already stamped fresh for the CURRENT
	# Uncertainty.SAMPLER_VERSION need work -- pool_cache_is_fresh is the
	# same check DesignMatrix.refresh_class_pools itself uses, checked here
	# first purely to report an accurate "N of M need refreshing" count and
	# to size the parallel pass correctly.
	_stale_cases = []
	for case in case_dir:
		case_index = int(case)
		cdir = os.path.join(home_dir, "DM_FOR_PARTIAL_PRS", str(case_index))
		if not DM.pool_cache_is_fresh(cdir, outlier_cfg["method"]):
			_stale_cases.append(case_index)

	if not _stale_cases:
		item(f"All {len(case_dir)} case class-pools already fresh "
		     f"(SAMPLER_VERSION={uncertainty.SAMPLER_VERSION!r}) -- nothing to refresh.", "ok")
	else:
		warn(f"{len(_stale_cases)}/{len(case_dir)} case class-pools predate "
		     f"SAMPLER_VERSION={uncertainty.SAMPLER_VERSION!r} (or were built with a "
		     f"different method) -- these are what let Stage-3 D-optimal resampling "
		     f"silently reject 80-100% of its candidates (see Case 52/56 in "
		     f"report_resampling/DEBUG_REPORT.md). Refreshing now: pure numpy, zero "
		     f"new simulations, serially per case (nested pools are illegal under "
		     f"a daemonic parent -- see the 2026-08-26 fix below).")
		_refresh_args = [
			(ci, getSim(cost_estimate_dict[ci]["Active_Rxns"], design_type),
			 len(activeParameters), outlier_cfg["method"])
			for ci in _stale_cases
		]
		step(f"Refreshing {len(_stale_cases)} case pool(s) serially")
		_t0 = time.time()
		# 2026-08-26 fix: refresh SERIALLY in the main process. The previous
		# multiprocessing.Pool wrapper failed with "AssertionError: daemonic
		# processes are not allowed to have children": the pool's workers are
		# daemonic, and refresh_class_pools itself spawns its own
		# multiprocessing pool for the class-B sampler -- Python forbids a
		# daemonic process from spawning children. Serial execution keeps the
		# inner pool legal (the main process is not daemonic). Per-case
		# refresh is pure numpy and takes seconds-to-minutes per case, so the
		# lost cross-case parallelism is negligible next to the Stage-3
		# simulation budget that follows.
		_results = [_refresh_pool_worker(args) for args in _refresh_args]
		_ok = [r for r in _results if r[1]]
		_bad = [r for r in _results if not r[1]]
		item(f"Refreshed {len(_ok)}/{len(_stale_cases)} case pool(s) in "
		     f"{time.time()-_t0:.1f}s",
		     "ok" if not _bad else "warn")
		if _bad:
			for ci, _, _, err_msg in _bad:
				error(f"Case-{ci} pool refresh FAILED: {err_msg}")
			raise RuntimeError(
				f"Class-pool refresh failed for {len(_bad)} case(s): "
				f"{[ci for ci, *_ in _bad]}. Stage 3 cannot safely run for these "
				f"cases until the underlying sampler issue is fixed -- see the "
				f"errors printed above.")
else:
	item(f"Class-pool refresh skipped: design_type={design_type!r} or "
	     f"outlier_cfg['enable']={outlier_cfg['enable']} (Stage 3 only applies "
	     f"to A1+B1+C1 with resampling enabled).")

phase_end(9)
phase(10, TOTAL_PHASES, "Per-case PRS fitting")

# ── Phase 10 (per-case p-PRS fitting) now lives in phase10_prs.py ──────────
# (2026-08-27): the whole pipeline -- screening -> 3-way split -> (tol, k,
# order, ridge) search -> identifiable-subspace PLS + PRS fit -> Stage-3
# D-optimal outlier retraining -> one-time hold-out verdict -- was moved into
# the Phase10PRS class with step-by-step STEP tracing. See phase10_prs.py's
# module docstring for the full walkthrough and the offline reproducer
# (Phase10CaseDiagnostics) that re-runs one case's fit with no Chemkin.
from phase10_prs import Phase10PRS
_phase10 = Phase10PRS({
	"target_list": target_list, "unsrt_data": unsrt_data, "optInputs": optInputs,
	"design_matrix_dict": design_matrix_dict, "kept_row_index": kept_row_index,
	"selected_params_dict": selected_params_dict, "temp_sim_opt": temp_sim_opt,
	"no_of_sim": no_of_sim, "cost_estimate_dict": cost_estimate_dict,
	"optDir": optDir, "home_dir": home_dir, "fuel": fuel,
	"PRS_type": PRS_type, "design_type": design_type, "MRE_criteria": MRE_criteria,
	"prs_fit_cfg": prs_fit_cfg, "outlier_cfg": outlier_cfg,
	"experiment_cfg": experiment_cfg,
	"modules": {"IPRS": IPRS, "PRS": PRS, "AR": AR,
	            "simulator": simulator, "data_management": data_management},
	"csv_files": {"PRS_ERROR_FILE": PRS_ERROR_FILE, "PRS_DETAIL_FILE": PRS_DETAIL_FILE,
	              "SCREENING_SUMMARY_FILE": SCREENING_SUMMARY_FILE, "COST_FILE": COST_FILE},
	"log": {"step": step, "kv": kv, "table": table, "result": result,
	        "warn": warn, "error": error, "append_csv_row": append_csv_row},
})
ResponseSurfaces = _phase10.run()

phase_end(10)

##################################################
##        Optimization Procedure                ##
##   Inputs: Target list and Response Surfaces  ##
##################################################
phase(11, TOTAL_PHASES, "Optimization & posterior")

from VisualAid import PostOptPlotter

if "solution_zeta.save" not in os.listdir():

    optimizer_obj  = Optimizer(target_list)
    opt, opt_zeta, posterior_cov = optimizer_obj.run_optimization_with_selected_PRS(
        unsrt_data, ResponseSurfaces, optInputs)

    # run_optimization_with_selected_PRS's fresh-optimization paths never
    # compute a real posterior covariance -- it returns cov=[] unconditionally
    # in every branch (OptimizationTool.py's own "icov = J^T J; cov = inv(icov)"
    # posterior-covariance lines are commented out). Recompute it here with the
    # same MUM-PCE Gauss-Newton formula the RELOAD branch below already uses
    # (OptimizationTool2_0.build_posterior_covariance) instead of saving/using
    # an empty placeholder -- an empty posterior_cov used to crash the very
    # next section ("Sigma_p = posterior_cov['Sigma_p']") for every case
    # needing a fresh (not reloaded) optimization, killing the whole run.
    if not isinstance(posterior_cov, dict):
        from OptimizationTool2_0 import OptimizationTool as _OT
        from OptimizationTool2_0 import CodecEngine
        _tmp = _OT(target_list)
        _tmp.unsrt            = unsrt_data
        _tmp.ResponseSurfaces = ResponseSurfaces
        _tmp.rxn_index        = list(unsrt_data.keys())
        _tmp.codec            = CodecEngine(unsrt_data)
        Sigma_zeta, Sigma_p, H_gn, rxn_slices = \
            _tmp.build_posterior_covariance(opt_zeta)
        posterior_cov = {"Sigma_zeta": Sigma_zeta, "Sigma_p": Sigma_p,
                         "H": H_gn, "rxn_slices": rxn_slices}

    # ── Save solution ──────────────────────────────────────────────────
    string_save = ""
    for i, j in enumerate(activeParameters):
        print("\n{}=\t{}".format(activeParameters[i], opt_zeta[i]))
        string_save += "{}=\t{}\n".format(activeParameters[i], opt_zeta[i])
    open("solution_zeta.save", "w").write(string_save)

    # ── Save posterior covariance matrices ─────────────────────────────
    with open("posterior_cov.pkl", "wb") as _f:
        pickle.dump(posterior_cov, _f)

    # ── Write optimised mechanism ──────────────────────────────────────
    originalMech  = Parser(mech_file_location).mech
    copy_of_mech  = deepcopy(originalMech)
    new_mechanism, a = Manipulator(copy_of_mech, unsrt_data, opt_zeta).doPerturbation()
    open("Optimized_mech.yaml", "w").write(yaml.safe_dump(new_mechanism, default_flow_style=False))

else:
    # ── Reload existing solution ───────────────────────────────────────
    save     = open("solution_zeta.save", "r").readlines()
    opt_zeta = [float(i.split("=")[1].strip()) for i in save]

    originalMech  = Parser(mech_file_location).mech
    copy_of_mech  = deepcopy(originalMech)
    new_mechanism, a = Manipulator(copy_of_mech, unsrt_data, opt_zeta).doPerturbation()
    open("Optimized_mech.yaml", "w").write(yaml.safe_dump(new_mechanism, default_flow_style=False))

    # Recompute posterior covariance if cache exists, else recalculate
    if "posterior_cov.pkl" in os.listdir():
        with open("posterior_cov.pkl", "rb") as _f:
            posterior_cov = pickle.load(_f)
    else:
        # ResponseSurfaces and unsrt_data are already in scope from above
        from OptimizationTool2_0 import OptimizationTool as _OT
        _tmp = _OT(target_list)
        _tmp.unsrt          = unsrt_data
        _tmp.ResponseSurfaces = ResponseSurfaces
        _tmp.rxn_index      = list(unsrt_data.keys())
        _tmp.codec          = None
        # Build codec so unflatten_zeta works
        from OptimizationTool2_0 import CodecEngine
        _tmp.codec = CodecEngine(unsrt_data)
        Sigma_zeta, Sigma_p, H_gn, rxn_slices = \
            _tmp.build_posterior_covariance(opt_zeta)
        posterior_cov = {"Sigma_zeta": Sigma_zeta, "Sigma_p": Sigma_p,
                         "H": H_gn, "rxn_slices": rxn_slices}
        with open("posterior_cov.pkl", "wb") as _f:
            pickle.dump(posterior_cov, _f)


##################################################
##   Per-reaction optimisation + posterior plots ##
##################################################

os.makedirs("Opt/Plots/Posterior", exist_ok=True)

Sigma_p    = posterior_cov["Sigma_p"]
rxn_slices = posterior_cov["rxn_slices"]
opt_zeta_arr = np.asarray(opt_zeta, dtype=float)

# Build per-reaction zeta from the flat vector
# (same logic as codec.unflatten_zeta but done inline so no dependency)
_offset = 0
for rxn in unsrt_data:
    m_j      = len(unsrt_data[rxn].activeParameters)
    z_rxn    = opt_zeta_arr[_offset: _offset + m_j]
    _offset += m_j

    sl       = rxn_slices[rxn]
    Sp_rxn   = Sigma_p[sl, sl]

    # Full 3-element zeta (pad zeros for partial-PRS reactions)
    z_full = np.zeros(3, dtype=float)
    sel    = np.asarray(unsrt_data[rxn].selection, dtype=int)
    act    = [i for i, s in enumerate(sel) if s == 1]
    for local_i, global_i in enumerate(act):
        if local_i < len(z_rxn):
            z_full[global_i] = z_rxn[local_i]

    plotter = PostOptPlotter(
        unsrt_data = unsrt_data,
        rxn        = rxn,
        zeta_opt   = z_full,
        Sigma_p_rxn = Sp_rxn,
        n_sigma    = 1,
    )
    out = plotter.plot(location="Opt/Plots/Posterior")
    print(f"  Saved posterior plot → {out}")

print("\nAll posterior plots saved to  Opt/Plots/Posterior/")
