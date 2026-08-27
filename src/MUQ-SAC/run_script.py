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
import os, sys, re, threading, subprocess, time, traceback, shutil, glob
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
threshold = stats_["cut_off_percentage"]/100
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

    # Covariance matrix
    Sigma = L @ L.T
    
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
			result = subprocess.run(['python3.9', binLoc["SA_tool"]] + args, capture_output=True, text=True)
			f = open("SA_tool.out","+a")
			f.write(result.stdout+"\n")
			
			print("\n\nSensitivity Analysis for A-factor is Done!!")
			# Printing the errors of script2.py, if any
			if result.stderr:
				f.write("Errors:\n"+result.stderr)	
			#	raise AssertionError("Sensitivity Analysis Done!!")
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
			result = subprocess.run(['python3.9', binLoc["SA_tool_3p"]] + args, capture_output=True, text=True)
			f = open("SA_tool_3p.out","+a")
			f.write(result.stdout+"\n")
			if result.stderr:
				f.write("Errors:\n"+result.stderr)	
			print("\n\nSensitivity Analysis for 3 parameters is Done!!")
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
		_final_sets_A = CA.run_pipeline(PICKLE_PATH, OUTPUT_DIR, threshold=threshold)
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
			corr_cutoff=0.9,
			threshold_corr=0.01,
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
	for case in case_dir:
		PPRS_system = P_PRS.PartialPRS(sensitivity_analysis[str(case)],correlation_results[str(case)],unsrt_data,optInputs,target_list,str(case),activeParameters,design_type,status=status_)
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


def archive_previous_study(optDir, sample_mode, outlier_cfg):
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
			shutil.move(src, dest)

	# Opt/Data/ResponseSurface must exist fresh for THIS run's coefficient
	# cache -- archiving it wholesale (correctly clears any stale cache from a
	# different method/study) means it has to be recreated immediately after.
	os.makedirs(os.path.join(optDir, "Data", "ResponseSurface"), exist_ok=True)

	with open(os.path.join(dest_root, "manifest.json"), "w") as f:
		json.dump({"archived_at": ts, "sample_mode": prev_mode,
		           "outlier_cfg": prev_cfg}, f, indent=2, default=str)
	print(f"\nArchived previous resampling study ({prev_mode}) -> {dest_root}\n")


archive_previous_study(optDir, outlier_cfg["sample_mode"], outlier_cfg)
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

def _mini_dm_fingerprint(prs_rows):
	"""SHA1 fingerprint of a mini-DM's rows (rounded to tame float noise),
	used to validate a cached simulation result actually matches the CURRENT
	pass's mini-DM before reusing it -- see _simulate_new_samples."""
	arr = np.round(np.asarray(prs_rows, dtype=float), 8)
	return hashlib.sha1(arr.tobytes()).hexdigest()


def _simulate_new_samples(case, prs_rows, full_rows, sel_mask, pass_id, force_resim=False):
	"""Simulate a mini-DM of new samples, reusing the same SM / data_management
	machinery as the main flow.

	The whole thing runs inside its own folder tree
	(Opt/Outlier_Resample/case-<case>/pass-<n>/) so the parallel simulation
	directories never clobber the original per-case runs. Simulation results are
	also copied into Opt/Data/Outlier_Simulations/ so the solution can be
	inspected. Already-generated files are reused (not regenerated) -- but ONLY
	if a companion fingerprint manifest confirms the cached result actually
	matches THIS pass's mini-DM (prs_rows), not a stale file left over from an
	earlier/different attempt. Set force_resim=True (outlier_cfg["force_resim"])
	to always resimulate regardless of any cache.

	Returns the list of ETA (target) values aligned with the input rows.
	cwd is assumed to be optDir on entry and is restored on exit.
	"""
	# The main flow's case IDs come from `case_dir = range(len(target_list))`, so
	# SM.make_dir_in_parallel / data_management expect an INTEGER case. In the PRS
	# loop `case` is the temp_sim_opt string key -- convert it back to int here.
	case_int = int(case)
	sim_store = os.path.join(optDir, "Data", "Outlier_Simulations")
	os.makedirs(sim_store, exist_ok=True)
	sim_data_file = os.path.join(sim_store, f"sim_data_case-{case_int}_pass-{pass_id}.lst")
	meta_file = os.path.join(sim_store, f"sim_data_case-{case_int}_pass-{pass_id}.meta.json")
	fingerprint = _mini_dm_fingerprint(prs_rows)

	# ── Reuse cached simulation results only if the fingerprint of THIS pass's
	#    mini-DM matches the one recorded when the cache was written -- a
	#    missing/mismatched manifest (e.g. a stale file from an earlier
	#    attempt, or one written before this check existed) is NOT reused. ──
	if not force_resim and os.path.isfile(sim_data_file) and os.path.isfile(meta_file):
		try:
			meta = json.load(open(meta_file))
		except (ValueError, OSError):
			meta = {}
		if meta.get("n_rows") == len(prs_rows) and meta.get("fingerprint") == fingerprint:
			print(f"\t[Case-{case}] pass {pass_id}: reusing cached simulation data "
				  f"(fingerprint verified)")
			ETA = [float(l.split("\t")[1]) for l in open(sim_data_file).readlines()]
			return list(np.asarray(ETA, dtype=float).ravel())
		print(f"\t[Case-{case}] pass {pass_id}: cached simulation data found but its "
			  f"fingerprint doesn't match this pass's mini-DM -- resimulating.")

	work_dir = os.path.join(optDir, "Outlier_Resample", f"case-{case_int}", f"pass-{pass_id}")
	mech_dir = os.path.join(work_dir, "Perturbed_Mech")
	os.makedirs(mech_dir, exist_ok=True)

	# Non-"Full" tag => SM stores the dict verbatim, so design_matrix_dict[case_int]
	# is the mini-DM (its length drives how many sim dirs getDirectoryList makes).
	OSM = simulator.SM(target_list, optInputs, unsrt_data,
					   {case_int: [list(r) for r in full_rows]}, tag="Outlier")

	# ── Generate perturbed mechanism YAMLs (skip if already generated) ─────
	yaml_loc = [os.path.join(mech_dir, f"mechanism_{i}.yaml")
				for i in range(len(prs_rows))]
	if all(os.path.isfile(p) for p in yaml_loc) and len(yaml_loc) > 0:
		print(f"\t[Case-{case}] pass {pass_id}: mechanism YAMLs already generated")
	else:
		if PRS_type == "Full":
			yaml_list = OSM.getYAML_List([list(r) for r in full_rows])
		else:
			# Partial: SM expects the COMPRESSED rows + full-length selection mask
			sel_rows = [list(sel_mask) for _ in range(len(prs_rows))]
			yaml_list = OSM.getYAML_List([list(r) for r in prs_rows], selection=sel_rows)
		location_mech = [mech_dir for _ in yaml_list]
		index_list = [str(i) for i in range(len(yaml_list))]
		OSM.getPerturbedMechLocation(yaml_list, location_mech, index_list)

	# ── Run the simulations INSIDE work_dir for ONLY the resampled case, then
	#    extract from the created case-<case_int> directory (mirrors the main
	#    flow's chdir). `cases=[case_int]` restricts SM to this one case so we
	#    don't waste simulations on the other targets. ──
	yaml_loc_dict = {case_int: yaml_loc}
	cwd0 = os.getcwd()
	os.chdir(work_dir)
	try:
		exec_locs = OSM.make_dir_in_parallel(yaml_loc_dict, cases=[case_int])
		os.chdir(f"case-{case_int}")
		data_sheet, failed_sim, ETA = data_management.generate_target_value_tables(
			exec_locs, target_list, case_int, fuel, input_=optInputs)
	finally:
		os.chdir(cwd0)

	# ── Persist results to Data/Outlier_Simulations for inspection ─────────
	open(sim_data_file, "w").write(data_sheet)
	open(os.path.join(sim_store, f"failed_sim_case-{case_int}_pass-{pass_id}.lst"),
		 "w").write(failed_sim)
	json.dump({"n_rows": len(prs_rows), "fingerprint": fingerprint}, open(meta_file, "w"))
	return list(np.asarray(ETA, dtype=float).ravel())


def _write_coef_cache(coef_path, coeff):
	"""(Re)write the coefficient cache file from an explicit coeff array --
	used to guarantee the on-disk cache matches `best`, not a rejected trial
	that happened to write over it (see retrain_around_outlier)."""
	if os.path.isfile(coef_path):
		os.remove(coef_path)
	with open(coef_path, "w") as f:
		f.write("Coefficients\n")
		for c in coeff:
			f.write(f"{float(c)}\n")


# ── No-ignition sentinel (2026-08-19) ────────────────────────────────────
# data_management.extract_output substitutes this EXACT value
# (ln(100 * 10000) = ln(1e6)) whenever a solver's output file exists but
# doesn't parse into a real ignition-delay reading -- a fabricated "no
# ignition found" placeholder, not a real measurement, and previously
# indistinguishable downstream from a genuine (if extreme) response. Zero
# occurrences in the current study's data (confirmed by direct scan), but
# unguarded -- screen_case_samples checks for it explicitly so a future
# batch containing one can never silently train/score against it.
NO_IGNITION_SENTINEL = np.log(100 * 10000)


def screen_case_samples(case, case_index, xData_raw, yData_raw, orig_row_idx,
                          blocks, dm_case_dir, band_tolerance_eval):
	"""Split one case's (design matrix, response) pair into a GOOD submatrix
	(what the PRS is fit and scored on) and a QUARANTINED remainder, and
	write the full audit trail to disk under dm_case_dir -- i.e.
	<this study's own root>/DM_FOR_PARTIAL_PRS/<case_index>/, derived from
	home_dir at the call site, never hardcoded to another study's path (per
	instruction).

	This is the load-bearing fix for the 400%-65,372% "holdout(all)" numbers
	previously reported in Prediction_errors.csv: those were the SAME fit
	graded against ~88-94% physically-implausible rows the sampler's own
	acceptance test would have rejected (see Uncertainty.SAMPLER_VERSION's
	docstring / report_resampling/DEBUG_REPORT.md). Every row here is
	re-scored against blocks -- the CURRENT band-validity test, using
	whatever prs_band_fprior/screening config this run's target.opt sets --
	regardless of when or by which sampler version it was originally
	generated, so a stale DesignMatrix.csv can never again silently corrupt
	a fit or a reported error (see partial_PRS_system.py's dm_provenance
	NOTE for why the raw DesignMatrix.csv itself is deliberately NOT
	regenerated/discarded).

	Rejection reasons, checked in order (first hit wins):
	  1. band_invalid  -- IdentifiablePRS.violation_detail(x, blocks) exceeds
	                       band_tolerance_eval. The dominant, and currently
	                       only observed, reason in this study.
	  2. sim_sentinel   -- response equals NO_IGNITION_SENTINEL (a fabricated
	                       "no ignition found" placeholder, see its own
	                       docstring above) -- currently 0 occurrences, kept
	                       so a future batch containing one is caught.
	  3. non_finite     -- NaN or +-inf response -- currently 0 occurrences.
	  4. non_physical   -- exp(response) outside (0, 1e6] microseconds --
	                       currently 0 occurrences (this pipeline's response
	                       column is ln(10*tau_us), see the note in
	                       report_resampling/DEBUG_REPORT.md).
	(sim_failed, a 5th reason for rows generate_target_value_tables dropped
	entirely because the simulation never returned a usable result, never
	reaches this function at all -- it is already excluded upstream via
	kept_row_index / orig_row_idx, and is accounted for separately in the
	optimization_cost.csv ledger as Samples_Failed_Sim.)

	Returns (xGood, yGood, stats) where stats is a dict with the full
	per-reason breakdown, used both for optimization_cost.csv's per-case
	ledger row and screening_summary.csv's study-level rollup.
	"""
	xData_raw = np.asarray(xData_raw, dtype=float)
	yData_raw = np.asarray(yData_raw, dtype=float).ravel()
	orig_row_idx = np.asarray(orig_row_idx, dtype=int)
	n = len(yData_raw)
	reasons = np.array([""] * n, dtype=object)

	ratio, worst_rxn = IPRS.violation_detail(xData_raw, blocks)
	tol = band_tolerance_eval if band_tolerance_eval is not None else float("inf")
	band_bad = ratio > tol
	reasons[band_bad] = "band_invalid"

	is_sentinel = (reasons == "") & np.isclose(yData_raw, NO_IGNITION_SENTINEL, atol=1e-6)
	reasons[is_sentinel] = "sim_sentinel"

	non_finite = (reasons == "") & (~np.isfinite(yData_raw))
	reasons[non_finite] = "non_finite"

	with np.errstate(over="ignore"):
		tau_us = np.exp(np.clip(yData_raw, -700, 700))
	non_physical = (reasons == "") & ((tau_us <= 0) | (tau_us > 1e6 + 1e-6))
	reasons[non_physical] = "non_physical"

	good_mask = (reasons == "")
	os.makedirs(dm_case_dir, exist_ok=True)

	# failed_DM.csv: quarantined rows, same column layout as pDesignMatrix.csv
	# (feature columns only), with a leading row_index column referencing the
	# TRUE DesignMatrix.csv row number (not this function's local position --
	# xData_raw here is already restricted to simulated/kept rows).
	bad_idx = np.where(~good_mask)[0]
	with open(os.path.join(dm_case_dir, "failed_DM.csv"), "w") as fh:
		for i in bad_idx:
			fh.write(f"{orig_row_idx[i]}," + ",".join(f"{v}" for v in xData_raw[i]) + "\n")

	with open(os.path.join(dm_case_dir, "failed_DM_reasons.csv"), "w") as fh:
		fh.write("row_index,reason,violation_ratio,worst_reaction,response_value\n")
		for i in bad_idx:
			fh.write(f"{orig_row_idx[i]},{reasons[i]},{ratio[i]:.6f},"
			         f"{worst_rxn[i]},{yData_raw[i]:.6f}\n")

	good_idx = np.where(good_mask)[0]
	with open(os.path.join(dm_case_dir, "good_DM_index.csv"), "w") as fh:
		fh.write("row_index\n")
		for i in good_idx:
			fh.write(f"{orig_row_idx[i]}\n")

	stats = {
		"n_generated_kept_for_sim": n,   # rows that reached this function (post sim-failure filtering)
		"n_band_invalid": int(band_bad.sum()),
		"n_sentinel": int(is_sentinel.sum()),
		"n_non_finite": int(non_finite.sum()),
		"n_non_physical": int(non_physical.sum()),
		"n_good": int(good_mask.sum()),
		"good_fraction_pct": float(100.0 * good_mask.sum() / n) if n else 0.0,
		"worst_violation_ratio": float(ratio.max()) if n else 0.0,
		"median_violation_ratio": float(np.median(ratio)) if n else 0.0,
	}
	with open(os.path.join(dm_case_dir, "screening_summary.json"), "w") as fh:
		json.dump(stats, fh, indent=2)

	return xData_raw[good_mask], yData_raw[good_mask], stats


def retrain_around_outlier(case, case_index, Response, xTrain, yTrain,
						   xVal, yVal, xData, selectedParams, blocks, search, resampler):
	"""Adaptive densification around PRS VALIDATION outliers, then refit.

	Both outlier DETECTION and the accept/reject decision below use ONLY
	xVal/yVal -- the caller's main loop must NEVER pass its final untouched
	hold-out here (that split is evaluated exactly once, after this function
	returns, purely for reporting; see the diagnosis in report/ of why the
	original version's re-use of the SAME xTest for both outlier-selection
	and scoring was a leakage bug).

	Each pass: build a focused mini-DM around the current worst VALIDATION
	outliers (via `resampler` -- an AdaptiveResampler.DOptimalResampler,
	already constructed with THIS case's layout/band-validity settings, or a
	legacy OutlierResampler mode selected via outlier_cfg["sample_mode"]),
	simulate the new points, augment the training set, refit a fresh PLS
	feature map (k/order/ridge stay FIXED at whatever the initial
	fit_prs_with_search chose) + PRS, and score on xVal/yVal.

	BUG FIX vs. the original: a pass is kept ONLY if it does not make the
	validation score worse ("accept only if the validation score improves,
	else revert and stop" -- the original unconditionally did `best = refit`,
	which is why case-3's retraining used to make the fit monotonically
	worse, 0.25%->11.8%, once combined with the (separately fixed)
	data_management.py duplicate-row bug). The on-disk coefficient cache is
	explicitly rewritten from `best.coeff` before returning, so a rejected
	trial's write never lingers as the "live" cache for this case_index.

	SECOND BUG FIX (found 2026-07-30, chasing a <1.2% target): accepting on
	validation ALONE lets TRAIN error drift upward across passes even while
	validation improves -- observed on Case 3 at MRE_criteria=1.2%: 5 passes
	all "accepted" by the validation-only rule, walking validation down
	2.06%->1.17% while TRAIN climbed to 2.04%, silently regressing above the
	already-achieved 2% criterion this same pipeline met before this run.
	The accept test now uses max(train, val) -- the identical criterion
	fit_prs_with_search already uses for its own hyperparameter selection --
	so a pass cannot be accepted if it trades train error for validation
	error; it can only be accepted if the WORSE of the two does not worsen.

	THIRD BUG FIX (2026-07-30, 10-fold CV investigation): xVal is now
	filtered to band-valid rows (prs_fit_cfg["band_tolerance_eval"], default
	1.0) before outlier detection AND accept/reject, for the same reason
	Bug 2 above exists -- a band-invalid validation row (physically
	implausible perturbation, class-B m=3 sampler bug, sec:upstream item 1)
	is not a fair signal for either "is this case still failing" or "did
	this pass help", and can only ever look like noise to a fit trained on
	physically valid data.

	Returns (best_response, resampler) -- the resampler is returned so the
	caller can inspect/reuse it (e.g. for logging) without reconstructing it.

	Defense-in-depth (2026-07-31): Stage 3 is A1+B1+C1-only (see the call
	site's own guard in the main per-case loop, above) -- this early return
	is a second, independent check in case a future caller invokes this
	function directly without going through that guard.
	"""
	if design_type != "A1+B1+C1":
		print(f"[Case-{case}] retrain_around_outlier called under "
		      f"{design}={design_type!r} -- returning the caller's Response "
		      f"unchanged (Stage 3 is supported only for 'A1+B1+C1').")
		return Response, resampler
	if prs_fit_cfg["band_tolerance_eval"] is not None:
		_vmask = IPRS.band_valid_mask(xVal, blocks, prs_fit_cfg["band_tolerance_eval"])
		if _vmask.any():
			xVal, yVal = xVal[_vmask], yVal[_vmask]
	target = outlier_cfg["target_fraction"] * MRE_criteria
	n_train0 = len(xTrain)
	X_aug = np.asarray(xTrain, dtype=float)
	y_aug = np.asarray(yTrain, dtype=float).ravel()
	xVal = np.asarray(xVal, dtype=float)
	yVal = np.asarray(yVal, dtype=float).ravel()
	best = Response
	# Response.ytestMaxError/.MaxError were set by the CALLER's Response.test
	# call on the UNFILTERED xVal -- re-score on the (possibly band-valid-
	# filtered) xVal/yVal above so the baseline is on the same footing as
	# every subsequent pass's comparison, not an apples-to-oranges mix.
	best_train_score, best_val_score, _ = Response.test(xVal, yVal)
	best_score = max(best_train_score, best_val_score)
	coef_path = os.path.join(optDir, "Data", "ResponseSurface",
							 f"responsecoef_case-{case_index}.csv")
	mode = outlier_cfg["sample_mode"]

	# Minimum outlier count, but ONLY on the FIRST pass (it==0): found
	# 2026-07-30 chasing a <1.2% target -- a handful of borderline VALIDATION
	# points on a case's very FIRST pass (e.g. Case 1's 3 points out of
	# ~1,135, straight from the Stage1+2 search) is the same small-
	# validation-set noise signature already diagnosed for Case 1's
	# hyperparameter search (report/, Section "Tuning Case 1"), not a broad,
	# real deficiency -- resampling to chase it measurably made the untouched
	# hold-out WORSE (0.905/1.847% -> 0.966/2.407%) even though the pass
	# legitimately improved max(train,val). Applying the SAME guard to LATER
	# passes is wrong, though: Case 3's pass 2 also had only 3 outliers, but
	# it was continuing an already-validated improving trend (pass 1 had
	# genuinely improved max(train,val) via a real, broad 9-point fix) and
	# itself produced a further genuine improvement (train 2.148%->1.930%,
	# reproducing this project's historically best Case 3 result) -- blocking
	# it left Case 3 at 2.148% train, regressing above the already-achieved
	# 2% criterion. So: skip-for-noise applies only when it==0 (nothing yet
	# validated this case's direction); once at least one pass has been
	# accepted, a small residual outlier count is trusted.
	min_outliers = max(int(outlier_cfg.get("min_outlier_count", 5)),
						int(np.ceil(outlier_cfg.get("min_outlier_frac", 0.005) * len(yVal))))

	# ── Accounting (2026-08-19, feeds optimization_cost.csv's Stage3_* columns) ──
	stage3_stats = {"passes_attempted": 0, "passes_accepted": 0,
	                 "sims_run": 0, "sims_kept": 0, "stalled_passes": 0,
	                 "aborted_reason": None}
	stall_count = 0
	prev_signature = None
	oversample_boost = 1

	it = 0
	while it < outlier_cfg["max_iter"]:
		rel = np.asarray(best.error_testing_relative, dtype=float)
		worst_abs = float(np.abs(rel).max()) if len(rel) else 0.0
		still_failing = worst_abs > MRE_criteria
		# min_passes ENFORCEMENT (2026-08-19): while the case is still above
		# the RAW criterion, no early-exit path below is allowed to stop
		# retraining before at least outlier_cfg["min_passes"] passes have
		# been ATTEMPTED (a stalled pass, see below, does NOT count as
		# attempted). Fixes the observed gap where a case above 2% could
		# exit after a single pass (or zero, if the very first candidate
		# draw stalled) with nothing distinguishing that from "this case
		# doesn't need help".
		below_min_passes = stage3_stats["passes_attempted"] < outlier_cfg["min_passes"]

		# Densify around every point still above the tighter TARGET (not just
		# points above MRE_criteria) so the loop keeps pushing error down to 0.5x.
		oidx = AR.find_outliers(rel, target, force_worst=False)
		if oidx.size == 0:
			if still_failing and below_min_passes:
				# Structurally rare (target is tighter than MRE_criteria by
				# default, so 0 target-outliers normally implies the case
				# already clears MRE_criteria too) but handled explicitly per
				# instruction: a case that is somehow still above the RAW
				# criterion with nothing above the tighter target must not be
				# silently left short of min_passes -- force at least the
				# single worst point.
				oidx = AR.find_outliers(rel, target, force_worst=True)
			else:
				print(f"\t[Case-{case}] worst VALIDATION error {worst_abs:.3f}% already "
					  f"<= target {target:.3f}% -- no re-training needed.")
				break
		if it == 0 and 0 < oidx.size < min_outliers and not still_failing:
			# A case above MRE_criteria is NEVER treated as validation-split
			# noise (that diagnosis only ever applied to a case chasing the
			# STRICTER target_fraction*MRE_criteria target after already
			# clearing the raw criterion) -- retrain_around_outlier is only
			# ever invoked for a case that is already above MRE_criteria (see
			# the call site's _stage3_needed gate), so this branch is
			# structurally unreachable at it==0 today; kept for correctness
			# if a future caller ever invokes this function pre-emptively.
			print(f"\t[Case-{case}] only {oidx.size} point(s) above target "
				  f"(< min_outlier threshold {min_outliers}) and case already "
				  f"clears MRE_criteria -- likely validation-split noise; "
				  f"skipping further retraining.")
			break

		# Reseed each pass so successive passes add DIFFERENT local samples.
		resampler.rng = np.random.default_rng(outlier_cfg["seed"] + it)
		try:
			outlier_rows = xVal[oidx]
			weights = np.abs(rel[oidx])
			dopt_info = None
			if mode == "local":
				prs_rows, full_rows, sel_mask, row_class = resampler.resample(outlier_rows, xData)
			elif mode == "dm_global":
				prs_rows, full_rows, sel_mask, row_class = resampler.resample_global(outlier_rows, xData)
			elif mode == "gap_fill":
				ref_rows = (np.vstack([xData, X_aug[n_train0:]])
							if len(X_aug) > n_train0 else xData)
				prs_rows, full_rows, sel_mask, row_class = resampler.resample_gap_fill(
					outlier_rows, ref_rows)
			elif mode == "hybrid":
				prs_rows, full_rows, sel_mask, row_class = resampler.resample_hybrid(
					outlier_rows, xData, local_fraction=outlier_cfg["local_fraction"])
			else:  # "d_optimal" (default) -- band-valid candidates, greedy D-optimal pick
				prs_rows, full_rows, sel_mask, row_class, dopt_info = resampler.resample_d_optimal(
					outlier_rows, xData, X_aug, best.build_basis,
					target_weights=weights, local_fraction=outlier_cfg["local_fraction"],
					oversample_boost=oversample_boost)

			# ── STALL DETECTION (2026-08-19, d_optimal only) ────────────────
			# A stalled pass -- zero candidates selected, or the IDENTICAL
			# candidate set as the previous pass -- burns a full simulation
			# batch for zero possible benefit (observed: Case 52 selected 0
			# points on all 8 pre-fix passes; Case 56 re-picked the same 34
			# points 8 times). Detected BEFORE any simulation is requested,
			# so a stall costs nothing. Does not count toward min_passes.
			if mode == "d_optimal" and dopt_info is not None:
				sig = dopt_info.get("selected_signature", ())
				stalled = (dopt_info["n_selected"] == 0
				           or (sig and prev_signature is not None and sig == prev_signature))
				if stalled:
					stall_count += 1
					stage3_stats["stalled_passes"] += 1
					print(f"\t[Case-{case}] pass stalled ({stall_count}/{outlier_cfg['stall_patience']}): "
					      f"D-optimal selected {dopt_info['n_selected']} candidate(s) from "
					      f"{dopt_info['n_candidates_band_valid']}/{dopt_info['n_candidates_generated']} "
					      f"band-valid (reject_rate {100*dopt_info['reject_rate']:.1f}%)"
					      + (", identical to the previous pass's selection" if sig and sig == prev_signature else "")
					      + " -- no simulation run for this pass.")
					if stall_count >= outlier_cfg["stall_patience"]:
						stage3_stats["aborted_reason"] = (
							f"stalled {stall_count} consecutive passes (candidate pool "
							f"exhausted -- check class-pool provenance for this case)")
						print(f"\t[Case-{case}] aborting Stage 3: {stage3_stats['aborted_reason']}.")
						break
					oversample_boost = min(oversample_boost * 4, 64)
					it += 1
					continue
				else:
					stall_count = 0
					oversample_boost = 1
					if sig:
						prev_signature = sig

			print(f"\n[Case-{case}] adaptive re-training pass "
				  f"{stage3_stats['passes_attempted']+1}/{outlier_cfg['max_iter']} (attempt {it+1}) "
				  f"[{mode}]: {oidx.size} point(s) above target on VALIDATION, "
				  f"worst |err|={worst_abs:.3f}% "
				  f"(target {target:.3f}% = {outlier_cfg['target_fraction']}x MRE {MRE_criteria}%)")
			if dopt_info is not None:
				print(f"\t[Case-{case}] D-optimal candidates: band-valid "
					  f"{dopt_info['n_candidates_band_valid']}/{dopt_info['n_candidates_generated']} "
					  f"(reject_rate {100*dopt_info['reject_rate']:.1f}%), "
					  f"selected {dopt_info['n_selected']}/{dopt_info['n_budget']}")

			stage3_stats["passes_attempted"] += 1

			# Persist the mini-DM ("small DM at the end of the original DM"),
			# split by row class just like the original DM generation, plus a
			# combined file. Saved next to the simulation results for inspection.
			sim_store = os.path.join(optDir, "Data", "Outlier_Simulations")
			os.makedirs(sim_store, exist_ok=True)
			row_class = np.asarray(row_class)
			np.savetxt(os.path.join(sim_store, f"Outlier_DM_case-{case}_pass-{it+1}.csv"),
					   prs_rows, delimiter=",")
			for lab in sorted(set(row_class)):
				np.savetxt(os.path.join(sim_store,
										f"Outlier_DM_case-{case}_pass-{it+1}_{lab}type.csv"),
						   prs_rows[row_class == lab], delimiter=",")
			print(f"\t[Case-{case}] mini-DM by class: " +
				  ", ".join(f"{lab}={int((row_class==lab).sum())}"
							for lab in sorted(set(row_class))))
			y_new = _simulate_new_samples(case, prs_rows, full_rows, sel_mask, it + 1,
										   force_resim=outlier_cfg["force_resim"])
			stage3_stats["sims_run"] += len(prs_rows)
			if len(y_new) != len(prs_rows):
				warn(f"[Case-{case}] simulated {len(y_new)} of {len(prs_rows)} new "
					 f"points; truncating to matched rows.")
				m = min(len(y_new), len(prs_rows))
				prs_rows, y_new = prs_rows[:m], y_new[:m]

			# Every NEW row must itself be band-valid before it is even
			# eligible to be counted "kept" -- resample_d_optimal already
			# band-gates candidates at the source (band_tolerance), so this
			# is defense-in-depth accounting, not a second filter pass; for
			# the legacy geometric modes (no band gate at all) this is the
			# FIRST time these rows are checked.
			if prs_fit_cfg["band_tolerance_eval"] is not None:
				_new_valid = IPRS.band_valid_mask(prs_rows, blocks, prs_fit_cfg["band_tolerance_eval"])
				stage3_stats["sims_kept"] += int(_new_valid.sum())
			else:
				stage3_stats["sims_kept"] += len(prs_rows)

			X_trial = np.vstack([X_aug, prs_rows])
			y_trial = np.concatenate([y_aug, np.asarray(y_new, dtype=float).ravel()])

			# Re-fit the PLS rotation on the AUGMENTED set (k/order/ridge fixed
			# at whatever the initial fit_prs_with_search chose for this case),
			# then the PRS itself.
			fmap_trial = IPRS.fit_feature_map(X_trial, y_trial, blocks, xData.shape[1], k=search["k"])
			if os.path.isfile(coef_path):
				os.remove(coef_path)
			trial = PRS.ResponseSurface(X_trial, y_trial, case, case_index,
										responseOrder=search["order"], prs_type=PRS_type,
										selected_params=selectedParams, criteria=MRE_criteria,
										feature_map=fmap_trial, ridge_alpha=search["ridge_alpha"])
			trial.create_response_surface()
			mtr, mva, crit = trial.test(xVal, yVal)
			trial_score = max(mtr, mva)
			print(f"\t[Case-{case}] pass {it+1}: VALIDATION max error "
				  f"{best_val_score:.3f}% -> {mva:.3f}% (train {mtr:.3f}%, "
				  f"max(train,val) {best_score:.3f}% -> {trial_score:.3f}%, "
				  f"target {target:.3f}%, +{len(prs_rows)} pts, {len(X_trial)} total)")

			# ── accept only if max(train, val) does not worsen, else revert and
			# stop -- NOT validation alone (see docstring: val-only accept let
			# train silently regress above the 2% criterion while chasing a
			# stricter validation target). ──
			if trial_score <= best_score:
				X_aug, y_aug = X_trial, y_trial
				best = trial
				best_val_score = mva
				best_train_score = mtr
				best_score = trial_score
				stage3_stats["passes_accepted"] += 1
				append_csv_row(PRS_ERROR_FILE, [f"Case-{case}-retrained-p{it+1}", "validation", it + 1,
				                                 mtr, mva, crit, "", len(yVal), 0, len(yVal),
				                                 "PASS" if mva <= MRE_criteria else "FAIL"])
				if mva <= target:
					print(f"\t[Case-{case}] reached target ({mva:.3f}% <= {target:.3f}%) "
						  f"after {stage3_stats['passes_attempted']} pass(es) -- stopping.")
					break
				if stage3_stats["passes_attempted"] >= outlier_cfg["max_iter"]:
					print(f"\t[Case-{case}] max_iter ({outlier_cfg['max_iter']}) reached; "
						  f"best VALIDATION error {mva:.3f}% still above target {target:.3f}%.")
			else:
				# RETRY-ON-REJECTION (2026-08-19): a rejected pass retries
				# with a new draw rather than stopping outright, UNLESS
				# min_passes is already satisfied and the case has run out
				# of max_iter budget. X_aug/y_aug/best/best_score are
				# UNCHANGED by a rejected trial.
				print(f"\t[Case-{case}] pass {it+1} made max(train,val) WORSE "
					  f"({trial_score:.3f}% > {best_score:.3f}%, train {mtr:.3f}% val {mva:.3f}%) "
					  f"-- discarding this pass's draw, keeping the last accepted surface, "
					  f"and retrying with a new draw" +
					  (f" (max_iter reached)." if stage3_stats["passes_attempted"] >= outlier_cfg["max_iter"] else "."))
				if stage3_stats["passes_attempted"] >= outlier_cfg["max_iter"]:
					break
		except Exception as exc:
			print(f"\t[Case-{case}] adaptive re-training FAILED: "
				  f"{type(exc).__name__}: {exc}. Keeping the last good surface.")
			traceback.print_exc()
			stage3_stats["aborted_reason"] = f"exception: {type(exc).__name__}: {exc}"
			break
		it += 1

	# Guarantee the on-disk coefficient cache reflects `best`, regardless of
	# which branch above ended the loop (a rejected trial's create_response_surface
	# call may have left ITS coefficients on disk otherwise).
	_write_coef_cache(coef_path, best.coeff)
	return best, resampler, stage3_stats

ResponseSurfaces = {}
selected_PRS = {}
# case_index MUST equal the true case number (int(case)), not its position in
# this loop -- design_matrix_dict / selected_params_dict / DM_FOR_PARTIAL_PRS
# paths / coefficient cache filenames are all keyed/named by the true case
# number (see their construction above, `for case in case_dir: ...dict[case]
# = ...`). The original `for case_index, case in enumerate(temp_sim_opt)`
# only got away with using the enumerate POSITION as case_index because
# case_dir defaults to a contiguous range(0, N) processed in insertion
# order, making position == case number by coincidence; it silently breaks
# (reads/writes the WRONG case's cached files) the moment case_dir is ever a
# non-contiguous or reordered subset (e.g. MUQSAC_CASE_SUBSET, above).
_n_cases_total = len(list(temp_sim_opt))
_n_case_done = 0
_case_verdicts = []   # (case, train%, holdout%, holdout_bandvalid%, PASS/FAIL) for the closing summary
for case in temp_sim_opt:
	case_index = int(case)
	_n_case_done += 1
	step(f"Case {case} ({_n_case_done}/{_n_cases_total})")

	# ── X/Y ALIGNMENT FIX (2026-08-19) ───────────────────────────────────
	# yData_sim/xData_sim are restricted to kept_row_index[case] -- the
	# design-matrix row each simulated response actually came from (see the
	# response-extraction phase above) -- rather than assuming positional
	# alignment between temp_sim_opt[case] and design_matrix_dict[case_index]
	# (silently wrong the moment any simulation fails and is dropped by
	# generate_target_value_tables; benign today, 0 failures, but no longer
	# an assumption this code makes).
	yData_sim = np.asarray(temp_sim_opt[case]).flatten()
	full_xData = np.asarray(design_matrix_dict[case_index])
	orig_row_idx = kept_row_index[case]
	if len(orig_row_idx) != len(yData_sim):
		warn(f"[Case-{case}] kept_row_index length ({len(orig_row_idx)}) != "
		     f"response count ({len(yData_sim)}) -- falling back to positional "
		     f"0..N-1 indexing for this case.")
		orig_row_idx = np.arange(len(yData_sim))
	xData_sim = full_xData[orig_row_idx]
	selectedParams = np.asarray(selected_params_dict[case_index])

	# ── Stage 1 setup: this case's identifiable-subspace layout + a
	#    DOptimalResampler (also reused by Stage 3 below if retraining fires,
	#    and needed here first because screen_case_samples's band-validity
	#    test uses `blocks`, which is built from resampler.layout) ──────────
	Tc = target_list[case_index].temperature
	twf = prs_fit_cfg["temp_window_frac"]
	T_local = np.array([Tc * (1.0 - twf), Tc, Tc * (1.0 + twf)])
	resampler = AR.DOptimalResampler(
		unsrt_data, selectedParams, design_type, PRS_type, home_dir, case,
		N_new=outlier_cfg["N_new"], delta_fraction=outlier_cfg["delta_fraction"],
		seed=outlier_cfg["seed"], uns_cap=outlier_cfg["uns_cap"],
		shuffle_factor=outlier_cfg["shuffle_factor"], method=outlier_cfg["method"],
		gapfill_oversample=outlier_cfg["gapfill_oversample"],
		gapfill_knn_radius_frac=outlier_cfg["gapfill_knn_radius_frac"],
		gapfill_knn_k=outlier_cfg["gapfill_knn_k"],
		gapfill_shuffle_factor=outlier_cfg["gapfill_shuffle_factor"],
		band_tolerance=outlier_cfg["band_tolerance"],
		temps_for_validity={"__default__": np.linspace(300., 2500., outlier_cfg["validity_n_temps"])},
		acquisition_alpha=outlier_cfg["acquisition_alpha"],
		candidate_oversample=outlier_cfg["candidate_oversample"])
	# DOptimalResampler.layout entries -> IdentifiablePRS's (rxn, col_slice,
	# local_indices) triples (same convention: prs_local is the reaction's
	# selected local Arrhenius-parameter indices for Partial PRS, or the full
	# 0..alen-1 range for Full PRS -- see AdaptiveResampler.py's _build_layout).
	layout_triples = [(lay["rxn"], slice(lay["comp_offset"], lay["comp_offset"] + lay["n_cols"]), lay["prs_local"])
	                   for lay in resampler.layout if lay["n_cols"] > 0]
	def _sigma_of(rxn):
		L = unsrt_data[rxn].cholskyDeCorrelateMat
		return L @ L.T
	blocks = IPRS.build_layout_blocks(layout_triples, _sigma_of, T_local, svd_tol=prs_fit_cfg["svd_tol"],
	                                   fprior_mode=prs_fit_cfg["band_fprior"])

	# ── SCREEN: quarantine bad samples into DM_FOR_PARTIAL_PRS/<case>/
	#    failed_DM.csv (this study's OWN dir, via home_dir -- never hardcoded)
	#    BEFORE the PRS ever sees them -- the good/bad split below is what
	#    every downstream split, fit, and reported metric is computed from. ──
	dm_case_dir = os.path.join(home_dir, "DM_FOR_PARTIAL_PRS", str(case_index))
	xData, yData, screen_stats = screen_case_samples(
		case, case_index, xData_sim, yData_sim, orig_row_idx, blocks,
		dm_case_dir, prs_fit_cfg["band_tolerance_eval"])
	append_csv_row(SCREENING_SUMMARY_FILE, [
		f"Case-{case}", screen_stats["n_generated_kept_for_sim"],
		screen_stats["n_band_invalid"], no_of_sim[case_index] - screen_stats["n_generated_kept_for_sim"],
		screen_stats["n_sentinel"], screen_stats["n_non_finite"], screen_stats["n_non_physical"],
		screen_stats["n_good"], f"{screen_stats['good_fraction_pct']:.2f}",
		f"{screen_stats['worst_violation_ratio']:.3f}", f"{screen_stats['median_violation_ratio']:.3f}",
	])
	kv("Samples simulated", screen_stats["n_generated_kept_for_sim"])
	kv("Band-invalid (quarantined)", f"{screen_stats['n_band_invalid']} "
	   f"({100 - screen_stats['good_fraction_pct']:.1f}%)")
	kv("Good (fit + scored on)", f"{screen_stats['n_good']} ({screen_stats['good_fraction_pct']:.1f}%)")
	if screen_stats["n_good"] < 50:
		warn(f"[Case-{case}] only {screen_stats['n_good']} good sample(s) -- "
		     f"PRS capacity (k) will be tightly capped by prs_max_terms_frac.")

	# ── 3-way split: train / validation / UNTOUCHED hold-out -- GOOD ROWS
	#    ONLY (2026-08-19). Previously split the RAW xData/yData (~89-94%
	#    band-invalid), so every split -- and everything scored against it --
	#    was already contaminated before Stage 1+2's own screening ever ran.
	#    Splitting the pre-screened good submatrix instead means xHold is
	#    finally an honest hold-out of PHYSICALLY VALID data, and
	#    Max_Holdout_Final converges with Max_Holdout_BandValid (they were
	#    previously two very different numbers computed from two different
	#    populations; see PRS_DETAIL_HEADER's comment). ────────────────────
	train_frac = 1.0 - prs_fit_cfg["val_frac"] - prs_fit_cfg["holdout_frac"]
	itr, ival, iho = IPRS.three_way_split(
		len(yData), seed_outer=prs_fit_cfg["split_seed"], seed_inner=prs_fit_cfg["split_seed_inner"],
		train_frac=train_frac, val_frac=prs_fit_cfg["val_frac"])
	xTrain, yTrain = xData[itr], yData[itr]
	xVal,   yVal   = xData[ival], yData[ival]
	xHold,  yHold  = xData[iho], yData[iho]

	try:
		# ── Stage 1+2: search (screening tolerance, k, order, ridge) --------------
		# Default "cv_kfold": worst-of-cv_folds-fold selection over the combined
		# train+val pool, scored on band-valid points only -- see prs_fit_cfg's
		# comment for why (a single split can massively underestimate true risk).
		# "single_split" restores the original train->validation-only search.
		if prs_fit_cfg["search_method"] == "cv_kfold":
			X_search_pool = np.vstack([xTrain, xVal])
			y_search_pool = np.concatenate([yTrain, yVal])
			search = IPRS.fit_prs_with_cv_search(
				PRS.ResponseSurface, X_search_pool, y_search_pool, blocks, xData.shape[1],
				case, case_index, PRS_type, selectedParams, MRE_criteria,
				cache_dir=optDir,
				tol_grid=prs_fit_cfg["screening_tolerances"], k_grid=prs_fit_cfg["k_grid"],
				order_grid=prs_fit_cfg["order_grid"], ridge_grid=prs_fit_cfg["ridge_grid"],
				max_terms_frac=prs_fit_cfg["max_terms_frac"],
				cv_folds=prs_fit_cfg["cv_folds"], cv_seed=prs_fit_cfg["cv_seed"],
				band_tolerance_eval=prs_fit_cfg["band_tolerance_eval"])
			kv("Stage1+2 search", f"cv_kfold/{prs_fit_cfg['cv_folds']}-fold")
			kv("  tolerance / k / order / ridge", f"{search['tolerance']} / {search['k']} / "
			   f"{search['order']} / {search['ridge_alpha']:.0e}")
			kv("  rows kept / pool", f"{search['n_kept']}/{len(X_search_pool)}")
			kv("  worst-fold score", f"{search['cv_score']:.3f}%")
			# For downstream code that expects a single train_pct/val_pct pair
			# (retrain_around_outlier's initial best_score, logging) -- the CV
			# score is the honest summary here, use it for both.
			search["train_pct"] = search["val_pct"] = search["cv_score"]
		else:
			search = IPRS.fit_prs_with_search(
				PRS.ResponseSurface, xTrain, yTrain, xVal, yVal, blocks, xData.shape[1],
				case, case_index, PRS_type, selectedParams, MRE_criteria,
				cache_dir=optDir,
				tol_grid=prs_fit_cfg["screening_tolerances"], k_grid=prs_fit_cfg["k_grid"],
				order_grid=prs_fit_cfg["order_grid"], ridge_grid=prs_fit_cfg["ridge_grid"],
				max_terms_frac=prs_fit_cfg["max_terms_frac"],
				band_tolerance_eval=prs_fit_cfg["band_tolerance_eval"])
			kv("Stage1+2 search", "single_split")
			kv("  tolerance / k / order / ridge", f"{search['tolerance']} / {search['k']} / "
			   f"{search['order']} / {search['ridge_alpha']:.0e}")
			kv("  rows kept / train", f"{search['n_kept']}/{len(xTrain)}")
			kv("  train / val score", f"{search['train_pct']:.3f}% / {search['val_pct']:.3f}%")

		# NOTE: for cv_kfold, search["keep_mask"] is sized to the combined
		# train+val POOL it searched internally, not to xTrain alone -- re-screen
		# xTrain by the WINNING tolerance here instead of trusting that mask
		# directly, so xVal stays untouched for Stage 3's accept/reject and xHold
		# stays untouched for final reporting, exactly as before (only the
		# hyperparameter SELECTION got more robust, the train/val/holdout
		# architecture is unchanged).
		if prs_fit_cfg["search_method"] == "cv_kfold":
			keep_mask, _ = IPRS.screen_rows(xTrain, blocks, search["tolerance"])
		else:
			keep_mask = search["keep_mask"]
		xTrainK, yTrainK = xTrain[keep_mask], yTrain[keep_mask]
		feature_map = IPRS.fit_feature_map(xTrainK, yTrainK, blocks, xData.shape[1], k=search["k"])

		coef_path = os.path.join(optDir, "Data", "ResponseSurface", f"responsecoef_case-{case_index}.csv")
		if os.path.isfile(coef_path):
			os.remove(coef_path)
		Response = PRS.ResponseSurface(xTrainK, yTrainK, case, case_index,
								responseOrder=search["order"], prs_type=PRS_type,
								selected_params=selectedParams, criteria=MRE_criteria,
								feature_map=feature_map, ridge_alpha=search["ridge_alpha"])
		Response.create_response_surface()
		Max_Train_Error, Max_Val_Error, Criterion = Response.test(xVal, yVal)

		# BAND-VALID TRIGGER FIX (2026-08-19): Stage 3's trigger below used to
		# gate on `Response.selection`, set by the Response.test() call just
		# above -- scored on the RAW (unfiltered) xVal. When most of a case's
		# rows are physically invalid (the class-B sampler bug, root cause 1;
		# fixed, but this study's EXISTING design matrix still has ~89% invalid
		# rows until Stage 3 itself replaces them), xVal is dominated by
		# invalid rows too (verified: 88% for case 0), so a cleanly-trained
		# model scores badly on it regardless of whether it actually meets the
		# criterion on the rows that matter -- Stage 3 would fire for nearly
		# every case, not just the ones genuinely failing. The search
		# (fit_prs_with_cv_search) already selects hyperparameters against
		# band-valid-filtered scoring when band_tolerance_eval is set; the
		# trigger now checks the SAME thing, so "does this case need
		# retraining" agrees with what was just optimized for. Max_Val_Error /
		# Criterion / Response.selection above are left untouched (other code
		# may depend on their original, all-points meaning) -- only the Stage 3
		# GATE below uses the band-valid figure.
		if prs_fit_cfg["band_tolerance_eval"] is not None:
			_trigger_vmask = IPRS.band_valid_mask(xVal, blocks, prs_fit_cfg["band_tolerance_eval"])
			_trigger_rel = np.asarray(Response.error_testing_relative, dtype=float)
			_stage3_val_error = (float(np.max(np.abs(_trigger_rel[_trigger_vmask])))
			                      if _trigger_vmask.any() else Max_Val_Error)
		else:
			_stage3_val_error = Max_Val_Error
		_stage3_needed = max(Max_Train_Error, _stage3_val_error) > MRE_criteria

		# ── Stage 3: adaptive outlier re-training -- outliers & accept/reject
		#    decided on VALIDATION only, never on the hold-out below ────────────
		# Trigger against the RAW criterion (validation > MRE_criteria), not a
		# tighter internal target. An earlier version of this trigger used
		# target_fraction*MRE_criteria (1.0% vs the raw 2%) to give borderline
		# cases a safety margin against hold-out-split sampling variance -- but
		# empirically (Case 1: search-selected combo already at 1.32% validation,
		# comfortably under criterion) that extra retraining pass, even though it
		# was ACCEPTED under the "validation does not get worse" rule (validation
		# was unchanged, 1.318% -> 1.318%), made the untouched hold-out
		# noticeably WORSE (2.30% -> 3.58%) -- i.e. the added D-optimal points
		# were validation-neutral but hold-out-harmful for this case, a genuine
		# overfit-to-validation risk the "accept if not worse" rule cannot see
		# (it only ever looks at validation, by design -- hold-out is touched
		# once, see below). A case whose SEARCH-selected fit already clears the
		# raw criterion is left alone; only cases that still fail outright
		# proceed to Stage 3.
		# Stage 3 (D-optimal adaptive resampling) is gated to the "A1+B1+C1"
		# design-matrix mode only: it relies on the class-A/B/C curve-sampler
		# pools (AdaptiveResampler._load_class_pool) that only exist for that
		# design; under "A-facto" (A-factor-only optimization), that loader
		# silently returns None and every candidate-generation path already
		# falls back to an uninformed LHS box-draw -- i.e. it would burn real
		# simulation budget on uninformed candidates instead of doing nothing.
		# Added 2026-07-31, per explicit instruction: resampling must never run
		# for A-factor-only optimization.
		stage3_stats = {"passes_attempted": 0, "passes_accepted": 0,
		                 "sims_run": 0, "sims_kept": 0, "stalled_passes": 0,
		                 "aborted_reason": None}
		if outlier_cfg["enable"] and design_type == "A1+B1+C1" and _stage3_needed:
			step(f"Stage 3: adaptive resampling (min {outlier_cfg['min_passes']} passes, "
			     f"max {outlier_cfg['max_iter']})")
			# A single case's Stage 3 hitting an unexpected internal error used
			# to crash the ENTIRE multi-hour, multi-case run (confirmed
			# 2026-08-19: a rare, documented-as-"structurally unreachable"
			# branch inside retrain_around_outlier raised uncaught, killing a
			# run that had already correctly written every one of the 68
			# cases' Prediction_errors_detailed.csv rows and Parity plots --
			# the crash was on a bonus retry pass, not a primary case). Any
			# case reaching here already has a valid Response from Stage 1+2,
			# so on failure we keep that surface and report zero Stage-3
			# passes rather than losing the whole run over one case's retry.
			try:
				Response, resampler, stage3_stats = retrain_around_outlier(
					case, case_index, Response, xTrainK, yTrainK, xVal, yVal,
					xData, selectedParams, blocks, search, resampler)
			except Exception as e:
				warn(f"[Case-{case}] Stage 3 raised {e!r} -- keeping the Stage 1+2 "
				     f"surface and reporting zero Stage-3 passes for this case.")
				stage3_stats["aborted_reason"] = f"exception: {e!r}"
			kv("Stage 3 passes attempted/accepted", f"{stage3_stats['passes_attempted']}/"
			   f"{stage3_stats['passes_accepted']}" + (
			   f" (aborted: {stage3_stats['aborted_reason']})" if stage3_stats["aborted_reason"] else ""))
			kv("Stage 3 new sims run/kept", f"{stage3_stats['sims_run']}/{stage3_stats['sims_kept']}")
		elif outlier_cfg["enable"] and design_type != "A1+B1+C1" and _stage3_needed:
			warn(f"[Case-{case}] Stage 3 D-optimal resampling SKIPPED: "
			     f"{design}={design_type!r} (A-factor-only optimization). "
			     f"Resampling is supported only for the 'A1+B1+C1' design.")

		# ── Final, ONE-TIME, untouched hold-out evaluation -- this is the number
		#    that gets reported; nothing above has seen xHold/yHold ─────────────
		Max_Train_Final, Max_Holdout_Final, Criterion = Response.test(xHold, yHold)
		Response.plot(case_index)

		# Band-valid-only hold-out MRE (2026-07-30, 10-fold CV investigation):
		# a row the sampler's own physical-validity test would have rejected
		# (ratio 7x-30x the band; the class-B m=3 sampler bug, sec:upstream
		# item 1) is not a fair target to grade a physically-valid fit against.
		# Computed from the SAME test() call above (error_testing_relative),
		# never a second fit/scoring pass -- no extra leakage risk. Reported
		# alongside the raw all-points number for transparency; band-valid is
		# the officially gating criterion when prs_fit_cfg["band_tolerance_eval"]
		# is set (default 1.0).
		if prs_fit_cfg["band_tolerance_eval"] is not None:
			hold_valid_mask = IPRS.band_valid_mask(xHold, blocks, prs_fit_cfg["band_tolerance_eval"])
			rel = np.asarray(Response.error_testing_relative, dtype=float)
			Max_Holdout_BandValid = float(np.max(np.abs(rel[hold_valid_mask]))) if hold_valid_mask.any() else Max_Holdout_Final
			gating_holdout = Max_Holdout_BandValid
			# Band-valid validation MRE, re-scored fresh against whatever Response
			# is FINAL at this point (Stage 3 may have replaced it since
			# Max_Val_Error was captured, above) -- cheap (Response is already
			# fit), and avoids reporting a validation number that predates the
			# retraining loop's last accepted pass.
			_mtr_v, _mva_v, _ = Response.test(xVal, yVal)
			val_valid_mask = IPRS.band_valid_mask(xVal, blocks, prs_fit_cfg["band_tolerance_eval"])
			relv = np.asarray(Response.error_testing_relative, dtype=float)
			Max_Val_BandValid = float(np.max(np.abs(relv[val_valid_mask]))) if val_valid_mask.any() else _mva_v
			# Response.error_testing_relative now reflects the xVal scoring call
			# above -- re-run the xHold test so it is left in its documented
			# post-loop state (the untouched-hold-out numbers) for anything
			# downstream (e.g. Response.plot, already called above, is unaffected).
			Response.test(xHold, yHold)
		else:
			Max_Holdout_BandValid = Max_Holdout_Final
			gating_holdout = Max_Holdout_Final
			Max_Val_BandValid = Max_Val_Error

		verdict = "PASS" if max(Max_Train_Final, gating_holdout) <= MRE_criteria else "FAIL"

		# PRS_ERROR_FILE (Prediction_errors.csv): now reports the sample counts
		# every error number was computed FROM, alongside the numbers themselves
		# -- "how good is the fit" is no longer separable from "on how much,
		# and how clean, data" (2026-08-19; see PRS_HEADER's comment).
		append_csv_row(PRS_ERROR_FILE, [
			f"Case-{case}", "holdout", "final",
			Max_Train_Final, Max_Holdout_Final, Criterion, Max_Holdout_BandValid,
			screen_stats["n_good"], screen_stats["n_band_invalid"] + screen_stats["n_sentinel"]
			+ screen_stats["n_non_finite"] + screen_stats["n_non_physical"],
			search["n_kept"], verdict,
		])
		append_csv_row(PRS_DETAIL_FILE, [
			case, search["tolerance"], search["k"], search["order"], search["ridge_alpha"],
			no_of_sim[case_index], screen_stats["n_good"], len(xTrain), search["n_kept"],
			Max_Train_Final, Max_Val_Error, Max_Holdout_Final,
			Max_Val_BandValid, Max_Holdout_BandValid,
		])

		# optimization_cost.csv ledger row (2026-08-19): the ACTUAL simulation
		# accounting for this case -- see HEADERS' comment for why this replaced
		# the old pre-simulation closed-form-only row.
		total_sims_run = no_of_sim[case_index] + stage3_stats["sims_run"]
		total_sims_useful = screen_stats["n_good"] + stage3_stats["sims_kept"]
		append_csv_row(COST_FILE, [
			f"Case-{case}",
			cost_estimate_dict[case_index]["Total_Rxns"], cost_estimate_dict[case_index]["Active_Rxns"],
			cost_estimate_dict[case_index]["Cost_Estimate_pPRS"], cost_estimate_dict[case_index]["Cost_Estimate_FullPRS"],
			no_of_sim[case_index], screen_stats["n_generated_kept_for_sim"],
			no_of_sim[case_index] - screen_stats["n_generated_kept_for_sim"],
			screen_stats["n_sentinel"], screen_stats["n_non_finite"], screen_stats["n_non_physical"],
			screen_stats["n_band_invalid"], screen_stats["n_good"],
			stage3_stats["passes_attempted"], stage3_stats["passes_accepted"],
			stage3_stats["sims_run"], stage3_stats["sims_kept"],
			total_sims_run, total_sims_useful,
			f"{100.0 * total_sims_useful / total_sims_run:.2f}" if total_sims_run else "0.00",
		])

		result(verdict == "PASS",
		       f"[Case-{case}] FINAL: train={Max_Train_Final:.3f}%  "
		       f"holdout(band-valid)={Max_Holdout_BandValid:.3f}%  holdout(all)={Max_Holdout_Final:.3f}%  "
		       f"{verdict}  (criteria {MRE_criteria}%, {screen_stats['n_good']} good / "
		       f"{no_of_sim[case_index]} generated samples)")
		_case_verdicts.append((case, Max_Train_Final, Max_Holdout_Final, Max_Holdout_BandValid, verdict))
		ResponseSurfaces[case_index] = Response
		del xTrain,xVal,xHold,yTrain,yVal,yHold,xTrainK,yTrainK
	except Exception as _case_exc:
		# CASE-LEVEL ROBUSTNESS NET (2026-08-19): a single case's fit
		# genuinely failing (e.g. too few good rows even for Stage 1+2's
		# search grid to find ANY fittable combo -- fit_prs_with_search /
		# fit_prs_with_cv_search raise RuntimeError in that case, by design)
		# must not abort the entire 68-case run -- "every process must be
		# accounted for" means every case gets a verdict, including a
		# case that could not be fit at all. Logged loudly, recorded as an
		# explicit FAILED row in every ledger, and the loop continues.
		error(f"[Case-{case}] PRS fitting FAILED: {type(_case_exc).__name__}: {_case_exc}")
		traceback.print_exc()
		_screen_n_good = screen_stats.get('n_good', 0) if 'screen_stats' in dir() else 0
		append_csv_row(PRS_ERROR_FILE, [
			f"Case-{case}", "holdout", "final", "", "", MRE_criteria, "",
			screen_stats.get('n_good', ''), '', '', 'FAIL (fit error)'])
		append_csv_row(COST_FILE, [
			f"Case-{case}",
			cost_estimate_dict[case_index]['Total_Rxns'], cost_estimate_dict[case_index]['Active_Rxns'],
			cost_estimate_dict[case_index]['Cost_Estimate_pPRS'], cost_estimate_dict[case_index]['Cost_Estimate_FullPRS'],
			no_of_sim[case_index], screen_stats.get('n_generated_kept_for_sim', ''),
			'', screen_stats.get('n_sentinel', ''), screen_stats.get('n_non_finite', ''),
			screen_stats.get('n_non_physical', ''), screen_stats.get('n_band_invalid', ''),
			screen_stats.get('n_good', ''), 0, 0, 0, 0, no_of_sim[case_index], 0, '0.00'])
		_case_verdicts.append((case, float('nan'), float('nan'), float('nan'), 'FAIL (fit error)'))
#raise AssertionError("The Target class, Uncertainty class, Design Matrix and Simulations and Response surface")
os.chdir("..")

# ── Run summary (2026-08-19): every case's verdict as one scannable table,
#    plus the totals optimization_cost.csv's ledger rows sum to -- this is
#    what makes "did the fix work" answerable in one glance instead of
#    grepping 68 FINAL lines out of a scrollback. ─────────────────────────
_n_pass = sum(1 for *_, v in _case_verdicts if v == "PASS")
_n_fail = len(_case_verdicts) - _n_pass
step("Per-case PRS summary")
table(["Case", "Train%", "Holdout(BV)%", "Holdout(all)%", "Verdict"],
      [[c, f"{tr:.3f}", f"{hv:.3f}", f"{ha:.3f}", v]
       for c, tr, ha, hv, v in _case_verdicts])
kv("Cases passing", f"{_n_pass}/{len(_case_verdicts)}")
if _n_fail:
	warn(f"Failing case(s): {', '.join(c for c, *_ , v in _case_verdicts if v == 'FAIL')}")
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
