import os
import Uncertainty # get the uncertainty extractor
import numpy as np
import math
import multiprocessing
import multiprocessing as mp
from multiprocessing import shared_memory
import subprocess
import time
import sys
import copy
import json
import traceback
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import tempfile
sys.path.append('/shuffle.so')
import shuffle
import warnings
import contextlib
import qmc_sampling
import datetime
from Uncertainty import SAMPLER_VERSION
#print(dir(shuffle))

def _pool_provenance_path(cache_dir):
	return os.path.join(cache_dir, "pool_provenance.json")


def pool_cache_is_fresh(cache_dir, tag):
	"""True iff cache_dir/pool_provenance.json exists, is stamped with the
	CURRENT Uncertainty.SAMPLER_VERSION, and was built with sampling method
	`tag`. False (stale/missing) means the cached a/b/c_type_samples_partial
	pickles in cache_dir must NOT be reused -- they may have been produced by
	a since-fixed sampler (see SAMPLER_VERSION's docstring for why this
	matters: this exact gap silently shadowed the 2026-08-19 class-B m=3
	band-gating fix for THREE-PARAM-MSO-LLNL-N-HEPTANE_159_CORRELATION_AWARE,
	leaving 93.2% of its rows band-invalid two weeks after the fix landed).
	"""
	p = _pool_provenance_path(cache_dir)
	if not os.path.isfile(p):
		return False
	try:
		with open(p) as fh:
			prov = json.load(fh)
	except Exception:
		return False
	return (prov.get("sampler_version") == SAMPLER_VERSION
	        and prov.get("method") == tag)


def _write_pool_provenance(cache_dir, tag, seed, n_a, n_b, n_c, extra=None):
	prov = {
		"sampler_version": SAMPLER_VERSION,
		"method": tag,
		"seed": seed,
		"n_a": int(n_a), "n_b": int(n_b), "n_c": int(n_c),
		"band_check": {"tolerance": 1.0, "source": "_safeguard_validate_curves"},
		"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
	}
	if extra:
		prov.update(extra)
	with open(_pool_provenance_path(cache_dir), "w") as fh:
		json.dump(prov, fh, indent=2)
	return prov


class DesignMatrix(object):
	def __init__(self,UnsrtData,design,sample_length,ind=None,method='numpy',seed=None,
				 lhs_optimization='random-cd'):
		"""
		method : 'numpy' | 'sobol' | 'lhs'  (default 'sobol')
			Sampling method used to build the design matrix. 'numpy'
			reproduces the original pseudo-random behaviour exactly.
			'sobol' and 'lhs' build a joint low-discrepancy design across
			all reactions simultaneously (see qmc_sampling.py). Can also
			be overridden per call via getSamples(method=...) /
			getSample_partial(..., method=...).
		seed : int or None
			Master seed. None reproduces the original "no fixed seed"
			behaviour. Any other value makes the chosen method fully
			reproducible (recommended for comparing methods fairly).
		lhs_optimization : passed through to scipy.stats.qmc.LatinHypercube's
			`optimization` argument when method='lhs' (None to disable).
		"""
		self.unsrt = UnsrtData
		self.sim = sample_length
		self.design = design
		#self.n = ind
		self.n = ind
		allowed_count = int(0.70*multiprocessing.cpu_count())
		self.allowed_count = allowed_count
		self.sample_method = method
		self.sample_seed = seed
		self.lhs_optimization = lhs_optimization
		# Diagnostics collected during partial-parameter generation (rows =
		# one dict per reaction/class QuasiRandomGenerator), used to report
		# how often rejection sampling fell back off the shadow matrix.
		self.sampling_diagnostics = []
	
	def main(self,V_, num_threads, unsrt, sim):
		shuffle.shuffle_arrays(V_, num_threads, unsrt, sim)
		return V_
	def create_memory_mapped_files(self,data_dict):
		mmap_dict = {}
		temp_dir = tempfile.mkdtemp()
		for key, value in data_dict.items():
			filename = os.path.join(temp_dir, f"{key}.dat")
			mmap = np.memmap(filename, dtype=value.dtype, mode='w+', shape=value.shape)
			mmap[:] = value[:]
			mmap_dict[key] = filename
		return mmap_dict, temp_dir
	def create_shared_memory_dict(self,data_dict):
		shm_dict = {}
		for key, value in data_dict.items():
			shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
			np_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
			np_array[:] = value[:]
			shm_dict[key] = (shm, np_array)
		return shm_dict

	def cleanup_shared_memory(shm_dict):
		for shm, _ in shm_dict.values():
			shm.close()
			shm.unlink()
		
	def shuffle_values(self,rxn, filename, shape, dtype, num_shuffles):
		values = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)

		column = []
		for _ in range(num_shuffles):
			np.random.shuffle(values)
			column.append(values.copy())

		column = np.concatenate(column)
		return rxn, column

	# ── NEW: sampling-method plumbing (numpy / sobol / lhs) ────────────────

	def _resolve_method_seed(self, method=None, seed=None):
		"""Per-call override falls back to the constructor's defaults."""
		m = method if method is not None else self.sample_method
		s = seed if seed is not None else self.sample_seed
		return m, s

	def _reaction_fallback_seed(self, base_seed, rxn, *tag):
		"""Deterministic per-(reaction, class) fallback seed, or None if
		base_seed is None (matches legacy 'no fixed seed' behaviour)."""
		return qmc_sampling.derive_seed(base_seed, rxn, *tag)

	def _safeguard_validate_curves(self, V_, rxn_param_indices, tol=1.0,
	                                temps_for_validity=None, base_seed=None, label=""):
		"""
		Defense-in-depth band-validity safeguard, applied to each reaction's
		pooled class-A/B/C curves in `V_` BEFORE they are combinatorially
		shuffled together (2026-08-19 addition, per explicit instruction --
		the "Random shuffling" block just below reuses each reaction's own
		curve list 4x via `num_shuffles = 4 * unshuffled`, so a single
		invalid raw curve that slipped past one sampler would otherwise be
		replicated across up to 4x as many design-matrix rows, in
		combination with many different other reactions' curves on the
		same row -- one bad value, amplified).

		This check is independent of, and IN ADDITION TO, whatever
		validity guarantee each individual class-A/B/C sampler already
		provides (e.g. Uncertainty._psac_class_B_m3_'s own rejection step,
		fixed the same day) -- it exists so a future bug in ANY one
		sampler, or an edge case an existing one misses, cannot silently
		reach the shuffled design matrix. It is the row-assembly-stage
		counterpart to IdentifiablePRS.screen_rows -- that screens ALREADY-
		WRITTEN design-matrix rows at PRS-fit time; this catches a bad
		curve BEFORE it is ever written, so training and hold-out never see
		it and it never gets replicated by the shuffle.

		Uses the CANONICAL (positive-diagonal) Cholesky factor of each
		reaction's Sigma -- np.linalg.cholesky(cholskyDeCorrelateMat @
		cholskyDeCorrelateMat.T), not cholskyDeCorrelateMat directly -- for
		consistency with IdentifiablePRS.build_layout_blocks, which is what
		actually screens/scores this data downstream. See
		Uncertainty.run_sampling_b_partial's 2026-08-19 canonical-sign fix
		docstring for why this distinction is not cosmetic: the two factors
		can disagree by sign on some components for the SAME physical
		covariance, and a "valid" sample under the wrong one can score as
		invalid under the one actually used for screening.

		A curve failing the check is REPLACED with a fresh class-A draw
		(rejection-sampled, always valid) rather than dropped, so every
		reaction keeps exactly its intended sample count -- mirrors the
		fallback already used for m<2 padding and for
		Uncertainty.run_sampling_b_partial's own class-B fallback.
		Mutates and returns V_ in place; repairs are counted and returned
		(never silent).
		"""
		if temps_for_validity is None:
			temps_for_validity = np.linspace(300., 2500., 12)
		temps_for_validity = np.asarray(temps_for_validity, dtype=float)
		theta_full = np.column_stack([
			np.ones_like(temps_for_validity),
			np.log(temps_for_validity),
			-1.0 / temps_for_validity,
		])  # (nT, 3)

		repaired = {}
		for rxn, curves in V_.items():
			idx = rxn_param_indices.get(rxn, ())
			if len(idx) == 0 or len(curves) == 0:
				continue
			L = self.unsrt[rxn].cholskyDeCorrelateMat
			Sigma = L @ L.T
			try:
				L_full = np.linalg.cholesky(Sigma)
			except np.linalg.LinAlgError:
				eps = 1e-12 * max(float(np.trace(Sigma)), 1e-10) / 3
				L_full = np.linalg.cholesky(Sigma + eps * np.eye(3))

			Z = np.asarray(curves, dtype=float)              # (n, 3) full-space zeta
			W = theta_full @ L_full                            # (nT, 3)
			fp = np.linalg.norm(W, axis=1)                     # (nT,)
			dk = W @ Z.T                                        # (nT, n)
			ratio = (np.abs(dk) / np.maximum(fp[:, None], 1e-300)).max(axis=0)  # (n,)

			bad = np.where(ratio > tol)[0]
			if len(bad) == 0:
				continue
			pad_seed = self._reaction_fallback_seed(base_seed, rxn, "safeguard_repair", label)
			replacement = self.unsrt[rxn].getClassA_partial(
				idx, len(bad), np.random.default_rng(pad_seed))
			for j, i in enumerate(bad):
				if j < len(replacement):
					curves[int(i)] = replacement[j]
			repaired[rxn] = int(len(bad))

		if repaired:
			total = sum(repaired.values())
			print(f"\n[safeguard{(' ' + label) if label else ''}] repaired {total} "
			      f"band-invalid curve(s) across {len(repaired)} reaction(s) "
			      f"before shuffling: {repaired}")
		return V_, repaired

	def _record_diagnostics(self, diag):
		if diag is not None:
			self.sampling_diagnostics.append(diag)

	def _aggregate_diagnostics(self, diag_list, label=""):
		"""
		Combine the per-task QuasiRandomGenerator diagnostics dicts coming
		back from a multiprocessing dispatch (do_unsrt_b_partial /
		do_unsrt_c_partial each return one dict per worker task, since each
		task only ever generates a single sample) into one per-reaction
		summary, comparable in shape to the per-reaction dicts produced
		directly by the sequential class-A/f-SAC adapters.
		"""
		if not diag_list:
			return None
		rows_reserved = sum(d["rows_reserved"] for d in diag_list)
		rows_used = sum(d["rows_used"] for d in diag_list)
		draws_from_shadow = sum(d["draws_from_shadow"] for d in diag_list)
		draws_from_fallback = sum(d["draws_from_fallback"] for d in diag_list)
		samples_that_overflowed = sum(d["samples_that_overflowed"] for d in diag_list)
		total = draws_from_shadow + draws_from_fallback
		return {
			"label": label,
			"n_tasks": len(diag_list),
			"rows_reserved": rows_reserved,
			"rows_used": rows_used,
			"draws_from_shadow": draws_from_shadow,
			"draws_from_fallback": draws_from_fallback,
			"fallback_draw_fraction": (draws_from_fallback / total) if total else 0.0,
			"samples_that_overflowed": samples_that_overflowed,
			"overflow_rate": (samples_that_overflowed / len(diag_list)) if diag_list else 0.0,
		}

	def save_sampling_diagnostics(self, path="sampling_diagnostics.json"):
		"""
		Dump the QuasiRandomGenerator diagnostics collected so far (rows
		used, fraction of draws that had to fall back off the shadow
		matrix, etc.) to a JSON file -- useful as empirical evidence of
		how much rejection sampling erodes the low-discrepancy structure
		for the partial-parameter path (see the accompanying LaTeX note).
		"""
		with open(path, "w") as fh:
			json.dump(self.sampling_diagnostics, fh, indent=2)
		return path

	def getClassA_Curves(self,n_a,joint_block=None):
		"""
			This defination generates n_a numbers of class-A type curves.

			joint_block : optional (n_a, n_reactions) array. When given (see
			getSamples()), it is the joint low-discrepancy design spanning
			every reaction at once; reaction j's column is sliced out and
			used directly as that reaction's generator, replacing the
			per-reaction np.random.random_sample draw used otherwise. The
			full (non-partial) class-A/B/C path never rejects a draw, so
			there is exactly one output curve per input row -- no further
			machinery is needed here.
		"""
		ClassACurve_dict = {}
		Generator_dict = {}
		rxn_list = list(self.unsrt.keys())
		for j,rxn in enumerate(rxn_list):
			if joint_block is not None:
				generator = joint_block[:, j:j+1]
			else:
				generator = np.random.random_sample((n_a,1)) #n_a: number of A-type curves required 
			data = self.unsrt[rxn].data
			zeta_max = self.unsrt[rxn].zeta.x
			ClassA_curves = []
			for gen in generator:
				ClassA_curves.append(gen[0]*zeta_max)#getting new A-type curves
			ClassACurve_dict[rxn] = ClassA_curves
			Generator_dict[rxn] = generator	
		return ClassACurve_dict,Generator_dict
			
	def getClassB_Curves(self,n_b,joint_block=None):
		"""
			This defination generates n_b numbers of class-B type curves
			ClassB_curves: np.asarray, size of (450,3)

			joint_block : optional (n_b, 2*n_reactions) array, sliced two
			columns per reaction (a1,a2), replacing the per-reaction
			np.random.random_sample((n_b,2)) draw. See getClassA_Curves.
		"""
		ClassBCurve_dict = {}
		Generator_dict = {}
		rxn_list = list(self.unsrt.keys())
		for j,rxn in enumerate(rxn_list):
			if joint_block is not None:
				generator = joint_block[:, 2*j:2*j+2]
			else:
				generator = np.random.random_sample((n_b,2))
			data = self.unsrt[rxn].data
			data["generators_b"] = generator
			callWorkForce = Worker(self.allowed_count)	
			ClassB_curves,generator_b = callWorkForce.do_unsrt_b(data,len(generator))
			del callWorkForce
			ClassBCurve_dict[rxn] = np.asarray(ClassB_curves)
			Generator_dict[rxn] = generator_b	
		return ClassBCurve_dict,Generator_dict
	
	def getClassC_Curves(self,n_c,joint_block=None):
		"""
			This defination generates n_c numbers of class-C type curves 

			joint_block : optional (n_c, 2*n_reactions) array, sliced two
			columns per reaction (a1,a2). See getClassA_Curves.
		"""
		
		ClassC_Curve_dict = {}
		Generator_dict = {}
		rxn_list = list(self.unsrt.keys())
		for j,rxn in enumerate(rxn_list):
			if joint_block is not None:
				generator = joint_block[:, 2*j:2*j+2]
			else:
				generator = np.random.random_sample((n_c,2))
			data = self.unsrt[rxn].data
			data["generators_c"] = generator
			callWorkForce = Worker(self.allowed_count)	
			ClassC_curves,generator_c = callWorkForce.do_unsrt_c(data,len(generator))
			del callWorkForce
			ClassC_Curve_dict[rxn] = np.asarray(ClassC_curves)
			Generator_dict[rxn] = generator_c	
		return ClassC_Curve_dict,Generator_dict
	
	def generatePointOnSphere(self):
		temp = np.random.random(2)
		u1 = temp[0]
		u2 = temp[1]
		lambda_ = np.arccos(2*u1-1)-math.pi/2
		phi = 2*math.pi*u2
		x = np.cos(lambda_)*np.cos(phi)
		y = np.cos(lambda_)*np.sin(phi)
		z = np.sin(lambda_)
		#print(x,y,z)
		return x,y,z
	
	def getSA_samples(self,factor):
		design_matrix = []
		design_matrix.extend(list(float(factor)*np.eye(self.rxn_len)))
		return np.asarray(design_matrix)
	
	def getNominal_samples(self):
		design_matrix = []
		design_matrix.append(list(np.zeros(self.rxn_len)))
		return	design_matrix
	
	# ── NEW: partial-parameter curve generators ────────────────────────────

	def _get_rxn_param_indices(self, selected_params):
		"""
		Parse the flat selected_params list (one entry per active parameter,
		ordered [R1_A, R1_n, R1_Ea, R2_A, R2_n, R2_Ea, ...]) and return a
		dict mapping each reaction key to its tuple of selected parameter
		indices for that reaction.

		0 = ln(A),  1 = n,  2 = ε = Ea/R.

		Reactions whose perturbation_type is 'factor' have only 1 active
		parameter; all others have 3.
		"""
		rxn_list = list(self.unsrt.keys())
		rxn_param_indices = {}
		offset = 0
		for rxn in rxn_list:
			n_params = len(self.unsrt[rxn].activeParameters)
			chunk = selected_params[offset: offset + n_params]
			selected = tuple(i for i, s in enumerate(chunk) if s == 1)
			# Fall back to at least index 0 (A-factor) so we never get
			# an empty tuple, which would crash L_r construction.
			rxn_param_indices[rxn] = selected if selected else (0,)
			offset += n_params
		return rxn_param_indices

	def getClassA_Curves_partial(self, n_a, rxn_param_indices, shadow_a, base_seed=None):
		"""
		Class-A samples per reaction using its selected parameter subset.

		shadow_a : (n_a, 2*n_reactions) joint low-discrepancy block (see
		           getSample_partial), 2 columns per reaction: [alpha_s, sign].
		           Each reaction gets its OWN QuasiRandomGenerator wrapping
		           its column-slice, with row i reserved for that reaction's
		           i-th accepted output sample's first attempt; any retry is
		           routed to an independent fallback stream so a rejection
		           in one reaction never perturbs another reaction's rows,
		           or that reaction's own later samples.
		"""
		curves = {}
		rxn_list = list(self.unsrt.keys())
		for j, rxn in enumerate(rxn_list):
			idx = rxn_param_indices[rxn]
			sub = shadow_a[:, 2*j:2*j+2]
			fb_seed = self._reaction_fallback_seed(base_seed, rxn, "A")
			rng = qmc_sampling.QuasiRandomGenerator(sub, fallback_seed=fb_seed, label=f"A:{rxn}")
			curves[rxn] = self.unsrt[rxn].getClassA_partial(idx, n_a, rng)
			self._record_diagnostics(rng.diagnostics())
		return curves

	
	def getClassB_Curves_partial(self, n_b, rxn_param_indices, shadow_b, base_seed=None):
		"""
		Class-B samples per reaction, generated in parallel via Worker.

		Each worker process calls Uncertainty.run_sampling_b_partial which
		instantiates a fresh UncertaintyExtractor and uses the analytical
		instance methods (_psac_class_B_m2 for m=2, _psac_class_B_m3 for m=3)
		— no SLSQP.  m=1 reactions are skipped (class-B infeasible).

		shadow_b : (n_b, 4*n_reactions) joint low-discrepancy block (see
		           getSample_partial), 4 columns per reaction: [a1, a2, r1,
		           r2]. a1/a2 feed populateValues() (meaningful for the m==3
		           branch only); r1/r2 are reserved as the first-attempt
		           draw for the m<3 rejection-sampling branches. Each of the
		           n_b worker tasks for this reaction gets its own row, so
		           multiprocessing dispatch order doesn't matter — sample i
		           always means shadow row i regardless of which process or
		           in what order it finishes.

		Returns
		-------
		curves : dict  rxn -> list of full-space (length-3) zeta vectors
		"""
		curves = {}
		rxn_list = list(self.unsrt.keys())
		total_rxns = len(rxn_list)

		pbar = tqdm(
			total=total_rxns,
			desc="Generating Class-B curves",
			dynamic_ncols=True,
			leave=True
			)

		for i, rxn in enumerate(rxn_list):

			pbar.set_postfix(
				reaction=f"{i+1}/{total_rxns}",
				samples=n_b
				)

			idx = rxn_param_indices[rxn]

			if len(idx) < 2:
				curves[rxn] = []
				pbar.update(1)
				continue

			generator = np.asarray(shadow_b[:, 4*i:4*i+4])

			data = self.unsrt[rxn].data
			data["generators_b_partial"] = generator
			data["param_indices"] = idx
			data["fallback_seed_b_base"] = self._reaction_fallback_seed(base_seed, rxn, "B")

			callWorkForce = Worker(self.allowed_count)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")

				ClassB_curves, _, diag_list = callWorkForce.do_unsrt_b_partial(
					data,
					n_b
					)

			del callWorkForce

			curves[rxn] = ClassB_curves
			self._record_diagnostics(self._aggregate_diagnostics(diag_list, label=f"B:{rxn}"))
			pbar.update(1)

		pbar.close()

		return curves

	
	def getClassC_Curves_partial(self, n_c, rxn_param_indices, shadow_c, base_seed=None):
		"""
		Class-C samples per reaction, generated in parallel via Worker.

		Each worker process calls Uncertainty.run_sampling_c_partial.
		m=1 reactions are skipped (class-C infeasible).

		shadow_c : (n_c, 2*n_reactions) joint low-discrepancy block, 2
		           columns per reaction: [r1, r2], reserved as the
		           first-attempt draw for the rejection-sampling routine.

		Returns
		-------
		curves : dict  rxn -> list of full-space (length-3) zeta vectors
		"""
		curves = {}
		rxn_list = list(self.unsrt.keys())
		total_rxns = len(rxn_list)

		pbar = tqdm(
			total=total_rxns,
			desc="Generating Class-C curves",
			dynamic_ncols=True,
			leave=True
			)

		for i, rxn in enumerate(rxn_list):

			pbar.set_postfix(
				reaction=f"{i+1}/{total_rxns}",
				samples=n_c
				)

			idx = rxn_param_indices[rxn]

			generator = np.asarray(shadow_c[:, 2*i:2*i+2])

			data = self.unsrt[rxn].data
			data["generators_c_partial"] = generator
			data["param_indices"] = idx
			data["fallback_seed_c_base"] = self._reaction_fallback_seed(base_seed, rxn, "C")

			callWorkForce = Worker(self.allowed_count)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")

				ClassC_curves, _, diag_list = callWorkForce.do_unsrt_c_partial(
					data,
					n_c
					)

			del callWorkForce

			curves[rxn] = ClassC_curves
			self._record_diagnostics(self._aggregate_diagnostics(diag_list, label=f"C:{rxn}"))
			pbar.update(1)

		pbar.close()

		return curves


	def get_fSAC_partial(self, n_fsac, rxn_param_indices, shadow_fsac, base_seed=None):
		"""
		f-SAC samples per reaction using the analytical class method
		UncertaintyExtractor.get_fsac_partial().

		m=1 → class-A (f-SAC undefined for single parameter; see the
		      recursion note in Uncertainty._psac_fsac about looser row
		      alignment in this edge case).
		m=2 → 2×2 system (T_min + T_mid).
		m=3 → 3×3 system (T_min + T_mid + T_max).

		shadow_fsac : (n_fsac, 3*n_reactions) joint low-discrepancy block,
		              3 columns per reaction: [r1, r2, r_mid] -- covers the
		              up-to-3 raw draws _psac_fsac consumes per attempt.
		              Same per-reaction QuasiRandomGenerator + fallback
		              treatment as class-A/B/C-partial above.
		"""
		curves = {}
		rxn_list = list(self.unsrt.keys())
		for j, rxn in enumerate(tqdm(rxn_list, desc="Generating f-SAC (partial)")):
			idx = rxn_param_indices[rxn]
			sub = shadow_fsac[:, 3*j:3*j+3]
			fb_seed = self._reaction_fallback_seed(base_seed, rxn, "fSAC")
			rng = qmc_sampling.QuasiRandomGenerator(sub, fallback_seed=fb_seed, label=f"fSAC:{rxn}")
			curves[rxn] = self.unsrt[rxn].get_fsac_partial(idx, n_fsac, rng)
			self._record_diagnostics(rng.diagnostics())
		return curves

	# ── END NEW partial generators ─────────────────────────────────────────

	def refresh_class_pools(self, case_index, selected_params, method=None, seed=None,
	                          force=False, base_dir=None):
		"""Regenerate ONLY this case's class-A/B/C curve pools
		({a,b,c}_type_samples_partial_<method>.pkl) through the CURRENT
		(band-gated) samplers, without touching DesignMatrix.csv,
		pDesignMatrix.csv, the shuffle cache, or running any simulation.

		This is the zero-simulation-cost half of the 2026-08-19 bad-sample
		fix: a case's DesignMatrix.csv/pDesignMatrix.csv (and the
		simulations that were actually run against those specific rows) are
		a sunk cost worth keeping -- band-valid rows in there stay band-valid
		regardless of what pool built them. But AdaptiveResampler.
		DOptimalResampler's Stage-3 candidate generator draws NEW candidates
		straight from these class pools, so if the pools are stale (e.g. the
		2026-08-05 pre-fix batch, 60.7% class-B rows band-invalid, worst
		85.2x), Stage 3 spends its whole candidate budget on points that get
		rejected by the band gate before a single new simulation is even
		requested -- exactly what happened to Case 52 (0/2940 band-valid,
		every one of 8 passes) and Case 56 (34/2748, the same 34 points
		re-picked 8 times). Refreshing the pools with pure numpy (seconds to
		low minutes per case, no Chemkin/FlameMaster call) fixes that at the
		source: run this once per case before constructing DOptimalResampler
		for it, whenever pool_cache_is_fresh(cache_dir, tag) is False.

		Sizing mirrors getSample_partial's A1+B1+C1 branch exactly (n_a/n_b/
		n_c nominal ratios, sobol power-of-2 rounding) so a refreshed pool is
		drawn from the same class-count budget the original design matrix
		used -- this is a like-for-like resampler-quality fix, not a
		resizing.

		Returns the written pool_provenance.json dict. Raises AssertionError
		if any refreshed pool is not 100% band-valid (the whole point of
		routing through the fixed samplers + _safeguard_validate_curves is
		that it should be -- a failure here means a NEW sampler defect, not
		this case's data, and must not be silently absorbed).
		"""
		method, seed = self._resolve_method_seed(method, seed)
		tag = method
		# base_dir: the STUDY ROOT (home_dir in run_script.py), NOT cwd --
		# by the time Stage 3 constructs a resampler the per-case PRS loop
		# has already os.chdir()'d into optDir (home_dir/Opt), so a bare
		# relative "DM_FOR_PARTIAL_PRS" here would resolve to the WRONG
		# place (home_dir/Opt/DM_FOR_PARTIAL_PRS instead of
		# home_dir/DM_FOR_PARTIAL_PRS). Mirrors AdaptiveResampler.
		# OutlierResampler's own `base_dir` convention exactly (both are
		# always called with home_dir by run_script.py) so a case's pools
		# and its resampler agree on the same directory by construction.
		root = base_dir if base_dir is not None else os.getcwd()
		cache_dir = os.path.join(root, "DM_FOR_PARTIAL_PRS", str(case_index))
		os.makedirs(cache_dir, exist_ok=True)

		if not force and pool_cache_is_fresh(cache_dir, tag):
			print(f"[Case-{case_index}] class pools already fresh "
			      f"(SAMPLER_VERSION={SAMPLER_VERSION!r}, method={tag!r}) -- skipping refresh.")
			with open(_pool_provenance_path(cache_dir)) as fh:
				return json.load(fh)

		n_a_req = int(0.1  * self.sim * 0.2)
		n_b_req = int(0.45 * self.sim * 0.2)
		n_c_req = int(self.sim * 0.2) - n_a_req - n_b_req
		if method == "sobol":
			n_a, n_b, n_c = (qmc_sampling.ceil_pow2(n_a_req),
			                 qmc_sampling.ceil_pow2(n_b_req),
			                 qmc_sampling.ceil_pow2(n_c_req))
		else:
			n_a, n_b, n_c = n_a_req, n_b_req, n_c_req

		rxn_param_indices = self._get_rxn_param_indices(selected_params)
		rxn_list = list(self.unsrt.keys())
		J = len(rxn_list)

		shadow_a, n_a = qmc_sampling.build_joint_shadow(
			method, n_a, J, dims_per_reaction=2, seed=seed, optimization=self.lhs_optimization)
		shadow_b, n_b = qmc_sampling.build_joint_shadow(
			method, n_b, J, dims_per_reaction=4, seed=seed, optimization=self.lhs_optimization)
		shadow_c, n_c = qmc_sampling.build_joint_shadow(
			method, n_c, J, dims_per_reaction=2, seed=seed, optimization=self.lhs_optimization)

		print(f"[Case-{case_index}] refreshing class pools through SAMPLER_VERSION="
		      f"{SAMPLER_VERSION!r}: n_a={n_a} n_b={n_b} n_c={n_c} per reaction, method={method}")
		a_curves = self.getClassA_Curves_partial(n_a, rxn_param_indices, shadow_a, base_seed=seed)
		b_curves = self.getClassB_Curves_partial(n_b, rxn_param_indices, shadow_b, base_seed=seed)
		c_curves = self.getClassC_Curves_partial(n_c, rxn_param_indices, shadow_c, base_seed=seed)

		# Same defense-in-depth safeguard the main generation path uses --
		# repairs (rather than drops) any curve a class sampler still let
		# through invalid, so every reaction keeps its full requested count.
		pools = {"A": a_curves, "B": b_curves, "C": c_curves}
		repairs_total = {}
		for label, curves in pools.items():
			curves, repaired = self._safeguard_validate_curves(
				curves, rxn_param_indices, tol=1.0, base_seed=seed,
				label=f"refresh-case-{case_index}-{label}")
			pools[label] = curves
			if repaired:
				repairs_total[label] = repaired

		for label, curves in pools.items():
			with open(f'{cache_dir}/{label.lower()}_type_samples_partial_{tag}.pkl', 'wb') as fh:
				pickle.dump(curves, fh)

		# Verify 100% band-valid -- this is the whole point of the refresh,
		# so assert it rather than trust the sampler + safeguard silently.
		T_check = np.linspace(300., 2500., 12)
		theta_full = np.column_stack([np.ones_like(T_check), np.log(T_check), -1.0 / T_check])
		worst_ratio, worst_label, worst_rxn = 0.0, None, None
		for label, curves in pools.items():
			for rxn, Z in curves.items():
				Z = np.asarray(Z, dtype=float)
				if Z.ndim != 2 or Z.shape[0] == 0:
					continue
				L = self.unsrt[rxn].cholskyDeCorrelateMat
				Sigma = np.asarray(L, dtype=float) @ np.asarray(L, dtype=float).T
				try:
					L_full = np.linalg.cholesky(Sigma)
				except np.linalg.LinAlgError:
					eps = 1e-12 * max(float(np.trace(Sigma)), 1e-10) / 3
					L_full = np.linalg.cholesky(Sigma + eps * np.eye(3))
				W = theta_full @ L_full
				fp = np.linalg.norm(W, axis=1)
				ratio = (np.abs(W @ Z.T) / np.maximum(fp[:, None], 1e-300)).max()
				if ratio > worst_ratio:
					worst_ratio, worst_label, worst_rxn = float(ratio), label, rxn
		assert worst_ratio <= 1.0 + 1e-9, (
			f"[Case-{case_index}] refresh_class_pools produced a band-invalid curve "
			f"after safeguard repair -- class {worst_label}, reaction {worst_rxn!r}, "
			f"ratio {worst_ratio:.3f}x. This means a NEW sampler defect, not stale "
			f"data -- do not proceed until the sampler is fixed.")

		prov = _write_pool_provenance(
			cache_dir, tag, seed, n_a, n_b, n_c,
			extra={"refreshed": True, "repairs": repairs_total,
			       "verified_max_ratio": worst_ratio})
		print(f"[Case-{case_index}] class pools refreshed: 100% band-valid "
		      f"(max ratio {worst_ratio:.3f}x <= 1.0), repairs={repairs_total or 'none'}")
		return prov

	def getSample_partial(self,case_index,selected_params,method=None,seed=None):
		print("\nStarting to generate partial-design matrix!!\n")
		method, seed = self._resolve_method_seed(method, seed)

		if self.design == 'A-facto': 
			if "DM_FOR_PARTIAL_PRS" not in os.listdir():
				os.mkdir("DM_FOR_PARTIAL_PRS")
				os.chdir("DM_FOR_PARTIAL_PRS")
				if f"{case_index}" not in os.listdir():
					os.mkdir(case_index)
				os.chdir("..")
			else:
				os.chdir("DM_FOR_PARTIAL_PRS")
				if f"{case_index}" not in os.listdir():
					os.mkdir(case_index)
				os.chdir("..")

			# Cache filename is tagged with the sampling method, so switching
			# methods between runs never silently reuses another method's
			# stale cached samples (the original cache name never depended
			# on how the samples were generated -- a real bug).
			cache_name = f"DesignMatrix_{method}.pkl"
			if cache_name not in os.listdir(f"DM_FOR_PARTIAL_PRS/{case_index}"):
				pts, n_actual = qmc_sampling.joint_uniform_block(method, self.sim, self.n, seed=seed)
				design_matrix = list(2.0*pts - 1.0)
				if n_actual != self.sim:
					print(f"\n[{method}] A-facto sample count {self.sim} -> {n_actual} "
						  f"(rounded up to a power of 2 for Sobol's balance property).")
				with open(f'DM_FOR_PARTIAL_PRS/{case_index}/{cache_name}', 'wb') as file_:
					pickle.dump(design_matrix, file_)
			else: 
				with open(f'DM_FOR_PARTIAL_PRS/{case_index}/{cache_name}', 'rb') as file_:
					design_matrix = pickle.load(file_)
				print(f"\nA-factor samples ({method}) are generated")

			s =""
			p_s = "" 
			p_ss = ""
			ss = ""
			p_design_matrix = []
			p_selection_matrix = []
			selection_matrix = []
			for row in design_matrix:
				row_ = []
				temo = []
				temp = []
				for index,element in enumerate(row):
					if selected_params[index] == 1:
						s+=f"{element},"
						p_s+=f"{element},"
						p_ss+="1.0,"
						ss+="1.0,"
						row_.append(element)
						temo.append(1.0)
						temp.append(1.0)
					else:
						s+="0.0,"
						p_ss+="0.0,"
						ss+="1.0,"
						temo.append(0.0)
						temp.append(1.0)
				p_design_matrix.append(row_)
				p_selection_matrix.append(temo)
				selection_matrix.append(temp)
				s+="\n"
				p_s+="\n"
				p_ss+="\n"
				ss+="\n"

			ff = open(f'DM_FOR_PARTIAL_PRS/{case_index}/DesignMatrix.csv','w').write(s)
			ff = open(f'DM_FOR_PARTIAL_PRS/{case_index}/pDesignMatrix.csv','w').write(p_s)
			ff = open(f'DM_FOR_PARTIAL_PRS/{case_index}/pSelectionMatrix.csv','w').write(p_ss)
			ff = open(f'DM_FOR_PARTIAL_PRS/{case_index}/SelectionMatrix.csv','w').write(ss)
			
			return np.asarray(design_matrix),np.asarray(selection_matrix),np.asarray(p_design_matrix),np.asarray(p_selection_matrix)
			#return np.asarray(design_matrix)
		
		elif self.design == "A1+B1+C1":
			# ──────────────────────────────────────────────────────────────
			# Partial-parameter A1+B1+C1 design matrix
			# Uses the reduced Cholesky L_r for each reaction's selected
			# parameter subset.  The full-space zeta vector (zeros at
			# inactive positions) is returned so that downstream YAML
			# generators require no changes.
			#
			# Sampling architecture: one joint low-discrepancy "shadow"
			# block per class (A/B/C/f-SAC) is built ONCE here, spanning
			# every reaction simultaneously (qmc_sampling.build_joint_shadow).
			# Each reaction is then handed its own column-slice of that
			# block by getClassA/B/C_Curves_partial / get_fSAC_partial,
			# which is what gives the unshuffled block its joint (not just
			# per-reaction-marginal) low-discrepancy coverage. See the
			# accompanying note for why the subsequent random-shuffling
			# block (kept unchanged below) partially trades that joint
			# structure back for combinatorial variety.
			# ──────────────────────────────────────────────────────────────
			#self.sim = self.sim*2
			tic = time.time()
			design_matrix = []

			# Nominal sample-count ratios (identical to the original code)
			n_a_req    = int(0.1  * self.sim * 0.2)
			n_b_req    = int(0.45 * self.sim * 0.2)
			#n_c_req    = int(0.35 * self.sim * 0.2)
			#n_fsac_req    = int(self.sim * 0.2) - n_a_req - n_b_req-n_c_req
			n_c_req = int(self.sim * 0.2) - n_a_req - n_b_req
			n_fsac_req = 0
			# Sobol needs a full power-of-2 batch per class. Per your call,
			# that batch is grown (never truncated) and the resulting design
			# matrix is allowed to grow accordingly. 'lhs'/'numpy' keep the
			# nominal counts exactly, so a side-by-side method comparison at
			# "the same requested n" stays meaningful.
			if method == "sobol":
				n_a    = qmc_sampling.ceil_pow2(n_a_req)
				n_b    = qmc_sampling.ceil_pow2(n_b_req)
				n_c    = qmc_sampling.ceil_pow2(n_c_req)
				n_fsac = qmc_sampling.ceil_pow2(n_fsac_req)
			else:
				n_a, n_b, n_c, n_fsac = n_a_req, n_b_req, n_c_req, n_fsac_req
			
			unshuffled = n_a + n_b + n_c + n_fsac_req
			#unshuffled = 0

			print(f"\n[{method}] partial A1+B1+C1 sample counts: "
				  f"n_a={n_a} (req {n_a_req}), n_b={n_b} (req {n_b_req}), "
				  f"n_c={n_c} (req {n_c_req}), n_fsac={n_fsac} (req {n_fsac_req}), "
				  f"unshuffled={unshuffled}")

			# ── Directory setup ────────────────────────────────────────────
			if "DM_FOR_PARTIAL_PRS" not in os.listdir():
				os.mkdir("DM_FOR_PARTIAL_PRS")
				os.chdir("DM_FOR_PARTIAL_PRS")
				if f"{case_index}" not in os.listdir():
					os.mkdir(case_index)
				os.chdir("..")
			else:
				os.chdir("DM_FOR_PARTIAL_PRS")
				if f"{case_index}" not in os.listdir():
					os.mkdir(case_index)
				os.chdir("..")

			cache_dir = f"DM_FOR_PARTIAL_PRS/{case_index}"
			# All cache filenames below are tagged with `tag` (the sampling
			# method) -- fixes the original bug where the cache never
			# depended on how the samples were generated, so switching
			# methods would silently reuse another method's stale samples.
			tag = method

			# ── Cache-freshness gate (2026-08-19) ───────────────────────────
			# A cached a/b/c_type_samples_partial_{tag}.pkl from a sampler
			# version older than the CURRENT Uncertainty.SAMPLER_VERSION must
			# never be silently reused -- see pool_cache_is_fresh's docstring
			# for why (this exact gap is what let the 2026-08-19 class-B m=3
			# band-gating fix sit unused against a 2026-08-05 cache for two
			# weeks, leaving 93.2% of this study's rows band-invalid). If
			# stale, delete the class-pool pickles + the shuffled cache (NOT
			# DesignMatrix.csv/pDesignMatrix.csv themselves -- those are
			# gated separately, at the higher partial_PRS_system.py level,
			# via dm_provenance.json) so the checks below regenerate them.
			if not pool_cache_is_fresh(cache_dir, tag):
				stale = [f for f in (f"a_type_samples_partial_{tag}.pkl",
				                      f"b_type_samples_partial_{tag}.pkl",
				                      f"c_type_samples_partial_{tag}.pkl",
				                      f"RANDOM_SHUFFLING_partial_{tag}.pkl")
				         if f in os.listdir(cache_dir)]
				if stale:
					print(f"\n[Case-{case_index}] cached class-pool samples predate "
					      f"SAMPLER_VERSION={SAMPLER_VERSION!r} (or used a different "
					      f"method) -- regenerating {len(stale)} stale pool file(s) "
					      f"instead of reusing them: {', '.join(stale)}")
					for f in stale:
						os.remove(os.path.join(cache_dir, f))

			# ── Build per-reaction active-parameter index tuples ──────────
			rxn_param_indices = self._get_rxn_param_indices(selected_params)
			rxn_list = list(self.unsrt.keys())
			J = len(rxn_list)

			# ── Build the joint low-discrepancy shadow blocks ONCE ─────────
			# One joint draw per class, spanning every reaction at once.
			# These stay in their natural [0,1) form -- every downstream
			# consumer (populateValues' raw a1/a2, or the
			# QuasiRandomGenerator-wrapped rejection-sampling draws) already
			# does its own (and possibly different, e.g. (-1,1) vs (0.3,1))
			# affine rescaling per call, exactly mirroring what
			# np.random.random_sample()/rng.uniform() did before.
			shadow_a, n_a_actual = qmc_sampling.build_joint_shadow(
				method, n_a, J, dims_per_reaction=2, seed=seed, optimization=self.lhs_optimization)
			shadow_b, n_b_actual = qmc_sampling.build_joint_shadow(
				method, n_b, J, dims_per_reaction=4, seed=seed, optimization=self.lhs_optimization)
			shadow_c, n_c_actual = qmc_sampling.build_joint_shadow(
				method, n_c, J, dims_per_reaction=2, seed=seed, optimization=self.lhs_optimization)
			shadow_fsac, n_fsac_actual = qmc_sampling.build_joint_shadow(
				method, n_fsac, J, dims_per_reaction=3, seed=seed, optimization=self.lhs_optimization)
			# n_a/n_b/n_c/n_fsac were already rounded up above for 'sobol',
			# so n_*_actual == n_* here; kept as separate variables only as
			# a defensive cross-check (build_joint_shadow is the single
			# source of truth for how many rows a method actually produces).
			n_a, n_b, n_c, n_fsac = n_a_actual, n_b_actual, n_c_actual, n_fsac_actual
			unshuffled = n_a + n_b + n_c #+ n_fsac
			#unshuffled = 0
			# ── Class-A (partial) ─────────────────────────────────────────
			if f"a_type_samples_partial_{tag}.pkl" not in os.listdir(cache_dir):
				print(f"\nGenerating partial class-A curves (n={n_a} per rxn, method={method}) ...")
				a_curves = self.getClassA_Curves_partial(n_a, rxn_param_indices, shadow_a, base_seed=seed)
				with open(f'{cache_dir}/a_type_samples_partial_{tag}.pkl', 'wb') as fh:
					pickle.dump(a_curves, fh)
			else:
				with open(f'{cache_dir}/a_type_samples_partial_{tag}.pkl', 'rb') as fh:
					a_curves = pickle.load(fh)
				print(f"\nPartial class-A curves ({method}) loaded from cache")

			# ── Class-B (partial, analytical) ─────────────────────────────
			if f"b_type_samples_partial_{tag}.pkl" not in os.listdir(cache_dir):
				print(f"\nGenerating partial class-B curves (n={n_b} per rxn, method={method}) ...")
				b_curves = self.getClassB_Curves_partial(n_b, rxn_param_indices, shadow_b, base_seed=seed)
				with open(f'{cache_dir}/b_type_samples_partial_{tag}.pkl', 'wb') as fh:
					pickle.dump(b_curves, fh)
			else:
				with open(f'{cache_dir}/b_type_samples_partial_{tag}.pkl', 'rb') as fh:
					b_curves = pickle.load(fh)
				print(f"\nPartial class-B curves ({method}) loaded from cache")

			# ── Class-C (partial) ─────────────────────────────────────────
			if f"c_type_samples_partial_{tag}.pkl" not in os.listdir(cache_dir):
				print(f"\nGenerating partial class-C curves (n={n_c} per rxn, method={method}) ...")
				c_curves = self.getClassC_Curves_partial(n_c, rxn_param_indices, shadow_c, base_seed=seed)
				with open(f'{cache_dir}/c_type_samples_partial_{tag}.pkl', 'wb') as fh:
					pickle.dump(c_curves, fh)
			else:
				with open(f'{cache_dir}/c_type_samples_partial_{tag}.pkl', 'rb') as fh:
					c_curves = pickle.load(fh)
				print(f"\nPartial class-C curves ({method}) loaded from cache")

			# Stamp/refresh provenance now that all three class pools are
			# known-fresh (either just regenerated above, or already valid
			# per the gate at the top of this branch) -- this is what lets a
			# FUTURE run's freshness check trust this cache without having to
			# re-derive it, and what AdaptiveResampler.DOptimalResampler
			# asserts against before it will draw Stage-3 candidates from
			# these same pkl files (see AdaptiveResampler.py's
			# _load_class_pool / __init__ provenance assertion).
			_write_pool_provenance(cache_dir, tag, seed, n_a, n_b, n_c)

			"""
			if f"fSAC_samples_partial_{tag}.pkl" not in os.listdir(cache_dir):
				print(f"\nGenerating partial f-SAC samples (n={n_fsac} per rxn, method={method}) ...")
				raw_fsac = self.get_fSAC_partial(
				    n_fsac, rxn_param_indices, shadow_fsac, base_seed=seed
				)
				V = {}
				for rxn in rxn_list:
					idx    = rxn_param_indices[rxn]
					fsac_l = list(raw_fsac.get(rxn, []))
					if len(fsac_l) < n_fsac:
						pad_seed = self._reaction_fallback_seed(seed, rxn, "fsac_pad")
						pad = self.unsrt[rxn].getClassA_partial(
						    idx, n_fsac - len(fsac_l), np.random.default_rng(pad_seed)
						)
						fsac_l += pad
					V[rxn] = fsac_l[:n_fsac]
				with open(f'{cache_dir}/fSAC_samples_partial_{tag}.pkl', 'wb') as fh:
					pickle.dump(V, fh)
			else:
				with open(f'{cache_dir}/fSAC_samples_partial_{tag}.pkl', 'rb') as fh:
					V = pickle.load(fh)
				print(f"\nPartial f-SAC curves ({method}) loaded from cache")
			"""
			# ── Build V_ (unshuffled direct samples, exactly `unshuffled` rows) ─
			# For m=1 reactions, class-B and class-C lists are empty;
			# those slots are filled with extra class-A samples so V_[rxn]
			# always has exactly `unshuffled` rows. This backfill is a
			# secondary, degenerate-case correction (not part of the primary
			# joint design above), so it uses a plain but deterministically
			# seeded RNG rather than the joint Sobol/LHS shadow blocks.
			V_ = {}
			for rxn in tqdm(rxn_list, desc="Populating V_ (partial)"):
				idx   = rxn_param_indices[rxn]
				m     = len(idx)
				a_p   = list(a_curves.get(rxn, []))
				b_p   = list(b_curves.get(rxn, []))
				c_p   = list(c_curves.get(rxn, []))

				pad_seed = self._reaction_fallback_seed(seed, rxn, "pad")
				pad_rng  = np.random.default_rng(pad_seed)

				if m < 2:
					# Pad B and C slots with extra class-A
					extra = self.unsrt[rxn].getClassA_partial(
					    idx, n_b + n_c, pad_rng
					)
					b_p = extra[:n_b]
					c_p = extra[n_b:]


				combined = a_p + b_p + c_p
				if len(combined) < unshuffled:
					pad = self.unsrt[rxn].getClassA_partial(
					    idx, unshuffled - len(combined), pad_rng
					)
					combined += pad
				V_[rxn] = np.asarray(combined[:unshuffled])

			#----Appedning the f-SAC samples in V_
			#for rxn in rxn_list:
			#	V_[rxn] = np.concatenate(
			#		[V_[rxn], np.asarray(V[rxn])],
			#		axis=0
			#		)

			# ── Safeguard: validate every reaction's pooled curves against
			# the reaction's TRUE physical band before they get shuffled
			# together (see _safeguard_validate_curves's docstring -- this
			# is what stops one bad curve from being amplified into up to
			# 4x as many design-matrix rows by the shuffle block below).
			V_, _curve_repairs = self._safeguard_validate_curves(
				V_, rxn_param_indices, tol=1.0, base_seed=seed, label=f"case-{case_index}")
			if _curve_repairs:
				try:
					with open(f'{cache_dir}/curve_safeguard_repairs_{tag}.json', 'w') as fh:
						json.dump(_curve_repairs, fh, indent=2)
				except Exception as exc:
					print(f"[warn] could not write curve_safeguard_repairs: {exc}")

			# ── Random shuffling ──────────────────────────────────────────
			# Left exactly as in the original code, per your instruction.
			# Note: independently permuting each reaction's already-jointly-
			# sampled curves means the joint (cross-reaction) structure the
			# shadow matrix built above is *not* preserved through this
			# step -- only each reaction's own marginal distribution is
			# (every value that exists survives, just at a different row).
			# Structurally this makes the shuffled block equivalent to a
			# Latin-Hypercube-style joint construction regardless of which
			# method seeded the unshuffled block; see the accompanying note
			# for the measured numbers and a suggested designed-permutation
			# alternative.
			if f"RANDOM_SHUFFLING_partial_{tag}.pkl" not in os.listdir(cache_dir):
				V_s    = {}
				V_copy = copy.deepcopy(V_)
				for rxn in tqdm(rxn_list, desc="Random shuffling (partial)"):
					column = []
					values = V_copy[rxn]
					for _ in range(4):
						np.random.shuffle(values)
						column.append(values.copy())
					V_s[rxn] = np.concatenate(column)
				with open(f'{cache_dir}/RANDOM_SHUFFLING_partial_{tag}.pkl', 'wb') as fh:
					pickle.dump(V_s, fh)
			else:
				with open(f'{cache_dir}/RANDOM_SHUFFLING_partial_{tag}.pkl', 'rb') as fh:
					V_s = pickle.load(fh)
				print(f"\nPartial shuffled curves ({method}) loaded from cache")

			num_shuffles = 4 * unshuffled   # matches original ratio

			# ── f-SAC (partial, analytical class method) ──────────────────
			

			# ── Assemble design-matrix rows ────────────────────────────────
			# Each row concatenates the full-length (3-element) zeta from
			# every reaction.  Inactive positions are 0.

			# Block 1: unshuffled direct samples
			
			for i in range(unshuffled):
				row = []
				for rxn in rxn_list:
					row.extend(V_[rxn][i])
				design_matrix.append(row)
			
			# Block 2: shuffled rows (4x repeat)
			for i in range(num_shuffles):
				row = []
				for rxn in rxn_list:
					row.extend(V_s[rxn][i])
				design_matrix.append(row)
			"""
			# Block 3: f-SAC rows
			for i in range(n_fsac):
				row = []
				for rxn in rxn_list:
					row.extend(V[rxn][i])
				design_matrix.append(row)
			"""
			tok = time.time()
			nominal_total = int(self.sim*0.2) + 4*int(self.sim*0.2)
			print(f"\nTime to build partial A1+B1+C1 DM ({method}): {tok-tic:.1f} s "
				  f"-- {len(design_matrix)} total rows (nominal target was {nominal_total})")

			try:
				diag_path = self.save_sampling_diagnostics(f"{cache_dir}/sampling_diagnostics_{tag}.json")
				print(f"Sampling diagnostics written to {diag_path}")
			except Exception as exc:
				print(f"[warn] could not write sampling diagnostics: {exc}")

			# ── Build selection matrices and write CSV files ───────────────
			s                 = ""
			p_s               = ""
			selected_string   = ""
			p_selected_string = ""
			p_design_matrix   = []
			p_selection_matrix = []
			selection_matrix   = []

			for row in design_matrix:
				row_   = []
				temo   = []
				temp   = []
				for index, element in enumerate(row):
					if selected_params[index] == 1:
						s                 += f"{element},"
						p_s               += f"{element},"
						selected_string   += "1.0,"
						p_selected_string += "1.0,"
						row_.append(element)
						temo.append(1.0)
						temp.append(1.0)
					else:
						selected_string   += "1.0,"
						p_selected_string += "0.0,"
						s   += f"{element},"
						temo.append(0.0)
						temp.append(1.0)
				p_design_matrix.append(row_)
				p_selection_matrix.append(temo)
				selection_matrix.append(temp)
				s                 += "\n"
				p_s               += "\n"
				selected_string   += "\n"
				p_selected_string += "\n"

			open(f'{cache_dir}/DesignMatrix.csv',    'w').write(s)
			open(f'{cache_dir}/pDesignMatrix.csv',   'w').write(p_s)
			open(f'{cache_dir}/pSelectionMatrix.csv','w').write(p_selected_string)
			open(f'{cache_dir}/SelectionMatrix.csv', 'w').write(selected_string)

			return (np.asarray(design_matrix),
			        np.asarray(selection_matrix),
			        np.asarray(p_design_matrix),
			        np.asarray(p_selection_matrix))
	
	def getSamples(self,method=None,seed=None):
		print("\nStarting to generate design matrix!!\n")
		method, seed = self._resolve_method_seed(method, seed)
		if self.design == 'A-facto': 
			#self.sim = 1 + 2* self.n+self.n*(self.n-1)/2
			pts, n_actual = qmc_sampling.joint_uniform_block(method, self.sim, self.n, seed=seed)
			design_matrix = list(2.0*pts - 1.0)
			if n_actual != self.sim:
				print(f"\n[{method}] A-facto sample count {self.sim} -> {n_actual} "
					  f"(rounded up to a power of 2 for Sobol's balance property).")
			design_matrix.extend(list(np.eye(self.n)))
			s =""
			for row in design_matrix:
				for element in row:
					s+=f"{element},"
				s+="\n"
			ff = open('DesignMatrix.csv','w').write(s)
			return np.asarray(design_matrix)
		elif self.design == "A1+B1+C1":
			tic = time.time()
			design_matrix = []

			n_a_req = int(0.1*self.sim*0.2)
			n_b_req = int(0.45*self.sim*0.2)
			n_c_req = int(self.sim*0.2-n_a_req-n_b_req)

			# Sobol needs a full power-of-2 batch; the batch is grown (not
			# truncated) and the design matrix grows to match. 'lhs'/'numpy'
			# use the nominal counts exactly.
			if method == "sobol":
				n_a = qmc_sampling.ceil_pow2(n_a_req)
				n_b = qmc_sampling.ceil_pow2(n_b_req)
				n_c = qmc_sampling.ceil_pow2(n_c_req)
			else:
				n_a, n_b, n_c = n_a_req, n_b_req, n_c_req

			print(f"\n[{method}] A1+B1+C1 sample counts: n_a={n_a} (req {n_a_req}), "
				  f"n_b={n_b} (req {n_b_req}), n_c={n_c} (req {n_c_req})")

			rxn_list = list(self.unsrt.keys())
			J = len(rxn_list)
			# One joint low-discrepancy draw per class, spanning every
			# reaction at once -- see getClassA_Curves/B/C above for how
			# each reaction's column-block gets sliced out.
			shadow_a, n_a = qmc_sampling.build_joint_shadow(method, n_a, J, 1, seed=seed)
			shadow_b, n_b = qmc_sampling.build_joint_shadow(method, n_b, J, 2, seed=seed, optimization=self.lhs_optimization)
			shadow_c, n_c = qmc_sampling.build_joint_shadow(method, n_c, J, 2, seed=seed, optimization=self.lhs_optimization)

			tag = method  # cache files are tagged by method -- see partial path

			if f"a_type_samples_{tag}.pkl" not in os.listdir():
				a_curves_dict, generator_a = self.getClassA_Curves(n_a, joint_block=shadow_a)# Returns 100 class-A Arrhenius samples
				with open(f'a_type_samples_{tag}.pkl', 'wb') as file_:
					pickle.dump(a_curves_dict, file_)
				print("\nA-type curves generated")
			else:
				# Load the object from the file
				with open(f'a_type_samples_{tag}.pkl', 'rb') as file_:
					a_curves_dict = pickle.load(file_)
				print(f"\nA-type curves ({method}) loaded from cache")
			
			if f"b_type_samples_{tag}.pkl" not in os.listdir():	
				b_curves_dict, generator_b = self.getClassB_Curves(n_b, joint_block=shadow_b)# Returns 450 class-B Arrhenius samples
				with open(f'b_type_samples_{tag}.pkl', 'wb') as file_:
					pickle.dump(b_curves_dict, file_)
				print("\nB-type curves generated")
			else:
				# Load the object from the file
				with open(f'b_type_samples_{tag}.pkl', 'rb') as file_:
					b_curves_dict = pickle.load(file_)
				print(f"\nB-type curves ({method}) loaded from cache")
			
			if f"c_type_samples_{tag}.pkl" not in os.listdir():
				c_curves_dict, generator_c = self.getClassC_Curves(n_c, joint_block=shadow_c)# Returns 450 class-C Arrhenius samples	
				with open(f'c_type_samples_{tag}.pkl', 'wb') as file_:
					pickle.dump(c_curves_dict, file_)
				print("\nC-type curves generated")
			else:
				# Load the object from the file
				with open(f'c_type_samples_{tag}.pkl', 'rb') as file_:
					c_curves_dict = pickle.load(file_)
				print(f"\nC-type curves ({method}) loaded from cache")
				
			V_ = {}
			for rxn in tqdm(self.unsrt,desc="Populating V_"):
				temp = []
				for sample_a in a_curves_dict[rxn]: 
					temp.append(sample_a)
				for sample_b in b_curves_dict[rxn]: 
					temp.append(sample_b)
				for sample_c in c_curves_dict[rxn]: 
					temp.append(sample_c)
				V_[rxn] = np.asarray(temp)
			
			V_s = {}#Doing random shuffling
			#Deepcopy the unshuffled samples first
			#print(n_a,n_b,n_c)
			#print(len(n_a),len(n_b),len(n_c))
			V_copy = copy.deepcopy(V_)
			
			for rxn in tqdm(self.unsrt, desc="Doing random shuffling"):
				column = []
				# Number of shuffles
				num_shuffles = int(self.sim * 0.8)
				# Get the values to shuffle
				values = V_copy[rxn]

				# Repeat the shuffle process
				for i in range(4):
					np.random.shuffle(values)
					column.append(values.copy())

				# Flatten the list of arrays
				column = np.concatenate(column)
				V_s[rxn] = column
			
			
			#for rxn in V_s:
			#	print(len(V_s[rxn]))
			
			#	print(len(V_s[rxn][0]))
			#	raise AssertionError("Ho stop!")
			#num_shuffles = int(self.sim * 0.8)
			# Initialize the result dictionary
			#V_s = {}

			# Create a pool of workers
			#pool = mp.Pool(mp.cpu_count()-int(0.1*mp.cpu_count()))#Leaving 10% of CPU power
			#results = []

			# Use apply_async to distribute the work
			#for rxn in tqdm(self.unsrt, desc="Doing random shuffling"):
			#	filename = mmap_dict[rxn]
				#shm, np_array = shm_dict[rxn]
			#	result = pool.apply_async(self.shuffle_values, args=(rxn, filename, V_[rxn].shape, V_[rxn].dtype, num_shuffles))
			#	results.append(result)

			# Close the pool and wait for the work to finish
			#pool.close()
			#pool.join()

			# Collect the results
			#for result in results:
			#	rxn, column = result.get()
			#	V_s[rxn] = column
			
			
			# Cleanup shared memory
			#self.cleanup_shared_memory(shm_dict)
			
			# Cleanup memory-mapped files
			#for filename in mmap_dict.values():
			#	os.remove(filename)
			#os.rmdir(temp_dir)
			
			
			"""
			V_linear_comb = {}#Doing linear combination to populate the matrix
			for rxn in self.unsrt:
				temp = []
				for i in range(1000):
					zeta_a = np.array(a_curves_dict[rxn][np.random.randint(0,10)])
					zeta_b = np.array(b_curves_dict[rxn][np.random.randint(0,10)])
					zeta_c = np.array(c_curves_dict[rxn][np.random.randint(0,10)])
					x,y,z = self.generatePointOnSphere()
					####
					temp.append(x*zeta_a+y*zeta_b+z*zeta_c)
				
				V_linear_comb[rxn] = np.asarray(temp)
				
			"""
									
			delta_n = {}
			p = {}
			V_opt = {}
			V = {}#Populating V for unshuffled portion of the design matrix
			ch = {}
			nominal = {}
			p_max = {}
			p_min = {}
			theta = {}
			Temp ={}
			zeta = {}
			for rxn in tqdm(self.unsrt,desc="Populating unshuffled portion of DM"):
				ch[rxn] = self.unsrt[rxn].cholskyDeCorrelateMat
				nominal[rxn] = self.unsrt[rxn].nominal
				p_max[rxn] = self.unsrt[rxn].P_max
				p_min[rxn] = self.unsrt[rxn].P_min
				theta[rxn] = self.unsrt[rxn].Theta
				Temp[rxn] = self.unsrt[rxn].temperatures
				zeta[rxn] = self.unsrt[rxn].zeta.x
		
			#fast SAC_3
			
			#plot the zetas
			#print(percentage)
			#Solving for a line passing from 3 points
			d_n = {}
			_delta_n = {}
			dict_delta_n = {}
			percentage = {}
			for rxn in tqdm(self.unsrt,desc="Generating fSAC samples"):
				T =np.array([Temp[rxn][0],(Temp[rxn][0] + Temp[rxn][-1])/2,Temp[rxn][-1]])	
				Theta = np.array([T/T,np.log(T),-1/(T)])
				P = nominal[rxn]
				cov = ch[rxn]
				zet = zeta[rxn]
				Tp = Temp[rxn]
				#Tp = np.linspace(300,2500,50)
				Theta_p = np.array([Tp/Tp,np.log(Tp),-1/(Tp)])
				P_max = P + np.asarray(np.dot(cov,zet)).flatten();
				P_min = P - np.asarray(np.dot(cov,zet)).flatten();
				
				d_n[rxn] = abs(P[1]-P_max[1])
				kmax = Theta_p.T.dot(P_max)
				kmin = Theta_p.T.dot(P_min)
				ka_o = Theta_p.T.dot(P)
				M = Theta.T.dot(cov)
				#fig = plt.figure()
				#plt.plot(1/Tp,kmax)
				#plt.plot(1/Tp,kmin)
				#plt.plot(1/Tp,ka_o)
				f = abs(kmax-ka_o)
				temp = []
				outside = []
				not_selected = []
				temp_n = []
				while len(temp)<1000:
					random = 2*np.random.rand(3)-1
					P_right = P + random[0]*np.asarray(np.dot(cov,zet)).flatten()
					P_mid = P + random[1]*(7/8)*np.asarray(np.dot(cov,zet)).flatten()
					P_left = P + random[2]*np.asarray(np.dot(cov,zet)).flatten()
					Theta_left = np.array([T[0]/T[0],np.log(T[0]),-1/(T[0])])
					Theta_mid = np.array([T[1]/T[1],np.log(T[1]),-1/(T[1])])
					Theta_right = np.array([T[2]/T[2],np.log(T[2]),-1/(T[2])])
					kappa_left = Theta_left.T.dot(P_left)
					kappa_mid = Theta_mid.T.dot(P_mid)
					kappa_right = Theta_right.T.dot(P_right)
					kappa_o1 = Theta_left.T.dot(P)
					kappa_o2= Theta_mid.T.dot(P)
					kappa_o3 = Theta_right.T.dot(P)
					kappa = np.array([kappa_left,kappa_mid,kappa_right])
					kappa_o = np.array([kappa_o1,kappa_o2,kappa_o3])
					y = kappa - kappa_o
					zeta_ = np.linalg.solve(M,y)
					func = np.asarray([(i.T.dot(cov.dot(zeta_))) for i in Theta_p.T])
					P_found = P + np.asarray(np.dot(cov,zeta_)).flatten()
					n_ = abs(P[1]-P_found[1])
					temp_n.append(n_)
					kappa_found = Theta_p.T.dot(P_found)
					f_found = abs(kappa_found-ka_o)
					if max(f)<max(f_found):
						outside.append(zeta_)
					if n_ > 2:	
						#plt.plot(1/Tp,kappa_found,"r-",linewidth=0.25)
						#temp.append(zeta_)#for unconstrained zeta
						not_selected.append(zeta_)
						_delta_n[n_] = zeta_
						
					else:
						temp.append(zeta_)
						_delta_n[n_] = zeta_
					#print(zeta_)
				percentage[rxn] = (len(not_selected)/len(temp))*100
				delta_n[rxn] = max(temp_n)
				
				dict_delta_n[rxn] = _delta_n[max(temp_n)]
				V[rxn] = temp
					
			
					
			# Use the ACTUAL number of rows produced (n_a+n_b+n_c per
			# reaction, and 4x that for the shuffled block) rather than a
			# fixed int(self.sim*0.2)/int(self.sim*0.8) -- otherwise Sobol's
			# power-of-2 inflation would be silently truncated back down
			# here, defeating the point of growing the batch in the first
			# place. For method='numpy'/'lhs' these counts equal the
			# original int(self.sim*0.2)/int(self.sim*0.8) exactly, so
			# behaviour there is unchanged.
			unshuffled_actual = len(next(iter(V_.values())))
			shuffled_actual = len(next(iter(V_s.values())))

			for i in range(unshuffled_actual):
				row = []
				for rxn in self.unsrt:
					row.extend(V_[rxn][i])
					
				design_matrix.append(row)
			
			
			#RANDOM SHUFFLING
			for i in range(shuffled_actual):
				row = []
				for rxn in self.unsrt:
					row.extend(V_s[rxn][i])
				design_matrix.append(row)
				
			
			for i in range(1000):
				row = []
				for rxn in self.unsrt:
					row.extend(V[rxn][i])
					
				design_matrix.append(row)
			tok = time.time()
			print(f"\n[{method}] total design-matrix rows: {len(design_matrix)} "
				  f"(unshuffled={unshuffled_actual}, shuffled={shuffled_actual}, fsac=1000)")
			print("Time taken to construct Design Matrix: {}".format(tok - tic))
			s =""
			for row in design_matrix:
				for element in row:
					s+=f"{element},"
				s+="\n"
			ff = open('DesignMatrix.csv','w').write(s)	
			
			return np.asarray(design_matrix)
	
#########################################
## Definitions to run the generation   ##
## of samples using parallel computing ##
#########################################		
	
def run_sampling_b(sample,data,generator,length):
	A = Uncertainty.UncertaintyExtractor(data)
	a1 = generator[0]
	a2 = generator[1]
	A.populateValues(a1,a2)
	A.getCovariance(flag=False)
	A.getUnCorrelated(flag = False)
	zeta = A.getB2Zeta(flag=True)
	del A
	return (sample,generator,zeta,length)
	
def run_sampling_c(sample,data,generator,length):
	A = Uncertainty.UncertaintyExtractor(data)
	a1 = generator[0]
	a2 = generator[1]
	A.populateValues(a1,a2)
	A.getCovariance(flag=False)
	A.getUnCorrelated(flag = False)
	zeta = A.getC2Zeta(flag=True)
	del A
	return (sample,generator,zeta,length)
	
class Worker():
	
	def __init__(self, workers):
		self.pool = multiprocessing.Pool(processes=workers)
		self.progress = []
		self.parallized_zeta = []
		self.generator = []
		self.parallel_zeta_dict = {}
		self.diagnostics_list = []

		self.sample_pbar = None
		self.error_count = 0

	def callback(self, result):

		self.progress.append(result[0])
		self.generator.append(result[1])
		self.parallized_zeta.append(result[2])

		if self.sample_pbar is not None:
			self.sample_pbar.update(1)

	def callback_partial(self, result):
		"""Same as callback(), but also collects the 5th element -- the
		per-task QuasiRandomGenerator diagnostics dict (or None for the
		class-B m==3 branch, which uses no adapter) -- returned by
		run_sampling_b_partial / run_sampling_c_partial."""
		self.progress.append(result[0])
		self.generator.append(result[1])
		self.parallized_zeta.append(result[2])
		if len(result) > 4 and result[4] is not None:
			self.diagnostics_list.append(result[4])

		if self.sample_pbar is not None:
			self.sample_pbar.update(1)

	def callback_error(self, result):
		"""Runs (in the main process) whenever a worker task raises. A bare
		print() here used to land in the middle of whatever tqdm bar is
		currently drawn (the outer per-reaction bar and/or self.sample_pbar),
		tearing up the live display -- tqdm.write() instead clears every
		active bar, writes the line, then redraws them, so the signal shows
		up cleanly above the bars instead of corrupting them. Full details
		(exception + best-effort traceback) go to worker_errors.log so
		nothing is lost even though the console line is kept short; the
		console only ever prints a one-line-per-error counter.
		"""
		self.error_count += 1
		exc_line = f"{type(result).__name__}: {result}"
		tqdm.write(f"    [worker error #{self.error_count}] {exc_line}")

		tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
		with open("worker_errors.log", "a") as fh:
			fh.write(f"--- worker error #{self.error_count} ---\n{tb}\n")

		# Errored tasks never reach callback()/callback_partial(), so without
		# this the bar would stall short of `total` even though pool.join()
		# doesn't actually wait on it -- keep the displayed count honest.
		if self.sample_pbar is not None:
			self.sample_pbar.update(1)

	def do_unsrt(self,data,sampling_points):
		#generators = data["generators"]
		for args in range(sampling_points):
			self.pool.apply_async(run_sampling, 
				  args=(1,data,data["generators"][args],sampling_points,), 
				  callback=self.callback)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()
		return self.generator,self.parallized_zeta
	def do_unsrt_b(self,data,sampling_points):
		#generators = data["generators"]
		for args in range(sampling_points):
			self.pool.apply_async(run_sampling_b, 
				  args=(1,data,data["generators_b"][args],sampling_points,), 
				  callback=self.callback,error_callback=self.callback_error)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()
		return self.parallized_zeta,self.generator
	def do_unsrt_c(self,data,sampling_points):
		#generators = data["generators"]
		for args in range(sampling_points):
			self.pool.apply_async(run_sampling_c, 
				  args=(1,data,data["generators_c"][args],sampling_points,), 
				  callback=self.callback,error_callback=self.callback_error)
		self.pool.close()
		self.pool.join()
		self.pool.terminate()
		return self.parallized_zeta,self.generator
	
	def do_unsrt_b_partial(self, data, sampling_points):
		"""
		Parallel class-B dispatch for partial-parameter sampling.

		Routes every task to Uncertainty.run_sampling_b_partial, which
		instantiates a fresh UncertaintyExtractor and calls the analytical
		instance methods (_psac_class_B_m2 / _psac_class_B_m3) — no SLSQP.

		Expects data to contain:
		  data["param_indices"]        – active parameter index tuple
		  data["generators_b_partial"] – joint shadow-block slice for this
		                                  reaction, shape (sampling_points,4):
		                                  columns [a1, a2, r1, r2]
		  data["fallback_seed_b_base"] – per-reaction fallback seed (or None)

		Returns (parallized_zeta, generator, diagnostics_list).
		"""
		self.sample_pbar = tqdm(
		total=sampling_points,
			desc="    Samples",
			leave=False,
			dynamic_ncols=True
			)
		for args in range(sampling_points):
			self.pool.apply_async(
				Uncertainty.run_sampling_b_partial,
				args=(
					1,
					data,
					data["generators_b_partial"][args],
					sampling_points,
					),
				callback=self.callback_partial,
				error_callback=self.callback_error
			)
		self.pool.close()
		self.pool.join()
		self.sample_pbar.close()
		self.pool.terminate()
		return self.parallized_zeta, self.generator, self.diagnostics_list
	

	def do_unsrt_c_partial(self, data, sampling_points):			
		"""
		Parallel class-C dispatch for partial-parameter sampling.

		Routes every task to Uncertainty.run_sampling_c_partial.

		Expects data to contain:
		  data["param_indices"]        – active parameter index tuple
		  data["generators_c_partial"] – joint shadow-block slice for this
		                                  reaction, shape (sampling_points,2):
		                                  columns [r1, r2]
		  data["fallback_seed_c_base"] – per-reaction fallback seed (or None)

		Returns (parallized_zeta, generator, diagnostics_list).
		"""

		self.sample_pbar = tqdm(
			total=sampling_points,
			desc="    Samples",
			leave=False,
			dynamic_ncols=True
			)
		for args in range(sampling_points):
			self.pool.apply_async(
				Uncertainty.run_sampling_c_partial,
					args=(
					1,
					data,
					data["generators_c_partial"][args],
					sampling_points,
					),
				callback=self.callback_partial,
				error_callback=self.callback_error
			)
		self.pool.close()
		self.pool.join()
		self.sample_pbar.close()
		self.pool.terminate()
		return self.parallized_zeta, self.generator, self.diagnostics_list

