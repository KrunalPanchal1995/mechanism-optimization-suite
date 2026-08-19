import os
import numpy as np
from copy import deepcopy
from MechanismParser import Parser

###############################################################################
# MechManipulator2_0.py  –  Upgraded Arrhenius Manipulator
#
# Supported design modes (detected automatically from unsrt + select_vector):
#
#   ┌──────────────────┬───────────┬───────────────────────────────────────┐
#   │ Mode			 │ n_params  │ Perturbation formula				  │
#   ├──────────────────┼───────────┼───────────────────────────────────────┤
#   │ Full A-factor	│	 1	 │ p[0] = p0[0] + σ_A · ζ_A			 │
#   │ Full A1+B1+C1	│	 3	 │ p	= p0   + L · ζ		(3×3 L)   │
#   │ Partial A-factor │	 1	 │ p[0] = p0[0] + σ_A · ζ_A · sel[0]   │
#   │ Partial A1+B1+C1 │	 3	 │ p_S  = p0_S + L_r · ζ_r   (m×m L_r) │
#   └──────────────────┴───────────┴───────────────────────────────────────┘
#
# For Partial A1+B1+C1, L_r is the Cholesky factor of the principal submatrix
# Σ_r = Σ[I,I] where I = active_indices.  This is mathematically correct;
# applying the full L to a sparse ζ (zeros at inactive positions) is NOT
# equivalent due to cross-correlation terms.
###############################################################################


class Manipulator:
	def __init__(self, copy_of_mech, unsrt_object, perturbation,
				 selection=np.array([])):
		"""
		Parameters
		----------
		copy_of_mech  : dict   Parsed YAML mechanism (deepcopy is taken internally).
		unsrt_object  : dict   {rxn_key → UncertaintyExtractor subclass instance}.
		perturbation  : list   Flat design-matrix row (full-space, zeros at inactive
							   positions for partial PRS).
		selection	 : array  Flat selection row from the design matrix
							   (1 = active, 0 = inactive).  If empty, defaults to
							   all-ones (full PRS / backward-compatible behaviour).
		"""
		self.mechanism  = deepcopy(copy_of_mech)
		self.unsrt	  = unsrt_object
		self.perturbation = perturbation
		self.rxn_list   = [rxn for rxn in self.unsrt]

		if len(selection) != 0:
			self.Arrhenius_Params_Selection = "some"
			self.selection = selection
		else:
			self.Arrhenius_Params_Selection = "some"
			self.selection = np.ones(len(perturbation))

		# Initialised here; populated by getRxnPerturbationDict()
		self.select_dict = {}

	# ═════════════════════════════════════════════════════════════════════════
	# Helpers
	# ═════════════════════════════════════════════════════════════════════════

	def getRxnPerturbationDict(self):
		"""
		Split the perturbation and selection vectors into per-reaction dicts.

		Handles two formats, auto-detected by len(perturbation):

		Full-space format  (full PRS / A1+B1+C1, original behaviour)
		─────────────────────────────────────────────────────────────
		  len(perturbation) == sum(activeParameters per rxn)  (e.g. 3*N_rxns)
		  perturbation and selection are consumed sequentially,
		  n = len(activeParameters) elements per reaction.

		Compressed-diagonal format  (SA mode, one value per reaction)
		──────────────────────────────────────────────────────────────
		  len(perturbation) == N_rxns  (one scalar zeta per reaction, indexed
		  directly by position j — this stride is always 1 and never varies).

		  selection does NOT have a fixed stride of 3 per reaction. Its
		  per-reaction width is n = len(self.unsrt[rxn].activeParameters):
		  1 for an "A-factor only" reaction, 3 for an "A1+B1+C1" reaction.
		  A reaction not selected for perturbation still occupies its n-width
		  slot in selection — it's just all-zero. Reading a hardcoded 3
		  elements per reaction (then trimming to n) silently misaligns the
		  cursor as soon as an A-factor-only reaction is encountered, and the
		  reactions after it would read someone else's flags. So selection
		  must be walked with a running cursor that advances by n, not 3*j.

		  For reaction j, with sel_n = selection[sel_count : sel_count+n]:
			n=1: [1] → A-factor active        [0] → inactive
			n=3: [1,0,0] → A-factor   [0,1,0] → n   [0,0,1] → Ea/R
				 [0,0,0] → inactive  → beta = zeros → _compute returns nominal
		  perturbation[j] is the single zeta scalar placed at whichever
		  position within sel_n is nonzero.

		Returns
		-------
		perturb	 : dict  rxn → np.ndarray  zeta vector (length n_params)
		select_dict : dict  rxn → np.ndarray  selection flags (length n_params)
		"""
		perturb	 = {}
		select_dict = {}
		N_rxns = len(self.rxn_list)

		if len(self.perturbation) == N_rxns:
			# ── Compressed-diagonal format ─────────────────────────────────────
			sel_count = 0   # cursor into self.selection (advances by n, NOT 3*j)

			for j, rxn in enumerate(self.rxn_list):
				n = len(self.unsrt[rxn].activeParameters)   # 1 or 3

				# Read exactly n selection flags for this reaction — its own
				# width, whether or not it's one of the selected/perturbed ones.
				sel_n = np.asarray(
					self.selection[sel_count : sel_count + n], dtype=float)
				#print(f"sel_n:{sel_n}|Rxn:{rxn}\n")
				m = int(np.sum(sel_n != 0.0))   # number of active params (0 or 1 for SA)
				if m == 0:
					# Inactive: zero beta + zero selection → _compute returns nominal
					perturb[rxn]	 = np.zeros(n)
					select_dict[rxn] = np.zeros(n)
				else:
					# Active: place the single diagonal zeta at the selected position
					beta_val  = float(self.perturbation[j])
					beta_full = np.zeros(n)
					for k in range(n):
						if sel_n[k] != 0.0:
							beta_full[k] = beta_val
					perturb[rxn]	 = beta_full
					select_dict[rxn] = sel_n

				sel_count += n   # advance by THIS reaction's own width

			# ── Sanity check ──────────────────────────────────────────────────
			if sel_count != len(self.selection):
				raise ValueError(
					f"getRxnPerturbationDict: consumed {sel_count} selection "
					f"entries but self.selection has {len(self.selection)}. "
					f"Check that self.selection's per-reaction layout matches "
					f"activeParameters (1 for A-factor-only, 3 for A1+B1+C1)."
				)

		else:
			# ── Full-space format ──────────────────────────────────────────────
			#
			# Two vectors with DIFFERENT, *reaction-dependent* strides must use
			# SEPARATE cursors:
			#
			#   self.selection  (pSelectionMatrix row)
			#	   Stride = n = len(self.unsrt[rxn].activeParameters) per
			#	   reaction — i.e. 1 for an "A-factor only" reaction, 3 for an
			#	   "A1+B1+C1" reaction (same n_params used by the
			#	   compressed-diagonal branch above and by
			#	   _compute_perturbed_params). sel_count must advance by n,
			#	   NOT by a hardcoded 3: a rxn_list that mixes both kinds (as
			#	   happens under p-SAC partial sampling) makes a fixed stride
			#	   of 3 read past the end of self.selection the moment an
			#	   n=1 reaction is hit, producing an IndexError.
			#
			#   self.perturbation  (pDesignMatrix row)
			#	   Stride = n_active per reaction (variable: 0..n).
			#	   dm_count advances by the number of 1s found within THIS
			#	   reaction's n-length selection slice (sel_n) — not by n
			#	   itself.
			#
			# beta_full / sel_n are built at length n (1 or 3), matching the
			# compressed-diagonal branch and what _compute_perturbed_params
			# expects (it trims select_vector to n_params anyway, so a
			# length-n vector is sufficient — and for n_params == 3 it's
			# still a full 3-element vector, so Mode-4 partial-A1+B1+C1
			# indexing via active_indices ⊆ {0,1,2} keeps working exactly
			# as before).
			dm_count  = 0   # cursor into self.perturbation (advances by n_active)
			sel_count = 0   # cursor into self.selection	(advances by n)

			for rxn in self.rxn_list:
				n = len(self.unsrt[rxn].activeParameters)   # 1 or 3

				# Read exactly n selection flags for this reaction
				sel_n	 = np.asarray(
					self.selection[sel_count : sel_count + n], dtype=float)
				n_active = int(np.sum(sel_n != 0.0))
				#print(f"Sel_n:{sel_n}|Rxn:{rxn}\n")

				# Extract the n_active DM values and place them at their
				# local parameter positions (0=lnA, 1=n, 2=Ea/R, truncated
				# to n) in a length-n beta vector.
				dm_vals   = np.asarray(
					self.perturbation[dm_count : dm_count + n_active],
					dtype=float)
				beta_full = np.zeros(n)
				k = 0
				for param_i in range(n):
					if sel_n[param_i] != 0.0:
						beta_full[param_i] = dm_vals[k]
						k += 1

				perturb[rxn]	 = beta_full
				select_dict[rxn] = sel_n

				sel_count += n		  # pSelectionMatrix: stride = n
				dm_count  += n_active   # pDesignMatrix:	variable stride

			# ── Sanity check ──────────────────────────────────────────────────
			# Catches any remaining rxn_list/selection/perturbation length
			# mismatch immediately, with a clear message, instead of a deep,
			# confusing IndexError mid-loop.
			if sel_count != len(self.selection):
				raise ValueError(
					f"getRxnPerturbationDict: consumed {sel_count} selection "
					f"entries but self.selection has {len(self.selection)}. "
					f"Check that self.selection's per-reaction layout matches "
					f"activeParameters (1 for A-factor-only, 3 for A1+B1+C1)."
				)
			if dm_count != len(self.perturbation):
				raise ValueError(
					f"getRxnPerturbationDict: consumed {dm_count} perturbation "
					f"entries but self.perturbation has {len(self.perturbation)}."
				)

		return perturb, select_dict

	def del_mech(self):
		del self.mechanism

	def getRxnType(self):
		return {rxn: self.unsrt[rxn].classification for rxn in self.rxn_list}

	# ═════════════════════════════════════════════════════════════════════════
	# Core perturbation kernel  (all four modes handled here)
	# ═════════════════════════════════════════════════════════════════════════

	def _compute_perturbed_params(self, rxn, beta, select_vector=None):
		"""
		Compute the perturbed Arrhenius parameter vector p = [ln A, n, Ea/R].

		Mode is inferred automatically:

		  n_params = 1  →  A-factor only (factor perturbation type)
		  n_params = 3
			len(active_indices) = 3  →  Full A1+B1+C1
			len(active_indices) < 3  →  Partial A1+B1+C1  (reduced L_r)
			len(active_indices) = 0  →  No perturbation   (return nominal)

		Parameters
		----------
		rxn		   : str   Reaction key in self.unsrt.
		beta		  : array Per-reaction zeta vector from the design matrix.
							  Full-space (length = n_active_params of reaction);
							  zeros occupy inactive positions for partial PRS.
		select_vector : array 1/0 flags, one per active parameter of this reaction.
							  Derived from the design-matrix selection row.
							  If None, falls back to self.unsrt[rxn].selection
							  (backward-compatible: uses the reaction's own
							  perturbation-type selection, not partial PRS).

		Returns
		-------
		p : np.ndarray, shape (3,)
			Perturbed [ln A, n, Ea/R].  Unperturbed parameters keep nominal values.

		Notes
		-----
		Partial A1+B1+C1 uses UncertaintyExtractor.get_reduced_cholesky(indices),
		which computes  Σ_r = Σ[I,I]  (principal submatrix)  then  L_r = chol(Σ_r).
		This is the only mathematically correct path; L_full · beta_sparse is wrong.

		Bug-fixes applied (vs original)
		--------------------------------
		FIX-1  p0 shape:
			np.asarray(self.unsrt[rxn].nominal) can produce a 2-D array (shape
			(3,1)) when `nominal` stores each value as a length-1 ndarray.
			.ravel() guarantees p0 is always 1-D shape (3,).

		FIX-2  beta_arr shape:
			`beta` is often a 2-D slice of the design matrix  (shape (1,3) or
			(3,1)).  np.asarray alone preserves that shape, so indexing
			beta_arr[list(active_indices)] selects ROWS not ELEMENTS, producing
			a 2-D zeta_r and therefore a 2-D delta_r.  delta_r[local_i] then
			returns a length-3 array, and the assignment
				p[global_i] = p0[global_i] + delta_r[local_i]
			raises the chained TypeError → ValueError seen in the traceback.
			.ravel() collapses beta_arr to 1-D before any indexing.

		FIX-3  delta_r shape:
			Even after fixing beta_arr, L_r @ zeta_r can return a 2-D array
			when L_r is exactly (1,1).  .ravel() ensures delta_r is always 1-D.
			The explicit float() wrappers in the assignment loop give one more
			guard against residual array elements reaching p.

		FIX-4  sigma_A extraction (n_params == 1):
			If cholskyDeCorrelateMat is a full 3×3 matrix (not trimmed to the
			active-parameter count), cov[0] returns the first ROW (length 3),
			and float(cov[0]) raises
				TypeError: only size-1 arrays can be converted to Python scalars
			which NumPy re-raises as the ValueError seen first in the traceback.
			Using np.asarray(cov).ravel()[0] always extracts the scalar
			regardless of whether cov is a scalar, 1-D, or 2-D array.

		FIX-5  active_indices tolerance:
			The original check `float(sel[i]) != 0.0` triggers on floating-point
			residues (e.g. 1e-15 from upstream arithmetic) that should be treated
			as "inactive".  Replacing it with `abs(float(sel[i])) > 1e-10` makes
			the boundary robust.  This was causing reactions that should be
			unperturbed (active_indices should be ()) to instead enter the
			partial-Cholesky branch and hit FIX-2/FIX-3.
		"""
		# ── FIX-1: ravel() guarantees 1-D regardless of how nominal is stored ──
		p0	   = np.asarray(self.unsrt[rxn].nominal, dtype=float).ravel()
		cov	  = self.unsrt[rxn].cholskyDeCorrelateMat
		n_params = len(self.unsrt[rxn].activeParameters)

		# Resolve selection vector ────────────────────────────────────────────
		if select_vector is None:
			sel = np.asarray(self.unsrt[rxn].selection, dtype=float).ravel()
			sel = sel[:n_params]
		else:
			sel = np.asarray(select_vector, dtype=float).ravel()[:n_params]

		# ── FIX-2: ravel() collapses 2-D DM slices to 1-D ──────────────────
		beta_arr = np.asarray(beta, dtype=float).ravel()
		#print(beta_arr)
		# ── Mode 1: A-factor only  (n_params = 1) ───────────────────────────
		if n_params == 1:
			# ── FIX-4: ravel()[0] handles scalar, 1-D, or 2-D cov ──────────
			sigma_A = float(np.asarray(cov, dtype=float).ravel()[0])
			active  = float(sel[0]) if len(sel) > 0 else 1.0
			p	   = p0.copy()
			p[0]   += sigma_A * float(beta_arr[0]) * active
			#print(f"Rxn: {rxn}  Active parameters: 1  selected params:{select_vector}"
			#	  f"  Nominal:{p0}  Perturbed: {p}\n")
			return p

		# ── Three-parameter cases  (n_params = 3) ────────────────────────────
		# ── FIX-5: tolerance-based zero check ───────────────────────────────
		active_indices = tuple(
			i for i in range(3) if i < len(sel) and abs(float(sel[i])) > 1e-10
		)

		if not active_indices:
			# Nothing selected — return nominal unchanged.
			#print(f"Rxn: {rxn}  Active parameters: 0  selected params:{select_vector}"
			#	  f"  Nominal:{p0}  Perturbed: {p0}  [no perturbation]\n")
			return p0.copy()

		if len(active_indices) == 3:
			# ── Mode 2: Full A1+B1+C1 ────────────────────────────────────────
			# p = p0 + L · ζ   (full 3×3 Cholesky)
			p = p0 + np.asarray(cov, dtype=float).dot(beta_arr).ravel()

		else:
			# ── Mode 4: Partial A1+B1+C1 ─────────────────────────────────────
			# Build reduced Cholesky from the principal submatrix of Σ = L L^T.
			#   Σ_r = Σ[active_indices, active_indices],  L_r = chol(Σ_r)
			#   ζ_r = beta_arr[active_indices]   (m active zeta values)
			#   Δp_r = L_r · ζ_r				(m-vector)
			#   p[active_indices] += Δp_r
			_,(_, L_r)  = self.unsrt[rxn].get_reduced_cholesky(active_indices)
			zeta_r  = beta_arr[list(active_indices)]			 # 1-D, length m
			# ── FIX-3: ravel() so delta_r[local_i] is always a scalar ───────
			delta_r = np.asarray(L_r, dtype=float).dot(zeta_r).ravel()
			p	   = p0.copy()
			for local_i, global_i in enumerate(active_indices):
				# explicit float() is the final guard against residual array dims
				p[global_i] = float(p0[global_i]) + float(delta_r[local_i])

		#print(f"Rxn: {rxn}  Active parameters: {len(active_indices)}"
		#	  f"  selected params:{select_vector}"
		#	  f"  Nominal:{p0}  Perturbed: {p}\n")
		return p

	def _write_rate_constant(self, p, reaction_details):
		"""
		Write the perturbed Arrhenius parameters into a rate-constant dict.

		Parameters
		----------
		p				: array  [ln A, n, Ea/R]
		reaction_details : dict   YAML rate-constant entry (mutated in place).
		"""
		reaction_details["A"]  = float(np.exp(p[0]))
		reaction_details["b"]  = float(p[1])
		reaction_details["Ea"] = float(p[2] * 1.987)

	# ═════════════════════════════════════════════════════════════════════════
	# Reaction-type perturbation methods
	# ═════════════════════════════════════════════════════════════════════════

	def ElementaryPerturbation(self, rxn, beta, mechanism, select_vector=None):
		"""
		Perturb an elementary, duplicate, or third-body reaction.

		Calls _compute_perturbed_params which transparently handles all four
		design modes.  select_vector is the per-reaction row from the design
		matrix selection (1 = active, 0 = inactive).  When omitted the
		reaction's own .selection attribute is used (backward compatible).
		"""
		p	 = self._compute_perturbed_params(rxn, beta, select_vector = select_vector )
		index = self.unsrt[rxn].index

		reaction_details = mechanism["reactions"][index]["rate-constant"]
		self._write_rate_constant(p, reaction_details)
		mechanism["reactions"][index]["rate-constant"] = deepcopy(reaction_details)
		return mechanism

	def PlogPerturbation(self, rxn, beta, mechanism, select_vector=None):
		"""
		Perturb a PLOG or PLOG-Duplicate pressure-dependent reaction.

		The pressure level (High / Low / interpolated) is determined from
		self.unsrt[rxn].pressure_limit as in the original code.
		"""
		p = self._compute_perturbed_params(rxn, beta, select_vector = select_vector)

		# Determine PLOG pressure index ───────────────────────────────────────
		rxn_split_index = None
		if "PLOG" in rxn:
			try:
				rxn_split_index = int(rxn.split(":")[1].split("_")[1])
			except (IndexError, ValueError):
				rxn_split_index = 0

		index		  = self.unsrt[rxn].index
		pressure_limit = self.unsrt[rxn].pressure_limit

		if pressure_limit == "High":
			pressure_index = -1
		elif pressure_limit == "Low":
			pressure_index = 0
		else:
			pressure_index = rxn_split_index if rxn_split_index is not None else 0

		reaction_details = mechanism["reactions"][index]["rate-constants"][pressure_index]
		self._write_rate_constant(p, reaction_details)
		mechanism["reactions"][index]["rate-constants"][pressure_index] = deepcopy(
			reaction_details)
		return mechanism

	def BranchingReactions(self, rxn, beta, mechanism, select_vector=None):
		"""
		Perturb all branches of a branching reaction with the same zeta.

		The perturbation delta [Δ(ln A), Δn, Δ(Ea/R)] is computed from the
		base reaction's nominal.  The same delta is applied to each branch's
		own nominal so that all branches receive an identical relative shift
		regardless of their individual nominal values.

		This correctly supports all four design modes because the delta is
		computed by _compute_perturbed_params (which handles Full / Partial
		A-factor and Full / Partial A1+B1+C1).
		"""
		p0_base = np.asarray(self.unsrt[rxn].nominal, dtype=float)
		p_base  = self._compute_perturbed_params(rxn, beta, select_vector = select_vector)
		delta   = p_base - p0_base		# [Δ(lnA), Δn, Δ(Ea/R)]

		# Collect base reaction index + all branch indices ─────────────────────
		indexes = [self.unsrt[rxn].index]
		indexes.extend(self.unsrt[rxn].branches)

		for index in indexes:
			reaction_details = mechanism["reactions"][index]["rate-constant"]
			# Each branch has its own nominal; apply the shared delta to it.
			p0_branch = np.array([
				float(np.log(reaction_details["A"])),
				float(reaction_details["b"]),
				float(reaction_details["Ea"] / 1.987),
			])
			p_branch = p0_branch + delta
			self._write_rate_constant(p_branch, reaction_details)
			mechanism["reactions"][index]["rate-constant"] = deepcopy(reaction_details)

		return mechanism

	def ThirdBodyPerturbation(self, rxn, beta, mechanism, select_vector=None):
		"""Third-body efficiency perturbation (not yet implemented)."""
		pass

	def TroePerturbation(self, rxn, beta, mechanism, select_vector=None):
		"""
		Perturb a Troe / falloff reaction.

		The high-pressure or low-pressure Arrhenius block is selected via
		self.unsrt[rxn].pressure_limit, matching the original behaviour.
		All four design modes are supported through _compute_perturbed_params.
		"""
		p	   = self._compute_perturbed_params(rxn, beta, select_vector = select_vector)
		P_limit = self.unsrt[rxn].pressure_limit
		index   = self.unsrt[rxn].index

		if P_limit == "High":
			reaction_details = mechanism["reactions"][index]["high-P-rate-constant"]
			self._write_rate_constant(p, reaction_details)
			mechanism["reactions"][index]["high-P-rate-constant"] = deepcopy(
				reaction_details)
		else:
			reaction_details = mechanism["reactions"][index]["low-P-rate-constant"]
			self._write_rate_constant(p, reaction_details)
			mechanism["reactions"][index]["low-P-rate-constant"] = deepcopy(
				reaction_details)

		return mechanism

	# ═════════════════════════════════════════════════════════════════════════
	# Main driver
	# ═════════════════════════════════════════════════════════════════════════

	def doPerturbation(self):
		"""
		Apply Arrhenius perturbations to every reaction in the mechanism.

		Builds the per-reaction beta and select_vector from the flat design
		matrix row and selection row stored in self.perturbation / self.selection,
		then dispatches to the appropriate perturbation method based on reaction
		classification.

		Returns
		-------
		mechanism : dict   The perturbed YAML mechanism.
		perturb   : dict   Per-reaction zeta vectors {rxn → np.ndarray}.
		"""
		rxn_type			  = self.getRxnType()
		perturb, self.select_dict = self.getRxnPerturbationDict()
		mechanism			 = self.mechanism

		for rxn in self.rxn_list:
			beta		= np.asarray(perturb[rxn])
			sel		 = self.select_dict[rxn]   # per-reaction selection flags
			#print(f"Sel: {sel}| Rxn:{rxn}\n")
			type_of_rxn = rxn_type[rxn]

			if type_of_rxn == "Elementary":
				mechanism = self.ElementaryPerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "PLOG":
				mechanism = self.PlogPerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "PLOG-Duplicate":
				mechanism = self.PlogPerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "Duplicate":
				mechanism = self.ElementaryPerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "Falloff":
				mechanism = self.TroePerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "ThirdBody":
				mechanism = self.ElementaryPerturbation(rxn, beta, mechanism, sel)

			elif type_of_rxn == "BranchingRxn":
				mechanism = self.BranchingReactions(rxn, beta, mechanism, sel)

		return mechanism, perturb
