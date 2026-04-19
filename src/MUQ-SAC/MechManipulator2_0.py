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
#   │ Mode             │ n_params  │ Perturbation formula                  │
#   ├──────────────────┼───────────┼───────────────────────────────────────┤
#   │ Full A-factor    │     1     │ p[0] = p0[0] + σ_A · ζ_A             │
#   │ Full A1+B1+C1    │     3     │ p    = p0   + L · ζ        (3×3 L)   │
#   │ Partial A-factor │     1     │ p[0] = p0[0] + σ_A · ζ_A · sel[0]   │
#   │ Partial A1+B1+C1 │     3     │ p_S  = p0_S + L_r · ζ_r   (m×m L_r) │
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
		selection     : array  Flat selection row from the design matrix
		                       (1 = active, 0 = inactive).  If empty, defaults to
		                       all-ones (full PRS / backward-compatible behaviour).
		"""
		self.mechanism  = deepcopy(copy_of_mech)
		self.unsrt      = unsrt_object
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
		Split the flat perturbation and selection vectors into per-reaction dicts.

		Returns
		-------
		perturb     : dict  rxn → np.ndarray of zeta values for this reaction
		select_dict : dict  rxn → np.ndarray of selection flags (0 or 1)
		"""
		perturb     = {}
		select_dict = {}
		count       = 0
		for rxn in self.rxn_list:
			n = len(self.unsrt[rxn].activeParameters)
			perturb[rxn]     = np.asarray(self.perturbation[count: count + n])
			select_dict[rxn] = np.asarray(self.selection[count:   count + n],
			                              dtype=float)
			count += n
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
		rxn           : str   Reaction key in self.unsrt.
		beta          : array Per-reaction zeta vector from the design matrix.
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
		"""
		p0       = np.asarray(self.unsrt[rxn].nominal, dtype=float)
		cov      = self.unsrt[rxn].cholskyDeCorrelateMat
		n_params = len(self.unsrt[rxn].activeParameters)

		# Resolve selection vector ────────────────────────────────────────────
		if select_vector is None:
			# Backward-compatible default: use the reaction's own selection attr.
			# For factor-type: [1,0,0]; for all-param type: [1,1,1].
			sel = np.asarray(self.unsrt[rxn].selection, dtype=float)
			# Trim to n_params entries (factor type has [1,0,0] but n_params=1)
			sel = sel[:n_params]
		else:
			sel = np.asarray(select_vector, dtype=float)[:n_params]

		beta_arr = np.asarray(beta, dtype=float)

		# ── Mode 1 & 3: A-factor only  (n_params = 1) ───────────────────────
		# This covers both "Full A-factor" and "Partial A-factor" reactions.
		# cov is a 1-element array [σ_A] (the scalar Cholesky factor for ln A).
		if n_params == 1:
			sigma_A = float(cov[0]) if hasattr(cov, '__len__') else float(cov)
			active  = float(sel[0]) if len(sel) > 0 else 1.0
			p       = p0.copy()
			p[0]   += sigma_A * float(beta_arr[0]) * active
			return p

		# ── Three-parameter cases  (n_params = 3) ────────────────────────────
		# Determine which of the three Arrhenius parameters are active.
		# active_indices maps into p0-space (0=lnA, 1=n, 2=Ea/R).
		active_indices = tuple(
			i for i in range(3) if i < len(sel) and float(sel[i]) != 0.0
		)

		if not active_indices:
			# Nothing selected for this reaction — return nominal unchanged.
			return p0.copy()

		if len(active_indices) == 3:
			# ── Mode 2: Full A1+B1+C1 ─────────────────────────────────────
			# Use the full 3×3 L matrix (identical to original MUQ-SAC formula).
			#   p = p0 + L · ζ
			p = p0 + np.asarray(cov.dot(beta_arr)).flatten()

		else:
			# ── Mode 4: Partial A1+B1+C1 ─────────────────────────────────
			# Build reduced Cholesky from the principal submatrix of Σ = L L^T.
			# get_reduced_cholesky(indices) → (Σ_r, L_r)
			# where Σ_r = Σ[active_indices, active_indices], L_r = chol(Σ_r).
			#
			# Then:
			#   ζ_r = beta[active_indices]        (active components only)
			#   Δp_r = L_r · ζ_r                 (m-vector, m = |active_indices|)
			#   p[active_indices] += Δp_r
			_, L_r  = self.unsrt[rxn].get_reduced_cholesky(active_indices)
			zeta_r  = beta_arr[list(active_indices)]
			delta_r = L_r @ zeta_r
			p       = p0.copy()
			for local_i, global_i in enumerate(active_indices):
				p[global_i] = p0[global_i] + delta_r[local_i]

		return p

	def _write_rate_constant(self, p, reaction_details):
		"""
		Write the perturbed Arrhenius parameters into a rate-constant dict.

		Parameters
		----------
		p                : array  [ln A, n, Ea/R]
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
		p     = self._compute_perturbed_params(rxn, beta, select_vector)
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
		p = self._compute_perturbed_params(rxn, beta, select_vector)

		# Determine PLOG pressure index ───────────────────────────────────────
		rxn_split_index = None
		if "PLOG" in rxn:
			try:
				rxn_split_index = int(rxn.split(":")[1].split("_")[1])
			except (IndexError, ValueError):
				rxn_split_index = 0

		index          = self.unsrt[rxn].index
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
		p_base  = self._compute_perturbed_params(rxn, beta, select_vector)
		delta   = p_base - p0_base        # [Δ(lnA), Δn, Δ(Ea/R)]

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
		p       = self._compute_perturbed_params(rxn, beta, select_vector)
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
		rxn_type              = self.getRxnType()
		perturb, self.select_dict = self.getRxnPerturbationDict()
		mechanism             = self.mechanism

		for rxn in self.rxn_list:
			beta        = np.asarray(perturb[rxn])
			sel         = self.select_dict[rxn]   # per-reaction selection flags
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
