import os,sys
import numpy as np
import DesignMatrix as DM
import pickle
import simulation_manager2_0 as simulator
import json
import hashlib
import datetime
from Uncertainty import SAMPLER_VERSION

class PartialPRS(object):
	"""
	Partial Polynomial Response Surface class.

	Selects the active Arrhenius parameters for each reaction based on
	normalised sensitivity coefficients, builds a partial design matrix,
	and generates the perturbed YAML mechanism files required for
	combustion-target simulations.

	Supports two design modes
	--------------------------
	* ``"A-facto"``	  – single A-factor perturbation (sensitivity cut-off
						   plus PLOG-linked propagation, plus -- NEW --
						   correlation-aware rescue from `correlation_list`).
	* ``"A1+B1+C1"``	 – full three-parameter (A, n, Ea) perturbation using
						   the sparsity framework; correlation-aware rescue
						   wired in via ``correlation_list`` (same as A-facto).

	Parameters
	----------
	sensitivity_dict : dict
		For ``"A-facto"``: ``{rxn: sensitivity_value}``.
		For three-param:  ``{rxn: [s_A, s_n, s_Ea]}``.
	correlation_list : list
		Active-parameter strings (same naming convention as
		``self.active_params``, e.g. ``"12_A"``) identified by the
		cross-target SA correlation analysis as needing to be in the
		active set, even when their own sensitivity is below the cut-off.
		Only PARENT reactions appear here -- PLOG duplicates/linked
		reactions sharing a parent are represented once. ``A-facto`` only
		for now.
	unsrt_data : dict
		Reaction → uncertainty object (provides ``activeParameters``,
		``linked_rIndex``, ``cholskyDeCorrelateMat``).
	optInputs : dict
		Optimisation configuration; must contain ``Stats.cut_off_percentage``
		and ``Stats.sensitive_parameters``.
	target_list : list
		Combustion-target objects.
	case_index : int / str
		Index used to label output directories.
	active_parameters : list
		Ordered list of active parameter strings, length ``3 × N_reactions``
		for the three-param design (order: A, n, Ea per reaction).
	design : str
		``"A-facto"`` or ``"A1+B1+C1"``.
	status : str
		``"Pending"`` triggers a fresh selection run.
		Any other value attempts to reload state from previously saved CSV
		files (restart mode).
	"""
	def __init__(
		self,
		sensitivity_dict,
		correlation_list,
		unsrt_data,
		optInputs,
		target_list,
		case_index,
		active_parameters,
		design,
		status="Pending",
	):
		self.target_list  = target_list
		self.optInputs	= optInputs
		self.cut_off	  = float(optInputs["Stats"]["cut_off_percentage"])
		#self.sens_param   = str(optInputs["Stats"]["sensitive_parameters"])
		# NOTE: Arrhenius_Selection_Type removed — always using "some" framework.
		self.design	   = design
		self.case_index   = case_index
		self.unsrt		= unsrt_data
		self.correlation_list = correlation_list or []

		self.s_A  = []
		self.s_n  = []
		self.s_Ea = []
		self.no_of_sim = None
		self.selection_detail = []   # populated in __init__; written by partial_DesignMatrix

		# Build linked-reaction map keyed on each reaction's A-param string.
		# Used to propagate selection to PLOG-linked reactions.
		self.linked_list = {}
		self.check_list  = []
		self.active_params = []
  
		# ------------------------------------------------------------------ #
		# A-factor design – sensitivity cut-off + linked propagation +	   #
		# correlation-aware rescue (NEW)									   #
		# ------------------------------------------------------------------ #
		if self.design == "A-facto":
			for rxn in unsrt_data:
				self.linked_list[unsrt_data[rxn].activeParameters[0]] = (
					unsrt_data[rxn].linked_rIndex
				)
				self.active_params.append(unsrt_data[rxn].activeParameters[0])		
			sens_SA_dict = {}

			for rxn in unsrt_data:
				for rxn_ in sensitivity_dict:
					if rxn_ == rxn.split(":")[0]:
						sens_SA_dict[unsrt_data[rxn].activeParameters[0]] = float(
							sensitivity_dict[rxn_]
						)
						self.s_A.append(float(sensitivity_dict[rxn_]))

			self.partial_active	  = {}
			self.partial_active_list = []
			self.selected			= []
			self.coeff			   = []
			self.abs_coeff		   = []

			# Populate check_list with A-params above the absolute cut-off
			for ac in self.active_params:
				for sc in sens_SA_dict:
					if ac == sc:
						self.populateCheckList(sens_SA_dict[sc], ac)

			# Build abs_coeff with linked-reaction propagation
			for ap in self.active_params:
				for sc in sens_SA_dict:
					if ap == sc:
						self.coeff.append(sens_SA_dict[sc])
						self.abs_coeff.extend(
							self.getSelectedLinkedRxn(sens_SA_dict[sc], ap)
						)

			# ---------------------------------------------------------- #
			# Correlation-aware rescue: force-select any active_param   #
			# that appears in `correlation_list`, even if its own       #
			# sensitivity is below the cut-off.  abs_coeff is already   #
			# a binary 1.0/0.0 indicator at this point.                 #
			# ---------------------------------------------------------- #
			# Capture pre-rescue state for selection_detail tracking
			_pre_rescue_a = {
				ap for ind, ap in enumerate(self.active_params)
				if ind < len(self.abs_coeff) and self.abs_coeff[ind] >= 1.0
			}
			correlation_set = set(self.correlation_list)
			for ind, ap in enumerate(self.active_params):
				if self.abs_coeff[ind] < 1.0 and self.getCorrelationRescue(ap, correlation_set):
					self.abs_coeff[ind] = 1.0
			# Params that moved 0→1 due to correlation rescue
			_corr_rescued_a = {
				ap for ind, ap in enumerate(self.active_params)
				if ind < len(self.abs_coeff) and self.abs_coeff[ind] >= 1.0
				and ap not in _pre_rescue_a
			}

			self.check_list		= []
			self.selected_rxn_string = ""

			cf = self.cut_off / 100
			for ind, active_param in enumerate(self.active_params):
				if abs(self.abs_coeff[ind]) >= cf:
					self.partial_active[active_param]  = 1
					self.partial_active_list.append(active_param)
					self.selected.append(1)
					self.selected_rxn_string += f"{active_param}\n"
				else:
					self.partial_active[active_param] = 0
					self.selected.append(0)

			self.selected_rxn_count = sum(self.selected)

			# Build selection_detail (written alongside DesignMatrix)
			for ind, active_param in enumerate(self.active_params):
				s_val = self.coeff[ind] if ind < len(self.coeff) else float("nan")
				import math
				abs_s = abs(s_val) if not math.isnan(s_val) else float("nan")
				in_final = self.selected[ind] == 1
				by_thresh = not math.isnan(abs_s) and abs_s >= cf
				by_rescue = active_param in _corr_rescued_a
				by_linked = in_final and (not by_thresh) and (not by_rescue)
				self.selection_detail.append({
					"param": active_param,
					"S_value": round(s_val, 6) if not math.isnan(s_val) else None,
					"abs_S": round(abs_s, 6) if not math.isnan(abs_s) else None,
					"selected_by_threshold": by_thresh,
					"selected_by_linked_propagation": by_linked,
					"rescued_by_correlation": by_rescue,
					"final_in_PRS": in_final,
				})

		# ------------------------------------------------------------------ #
		# Three-parameter design (A, n, Ea) – new implementation			  #
		# ------------------------------------------------------------------ #
		else:
			csv_path = f"DM_FOR_PARTIAL_PRS/{case_index}/selected_parameters.csv"
			if status == "Pending" and not os.path.exists(csv_path):
				self._build_three_param_selection(sensitivity_dict, active_parameters)
			else:
				self._reload_three_param_selection(active_parameters)

	# ====================================================================== #
	# Three-parameter selection helpers										#
	# ====================================================================== #

	def _build_three_param_selection(self, sensitivity_dict, active_parameters):
		"""
		Build the selection vectors for the three-parameter (A, n, Ea) case
		from scratch.

		Ordering note
		-------------
		``sensitivity_dict`` is sorted externally by ``|s_A|`` before being
		passed in, so iterating it directly would populate ``s_A / s_n / s_Ea``
		in sorted order, which is misaligned with ``active_parameters`` (and
		the DesignMatrix columns) that both follow ``unsrt_data`` order.
		The fix: iterate ``self.unsrt`` and look up each reaction in
		``sensitivity_dict`` — identical to how the A-facto branch resolves
		the same mismatch via the ``rxn.split(":")[0]`` key.

		Cutoff criterion (per parameter)
		----------------------------------
		A parameter is *selected* when its sensitivity magnitude exceeds
		``(cut_off / 100) × max(|sensitivity|)`` computed separately for each
		of the three Arrhenius parameters over all reactions.  This introduces
		sparsity while respecting the relative importance scale of each
		parameter type.

		Linked-reaction propagation
		----------------------------
		PLOG-linked reaction pairs are handled identically to the A-facto
		case: if a parameter of the linked partner exceeds the threshold, the
		corresponding parameter of this reaction is also selected.

		NOTE: correlation-aware rescue (`self.correlation_list`) is NOT yet
		wired into this branch -- A-facto only, for now.
		"""
		# Populate s_A / s_n / s_Ea in unsrt_data order so that abs_coeff
		# is aligned 1-to-1 with active_parameters / DesignMatrix columns.
		for rxn in self.unsrt:
			for rxn_ in sensitivity_dict:
				if rxn_ == rxn.split(":")[0]:
					self.s_A.append(float(sensitivity_dict[rxn_][0]))
					self.s_n.append(float(sensitivity_dict[rxn_][1]))
					self.s_Ea.append(float(sensitivity_dict[rxn_][2]))

		self.active_params	     = active_parameters
		self.partial_active	     = {}
		self.partial_active_list = []
		self.selected			= []
		self.abs_coeff		     = []
		self.selected_rxn_string = ""

		# Per-parameter maxima – guard against empty lists
		max_sA  = max((abs(v) for v in self.s_A),  default=1.0)
		max_sn  = max((abs(v) for v in self.s_n),  default=1.0)
		max_sEa = max((abs(v) for v in self.s_Ea), default=1.0)

		thresh_A  = (self.cut_off / 100) * max_sA
		thresh_n  = (self.cut_off / 100) * max_sn
		thresh_Ea = (self.cut_off / 100) * max_sEa
		# Cycles with (ind % 3): 0 → A,  1 → n,  2 → Ea
		thresholds = [thresh_A, thresh_n, thresh_Ea]

		# Build abs_coeff: [|sA_r1|, |sn_r1|, |sEa_r1|, |sA_r2|, ...]
		# aligned 1-to-1 with active_params (3 entries per reaction).
		for idx in range(len(self.s_A)):
			self.abs_coeff.extend(
				self._get_sensitivity_vector(
					abs(self.s_A[idx]),
					abs(self.s_n[idx]),
					abs(self.s_Ea[idx]),
				)
			)

		# First pass – build check_list for linked-reaction propagation.
		# A param name is added when it directly exceeds its threshold.
		self.check_list = []
		for ind, active_param in enumerate(self.active_params):
			if self.abs_coeff[ind] >= thresholds[ind % 3]:
				self.check_list.append(active_param)

		# ---------------------------------------------------------- #
		# Correlation-aware rescue (3-param) — wired in here         #
		# Mirrors the A-facto branch: force abs_coeff up to threshold #
		# for any feature listed in correlation_list, and add it to   #
		# check_list so linked-reaction propagation works correctly   #
		# in the second pass below.                                   #
		# ---------------------------------------------------------- #
		_corr_rescued_3p = set()
		if self.correlation_list:
			_corr_set_3p = set(self.correlation_list)
			for ind, active_param in enumerate(self.active_params):
				param_type = ind % 3
				if (self.abs_coeff[ind] < thresholds[param_type]
						and active_param in _corr_set_3p):
					self.abs_coeff[ind] = thresholds[param_type]
					_corr_rescued_3p.add(active_param)
					if active_param not in self.check_list:
						self.check_list.append(active_param)

		# Suffix table aligned with param_type = ind % 3
		_param_suffixes = ["_A", "_n", "_Ea"]

		# Second pass – final selection with linked-reaction propagation.
		for ind, active_param in enumerate(self.active_params):
			param_type = ind % 3
			threshold  = thresholds[param_type]

			directly_selected = self.abs_coeff[ind] >= threshold

			# Linked-reaction check:
			# The A-param of this reaction is at active_params[ind - param_type].
			a_param_key  = self.active_params[ind - param_type]
			linked_idx   = self.linked_list.get(a_param_key)
			linked_param = (
				str(linked_idx) + _param_suffixes[param_type]
				if linked_idx is not None
				else None
			)
			linked_selected = (linked_param is not None) and (
				linked_param in self.check_list
			)

			if directly_selected or linked_selected:
				self.partial_active[active_param] = 1
				self.partial_active_list.append(active_param)
				self.selected.append(1)
				self.selected_rxn_string += f"{active_param}\n"
			else:
				self.partial_active[active_param] = 0
				self.selected.append(0)

		self.selected_rxn_count = sum(self.selected)

		# Build selection_detail (written alongside DesignMatrix)
		import math
		for ind, active_param in enumerate(self.active_params):
			rxn_idx = ind // 3
			p_idx   = ind % 3
			if rxn_idx < len(self.s_A):
				s_raw = [self.s_A[rxn_idx], self.s_n[rxn_idx], self.s_Ea[rxn_idx]][p_idx]
			else:
				s_raw = float("nan")
			in_final  = self.selected[ind] == 1
			by_thresh = not math.isnan(s_raw) and abs(s_raw) >= thresholds[p_idx]
			by_rescue = active_param in _corr_rescued_3p
			by_linked = in_final and (not by_thresh) and (not by_rescue)
			self.selection_detail.append({
				"param": active_param,
				"S_value": round(s_raw, 6) if not math.isnan(s_raw) else None,
				"abs_S": round(abs(s_raw), 6) if not math.isnan(s_raw) else None,
				"selected_by_threshold": by_thresh,
				"selected_by_linked_propagation": by_linked,
				"rescued_by_correlation": by_rescue,
				"final_in_PRS": in_final,
			})

	def _reload_three_param_selection(self, active_parameters):
		"""
		Reconstruct all selection variables from the previously saved
		``selected_parameters.csv`` file when restarting a run.

		This avoids repeating the sensitivity-based selection and ensures
		that the downstream design-matrix generation can continue seamlessly.
		All variables (``partial_active``, ``partial_active_list``,
		``selected``, ``selected_rxn_count``, ``selected_rxn_string``)
		are fully reconstructed.

		``abs_coeff`` is populated as a binary sentinel (1.0 / 0.0) reflecting
		the saved selection state; the original sensitivity magnitudes are not
		available on reload.
		"""
		self.active_params	   = active_parameters
		self.partial_active	  = {}
		self.partial_active_list = []
		self.selected			= []
		self.abs_coeff		   = []
		self.selected_rxn_string = ""

		csv_path = f"DM_FOR_PARTIAL_PRS/{self.case_index}/selected_parameters.csv"

		if os.path.exists(csv_path):
			selected_set = {
				line.strip()
				for line in open(csv_path).readlines()
				if line.strip()
			}
			for active_param in self.active_params:
				if active_param in selected_set:
					self.partial_active[active_param] = 1
					self.partial_active_list.append(active_param)
					self.selected.append(1)
					self.abs_coeff.append(1.0)
					self.selected_rxn_string += f"{active_param}\n"
				else:
					self.partial_active[active_param] = 0
					self.selected.append(0)
					self.abs_coeff.append(0.0)
			self.selected_rxn_count = len(selected_set)
		else:
			# No saved state found – initialise empty and warn the caller.
			self.selected_rxn_count = 0
			print(
				f"[Warning] Case {self.case_index}: no saved selection found at "
				f"'{csv_path}'. Re-run with status='Pending' to regenerate."
			)

	@staticmethod
	def _get_sensitivity_vector(sa, sn, se):
		"""
		Return the raw sensitivity magnitudes ``[sa, sn, se]`` for use in
		sparsity-based parameter selection (the *some* framework).

		Each element corresponds to one Arrhenius parameter (A, n, Ea).
		Selection is decided upstream by comparing these values against the
		per-parameter normalised threshold.

		Replaces the former ``getSelectionList`` / ``getSelectionListZeta``
		methods; the *all* vs *some* mode switch is no longer required.
		"""
		return [sa, sn, se]

	# ====================================================================== #
	# A-factor helpers – sensitivity cut-off, linked propagation,			#
	# and correlation-aware rescue (NEW)									   #
	# ====================================================================== #

	def populateCheckList(self, sa, activeParams):
		"""Add *activeParams* to check_list when |sa| exceeds the cut-off."""
		if abs(sa) > self.cut_off / 100:
			self.check_list.append(activeParams)

	def getSelectedLinkedRxn(self, sa, activeParams):
		"""
		Return ``[1.0]`` if *activeParams* should be selected either because
		its own sensitivity exceeds the cut-off or because its PLOG-linked
		partner was already placed in ``check_list``.
		"""
		cf = self.cut_off / 100
		if abs(sa) >= cf:
			return [1.0]
		if str(self.linked_list[activeParams]) + "_A" in self.check_list:
			return [1.0]
		return [0.0]

	def getCorrelationRescue(self, active_param, correlation_set):
		"""
		Return True if *active_param* should be force-selected because of
		the correlation analysis, either:
		  (a) it appears directly in `correlation_set`, or
		  (b) its PLOG-linked partner appears in `correlation_set` --
			  `correlation_list` only lists parent reactions, so linked
			  duplicates have to be pulled in this way, mirroring how
			  `getSelectedLinkedRxn` propagates the sensitivity cut-off.

		NOTE on naming: `correlation_set` is assumed to already use the
		same active-parameter string format as `self.active_params`
		(e.g. "12_A"), per how the correlation list was described. If your
		correlation list instead uses raw reaction-equation keys, match it
		against `self.unsrt` first (the same `rxn.split(":")[0]` pattern
		used elsewhere in this class) before calling this method.
		"""
		if active_param in correlation_set:
			return True
		linked_idx = self.linked_list.get(active_param)
		if linked_idx is not None and (str(linked_idx) + "_A") in correlation_set:
			return True
		return False

	# ====================================================================== #
	# Design-matrix sizing utilities – unchanged							   #
	# ====================================================================== #

	def getTotalUnknowns(self, N):
		n_ = 1 + 2 * N + (N * (N - 1)) / 2
		return int(n_)

	def getSim(self, n, design):
		n_ = self.getTotalUnknowns(n)
		return 4 * n_ if design == "A-facto" else 7 * n_

	# ====================================================================== #
	# Selection-detail CSV writer                                             #
	# ====================================================================== #

	def _write_selection_details(self, case_dir):
		"""
		Write selection_details.csv to *case_dir* explaining why each
		parameter is (or is not) in the final p-PRS active set.
		Columns: param, S_value, abs_S, selected_by_threshold,
		         selected_by_linked_propagation, rescued_by_correlation,
		         final_in_PRS.
		"""
		if not self.selection_detail:
			return
		import csv
		path = os.path.join(case_dir, "selection_details.csv")
		headers = [
			"param", "S_value", "abs_S",
			"selected_by_threshold", "selected_by_linked_propagation",
			"rescued_by_correlation", "final_in_PRS",
		]
		with open(path, "w", newline="") as f:
			w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
			w.writeheader()
			w.writerows(self.selection_detail)
		# Also write a compact summary
		n_thresh  = sum(1 for r in self.selection_detail if r["selected_by_threshold"])
		n_linked  = sum(1 for r in self.selection_detail if r["selected_by_linked_propagation"])
		n_rescue  = sum(1 for r in self.selection_detail if r["rescued_by_correlation"])
		n_final   = sum(1 for r in self.selection_detail if r["final_in_PRS"])
		summary = (
			f"# Selection summary for Case {self.case_index}\n"
			f"total_active_params,{len(self.selection_detail)}\n"
			f"selected_by_threshold,{n_thresh}\n"
			f"selected_by_linked_propagation,{n_linked}\n"
			f"rescued_by_correlation,{n_rescue}\n"
			f"final_in_PRS,{n_final}\n"
		)
		open(os.path.join(case_dir, "selection_summary.csv"), "w").write(summary)

	# ====================================================================== #
	# DesignMatrix.csv provenance (2026-08-19, audit trail) — see the        #
	# "NOTE" print in partial_DesignMatrix's else-branch for why this is a   #
	# WARN-and-keep record, not a regeneration gate: the sunk simulation     #
	# cost behind a cached DesignMatrix.csv is deliberately preserved, and   #
	# per-row correctness is enforced downstream at fit time instead.       #
	# ====================================================================== #

	def _write_dm_provenance(self, dm_dir, n_rows, n_cols):
		path = os.path.join(dm_dir, "DesignMatrix.csv")
		sha1 = ""
		try:
			h = hashlib.sha1()
			with open(path, "rb") as fh:
				for chunk in iter(lambda: fh.read(1 << 20), b""):
					h.update(chunk)
			sha1 = h.hexdigest()
		except Exception:
			pass
		prov = {
			"sampler_version": SAMPLER_VERSION,
			"design": self.design,
			"prs_type": "Partial",
			"n_rows": int(n_rows),
			"n_cols": int(n_cols),
			"sha1": sha1,
			"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
		}
		with open(os.path.join(dm_dir, "dm_provenance.json"), "w") as fh:
			json.dump(prov, fh, indent=2)
		return prov

	def _read_dm_provenance(self, dm_dir):
		path = os.path.join(dm_dir, "dm_provenance.json")
		if not os.path.isfile(path):
			return None
		try:
			with open(path) as fh:
				return json.load(fh)
		except Exception:
			return None

	# ====================================================================== #
	# Partial design matrix builder                    						#
	# ====================================================================== #

	def partial_DesignMatrix(self):
		na = len(self.partial_active_list)
		n_rxn = len(self.active_params)
		self.no_of_sim = self.getSim(na, self.design)   # formula estimate; updated to actual below

		if "DM_FOR_PARTIAL_PRS" not in os.listdir():
			os.mkdir("DM_FOR_PARTIAL_PRS")
			os.mkdir(f"DM_FOR_PARTIAL_PRS/{self.case_index}")
		else:
			if f"{self.case_index}" not in os.listdir("DM_FOR_PARTIAL_PRS/"):
				os.mkdir(f"DM_FOR_PARTIAL_PRS/{self.case_index}")

		# Write selection details alongside the DesignMatrix directory
		self._write_selection_details(f"DM_FOR_PARTIAL_PRS/{self.case_index}")

		#######################################################################
		### Goes to DesignMatrix Modules to create Matrix for Partial PRS ###
		#######################################################################

		dm_dir = f"DM_FOR_PARTIAL_PRS/{self.case_index}"
		if "DesignMatrix.csv" not in os.listdir(dm_dir + "/"):
			design_matrix,selection_matrix,p_design_matrix,p_selection_matrix = DM.DesignMatrix(self.unsrt,self.design,self.getSim(na,self.design),n_rxn).getSample_partial(self.case_index,self.selected)
			g = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/selected_parameters.csv","w").write(self.selected_rxn_string)
			self._write_dm_provenance(dm_dir, n_rows=len(p_design_matrix), n_cols=(len(p_design_matrix[0]) if len(p_design_matrix) else 0))

		# partial_PRS_system.py  →  partial_DesignMatrix(), else branch
		else:
			# ── Provenance check (2026-08-19, audit-only, does NOT force
			# regeneration) ──────────────────────────────────────────────
			# A cached DesignMatrix.csv predating the current
			# Uncertainty.SAMPLER_VERSION is EXPECTED here, by design: this
			# study's ~30k already-simulated band-valid rows (and the real
			# Chemkin simulation cost behind them) are a sunk cost worth
			# keeping, not discarding and re-simulating from scratch (that
			# option -- "full regeneration + full re-simulation" -- was
			# offered and explicitly declined in favour of reusing existing
			# band-valid rows). So this check WARNS (loudly, and writes an
			# audit record) rather than deletes/regenerates -- the actual
			# guarantee that a pre-fix bad sample can never again silently
			# corrupt a fit or a reported error is enforced downstream, at
			# FIT time, by run_script.py's per-case screen_case_samples
			# (Part 2 of the 2026-08-19 fix): every row is re-scored against
			# TODAY's band-validity test regardless of when/how it was
			# generated, and only band-valid rows ever reach the PRS or the
			# reported metrics. What DOES get force-regenerated on a stale
			# sampler version is the case's class-A/B/C POOL
			# ({a,b,c}_type_samples_partial_*.pkl, via
			# DesignMatrix.refresh_class_pools, called from run_script.py
			# before Stage 3 -- those feed NEW candidate generation, where
			# there is no sunk cost to protect and a stale pool is actively
			# harmful (see AdaptiveResampler.DOptimalResampler's own
			# provenance assertion).
			prov = self._read_dm_provenance(dm_dir)
			if prov is None or prov.get("sampler_version") != SAMPLER_VERSION:
				print(f"\n[Case-{self.case_index}] NOTE: cached DesignMatrix.csv predates "
				      f"(or has no record of) SAMPLER_VERSION={SAMPLER_VERSION!r} -- "
				      f"reusing it anyway (it is a sunk simulation cost), but every row "
				      f"will be re-screened against the CURRENT band-validity test before "
				      f"it can reach the PRS fit or any reported error (see run_script.py's "
				      f"screen_case_samples). Class pools for Stage-3 resampling are "
				      f"refreshed independently and are NOT covered by this note.")

			design_matrix_file      = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/DesignMatrix.csv").readlines()
			selection_matrix_file   = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/SelectionMatrix.csv").readlines()
			p_design_matrix_file    = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/pDesignMatrix.csv").readlines()
			p_selection_matrix_file = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/pSelectionMatrix.csv").readlines()
			selected_parameters     = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/selected_parameters.csv").readlines()
			selected_parameters     = [i.strip() for i in selected_parameters]
			self.selected_rxn_count = len(selected_parameters)
			
			design_matrix = []
			for row in design_matrix_file:
				design_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
			
			selection_matrix = []
			for row in selection_matrix_file:
				selection_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
			
			p_design_matrix = []
			for row in p_design_matrix_file:
				p_design_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
			
			p_selection_matrix = []
			for row in p_selection_matrix_file:
				p_selection_matrix.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
		
		# Update no_of_sim to the ACTUAL row count of the design matrix
		# (Sobol or other samplers may produce more rows than the formula estimate)
		self.no_of_sim = len(p_design_matrix)
		print(
			f"\n[Case-{self.case_index}]"
			f"\n\tSelected features/reactions  : {self.selected_rxn_count}"
			f"\n\tEstimated sims (2nd-order formula) : {self.getSim(na, self.design)}"
			f"\n\tActual sims (design matrix rows)   : {self.no_of_sim}\n"
		)

		####################################################################################################
		### Goes to Simulation Manager to generate YAML mechanisms based on DesignMatrix for Partial PRS ###
		####################################################################################################

		SSM = simulator.SM(self.target_list,self.optInputs,self.unsrt,design_matrix,tag="Partial")
		if f"YAML_FILES_FOR_PARTIAL_PRS" not in os.listdir():
			os.mkdir("YAML_FILES_FOR_PARTIAL_PRS")
		if f"{self.case_index}" not in os.listdir("YAML_FILES_FOR_PARTIAL_PRS/"):
			os.mkdir(f"YAML_FILES_FOR_PARTIAL_PRS/{self.case_index}")
			print("\nPerturbing the Mechanism files\n")
			
			chunk_size = 500
			params_yaml = [p_design_matrix[i:i+chunk_size] for i in range(0, len(p_design_matrix), chunk_size)]
			params_selection_yaml = [p_selection_matrix[i:i+chunk_size] for i in range(0, len(p_selection_matrix), chunk_size)]
			count = 0
			yaml_loc = []
			for index,params in enumerate(params_yaml):
				if self.design == "A-facto":
					yaml_list = SSM.getYAML_List(params,selection=params_selection_yaml[index])
				else:
					yaml_list = SSM.getYAML_List(params,selection=params_selection_yaml[index])
				#yaml_loc = []
				location_mech = []
				index_list = []
				for i,dict_ in enumerate(yaml_list):
					index_list.append(str(count+i))
					location_mech.append(os.getcwd()+f"/YAML_FILES_FOR_PARTIAL_PRS/{self.case_index}/")
					yaml_loc.append(os.getcwd()+f"/YAML_FILES_FOR_PARTIAL_PRS/{self.case_index}/mechanism_"+str(count+i)+".yaml")
				count+=len(yaml_list)
				#gen_flag = False
				#SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
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
				location_mech.append(os.getcwd()+f"/YAML_FILES_FOR_PARTIAL_PRS/{self.case_index}")
				yaml_loc.append(os.getcwd()+f"/YAML_FILES_FOR_PARTIAL_PRS/{self.case_index}/mechanism_"+str(i)+".yaml")
		return yaml_loc,p_design_matrix,self.selected, self.selected_rxn_count
