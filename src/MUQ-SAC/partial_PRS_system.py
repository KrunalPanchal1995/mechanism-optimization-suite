import os,sys
import numpy as np
import DesignMatrix as DM
import pickle
import simulation_manager2_0 as simulator

class PartialPRS(object):
	"""
	Partial Polynomial Response Surface class.

	Selects the active Arrhenius parameters for each reaction based on
	normalised sensitivity coefficients, builds a partial design matrix,
	and generates the perturbed YAML mechanism files required for
	combustion-target simulations.

	Supports two design modes
	--------------------------
	* ``"A-facto"``	  – single A-factor perturbation (tested, unchanged).
	* ``"A1+B1+C1"``	 – full three-parameter (A, n, Ea) perturbation using
						   the *some* sparsity framework (new).

	Parameters
	----------
	sensitivity_dict : dict
		For ``"A-facto"``: ``{rxn: sensitivity_value}``.
		For three-param:  ``{rxn: [s_A, s_n, s_Ea]}``.
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
		self.sens_param   = str(optInputs["Stats"]["sensitive_parameters"])
		# NOTE: Arrhenius_Selection_Type removed — always using "some" framework.
		self.design	   = design
		self.case_index   = case_index
		self.unsrt		= unsrt_data

		self.s_A  = []
		self.s_n  = []
		self.s_Ea = []
		self.no_of_sim = None

		# Build linked-reaction map keyed on each reaction's A-param string.
		# Used to propagate selection to PLOG-linked reactions.
		self.linked_list = {}
		self.check_list  = []
		self.active_params = []
  
		# ------------------------------------------------------------------ #
		# A-factor design – tested and frozen, do not alter				   #
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

			self.check_list		= []
			self.selected_rxn_string = ""

			for ind, active_param in enumerate(self.active_params):
				if abs(self.abs_coeff[ind]) >= self.cut_off / 100:
					self.partial_active[active_param]  = 1
					self.partial_active_list.append(active_param)
					self.selected.append(1)
					self.selected_rxn_string += f"{active_param}\n"
				else:
					self.partial_active[active_param] = 0
					self.selected.append(0)

			self.selected_rxn_count = sum(self.selected)

		# ------------------------------------------------------------------ #
		# Three-parameter design (A, n, Ea) – new implementation			  #
		# ------------------------------------------------------------------ #
		else:
			if status == "Pending":
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
		"""
		# Populate s_A / s_n / s_Ea in unsrt_data order so that abs_coeff
		# is aligned 1-to-1 with active_parameters / DesignMatrix columns.
		for rxn in self.unsrt:
			for rxn_ in sensitivity_dict:
				if rxn_ == rxn.split(":")[0]:
					self.s_A.append(float(sensitivity_dict[rxn_][0]))
					self.s_n.append(float(sensitivity_dict[rxn_][1]))
					self.s_Ea.append(float(sensitivity_dict[rxn_][2]))

		self.active_params	   = active_parameters
		self.partial_active	  = {}
		self.partial_active_list = []
		self.selected			= []
		self.abs_coeff		   = []
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
	# A-factor helpers – unchanged											 #
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
	# Partial design matrix builder                    						#
	# ====================================================================== #
	
	def partial_DesignMatrix(self):
		na = len(self.partial_active_list)
		n_rxn = len(self.active_params)
		self.no_of_sim = self.getSim(na,self.design)
		
		print(f"\n[Case-{self.case_index}]\n\tNo. of Simulations required: {self.getSim(na,self.design)}\n\tNo. of selected reactions: {self.selected_rxn_count}\n")
		if "DM_FOR_PARTIAL_PRS" not in os.listdir():
			os.mkdir("DM_FOR_PARTIAL_PRS")
			os.mkdir(f"DM_FOR_PARTIAL_PRS/{self.case_index}")
		else:
			if f"{self.case_index}" not in os.listdir("DM_FOR_PARTIAL_PRS/"):
				os.mkdir(f"DM_FOR_PARTIAL_PRS/{self.case_index}")
		
		#######################################################################
		### Goes to DesignMatrix Modules to create Matrix for Partial PRS ###
		#######################################################################
		
		if "DesignMatrix.csv" not in os.listdir(f"DM_FOR_PARTIAL_PRS/{self.case_index}/"):
			design_matrix,selection_matrix,p_design_matrix,p_selection_matrix = DM.DesignMatrix(self.unsrt,self.design,self.getSim(na,self.design),n_rxn).getSample_partial(self.case_index,self.selected)
			g = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/selected_parameters.csv","w").write(self.selected_rxn_string)
		else:
			for rxn in self.unsrt:
				self.active_params.extend(list(self.unsrt[rxn].activeParameters))
			#print(len(self.active_params))
			design_matrix_file = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/DesignMatrix.csv").readlines()
			selection_matrix_file = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/SelectionMatrix.csv").readlines()
			p_design_matrix_file = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/pDesignMatrix.csv").readlines()
			p_selection_matrix_file = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/pSelectionMatrix.csv").readlines()
			selected_parameters = open(f"DM_FOR_PARTIAL_PRS/{self.case_index}/selected_parameters.csv").readlines()
			selected_parameters = [i.strip() for i in selected_parameters]
			self.selected_rxn_count = len(selected_parameters)
			#print(p_selection_matrix_file)
			for rxn in self.active_params:
				if rxn in selected_parameters:
					self.selected.append(1)
				else:
					self.selected.append(0)
			#print(len(self.selected))
			#raise AssertionError("Stop!")
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
			params_yaml = [design_matrix[i:i+chunk_size] for i in range(0, len(design_matrix), chunk_size)]
			params_selection_yaml = [selection_matrix[i:i+chunk_size] for i in range(0, len(selection_matrix), chunk_size)]
			"""
			print(self.sens_param)
			if self.sens_param == "zeta":
				params_selection_yaml = [selection_matrix[i:i+chunk_size] for i in range(0, len(selection_matrix), chunk_size)]
			else:
				params_selection_yaml = [p_selection_matrix[i:i+chunk_size] for i in range(0, len(p_selection_matrix), chunk_size)]
			"""	
			count = 0
			yaml_loc = []
			for index,params in enumerate(params_yaml):
				if self.design == "A-facto":
					yaml_list = SSM.getYAML_List(params)#,selection=params_selection_yaml[index])
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
		return yaml_loc,p_design_matrix,self.selected
