def species_selection(species,species_data,carbon_number):
	selected_species = []
	for kind in species:
		for data in species_data:
			if kind in data["name"]:
				kind_composition = data["composition"]
				if "C" in kind_composition:
					if kind_composition["C"] >= carbon_number:
						selected_species.append(data["name"])
			
	selected_species = set(selected_species)
	return selected_species

def _species_selection(species,species_data,carbon_number,start=None,stop=None):
	selected_species = []
	start_species = start
	stop_species = stop
	for kind in species:
		for data in species_data:
			if kind in data["name"]:
				kind_composition = data["composition"]
				if "C" in kind_composition:
					if kind_composition["C"] >= stop and kind_composition["C"] <= start:
						selected_species.append(data["name"])
			
	selected_species = set(selected_species)
	return selected_species

def reaction_selection(selected_species,reactions):
	selected_reactions = []
	for species in selected_species:
		for reaction in reactions:
			if species in reaction["equation"]:
				selected_reactions.append(reaction["equation"])

	selected_reactions = set(selected_reactions)
	return selected_reactions

def _norm_eq(eq):
	"""Order- and whitespace-insensitive equation key: 'H+O2<=>O+OH',
	'H + O2 <=> O + OH', and 'O2 + H <=> O + OH' must all collide.

	Found 2026-07-22 running the ammonia campaign's Phase 6 target-opt (the
	first real end-to-end exercise of this A-factor path this session): the
	shipped uncertainty-XML convention (DATASET/H2_O2_TEST/uncertainty/
	H2_uncertainty.xml's own `rxn="H+O2&lt;=&gt;O+OH"`, no spaces -- the
	same convention uncertainty_generator_gui.py's generate_uncertainty_xml()
	uses) never matches a real Cantera-YAML mechanism's equation string
	under a plain exact-string (or even whitespace-only-normalized)
	comparison: ck2yaml always emits full spacing ("H + O2 <=> O + OH"),
	*and* reactant/product order follows the mechanism's own declaration,
	not necessarily the order a user or an uncertainty-XML author happened
	to type them in (Konnov 2019's own H+O2<=>O+OH is declared
	'O2 + H <=> O + OH'). Every A-factor reaction-index lookup silently
	returned an empty dict for any real mechanism -- no crash here, just an
	empty result -- until DesignMatrix's nominal/SA sample construction
	(which sizes its perturbation vector off this same dict) produced a
	too-short row and simulation_manager.py's getDirectoryList() indexed
	past the end of the (also-empty) reaction_index list a few calls later.
	Normalizing here (not in DesignMatrix/simulation_manager) fixes it at
	the actual mismatch, matching how the rest of this repo already handles
	the same class of problem (see rmip.analysis.mechanism_check's
	_yaml_reaction_signature, which also sorts each side's species)."""
	s = str(eq).replace("<=>", "=").replace("=>", "=").replace("<=", "=")
	sides = [side.strip() for side in s.split("=")]
	norm_sides = []
	for side in sides:
		# drop a leading numeric stoichiometric coefficient per term, and
		# any surrounding whitespace, then sort so term order doesn't matter
		terms = []
		for term in side.split("+"):
			t = term.strip()
			parts = t.split(None, 1)
			if len(parts) == 2:
				try:
					float(parts[0])
					t = parts[1].strip()
				except ValueError:
					pass
			terms.append(t)
		norm_sides.append(tuple(sorted(terms)))
	return tuple(norm_sides)

def reaction_index(selected_reactions,reactions):
	reaction_dict = {}
	for rxn in selected_reactions:
		rxn_key = rxn.split(":")[0]
		for index,reaction in enumerate(reactions):
			if _norm_eq(rxn_key) == _norm_eq(reaction["equation"]):
				reaction_dict[index+1] = rxn_key
				#break

	return reaction_dict

def getRxnDetails(mechanism,rxn_list):
	rxn_dict = {}
	rxn_data = mechanism["reactions"]
	for rxn in rxn_list:
		new_rxn_data = {}
		temp = []
		index_ = []
		rxn_ = []
		for index,data in enumerate(rxn_data):
			if _norm_eq(rxn) == _norm_eq(data["equation"]):
				temp.append(data)
				index_.append(index+1)
		new_rxn_data["temp"] = temp
		new_rxn_data["index"] = index_
		rxn_dict[rxn] = new_rxn_data
	return rxn_dict

"""
#Shutting this down to test the hypothesis:	
	-Dulicate reactions are treated as elementary reactions in FlameMaster
def getRxnType(mechanism,rxn_list):
	rxn_type = {}
	rxn_data = mechanism["reactions"]
	for rxn in rxn_list:
		for data in rxn_data:
			if rxn in data["equation"]:
				if "type" in data:
					if data["type"] == "three-body":
						rxn_type[data["equation"]] = "ThirdBody"
						#print("Rxn is three-body")
					elif data["type"] == "falloff":
						rxn_type[data["equation"]] = "Falloff"
						#print("Rxn is fall-off")
					elif data["type"] == "Chebyshev":
						rxn_type[data["equation"]] = "Chebyshev"
						
					elif data["type"] == "pressure-dependent-Arrhenius" and "duplicate" not in data:
						rxn_type[data["equation"]] = "PLOG"
						#print("Rxn is Plog")
					elif data["type"] == "pressure-dependent-Arrhenius" and "duplicate" in data:
						rxn_type[data["equation"]] = "PLOG-Duplicate"
						break
						#print("Rxn is Plog duplicate")
				elif "duplicate" in data:
					rxn_type[data["equation"]] = "Duplicate"
					break
					#print("Rxn is duplicate")
				
				else:
					rxn_type[data["equation"]] = "Elementary"
					#print("Rxn is elementary")
	return rxn_type
"""
def getRxnType(mechanism,rxn_list):
	rxn_type = {}
	rxn_data = mechanism["reactions"]
	for rxn in rxn_list:
		for index,data in enumerate(rxn_data):
			if _norm_eq(rxn) == _norm_eq(data["equation"]):
				#print(rxn,data["equation"])
				if "type" in data:
					if data["type"] == "three-body":
						rxn_type[index+1] = "ThirdBody"
						#print("Rxn is three-body")
					elif data["type"] == "falloff":
						rxn_type[index+1] = "Falloff"
						#print("Rxn is fall-off")
					elif data["type"] == "Chebyshev":
						rxn_type[index+1] = "Chebyshev"
						
					elif data["type"] == "pressure-dependent-Arrhenius":# and "duplicate" not in data:
						rxn_type[index+1] = "PLOG"
						#print("Rxn is Plog")
					#elif data["type"] == "pressure-dependent-Arrhenius" and "duplicate" in data:
						#rxn_type[data["equation"]] = "PLOG-Duplicate"
						#break
						#print("Rxn is Plog duplicate")
				#elif "duplicate" in data:
				#	rxn_type[data["equation"]] = "Duplicate"
				#	break
					#print("Rxn is duplicate")
				
				else:
					rxn_type[index+1] = "Elementary"
					#print("Rxn is elementary")
	return rxn_type
def getMAX_B_TYPE_ZETA(unsrt):
	
	return zeta_b2
