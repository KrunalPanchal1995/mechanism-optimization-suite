import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from numpy import linalg as LA
import json
from scipy.interpolate import InterpolatedUnivariateSpline
import statistics
import pandas as pd
import time
from scipy.optimize import minimize

# Task 2D: load MW/stoich from shared chem_data module; fall back to inline tables
# if chem_data is not on the path (e.g., running from src/MUQ-SAC/ directly).
try:
	_chem_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
	if _chem_data_dir not in sys.path:
		sys.path.insert(0, _chem_data_dir)
	from chem_data import DEFAULT_FUEL_MW as _DEFAULT_MW, DEFAULT_STOICH_O2 as _DEFAULT_NU, compute_nu_o2 as _compute_nu_o2
	_HAS_CHEM_DATA = True
except ImportError:
	_HAS_CHEM_DATA = False

class combustion_target():
###Class definition and acquisition of input parameters.
	def __init__(self,data,addendum,index):
		# Task 2D: MW and stoichiometry tables now sourced from chem_data.py.
		# Hard-coded fallback retained for backward compatibility when
		# chem_data is unavailable.
		if _HAS_CHEM_DATA:
			self.molecularWt = dict(_DEFAULT_MW)
			self.stoichiometry = dict(_DEFAULT_NU)
		else:
			self.molecularWt = {}
			self.molecularWt["CO"] = 28
			self.molecularWt["H2"] = 2
			self.molecularWt["O2"] = 32
			self.molecularWt["co"] = 28
			self.molecularWt["h2"] = 2
			self.molecularWt["o2"] = 32
			self.molecularWt["CO2"] = 0
			self.molecularWt["AR"] = 0
			self.molecularWt["HE"] = 0
			self.molecularWt["Ar"] = 0
			self.molecularWt["He"] = 0
			self.molecularWt["N2"] = 0
			self.molecularWt["H2O"] = 0
			self.molecularWt["CH4"] = 16
			self.molecularWt["NC7H16"] = 100.21
			self.molecularWt["nc7h16"] = 100.21
			self.molecularWt["C4H6"] = 54
			self.molecularWt["OME3"] = 136
			self.molecularWt["MB-C5H10O2"] = 86
			self.molecularWt["CH2O"] = 30
			self.molecularWt["NC12H26"] = 170
			self.molecularWt["C7H8"] = 92
			self.molecularWt["IC16H34"] = 226
			self.stoichiometry = {}
			self.stoichiometry["H2"] = 0.5
			self.stoichiometry["CO"] = 0.5
			self.stoichiometry["O2"] = 0.0
			self.stoichiometry["N2"] = 0.0
			self.stoichiometry["h2"] = 0.5
			self.stoichiometry["co"] = 0.5
			self.stoichiometry["o2"] = 0.0
			self.stoichiometry["n2"] = 0.0
			self.stoichiometry["HE"] = 0.0
			self.stoichiometry["AR"] = 0.0
			self.stoichiometry["He"] = 0.0
			self.stoichiometry["Ar"] = 0.0
			self.stoichiometry["H2O"] = 0.0
			self.stoichiometry["CO2"] = 0.0
			self.stoichiometry["CH4"] = 2.0
			self.stoichiometry["NC7H16"] = 11
			self.stoichiometry["nc7h16"] = 11
			self.stoichiometry["C4H6"] = 5.5
			self.stoichiometry["OME3"] = 6
			self.stoichiometry["MB-C5H10O2"] = 6.5
			self.stoichiometry["CH2O"] = 1
			self.stoichiometry["NC12H26"] = 25
			self.stoichiometry["C7H8"] = 11
			self.stoichiometry["IC16H34"] = 33
			self.stoichiometry["C12H23"] = 17.75
			self.stoichiometry["C3H8"] = 5.0
		self.data = data
		#print(self.data)
		parameters = self.data.split('|')
		#print(parameters)
		self.dataSet_id = parameters[1].strip('\t')
		self.calculated = 0
		self.x = None
		self.index = index
		self.case_index = "case-"+str(index)
		self.units = self.f_unit = self.t_unit = self.temperature_factor = self.p_unit = self.pressure_factor = self.target_unit = self.target_factor = self.flow_unit  = self.flow_rate = self.target_key = self.input_file = self.target = self.simulation = self.temperature = self.fuel = self.oxygen = self.nitrogen = self.argon = self.pressure = self.phi = self.observed = self.d_weight = self.d_set = self.species = self.s_p_name = None
		self.add = {}	
		self.simulated = None
		#raise ValueError("addendum")
		if self.dataSet_id in addendum:
			self.add = addendum[self.dataSet_id]
			if self.add == None:
				self.add = {}
				self.add["solver"] = "FlameMaster"
			#print(self.add)
		else:
			self.add = {}
			
			self.add["solver"] = "FlameMaster"
		
		self.uniqueID = str(self.dataSet_id)+"_"+str(parameters[0].strip("\t"))
		
		#print(self.uniqueID)
		#By default  
#		default = {}
#		cantera_def = {}
#		cantera_def["saveAll"] = '#' 
#		cantera_def["ign_species"] = 'OH'
#		cantera_def["ign_cond"] = 'max'
#		cantera_def["ign_specific_cond"] = "None;None"
#		FlameMan_def = {}
#		default["cantera"] = cantera_def
#		default["FlameMaster"] = FlameMan_def
		self.species_dict = {}
		self.fuel_dict = {}
		self.BG_dict = {}
		self.ignition_type = "reflected"
		self.flame = "laminar"
		self.reactor = "JSR" 
		self.std_dvtn = 1.0
		
		for param in parameters:
			#print(param)
			key_and_value = param.split("--")
			
			if len(key_and_value) == 1:
				continue
			
			key = key_and_value[0].strip()
			#print(key)
			content = key_and_value[1].strip()
			
			if "input_file" in key:
				self.input_file = os.path.abspath(content)
				continue
					
			if "target" in key:
				self.target = content
#				print(content)
#				if content.strip(" ") == "ignition delay measurement" or "Tig":
#					self.target = "Tig"
#				
#				elif content.strip(" ") == "laminar burning velocity measurement" or "Fsl" or "Sl":
#					self.target = "Fsl"
#					
#				elif content.strip(" ") == "Heat Flux Burner" or "Flf" or "HeatFlux":
#					self.target = "Flf"
#					
#				elif content.strip(" ") == "laminar burning velocity measurement" or "Fsl" or "Sl":
#					self.target = "Flw"
#					
#				else:
#					raise AssertionError("Invalid target type")
				#print(self.target)
			if "simulation" in key :
				self.simulation = content
			
			if "measurnment_type" in key:
				self.measure_type = content
			
			if "Flame_type" in key:
				self.flame_type = content
			
			if "Ignition_mode" in key:
				self.ig_mode = content.strip("[]")
			
			if "Reactor_type" in key:
				self.reactor = content
			
			if "Residence_time" in key:
				self.residenceTime = content
			
			if "startProfile" in key:
				self.s_p_name = content
				
			if "Fuel_type" in key:
				self.fuel_type = content
	
			if key.strip() == "Fuel":
				#print(content.split("->"))
				#print(content)
				# Real generated .target files (e.g. DATASET/H2_O2_TEST/targets/
				# H2_IDT.target) write the fuel spec as a dict even for a
				# single-species "Mono" fuel, e.g. "x->{'a': 'H2'} = {'a': 0.0050}"
				# -- the generator does not special-case Mono to a bare scalar.
				# Dispatch on the ACTUAL content shape (does it start with "{"),
				# not just the declared Fuel_type, so a real Mono target doesn't
				# crash trying float() on a dict-repr string. A genuine bare-
				# scalar Mono line (no "{") still takes the original branch.
				_fuel_rhs = content.split("->")[1].split("=")[0].strip()
				if "Multi" in self.fuel_type or _fuel_rhs.startswith("{"):
					self.fuel_id = json.loads(content.split("=")[0].split("->")[1].replace("'",'"'))
					self.fuel_x = json.loads(content.split("=")[1].replace("'",'"'))
					self.fuel_is = ""
					for i in self.fuel_id:
						self.species_dict[str(self.fuel_id[i])] = float(self.fuel_x[i])
						self.fuel_dict[str(self.fuel_id[i])] = float(self.fuel_x[i])
						self.fuel_is+="fuel is "+str(self.fuel_id[i])+"\n"
				else:
					self.fuel_id = content.split("->")[1].split("=")[0]
					self.fuel_x = float(content.split("->")[1].split("=")[1])
					self.species_dict[str(self.fuel_id)] = float(self.fuel_x)
					self.fuel_dict[str(self.fuel_id)] = float(self.fuel_x)
			
			if "Oxidizer" in key:
				# Same dict-shape dispatch as the Fuel field above (B4,
				# Round-2 audit): no shipped .target currently writes a
				# dict-valued Oxidizer, but generate_target_opt.py's writer
				# does not special-case Oxidizer to a bare scalar any more
				# than it did for Fuel, so a Multi oxidizer blend (e.g. an
				# O2/N2/Ar custom oxidizer, as opposed to Bath_gas) would hit
				# the same "ValueError: could not convert string to float"
				# this dispatch already fixed for Fuel. actual_oxidizer
				# collapses to the summed oxidizer mole fraction, matching
				# how Fact (line ~524) uses a single scalar amount
				# regardless of how many oxidizer species compose it.
				#
				# Unlike Fuel, this dispatches on content shape ALONE, not
				# on self.fuel_type -- real data (DATASET/METHYL_BUTANOATE_2026/
				# MB_LTC_no_dummy.target) has Fuel_type=Multi with a plain
				# scalar Oxidizer ("x->O2 = 0.2035"): Fuel_type governs the
				# FUEL field's own shape, not the Oxidizer field's, and
				# gating on it here sent that real scalar through json.loads
				# and crashed with JSONDecodeError (caught by
				# test_write_curated_real_bridge_gates_clean going from pass
				# to fail during this same fix -- corrected before commit).
				_ox_rhs = content.split("->")[1].split("=")[0].strip()
				if _ox_rhs.startswith("{"):
					self.oxidizer = json.loads(content.split("=")[0].split("->")[1].replace("'",'"'))
					self.oxidizer_x = json.loads(content.split("=")[1].replace("'",'"'))
					for i in self.oxidizer:
						self.species_dict[str(self.oxidizer[i])] = float(self.oxidizer_x[i])
					actual_oxidizer = sum(float(v) for v in self.oxidizer_x.values())
				else:
					self.oxidizer = content.split("->")[1].split("=")[0].strip(" ")
					self.oxidizer_x = float(content.split("->")[1].split("=")[1])# to convert to percentage
					self.species_dict[str(self.oxidizer)] = float(self.oxidizer_x)
					actual_oxidizer = float(self.oxidizer_x)
			
			
			if "Bath_gas" in key:
				self.bath_gas_id = "Multi"
				self.bath_gas = json.loads(content.split("=")[0].split("->")[1].replace("'",'"'))
				self.bath_gas_x = json.loads(content.split("=")[1].replace("'",'"'))
	
				for i in self.bath_gas:
					self.species_dict[str(self.bath_gas[i])] = float(self.bath_gas_x[i])
					self.BG_dict[str(self.bath_gas[i])] = float(self.bath_gas_x[i])
						
			if "BG1" in key:
				if content.split("=")[1].strip() != '':
					self.bath_gas1 = content.split("->")[1].split("=")[0]
					self.bath_gas1_x = float(content.split("->")[1].split("=")[1])# to convert to percentage
					self.bg1_tag = ""
				else:
					self.bath_gas1 = ""
					self.bath_gas1_x = ""# to convert to percentage
					self.bg1_tag = "#"
				
			if "BG2" in key:#mole %/100
				if content.split("=")[1].strip() != '':
					self.bath_gas2 = content.split("->")[1].split("=")[0]
					self.bath_gas2_x = float(content.split("->")[1].split("=")[1])# to convert to percentage
					self.bg2_tag = ""
				else:
					self.bath_gas2 = ""
					self.bath_gas2_x = ""# to convert to percentage
					self.bg2_tag = "#"
				
			if "BG3" in key:#mole %/100
				
				if content.split("=")[1].strip() != '':
					self.bath_gas3 = content.split("->")[1].split("=")[0]
					self.bath_gas3_x = float(content.split("->")[1].split("=")[1])# to convert to percentage
					self.bg3_tag = ""
				else:
					self.bath_gas3 = ""
					self.bath_gas3_x = ""# to convert to percentage
					self.bg3_tag = "#"
			
			if  key == "T":#K
				self.temperature = float(content)
			if  key == "Ti":#K
				self.temperature_i = float(content)
			
			if key.strip() == "units":
				self.units = json.loads(content.replace("'",'"'))
			
			if key.strip() == "flow_rate":
				self.flow_rate = content
			
			if key.strip() == "Fuel_units":		
				self.f_unit = content
			
			if key.strip() == "T_units":
				self.t_unit = content
							
			if key.strip() == "P_units":
				self.p_unit = content
			
			if key.strip() == "obs_unit":
				self.target_unit = content
			
			if key.strip() == "flow_units":
				self.flow_unit = content
				
			if key.strip() == "P":
				self.pressure = float(content) #atm
			
			if key.strip() == "Pi":
				self.pressure_i = float(content) #atm
				
			if key.strip() == "Phi":
				self.phi = str(content)
					
			if key.strip() == "observed":
				self.observed = float(content);
				
			if "deviation" in key:
				#print(self.uniqueID,content)
				self.std_dvtn = float(content);
			
			if "dataSet" in key:
				self.d_set = content;#index of a data set
			else:
				self.d_set = self.dataSet_id

			if "data_weight" in key:
				self.d_weight = 1/float(content.strip());#weight of each data point in a perticular data set
			
			if "species" in key:
				self.species = content
				#print(content)
			
			
			#print(self.target)		
		if "Pi" in self.units:
			if self.units["Pi"].strip(" ") == "mbar":
				self.pressure_i = float(self.pressure_i)*1e2
			if self.units["Pi"].strip(" ") == "bar":
				self.pressure_i = float(self.pressure_i)*1e5# 1 bar = 1e5 Pa
			if self.units["Pi"].strip(" ") == "torr":
				self.pressure_i = float(self.pressure_i)*133.322# 1 torr = 133.322 Pa
			if self.units["Pi"].strip(" ") == "atm":
				self.pressure_i = float(self.pressure_i)*101325
			if self.units["Pi"].strip("	") == "Pa":
				self.pressure_i = float(self.pressure_i)
			
		if self.units["P"].strip(" ") == "mbar":
			self.pressure = float(self.pressure)*1e2
		if self.units["P"].strip(" ") == "bar":
			self.pressure = float(self.pressure)*1e5# 1 bar = 1e5 Pa
		if self.units["P"].strip(" ") == "torr":
			self.pressure = float(self.pressure)*133.322# 1 torr = 133.322 Pa
		if self.units["P"].strip(" ") == "atm":
			self.pressure = float(self.pressure)*101325
		if self.units["P"].strip("	") == "Pa":
			self.pressure = float(self.pressure)
		if "solver" not in self.add:
			self.add["solver"] = "FlameMaster"
		if "Thermodiffusion" not in self.add:
			self.add["Thermodiffusion"] = True
		if "ComputeWithRadiation" not in self.add:
			self.add["ComputeWithRadiation"] = True
		if "saveAll" not in self.add:
			self.add["saveAll"] = False
		if "BoundaryLayer" not in self.add:
			self.add["BoundaryLayer"] = False
		if "ign_delay_def" not in self.add:
			self.add["ign_delay_def"] = "OH"
		if "ign_cond" not in self.add:
			self.add["ign_cond"] = "max"
		if "specific_cond" not in self.add:
			self.add["specific_cond"] = "None;None"
		if "rcm_vt_profile" not in self.add:
			self.add["rcm_vt_profile"] = ""
		if "exp_conc" not in self.add:
			self.add["exp_conc"] = {}
			self.add["exp_conc"][float(self.temperature)] = ""
		else:
			if int(self.temperature) not in self.add["exp_conc"]:
				self.add["exp_conc"][float(self.temperature)] = "" 
		if "width" not in self.add:
			self.add["width"] = 0.014 
		if "ratio" not in self.add:
			self.add["ratio"] = 3
		if "slope" not in self.add:
			self.add["slope"] = 0.015
		if "curve" not in self.add:
			self.add["curve"] = 0.015
		if "volume_profile_type" not in self.add:
		    self.add["volume_profile_type"] = "csv_file"
		if "dt" not in self.add:
			self.add["dt"] = 1e-5
		if "loglevel" not in self.add:
			self.add["loglevel"] = 1		
		if "auto" not in self.add:
			self.add["auto"] = "True"
		if "width_bsf" not in self.add:
			self.add["width_bsf"] = 0.5 #m 
		if "slope_bsf" not in self.add:
			self.add["slope_bsf"] = 0.05
		if "phi" not in self.add:
			self.add["phi"] = "N/A"
		if "type" not in self.add:
			self.add["type"] = "N/A"
		if "flw_length" not in self.add:
			self.add["flw_length"] = 1 #m 
		if "flow_velocity" not in self.add:
			self.add["flow_velocity"] = 0.06  #m/s 
		if "crossSectionalArea" not in self.add:
			self.add["crossSectionalArea"] = 0.000314 # m**2,pi,r=0.01 m
		if "reactorVolume" not in self.add:
			self.add["reactorVolume"] = 0.000314 #m**3,area*l	
		if "residenceTime" not in self.add:
			self.add["residenceTime"] = 1
		if "time_step" not in self.add:
			self.add["time_step"] = 2000 #steps		
		if "flw_method" not in self.add:
			self.add["flw_method"] = "slope"
		if "flw_species" not in self.add:
			self.add["flw_species"] = "H2" #species 
		if "flw_limits" not in self.add:
			self.add["flw_limits"] = (40,60) #two points between which slope is to be found 
		if "anchor" not in self.add:
			self.add["anchor"] = 50 # anchor point 
		if "limit_units" not in self.add:
			self.add["limit_units"] = "percentage"
		if "heat" not in self.add:
			self.add["heat"] = ""
		else:
			self.add["heat"] = "HeatTransCoeff is {}".format(self.add["heat"])
		if "flow_rate" not in self.add:
			self.add["flow_rate"] = "0.06"
		if "flf_target" not in self.add:
			self.add["flf_target"] = "H" #m**3	
		if "flf_cond" not in self.add:
			self.add["flf_cond"] = "max" #m**3	
		if "T_amb" not in self.add:
			self.add["T_ambient"] = 298
		else:
			self.add["T_amb"] = "AmbientTemp is {}".format(self.add["T_ambient"])
		if "isIsotherm" not in self.add:
			self.add["isIsotherm"] = ""
		else:
			self.add["isIsotherm"] = "Isothermal is {}".format(self.add["isIsotherm"])
		if "transport_model" not in self.add:
			self.add["transport_model"] = "Multi"
			self.add["solve_bsf"] = "loglevel,refine_grid"
			self.add["group"] = "multi"
			self.add["description"] = "solution with multicomponent transport"
		else:
			if "Mix" in self.add["transport_model"]:
				self.add["solve_bsf"] = "loglevel,auto=True"
				self.add["group"] = "mix"
				self.add["description"] = "solution with mixture-averaged transport"
			
		# --- RMIP Phase-6 additions (Future.md pt. 3 + 6) -------------------
		# JSR defaults (residenceTime already defaulted above; these fill the
		# remaining keys the JSR templates in make_input_file.py read).
		# RMIP: unified simulation-stop-time key across all target types
		# (previously split across estimateTIG/maxSimulationTime/total_time,
		# which could silently collide if defaulted independently).
		if "MaxSimulationTime" not in self.add:
			if "JSR" in self.target or "Flf" in self.target or "PFR" in self.target or "Flw" in self.target:
				self.add["MaxSimulationTime"] = 10  # s
			elif "Tig" in str(self.target) and self.observed:
				# 2026-08-27: the old flat 0.4s default made AURORA integrate
				# ~5000x past the ignition event for µs-scale HTT targets
				# (pilot_5case_case0: tau_exp = 76 µs) -- ~5-6 min of solver
				# time AND license-seat hold per sample, all of it wasted on
				# post-ignition chemistry the tau extractor never reads.
				# Scale the stop time to the observed ignition delay instead:
				# 100x covers any perturbation within uncertainty bands
				# (self.observed is in µs: see the "ms" -> *1000 conversion
				# above), floored at 10 ms so µs-scale delays still give the
				# solver room past the ignition spike. Tig rows without an
				# observed value keep the legacy 0.4s.
				self.add["MaxSimulationTime"] = max(0.01, 100.0 * self.observed * 1e-6)
			else:  # RCM, Fls, Flf_a, etc.
				self.add["MaxSimulationTime"] = 0.4  # s
		if "species" not in self.add:
			self.add["species"] = list(self.species_dict.keys())[0] if self.species_dict else "CO2"
		# PFR (Plug-Flow Reactor) defaults — new target type, all three solvers.
		if "pfr_length" not in self.add:
			self.add["pfr_length"] = 0.1  # m
		if "pfr_diameter" not in self.add:
			self.add["pfr_diameter"] = 0.02  # m
		if "pfr_mdot" not in self.add:
			self.add["pfr_mdot"] = 1e-3  # kg/s
		if "pfr_n_points" not in self.add:
			self.add["pfr_n_points"] = 200
		# Reaction flux / ROP analysis toggle — applies to ANY target type
		# (Tig/Fls/Flf/JSR/PFR/RCM) across all three solvers:
		#   FlameMaster  -> uncomments 'ReactionFluxAnal is TRUE'
		#   CHEMKIN_PRO  -> job.postprocess(rop=True) + VARIABLE ROP ALL
		#   cantera      -> dumps gas.net_rates_of_progress to output/flux.json
		if "flux_analysis" not in self.add:
			self.add["flux_analysis"] = False
		if "flux_element" not in self.add:
			self.add["flux_element"] = "C"

		# Zero-RK solver defaults (Tig/JSR only — see ZERORK_GSA_PLAN.md; Zero-RK
		# has no PFR/flame reactor, those stay on cantera/CHEMKIN_PRO/FlameMaster
		# regardless of this target's solver= setting). Per the same rule as
		# every other solver's parameters above: these are the ONLY place
		# defaults are set — make_input_file.py's zerork branches read
		# target.add[...] exclusively, never hardcode a value themselves.
		if "zerork_install_dir" not in self.add:
			self.add["zerork_install_dir"] = "/home/krithika/zero-rk/build2/inst_dir"
		if "zerork_idt_temp_delta" not in self.add:
			self.add["zerork_idt_temp_delta"] = 400.0  # K, ignition-delay definition
		if "zerork_stop_time" not in self.add:
			self.add["zerork_stop_time"] = float(self.add["MaxSimulationTime"])  # s
		if "zerork_rtol" not in self.add:
			self.add["zerork_rtol"] = 1e-8
		if "zerork_atol" not in self.add:
			self.add["zerork_atol"] = 1e-20
		if "zerork_eps_lin" not in self.add:
			self.add["zerork_eps_lin"] = 0.05
		if "zerork_nonlinear_convergence_coefficient" not in self.add:
			self.add["zerork_nonlinear_convergence_coefficient"] = 0.05
		if "zerork_preconditioner_threshold" not in self.add:
			self.add["zerork_preconditioner_threshold"] = 2.048e-3
		# JSR (constVolumePSR) uses its own preconditioner threshold, not the
		# Tig/IDT (constVolumeWSR) one above -- Zero-RK's own example decks use
		# different tuned values per reactor type (1.0e-6 for PSR vs 2.048e-3
		# for WSR; see zerork_bridge.py's build_jsr_input default), so reusing
		# the Tig value here would be wrong, not just untuned.
		if "zerork_jsr_preconditioner_threshold" not in self.add:
			self.add["zerork_jsr_preconditioner_threshold"] = 1.0e-6
		if "zerork_n_mpi_ranks" not in self.add:
			self.add["zerork_n_mpi_ranks"] = 1

		# Zero-RK RCM defaults (variable_volume.x, see ZERORK_GSA_PLAN.md section
		# 1.1 table). RCM's own ignition-delay DEFINITION differs from the cantera/
		# CHEMKIN_PRO/FlameMaster RCM branches above (which compute tau as the time
		# between the first-stage and main temperature-rise peaks): Zero-RK's own
		# "Time at max dP/dt" summary line is used instead -- a real, documented
		# definitional difference (same class of gap as zerork_idt_temp_delta vs.
		# the flexible ign_delay_def/ign_cond options for Tig), not an oversight.
		if "zerork_rcm_volume_mult" not in self.add:
			self.add["zerork_rcm_volume_mult"] = 1.0e-6  # cm^3 -> m^3 (volume_profile files are cm^3)
		if "zerork_rcm_stroke_mult" not in self.add:
			self.add["zerork_rcm_stroke_mult"] = 1.0
		if "zerork_rcm_max_dt_internal" not in self.add:
			self.add["zerork_rcm_max_dt_internal"] = 1.0e-3  # s
		if "zerork_rcm_max_steps" not in self.add:
			self.add["zerork_rcm_max_steps"] = 1000000
		if "zerork_rcm_ref_temp" not in self.add:
			self.add["zerork_rcm_ref_temp"] = 1000.0  # K, ODE-state normalization only
		if "zerork_rcm_ref_moles" not in self.add:
			self.add["zerork_rcm_ref_moles"] = 0.25  # ODE-state normalization only
		if "zerork_rcm_max_order" not in self.add:
			self.add["zerork_rcm_max_order"] = 5  # max BDF order
		if "zerork_rcm_max_num_nonlin_iters" not in self.add:
			self.add["zerork_rcm_max_num_nonlin_iters"] = 3

		self.initialCond = self.getInitialCond()
		if "Flf" in self.target:
			#print(self.add)
			if "ExpTempFile" not in self.add:
				self.add["ExpTempFile"] = ""
				self.burner_temp = 0
			else:
				# Populating the expTempProfile
				# Using spline function in scipy
				read = open(str(self.add["ExpTempFile"]),'r').readlines()
				data = []
				for i in read:
					if "#" not in i:
						data.append(i)
				header = self.add["file_header"]
				x,y,xx,yy,self.burner_temp = self.getBurnerTemp(data,header)
				string = ""
				for ind,val in enumerate(xx):
					if self.add["solver"] == 'cantera':
						string+="{},{}".format(val,yy[ind])
					else:
						string+="{}\t{}".format(val,yy[ind])
					if ind<len(xx):
						string+="\n"
					else:
						continue
				
				for ind,i in enumerate(x):
					if float(i)>1.0:
						if self.add["solver"] == 'cantera':
							string+="{},{}".format(i,y[ind])
						else:
							string+="{}\t{}".format(i,y[ind])
						if ind<len(x):
							string+="\n"
						else:
							continue
					else:
						continue
				
				self.add["flf_grid"] = x[-1]
				file_dump = open(self.add["ExpTempFile"],"w+")
				file_dump.write(string)
				file_dump.close()
		    
		if "Fsl" in self.target:
			if key == "start_profile":
				self.startProfile_location = content	
		if "N/A" in self.phi:
			self.phi = self.add["phi"]
		
		
		# Compute equivalence ratio phi for multi-fuel mixtures
		# Assumptions:
		# - self.fuel_dict: {species: mole_fraction}
		# - self.oxidizer_dict: {species: mole_fraction} and contains O2
		# - self.stoichiometry: {fuel_species: nu_O2_per_mol_fuel}

		# 1) Get actual oxidizer (O2) mole fraction
		x_o2 = self.oxidizer_x
		if x_o2 <= 0.0:
			raise ValueError("Cannot compute phi: oxidizer_dict has no O2 or O2 mole fraction is zero.")

		# 2) Total fuel (mole fraction) and stoichiometric O2 required for the mixture
		total_fuel = 0.0
		o2_required = 0.0

		for fuel, x_fuel in self.fuel_dict.items():
			x_fuel = float(x_fuel)
			if x_fuel <= 0.0:
				continue

			nu_o2 = float(self.stoichiometry.get(fuel, 0.0))
			# For fuels not in the table, compute from atom balance via chem_data
			if nu_o2 <= 0.0 and _HAS_CHEM_DATA:
				nu_o2 = _compute_nu_o2(fuel)
			# Only include fuels for which a valid stoichiometric O2 coefficient exists
			if nu_o2 <= 0.0:
				continue

			total_fuel += x_fuel
			o2_required += nu_o2 * x_fuel

		if total_fuel <= 0.0:
			raise ValueError("Cannot compute phi: total_fuel is zero (or no fuels had valid stoichiometry).")
		if o2_required <= 0.0:
			raise ValueError("Cannot compute phi: computed O2_required is zero (check stoichiometry map).")

		# 3) Compute phi using your stated formula (shown explicitly)
		F_over_A_actual = total_fuel / x_o2
		F_over_A_stoich = total_fuel / o2_required
		self.phi = float(F_over_A_actual / F_over_A_stoich)

		#print(self.phi)
		# Equivalent compact form:
		# self.phi = float(o2_required / x_o2)
		
		if self.units["observed"] == "ms" and self.target == "Tig":
			self.observed = self.observed*1000
		#print(self.dataSet_id,self.phi)
		#raise AssertionError("Stop")
		#if "Fls" in self.target:
			#print("Target {} has phi of {}\n".format(self.uniqueID,self.phi))
		#self.equivalence_ratio = (F/A)act/(F/A)st
#This function goes through a file previously created by the "generate_target_value_tables" function in data management module 
#and extract target value information for each case and sort them based on the extend of perturbation.
	def getBurnerTemp(self,data,structure):
		x = []
		y = []
		for i in data:
			if len(i)>1:
				for ind,s in enumerate(structure):
					if "distance" in s:
						if self.add["solver"] == 'cantera':
							x.append(float(i.split(",")[ind].strip("\n")))
						else:
							x.append(float(i.split("	")[ind].strip("\n")))
					if "T" in s:
						if self.add["solver"] == 'cantera':
							y.append(float(i.split(",")[ind].strip("\n")))
						else:
							y.append(float(i.split("	")[ind].strip("\n")))	
		x = np.asarray(x)
		y = np.asarray(y)
		xo = 0
		order = 1
		s = InterpolatedUnivariateSpline(x, y, k=order)
		inter = 4
		xx = np.arange(x[0],1,0.02)
		ss = InterpolatedUnivariateSpline(x, y, k=inter)
		yy = ss(xx)
		yo = s(xo)
		return x,y,xx,yy,float(yo)
		
	def getInitialCond(self):
		species = self.species_dict
		str_syntax = "x->"
		dirichlet = ""
		for i in species:
			dirichlet += str_syntax+str(i)+"="+str(species[i])+"\n\t\t"
		return dirichlet
	
	def make_eta_lists(self,reaction_index,unsrt):
		xdata = []
		self.xdata = []		
		self.ydata = []
		home_dir = os.getcwd()
		os.chdir(self.case_index)
		eta_values = open(home_dir+'/Data/Simulations/sim_data_'+self.case_index+'.lst','r').readlines()
		for eta in eta_values:
			params = eta.split("\t")
			self.ydata.append(float(params[1]))
		
		#beta = open(home_dir+'/Data/Simulations/Generator_list_'+self.case_index+'.csv','r').readlines()		
		beta = open(home_dir+'/Data/Simulations/Beta_list_'+self.case_index+'.csv','r').readlines()		
		#beta = open(home_dir+'/Data/Simulations/NormalizedBeta_list_'+self.case_index+'.csv','r').readlines()	
		for i in beta[0:len(self.ydata)]:
			X = []			
			line = i.split(',')
			for j in line:
				if j != '\n':				
					X.append(float(j))
			self.xdata.append(X)	
		#print(np.shape(self.xdata))
		os.chdir(home_dir)			

