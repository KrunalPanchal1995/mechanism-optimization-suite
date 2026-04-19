#python default modules
import json
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
import os, sys, re, threading, subprocess, time
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
import combustion_target_class
import Uncertainty as uncertainty
from MechanismParser import Parser
import simulation_manager2_0 as simulator
from MechManipulator2_0 import Manipulator
from Target2CSV import Exporter as Target2CSV
from VisualAid import TPhiPlotter, ArrheniusPlotter
from OptimizationTool import OptimizationTool as Optimizer


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

################################################
# Open the input file and check for arguements #
################################################

if len(sys.argv) > 1:
	input_file = open(sys.argv[1],'r')
	optInputs = yaml.safe_load(input_file)
	print("Input file found\n")
else:
	print("Please enter a valid input file name as arguement. \n For details of preparing the input file, please see the UserManual\n\nProgram exiting")
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
parallel_threads = dataCounts[countThreads]
targets_count = int(dataCounts["targets_count"])
rps_order = stats_[order]
PRS_type = stats_["PRS_type"]
print("\nParallel threads are {}".format(parallel_threads))
targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())

####################################################
##  Unloading the target data	  	           ##
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
print(f"Case directory: (case_dir)")
print("\n\nOptimization targets identified.\n")

# SAVING THE TARGETS IN A CSV FILE
df = Target2CSV(target_list).to_dataframe()
df.to_csv("targets.csv", index=False)
print("\t Targets exported to targets.csv\n\n")

#Plotting the T-P-phi space
print("\t Plotting the T-P-phi space...")
TPhiPlotter(df).plot_t_p_phi(color="Phi", save_path="T-P-phi_space.png", show=True)
print("T-P-phi space plot saved as T-P-phi_space.png\n\n")
print("\n\nOptimization targets identified.\nStarted the MUQ process.......\n")


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
	print("Uncertainty analysis finished")

else:
	# Load the object from the file
	with open('unsrt.pkl', 'rb') as file_:
		unsrt_data = pickle.load(file_)
	print("Uncertainty analysis already finished")

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
print("\t"+f"{manipulationDict['activeParameters']}")
print("\n################################################\n\n")


"""
For Partial PRS (p-PRS), we need to do a second level of sensitivity analysis:	
	
	sensitivity_analysis: a dictionary that stores the sensitivity coefficients of all the active
			      parameters (selected in the first set of sensitivity analysis) reaction wise
			      as in unsrt_data dictionary
"""

if PRS_type == "Partial":
	####################################################
	## Sensitivity Analysis of the selected reactions ##
	####################################################
	
	print("\nPartial Polynomial Response Surface is choosen as a Solution Mapping Technique\n\n\tStarting the Sensitivity Analysis: \n\t\tKindly be Patient\n")
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
		else:
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
		else:
			status_ = "Pending"
			print("\n\t3-Parameter Sensitivity analysis is already over!!\n")
			with open('sens_3p_parameters.pkl', 'rb') as file_:
				sensitivity_analysis = pickle.load(file_)
	
	partialPRS_Object = []
	selected_params_dict = {}
	design_matrix_dict = {}
	yaml_loc_dict = {}
	no_of_sim = {}
	print("\n################################################\n###  Starting to generate Design Matrix      ###\n###  for all targets                        ###\n################################################\n\n")
	for case in case_dir:
		PPRS_system = P_PRS.PartialPRS(sensitivity_analysis[str(case)],unsrt_data,optInputs,target_list,str(case),activeParameters,design_type,status=status_)
		yaml_loc,design_matrix,selected_params = PPRS_system.partial_DesignMatrix()
		partialPRS_Object.append(PPRS_system)
		no_of_sim[case] = int(PPRS_system.no_of_sim)
		yaml_loc_dict[case] = yaml_loc
		design_matrix_dict[case] = design_matrix
		selected_params_dict[case] = selected_params
##################################################################
##  Use the unsrt data to sample the Arrhenius curves           ##
##  MUQ-SAC: Method of Uncertainty Quantification and           ##
##	Sampling of Arrhenius Curves                                ##
##################################################################
	print("\n\n Generating the perturbed mechanism files for partial-PRS is finished!!\n")
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
		print(f"\n\n\tNo. of Simulations required: {getSim(len(manipulationDict['activeParameters']),design)}\n")
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

	if "Perturbed_Mech" not in os.listdir():
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
print("\n\n Generating the perturbed mechanism files for full-PRS is finished!!\n")
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
	print("Progress file detected")
	progress = open(optDir+"/progress",'r').readlines()
	FlameMaster_Execution_location = []
	with open(optDir+"/locations") as infile:
		for line in infile:
			FlameMaster_Execution_location.append(line)
	#FlameMaster_Execution_location = open(optDir+"/locations",'r').readlines()
	#missing_location = data_management.find_missing_location(FlameMaster_Execution_location, progress)

#############################################
##     Extracting simulation results       ##
##       (The module if validated          ##
#############################################
temp_sim_opt = {}
for case in case_dir:	
	os.chdir("Data/Simulations")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		ETA = [float(i.split("\t")[1]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		temp_sim_opt[str(case)] = ETA
		os.chdir(optDir)
		#print(ETA)
		#raise AssertionError("Generating ETA list for all cases")	
	else:
		os.chdir(optDir)
		#print(os.getcwd())
		os.chdir("case-"+str(case))	
		data_sheet,failed_sim, ETA = data_management.generate_target_value_tables(FlameMaster_Execution_location, target_list, case, fuel,input_=optInputs)
		#print(data_sheet)
		#raise AssertionError("!STOP")
		temp_sim_opt[str(case)] = ETA
		f = open('../Data/Simulations/sim_data_case-'+str(case)+'.lst','w').write(data_sheet)
		g = open('../Data/Simulations/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim)
		#f.write(data_sheet)
		#f.close()
		os.chdir(optDir)

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

ResponseSurfaces = {}
selected_PRS = {}
for case_index,case in enumerate(temp_sim_opt):
	yData = np.asarray(temp_sim_opt[case]).flatten()
	xData = np.asarray(design_matrix_dict[case_index])#[0:no_of_sim[case_index]]
	#print(np.shape(xData))
	#print(np.shape(yData))
	#raise AssertionError("Stop!!")
	
	xTrain,xTest,yTrain,yTest = train_test_split(xData,yData,
									random_state=104, 
                                	test_size=0.1, 
                                   	shuffle=True)
	#print(np.shape(xTest))
	#print(np.shape(yTrain))
	Response = PRS.ResponseSurface(xTrain,yTrain,case,case_index,prs_type=PRS_type,selected_params=selected_params_dict[case_index])
	Response.create_response_surface()
	Response.test(xTest,yTest)
	Response.plot(case_index)
	#Response.DoStats_Analysis() #Generates stastical analysis report
	#print(Response.case)
	ResponseSurfaces[case_index] = Response
	#print(Response.selection)
	del xTrain,xTest,yTrain,yTest
#raise AssertionError("The Target class, Uncertainty class, Design Matrix and Simulations and Response surface")
os.chdir("..")

##################################################
##        Optimization Procedure                ##
##   Inputs: Target list and Response Surfaces  ##
##################################################

from VisualAid import PostOptPlotter

if "solution_zeta.save" not in os.listdir():

    optimizer_obj  = Optimizer(target_list)
    opt, opt_zeta, posterior_cov = optimizer_obj.run_optimization_with_selected_PRS(
        unsrt_data, ResponseSurfaces, optInputs)

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
