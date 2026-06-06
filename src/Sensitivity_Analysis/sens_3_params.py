#python default modules
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import minimize
import os, sys, re, threading, subprocess, time
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import PolynomialFeatures
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
from scipy.linalg import block_diag
from scipy import optimize as spopt
import json
import multiprocessing
import concurrent.futures
import asyncio
import pickle
#import yaml
#import ruamel.yaml as yaml
try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
import pandas as pd
import yaml
sys.path.append('/parallel_yaml_writer.so')
import parallel_yaml_writer

####################################
##  Importing the sampling file   ##
##                                ##
####################################
import reaction_selection as rs
from MechManipulator2_0 import Manipulator
#program specific modules
from copy import deepcopy
from MechanismParser import Parser

import combustion_target_class
import data_management
#import data_management as dm
import simulation_manager2_0 as simulator
import Uncertainty as uncertainty
from mpire import WorkerPool
import DesignMatrix2_0 as DM
import ResponseSurface as PRS
import VisualAid as VA

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

#!!!!!!! GET MECHANISM FILE , number of targets  from the input file !!!!!!!!!
iFile = str(os.getcwd())+"/"+str(sys.argv[1])
dataCounts = optInputs[count]
binLoc = optInputs["Bin"]
inputs = optInputs["Inputs"]
locations = optInputs["Locations"]
startProfile_location = optInputs[startProfile]
stats_ = optInputs["Stats"]
global A_fact_samples

A_fact_samples = stats_["Sampling_of_PRS"]
if "sensitive_parameters" not in stats_:
	stats_["sensitive_parameters"] = "Principle_SubMatrix"
	optInputs["Stats"]["sensitive_parameters"] = "Principle_SubMatrix"
if "Arrhenius_Selection_Type" not in stats_:
	stats_["Arrhenius_Selection_Type"] = "some"
	optInputs["Stats"]["Arrhenius_Selection_Type"] = "some"
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
#######################READ TARGET FILE ###################

print("\nParallel threads are {}".format(parallel_threads))
targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())

####################################################
##  Unloading the target data	  	               ##
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
	string_target+=f"{t.dataSet_id}|{t.target}|{t.species_dict}|{t.temperature}|{t.pressure}|{t.phi}|{t.observed}|{t.std_dvtn}\n"
	c_index +=1
	target_list.append(t)
case_dir = range(0,len(target_list))
print(case_dir)
print("\n\nOptimization targets identified.\nStarted the MUQ process.......\n")

############################################
##  Uncertainty Quantification            ##
##  					                 ##
############################################

if "unsrt.pkl" not in os.listdir():
	UncertDataSet = uncertainty.uncertaintyData(locations,binLoc);
	############################################
	##   Get unsrt data from UncertDataSet    ##
	############################################

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

with open(mech_file_location,'r') as file_:
	yaml_mech = file_.read()

mechanism = yaml.safe_load(yaml_mech)
species = mechanism['phases'][0]["species"]
species_data = mechanism["species"]
reactions = mechanism["reactions"]
selected_reactions = [rxn for rxn in unsrt_data]
reaction_dict = rs.reaction_index(selected_reactions,reactions)

rxn_type = rs.getRxnType(mechanism,selected_reactions)
string_f = ""
string_g = ""
index_dict = {}
for index in reaction_dict:
	index_dict[reaction_dict[index]] = index
	string_f+=f"{index}\t{reaction_dict[index]}\n"

for rxn in rxn_type:
	string_g+=f"{rxn}\t{rxn_type[rxn]}\n"
f = open("Reaction_dict.txt","w").write(string_f)
g = open("Reaction_type.txt","w").write(string_g)
rxn_dict = {}
rxn_dict["reaction"] = reaction_dict
rxn_dict["type"] = rxn_type
rxn_dict["data"] = rs.getRxnDetails(mechanism,selected_reactions)
string_reaction = ""
for index in reaction_dict:
	string_reaction+=f"{index}\t{reaction_dict[index]}\n"
g = open("selected_rxn.txt","+w").write(string_reaction)


######################################################################
##  CREATING A DICTIONARY CONTAINING ALL THE DATA FROM UNSRT CLASS  ##
##													   ##
######################################################################
rxn_list = []
activeParameters = []

for rxn in unsrt_data:
	rxn_list.append(rxn)
	activeParameters.extend(unsrt_data[rxn].activeParameters)
ap = len(activeParameters)
print("Active Parameters:", activeParameters)
#raise AssertionError("Stop")

############################################
### Plotting all the samples taken for    ##
### Sensitivity analysis                  ##
############################################
global T, theta
T = np.linspace(300,2500,100)
theta = np.array([T/T,np.log(T),-1/T])

def getUnsrtLimit(Po,P_u,P_l):
	K_o = np.asarray([ i.dot(Po) for i in theta.T ]).flatten()
	K_u = np.asarray([ i.dot(P_u) for i in theta.T ]).flatten()
	K_l = np.asarray([ i.dot(P_l) for i in theta.T ]).flatten()
	return K_o,K_u,K_l
	
def getKappa(P):
	K = np.asarray([ i.dot(P) for i in theta.T ]).flatten()
	return np.exp(K)

#print((getKappa(P_multiply_n[0])[0])/(getKappa(P_multiply_n[0])[0]-getKappa(P_nominal_list[0])[0] ))
#raise AssertionError("Stop")

#for rxn in unsrt_data:
#	VA.ArrheniusPlotter(unsrt_data,rxn).plot_uncertainty_limits(location="Plots/UQ")
#	VA.ArrheniusPlotter(unsrt_data,rxn).plot_perturbed_Arrhenius_parameters(location="Plots/SA")
	
#########################################
###    Creating Design Matrix for    ####
###    sensitivity analysis          ####
#########################################

"""
For sensitivity analysis we create two design matrix
	- for one, we multiply all reactions by a factor of 2
	- for second, we devide all reactions by a factor of 0.5
"""
#########################################
###    Creating Design Matrix for    ####
###    sensitivity analysis          ####
#########################################
select_param_a = np.asarray([1,0,0]*len(rxn_list))
select_param_n = np.asarray([0,1,0]*len(rxn_list))
select_param_ea = np.asarray([0,0,1]*len(rxn_list))
perturb_fact = 0.1

"""
For sensitivity analysis we create two design matrix
	- for one, we multiply all reactions by a factor of 2
	- for second, we devide all reactions by a factor of 0.5
"""
if "DesignMatrix_x0_3P.csv" not in os.listdir():
	design_matrix_x0_3P = DM.DesignMatrix(unsrt_data,design_type,1,ind=len(activeParameters)).getNominal_samples()
	s =""
	for row in design_matrix_x0_3P:
		for element in row:
			s+=f"{element},"
		s+="\n"
	ff = open('DesignMatrix_x0_3P.csv','w').write(s)
else:
	design_matrix_file = open("DesignMatrix_x0_3P.csv").readlines()
	design_matrix_x0_3P = []
	for row in design_matrix_file:
		design_matrix_x0_3P.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])

if "DesignMatrix_A.csv" not in os.listdir():
	selection_matrix_A,design_matrix_A = DM.DesignMatrix(unsrt_data,design_type,len(reaction_dict)).getSA_3P_samples(select_param_a,param_type="A",perturb_fact=perturb_fact)
	s =""
	for row in design_matrix_A:
		for element in row:
			s+=f"{element},"
		s+="\n"
	ff = open('DesignMatrix_A.csv','w').write(s)
else:
	design_matrix_file = open("DesignMatrix_A.csv").readlines()
	selection_matrix_file = open("pSelectionMatrix_A.csv").readlines()
	selection_matrix_A = []
	design_matrix_A = []
	for row in design_matrix_file:
		design_matrix_A.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
	for row in selection_matrix_file:
		selection_matrix_A.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])


if "DesignMatrix_n.csv" not in os.listdir():
	selection_matrix_n,design_matrix_n = DM.DesignMatrix(unsrt_data,design_type,len(reaction_dict)).getSA_3P_samples(select_param_n,param_type="n",perturb_fact=perturb_fact)
	s =""
	for row in design_matrix_n:
		for element in row:
			s+=f"{element},"
		s+="\n"
	ff = open('DesignMatrix_n.csv','w').write(s)
else:
	design_matrix_file = open("DesignMatrix_n.csv").readlines()
	selection_matrix_file = open("pSelectionMatrix_n.csv").readlines()
	selection_matrix_n = []
	design_matrix_n = []
	for row in design_matrix_file:
		design_matrix_n.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
	for row in selection_matrix_file:
		selection_matrix_n.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])



if "DesignMatrix_Ea.csv" not in os.listdir():
	selection_matrix_Ea,design_matrix_Ea = DM.DesignMatrix(unsrt_data,design_type,len(reaction_dict)).getSA_3P_samples(select_param_ea,param_type="Ea",perturb_fact=perturb_fact)
	s =""
	for row in design_matrix_Ea:
		for element in row:
			s+=f"{element},"
		s+="\n"
	ff = open('DesignMatrix_Ea.csv','w').write(s)
else:
	design_matrix_file = open("DesignMatrix_Ea.csv").readlines()
	selection_matrix_file = open("pSelectionMatrix_Ea.csv").readlines()
	selection_matrix_Ea = []
	design_matrix_Ea = []
	for row in design_matrix_file:
		design_matrix_Ea.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
	for row in selection_matrix_file:
		selection_matrix_Ea.append([float(ele) for ele in row.strip("\n").strip(",").split(",")])
		
VA.DesignMatrixPlotter(unsrt_data).plot_dm_samples()

##############################################################################
## Extract delta_A, delta_n, delta_Ea from the SA Design Matrices           ##
##                                                                          ##
## Formula:  P = P_o + L_r @ zr                                             ##
##                                                                          ##
## zr   — diagonal element of p_design_matrix (1×1 scalar):                ##
##         For reaction i, the (i,i) element of design_matrix_{A|n|Ea}      ##
##         is the class-A ζ-component for that parameter.                   ##
##         (The DM is diagonal because each row perturbs exactly one rxn.)  ##
##                                                                          ##
## L_r  — reduced Cholesky column for the selected parameter:               ##
##         L_r_A  = cholskyDeCorrelateMat[:, 0]   (A-factor column)        ##
##         L_r_n  = cholskyDeCorrelateMat[:, 1]   (n column)               ##
##         L_r_Ea = cholskyDeCorrelateMat[:, 2]   (Ea column)              ##
##         This projects the scalar zr back into full parameter space.      ##
##                                                                          ##
## delta = L_r * zr  (shape (3,): [Δln(A), Δn, ΔEa/R])                    ##
##############################################################################

design_matrix_A  = np.asarray(design_matrix_A,  dtype=float)
design_matrix_n  = np.asarray(design_matrix_n,  dtype=float)
design_matrix_Ea = np.asarray(design_matrix_Ea, dtype=float)

delta_dict = {}  # rxn → {zr_A, zr_n, zr_Ea, L_r_A, L_r_n, L_r_Ea,
                 #         delta_A, delta_n, delta_Ea, P_o, P_A, P_n, P_Ea}

for i, rxn in enumerate(rxn_list):
    P_o = np.asarray(unsrt_data[rxn].nominal, dtype=float).flatten()   # (3,)
    L = unsrt_data[rxn].cov
    # ── L_r via principal submatrix of Σ=LL^T, then re-Cholesky ─────────────
    # param index: 0=ln(A), 1=n, 2=Ea/R
    # Returns (Sigma_r, L_r); L_r is (1×1) for single-param selection.
    _, L_r_A  = unsrt_data[rxn].get_reduced_cholesky((0,))   # (1,1): sqrt(Σ[0,0])
    _, L_r_n  = unsrt_data[rxn].get_reduced_cholesky((1,))   # (1,1): sqrt(Σ[1,1])
    _, L_r_Ea = unsrt_data[rxn].get_reduced_cholesky((2,))   # (1,1): sqrt(Σ[2,2])

    # ── zr: diagonal scalar of each SA design matrix, reshaped to (1,1) ──────
    # design_matrix_{A|n|Ea} is (N_rxns × N_rxns) diagonal:
    #   row i has only column i non-zero = class-A ζ for rxn i's parameter.
    zr_A  = np.array([[design_matrix_A [i, i]]])   # (1,1)
    zr_n  = np.array([[design_matrix_n [i, i]]])   # (1,1)
    zr_Ea = np.array([[design_matrix_Ea[i, i]]])   # (1,1)

    # ── delta = L_r @ zr  (scalar result for m=1, via matrix multiply) ───────
    # L_r (1,1) @ zr (1,1) → (1,1); flatten to scalar.
    delta_A  = float((L_r_A  @ zr_A ).item())   # scalar: Δln(A)
    delta_n  = float((L_r_n  @ zr_n ).item())   # scalar: Δn
    delta_Ea = float((L_r_Ea @ zr_Ea).item())   # scalar: ΔEa/R

    # ── embed deltas into full parameter space [ln(A), n, Ea/R] ──────────────
    # Only the selected index is perturbed; all other indices stay at P_o.
    P_A  = P_o + np.array([delta_A,  0.0,     0.0     ])   # A-factor perturbed
    P_n  = P_o + np.array([0.0,      delta_n,  0.0     ])  # n perturbed
    P_Ea = P_o + np.array([0.0,      0.0,      delta_Ea])  # Ea perturbed

    delta_dict[rxn] = {
        # ── raw zr scalars (1×1 diagonal from DM) ────────────────────────────
        "zr_A"    : float(zr_A.item()),
        "zr_n"    : float(zr_n.item()),
        "zr_Ea"   : float(zr_Ea.item()),
        # ── reduced Cholesky matrices (1×1 each) ──────────────────────────────
        "L"	   : L,
        "L_r_A"   : L_r_A,
        "L_r_n"   : L_r_n,
        "L_r_Ea"  : L_r_Ea,
        # ── reduced-space perturbation scalars (delta = L_r @ zr) ─────────────
        "delta_A" : delta_A,    # scalar Δln(A)
        "delta_n" : delta_n,    # scalar Δn
        "delta_Ea": delta_Ea,   # scalar ΔEa/R
        # ── nominal and full-space perturbed parameter vectors ────────────────
        "P_o"     : P_o,    # nominal [ln(A), n, Ea/R]
        "P_A"     : P_A,    # P_o with ln(A) shifted by delta_A
        "P_n"     : P_n,    # P_o with n     shifted by delta_n
        "P_Ea"    : P_Ea,   # P_o with Ea/R  shifted by delta_Ea
    }

print(f"\n[delta_dict] Deltas extracted for {len(delta_dict)} reactions.")
_r = rxn_list[1]
print(f"  Sample reaction : {_r}")
print(f"  P_o             : {delta_dict[_r]['P_o']}")
print(f"  L               : {delta_dict[_r]['L']}")
print(f"  L_r_A  (1×1)    : {delta_dict[_r]['L_r_A']}   zr_A  = {delta_dict[_r]['zr_A']:.6f}  →  delta_A  = {delta_dict[_r]['delta_A']:.6f}")
print(f"  L_r_n  (1×1)    : {delta_dict[_r]['L_r_n']}   zr_n  = {delta_dict[_r]['zr_n']:.6f}  →  delta_n  = {delta_dict[_r]['delta_n']:.6f}")
print(f"  L_r_Ea (1×1)    : {delta_dict[_r]['L_r_Ea']}  zr_Ea = {delta_dict[_r]['zr_Ea']:.6f}  →  delta_Ea = {delta_dict[_r]['delta_Ea']:.6f}")
#raise AssertionError("Stop")
#########################################
###   Generating YAML files for      ####
###    sensitivity analysis          ####
#########################################

yaml_loc_nominal = []
yaml_loc_nominal.append(mech_file_location)
SSM = simulator.SM(target_list,optInputs,unsrt_data,design_matrix_A)
if "Perturbed_Mech_SA_3P_BruteForce" not in os.listdir():
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/A_factor")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/n")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/Ea")
	print("\nPerturbing the Mechanism files for 3-Parameter brute force sensitivity analysis\n")
	chunk_size = 500
	params_DM_A = [design_matrix_A[i:i+chunk_size] for i in range(0, len(design_matrix_A), chunk_size)]
	params_DM_n = [design_matrix_n[i:i+chunk_size] for i in range(0, len(design_matrix_n), chunk_size)]
	params_DM_Ea = [design_matrix_Ea[i:i+chunk_size] for i in range(0, len(design_matrix_Ea), chunk_size)]
	selection_params_A = [selection_matrix_A[i:i+chunk_size] for i in range(0, len(selection_matrix_A), chunk_size)]
	selection_params_n = [selection_matrix_n[i:i+chunk_size] for i in range(0, len(selection_matrix_n), chunk_size)]
	selection_params_Ea = [selection_matrix_Ea[i:i+chunk_size] for i in range(0, len(selection_matrix_Ea), chunk_size)]
	count = 0
	yaml_loc_A = []
	yaml_loc_n = []
	yaml_loc_Ea = []
	for i,params in enumerate(params_DM_A):
		yaml_list = SSM.getYAML_List(params,selection_params_A[i])
		#yaml_loc = []
		location_mech = []
		index_list = []
		for i,dict_ in enumerate(yaml_list):
			index_list.append(str(count+i))
			location_mech.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor")
			yaml_loc_A.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor/mechanism_"+str(count+i)+".yaml")
		count+=len(yaml_list)
		#gen_flag = False
		#SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		print(f"\n\tGenerated {count} files!!\n")
	count = 0
	for i,params in enumerate(params_DM_n):
		yaml_list = SSM.getYAML_List(params,selection_params_n[i])
		#yaml_loc = []
		location_mech = []
		index_list = []
		for i,dict_ in enumerate(yaml_list):
			index_list.append(str(count+i))
			location_mech.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n")
			yaml_loc_n.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n/mechanism_"+str(count+i)+".yaml")
		count+=len(yaml_list)
		#gen_flag = False
		#SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		print(f"\n\tGenerated {count} files!!\n")
	count = 0
	for i,params in enumerate(params_DM_Ea):
		yaml_list = SSM.getYAML_List(params,selection_params_Ea[i])
		#yaml_loc = []
		location_mech = []
		index_list = []
		for i,dict_ in enumerate(yaml_list):
			index_list.append(str(count+i))
			location_mech.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea")
			yaml_loc_Ea.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea/mechanism_"+str(count+i)+".yaml")
		count+=len(yaml_list)
		#gen_flag = False
		#SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		SSM.getPerturbedMechLocation(yaml_list,location_mech,index_list)
		print(f"\n\tGenerated {count} files!!\n")
	print("\n\tGenerated the YAML files required for simulations!!\n")
else:
	print("\nYAML files for 3-Parameter Bruteforce analysis is already generated!!")
	yaml_loc_A = []
	yaml_loc_n = []
	yaml_loc_Ea = []
	location_mech_A = []
	location_mech_n = []
	location_mech_Ea = []
	index_list_A = []
	index_list_n = []
	index_list_Ea = []
	for i,sample in enumerate(design_matrix_A):
		index_list_A.append(i)
		location_mech_A.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor")
		yaml_loc_A.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor/mechanism_"+str(i)+".yaml")
	for i,sample in enumerate(design_matrix_n):
		index_list_n.append(i)
		location_mech_n.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n")
		yaml_loc_n.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n/mechanism_"+str(i)+".yaml")
	for i,sample in enumerate(design_matrix_Ea):
		index_list_Ea.append(i)
		location_mech_Ea.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea")
		yaml_loc_Ea.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea/mechanism_"+str(i)+".yaml")


#########################################
###    Creating case dict		   ###
#########################################
yaml_loc_nominal_case = {}
yaml_loc_A_case = {}
yaml_loc_n_case = {}
yaml_loc_Ea_case = {}

for case in case_dir:
	yaml_loc_nominal_case[case] = yaml_loc_nominal
	yaml_loc_A_case[case] = yaml_loc_A
	yaml_loc_n_case[case] = yaml_loc_n
	yaml_loc_Ea_case[case] = yaml_loc_Ea				
#########################################
###    Creating Simulation Field     ####
#########################################

print("\t\t#########################################\n\t\t###    Creating Simulation Field     ####\n\t\t#########################################")

if "SA_3P" not in os.listdir():
	os.mkdir("SA_3P")
	os.chdir("SA_3P")
	os.mkdir("multiply_A")
	os.mkdir("multiply_n")
	os.mkdir("multiply_Ea")
	os.mkdir("multiply")# we multiply reactions by 2 in this folder
	os.mkdir("divide")# we divide the reactions by 2 in this folder
	os.mkdir("nominal")
	os.mkdir("Data")
	os.chdir("Data")
	os.mkdir("Simulations")
	os.chdir("Simulations")
	os.mkdir("Multiply")
	os.mkdir("Multiply_A")
	os.mkdir("Multiply_n")
	os.mkdir("Multiply_Ea")
	os.mkdir("Divide")
	os.mkdir("Nominal")
	os.chdir("..")
	os.mkdir("ResponseSurface")
	os.chdir("..")
	os.chdir("multiply_A")
	SADir = os.getcwd()
else:
	os.chdir("SA_3P")
	os.chdir("multiply_A")
	SADir = os.getcwd()

################################################################################
#### Multiplying the A-factor of reactions within the uncertainty limits     ###
################################################################################
print("\n\t\t#####################################################\n\t\t#### Multiplying the A-factor of all reactions for 3-Params UQ   ###\n\t\t#####################################################")


if os.path.isfile("progress") == False:
	FlameMaster_Execution_location_A = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_A).make_dir_in_parallel(yaml_loc_A_case)
	
else:
	print("\t\tProgress file detected")
	progress = open(SADir+"/progress",'r').readlines()
	FlameMaster_Execution_location_A = []
	with open(SADir+"/locations") as infile:
		for line in infile:
			FlameMaster_Execution_location_A.append(line)

os.chdir("..")
os.chdir("multiply_n")
SADir = os.getcwd()

####################################################################################
#### Multiplying the "n" parameter of reactions within the uncertainty limits    ###
####################################################################################
print("\n\t\t#####################################################\n\t\t#### Multiplying the n of all reactions for 3-Params UQ   ###\n\t\t#####################################################")


if os.path.isfile("progress") == False:
	FlameMaster_Execution_location_n = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_n).make_dir_in_parallel(yaml_loc_n_case)
	
else:
	print("\t\tProgress file detected")
	progress = open(SADir+"/progress",'r').readlines()
	FlameMaster_Execution_location_n = []
	with open(SADir+"/locations") as infile:
		for line in infile:
			FlameMaster_Execution_location_n.append(line)

os.chdir("..")
os.chdir("multiply_Ea")
SADir = os.getcwd()

#######################################################################################
#### Multiplying the "Ea" paramteter of reactions within the uncertainty limits     ###
#######################################################################################
print("\n\t\t#####################################################\n\t\t#### Multiplying the Ea of all reactions for 3-Params UQ   ###\n\t\t#####################################################")


if os.path.isfile("progress") == False:
	FlameMaster_Execution_location_Ea = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_Ea).make_dir_in_parallel(yaml_loc_Ea_case)
	
else:
	print("\t\tProgress file detected")
	progress = open(SADir+"/progress",'r').readlines()
	FlameMaster_Execution_location_Ea = []
	with open(SADir+"/locations") as infile:
		for line in infile:
			FlameMaster_Execution_location_Ea.append(line)


os.chdir("..")
os.chdir("nominal")
SADir = os.getcwd()

#############################
#### Nominal simulations  ###
#############################
print("\n\t\t#############################\n\t\t#### Nominal simulations ###\n\t\t#############################")

if os.path.isfile("progress") == False:
	FlameMaster_Execution_location_x0 = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_x0_3P).make_dir_in_parallel(yaml_loc_nominal_case)
	
else:
	print("\t\tProgress file detected")
	progress = open(SADir+"/progress",'r').readlines()
	FlameMaster_Execution_location_x0 = []
	with open(SADir+"/locations") as infile:
		for line in infile:
			FlameMaster_Execution_location_x0.append(line)
os.chdir("..")
SAdir = os.getcwd()

########################################################
#### collecting sensitivity data from the simulation ###
########################################################
##################################
##### From Nominal Folder #######
##################################
temp_sim_opt_x0 = {}
for case in case_dir:	
	os.chdir("Data/Simulations/Nominal")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		ETA_x0 = [float(i.split("\t")[1]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		
		folderName_x0 = [float(i.split("\t")[0]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		temp_sim_opt_x0[str(case)] = {}
		temp_sim_opt_x0[str(case)]["ETA"] = ETA_x0
		temp_sim_opt_x0[str(case)]["index"] = folderName_x0
		os.chdir(SAdir)
		#print(ETA)
		#raise AssertionError("Generating ETA list for all cases")	
	else:
		os.chdir(SAdir)
		os.chdir("nominal/case-"+str(case))	
		data_sheet_x0,failed_sim_x0,index_x0, ETA_x0,eta_x0 = data_management.generate_SA_target_value_tables(FlameMaster_Execution_location_x0, target_list, case, fuel)
		#print(data_sheet)
		#raise AssertionError("!STOP")
		temp_sim_opt_x0[str(case)] = {}
		temp_sim_opt_x0[str(case)]["ETA"] = ETA_x0
		temp_sim_opt_x0[str(case)]["index"] = index_x0
		f = open('../../Data/Simulations/Nominal/sim_data_case-'+str(case)+'.lst','w').write(data_sheet_x0)
		g = open('../../Data/Simulations/Nominal/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim_x0)
		#f.write(data_sheet)
		#f.close()
		os.chdir(SAdir)

####################################
##### From Multiply_A Folder #######
####################################
temp_sim_opt_A = {}
for case in case_dir:	
	os.chdir("Data/Simulations/Multiply_A")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		try:
			ETA_A = [float(i.split("\t")[1]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		except ValueError:
			print("Case-"+str(case)+" has ValueError")

		folderName_A = [float(i.split("\t")[0]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		temp_sim_opt_A[str(case)] = {}
		temp_sim_opt_A[str(case)]["ETA"] = ETA_A
		temp_sim_opt_A[str(case)]["index"] = folderName_A
		os.chdir(SAdir)
		#print(ETA)
		#raise AssertionError("Generating ETA list for all cases")	
	else:
		os.chdir(SAdir)
		os.chdir("multiply_A/case-"+str(case))	
		data_sheet_A,failed_sim_A,index_A, ETA_A,eta_A = data_management.generate_SA_target_value_tables(FlameMaster_Execution_location_A, target_list, case, fuel)
		#print(data_sheet)
		#raise AssertionError("!STOP")
		temp_sim_opt_A[str(case)] = {}
		temp_sim_opt_A[str(case)]["ETA"] = ETA_A
		temp_sim_opt_A[str(case)]["index"] = index_A
		f = open('../../Data/Simulations/Multiply_A/sim_data_case-'+str(case)+'.lst','w').write(data_sheet_A)
		g = open('../../Data/Simulations/Multiply_A/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim_A)
		#f.write(data_sheet)
		#f.close()
		os.chdir(SAdir)

####################################
##### From Multiply_n Folder #######
####################################
temp_sim_opt_n = {}
for case in case_dir:	
	os.chdir("Data/Simulations/Multiply_n")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		try:
			ETA_n = [float(i.split("\t")[1]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		except ValueError:
			print("Case-"+str(case)+" has ValueError")

		folderName_n = [float(i.split("\t")[0]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		temp_sim_opt_n[str(case)] = {}
		temp_sim_opt_n[str(case)]["ETA"] = ETA_n
		temp_sim_opt_n[str(case)]["index"] = folderName_n
		os.chdir(SAdir)
		#print(ETA)
		#raise AssertionError("Generating ETA list for all cases")	
	else:
		os.chdir(SAdir)
		os.chdir("multiply_n/case-"+str(case))	
		data_sheet_n,failed_sim_n,index_n, ETA_n,eta_n = data_management.generate_SA_target_value_tables(FlameMaster_Execution_location_n, target_list, case, fuel)
		#print(data_sheet)
		#raise AssertionError("!STOP")
		temp_sim_opt_n[str(case)] = {}
		temp_sim_opt_n[str(case)]["ETA"] = ETA_n
		temp_sim_opt_n[str(case)]["index"] = index_n
		f = open('../../Data/Simulations/Multiply_n/sim_data_case-'+str(case)+'.lst','w').write(data_sheet_n)
		g = open('../../Data/Simulations/Multiply_n/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim_n)
		#f.write(data_sheet)
		#f.close()
		os.chdir(SAdir)

#####################################
##### From Multiply_Ea Folder #######
#####################################
temp_sim_opt_Ea = {}
for case in case_dir:	
	os.chdir("Data/Simulations/Multiply_Ea")
	if "sim_data_case-"+str(case)+".lst" in os.listdir():
		try:
			ETA_Ea = [float(i.split("\t")[1]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		except ValueError:
			print("Case-"+str(case)+" has ValueError")

		folderName_Ea = [float(i.split("\t")[0]) for i in open("sim_data_case-"+str(case)+".lst").readlines()]
		temp_sim_opt_Ea[str(case)] = {}
		temp_sim_opt_Ea[str(case)]["ETA"] = ETA_Ea
		temp_sim_opt_Ea[str(case)]["index"] = folderName_Ea
		os.chdir(SAdir)
		#print(ETA)
		#raise AssertionError("Generating ETA list for all cases")	
	else:
		os.chdir(SAdir)
		os.chdir("multiply_Ea/case-"+str(case))	
		data_sheet_Ea,failed_sim_Ea,index_Ea, ETA_Ea,eta_Ea = data_management.generate_SA_target_value_tables(FlameMaster_Execution_location_Ea, target_list, case, fuel)
		#print(data_sheet)
		#raise AssertionError("!STOP")
		temp_sim_opt_Ea[str(case)] = {}
		temp_sim_opt_Ea[str(case)]["ETA"] = ETA_Ea
		temp_sim_opt_Ea[str(case)]["index"] = index_Ea
		f = open('../../Data/Simulations/Multiply_Ea/sim_data_case-'+str(case)+'.lst','w').write(data_sheet_Ea)
		g = open('../../Data/Simulations/Multiply_Ea/failed_sim_data_case-'+str(case)+'.lst','w').write(failed_sim_Ea)
		#f.write(data_sheet)
		#f.close()
		os.chdir(SAdir)

os.makedirs("../Plots_SA",exist_ok = True)
os.makedirs("Data/SensitivityCoeffs",exist_ok = True)
selected_BRUTE_FORCE_PARAMETERS = {}
import traceback

def diagnose_rxn_case(rxn_index, case_index, case,
                      multiply_A, multiply_n, multiply_Ea,
                      nominal, P_multiply_A, P_multiply_n, P_multiply_Ea,
                      P_nominal_list, unsrt_data, rxn_list):
    """
    Runs all three Arrhenius parameter sensitivity computations for a single
    (case_index, rxn_index) pair and reports detailed diagnostics on failure.
    """
    rxn_name = list(unsrt_data.keys())[rxn_index] if hasattr(unsrt_data, 'keys') else str(rxn_index)

    def _check_array(label, arr, expected_dtype=float):
        """Validate shape, dtype, and finiteness of an array-like value."""
        arr = np.asarray(arr)
        issues = []
        if not np.issubdtype(arr.dtype, np.floating):
            issues.append(f"dtype is '{arr.dtype}' — expected float. Raw value: {arr}")
        if arr.size == 0:
            issues.append("array is EMPTY")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            issues.append(f"contains non-finite values (nan/inf): {arr}")
        return arr, issues

    report = []
    report.append("=" * 70)
    report.append(f"DIAGNOSTIC REPORT — case_index: {case_index}  |  rxn_index: {rxn_index}")
    report.append(f"Reaction identifier : {rxn_name}")
    report.append(f"Case key            : {case}")
    report.append("-" * 70)

    # --- Check nominal and multiply arrays ---
    for label, val in [
        ("nominal",              nominal),
        ("multiply_A[rxn_index]", multiply_A[rxn_index]),
        ("multiply_n[rxn_index]", multiply_n[rxn_index]),
        ("multiply_Ea[rxn_index]",multiply_Ea[rxn_index]),
    ]:
        arr, issues = _check_array(label, val)
        status = "OK" if not issues else "FAIL"
        report.append(f"  [{status}] {label}: shape={arr.shape}, dtype={arr.dtype}, value={arr}")
        for iss in issues:
            report.append(f"       ^^^ {iss}")

    # --- Check perturbed parameter vectors ---
    for label, val in [
        ("P_multiply_A[rxn_index]",  P_multiply_A[rxn_index]),
        ("P_multiply_n[rxn_index]",  P_multiply_n[rxn_index]),
        ("P_multiply_Ea[rxn_index]", P_multiply_Ea[rxn_index]),
        ("P_nominal_list[rxn_index]",P_nominal_list[rxn_index]),
    ]:
        arr, issues = _check_array(label, val)
        status = "OK" if not issues else "FAIL"
        report.append(f"  [{status}] {label}: shape={arr.shape}, dtype={arr.dtype}, value={arr}")
        for iss in issues:
            report.append(f"       ^^^ {iss}")

    report.append("-" * 70)
    report.append("Attempting each SA coefficient computation individually:")

    # --- Try each computation block separately ---
    blocks = {
        "A-parameter  (exp-based normalization)": lambda: (
            np.exp(P_multiply_A[rxn_index][0]),
            np.exp(P_nominal_list[rxn_index][0]),
            np.asarray((multiply_A[rxn_index] - nominal) / nominal).flatten()
        ),
        "n-parameter  (direct delta normalization)": lambda: (
            P_multiply_n[rxn_index][1],
            P_nominal_list[rxn_index][1],
            np.asarray((multiply_n[rxn_index] - nominal) / nominal).flatten()
        ),
        "Ea-parameter (ratio-based normalization)": lambda: (
            P_multiply_Ea[rxn_index][2],
            P_nominal_list[rxn_index][2],
            np.asarray((multiply_Ea[rxn_index] - nominal) / nominal).flatten()
        ),
    }

    all_ok = True
    for block_name, fn in blocks.items():
        try:
            result = fn()
            report.append(f"  [OK  ] {block_name} — result values: {result}")
        except Exception as e:
            all_ok = False
            report.append(f"  [FAIL] {block_name}")
            report.append(f"         Error type : {type(e).__name__}")
            report.append(f"         Error msg  : {e}")
            report.append(f"         Traceback  :\n{''.join(traceback.format_exc().splitlines(keepends=True))}")

    report.append("=" * 70)
    print("\n".join(report))
    return all_ok
    
for case_index,case in enumerate(temp_sim_opt_A):		
	rxn_Sa = {}
	rxn_Sa_1 = {}
	count = 0
	index = temp_sim_opt_A[str(case)]["index"]
	multiply_A = np.asarray(temp_sim_opt_A[str(case)]["ETA"])
	nominal = np.asarray(temp_sim_opt_x0[str(case)]["ETA"])
	
	index_n = temp_sim_opt_n[str(case)]["index"]
	multiply_n = np.asarray(temp_sim_opt_n[str(case)]["ETA"])
	
	index_ea = temp_sim_opt_Ea[str(case)]["index"]
	multiply_Ea = np.asarray(temp_sim_opt_Ea[str(case)]["ETA"])
	#Sensitivity analysis for A parameter
	T_ = float(target_list[case_index].temperature)
	SA_coeff_A = []
	SA_coeff_n = []
	SA_coeff_Ea = []
	SA_coeff_without_k_perturbed = []
	for rxn_index, rxn in enumerate(unsrt_data):
		
		#--------A------------------
		delta_a = float(delta_dict[rxn]["delta_A"])
		A_o = float(delta_dict[rxn]["P_o"][0])
		
		normalized_A = (A_o/delta_a)
		#print("multiply_A[rxn_index] = ", multiply_A[rxn_index], type(multiply_A[rxn_index]) )
		#print("Nominal", nominal, type(nominal))
		#raise AssertionError
		nominal = float(np.asarray(nominal).flatten()[0])	# nominal is an array not a scalar. converting to float
		fact_A = np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()
		#SA_coeff_A.append((k_o/(k_perturbed-k_o))*((multiply_A[rxn_index] - nominal)/nominal))
		SA_coeff_A.append(normalized_A*fact_A[0])
		SA_coeff_without_k_perturbed.append(np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()[0])
		#--------A------------------
		#Sensitivity analysis for n parameter
		
		
		delta_n = float(delta_dict[rxn]["delta_n"])
		n_o = float(delta_dict[rxn]["P_o"][1])
		
		fact_n = np.asarray(((multiply_n[rxn_index] - nominal)/nominal)).flatten()
		normalized_n = (1/delta_n)
		#SA_coeff_n.append((k_o/(k_perturbed-k_o))*((multiply_n[rxn_index] - nominal)/nominal))
		SA_coeff_n.append(normalized_n*fact_n[0])
		
		#--------Ea------------------
		#Sensitivity analysis for Ea parameter
		
		delta_Ea = float(delta_dict[rxn]["delta_Ea"])
		Ea_o = float(delta_dict[rxn]["P_o"][2])
		
		fact_ea = np.asarray(((multiply_Ea[rxn_index] - nominal)/nominal)).flatten()
		normalized_ea = Ea_o/delta_Ea
		#SA_coeff_Ea.append((k_o/(k_perturbed-k_o))*(multiply_Ea[rxn_index] - nominal)/nominal)
		SA_coeff_Ea.append(normalized_ea*fact_ea[0])
	

	
	for ind,rxn in enumerate(rxn_list):
		temp = []
		temp_1 = []
		temp.append(SA_coeff_A[count])
		temp_1.append(SA_coeff_without_k_perturbed[count])
		temp.append(SA_coeff_n[count])
		temp.append(SA_coeff_Ea[count])
		count+=1
		rxn_Sa[rxn] = temp
		rxn_Sa_1[rxn] = temp_1
		
	SA_dict = dict(sorted(rxn_Sa.items(), key=lambda item: abs(item[1][0]),reverse = True))
	SA_dict_1 = dict(sorted(rxn_Sa_1.items(), key=lambda item: abs(item[1][0]),reverse = True))
	#print(SA_dict)
	selected_BRUTE_FORCE_PARAMETERS[str(case_index)] = rxn_Sa
	sort_rlist = []
	sort_alist = []
	sort_alist_1 = []
	sort_nlist = []
	sort_ealist = []
	ticks = []
	for ind,rxn in enumerate(SA_dict):
		sort_rlist.append(rxn)
		sort_alist_1.append(SA_dict_1[rxn][0])
		sort_alist.append(SA_dict[rxn][0])
		sort_nlist.append(SA_dict[rxn][1])
		sort_ealist.append(SA_dict[rxn][2])
		ticks.append(ind)
		
	
	#print(sort_alist)
	fig = plt.figure()
	#y_pos = [i for i in range(0,len(sort_alist))]
	y_pos = range(0,len(sort_alist))
	#print(y_pos)
	plt.barh(y_pos,sorted(sort_alist,key =abs), alpha=0.51)
	#plt.barh(y_pos,sort_alist, alpha=0.51)
	plt.yticks(y_pos, sort_rlist)
	plt.xlabel(r'normalized sensitivities $S_i = \frac{\partial ln(S_{u}^{o})}{\partial x_i}}$')
	#plt.title('Sensitivity Analysis using Response surface method')
	plt.savefig('../Plots_SA/sensitivity_A_'+str(case_index)+'.png',bbox_inches="tight")
	plt.close()
	"""
	Plotting the sensitivity in same fig
	"""
	
	#print(sort_rlist)
	fake_data = pd.DataFrame({"index": list(sort_rlist), 0: sort_alist , 1: sort_nlist, 2: np.asarray(sort_ealist)*10})
	fake_data.set_index("index",drop=False)
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 8), frameon=False)
	
	fake_data[0].plot.barh(ax=ax1)
	fake_data[1].plot.barh(ax=ax2)
	fake_data[2].plot.barh(ax=ax3)
	ax1.set_yticks(ticks,sort_rlist)
	ax1.set_xlabel(r'$\partial ln(\eta) / \partial \zeta_{\alpha}$')
	ax2.set_xlabel(r'$\partial ln(\eta) / \partial \zeta_{n}$')
	ax3.set_xlabel(r'$\partial ln(\eta) / \partial \zeta_{\epsilon} (\times 10^{-1})$')
	fig.savefig('../Plots_SA/sensitivity_'+str(case_index)+'.png',bbox_inches="tight")
	plt.close()
	
	string_FM = "Sensitivity Analysis (using Cantera) Tig:\n"
	string_FM_1 = "Sensitivity Analysis (using Cantera) Tig:\n"
	for ind,rxn in enumerate(sort_rlist):
		string_FM +=f"\t{sort_alist[ind]:.8f}\t{index_dict[rxn.split(':')[0]]}\t{rxn}\n"
		string_FM_1 +=f"\t{sort_alist_1[ind]:.8f}\t{index_dict[rxn.split(':')[0]]}\t{rxn}\n"
		
	string_FM_n = "Sensitivity Analysis (using Cantera) Tig:\n"
	for ind,rxn in enumerate(sort_rlist):
		string_FM_n +=f"\t{sort_nlist[ind]:.8f}\t{index_dict[rxn.split(':')[0]]}\t{rxn}\n"
	
	string_FM_Ea = "Sensitivity Analysis (using Cantera) Tig:\n"
	for ind,rxn in enumerate(sort_rlist):
		string_FM_Ea +=f"\t{sort_ealist[ind]:.8f}\t{index_dict[rxn.split(':')[0]]}\t{rxn}\n"

	g = open(f"Data/SensitivityCoeffs/FM_sensitivity_T_{T_}_case_{case_index}.txt","w").write(string_FM)
	g = open(f"Data/SensitivityCoeffs/FM_sensitivity_T_{T_}_case_{case_index}_1.txt","w").write(string_FM_1)
	g_n = open(f"Data/SensitivityCoeffs/FM_sensitivity_T_{T_}_case_{case_index}_n.txt","w").write(string_FM_n)
	g_ea = open(f"Data/SensitivityCoeffs/FM_sensitivity_T_{T_}_case_{case_index}_ea.txt","w").write(string_FM_Ea)

	
os.chdir("..")
if "sens_3p_parameters.pkl" not in os.listdir():
	with open('sens_3p_parameters.pkl', 'wb') as file_:
			pickle.dump(selected_BRUTE_FORCE_PARAMETERS, file_)

raise AssertionError("3-PARAM SENSITIVITY ANALYSIS IS DONE!!")
