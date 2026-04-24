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
import DesignMatrix as DM
from DesignMatrix2_0 import DesignMatrixWriter
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
target_file = open("target_data.txt","w")
target_file.write(string_target)
target_file.close()

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
manipulationDict = {}
selection = []
Cholesky_list = []
zeta_list = []
activeParameters = []
P_nominal_list = []
P_upper = []
P_lower = []
P_multiply_A = []
P_multiply_n = []
P_multiply_Ea = []
rxn_list = []
rIndex = []
sigma_n = []
sigma_Ea = []
zeta_list_A = []
zeta_list_B = []
zeta_list_C = []

facto_a = 1.0
facto_n = 1.0
facto_ea = 50.0

for rxn in unsrt_data:
	activeParameters.extend(unsrt_data[rxn].activeParameters)
ap = len(activeParameters)
z_a = z_n = z_e = np.asarray([1,1,1])
p_a = np.asarray([facto_a,0,0])
p_n = np.asarray([0,facto_n,0])
p_e = np.asarray([0,0,facto_ea])
	
for rxn in unsrt_data:
	rxn_list.append(rxn)
	selection.extend(unsrt_data[rxn].selection)
	Cholesky_list.append(unsrt_data[rxn].cholskyDeCorrelateMat)
	covariance_mat = np.dot(unsrt_data[rxn].L,unsrt_data[rxn].L.T)
	sigma_vector = np.diag(covariance_mat)
	cov = unsrt_data[rxn].L
	zeta = np.asarray(unsrt_data[rxn].zeta.x)
	Po = np.asarray(unsrt_data[rxn].nominal)
	zeta_A = z_a*zeta
	zeta_B = z_n*zeta
	zeta_C = z_e*zeta
	zeta_list_A.append(zeta_A)
	zeta_list_B.append(zeta_B)
	zeta_list_C.append(zeta_C)
	P_upper.append(np.asarray(Po + np.array([1,1,1])*np.asarray(cov.dot(zeta)).flatten()))
	P_lower.append(np.asarray(Po - np.array([1,1,1])*np.asarray(cov.dot(zeta)).flatten()))
	P_multiply_A.append(np.asarray(Po + p_a*np.asarray(cov.dot(zeta_A)).flatten()))
	P_multiply_n.append(np.asarray(Po + p_n*np.asarray(cov.dot(zeta_B)).flatten()))
	P_multiply_Ea.append(np.asarray(Po + p_e*np.asarray(cov.dot(zeta_C)).flatten()))
	sigma_n.append(sigma_vector[1])
	sigma_Ea.append(sigma_vector[2])
	zeta_list.append(unsrt_data[rxn].perturb_factor)
	
	#activeParameters.extend(unsrt_data[rxn].activeParameters)
	P_nominal_list.append(unsrt_data[rxn].nominal)
	rIndex.append(unsrt_data[rxn].rIndex)
#print(zeta_list)
#print(P_nominal_list[0],P_multiply_n[0])

#raise AssertionError("Stop")

manipulationDict["selection"] = deepcopy(selection)#.deepcopy()
manipulationDict["Cholesky"] = deepcopy(Cholesky_list)#.deepcopy()
manipulationDict["zeta"] = deepcopy(zeta_list)#.deepcopy()
manipulationDict["activeParameters"] = deepcopy(activeParameters)#.deepcopy()
manipulationDict["nominal"] = deepcopy(P_nominal_list)#.deepcopy()
print("\nFollowing list is the choosen reactions\n")
print(manipulationDict["activeParameters"])


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
if "DesignMatrix_x0_3P.csv" not in os.listdir():
	perturbation_values = (facto_a,facto_n,facto_ea)
	output_dir = "."
	DesignMatrixWriter(unsrt_data,perturbation_values,output_dir).write_all()
	
	design_matrix_file_x0_3P = open("DesignMatrix_x0_3P.csv").readlines()
	design_matrix_x0_3P = []
	for row in design_matrix_file_x0_3P:
		design_matrix_x0_3P.append([float(ele) for ele in row.strip("\n").split(",")])
		
	design_matrix_file_3P = open("DesignMatrix_3P.csv").readlines()
	design_matrix_3P = []
	for row in design_matrix_file_3P:
		design_matrix_3P.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_A_file = open("convertor_A.csv").readlines()
	convertor_A = []
	for row in convertor_A_file:
		convertor_A.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_n_file = open("convertor_n.csv").readlines()
	convertor_n = []
	for row in convertor_n_file:
		convertor_n.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_Ea_file = open("convertor_Ea.csv").readlines()
	convertor_Ea = []
	for row in convertor_Ea_file:
		convertor_Ea.append([float(ele) for ele in row.strip("\n").split(",")])


else:
	design_matrix_file_x0_3P = open("DesignMatrix_x0_3P.csv").readlines()
	design_matrix_x0_3P = []
	for row in design_matrix_file_x0_3P:
		design_matrix_x0_3P.append([float(ele) for ele in row.strip("\n").split(",")])
		
	design_matrix_file_3P = open("DesignMatrix_3P.csv").readlines()
	design_matrix_3P = []
	for row in design_matrix_file_3P:
		design_matrix_3P.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_A_file = open("convertor_A.csv").readlines()
	convertor_A = []
	for row in convertor_A_file:
		convertor_A.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_n_file = open("convertor_n.csv").readlines()
	convertor_n = []
	for row in convertor_n_file:
		convertor_n.append([float(ele) for ele in row.strip("\n").split(",")])
	
	convertor_Ea_file = open("convertor_Ea.csv").readlines()
	convertor_Ea = []
	for row in convertor_Ea_file:
		convertor_Ea.append([float(ele) for ele in row.strip("\n").split(",")])

#########################################
###   Generating YAML files for      ####
###    sensitivity analysis          ####
#########################################


yaml_loc_nominal = []
yaml_loc_nominal.append(mech_file_location)
SSM = simulator.SM(target_list,optInputs,unsrt_data,design_matrix_3P)
if "Perturbed_Mech_SA_3P_BruteForce" not in os.listdir():
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/A_factor")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/n")
	os.mkdir("Perturbed_Mech_SA_3P_BruteForce/Ea")
	print("\nPerturbing the Mechanism files for 3-Parameter brute force sensitivity analysis\n")
	chunk_size = 500
	params_DM = [design_matrix_3P[i:i+chunk_size] for i in range(0, len(design_matrix_3P), chunk_size)]
	count = 0
	yaml_loc_A = []
	yaml_loc_n = []
	yaml_loc_Ea = []
	for params in params_DM:
		yaml_list = SSM.getYAML_List(params,"A",fact = facto_a)
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
	for params in params_DM:
		yaml_list = SSM.getYAML_List(params,"n",fact = facto_n)
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
	for params in params_DM:
		yaml_list = SSM.getYAML_List(params,"Ea",fact = facto_ea)
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
	for i,sample in enumerate(design_matrix_3P):
		index_list_A.append(i)
		location_mech_A.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor")
		yaml_loc_A.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/A_factor/mechanism_"+str(i)+".yaml")
	for i,sample in enumerate(design_matrix_3P):
		index_list_n.append(i)
		location_mech_n.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n")
		yaml_loc_n.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/n/mechanism_"+str(i)+".yaml")
	for i,sample in enumerate(design_matrix_3P):
		index_list_Ea.append(i)
		location_mech_Ea.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea")
		yaml_loc_Ea.append(os.getcwd()+"/Perturbed_Mech_SA_3P_BruteForce/Ea/mechanism_"+str(i)+".yaml")


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
	FlameMaster_Execution_location_A = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_3P).make_dir_in_parallel(yaml_loc_A)
	
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
	FlameMaster_Execution_location_n = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_3P).make_dir_in_parallel(yaml_loc_n)
	
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
	FlameMaster_Execution_location_Ea = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_3P).make_dir_in_parallel(yaml_loc_Ea)
	
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
	FlameMaster_Execution_location_x0 = simulator.SM(target_list,optInputs,rxn_dict,design_matrix_x0_3P).make_dir_in_parallel(yaml_loc_nominal)
	
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
		
		"""
		k_perturbed = getKappa(P_multiply_A[rxn_index])
		k_o = getKappa(P_nominal_list[rxn_index])
		normalized_A = np.asarray((k_o/(k_perturbed-k_o))).flatten()
		fact_A = np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()
		#SA_coeff_A.append((k_o/(k_perturbed-k_o))*((multiply_A[rxn_index] - nominal)/nominal))
		SA_coeff_A.append(np.asarray(max(list(normalized_A))*fact_A)[0])
		SA_coeff_without_k_perturbed.append(np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()[0])
		
		#Sensitivity analysis for n parameter
		#T = target_list[case_index].temperature
		k_perturbed = getKappa(P_multiply_n[rxn_index])
		k_o = getKappa(P_nominal_list[rxn_index])
		#print(len(k_o),len(k_perturbed),len(multiply_n))
		#print(multiply_n)
		fact_n = np.asarray(((multiply_n[rxn_index] - nominal)/nominal)).flatten()
		normalized_n = np.asarray((k_o/(k_perturbed-k_o))).flatten()
		print(normalized_n)
		max_morm_n = max(list(normalized_n))
		#SA_coeff_n.append((k_o/(k_perturbed-k_o))*((multiply_n[rxn_index] - nominal)/nominal))
		SA_coeff_n.append(max_morm_n*fact_n[0])
		#Sensitivity analysis for Ea parameter
		#T = target_list[case_index].temperature
		k_perturbed = getKappa(P_multiply_Ea[rxn_index])
		k_o = getKappa(P_nominal_list[rxn_index])
		fact_ea = np.asarray(((multiply_Ea[rxn_index] - nominal)/nominal)).flatten()
		normalized_ea = np.asarray((k_o/(k_perturbed-k_o))).flatten()
		max_morm_ea = max(list(normalized_ea))
		#SA_coeff_Ea.append((k_o/(k_perturbed-k_o))*(multiply_Ea[rxn_index] - nominal)/nominal)
		SA_coeff_Ea.append(max_morm_ea*fact_ea[0])
		"""
		
		A_perturbed = np.exp(P_multiply_A[rxn_index][0])
		A_o = np.exp(P_nominal_list[rxn_index][0])
		normalized_A = (A_o/(A_perturbed-A_o))
		fact_A = np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()
		#SA_coeff_A.append((k_o/(k_perturbed-k_o))*((multiply_A[rxn_index] - nominal)/nominal))
		SA_coeff_A.append(normalized_A*fact_A[0])
		SA_coeff_without_k_perturbed.append(np.asarray((multiply_A[rxn_index] - nominal)/nominal).flatten()[0])
		print(multiply_A[rxn_index])
		#Sensitivity analysis for n parameter
		#T = target_list[case_index].temperature
		n_perturbed = P_multiply_n[rxn_index][1]
		n_o = P_nominal_list[rxn_index][1]
		#print(len(k_o),len(k_perturbed),len(multiply_n))
		#print(multiply_n)
		fact_n = np.asarray(((multiply_n[rxn_index] - nominal)/nominal)).flatten()
		normalized_n = (1/(n_perturbed-n_o))
		#SA_coeff_n.append((k_o/(k_perturbed-k_o))*((multiply_n[rxn_index] - nominal)/nominal))
		SA_coeff_n.append(normalized_n*fact_n[0])
		#Sensitivity analysis for Ea parameter
		#T = target_list[case_index].temperature
		Ea_perturbed = P_multiply_Ea[rxn_index][2]
		Ea_o = P_nominal_list[rxn_index][2]
		fact_ea = np.asarray(((multiply_Ea[rxn_index] - nominal)/nominal)).flatten()
		normalized_ea = Ea_o/(Ea_perturbed-Ea_o)
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


