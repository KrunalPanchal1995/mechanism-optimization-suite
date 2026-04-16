try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
import yaml 
from copy import deepcopy
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import combustion_target_class
from Target2CSV import Exporter as Target2CSV
from VisualAid import TPhiPlotter, ArrheniusPlotter
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

add = "addendum"
targets = "targets"
count = "Counts"

dataCounts = optInputs[count]
targets_count = int(dataCounts["targets_count"])
locations = optInputs["Locations"]

targetLines = open(locations[targets],'r').readlines()
addendum = yaml.safe_load(open(locations[add],'r').read())


######################################################
##  Unloading the target data	  	                ##
## TARGET CLASS CONTAINING EACH TARGET AS AN OBJECT ##
######################################################


target_list = []
c_index = 0
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
print("Targets exported to targets.csv\n\n")

#Plotting the T-P-phi space
print("Plotting the T-P-phi space...")
TPhiPlotter(df).plot_t_p_phi(color="Phi", save_path="T-P-phi_space.png", show=True)
print("T-P-phi space plot saved as T-P-phi_space.png\n\n")