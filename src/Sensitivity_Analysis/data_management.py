import re, os, math
import time
import numpy as np
import shutil
import subprocess
import subprocess as sub
import threading
import combustion_target_class
import make_input_file as MakeFile
from tqdm import tqdm
import multiprocessing
import sys
import traceback
from pathlib import Path
import datetime




SUBFOLDER_TO_SUBDIR = {
    "multiply_A":  "A_factor",
    "multiply_n":  "n",
    "multiply_Ea": "Ea",
    "divide_A":    "A_factor_neg",
    "divide_n":    "n_neg",
    "divide_Ea":   "Ea_neg",
    "nominal":     "nominal",
}


class RunCmd(threading.Thread):
	def __init__(self, cmd, timeout):
		threading.Thread.__init__(self)
		self.cmd = cmd
		self.timeout = timeout

	def run(self):
		self.p = sub.Popen(self.cmd)
		self.p.wait()

	def Run(self):
		self.start()
		self.join(self.timeout)

		if self.is_alive():
			self.p.terminate()	  #use self.p.kill() if process needs a kill -9
			self.join()


def run_sim(index, case, input_, caseID, path, total):
    """
    Run a single Cantera simulation for SA_3P_BruteForce perturbations.

    Fixes over the original:
      - No os.chdir(): all paths are absolute; subprocess uses cwd=workdir.
        (chdir inside multiprocessing pool workers leaks state between tasks.)
      - Per-simulation log file `run_sim.log` written in workdir, so errors
        survive the run instead of vanishing into worker-process stdout.
      - Error message returned in the result tuple so the parent process can
        aggregate all failures into a master log.
      - Fixed NameError: AssertionError referenced `Perturbed_location`
        before assignment; now uses `perturbed_path`.
      - Fixed possible NameError on `sub_dir` for unknown subfolder names.
    """
    eta = "N/A"
    ETA = "N/A"
    string = path
    err_msg = ""

    # path example: /data/.../multiply_A/case-44/2610/output/tau.out
    loc = os.path.dirname(path)                 # .../2610/output
    file_name = os.path.basename(path)          # tau.out
    workdir = os.path.dirname(loc)              # .../2610

    # ---- per-simulation logger -------------------------------------------
    log_path = os.path.join(workdir, "run_sim.log")

    def log(msg, level="INFO"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] [{level}] {msg}\n"
        try:
            with open(log_path, "a") as lf:
                lf.write(line)
        except OSError:
            # last resort: at least try stdout
            sys.stdout.write(line)

    def log_exception(stage, exc):
        nonlocal err_msg
        tb = traceback.format_exc()
        err_msg = f"{stage}: {exc}"
        log(f"FAILED at {stage}: {exc}", level="ERROR")
        log(tb, level="TRACEBACK")

    try:
        log(f"--- run_sim start | index={index} caseID={caseID} ---")
        log(f"target path : {path}")
        log(f"workdir     : {workdir}")

        # -------------------------------
        # Step 1: Build paths
        # -------------------------------
        try:
            parts = path.split("/")
            dir_name = parts[-3]            # e.g. 2610
            case_name = parts[-4]           # e.g. case-44
            subfolder_name = parts[-5]      # e.g. multiply_A

            if subfolder_name not in SUBFOLDER_TO_SUBDIR:
                raise ValueError(
                    f"Unknown perturbation subfolder '{subfolder_name}' in path; "
                    f"expected one of {list(SUBFOLDER_TO_SUBDIR)}"
                )
            sub_dir = SUBFOLDER_TO_SUBDIR[subfolder_name]

            base_path = Path("/".join(parts[:-6]))
            perturbed_path = (
                base_path / "Perturbed_Mech_SA_3P_BruteForce" / sub_dir 
            )

            if sub_dir != "nominal" and not perturbed_path.exists():
                raise FileNotFoundError(
                    f"Perturbed mechanism directory not found: {perturbed_path}"
                )

            Perturbed_location = str(perturbed_path)
            run_cantera = os.path.join(workdir, "cantera_1.py")

            log(f"dir_name           : {dir_name}")
            log(f"case_name          : {case_name}")
            log(f"sub_dir            : {sub_dir}")
            log(f"Perturbed_location : {Perturbed_location}")
            log(f"run_cantera        : {run_cantera}")

        except Exception as e:
            log_exception("Step 1 (path construction)", e)
            return (index, eta, ETA, string, total, err_msg)

        # -------------------------------
        # Step 2: Create Cantera input string
        # -------------------------------
        try:
            if input_["Stats"]["Design_of_PRS"] != "A-facto":
                if sub_dir == "nominal":
                    mech_path = input_["Locations"]["mechanism"]
                else:
                    mech_path = f"{Perturbed_location}/mechanism_{dir_name}.yaml"
            else:
                # absolute, since we no longer chdir into workdir
                mech_path = os.path.join(workdir, "mechanism.yaml")

            if not os.path.exists(mech_path):
                raise FileNotFoundError(f"Mechanism file not found: {mech_path}")

            log(f"mechanism file: {mech_path}")

            instring, a, b, c = MakeFile.create_input_file(
                caseID, input_, case, mech_file=mech_path
            )
            if not instring:
                raise ValueError("MakeFile.create_input_file returned empty instring")

            log("Cantera input string created successfully")

        except Exception as e:
            log_exception("Step 2 (input file creation)", e)
            return (index, eta, ETA, string, total, err_msg)

        # -------------------------------
        # Step 3: Write cantera_1.py
        # -------------------------------
        try:
            with open(run_cantera, "w") as f:
                f.write(instring)
            log(f"cantera_1.py written at {run_cantera}")
        except Exception as e:
            log_exception("Step 3 (writing cantera_1.py)", e)
            return (index, eta, ETA, string, total, err_msg)

        # -------------------------------
        # Step 4: Run cantera_1.py
        # -------------------------------
        try:
            solve_out = os.path.join(workdir, "solve_1.out")
            log(f"running cantera_1.py | log: {solve_out}")

            with open(solve_out, "w") as f:
                result = subprocess.call(
                    [sys.executable, run_cantera],   # was hard-coded "python3.9"
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=workdir,                     # replaces os.chdir
                )

            log(f"subprocess return code: {result}")
            if result != 0:
                log(f"cantera_1.py exited non-zero ({result}); "
                    f"see {solve_out}", level="WARNING")

        except Exception as e:
            log_exception("Step 4 (running cantera_1.py)", e)
            return (index, eta, ETA, string, total, err_msg)

        # -------------------------------
        # Step 5: Read tau.out
        # -------------------------------
        try:
            tau_file = os.path.join(loc, file_name)
            log(f"reading output file: {tau_file}")

            if not os.path.exists(tau_file):
                raise FileNotFoundError(f"{file_name} not found at {tau_file}")

            with open(tau_file, "r") as f:
                out_file = f.readlines()

            string = tau_file

            if len(out_file) < 2:
                raise ValueError(
                    f"{file_name} has insufficient lines: {len(out_file)}"
                )

            line = out_file[1].split()
            log(f"parsed output line: {line}")

            if len(line) == 2:
                eta = np.log(float(line[1]) * 10)
                ETA = float(line[1])
            else:
                log("output second line does not have 2 columns; "
                    "using fallback values", level="WARNING")
                eta = np.log(100 * 10000)
                ETA = 100 * 10000

            log(f"SUCCESS | eta={eta} ETA={ETA}")

        except Exception as e:
            log_exception("Step 5 (reading output)", e)
            return (index, eta, ETA, string, total, err_msg)

    except Exception as e:
        # absolute safety net: nothing should escape a pool worker silently
        log_exception("run_sim (unexpected)", e)

    return (index, eta, ETA, string, total, err_msg)

class Worker():
    def __init__(self, workers, error_log="failed_simulations.log"):
        self.pool = multiprocessing.Pool(processes=workers)
        self.results = []
        self.progress = []
        self.error_log = error_log

    def callback_run(self, result):
        self.results.append(result)
        self.progress.append(result[0])

        # result = (index, eta, ETA, string, total, err_msg)
        err_msg = result[5] if len(result) > 5 else ""
        if err_msg:
            with open(self.error_log, "a") as f:
                f.write(f"index={result[0]}\tpath={result[3]}\terror={err_msg}\n")

        sys.stdout.write(
            "\t\t\r{:06.2f}% is complete".format(
                len(self.progress) / float(result[4]) * 100
            )
        )
        sys.stdout.flush()

    def callback_error(self, error):
        # fires only if run_sim itself raises (should now be never,
        # thanks to the safety net), but log it anyway
        with open(self.error_log, "a") as f:
            f.write(f"POOL-LEVEL ERROR: {error}\n")
        print("error", error)

    def do_job_async(self, location, case, optInput, caseID):
        for index, args in enumerate(location):
            self.pool.apply_async(
                run_sim,
                args=(index, case, optInput, caseID, args, len(location)),
                callback=self.callback_run,
                error_callback=self.callback_error,
            )
        self.pool.close()
        self.pool.join()

        self.results.sort(key=lambda x: x[0])
        sorted_eta = [r[1] for r in self.results]
        sorted_ETA = [r[2] for r in self.results]
        sorted_string = [r[3] for r in self.results]
        return sorted_eta, sorted_ETA, sorted_string

#Extract_reactinons pre exponential factor from mechanism file using sorted numbers. dummy function not called in main code. replaced with a regular expression.
def extract_reaction_coeff(mech):
	data = mech.readlines()
	reaction_set = []
	reaction = ''
	pre_exp_factor = {}
	for n,line in enumerate(data):
		if('->' in line):
			reaction = line
			i = 0
			while(True):
				if('}' not in reaction):
					reaction = reaction + data[n+i+1]
				else:
					break
				i +=1
			reaction_set.append(reaction)

	for r in reaction_set:
		param = r.split()
		index = param[0][:-1]
		for a in param:
			if a=="a" or a== "A":
				pre_exp_factor[index] = float(param[param.index(a)+2]) 
	
	return pre_exp_factor	
		
#extract uncertainity data for each reaction from user created uncertainty file. It should be a two column file with first column containing index and second one with uncertainties
def extract_index_and_uncertainty(uns):
	data = uns.readlines()
	uncertainty_a = {}
	uncertainty_n = {}
	uncertainty_e = {}
	index = []
	for i in data:
		if "#" in i:
			i = i[:i.index('#')]
		seg = i.split()
		if len(seg) == 4:
			uncertainty_a[seg[0]] = float(seg[1]);
			uncertainty_n[seg[0]] = float(seg[2]);
			uncertainty_e[seg[0]] = float(seg[3]);
			index.append(seg[0])
		else:
			continue
	return uncertainty_a, uncertainty_n, uncertainty_e, index

#Once the simulations are completed, this function goes through all the locations where FlameMaster simulation is performed and read the output value based on the type of target.
#and print as a text file containg four columns. First three contain location information and last column contain  target value.
def generate_SA_target_value_tables(locations, t_list, case, fuel, input_={}):
	list_fuel = []

	if "dict" in str(type(fuel)):
		for i in fuel:
			list_fuel.append(fuel[i])
		fuel = list_fuel[0]

	data_loc = []
	for location in locations:
		list_loc = location.split("/")
		for i in list_loc:
			if i == "case-" + str(case).strip():
				data_loc.append(location.strip("\n"))

	data = ''
	failed_sim = ""
	ETA = []
	eta = []

	# --------------------------------------------------
	# First pass: identify failed simulations
	# --------------------------------------------------
	re_run_sim_dict = {}
	count = 0

	for i in data_loc:
		eta_, ETA_, file_path = extract_output(
			t_list[case],
			fuel,
			i[:-4] + "/output/",
			data_loc.index(i),
			input_=input_,
			caseID=case
		)

		if "N/A" in str(eta_):
			re_run_sim_dict[count] = file_path
			count += 1

	re_run_loc = []
	if re_run_sim_dict:
		for i in re_run_sim_dict:
			re_run_loc.append(re_run_sim_dict[i])

	# --------------------------------------------------
	# Re-run failed simulations
	# --------------------------------------------------
	if len(re_run_loc) != 0:
		print("#############______________________################\n")
		print(
			f"\t\tFor Case-{case}:\n"
			f"\t\t\tTotal of {len(re_run_loc)} simulations out of {len(data_loc)} "
			f"did not return expected results\n"
			f"\t\tRe-running those simulations........\n\n\n"
		)

		W = Worker(100)

		sorted_eta, sorted_ETA, sorted_Path = W.do_job_async(
			re_run_loc,
			t_list[case],
			input_,
			case
		)

		del W

		print(
			f"\t\tSimulations ended successfully\n\n"
			f"#############______________________################\n"
		)

	# --------------------------------------------------
	# Second pass: collect final results
	# --------------------------------------------------
	folderName = []

	for i in data_loc:
		pathList = i.split("/")
		file_loc = open("./eta_file_location.txt", "+a")

		start = pathList.index("case-{}".format(case))

		eta_, ETA_, file_path = extract_output(
			t_list[case],
			fuel,
			i[:-4] + "/output/",
			data_loc.index(i),
			input_=input_,
			caseID=case
		)

		if "N/A" in str(eta_):
			file_loc.write(file_path + "\n")

			folderName.append(pathList[start + 1])
			fN = pathList[start + 1]

			failed_sim += "{}\t{}\n".format(fN, ETA_)
			file_loc.close()

		else:
			file_loc.write(file_path + "\n")

			folderName.append(pathList[start + 1])
			fN = pathList[start + 1]

			data += "{}\t{}\n".format(fN, ETA_)
			file_loc.close()

			ETA.append(ETA_)
			eta.append(eta_)

	return data, failed_sim, folderName, ETA, eta
	
	
def generate_target_value_tables(locations, t_list, case, fuel,input_={}):
	list_fuel = []
	optInputs = input_
	if "dict" in str(type(fuel)):
		for i in fuel:
			list_fuel.append(fuel[i])
		fuel = list_fuel[0]
	
	#print(fuel)
	data_loc = []
	for location in locations:
		#print(location)
		list_loc = location.strip("\n").split("/")
		#print(list_loc)
		for i in list_loc:
			if i == "case-"+str(case).strip():
				data_loc.append(location.strip("\n"))
		#print(data_loc)
		#raise AssertionError("Stop!!")
	#print(data_loc)
	data =''
	failed_sim = ""
	ETA = []
	eta = []
	re_run_sim_dict = {}
	count = 0
	for i in data_loc:
		eta_,_,file_path = extract_output(t_list[case],fuel,i[:-3]+"output/",data_loc.index(i),input_=optInputs,caseID = case)		
		if "N/A" in str(eta_):
			re_run_sim_dict[count] = file_path
			count+=1
	
	re_run_loc = []
	if re_run_sim_dict != {}:
		for i in re_run_sim_dict:
			re_run_loc.append(re_run_sim_dict[i])
	
	if len(re_run_loc)!=0:
		print("#############______________________################\n")
		print(f"\t\tFor Case-{case}:\n\t\t\tTotal of {len(re_run_loc)} simulations out of {len(data_loc)} did not returned expected results\n\t\tRe-running those simulations........\n\n\n")
		W = Worker(100)
		sorted_eta,sorted_ETA,sorted_Path = W.do_job_async(re_run_loc,t_list[case],optInputs,case)
		del W
		print(f"\t\tSimulations ended successfully\n\n#############______________________################\n")
		#raise AssertionError("Stop!")
	for i in data_loc:
		pathList = i.split("/")
		file_loc = open("./eta_file_location.txt","+a")
		start = pathList.index("case-{}".format(case))

		#print(i)
		eta_,ETA_,file_path = extract_output(t_list[case],fuel,i[:-3]+"output/",data_loc.index(i),input_=optInputs,caseID = case)
		if "N/A" in str(eta_):
			#print(eta)
			#print(file_path)
			file_loc.write(file_path+"\n")
			folderName = pathList[start+1]
			#print(folderName)
			#print(eta)
			failed_sim += "{}\t{}\n".format(folderName , eta_)
			file_loc.close()
			
		else:	
			#print(eta)
			#print(file_path)
			file_loc.write(file_path+"\n")
			folderName = pathList[start+1]
			#print(folderName)
			#print(eta)
			data += "{}\t{}\n".format(folderName , eta_)
			file_loc.close()
			ETA.append(ETA_)
			eta.append(eta_)
	return data,failed_sim,eta


def extract_direct_simulation_values(case,loc,target_list,fuel):
	data =''
	eta_list = []
	failed_list = []
	pathList = loc.strip("\n").split("/")
	#print(pathList)
	#start = pathList.index("case-{}".format(case))
	file_loc = open("./eta_file_location.txt","w")
	eta,file_path = extract_output(target_list[case],fuel,loc,case)
	if "N/A" in str(eta):
		file_loc.write(file_path+"\n")
		folderName = pathList[-3]
		data += "{}\t{}\n".format(folderName , eta)
		file_loc.close()
		failed_list.append(float(eta))	
	else:
		file_loc.write(file_path+"\n")
		folderName = pathList[-3]
		data += "{}\t{}\n".format(folderName , eta)
		file_loc.close()
		eta_list.append(float(eta))
	return eta_list


#function extracts output from the given text file based on the called location where it currently is. Called in the previous function.		
def extract_output(case,fuel,path,index,input_=None,caseID=None):
	eta = string = ETA = None
	#print(case.target)
	if case.target.strip() == "RCM":
		if "cantera" in case.add["solver"]:
			string = path +"RCM.out"
			if "RCM.out" in os.listdir(path):
				out_file = open(path+"RCM.out",'r').readlines()
				string = path +"RCM.out"
				#print(out_file)
				line = out_file[1].split()
				#print(len(line))
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1])*10)	#us/micro seconds
					ETA = float(line[1])
				else:
					eta = np.log(100*10000)
					ETA = 100
				
			else:
				eta = "N/A"
				ETA = eta
		
		elif "CHEMKIN_PRO" in case.add["solver"]:
			#print("True")
			#print(path)
			#print(os.listdir(path))
			if "RCM.out" in os.listdir(path):
				out_file = open(path+"RCM.out",'r').readlines()
				string = path +"RCM.out"
				#print(out_file)
				line = out_file[1].split()
				#print(len(line))
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	#us/micro seconds
				else:
					eta = np.log(100*10000)
					ETA = 100*10000
			else:
				eta = "N/A"
				ETA = eta
				string = path+"RCM.out"
				
	elif case.target.strip() == "JSR":
		string = path +"jsr.out"
		if "cantera" in case.add["solver"]:
			if "jsr.out" in os.listdir(path):
				out_file = open(path+"jsr.out",'r').readlines()
				string = path +"jsr.out"
				#print(out_file)
				line = out_file[1].split()
				#print(len(line))
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1]) 	#us/micro seconds
				else:
					eta = np.log(100*10000)
					ETA = 100
				
			else:
				eta = "N/A"
				ETA = eta
				string = path+"jsr.out"
	
	elif case.target.strip() == "Tig":
		string = path +"tau.out"
		if "cantera" in case.add["solver"]:
			if "tau.out" in os.listdir(path):
				out_file = open(path+"tau.out",'r').readlines()
				string = path +"tau.out"
				#print(out_file)
				line = out_file[1].split()
				#print(len(line))
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	#us/micro seconds
				else:
					eta = np.log(100*10000)
					ETA = 100*10000
			
			else:
				eta = "N/A"
				ETA = eta
				string = path+"tau.out"
		
		
		elif "CHEMKIN_PRO" in case.add["solver"]:
			string = path +"tau.out"
			#print("True")
			#print(path)
			#print(os.listdir(path))
			if "tau.out" in os.listdir(path):
				out_file = open(path+"tau.out",'r').readlines()
				string = path +"tau.out"
				#print(out_file)
				line = out_file[1].split()
				#print(len(line))
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	#us/micro seconds
				else:
					eta = np.log(100*10000)
					ETA = 100*10000
			else:
				eta = "N/A"
				ETA = eta
				string = path+"tau.out"
		
		
		else:
			#print("Tig is the target and FlameMaster is the solver")
			
			out_file = open(path+"tau.out",'r').readlines()
			string = path +"tau.out"
			line = out_file[0].split("	")
			#print(line)
			if len(line) == 3:
				#eta = np.log(float(line[2])*1000)
				#print(float(line[1]))
				eta = np.log(float(line[1])*1000*10)
				ETA = float(line[1])*1000#us /micro seconds
			else:
				eta = np.log(100*10000)
				ETA = 100*10000
			#return eta,string
	elif case.target.strip() == "Fls":
		if "cantera" in case.add["solver"]:
			out_file = open(path+"Su.out",'r').readlines()
			string = path +"Su.out"
			line = out_file[1].split()
			#print(line)
			if len(line) == 2:
				eta = np.log(float(line[1]))
				ETA = float(line[1])	#cm/seconds
			else:
				eta = np.log(200)
				ETA = 200
		else:
			#start = os.getcwd()
			#print(path)
			os.chdir(path)
			start = path
			outfile =""
			flist = os.listdir()
			for i in flist:
				if fuel in i:
					if i.endswith("{}".format(int(case.temperature))):
						outfile =open(i,'r').readlines()
						string = path+i
						for line in outfile:
							if "burningVelocity" in line:
								#print(line.split()[2])
								#eta = np.log(float(line.split()[2]))
								eta = float(line.split()[2])
								#os.chdir("..")
								#return eta,string
					else:
						if i.endswith("noC"):
							#os.remove(i)
							continue			
			if outfile == "":
				#print("{sim_index} in {case_index} is not converged. Changing the start profile and re-calculating the flame speed as outfile is {outfile}".format(sim_index = index, case_index = case.case_index,outfile = outfile))
				#print("\n Current location is {}".format(os.getcwd()))
				start_profiles = open("../../eta_file_location.txt","r").readlines()			
				count = 1
				os.chdir(path)
				os.chdir("..")
				start = os.getcwd() 
				while outfile == "":
					print(start_profiles)
					for start_profile in start_profiles:
						os.chdir(start)
						print("{}: Using profile location of {}".format(count,start_profile))
						print(os.getcwd())
						#print(os.listdir())
						FM_input = open("FlameMaster.input","r").readlines()
						temp = open("New.txt","+a")
						for line in FM_input:
							if "StartProfilesFile" in line.split():
								string = "StartProfilesFile is "+start_profile
								temp.write(line.replace(line,string))
							else:
								temp.write(line)
						temp.close()
						print("\n \t Creating FlameMaster input file with new start profile \n")
						#print(os.listdir())
						os.remove("FlameMaster.input")
						os.rename("New.txt","FlameMaster.input")
						#run FlameMaster locally
						#print(os.listdir())
						kill = int(180)
						print("\n \t Running the FlameMaster locally...\n \t Kill Time out of {} sec".format(kill))
						RunCmd(["./run"],kill).Run()
						 
						count +=1
						os.chdir("output")
						flist = os.listdir()
						for i in flist:
							if fuel in i:
								if i.endswith("{}".format(int(case.temperature))):
									outfile =open(i,'r').readlines()
									string = path+i
									for line in outfile:
										if "burningVelocity" in line:
											#print(line.split()[2])
											#eta = np.log(float(line.split()[2]))
											eta = float(line.split()[2])
											string = path+i
											os.chdir(start)
											#return eta,string	
								else:
									if i.endswith("noC"):
										#os.remove(i)
										print("Case not converged yet!!!")
										continue
									else:
										continue	
				'''
				for i in flist: 
					if i.endswith("noC"):
						outfile =open(i,'r').readlines()
						for line in outfile:
							if "burningVelocity" in line:
								print(line.split()[2])
								eta = line.split()[2]
								os.chdir(start)
								return eta
					'''
		
			#exit() 
		#for i in outfile:
		#	if "burningVelocity" in i:
		#		eta = i.split()[2]
		#return eta,string
	elif case.target.strip() == "Flf":
		if "cantera" in case.add["solver"]:
			out_file = open(path+"flf.out",'r').readlines()
			string = path +"flf.out"
			line = out_file[1].split("	")
			#print(line)
			if len(line) == 2:
				eta = float(line[1])
				ETA = eta	#%mole
			else:
				eta = 100
				ETA = eta
		else:
			#print("Tig is the target")
			out_file = open(path+"result.dout",'r').readlines()
			string = path +"result.dout"
			line = out_file[0].split()
			#print(line)
			if len(line) == 2:
				#eta = np.log(float(line[2])*1000)
				eta = float(line[1])*100
				ETA = eta #%mole
			else:
				eta = 100
				ETA = eta
	
	elif case.target.strip() == "Flw":
		if "cantera" in case.add["solver"]:
			if "slope" in case.add["flw_method"]:
				out_file = open(path+"rate.csv",'r').readlines()
				string = path +"rate.csv"
				line = out_file[0].split()
				if len(line) == 3:
					#eta = np.log(float(line[2])*1000)
					eta = float(line[1]) #ppm/ms
				else:
					eta = 100
			else:	
				#print("Tig is the target")
				out_file = open(path+"rate.csv",'r').readlines()
				string = path +"time.csv"
				line = out_file[0].split()
				#print(line)
				if len(line) == 3:
					#eta = np.log(float(line[2])*1000)
					eta = float(line[1]) #ms
				else:
					eta = 100
		else:
			if "slope" in case.add["flw_method"]:
				out_file = open(path+"rate.csv",'r').readlines()
				string = path +"rate.csv"
				line = out_file[0].split()
				if len(line) == 3:
					#eta = np.log(float(line[2])*1000)
					eta = float(line[1]) #ppm/ms
				else:
					eta = 100
			else:	
				#print("Tig is the target")
				out_file = open(path+"rate.csv",'r').readlines()
				string = path +"time.csv"
				line = out_file[0].split()
				#print(line)
				if len(line) == 3:
					#eta = np.log(float(line[2])*1000)
					eta = float(line[1]) #ms
				else:
					eta = 100
#			
	return eta,ETA,string	
#Generate an optimized mechanism based on the optimized vector. 
def generate_optimized_mechanism(mech_file_location, reaction_index, unsrt_data, opt_x):
	mech_file = open(mech_file_location,'r')
	mech = new_mech = mech_file.read()	
	for i in reaction_index:
		w = re.compile(r'\n{}:.*?->.*?a\s*?=.*?(\S*?E\S\d*?\s).*?\}}'.format(i), re.DOTALL | re.IGNORECASE) #Group(1) will extract pre exp factor
		match = re.search(w, mech)
		if match == None:
			print ("Unable to perturb reaction rate... Exiting")
			exit()
		reaction = match.group()
		pre_exp = match.group(1)
		pre_exp_p = "{:.4E} ".format(float(pre_exp)*math.exp(opt_x[reaction_index.index(i)]*math.log(unsrt_data[i])))
		new_reaction = reaction.replace(pre_exp, pre_exp_p)
		new_mech = new_mech.replace(reaction, new_reaction)
		#Perturb backward reaction
		if i[-1] == 'f':
			kb = i[:-1]+'b'
			w = re.compile(r'\n{}:.*?->.*?a\s*?=.*?(\S*?E\S\d*?\s).*?\}}'.format(kb), re.DOTALL | re.IGNORECASE) #Group(1) will extract pre exp factor
			match = re.search(w, mech)
			if match != None:
				reaction = match.group()
				pre_exp = match.group(1)
				pre_exp_p = "{:.4E} ".format(float(pre_exp)*math.exp(opt_x[reaction_index.index(i)]*math.log(unsrt_data[i])))
				new_reaction = reaction.replace(pre_exp, pre_exp_p)
				new_mech = new_mech.replace(reaction, new_reaction)
	
	mech_name = mech_file_location.split('/')[-1]
	opt_mech_name = mech_name.replace(".mech","_optimized.mech")		
	ff = open(opt_mech_name,'w')
	ff.write(new_mech)
	ff.close()	

def getTestingData(sensDir,case):
	#print(os.getcwd())
	#print(f"{sensDir}")
	y_file = open(sensDir+"/Data/Simulations/sim_data_case-"+str(case)+".lst","r").readlines()
	#print(y_file)
	#x_file = open(sensDir+"/Data/Simulations/Generator_list_case-"+str(case)+".csv","r").readlines()
	x_file = open(sensDir+"/Data/Simulations/Beta_list_case-"+str(case)+".csv","r").readlines()
	#x_file = open(sensDir+"/Data/Simulations/NormalizedBeta_list_case-"+str(case)+".csv","r").readlines()
	x_data = np.asarray([np.asarray(np.float_(i[:-1].strip(",").split(","))) for i in x_file])
	y_data = np.asarray([float(i.strip("\n").split()[1]) for i in y_file])
		
	return  x_data, y_data
#create log files containing errors response surface values and optimized values					
def make_log_files(case, reaction_index, opt_x):
	co_eff = open("response_surface_co_efficients.out",'w')
	co_eff.write("#Response surface coefficients\n")
	for i in case:
		co_eff.write("{}\t".format(i.case_index))
		for j in i.co_efficients:
			co_eff.write("{}\t".format(j))
		co_eff.write("\n")
	co_eff.close()
	
	vect = open("Optimized_vector",'w')
	vect.write("Normalized Pre Exponential Factors\n")
	for i, j in enumerate(reaction_index):
		vect.write("{}\t{}\n".format(j, opt_x[i]))
	vect.close()
	
	co_eff = open("results_and_errors.out",'w')
	co_eff.write("Unoptimized\tOptimized\t\Experiment\tOld_error\tNew_error\n")
	for i in case:
		opt_eta = i.calculated_target_value(opt_x)
		old_error = abs(i.observed - i.calculated)
		new_error = abs(i.observed - opt_eta)
		co_eff.write("{}\t{}\t{}\t{}\t{}\n".format(i.calculated, opt_eta, i.observed, old_error, new_error))
		
	co_eff.close()
	
#Checks the contents of two files created using simulations. One file containing all the locations where simulation
#should be performed and another containing the locations where it is performed. The differene is returned and used for 
#performing remaining simulations. Helps to resume simulations in case of power failure. 		
def find_missing_location(initial, progress):
	#print(len(initial))
	#print(len(progress))
	missing_locations = []
	if len(initial) == len(progress):
		print("All simulations are completed\n")
	else:	
		for i in initial:
			if i not in progress:
				missing_locations.append(i[:-1])
		print("Found all missing locations\n")
	#print(len(missing_locations))
	return missing_locations
	
