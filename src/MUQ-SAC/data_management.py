import re, os, math
import time
import numpy as np
import shutil
import subprocess
import subprocess as sub
import threading
import logging
import combustion_target_class
import make_input_file as MakeFile
from tqdm import tqdm
import multiprocessing
import sys
from pathlib import Path

# M22: the per-case solver interpreter was hardcoded to "python3.9". Keep that
# as the default (the pipeline's Cantera/solver deps live there) but let it be
# overridden without editing source, for machines where the interpreter differs.
_PY = os.environ.get("MUQSAC_PYTHON", "python3.9")

# ---------------------------------------------------------------------------
# Logging (2026-08-29): every diagnostic in this module goes to a per-study
# log file -- the terminal shows ONLY the tqdm progress bar of the re-run
# pool. The log path is captured at import time (the pipeline imports this
# module from the study directory) and is overridable via DATA_MGMT_LOG.
# propagate=False keeps records out of any root logger a host configures with
# a console handler (mso_compare's setup_logging does exactly that), so the
# terminal stays clean in every embedding context.
# ---------------------------------------------------------------------------
_LOG_PATH = os.path.abspath(
    os.environ.get("DATA_MGMT_LOG") or os.path.join(os.getcwd(), "data_management.log")
)


class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record.

    Worker.do_job_async ends with pool.terminate(), which SIGTERMs the forked
    workers without running logging shutdown -- any record still sitting in a
    worker's buffered stream would be silently lost. Per-record flush makes
    each record land on disk (O_APPEND, one write syscall) as soon as it is
    emitted, from any process.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()


logger = logging.getLogger("data_management")
logger.propagate = False
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fh = _FlushFileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(_fh)


def _supports_unicode():
    """Mirror run_script.py's glyph check so the progress bar's postfix symbols
    degrade to ASCII when stdout can't encode them."""
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    try:
        "✓✗".encode(enc or "utf-8")
        return bool(enc) and "utf" in enc
    except Exception:
        return False


_GLYPH = {
    "ok":   "✓" if _supports_unicode() else "[OK]",
    "fail": "✗" if _supports_unicode() else "[FAIL]",
}


def _new_pbar(total, desc):
    """The one progress bar the terminal gets for a re-run pool phase."""
    return tqdm(
        total=total,
        desc=desc,
        unit="sim",
        file=sys.stdout,
        ncols=100,
        disable=False,       # stdout is a pipe under run_script: the bar must render anyway
        mininterval=0.25,    # cap tee'd frame spam at ~4 fps
        bar_format="{desc:30s} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

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

def cleanup_files(workdir):
    """
    Deletes:
      1. Everything in ./output except RCM.out and tau.out.
      2. time_history.csv and ScanMan.log anywhere under workdir.

    A log of deleted files is written to:
      delete.log
      delete2.log
    """

    delete_log = os.path.join(workdir, "delete.log")
    delete2_log = os.path.join(workdir, "delete2.log")

    # ---------------------------------------------------------
    # Cleanup output directory
    # ---------------------------------------------------------
    output_dir = Path(workdir) / "output"

    with open(delete_log, "w") as log:
        if output_dir.exists():
            for file in output_dir.iterdir():
                if (
                    file.is_file()
                    and file.name not in ("RCM.out", "tau.out")
                ):
                    try:
                        file.unlink()
                        log.write(f"Deleted: {file}\n")
                    except Exception as e:
                        log.write(f"ERROR deleting {file}: {e}\n")

    # ---------------------------------------------------------
    # Cleanup time_history.csv and ScanMan.log everywhere
    # ---------------------------------------------------------
    with open(delete2_log, "w") as log:
        for root, _, files in os.walk(workdir):
            for name in files:
                if name in ("time_history.csv", "ScanMan.log"):
                    file = Path(root) / name
                    try:
                        file.unlink()
                        log.write(f"Deleted: {file}\n")
                    except Exception as e:
                        log.write(f"ERROR deleting {file}: {e}\n")

def _run_script_needs_soln2ck(solver):
	"""Whether the per-case `run` script make_input_file.py generated for this
	solver contains a `soln2ck_2.py` CHEMKIN-round-trip line.

	Must stay in lock-step with make_input_file.create_input_file's `s_run`
	templates: only the CHEMKIN_PRO and zerork branches emit `soln2ck_2.py`;
	the `solver: cantera` branch runs cantera_.py directly and never does, and
	the FlameMaster/else branch uses soln2ck_FM.py (a different command). Step 5
	of run_sim uses this so a cantera retry does not fail looking for a
	soln2ck_2.py line that by design was never written -- the bug that used to
	mask the real first-attempt failure (bug.md 2026-07-17).
	"""
	s = str(solver or "").strip()
	return ("CHEMKIN_PRO" in s) or ("zerork" in s)


def run_sim(index, case, input_, caseID, path, total):
	eta = "N/A"
	ETA = "N/A"
	
	string = path
	start = os.getcwd()
	loc = "/".join(path.split("/")[:-1])
	file_name = path.split("/")[-1]
	try:
		# -------------------------------
		# Step 1: Change directory
		# -------------------------------
		try:
			#print(f"[INFO] Initial cwd: {start}")
			#print(f"[INFO] Target path: {loc}")
			
			#path:/data/SA_SRIVATSAV/LTC/Opt/case-44/2610/output/tau.out
			os.chdir(loc)
			os.chdir("..")
			workdir = os.getcwd()
			loc_path = Path(loc)
			# 2026-08-26 idempotency fix: a retry must never destroy a
			# previously good result. If the expected output file already
			# exists and is non-empty, reuse it (parse + return) instead of
			# wiping the folder with rmtree and re-simulating -- so a failed
			# re-run can no longer destroy a good tau.out.
			_have_valid = os.path.isfile(path) and os.path.getsize(path) > 0
			if _have_valid:
				with open(path, "r") as f:
					_existing = f.readlines()
				if len(_existing) >= 2:
					_line = _existing[1].split()
					if len(_line) == 2:
						eta = np.log(float(_line[1]) * 10)
						ETA = float(_line[1])
						string = path
						return (index, eta, ETA, string, total)
				logger.warning("[WARN] %s exists but is malformed -- re-simulating.", path)
			shutil.rmtree(loc_path, ignore_errors=True)

			# Recreate the directory
			loc_path.mkdir()
			#print()
			#print(f"[INFO] Working directory changed to: {workdir}")

		except Exception:
			logger.exception("[ERROR] Failed while changing directory to path=%s", loc)
			return (index, eta, ETA, string, total)

		# -------------------------------
		# Step 2: Build paths
		# -------------------------------
		try:
			dir_name = path.split("/")[-3]
			case_name = path.split("/")[-4]
			base_path = Path("/".join(path.split("/")[:-5]))
			perturbed_path = base_path / "Perturbed_Mech"
			fallback_path = base_path / "YAML_FILES_FOR_PARTIAL_PRS" /f"{caseID}"
			
			if perturbed_path.exists():
				Perturbed_location = str(perturbed_path)
			else:
				Perturbed_location = str(fallback_path)
			
			run_cantera = os.path.join(workdir, "cantera_1.py")

			#print(f"[INFO] dir_name: {dir_name}")
			#print(f"[INFO] case_name: {case_name}")
			#print(f"[INFO] Perturbed_location: {Perturbed_location}")
			#print(f"[INFO] run_cantera path: {run_cantera}")

		except Exception:
			logger.exception("[ERROR] Failed while constructing paths")
			return (index, eta, ETA, string, total)

		# -------------------------------
		# Step 3: Create cantera input string
		# -------------------------------
		try:
			mech_path = f"{Perturbed_location}/mechanism_{dir_name}.yaml"
			#print(f"[INFO] Mechanism file path: {mech_path}")

			# Bug 18 (AMMONIA_CAMPAIGN_2026-07-22.md, bug.md): this used to
			# hardcode solver="CHEMKIN_PRO", bypassing create_input_file's own
			# `solver=None -> target.add["solver"]` fallback (make_input_file.py
			# :329-330) -- so every retry (this is the retry path; the first
			# attempt at getNominalDirectoryList/etc. never passes a solver
			# kwarg and correctly honors target.add["solver"]) silently
			# switched to Chemkin-Pro even when the target explicitly declared
			# `solver: cantera`, invoking ANSYS Chemkin-Pro and failing rc=41
			# instead of genuinely retrying with the declared solver. The
			# sibling Sensitivity_Analysis/data_management.py never passed a
			# solver kwarg at all -- that's the correct reference behavior.
			instring, a, b, c = MakeFile.create_input_file(
				caseID,
				input_,
				case,
				mech_file=mech_path,
			)

			if not instring:
				raise ValueError("MakeFile.create_input_file returned empty instring")

			#print("[INFO] cantera input string created successfully")

		except Exception:
			logger.exception("[ERROR] Failed while creating cantera input file content")
			return (index, eta, ETA, string, total)

		# -------------------------------
		# Step 4: Write cantera_1.py
		# -------------------------------
		try:
			with open(run_cantera, "w") as f:
				f.write(instring)

			#print(f"[INFO] cantera_1.py written successfully at: {run_cantera}")

			if not os.path.exists(run_cantera):
				raise FileNotFoundError(f"cantera_1.py was not created at {run_cantera}")

		except Exception:
			logger.exception("[ERROR] Failed while writing cantera_1.py")
			return (index, eta, ETA, string, total)

		# -------------------------------
		# Step 5: Run cantera_1.py
		# -------------------------------
		try:
			solve_out = os.path.join(workdir, "solve_1.out")
			log_file = os.path.join(workdir, "sol2ck_2.log")
			#print(f"[INFO] Running cantera_1.py in {workdir}")
			#print(f"[INFO] Output log: {solve_out}")
			env = os.environ.copy()
			env["NUMEXPR_MAX_THREADS"] = "1"
			
			# Execute the soln2ck command from the run script.
			#
			# The 'run' script's contents depend on which solver
			# make_input_file.py used to build it: only the CHEMKIN_PRO and
			# zerork templates contain a soln2ck_2.py CHEMKIN-round-trip line;
			# the `solver: cantera` template runs cantera_.py directly and never
			# writes a soln2ck_2.py line by design. Branch on the case's actual
			# solver so a cantera retry does NOT fail looking for a soln2ck_2.py
			# line that was never written -- that spurious
			# RuntimeError("Could not find the soln2ck_2.py command ...") used
			# to mask whatever really caused the first-attempt failure
			# (bug.md 2026-07-17 soln2ck retry-mask defect).
			solver = str(case.add.get("solver", "")).strip()
			needs_soln2ck = _run_script_needs_soln2ck(solver)

			if needs_soln2ck:
				run_script = os.path.join(workdir, "run")
				soln2ck_cmd = None
				with open(run_script, "r") as f:
					for line in f:
						# 2026-08-27: skip comment lines -- the 2026-08-26
						# empty-folder fix added a "# cleanup below only runs
						# when BOTH soln2ck_2.py and cantera_.py succeeded"
						# comment to the run template, and the old first-match
						# extraction grabbed that comment as the command.
						# Executing it (a no-op) meant mechanism.inp/thermo.dat
						# were never regenerated after a directory wipe, so the
						# Chemkin preprocess then failed rc=41 ("Failure
						# Pre-processing input files") on 2202/6025 samples of
						# pilot_5case_case0. Also split at the first '&&' so
						# only the soln2ck command itself runs here -- the rest
						# of the chain (cantera_.py + cleanup) would duplicate
						# the cantera_1.py run below.
						if "soln2ck_2.py" in line and not line.strip().startswith("#"):
							soln2ck_cmd = line.strip().split("&&")[0].strip()
							break
				if soln2ck_cmd is None:
					raise RuntimeError(
						f"solver={solver!r} run script at {run_script} is missing "
						"its expected soln2ck_2.py command."
					)

				# Regenerate mechanism.inp/thermo.dat only when missing (a
				# wiped or interrupted retry dir); a healthy dir already has
				# them, and re-running soln2ck there would rewrite files the
				# Chemkin preprocess is about to read.
				_missing_ck_inputs = (
					not os.path.isfile(os.path.join(workdir, "mechanism.inp"))
					or not os.path.isfile(os.path.join(workdir, "thermo.dat"))
				)
				if _missing_ck_inputs:
					logger.info("[INFO] mechanism.inp/thermo.dat missing in %s -- "
					            "regenerating via soln2ck", workdir)
					with open(log_file,"a") as f:

						subprocess.run(
						    soln2ck_cmd,
						    cwd=workdir,
						    shell=True,
						    check=True,
						    env=env,
						    stdout=f,
						    stderr=subprocess.STDOUT,
						)
			else:
				logger.info("[INFO] solver=%r: no soln2ck round-trip needed; "
				            "running cantera directly.", solver)
			
			
			with open(solve_out, "w") as f:
				result = subprocess.call(
					[_PY, run_cantera],
					stdout=f,
					stderr=subprocess.STDOUT,
					env=env
				)

			#print(f"[INFO] Subprocess return code: {result}")

			# M22: on a non-zero solver exit, do NOT fall through to Step 6 and
			# parse tau.out -- that file may be absent or stale from a prior run,
			# which would silently return a wrong ETA. Report N/A for this case.
			if result != 0:
				logger.warning("[WARNING] cantera_1.py exited with non-zero code: %s; "
				               "skipping tau.out parse and reporting N/A for this case.", result)
				return (index, eta, ETA, string, total)

		except Exception:
			logger.exception("[ERROR] Failed while executing cantera_1.py")
			return (index, eta, ETA, string, total)

		try:
			subprocess.run(
				"find ./output -type f ! -name 'RCM.out' ! -name 'tau.out' -delete",
				cwd=workdir,
				shell=True,
				check=True,
			)

			subprocess.run(
				"find . -type f \\( -name 'time_history.csv' -o -name 'ScanMan.log' \\) -delete",
				cwd=workdir,
				shell=True,
				check=True,
			)

		except subprocess.CalledProcessError as e:
			logger.warning("[WARNING] Cleanup command failed: %s", e)
		
		# -------------------------------
		# Step 6: Read tau.out
		# -------------------------------
		try:
			tau_file = os.path.join(loc,file_name)
			#print(f"[INFO] Looking for tau file at: {tau_file}")

			if not os.path.exists(tau_file):
				raise FileNotFoundError(f"{file_name} not found at {tau_file}")

			with open(tau_file, "r") as f:
				out_file = f.readlines()

			string = tau_file

			if len(out_file) < 2:
				raise ValueError(f"{file_name} has insufficient lines: {len(out_file)}")

			line = out_file[1].split()
			#print(f"[INFO] Parsed tau.out line: {line}")

			if len(line) == 2:
				eta = np.log(float(line[1]) * 10)
				ETA = float(line[1])
			else:
				logger.warning("[WARNING] tau.out second line does not have 2 columns, using fallback values")
				eta = np.log(100 * 10000)
				ETA = 100 * 10000

		except Exception:
			logger.exception("[ERROR] Failed while reading/parsing tau.out")
			return (index, eta, ETA, string, total)

	finally:
		try:
			os.chdir(start)
			#print(f"[INFO] Returned to original cwd: {start}")
		except Exception:
			logger.exception("[ERROR] Failed to return to original cwd: %s", start)

	return (index, eta, ETA, string, total)

class Worker():
	def __init__(self, workers):
		self.pool = multiprocessing.Pool(processes=workers)
		self.results = []
		self.progress = []
		self._pbar = None
		self._n_fail = 0
	def callback_run(self, result):
		self.results.append(result)
		self.progress.append(result[0])
		# Lazy bar: mso_compare/sim_runner._sweep_missing_simulations drives
		# this pool directly (apply_async + callbacks, no do_job_async), so
		# create the bar on the first callback using the total carried by the
		# task tuple (run_sim's last return element).
		if self._pbar is None:
			self._pbar = _new_pbar(int(result[-1]), "Re-running simulations")
		self._pbar.update(1)
		self._refresh_pbar_postfix()
		self._maybe_close_pbar()

	def callback_error(self,result):
		# Log-only: the failure count surfaces on the bar's postfix, never as
		# an extra terminal line (strictly progress-bar-only terminal).
		logger.error("pool worker failed: %r", result)
		self._n_fail += 1
		if self._pbar is not None:
			self._refresh_pbar_postfix()
			self._maybe_close_pbar()

	def _refresh_pbar_postfix(self):
		if self._pbar is None:
			return
		if self._n_fail:
			self._pbar.set_postfix_str(f"{_GLYPH['fail']}{self._n_fail}")
		elif self._pbar.n >= self._pbar.total:
			self._pbar.set_postfix_str(f"{_GLYPH['ok']} done")
		else:
			self._pbar.set_postfix_str("")

	def _maybe_close_pbar(self):
		"""Close once every task has settled (success callback or failure)."""
		if self._pbar is not None and self._pbar.n + self._n_fail >= self._pbar.total:
			self._refresh_pbar_postfix()
			self._pbar.close()
			self._pbar = None

	def do_job_async(self,location,case,optInput,caseID,desc="Re-running simulations"):
		# ONE tqdm bar is the terminal's only output for this pool phase; all
		# diagnostics go to data_management.log and errors show up solely as
		# the bar's postfix failure count.
		self._pbar = _new_pbar(len(location), f"{desc} (case-{caseID})")
		try:
			for index,args in enumerate(location):
				self.pool.apply_async(run_sim,
					  args=(index,case,optInput,caseID,args,len(location)),
					  callback=self.callback_run,error_callback=self.callback_error)
			self.pool.close()
			self.pool.join()
			self.pool.terminate()
		finally:
			if self._pbar is not None:
				self._refresh_pbar_postfix()
				self._pbar.close()
				self._pbar = None
		self.results.sort(key=lambda x: x[0])  # Sort by the second-to-last element (row identifier)

		# Extract sorted values
		sorted_eta = [result[1] for result in self.results]
		sorted_ETA = [result[2] for result in self.results]
		sorted_string = [result[3] for result in self.results]
		return sorted_eta,sorted_ETA,sorted_string

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
		
def select_case_rows(locations, case):
	"""
	Select every simulation-output location belonging to `case`, and the
	index (within that location's own path) of its case directory.

	BUG FIX (root cause of the adaptive-outlier-retraining corruption
	documented in report/ -- reproduced and confirmed on disk: retrained
	case-3's sim_data_case-3_pass-1.lst had 998 lines for a 499-row mini
	design matrix, every response duplicated, causing training error to climb
	monotonically 0.25%->11.8% over 5 retraining passes):

	The original per-caller logic was:
	    for location in locations:
	        for i in location.split("/"):
	            if i == "case-<N>":
	                data_loc.append(location)          # (A)
	    ...
	    start = pathList.index("case-<N>")              # (B)

	(A) appends the SAME location once per matching path COMPONENT, not once
	per location. Ordinary run paths (`Opt/case-3/17/run`) contain "case-3"
	exactly once, so this looked correct for years. But the adaptive
	outlier-retraining paths are nested
	(`Opt/Outlier_Resample/case-3/pass-1/case-3/17/run`) and contain
	"case-3" TWICE, so every such location was silently appended twice,
	doubling the response list.

	(B) then used `list.index(...)`, which always returns the FIRST match --
	the outer "case-3" -- so `pathList[start+1]` read "pass-1" (a directory
	name) instead of "17" (the sample index), for every retrained row.

	Fix: append each location at most once, and use the LAST ("innermost",
	closest to the actual per-sample subdirectory) occurrence of "case-<N>"
	as the anchor. For ordinary single-occurrence paths this is byte-for-byte
	identical to the old behaviour; for nested paths it is now correct.

	Returns
	-------
	list of (location, case_dir_index) tuples, in `locations` order, with no
	duplicates -- safe to enumerate() directly instead of the fragile
	`data_loc.index(i)` "row number" pattern the callers used to rely on
	(which is also silently wrong if two locations ever collide as strings).
	"""
	tag = "case-" + str(case).strip()
	rows = []
	for location in locations:
		loc = location.strip("\n")
		parts = loc.split("/")
		idx = None
		for i, p in enumerate(parts):
			if p == tag:
				idx = i   # keep overwriting -> ends on the LAST occurrence
		if idx is not None:
			rows.append((loc, idx))
	return rows


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

	# BUG FIX: see select_case_rows()'s docstring -- the previous per-component
	# `data_loc.append(...)` loop double-counted (and mis-indexed) every
	# location under a nested case-N directory, i.e. every location produced
	# by adaptive outlier retraining (Opt/Outlier_Resample/case-N/pass-P/case-N/...).
	rows = select_case_rows(locations, case)
	data_loc = [loc for loc, _ in rows]

	data = ''
	failed_sim = ""
	ETA = []
	eta = []

	# --------------------------------------------------
	# First pass: identify failed simulations
	# --------------------------------------------------
	re_run_sim_dict = {}
	count = 0

	for row_index, (i, _start) in enumerate(rows):
		eta_, ETA_, file_path = extract_output(
			t_list[case],
			fuel,
			i[:-4] + "/output/",
			row_index,
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
		logger.info("For Case-%s: total of %s simulations out of %s did not "
		            "return expected results -- re-running those simulations.",
		            case, len(re_run_loc), len(data_loc))

		W = Worker(100)

		sorted_eta, sorted_ETA, sorted_Path = W.do_job_async(
			re_run_loc,
			t_list[case],
			input_,
			case
		)

		del W

		logger.info("Re-run of %s simulations for Case-%s ended.", len(re_run_loc), case)

	# --------------------------------------------------
	# Second pass: collect final results
	# --------------------------------------------------
	folderName = []

	for row_index, (i, start) in enumerate(rows):
		pathList = i.split("/")
		file_loc = open("./eta_file_location.txt", "+a")

		eta_, ETA_, file_path = extract_output(
			t_list[case],
			fuel,
			i[:-4] + "/output/",
			row_index,
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
	# BUG FIX: see select_case_rows()'s docstring -- the previous per-component
	# `data_loc.append(...)` loop double-counted (and mis-indexed) every
	# location under a nested case-N directory, i.e. every location produced
	# by adaptive outlier retraining (Opt/Outlier_Resample/case-N/pass-P/case-N/...).
	rows = select_case_rows(locations, case)
	data_loc = [loc for loc, _ in rows]
	data =''
	failed_sim = ""
	ETA = []
	eta = []
	# ── Bounded retry loop (2026-08-26 empty-folder fix) ──────────────────
	# Previously this was a single one-shot retry with NO verification: a
	# sample whose output/ folder was missing, EMPTY, or failed was re-run
	# once, and if that attempt failed too the sample was silently dropped
	# to failed_sim below. Now the detect -> re-run -> re-extract cycle
	# repeats up to max_retries (3) times, and only then lands in
	# failed_sim -- the "regenerate the required files when the folder is
	# empty" behaviour: a transient failure (e.g. an Ansys license checkout
	# hiccup under heavy parallelism) gets up to 3 fresh attempts before a
	# case row is ever reported missing.
	max_retries = 3
	for attempt in range(1, max_retries + 1):
		re_run_loc = []
		for row_index, (i, _start) in enumerate(rows):
			eta_,_,file_path = extract_output(t_list[case],fuel,i[:-3]+"output/",row_index,input_=optInputs,caseID = case)
			if "N/A" in str(eta_):
				re_run_loc.append(file_path)

		if len(re_run_loc) == 0:
			break
		logger.info("For Case-%s (attempt %s/%s): total of %s simulations out of %s "
		            "did not return expected results -- re-running those simulations.",
		            case, attempt, max_retries, len(re_run_loc), len(data_loc))
		W = Worker(input_["Counts"]["parallel_threads"])
		sorted_eta,sorted_ETA,sorted_Path = W.do_job_async(
			re_run_loc,t_list[case],optInputs,case,
			desc=f"Re-running (attempt {attempt}/{max_retries})")
		del W
		logger.info("Re-run attempt %s/%s for Case-%s ended.", attempt, max_retries, case)
	for row_index, (i, start) in enumerate(rows):
		pathList = i.split("/")
		file_loc = open("./eta_file_location.txt","+a")

		#print(i)
		eta_,ETA_,file_path = extract_output(t_list[case],fuel,i[:-3]+"output/",row_index,input_=optInputs,caseID = case)
		# H2/O2 extreme-condition sweep fix: extract_output() can legitimately
		# return file_path=None for a case that still fails after the retry
		# above (e.g. a genuinely non-physical/non-convergent condition at
		# the deliberately extreme edges of a sweep, not a transient error).
		# file_loc.write(file_path+"\n") then raised TypeError on the `+`,
		# aborting post-processing for the WHOLE batch and losing every
		# already-successful case's results. Write a placeholder instead so
		# one bad condition can't take down hundreds of good ones.
		if file_path is None:
			file_path = "N/A (extract_output returned no file_path)"
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
def _log_missing_solver_output(path, filename):
	"""(Round-2 audit fix, B8 -- WORKBENCH_AUDIT_2026-08-10.md) Print a
	diagnostic naming the case directory and, if present, the tail of the
	adjacent solver log (the `solve` file each per-case run/ directory
	captures its stdout+stderr into, one level above the `output/`
	directory extract_output is passed) -- instead of the bare
	FileNotFoundError this used to be, which gave zero indication of
	*why* the run failed (solver non-convergence vs. a crashed
	generated script vs. a missing binary all look identical as a plain
	"file not found"). Confirmed real cause in a live FFCM1 campaign:
	an unquoted fuel name in the generated cantera_.py
	(make_input_file2_0.py's `set_equivalence_ratio` template) raised
	`NameError: name 'H2' is not defined` before any simulation ran --
	invisible without the log tail this prints.
	"""
	case_dir = os.path.dirname(os.path.dirname(os.path.join(path, "")))
	log_path = os.path.join(case_dir, "solve")
	logger.warning("[extract_output] WARNING: %s missing in %r (case dir: %r) -- "
	               "treating this case as failed (eta=N/A) rather than aborting "
	               "the run.", filename, path, case_dir)
	if os.path.isfile(log_path):
		try:
			tail = open(log_path).readlines()[-10:]
			_tail_msg = "\n".join(f"[extract_output]     {line.rstrip()}" for line in tail)
			logger.warning("[extract_output]   tail of %s:\n%s", log_path, _tail_msg)
		except OSError as exc:
			logger.warning("[extract_output]   (could not read %s: %s)", log_path, exc)


def extract_output(case,fuel,path,index,input_=None,caseID=None):
	eta = string = ETA = None
	# 2026-08-26 empty-folder fix. A missing output/ folder used to be
	# silently CREATED here (os.makedirs), turning "never simulated" into
	# an empty folder that the resume machinery (simulation_manager2_0.
	# getDirectoryList) then mistook for "simulation complete". Now:
	#  - folder missing -> N/A (explicit); run_sim (re)creates it;
	#  - folder empty   -> N/A (explicit): the terminal state of a failed
	#    simulation (license failure before tau.out was written, an
	#    interrupted ./run, or the run script's find -delete cleanup).
	# The bounded retry loop in generate_target_value_tables re-runs both,
	# so neither can silently pass a case row as "done" any more.
	_outfile_by_target = {"RCM": "RCM.out", "JSR": "jsr.out", "Tig": "tau.out",
	                      "Fls": "Su.out", "PFR": "pfr.out"}
	_outname = _outfile_by_target.get(case.target.strip(), "tau.out")
	if not os.path.isdir(path):
		eta = f"N/A (output folder missing: {path})"
		ETA = eta
		return eta, ETA, path + _outname
	if not os.listdir(path):
		eta = f"N/A (output folder empty: {path})"
		ETA = eta
		return eta, ETA, path + _outname
	#print(path)
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

		elif "zerork" in case.add["solver"]:
			# make_input_file.py's zerork JSR branch writes jsr.out in the
			# exact same "#T(K)\tmole fraction\n{T}\t{MF}" shape the cantera
			# branch above does (see the Tig zerork branch's comment for why
			# this explicit branch matters -- without it "zerork" would fall
			# through with no branch at all here, same silent-fallback risk).
			string = path + "jsr.out"
			if "jsr.out" in os.listdir(path):
				out_file = open(path+"jsr.out",'r').readlines()
				line = out_file[1].split()
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	#mole fraction
				else:
					eta = np.log(100*10000)
					ETA = 100
			else:
				eta = "N/A"
				ETA = eta

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

		elif "zerork" in case.add["solver"]:
			# zerork_bridge.py / make_input_file.py's zerork Tig branch
			# writes tau.out in the exact same "#T(K)\ttau(us)\n{T}\t{tau}"
			# shape the cantera branch above does -- without this explicit
			# branch, "zerork" falls through to the bare `else:` below
			# (labeled/tuned for FlameMaster's different tau.out layout:
			# 3 tab-fields on line 0, not 2 on line 1), which would
			# silently read the wrong line and fall back to a bogus
			# eta=np.log(100*10000) default instead of erroring.
			string = path + "tau.out"
			if "tau.out" in os.listdir(path):
				out_file = open(path+"tau.out",'r').readlines()
				line = out_file[1].split()
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	#us/micro seconds
				else:
					eta = np.log(100*10000)
					ETA = 100*10000
			else:
				eta = "N/A"
				ETA = eta
		
		
		else:
			#print("Tig is the target and FlameMaster is the solver")
			string = path + "tau.out"
			# (Round-2 audit fix, B8) This was the only branch in this
			# function that opened its output file without first checking
			# `"tau.out" in os.listdir(path)` the way every sibling branch
			# above does -- a missing file (solver crash/non-convergence)
			# raised a bare, undiagnosed FileNotFoundError that aborted the
			# entire multi-hour optimization run over one bad case, instead
			# of degrading to the same "N/A" sentinel every other branch
			# already uses.
			if "tau.out" in os.listdir(path):
				out_file = open(path+"tau.out",'r').readlines()
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
			else:
				_log_missing_solver_output(path, "tau.out")
				eta = "N/A"
				ETA = eta
			#return eta,string
	elif case.target.strip() == "Fls":
		if "cantera" in case.add["solver"]:
			string = path + "Su.out"
			# (Round-2 audit fix, B8) Same missing guard as the Tig branch
			# above -- see that comment. This is the exact site that
			# crashed on the real FFCM1 campaign's SA/nominal/case-7:
			# make_input_file2_0.py's generated cantera_.py had an
			# unquoted fuel name (NameError before any simulation ran),
			# so Su.out was never written, and this open() raised a bare
			# FileNotFoundError with no indication of the real cause.
			if "Su.out" in os.listdir(path):
				out_file = open(path+"Su.out",'r').readlines()
				line = out_file[1].split()
				#print(line)
				if len(line) == 2:
					eta = np.log(float(line[1]))
					ETA = float(line[1])	#cm/seconds
				else:
					eta = np.log(200)
					ETA = 200
			else:
				_log_missing_solver_output(path, "Su.out")
				eta = "N/A"
				ETA = eta
		else:
			# Task 5B: bounded, restart-safe FlameMaster extraction.
			# CWD is restored via try/finally regardless of any exception.
			saved_cwd = os.getcwd()
			try:
				os.chdir(path)
				outfile = ""
				eta = float("nan")

				# First pass: output file may already exist (restart-safe skip)
				for fname in os.listdir():
					if fuel in fname and fname.endswith("{}".format(int(case.temperature))):
						lines = open(fname, 'r').readlines()
						for line in lines:
							if "burningVelocity" in line:
								eta = float(line.split()[2])
								string = path + fname
								outfile = fname
								break
					if outfile:
						break

				# Second pass: try alternate start profiles (bounded — at most N attempts)
				if not outfile:
					case_dir = os.path.normpath(os.path.join(path, ".."))
					eta_file  = os.path.normpath(os.path.join(path, "..", "..", "eta_file_location.txt"))
					try:
						with open(eta_file, "r") as fh:
							start_profiles = [l.strip() for l in fh if l.strip()]
					except FileNotFoundError:
						start_profiles = []

					for count, start_profile in enumerate(start_profiles, 1):
						logger.info("%s: Using profile %s", count, start_profile)
						os.chdir(case_dir)

						fm_path  = os.path.join(case_dir, "FlameMaster.input")
						tmp_path = os.path.join(case_dir, "_fm_tmp.input")
						with open(fm_path, "r") as fh:
							fm_lines = fh.readlines()
						with open(tmp_path, "w") as fh:
							for line in fm_lines:
								if "StartProfilesFile" in line.split():
									fh.write("StartProfilesFile is {}\n".format(start_profile))
								else:
									fh.write(line)
						os.replace(tmp_path, fm_path)  # atomic rename

						logger.info("\t Running FlameMaster (timeout 180 s)...")
						RunCmd(["./run"], 180).Run()

						out_dir = os.path.join(case_dir, "output")
						if not os.path.isdir(out_dir):
							continue
						for fname in os.listdir(out_dir):
							if fuel in fname and fname.endswith("{}".format(int(case.temperature))):
								lines = open(os.path.join(out_dir, fname), 'r').readlines()
								for line in lines:
									if "burningVelocity" in line:
										eta = float(line.split()[2])
										string = os.path.join(out_dir, fname)
										outfile = fname
										break
							if outfile:
								break
						if outfile:
							logger.info("Converged with profile: %s", start_profile)
							break

					if not outfile:
						uid = getattr(case, "uniqueID", path)
						logger.warning("WARNING: Fls case %s did not converge after %s profiles.",
						               uid, len(start_profiles))
			finally:
				os.chdir(saved_cwd)
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

	elif case.target.strip() == "PFR":
		# H2/O2 extreme-condition sweep fix (H2/review/flag.md §1): this
		# function previously had no branch at all for "PFR", so eta/ETA
		# stayed None for every PFR case and no Nominal value was ever
		# extracted into the pipeline's Plot/Dataset/PFR CSVs -- the study
		# worked around this by reading each case's output/pfr_full.json
		# directly. This branch closes that gap for the cantera solver
		# (make_input_file.py's PFR/cantera template writes output/pfr.out
		# in the exact same "#header\n{T}\t{value}" shape as Tig's
		# tau.out), using the same eta=ln(10*ETA) encoding Tig uses so
		# sim_runner.py's shared exp(eta)/10 cache decode recovers the
		# correct exit mole fraction. CHEMKIN_PRO/FlameMaster PFR output
		# formats were not inspected/verified in this pass -- left as
		# "N/A" rather than guessed.
		if "cantera" in case.add["solver"]:
			string = path + "pfr.out"
			if "pfr.out" in os.listdir(path):
				out_file = open(path+"pfr.out",'r').readlines()
				line = out_file[1].split()
				if len(line) == 2:
					eta = np.log(float(line[1])*10)
					ETA = float(line[1])	# exit mole fraction of case.add["species"]
				else:
					eta = "N/A"
					ETA = eta
			else:
				_log_missing_solver_output(path, "pfr.out")
				eta = "N/A"
				ETA = eta
		else:
			eta = "N/A"
			ETA = eta
	return eta,ETA,string	
#Generate an optimized mechanism based on the optimized vector. 
def generate_optimized_mechanism(mech_file_location, reaction_index, unsrt_data, opt_x):
	mech_file = open(mech_file_location,'r')
	mech = new_mech = mech_file.read()	
	for i in reaction_index:
		w = re.compile(r'\n{}:.*?->.*?a\s*?=.*?(\S*?E\S\d*?\s).*?\}}'.format(i), re.DOTALL | re.IGNORECASE) #Group(1) will extract pre exp factor
		match = re.search(w, mech)
		if match == None:
			logger.error("Unable to perturb reaction rate... Exiting")
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
		logger.info("All simulations are completed")
	else:
		for i in initial:
			if i not in progress:
				missing_locations.append(i[:-1])
		logger.info("Found %s missing locations", len(missing_locations))
	#print(len(missing_locations))
	return missing_locations
	
