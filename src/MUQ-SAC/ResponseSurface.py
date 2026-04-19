import os
import math
import json
import statistics
import numpy as np
import scipy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt
from StastisticalAnalysis import StatisticalAnalysis
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

#
# ─── DATA-SPACE CONTRACT ────────────────────────────────────────────────────
#
# Two distinct input spaces coexist in this class.
#
# "COMPRESSED space"  (length n_comp)
#     Only the selected Arrhenius parameters.
#     self.X training rows live here.
#     self.coeff is fitted here.
#     self.a, self.b from resCoeffTransform live here.
#     Methods that receive compressed input (no filter needed):
#         evaluate_prs()   — internal polynomial evaluator
#         test()           — xTest rows come from pDesignMatrix split
#
# "FULL space"  (length n_full,  n_full >= n_comp)
#     The complete flat zeta/kappa vector from the optimiser, with zeros
#     at inactive positions for Partial PRS.
#     For Full PRS: n_full == n_comp (all params selected, no difference).
#     Methods that receive full-space input and MUST filter before use:
#         evaluate()                 — called by _eval_obj_core in OptimizationTool
#         Jacobian()                 — called by _jac_func_zeta_direct
#         evaluateResponse()         — legacy matrix-based evaluation
#         estimate()                 — legacy value + gradient
#         model_response_uncertainty()
#
# Two private helpers handle the conversion:
#     _compress(x)             full-space → compressed  (select by selected_params)
#     _expand_gradient(g_comp) compressed gradient → full-space (0 at inactive)
#
# ────────────────────────────────────────────────────────────────────────────

class ResponseSurface(object):
	def __init__(self, xdata, ydata, case, case_index,
	             responseOrder=2, selected_params=None, prs_type="Full"):
		if prs_type == "Full":
			self.selected_params = np.repeat(1, len(xdata[0]))
		else:
			self.selected_params = np.asarray(selected_params, dtype=int)

		self.X          = xdata
		self.Y          = ydata
		self.case       = case
		self.case_index = case_index
		self.order      = int(responseOrder)

		# ── space-bridging metadata ──────────────────────────────────────────
		# n_full     : length of the full optimiser vector
		# n_comp     : number of selected params = length of each xdata row
		# _active_idx: indices in full-space that are selected (selected_params==1)
		self.n_full      = len(self.selected_params)
		self.n_comp      = len(xdata[0])
		self._active_idx = [i for i, s in enumerate(self.selected_params) if s == 1]

	# ═══════════════════════════════════════════════════════════════════════
	# Space-bridging helpers
	# ═══════════════════════════════════════════════════════════════════════

	def _compress(self, x):
		"""
		Filter a full-space vector x (length n_full) down to the compressed
		subset (length n_comp) by keeping only the positions where
		selected_params == 1.

		For Full PRS this is an identity (n_full == n_comp, all selected).
		"""
		x = list(x)
		return [x[i] for i in self._active_idx]

	def _expand_gradient(self, grad_comp):
		"""
		Embed a compressed gradient (length n_comp) back into a full-space
		gradient (length n_full) with 0.0 at inactive parameter positions.

		For Full PRS this is an identity operation.
		"""
		grad_full = np.zeros(self.n_full, dtype=float)
		for local_i, global_i in enumerate(self._active_idx):
			grad_full[global_i] = grad_comp[local_i]
		return grad_full

	# ═══════════════════════════════════════════════════════════════════════
	# Unchanged methods (preserved verbatim)
	# ═══════════════════════════════════════════════════════════════════════

	def create_Neural_Network(self):
		self.regr = MLP(hidden_layer_sizes=(31,30,15),max_iter=1500).fit(self.X,self.Y)
		self.regr_fit = self.regr.predict(self.X)
		print(self.regr.score(self.X,self.Y))

	def create_SVR_response_surface(self):
		self.svr = SVR().fit(self.X,self.Y)
		self.svr_yfit = self.svr.predict(self.X)

	def DoStats_Analysis(self):
		stats_obj = StatisticalAnalysis(self.Y,self.case_index)
		stats_obj.generate_pdf(os.getcwd()+'/Data/ResponseSurface/stats_report_'+str(self.case_index)+'.pdf')

	def plot(self,index,slice_type="",slice_no=""):
		if "Plots" not in os.listdir(".."):
			os.mkdir("../Plots")		
		fig = plt.figure()
		ax = fig.add_subplot()
		for spine in ax.spines.values():
		    spine.set_edgecolor('black')
		    spine.set_linewidth(0.7) 
		plt.xlabel("Response Surface estimation")
		plt.ylabel("Direct Simulation")
		plt.plot(np.asarray(self.y_Test_simulation),np.asarray(self.y_Test_Predict),"k.",ms=8,label=f"Testing (max error = {self.ytestMaxError :.3f}%)")
		plt.scatter(np.asarray(self.Y), np.asarray(self.resFramWrk), color="none", edgecolor="green",label=f"Training (max error = {self.MaxError:.3f}%)")
		x = np.linspace(0,500,1000)
		plt.xlim(min(np.asarray(self.Y))*0.93,max(np.asarray(self.Y))*1.07)
		plt.ylim(min(np.asarray(self.Y))*0.93,max(np.asarray(self.Y))*1.07)
		ax.grid(False)
		plt.plot(x,x,"-", linewidth = 0.5, label="parity line")
		plt.legend(loc="upper left")
		print(os.getcwd(),index)
		plt.savefig('../Plots/Parity_plot_case_'+str(self.case_index) + str(slice_type) + str(slice_no) +'_TESTING.pdf',bbox_inches="tight")	

	def create_gauss_response_surface(self):
		kernel = DotProduct() + WhiteKernel()
		self.gpr = GaussianProcessRegressor(kernel=kernel,
											random_state=0).fit(self.X, self.Y)
		self.gpr_score = self.gpr.score(self.X,self.Y)
		self.gpr_yfit = self.gpr.predict(self.X)
	
	def create_Isotonic_response_surface(self):
		self.huber = QuantileRegressor(quantile=0.8).fit(self.X, self.Y)
		self.huber_yfit = self.huber.predict(self.X)
	
	def create_HuberRegressor_response_surface(self):
		self.huber = HuberRegressor().fit(self.X, self.Y)
		self.huber_yfit = self.huber.predict(self.X)
		self.coeff = self.huber.coef_
		rr = open(os.getcwd()+'/Data/ResponseSurface/huber_responsecoef_case-'+str(self.case_index)+'.csv','w')		
		res = "Coefficients\n"
		for i in self.coeff:
			res +='{}\n'.format(float(i))
		rr.write(res)
		rr.close()
		self.resFramWrk = []
		for i in self.X:
		    self.resFramWrk.append(self.evaluate_prs(i))
		fileResponse = open(os.getcwd()+'/Data/ResponseSurface/Huber_Response_comparison_.csv','w')
		simVSresp  = "Cantera,Response Surface,Error_(ln(tau)),Error(Tau)\n"
		self.RMS_error = []
		self.error = []
		self.relative_error = []
		TraError = []
		for i in range(len(self.resFramWrk)):
			self.error.append(abs(self.Y[i]-self.resFramWrk[i]))
			self.RMS_error.append(abs(self.Y[i]-self.resFramWrk[i])**2)
			self.relative_error.append((abs(self.Y[i]-self.resFramWrk[i])/(self.Y[i]))*100)
			TraError.append(1-np.exp(-(self.Y[i]-self.resFramWrk[i])))
			simVSresp +='{},{},{},{}\n'.format(self.Y[i],self.resFramWrk[i],(self.Y[i]-self.resFramWrk[i])/self.Y[i],1-np.exp(-(self.Y[i]-self.resFramWrk[i])))	
		fileResponse.write(simVSresp)		
		fileResponse.close()
		self.MaxError = max(self.error)
		self.MeanError = statistics.mean(self.error)
		self.RMS = math.sqrt(sum(self.RMS_error)/len(self.RMS_error))
		self.MaxError = max(self.relative_error)
		self.MeanError = statistics.mean(self.relative_error)
		del self.X

	def objective(self,z):
		"""(Ax-y) + penalty = 0"""
		A = self.BTrsMatrix
		prediction = A.dot(z)
		simulation = self.actualValue
		residual = (prediction - simulation)
		obj = np.dot(residual,residual.T)
		for i in z:
			obj+=i**2
		return obj

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: test()
	# ═══════════════════════════════════════════════════════════════════════

	def test(self, xTest, yTest):
		"""
		Hold-out accuracy evaluation.  xTest rows are COMPRESSED (from the
		pDesignMatrix train/test split), so evaluate_prs is called directly.

		BUG FIX — ytestMaxError used for the pass/fail criterion:
		    Original: max(error_testing_relative)
		    When all signed errors are negative (consistent over-prediction),
		    max() returns a near-zero value and the PRS is incorrectly accepted.
		    Fix: max(|error|) so the criterion is magnitude-based.
		    Signed values are kept in error_testing_relative for diagnostics.
		"""
		self.y_Test_Predict    = []
		self.y_Test_simulation = yTest

		for sample in xTest:
			# xTest is compressed — no filter needed
			self.y_Test_Predict.append(self.evaluate_prs(sample))

		self.error_testing          = []
		self.error_testing_relative = []
		for index, sample in enumerate(self.y_Test_simulation):
			pred = float(np.asarray(self.y_Test_Predict)[index])
			self.error_testing.append(pred - sample)
			# signed (%) — kept for diagnostics
			self.error_testing_relative.append(((pred - sample) / sample) * 100)

		# BUG FIX: was max(signed), now max(|signed|)
		self.ytestMaxError  = max(abs(e) for e in self.error_testing_relative)
		self.yTestMeanError = statistics.mean(self.error_testing_relative)

		if self.ytestMaxError > 6:
			self.selection = 0
		else:
			self.selection = 1

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: create_response_surface  (cache check)
	# ═══════════════════════════════════════════════════════════════════════

	def create_response_surface(self):
		"""
		Fit PRS by QR least-squares on COMPRESSED self.X.

		BUG FIX — cache check:
		    Original: if  os.getcwd() + '/.../responsecoef_case-N.csv'  is True:
		    A string is never 'is True' → the cache was NEVER loaded.
		    Fix: os.path.isfile(path).
		"""
		coef_path = (os.getcwd()
		             + '/Data/ResponseSurface/responsecoef_case-'
		             + str(self.case_index) + '.csv')

		# BUG FIX: was  ``if coef_path is True:``  → always False
		if os.path.isfile(coef_path):
			f          = open(coef_path, 'r').readlines()
			self.coeff = np.asarray([float(i) for i in f[1:]])
		else:
			self.BTrsMatrix = self.MatPolyFitTransform()
			self.Q, self.R  = np.linalg.qr(self.BTrsMatrix)
			y               = np.dot(np.transpose(self.Q), self.Y)
			self.coeff      = np.linalg.solve(self.R, np.transpose(y))

			rr  = open(coef_path, 'w')
			res = "Coefficients\n"
			for i in self.coeff:
				res += '{}\n'.format(float(i))
			rr.write(res)
			rr.close()

			if self.order == 2:
				self.zero, self.a, self.b = self.resCoeffTransform(self.order)

			del self.BTrsMatrix, self.Q, self.R

		# Training-error diagnostics (self.X rows are compressed)
		self.resFramWrk = []
		for i in self.X:
		    self.resFramWrk.append(self.evaluate_prs(i))

		fileResponse = open(os.getcwd()+'/Data/ResponseSurface/FlaMan_Response_comparison_.csv','w')
		simVSresp  = "FlameMaster,Response Surface,Error_(ln(tau)),Error(Tau)\n"
		self.RMS_error = []
		self.error = []
		self.relative_error = []
		TraError = []
		for i in range(len(self.resFramWrk)):
			self.error.append(abs(self.Y[i]-self.resFramWrk[i]))
			self.RMS_error.append(abs(self.Y[i]-self.resFramWrk[i])**2)
			self.relative_error.append((abs(self.Y[i]-self.resFramWrk[i])/(self.Y[i]))*100)
			TraError.append(1-np.exp(-(self.Y[i]-self.resFramWrk[i])))
			simVSresp +='{},{},{},{}\n'.format(self.Y[i],self.resFramWrk[i],(self.Y[i]-self.resFramWrk[i])/self.Y[i],1-np.exp(-(self.Y[i]-self.resFramWrk[i])))	
		fileResponse.write(simVSresp)		
		fileResponse.close()

		self.MaxError  = max(self.error)
		self.MeanError = statistics.mean(self.error)
		self.RMS       = math.sqrt(sum(self.RMS_error)/len(self.RMS_error))
		self.MaxError  = max(self.relative_error)
		self.MeanError = statistics.mean(self.relative_error)
		del self.X

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: MatPolyFitTransform  (order 3/4/5 index bug)
	# ═══════════════════════════════════════════════════════════════════════

	def MatPolyFitTransform(self):
		"""
		Build the polynomial feature B-matrix from COMPRESSED self.X.

		BUG FIX (order 3, 4, 5):
		    Original loops used  row.index(value)  which returns the FIRST
		    occurrence of that value in the row.  When two or more compressed
		    row elements are equal (common when many zeta samples cluster near
		    zero), row.index() always returns the same start index, producing
		    duplicate monomials and the wrong number of B-matrix columns.
		    Example with row = [0.5, 0.3, 0.5]:
		        Buggy:   21 order-3 terms  (duplicates because index(0.5) = 0 always)
		        Correct: 10 order-3 terms  = C(3+3-1, 3) = C(5,3)

		    Fix: all order-3/4/5 blocks use range-based index enumeration
		    (ii, jj, kk, ... iterating with ii<=jj<=kk<=...) which is
		    independent of element values.

		Order-1 and order-2 are unchanged (order-2 already used enumerate
		correctly with the index variable).
		"""
		BTrsMatrix = []
		for outer_row in self.X:
			tow  = self.order
			row  = list(outer_row)
			n    = len(row)
			row_ = [1]

			# order 1: linear terms  [x_i]
			if tow > 0:
				for v in row:
					row_.append(v)
				tow -= 1

			# order 2: [x_i * x_j,  i <= j]
			# (unchanged — enumerate was already correct)
			if tow > 0:
				for i, j in enumerate(row):
					for k in row[i:]:
						row_.append(j * k)
				tow -= 1

			# order 3: [x_i * x_j * x_k,  i <= j <= k]
			# BUG FIX: was  ``for i in row: for j in row[row.index(i):]``
			if tow > 0:
				for ii in range(n):
					for jj in range(ii, n):
						for kk in range(jj, n):
							row_.append(row[ii] * row[jj] * row[kk])
				tow -= 1

			# order 4
			# BUG FIX: was  ``for i,j in enumerate(row): for j in row[row.index(i):]``
			if tow > 0:
				for ii in range(n):
					for jj in range(ii, n):
						for kk in range(jj, n):
							for ll in range(kk, n):
								row_.append(row[ii] * row[jj] * row[kk] * row[ll])
				tow -= 1

			# order 5
			if tow > 0:
				for ii in range(n):
					for jj in range(ii, n):
						for kk in range(jj, n):
							for ll in range(kk, n):
								for mm in range(ll, n):
									row_.append(row[ii] * row[jj] * row[kk]
									            * row[ll] * row[mm])
				tow -= 1

			BTrsMatrix.append(row_)
		return BTrsMatrix

	# ═══════════════════════════════════════════════════════════════════════
	# Core polynomial evaluation — operates on COMPRESSED input only
	# ═══════════════════════════════════════════════════════════════════════

	def _evaluate_prs(self, x):
		"""Sklearn Huber PRS on a compressed sample."""
		return self.huber.predict(np.array([x]))[0]

	def evaluate_prs(self, x):
		"""
		Evaluate the QR polynomial on a COMPRESSED input x (length n_comp).

		NO filtering applied.  Used internally:
		  - create_response_surface: iterates self.X rows (compressed)
		  - test():  xTest rows are compressed (pDesignMatrix split)

		For the optimiser use evaluate(x) which accepts full-space input.
		"""
		BZeta = x
		coeff = self.coeff
		tow   = self.order
		val   = coeff[0]
		count = 1

		if tow > 0:
			for i in BZeta:
				val   += coeff[count] * i
				count += 1
			tow -= 1

		if tow > 0:
			for i, j in enumerate(BZeta):
				for k in BZeta[i:]:
					if count < len(coeff):
						val   += coeff[count] * j * k
						count += 1
			tow -= 1

		return val

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: evaluate  — accepts FULL-SPACE input from the optimiser
	# ═══════════════════════════════════════════════════════════════════════

	def evaluate(self, x):
		"""
		Evaluate the PRS at a FULL-SPACE optimiser point x (length n_full).

		Filters x to the compressed subset using _compress, then delegates to
		evaluate_prs.  For Full PRS _compress is an identity so behaviour is
		identical to the original.

		Called by OptimizationTool._eval_obj_core().
		"""
		return self.evaluate_prs(self._compress(x))

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: Jacobian  — accepts FULL-SPACE input, returns FULL-SPACE gradient
	# ═══════════════════════════════════════════════════════════════════════

	def Jacobian(self, x):
		"""
		Compute the full-space gradient  d(PRS)/d(x)  at FULL-SPACE point x.

		Returns a list of length n_full.  Inactive positions have gradient 0.

		BUG FIX:
		    Original: iterated over full-space x (length n_full) and passed it
		    as BZeta to jacobian_element:
		        for i, opt in enumerate(x):          # x has n_full elements
		            j.append(jacobian_element(coeff, x, opt, i, 2))
		    self.coeff was fitted on n_comp variables, so jacobian_element
		    needed 1 + n_full + n_full*(n_full+1)//2 coefficients but only had
		    1 + n_comp + n_comp*(n_comp+1)//2.
		    → IndexError on every Partial-PRS gradient call.

		    Fix:
		    Step 1. Compress x → x_comp  (length n_comp).
		    Step 2. Compute dPRS/dx_comp[k] for each k using jacobian_element
		            with the COMPRESSED BZeta — coeff and BZeta now match.
		    Step 3. Expand the compressed gradient to full space via
		            _expand_gradient  (0 at inactive positions).
		"""
		x_comp    = self._compress(x)   # length n_comp
		grad_comp = []

		for k in range(len(x_comp)):
			# BZeta is x_comp (compressed) — coefficient indexing is correct
			grad_comp.append(
			    self.jacobian_element(self.coeff, x_comp, x_comp[k], k, 2))

		return list(self._expand_gradient(grad_comp))

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: evaluateResponse  — accepts FULL-SPACE input
	# ═══════════════════════════════════════════════════════════════════════

	def evaluateResponse(self, x, cov_x=None):
		"""
		Evaluate using pre-decomposed coefficient matrices (zero, a, b).

		x is FULL-SPACE (length n_full).  cov_x is in COMPRESSED space.

		BUG FIX:
		    Original used full-space x directly:
		        val += np.dot(self.a, x)   # self.a has length n_comp
		    For Full PRS (n_full == n_comp) this was fine.
		    For Partial PRS numpy raises a shape mismatch or — worse — silently
		    computes np.dot on the first n_comp elements of full-space x,
		    which are NOT the n_comp selected parameters.
		    Fix: compress x before all dot products.
		"""
		# BUG FIX: filter to compressed space
		x_c = np.asarray(self._compress(x), dtype=float)

		val  = self.zero
		val += np.dot(self.a, x_c)

		if cov_x is not None:
			a_times_cov = np.dot(self.a, cov_x)
			variance    = np.dot(self.a, a_times_cov.T)

		if self.b is not None:
			b_times_x = np.asarray(np.dot(self.b, x_c)).flatten()
			val      += np.dot(b_times_x.T, x_c)
			if cov_x is not None:
				b_times_cov = np.dot(self.b, cov_x)
				variance   += 2 * np.trace(np.dot(b_times_cov, b_times_cov))

		if cov_x is not None:
			return val, math.sqrt(variance)
		return val

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: estimate  — accepts FULL-SPACE input, returns FULL-SPACE gradient
	# ═══════════════════════════════════════════════════════════════════════

	def estimate(self, x):
		"""
		Evaluate PRS and return (value, full-space gradient) using a/b matrices.

		x is FULL-SPACE.  Returns (float, np.ndarray of length n_full).

		BUG FIX (two issues):
		1. Same dimension mismatch as evaluateResponse: full-space x was used
		   with self.a/self.b (length n_comp).
		   Fix: compress x before dot products.
		2. Gradient was returned in compressed space (length n_comp), causing
		   a length mismatch with the optimiser, which expects n_full.
		   Fix: expand gradient to full space before returning.
		"""
		# BUG FIX: filter to compressed space
		x_c = np.asarray(self._compress(x), dtype=float)

		val           = self.zero
		val          += np.dot(self.a, x_c)
		response_grad = np.array(self.a, dtype=float)   # copy, don't mutate self.a

		if self.b is not None:
			b_times_x      = np.asarray(np.dot(self.b, x_c)).flatten()
			val           += np.dot(b_times_x.T, x_c)
			response_grad += 2 * b_times_x

		# BUG FIX: expand to full-space gradient
		return val, self._expand_gradient(response_grad)

	# ═══════════════════════════════════════════════════════════════════════
	# Coefficient decomposition (unchanged — always receives compressed data)
	# ═══════════════════════════════════════════════════════════════════════

	def resCoeffTransform(self, order):
		"""
		Decompose self.coeff into (zero, a, b) for order-2 PRS.

		self.X[0] is a COMPRESSED row (called before del self.X).
		a and b are in compressed space and are used by evaluateResponse and
		estimate, which now compress their input before calling np.dot.
		"""
		coeff = self.coeff
		tow   = order
		row   = self.X[0]      # compressed row, length n_comp
		zero  = []
		a     = []
		b     = []
		zero.append(coeff[0])
		count = 1

		if tow > 0:
			for _ in row:
				a.append(coeff[count])
				count += 1
			tow -= 1

		if tow > 0:
			for i, j in enumerate(row):
				temp = []
				for k, l in enumerate(row[i:]):
					if count < len(coeff):
						temp.append(coeff[count])
						count += 1
				b.append(list(np.zeros(len(row) - len(temp))) + temp)
			tow -= 1

		return float(self.coeff[0]), np.asarray(a), np.matrix(b)

	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: model_response_uncertainty
	# ═══════════════════════════════════════════════════════════════════════

	def model_response_uncertainty(self, x, cov, order):
		"""
		Estimate PRS model uncertainty at FULL-SPACE point x.

		BUG FIX (two issues):
		1. ``row = BZeta`` (original line 413) → NameError: BZeta undefined.
		   Fix: define BZeta from x using _compress.
		2. ``coeff = self.resCoef`` → AttributeError: no attribute 'resCoef'.
		   Fix: use self.coeff.
		"""
		# BUG FIX: was  row = BZeta  (NameError)
		BZeta = self._compress(x)

		# BUG FIX: was  coeff = self.resCoef  (AttributeError)
		coeff = self.coeff

		tow   = order
		val   = 0
		a     = 0
		b_ii  = 0
		b_ij  = 0
		count = 1

		if tow > 0:
			for i in BZeta:
				a     += coeff[count] ** 2
				count += 1
			tow -= 1

		if tow > 0:
			for i, j in enumerate(BZeta):
				for k, l in enumerate(BZeta[i:]):
					if count < len(coeff):
						if i == k:
							b_ii += coeff[count] ** 2
						else:
							b_ij += coeff[count] ** 2
						count += 1
			tow -= 1

		val = a + 2 * b_ii + b_ij
		return np.sqrt(val)

	# ═══════════════════════════════════════════════════════════════════════
	# Core Jacobian element — unchanged, but MUST receive COMPRESSED BZeta
	# ═══════════════════════════════════════════════════════════════════════


	def jacobian_element(self, coeff, BZeta, x, ind, resp_order):
		"""
		Compute d(PRS)/d(BZeta[ind]) analytically.

		BZeta MUST be the COMPRESSED input vector (length n_comp) so that the
		coefficient index count stays within len(coeff).  ind is a position
		within the compressed BZeta (0-based).

		This function is called only from Jacobian() after x has been
		compressed, ensuring coeff and BZeta are always aligned.

		For order-2 PRS:
		    f  = c0 + Σ_i c_i z_i + Σ_{i<=j} c_{ij} z_i z_j
		    df/dz_ind = c_ind
		               + 2 c_{ind,ind} z_ind          (diagonal quadratic)
		               + Σ_{j!=ind} c_{ind,j} z_j     (off-diagonal cross terms)
		"""
		tow   = resp_order
		row   = BZeta
		val   = 0
		count = 1
		index = []

		# ── linear contribution  (coefficient c_ind) ──────────────────────
		if tow > 0:
			for i, j in enumerate(BZeta):
				if i == ind:
					val   += coeff[count]
					index.append(count)
					count += 1
				else:
					count += 1
			tow -= 1

		# ── quadratic contribution ─────────────────────────────────────────
		if tow > 0:
			for i, j in enumerate(BZeta):
				l = i
				for k in BZeta[i:]:
					if i == ind and l == ind:
						# diagonal z_ind^2: d/dz_ind = 2 c * z_ind
						val   += 2 * coeff[count] * k
						index.append(count)
						count += 1
						l     += 1
					elif i == ind and l != ind:
						# cross z_ind * z_l  (l > ind): d/dz_ind = c * z_l
						val   += coeff[count] * k
						index.append(count)
						count += 1
						l     += 1
					elif i != ind and l == ind:
						# cross z_i * z_ind  (i < ind): d/dz_ind = c * z_i
						val   += coeff[count] * j
						index.append(count)
						count += 1
						l     += 1
					else:
						count += 1
						l     += 1
			tow -= 1

		return val
	
	def transformed_value_and_jacobian(self, x_full, case):
		"""
		Return the transformed model value, observed value, effective sigma,
		and the transformed Jacobian row in FULL-SPACE, exactly matching the
		residual definition in OptimizationTool2_0._eval_obj_core.

		Used by OptimizationTool2_0.build_posterior_covariance to assemble
		the weighted Gauss-Newton Hessian without duplicating the residual logic.

		Parameters
		----------
		x_full : array-like, length n_full
			Full-space flat zeta vector from the optimiser.
		case   : combustion_target object
			Must expose .target (str), .observed (float), .std_dvtn (float).

		Returns
		-------
		value : float      — model prediction in residual space
		y_obs : float      — observation in residual space
		sigma : float      — effective standard deviation in residual space
		J_eff : np.ndarray — d(value)/d(zeta), full-space, length n_full
		"""
		raw   = float(self.evaluate(x_full))
		J_raw = np.asarray(self.Jacobian(x_full), dtype=float)  # full-space gradient

		if case.target in ("Tig", "RCM"):
			# Residual: (PRS(ζ) − log(10·τ_obs)) / (σ/τ)
			value = raw
			y_obs = np.log(case.observed * 10.0)
			sigma = case.std_dvtn / case.observed   # relative sigma
			J_eff = J_raw                            # d(raw)/dζ  — linear in PRS

		elif case.target == "Fls":
			# Residual: (exp(PRS(ζ)) − S_u_obs) / σ
			value = np.exp(raw)
			y_obs = float(case.observed)
			sigma = float(case.std_dvtn)
			J_eff = value * J_raw                   # chain rule: d(exp(PRS))/dζ

		elif case.target == "Flw":
			# Residual: (PRS(ζ) − y_obs) / σ
			value = raw
			y_obs = float(case.observed)
			sigma = float(case.std_dvtn)
			J_eff = J_raw

		else:
			raise NotImplementedError(
				f"transformed_value_and_jacobian: target {case.target!r} not wired.")

		return value, float(y_obs), float(sigma), J_eff