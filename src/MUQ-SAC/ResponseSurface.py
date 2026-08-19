import os
import math
import json
import pickle
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
from IdentifiablePRS import FeatureMap, build_layout_blocks

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
	             responseOrder=2, selected_params=None, prs_type="Full", criteria = 2,
	             feature_map=None, ridge_alpha=0.0):
		"""
		feature_map : IdentifiablePRS.FeatureMap or None
			When given, the PRS is fitted/evaluated in the *identifiable
			subspace* `scores = compressed_zeta @ feature_map.M + feature_map.c`
			instead of directly on the raw (often near-unidentifiable, see
			IdentifiablePRS.py's module docstring) zeta coordinates. This is
			opt-in and fully backward compatible: feature_map=None reproduces
			the original raw-zeta polynomial exactly, unchanged.
			The map is affine, so evaluate()/Jacobian() keep their original
			FULL-space contract for the live SLSQP optimizer -- only what the
			compressed BZeta *means* changes; see Jacobian()'s feature_map
			branch for the chain-rule composition.
		ridge_alpha : float
			Optional L2 penalty (intercept excluded) added to the normal
			equations. 0.0 (default) reproduces the original QR solve exactly.
			Only meaningful together with feature_map; the raw-zeta path is
			so ill-conditioned that ridge alone cannot fix it (verified: the
			penalized directions carry the largest raw variance, so shrinkage
			barely helps -- see report/).
		"""
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
		self.criteria    = criteria

		self.feature_map = feature_map
		self.ridge_alpha = float(ridge_alpha)
		self._poly       = None   # lazily-built sklearn PolynomialFeatures, see _ensure_poly
		#self.flag = ""

	# ═══════════════════════════════════════════════════════════════════════
	# Identifiable-subspace polynomial engine (used only when feature_map is set)
	# ═══════════════════════════════════════════════════════════════════════

	def _ensure_poly(self, k):
		"""Lazily build/reuse the PolynomialFeatures(degree=self.order) object
		for the k-dimensional score space. Depends only on (order, k), never
		on data values, so it is safe to rebuild after a coefficient-cache
		hit (which skips MatPolyFitTransform entirely)."""
		if self._poly is None or self._poly.n_features_in_ != k:
			self._poly = PolynomialFeatures(degree=self.order, include_bias=True)
			self._poly.fit(np.zeros((1, k)))
		return self._poly

	def _poly_value_and_grad(self, scores):
		"""Generic multivariate-polynomial value + gradient at one point,
		using self._poly.powers_ (n_terms, k) and self.coeff. Replaces the
		hand-rolled, order-2-only jacobian_element() for the feature-map path
		so cubic (or higher) score-space fits get correct gradients for the
		live SLSQP optimizer without hand-deriving each order's index algebra.

		For a term coeff_t * prod_i scores_i^e_i, d/d(scores_j) is 0 unless
		e_j > 0, in which case it is
		    e_j * scores_j^(e_j-1) * prod_{i != j} scores_i^e_i.
		Computed via a zeroed-exponent-at-j "others" product so no division
		(and hence no 0/0) is ever needed, even where scores_j == 0.
		"""
		powers = self._poly.powers_                       # (n_terms, k)
		coeff  = np.asarray(self.coeff, dtype=float)
		scores = np.asarray(scores, dtype=float)
		term_vals = np.prod(np.power(scores[None, :], powers), axis=1)
		val = float(coeff @ term_vals)

		k = len(scores)
		grad = np.zeros(k)
		for j in range(k):
			e = powers[:, j]
			powers_no_j = powers.copy()
			powers_no_j[:, j] = 0
			others = np.prod(np.power(scores[None, :], powers_no_j), axis=1)
			dvj = e * np.power(scores[j], np.clip(e - 1, 0, None)) * others
			grad[j] = float(coeff @ dvj)
		return val, grad

	def build_basis(self, X_comp):
		"""Public, side-effect-free: compressed-space rows (n, n_comp) ->
		polynomial-basis matrix (n, p) in the identifiable subspace, using
		THIS instance's feature_map and order. Used internally by
		MatPolyFitTransform (feature_map path) and externally by
		AdaptiveResampler.DOptimalResampler to score new candidate rows
		against the CURRENT fit's basis for D-optimal augmentation, without
		needing self.X to be set or a fit to have happened yet (only
		feature_map + order matter -- self._poly is built/reused lazily).
		Requires feature_map to be set (raises otherwise; the legacy
		raw-zeta path has no single reusable basis builder of this kind --
		see MatPolyFitTransform for that path instead).
		"""
		if self.feature_map is None:
			raise ValueError("build_basis requires feature_map to be set")
		scores = self.feature_map.transform(np.asarray(X_comp, dtype=float))
		poly = self._ensure_poly(scores.shape[1])
		return poly.transform(scores)
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
		#if self.flag != "Zero_rxn_selected":
		if "Plots" not in os.listdir(".."):
			os.mkdir("../Plots")
		fig = plt.figure()
		ax = fig.add_subplot()
		for spine in ax.spines.values():
		    spine.set_edgecolor('black')
		    spine.set_linewidth(0.7)
		# BUG FIX: labels were swapped -- x is the simulation value (what
		# self.y_Test_simulation / self.Y hold), y is the PRS prediction.
		plt.xlabel("Direct Simulation")
		plt.ylabel("Response Surface estimation")
		plt.plot(np.asarray(self.y_Test_simulation),np.asarray(self.y_Test_Predict),"k.",ms=8,label=f"Testing (max error = {self.ytestMaxError :.3f}%)")
		plt.scatter(np.asarray(self.Y), np.asarray(self.resFramWrk), color="none", edgecolor="green",label=f"Training (max error = {self.MaxError:.3f}%)")
		x = np.linspace(0,500,1000)
		# BUG FIX: limits must cover BOTH train and test so an outlier test
		# point never falls outside the visible axes.
		lo = min(np.min(np.asarray(self.Y)), np.min(np.asarray(self.y_Test_simulation)))
		hi = max(np.max(np.asarray(self.Y)), np.max(np.asarray(self.y_Test_simulation)))
		plt.xlim(lo*0.93, hi*1.07)
		plt.ylim(lo*0.93, hi*1.07)
		ax.grid(False)
		plt.plot(x,x,"-", linewidth = 0.5, label="parity line")
		plt.legend(loc="upper left")
		print(f"Case-{index}: max-error: {self.ytestMaxError:.3f}%\n")
		plt.savefig('../Plots/Parity_plot_case_'+str(self.case_index) + str(slice_type) + str(slice_no) +'_TESTING.pdf',bbox_inches="tight")
		plt.close(fig)   # BUG FIX: figures were never closed, accumulating across cases

	def plot_parity_scaled(self, ratio=None, y_global=None, location="Plots",
	                        tag="", n_train_show=400):
		"""
		Two-panel parity plot at TWO deliberately different scales, so a
		reader can judge fit quality without either (a) an outlier point
		crushing everything else into an invisible sliver, or (b) a
		single-fold's own narrow y-range making the fit look artificially
		tight/loose relative to the PRS's true dynamic range.

		Must be called after .test(xTest, yTest) (uses self.y_Test_Predict,
		self.y_Test_simulation, self.error_testing_relative) and after
		create_response_surface() (uses self.Y, self.resFramWrk for the
		training overlay in the zoomed-out panel).

		Parameters
		----------
		ratio : optional, array of length len(y_Test_simulation)
			Band-violation ratio per test row (e.g. from
			IdentifiablePRS.violation_detail / band_valid_mask's underlying
			ratio). If given, points with ratio > 1 are drawn as red
			triangles and are EXCLUDED from the zoomed-in panel (they are
			the officially-non-scored subset -- see
			report_10fold_validation/main.tex, band_valid_mask). If not
			given, every test point is treated as valid.
		y_global : optional (lo, hi) tuple
			The PROPER scale of the PRS -- the full case's y range across
			every row in the design matrix (train+val+hold-out+outlier
			pool), NOT just this test split's own min/max. Anchors the
			LEFT (zoomed-out) panel so the viewer sees this split's error
			in the context of the response surface's true dynamic range.
			Defaults to this call's own train+test range if not supplied
			(same behaviour as the original `plot()` method).
		location, tag : output path is
			f"{location}/parity_scaled_case_{self.case_index}{tag}.pdf"

		Returns the output path.
		"""
		os.makedirs(location, exist_ok=True)
		y_true = np.asarray(self.y_Test_simulation, dtype=float)
		y_pred = np.asarray(self.y_Test_Predict, dtype=float)
		rel_err = np.asarray(self.error_testing_relative, dtype=float)
		y_train = np.asarray(self.Y, dtype=float)
		y_train_fit = np.asarray(self.resFramWrk, dtype=float)

		if ratio is None:
			valid = np.ones(len(y_true), dtype=bool)
		else:
			ratio = np.asarray(ratio, dtype=float)
			valid = ratio <= 1.0

		fig, (ax_out, ax_in) = plt.subplots(1, 2, figsize=(11.5, 5.4))

		# ---- LEFT: zoomed OUT, proper (global) PRS scale ----
		# y_global anchors the MINIMUM extent (so the panel is never scaled
		# tighter than the PRS's true dynamic range); the actual union with
		# this call's own data is always taken so an out-of-band prediction
		# is never silently clipped off the visible axes.
		lo = min(y_true.min(), y_pred.min(), y_train.min())
		hi = max(y_true.max(), y_pred.max(), y_train.max())
		if y_global is not None:
			lo = min(lo, float(y_global[0]))
			hi = max(hi, float(y_global[1]))
		pad = 0.05 * (hi - lo if hi > lo else 1.0)

		if n_train_show and len(y_train) > n_train_show:
			show = np.random.default_rng(0).choice(len(y_train), size=n_train_show, replace=False)
		else:
			show = slice(None)
		ax_out.scatter(y_train[show], y_train_fit[show], s=10, facecolors="none",
		               edgecolors="#2b8a3e", alpha=0.5, linewidths=0.6,
		               label=f"Training (max err={self.MaxError:.2f}%)", zorder=2)
		ax_out.scatter(y_true[valid], y_pred[valid], s=22, c="#2c7fb8", alpha=0.75,
		               edgecolors="none", label=f"Test, band-valid (n={int(valid.sum())})", zorder=3)
		if (~valid).any():
			ax_out.scatter(y_true[~valid], y_pred[~valid], s=40, c="#d7301f", marker="^",
			               alpha=0.85, edgecolors="k", linewidths=0.3,
			               label=f"Test, band-invalid (n={int((~valid).sum())})", zorder=4)
		ax_out.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=1, label="1:1")
		ax_out.set_xlim(lo - pad, hi + pad)
		ax_out.set_ylim(lo - pad, hi + pad)
		ax_out.set_aspect("equal", adjustable="box")
		ax_out.set_xlabel("Actual (simulated)")
		ax_out.set_ylabel("PRS-predicted")
		scale_note = "$\\geq$case-global scale" if y_global is not None else "train+test scale"
		ax_out.set_title(f"Zoomed OUT ({scale_note})\nall-points test max err={self.ytestMaxError:.2f}%",
		                  fontsize=9.5)
		ax_out.legend(fontsize=7, loc="lower right", framealpha=0.9)

		# ---- RIGHT: zoomed IN, band-valid points only ----
		yv, pv = y_true[valid], y_pred[valid]
		if len(yv):
			lo_z = min(yv.min(), pv.min()); hi_z = max(yv.max(), pv.max())
			pad_z = 0.08 * (hi_z - lo_z if hi_z > lo_z else 1.0)
		else:
			lo_z, hi_z, pad_z = lo, hi, pad
		ax_in.scatter(yv, pv, s=22, c="#2c7fb8", alpha=0.75, edgecolors="none", zorder=3)
		ax_in.plot([lo_z - pad_z, hi_z + pad_z], [lo_z - pad_z, hi_z + pad_z],
		           "k--", lw=1, zorder=1)
		ax_in.set_xlim(lo_z - pad_z, hi_z + pad_z)
		ax_in.set_ylim(lo_z - pad_z, hi_z + pad_z)
		ax_in.set_aspect("equal", adjustable="box")
		ax_in.set_xlabel("Actual (simulated)")
		ax_in.set_ylabel("PRS-predicted")
		valid_mre = float(np.max(np.abs(rel_err[valid]))) if valid.any() else float("nan")
		ax_in.set_title(f"Zoomed IN (band-valid only)\nvalid-only test max err={valid_mre:.2f}%",
		                 fontsize=9.5)

		fig.suptitle(f"Case {self.case_index}: PRS parity, proper scale", fontsize=11)
		fig.tight_layout(rect=[0, 0, 1, 0.93])
		out_path = os.path.join(location, f"parity_scaled_case_{self.case_index}{tag}.pdf")
		fig.savefig(out_path, bbox_inches="tight", dpi=150)
		fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
		plt.close(fig)
		return out_path

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
		#if self._active_idx:
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
		
		if self.ytestMaxError > self.criteria:
			self.selection = 0
		else:
			self.selection = 1
		#else:
		#	self.MaxError = 0
		#	self.ytestMaxError = 0
		return self.MaxError, self.ytestMaxError, self.criteria
	# ═══════════════════════════════════════════════════════════════════════
	# FIXED: create_response_surface  (cache check)
	# ═══════════════════════════════════════════════════════════════════════

	def create_response_surface(self):
		#if self._active_idx:
		"""
		Fit PRS on COMPRESSED self.X -- either the raw zeta polynomial (QR
		least squares, feature_map=None, byte-identical to the original) or,
		when feature_map is set, the identifiable-subspace polynomial
		(optionally ridge-penalized via ridge_alpha) built by MatPolyFitTransform.

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
			if self.feature_map is not None and self.ridge_alpha > 0:
				# Normal-equation ridge, intercept (column 0) unpenalized.
				B = np.asarray(self.BTrsMatrix, dtype=float)
				G = B.T @ B
				I = np.eye(G.shape[0]); I[0, 0] = 0.0
				self.coeff = np.linalg.solve(
					G + self.ridge_alpha * len(B) * I, B.T @ np.asarray(self.Y, dtype=float))
			else:
				self.Q, self.R  = np.linalg.qr(self.BTrsMatrix)
				y               = np.dot(np.transpose(self.Q), self.Y)
				self.coeff      = np.linalg.solve(self.R, np.transpose(y))

			rr  = open(coef_path, 'w')
			res = "Coefficients\n"
			for i in self.coeff:
				res += '{}\n'.format(float(i))
			rr.write(res)
			rr.close()

			if hasattr(self, "Q"):
				del self.BTrsMatrix, self.Q, self.R
			else:
				del self.BTrsMatrix

		# BUG FIX: resCoeffTransform (needed by the legacy evaluateResponse/
		# estimate a,b-matrix path) must run on EVERY fit, not only a fresh
		# one -- previously it was skipped entirely on a coefficient-cache
		# hit, leaving self.zero/.a/.b unset. Only meaningful for the raw
		# order-2 path (feature_map=None); score-space fits don't use it
		# (evaluate()/Jacobian(), the only methods the live optimizer calls,
		# go through evaluate_prs()/_poly_value_and_grad() instead).
		if self.order == 2 and self.feature_map is None:
			self.zero, self.a, self.b = self.resCoeffTransform(self.order)

		# Training-error diagnostics (self.X rows are compressed)
		self.resFramWrk = []
		for i in self.X:
		    self.resFramWrk.append(self.evaluate_prs(i))

		# BUG FIX: filename had no case index, so every case overwrote the
		# same file and only the last case's comparison data survived.
		fileResponse = open(os.getcwd()+'/Data/ResponseSurface/FlaMan_Response_comparison_case-'+str(self.case_index)+'.csv','w')
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
		#else:
		#	self.flag = "Zero_rxn_selected"
		#	self.selection = 0
		#	del self.X

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
		        Correct: 10 order-3 terms  = C(3+3-1, 3) = C(5,3).
		"""
		if self.feature_map is not None:
			return self.build_basis(np.asarray(self.X, dtype=float))

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
		if self.feature_map is not None:
			scores = self.feature_map.transform_row(np.asarray(x, dtype=float))
			poly   = self._ensure_poly(len(scores))
			Phi    = poly.transform(scores.reshape(1, -1))[0]
			return float(Phi @ self.coeff)

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
		x_comp = self._compress(x)   # length n_comp

		if self.feature_map is not None:
			# Chain rule through the affine feature map: scores = x_comp @ M + c,
			# so d(PRS)/d(x_comp) = M @ d(PRS)/d(scores). M is constant (affine
			# map), so this is exact for any polynomial order the map was
			# fitted with -- unlike jacobian_element below, which is hand-derived
			# for order 2 only.
			x_comp_arr = np.asarray(x_comp, dtype=float)
			scores     = self.feature_map.transform_row(x_comp_arr)
			self._ensure_poly(len(scores))
			_, grad_scores = self._poly_value_and_grad(scores)
			grad_comp = self.feature_map.M @ grad_scores
			return list(self._expand_gradient(grad_comp))

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


class RowGeneratorDiagnostics:
	"""
	Per-row, per-reaction band-violation + generator-class attribution plot.

	For one flagged design-matrix row (a row whose ResponseSurface.test()
	prediction was a large outlier -- see report_10fold_validation/
	main.tex's diagnosis), this draws, for EVERY reaction in the case's
	identifiable layout, one small panel showing:
	  - the physical uncertainty band  +-f_prior(T)  (IdentifiablePRS's own
	    quantity -- band-crossing here means band-invalid by the exact same
	    test IdentifiablePRS.violation_ratio uses, not an approximation)
	  - background samples from each of the sampler's three generator
	    classes (A/B/C -- see Uncertainty.py's _psac_class_* methods and
	    DM_FOR_PARTIAL_PRS/<case>/{a,b,c}_type_samples_partial_numpy.pkl)
	  - the flagged row's own curve for that reaction, highlighted and
	    colored by whichever generator class it matches

	The row's generator class is identified by exact nearest-neighbor
	lookup against the case's raw (uncompressed) DesignMatrix.csv -- this
	is not a guess: matches come back at essentially float-precision zero
	distance because each row's per-reaction perturbation is drawn verbatim
	from one of the three class pools before per-reaction shuffling
	assembles the joint row (confirmed 2026-07-31: both of
	report_10fold_validation's cited flagged rows -- Case 3 row 609, Case 0
	row 1702 -- matched Class B at distance ~1e-9, i.e. exactly, which is
	why Class B's broken rejection test, already diagnosed in report/
	main.tex sec:upstream item 1, is the right thing to blame).

	Usage
	-----
	    from IdentifiablePRS import build_layout_blocks
	    blocks = build_layout_blocks(layout, sigma_of, T_arr)
	    diag = RowGeneratorDiagnostics(case, blocks, rxn_index,
	                                    design_matrix_raw_csv_path,
	                                    class_pool_dir, T_min, T_max)
	    diag.plot_row(row_id=609, X_compressed=X, location="Plots")
	"""

	CLASS_COLORS = {"a": "#2ca25f", "b": "#d7301f", "c": "#2c7fb8"}
	CLASS_LABEL = {"a": "Class A (single-param sensitivity)",
	               "b": "Class B (correlated multi-param)",
	               "c": "Class C (broad exploration)"}

	def __init__(self, case, blocks, rxn_index, design_matrix_raw_csv_path,
	             class_pool_dir, T_min, T_max, n_T=150):
		"""
		case : int/str -- case identifier (for titles only).
		blocks : list of IdentifiablePRS.ReactionBlock -- this case's
		    identifiable layout (from IdentifiablePRS.build_layout_blocks).
		rxn_index : dict[str, int] -- reaction name -> column-block index
		    in the RAW (uncompressed) DesignMatrix.csv (3 columns/reaction,
		    i.e. raw_col = 3*rxn_index[rxn]).
		design_matrix_raw_csv_path : str -- path to
		    DM_FOR_PARTIAL_PRS/<case>/DesignMatrix.csv (the UNcompressed
		    per-reaction zeta, used only for generator-class lookup).
		class_pool_dir : str -- directory containing
		    {a,b,c}_type_samples_partial_numpy.pkl for this case.
		T_min, T_max : float -- temperature window (K), e.g. the case's own
		    0.85*T_target .. 1.25*T_target resampling window.
		"""
		self.case = case
		self.blocks = blocks
		self.rxn_index = rxn_index
		self.design_matrix_raw = np.asarray(
			[np.fromstring(l.strip().rstrip(","), sep=",")
			 for l in open(design_matrix_raw_csv_path) if l.strip()])
		self.pools = {}
		for cls in ("a", "b", "c"):
			path = os.path.join(class_pool_dir, f"{cls}_type_samples_partial_numpy.pkl")
			with open(path, "rb") as f:
				self.pools[cls] = pickle.load(f)
		self.T = np.linspace(T_min, T_max, n_T)
		self.Theta_full = np.column_stack(
			[np.ones_like(self.T), np.log(self.T), -1.0 / self.T])

	def identify_generator(self, row_id, rxn, tol=1e-4):
		"""Nearest-neighbor match of this row's raw per-reaction zeta
		against every class pool. Returns (class, pool_idx, distance);
		class is None if no pool entry is close (e.g. a D-optimal
		outlier-resample row, which was never part of the class-A/B/C
		pools to begin with)."""
		j = self.rxn_index[rxn]
		raw3 = self.design_matrix_raw[row_id, 3 * j: 3 * j + 3]
		best = (None, None, np.inf)
		for cls, pool_dict in self.pools.items():
			pool = pool_dict.get(rxn)
			if not pool:
				continue
			pool = np.asarray(pool)
			dist = np.linalg.norm(pool - raw3[None, :], axis=1)
			k = int(np.argmin(dist))
			if dist[k] < best[2]:
				best = (cls, k, float(dist[k]))
		cls, idx, dist = best
		if dist > tol * max(1.0, np.linalg.norm(raw3)):
			return None, None, dist
		return cls, idx, dist

	def plot_row(self, row_id, X_compressed, location="Plots", n_cols=4,
	             n_bg=10, figsize_per_panel=(4.2, 3.0)):
		"""
		X_compressed : the case's COMPRESSED design matrix (same array
		    ResponseSurface is fit/tested on, e.g. pDesignMatrix.csv) --
		    X_compressed[row_id, block.col_slice] gives each reaction's raw
		    zeta_r for this row (IdentifiablePRS's own convention).

		Returns the output path.
		"""
		os.makedirs(location, exist_ok=True)
		invT = 1.0 / self.T
		n = len(self.blocks)
		ncols = n_cols
		nrows = int(np.ceil(n / ncols))
		fig, axes = plt.subplots(nrows, ncols,
		                          figsize=(figsize_per_panel[0] * ncols,
		                                   figsize_per_panel[1] * nrows))
		axes = np.atleast_1d(axes).ravel()
		rng = np.random.default_rng(0)

		generator_by_rxn = {}
		for ax_i, block in enumerate(self.blocks):
			ax = axes[ax_i]
			rxn = block.rxn
			cls, pool_idx, dist = self.identify_generator(row_id, rxn)
			generator_by_rxn[rxn] = cls

			Theta_S = self.Theta_full[:, block.indices]
			zeta_r = X_compressed[row_id, block.col_slice]
			dk_row = Theta_S @ (block.L_r @ zeta_r)
			fp_fine = np.linalg.norm(Theta_S @ block.L_r, axis=1)

			# official ratio, exactly IdentifiablePRS.violation_ratio's per-block value
			dk3 = zeta_r @ block.W.T
			ratio = float((np.abs(dk3) / block.fp).max())

			ax.fill_between(invT, -fp_fine, fp_fine, color="steelblue", alpha=0.15, zorder=1)
			for cls_bg, pool_dict in self.pools.items():
				pool = pool_dict.get(rxn)
				if not pool:
					continue
				pool = np.asarray(pool)
				n_show = min(n_bg, len(pool))
				idxs = rng.choice(len(pool), size=n_show, replace=False)
				for k in idxs:
					zr = pool[k][list(block.indices)] if pool.shape[1] == 3 else pool[k]
					dk = Theta_S @ (block.L_r @ zr)
					ax.plot(invT, dk, color=self.CLASS_COLORS[cls_bg], lw=0.4, alpha=0.35, zorder=2)

			row_color = self.CLASS_COLORS.get(cls, "black")
			ax.plot(invT, dk_row, color=row_color, lw=2.0, zorder=4)

			status = "FAIL" if ratio > 1.0 else "ok"
			gen_tag = cls.upper() if cls else "?"
			ax.set_title(f"{rxn}\nratio={ratio:.2f} [{status}]  gen={gen_tag}",
			             fontsize=6.7, color=("#d7301f" if ratio > 1 else "#2b8a3e"))
			ax.tick_params(labelsize=6)
			for spine in ax.spines.values():
				spine.set_linewidth(0.6)

		for ax_i in range(n, len(axes)):
			axes[ax_i].axis("off")

		handles = [plt.Line2D([0], [0], color=self.CLASS_COLORS[c], lw=2, label=self.CLASS_LABEL[c])
		           for c in ("a", "b", "c")]
		handles.append(plt.Line2D([0], [0], color="black", lw=2.5,
		                           label=f"row {row_id}'s own curve (colored by its matched generator)"))
		handles.append(plt.Rectangle((0, 0), 1, 1, color="steelblue", alpha=0.3, label="physical band"))
		fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
		n_fail = sum(1 for b in self.blocks
		             if float((np.abs(X_compressed[row_id, b.col_slice] @ b.W.T) / b.fp).max()) > 1.0)
		fig.suptitle(f"Case {self.case}, row {row_id}: per-reaction band violation and generator "
		             f"attribution ({n_fail}/{n} reactions band-invalid for this row)\n"
		             r"($y=\Delta\ln\kappa(T)$ vs $1/T$; band $=\pm f_{\rm prior}(T)$)",
		             fontsize=11)
		fig.tight_layout(rect=[0, 0.04, 1, 0.93])
		out_path = os.path.join(location, f"violation_row{row_id}_case{self.case}_generators.pdf")
		fig.savefig(out_path, bbox_inches="tight", dpi=170)
		fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=170)
		plt.close(fig)
		return out_path, generator_by_rxn
