import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

class TPhiPlotter:
    """3D plotting utilities for T–P–phi space."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_t_p_phi(self, color: str = "dilution", save_path: str = None, show: bool = False) -> None:
        """
        Plot a 3D scatter of (T, P, phi) colored by a chosen column.

        Parameters
        ----------
        color:
            Column name used for coloring (default: 'dilution').
        save_path:
            If provided, saves the figure to this path.
        show:
            If True, calls plt.show().
        """
        d = self.df.dropna(subset=["Temperature_K", "Pressure_Pa", "Phi"])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(d["Temperature_K"], d["Pressure_Pa"], d["Phi"], c=d[color] if color in d else None)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Pressure [Pa]")
        ax.set_zlabel("Phi [-]")
        if color in d:
            fig.colorbar(sc, ax=ax, label=color)
        ax.set_title("T–P–phi space")
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


class ArrheniusPlotter(object):
	def __init__(self,unsrt_object,reaction):
		self.unsrt_data = unsrt_object
		self.rxn = reaction
		self.M = 3.0/np.log(10.0)
	
	def getNominalParams(self):
		Nom = self.unsrt_data[self.rxn].nominal
		return Nom
	
	def getCholeskyCovariance(self):
		return self.unsrt_data[self.rxn].cov
	
	def getZetaMax(self):
		return self.unsrt_data[self.rxn].zeta.x

	def getTemperatures(self):
		return self.unsrt_data[self.rxn].temperatures
	
	def getTheta(self):
		T = self.getTemperatures()
		Theta = np.array([T/T,np.log(T),-1/T])
		return Theta																						
	
	def getUncertFunc(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		func = [self.M*np.linalg.norm(np.dot(L.T,i)) for i in Theta.T]
		return np.asarray(func)
	
	def getZetaUnsrtFunc(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		z = self.getZetaMax()
		func = [(i.T.dot(L.dot(z))) for i in Theta.T]
		return np.asarray(func)
	
	def getPerturbed_A_curve(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		z = np.array([1,0,0])
		zeta = self.getZetaMax()
		func = [(i.T.dot(z*(L.dot(zeta)))) for i in Theta.T]
		return np.asarray(func)
	
	def getPerturbed_n_curve(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		zeta = self.getZetaMax()
		z = np.array([0,100,0])
		func = [(i.T.dot(z*(L.dot(zeta)))) for i in Theta.T]
		return np.asarray(func)
	
	def getPerturbed_Ea_curve(self):
		nom = self.getNominalParams()
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		z = np.array([0,0,200])
		zeta = self.getZetaMax()
		func = [(i.T.dot(z*(L.dot(zeta)))) for i in Theta.T]
		#raise AssertionError("Perturbing Ea")
		return np.asarray(func)
	
	def getNominalCurve(self):
		P = self.getNominalParams()
		Theta =   self.getTheta()
		func =  [(i.T.dot(P)) for i in Theta.T]
		return np.asarray(func)
	
	def plot_uncertainty_limits(self,location="Plots"):
		self.UQ_plot_loc = location
		os.makedirs(location,exist_ok = True)
		fig = plt.figure()
		T = self.getTemperatures()
		Kappa_o = self.getNominalCurve()
		Kappa_max = self.getZetaUnsrtFunc()
		UQ_limit = self.getUncertFunc()
		plt.plot(1/T,Kappa_o,"b-",label="Nominal Curve")
		plt.plot(1/T,Kappa_o + Kappa_max,"r-",label=r"Arrhenius Curve (f($\zeta$))")
		plt.plot(1/T,Kappa_o-Kappa_max,"r-")
		plt.plot(1/T,Kappa_o+UQ_limit,"k--",label=r"Uncertainty Limits")
		plt.plot(1/T,Kappa_o-UQ_limit,"k--")
		plt.xlabel("Temperatures (1/K)")
		plt.ylabel(r"Rate Coefficient $(\kappa)$")
		plt.legend()
		plt.savefig(location+f"/{self.rxn}.pdf",bbox_inches="tight")
	
	def plot_perturbed_Arrhenius_parameters(self,location="Plots"):
		self.UQ_plot_loc = location
		os.makedirs(location,exist_ok = True)
		#print(self.rxn)
		fig = plt.figure()
		T = self.getTemperatures()
		Kappa_o = self.getNominalCurve()
		Kappa_max = self.getZetaUnsrtFunc()
		UQ_limit = self.getUncertFunc()
		Z_a = self.getPerturbed_A_curve()
		Z_n = self.getPerturbed_n_curve()
		Z_e = self.getPerturbed_Ea_curve()
		#print(Kappa_o,Z_e)
		plt.plot(1/T,Kappa_o,"b-",label="Nominal Curve")
		plt.plot(1/T,Kappa_o + Kappa_max,"r-",label=r"Arrhenius Curve (f($\zeta$))")
		plt.plot(1/T,Kappa_o-Kappa_max,"r-")
		plt.plot(1/T,Kappa_o+UQ_limit,"k--",label=r"Uncertainty Limits")
		plt.plot(1/T,Kappa_o-UQ_limit,"k--")
		plt.plot(1/T,Kappa_o+Z_a,"b--",label="Perturbing A-factor")
		plt.plot(1/T,Kappa_o+Z_n,"c--",label="Perturbing n parameter")
		plt.plot(1/T,Kappa_o+Z_e,"y--",label="Perturbing Ea")
		plt.xlabel("Temperatures (1/K)")
		plt.ylabel(r"Rate Coefficient $(\kappa)$")
		plt.legend()
		#plt.show()
		plt.savefig(location+f"/{self.rxn}.pdf",bbox_inches="tight")		


class DesignMatrixPlotter(object):
	"""
	Plots class-A Design Matrix (DM) samples for the three Arrhenius parameters
	(A, n, Ea) on a per-reaction basis.

	For each reaction the figure contains:
	  - the nominal log-rate curve  κ₀(T)
	  - the full-3-param UQ bounds  κ₀ ± M·‖L^T·θ(T)‖
	  - class-A DM sample curves (upper and lower) for:
	        • only ln(A) perturbed to 0.1·f_prior(T)   [red]
	        • only n     perturbed to 0.1·f_prior(T)   [green]
	        • only Ea    perturbed to 0.1·f_prior(T)   [magenta]

	Figures are stored at:
	    <location>/<sanitised_rxn_name>/<sanitised_rxn_name>.pdf

	Parameters
	----------
	unsrt_data : dict
	    The unsrt_data object loaded from unsrt.pkl.
	dm_A_path : str
	    Path to DesignMatrix_A.csv  (output of getSA_3P_samples with param_type="A")
	dm_n_path : str
	    Path to DesignMatrix_n.csv  (output of getSA_3P_samples with param_type="n")
	dm_Ea_path : str
	    Path to DesignMatrix_Ea.csv (output of getSA_3P_samples with param_type="Ea")
	T_min, T_max : float
	    Temperature range for the Arrhenius plot (K).
	n_T : int
	    Number of temperature points.

	Usage
	-----
	    import pickle, VisualAid as VA
	    with open('unsrt.pkl', 'rb') as f:
	        unsrt_data = pickle.load(f)
	    plotter = VA.DesignMatrixPlotter(unsrt_data)
	    plotter.plot_dm_samples(location="DM_plots")
	"""

	def __init__(self, unsrt_data,
	             dm_A_path="DesignMatrix_A.csv",
	             dm_n_path="DesignMatrix_n.csv",
	             dm_Ea_path="DesignMatrix_Ea.csv",
	             T_min=900, T_max=1200, n_T=200):
		self.unsrt_data = unsrt_data
		self.rxn_list   = list(unsrt_data.keys())
		self.M          = 3.0 / np.log(10.0)   # same convention as ArrheniusPlotter

		# Temperature grid and log-rate basis vectors Θ(T) = [1, ln T, -1/T]
		self.T     = np.linspace(T_min, T_max, n_T)
		
		self.Theta = np.array([self.T / self.T,   # row 0: ones
		                       np.log(self.T),     # row 1: ln T
		                       -1.0 / self.T])     # row 2: -1/T
		# Theta.T has shape (n_T, 3); Theta.T[i] is the 3-vector at T[i]

		# Load block-diagonal design matrices  shape: (N_rxns, total_params)
		self.dm_A  = self._load_csv(dm_A_path)
		self.dm_n  = self._load_csv(dm_n_path)
		self.dm_Ea = self._load_csv(dm_Ea_path)

		# Print shape diagnostics to catch CSV/unsrt_data mismatches early
		self._validate_dm_shapes()

	# ── private helpers ────────────────────────────────────────────────────

	@staticmethod
	def _load_csv(path):
		"""Read a CSV (tolerates trailing commas) → 2-D numpy float array."""
		rows = []
		with open(path, 'r') as fh:
			for line in fh:
				line = line.strip().rstrip(',')
				if line:
					rows.append([float(v) for v in line.split(',')])
		return np.array(rows)

	def _validate_dm_shapes(self):
		"""
		Print shape diagnostics so any CSV/unsrt_data mismatch is immediately
		visible in the log, and raise early if a DM has zero columns.
		"""
		N = len(self.rxn_list)
		print(f"\n[DesignMatrixPlotter] N_rxns = {N}")
		for tag, dm in [("A", self.dm_A), ("n", self.dm_n), ("Ea", self.dm_Ea)]:
			print(f"  DesignMatrix_{tag}.csv shape : {dm.shape}")
			if dm.ndim != 2 or dm.shape[1] == 0:
				raise ValueError(
					f"DesignMatrix_{tag} loaded with unexpected shape {dm.shape}. "
					"Check that the CSV file exists and is non-empty."
				)
		print()

	def _extract_zr(self, dm, rxn_idx):
		"""
		Extract the scalar ζ_r for reaction rxn_idx from the diagonal
		p_design_matrix (N×N).  The p_design_matrix is diagonal: element
		[i, i] holds the single reduced-space zeta for reaction i.

		Also handles the N×3N full-block layout (legacy / unoverwritten file)
		by returning the entire 3-vector so _delta_curve_partial can use it
		via the reduced-Cholesky path.  In practice sens_3_params always
		overwrites with p_design_matrix so N×N is the normal case.
		"""
		N     = len(self.rxn_list)
		ncols = dm.shape[1]
		if ncols == N:
			# p_design_matrix (N×N diagonal): ζ_r is a scalar at [i, i]
			return np.array([dm[rxn_idx, rxn_idx]])
		elif ncols == N * 3:
			# full block-diagonal (N×3N): full ζ_r of length 3
			start = rxn_idx * 3
			return np.asarray(dm[rxn_idx, start:start + 3], dtype=float)
		else:
			raise ValueError(
				f"Unexpected DM shape {dm.shape} for {N} reactions. "
				f"Expected ({N}, {N}) for p_design_matrix "
				f"or ({N}, {N * 3}) for full design_matrix."
			)

	@staticmethod
	def _sanitize(name):
		"""Replace characters that are illegal in directory / file names."""
		return re.sub(r'[\\/:*?"<>|]', '_', name).replace(' ', '_')

	def _nominal_curve(self, rxn):
		"""κ₀(T) = Θ(T)^T · P₀  (log-space nominal rate)."""
		P0 = np.asarray(self.unsrt_data[rxn].nominal)
		return np.array([self.Theta.T[i].dot(P0) for i in range(len(self.T))])

	def _uq_band(self, rxn):
		"""
		Full-3-param symmetric UQ band: M · ‖L^T · θ(T)‖ at every temperature.
		L = unsrt_data[rxn].cov  (3×3 lower-triangular Cholesky factor of Σ).
		Identical formula to ArrheniusPlotter.getUncertFunc().
		"""
		L = self.unsrt_data[rxn].cov
		return np.array([self.M * np.linalg.norm(L.T.dot(self.Theta.T[i]))
		                 for i in range(len(self.T))])

	def _delta_curve_partial(self, rxn, zr, param_idx):
		"""
		Compute the perturbation δκ(T) for a single-parameter class-A sample.

		Correct math (mirrors _psac_class_A_SA in Uncertainty.py):
		  1. L_r = get_reduced_cholesky([param_idx])[1]
		           — Cholesky of the principal submatrix Σ_r of Σ.
		           For m=1 this is a (1,1) matrix: L_r[0,0] = √Σ[p,p].
		  2. δp_r = L_r @ ζ_r        (shape (m,))
		  3. δp_full = zeros(3);  δp_full[param_idx] = δp_r[0]
		  4. δκ(T) = Θ(T)^T · δp_full   at every temperature.

		This is NOT the same as (L_full @ ζ_full_embedded) because L_r is the
		Cholesky of the marginal variance of that single parameter, whereas
		L_full[param_idx, param_idx] is the diagonal of the full Cholesky —
		the two differ whenever the Arrhenius parameters are correlated.

		Parameters
		----------
		rxn       : str   — reaction key in unsrt_data
		zr        : array — reduced-space ζ_r (length 1 for N×N DM, 3 for N×3N)
		param_idx : int   — 0=ln(A), 1=n, 2=Ea/R
		"""
		_, L_r = self.unsrt_data[rxn].get_reduced_cholesky([param_idx])
		# ζ_r for this parameter is zr[0] (scalar) in the N×N case;
		# use only the component at param_idx for safety in the N×3N case.
		zr_scalar = np.array([float(zr[0]) if len(zr) == 1 else float(zr[param_idx])])
		delta_p_r   = L_r @ zr_scalar          # shape (1,)
		delta_p_full = np.zeros(3)
		delta_p_full[param_idx] = delta_p_r[0]
		return np.array([self.Theta.T[i].dot(delta_p_full) for i in range(len(self.T))])

	# ── public plotting method ─────────────────────────────────────────────

	def plot_dm_samples(self, location="DM_plots"):
		"""
		Generate and save one PDF per reaction to
		    ``<location>/<rxn_name>/<rxn_name>.pdf``

		Parameters
		----------
		location : str
		    Root output directory (created if absent).
		"""
		inv_T = 1.0 / self.T

		for rxn_idx, rxn in enumerate(self.rxn_list):
			safe_name  = self._sanitize(rxn)
			rxn_folder = os.path.join(location, safe_name)
			os.makedirs(rxn_folder, exist_ok=True)

			# ── compute all curves ────────────────────────────────────────
			K0    = self._nominal_curve(rxn)
			uq    = self._uq_band(rxn)

			# Extract scalar ζ_r from each diagonal p_design_matrix and
			# compute δκ(T) via the reduced Cholesky L_r for that parameter.
			zr_A  = self._extract_zr(self.dm_A,  rxn_idx)
			zr_n  = self._extract_zr(self.dm_n,  rxn_idx)
			zr_Ea = self._extract_zr(self.dm_Ea, rxn_idx)

			dK_A  = self._delta_curve_partial(rxn, zr_A,  param_idx=0)
			dK_n  = self._delta_curve_partial(rxn, zr_n,  param_idx=1)
			dK_Ea = self._delta_curve_partial(rxn, zr_Ea, param_idx=2)

			# ── build figure ──────────────────────────────────────────────
			fig, ax = plt.subplots(figsize=(8, 5))

			# Nominal curve
			ax.plot(inv_T, K0,
			        'b-', lw=2.0,
			        label=r'Nominal $\kappa_0(T)$')

			# Full-UQ limits (all 3 correlated Arrhenius params)
			ax.plot(inv_T, K0 + uq,
			        'k--', lw=1.5,
			        label=r'Full UQ limits (3-param Cholesky, $\pm$)')
			ax.plot(inv_T, K0 - uq,
			        'k--', lw=1.5)

			# Class-A sample: only ln(A) perturbed
			ax.plot(inv_T, K0 + dK_A,
			        'r-', lw=1.2,
			        label=r'Class-A: $\delta\!\ln A$ only '
			              r'$(0.1\,f_{\rm prior})$')
			ax.plot(inv_T, K0 - dK_A,
			        'r-', lw=1.2)

			# Class-A sample: only n perturbed
			ax.plot(inv_T, K0 + dK_n,
			        'g-', lw=1.2,
			        label=r'Class-A: $\delta n$ only '
			              r'$(0.1\,f_{\rm prior})$')
			ax.plot(inv_T, K0 - dK_n,
			        'g-', lw=1.2)

			# Class-A sample: only Ea perturbed
			ax.plot(inv_T, K0 + dK_Ea,
			        'm-', lw=1.2,
			        label=r'Class-A: $\delta E_a$ only '
			              r'$(0.1\,f_{\rm prior})$')
			ax.plot(inv_T, K0 - dK_Ea,
			        'm-', lw=1.2)

			ax.set_xlabel(r'$1/T\;\mathrm{(K^{-1})}$', fontsize=12)
			ax.set_ylabel(r'$\ln\kappa$', fontsize=12)
			ax.set_title(rxn, fontsize=11)
			ax.legend(fontsize=8, loc='best')
			fig.tight_layout()

			out_path = os.path.join(rxn_folder, f"{safe_name}.pdf")
			fig.savefig(out_path, bbox_inches='tight')
			plt.close(fig)
			print(f"  [DM Plot] Saved: {out_path}")

		print(f"\nAll DM plots written to '{location}/'")

class PostOptPlotter:
    """
    Per-reaction Arrhenius rate-constant plot showing:
      - Prior nominal curve
      - Prior uncertainty limits (from MUQ-SAC Cholesky factor)
      - Optimised (MAP) rate constant curve
      - Posterior ±n_sigma uncertainty envelope (from MUM-PCE Gauss-Newton Hessian)

    Usage
    -----
    plotter = PostOptPlotter(unsrt_data, rxn_key, zeta_opt_rxn, Sigma_p_rxn)
    plotter.plot(location="Opt/Plots")
    """

    M = 3.0 / np.log(10.0)   # ln → log10 conversion

    def __init__(self, unsrt_data, rxn, zeta_opt, Sigma_p_rxn,
                 n_sigma=1, n_points=120):
        """
        Parameters
        ----------
        unsrt_data  : dict  — the full unsrt_data dict
        rxn         : str   — reaction key
        zeta_opt    : array — full 3-element optimised zeta for this reaction
        Sigma_p_rxn : array — 3×3 (or m×m) posterior covariance block in
                              Arrhenius-param space for this reaction
                              (from build_posterior_covariance rxn_slices)
        n_sigma     : int   — how many σ bands to draw
        n_points    : int   — temperature-grid density
        """
        self.unsrt      = unsrt_data
        self.rxn        = rxn
        self.zeta_opt   = np.asarray(zeta_opt, dtype=float)
        self.Sigma_p    = np.asarray(Sigma_p_rxn, dtype=float)
        self.n_sigma    = n_sigma
        self.n_points   = n_points

        u        = unsrt_data[rxn]
        self.T   = np.linspace(float(u.temperatures[0]),
                               float(u.temperatures[-1]), n_points)
        self.Theta = np.array([np.ones(n_points),
                                np.log(self.T),
                                -1.0 / self.T])          # shape (3, N)
        self.p0  = np.asarray(u.nominal, dtype=float)
        self.L   = np.asarray(u.cholskyDeCorrelateMat, dtype=float)

    # ── Curve helpers ──────────────────────────────────────────────────

    def _nominal_curve(self):
        return self.M * (self.Theta.T @ self.p0)

    def _prior_limits(self):
        """±1σ prior uncertainty band in log10(k)."""
        half_width = np.array([
            self.M * np.linalg.norm(self.L.T @ self.Theta[:, i])
            for i in range(self.n_points)
        ])
        k0 = self._nominal_curve()
        return k0 - half_width, k0 + half_width

    def _optimised_curve(self):
        p_opt = self.p0 + self.L @ self.zeta_opt
        return self.M * (self.Theta.T @ p_opt)

    def _posterior_band(self):
        """Posterior ±n_sigma band in log10(k) via θ^T Σ_p θ."""
        Sp = self.Sigma_p
        # If block is smaller than 3×3 (partial PRS), embed into 3×3
        if Sp.shape[0] < 3:
            u   = self.unsrt[self.rxn]
            sel = np.asarray(u.selection, dtype=int)
            act = [i for i, s in enumerate(sel) if s == 1]
            Sp_full = np.zeros((3, 3), dtype=float)
            for a, gi in enumerate(act):
                for b, gj in enumerate(act):
                    Sp_full[gi, gj] = Sp[a, b]
            Sp = Sp_full

        sigma_k = np.array([
            self.M * float(np.sqrt(max(
                self.Theta[:, i] @ Sp @ self.Theta[:, i], 0.0)))
            for i in range(self.n_points)
        ])
        k_opt = self._optimised_curve()
        return k_opt - self.n_sigma * sigma_k, k_opt + self.n_sigma * sigma_k

    # ── Main plot ──────────────────────────────────────────────────────

    def plot(self, location="Plots", figsize=(5.5, 4.0)):
        """
        Generate and save the optimisation + posterior-covariance plot.

        Saved as:  {location}/{rxn}_posterior.pdf
        """
        os.makedirs(location, exist_ok=True)

        k_nom           = self._nominal_curve()
        k_prior_lo, k_prior_hi = self._prior_limits()
        k_opt           = self._optimised_curve()
        k_post_lo, k_post_hi   = self._posterior_band()

        x = 1e3 / self.T   # 1000/T  (conventional Arrhenius x-axis)

        fig, ax = plt.subplots(figsize=figsize)

        # Prior uncertainty envelope
        ax.fill_between(x, k_prior_lo, k_prior_hi,
                        color="steelblue", alpha=0.15, label="Prior $\\pm1\\sigma$")
        ax.plot(x, k_prior_lo, "b--", lw=0.7)
        ax.plot(x, k_prior_hi, "b--", lw=0.7)

        # Nominal
        ax.plot(x, k_nom,  "b-",  lw=1.4, label="Nominal (prior mean)")

        # Optimised MAP
        ax.plot(x, k_opt,  "r-",  lw=1.8, label="Optimised (MAP)")

        # Posterior uncertainty envelope
        ax.fill_between(x, k_post_lo, k_post_hi,
                        color="tomato", alpha=0.30,
                        label=f"Posterior $\\pm{self.n_sigma}\\sigma$")
        ax.plot(x, k_post_lo, "r--", lw=0.7)
        ax.plot(x, k_post_hi, "r--", lw=0.7)

        ax.set_xlabel(r"$1000\,/\,T$  (K$^{-1}$)", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(k)$", fontsize=10)
        ax.set_title(str(self.rxn), fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, lw=0.3, alpha=0.5)

        # Second x-axis showing T directly
        ax2 = ax.twiny()
        T_ticks = np.array([700, 1000, 1500, 2000, 2500])
        T_ticks = T_ticks[(T_ticks >= self.T[0]) & (T_ticks <= self.T[-1])]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(1e3 / T_ticks)
        ax2.set_xticklabels([f"{int(t)} K" for t in T_ticks], fontsize=7)
        ax2.tick_params(axis="x", length=3)

        fig.tight_layout()
        safe_rxn = str(self.rxn).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(location, f"{safe_rxn}_posterior.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class SampleViolationPlotter:
    """
    Plots REAL design-matrix / ResponseSurface input rows for one reaction
    against its physical uncertainty band, to visualize how far specific
    accepted rows exceed the sampler's own band-validity test
    (Uncertainty._psac_class_A / IdentifiablePRS.violation_ratio) --
    see report_10fold_validation/main.tex, "Diagnosis: worst folds are
    driven by band-invalid rows".

    Unlike ArrheniusPlotter/DesignMatrixPlotter (which draw single-parameter
    class-A sensitivity curves), this plots the FULL multi-parameter
    perturbation of whatever rows are handed in -- typically actual rows
    pulled straight from a case's compressed design matrix
    (DM_FOR_PARTIAL_PRS/<case>/pDesignMatrix.csv, the same array
    ResponseSurface.ResponseSurface is fit/tested on), so what is drawn is
    exactly the PRS's own input, not a synthetic sensitivity sample.

    Uses the SAME geometry as IdentifiablePRS.violation_ratio (reduced
    Cholesky L_r restricted to `indices`, Theta basis [1, ln T, -1/T], no
    log10 rescale) so the plotted +-band and the printed violation ratios
    are numerically identical to the ones the 10-fold CV diagnosis reports
    -- band-crossing on this plot means band-invalid by that same test, not
    an approximation of it.

    Parameters
    ----------
    rxn : str
        Reaction key (must match the folder name under Uncertainty_Matrices/).
    nominal : array (3,)
        [ln(A), n, Ea/R] nominal Arrhenius parameters (Ea/R in Kelvin).
        Build from Uncertainty_Matrices/<rxn>/nominal.csv via
        `SampleViolationPlotter.load_nominal(path)`.
    L_r : array (m, m)
        Reduced Cholesky factor for this reaction's SELECTED parameters
        (IdentifiablePRS.reduced_cholesky(Sigma, indices)[1] / a
        ReactionBlock's .L_r -- must be built with the same `indices`).
    indices : sequence of int
        Which of (0=ln A, 1=n, 2=Ea/R) are selected/identifiable for this
        reaction in this case (a ReactionBlock's .indices).
    T_min, T_max : float
        Temperature window to draw (K). Pass the case's own resampling
        window (e.g. 0.85*T_target to 1.25*T_target) to match the exact
        band the sampler/violation test used.
    """

    def __init__(self, rxn, nominal, L_r, indices, T_min, T_max, n_T=200):
        self.rxn = rxn
        self.P0 = np.asarray(nominal, dtype=float)          # (3,)
        self.L_r = np.asarray(L_r, dtype=float)              # (m, m)
        self.indices = list(indices)
        self.T = np.linspace(T_min, T_max, n_T)
        Theta_full = np.column_stack([np.ones_like(self.T), np.log(self.T), -1.0 / self.T])  # (n_T,3)
        self.Theta_full = Theta_full
        self.Theta_S = Theta_full[:, self.indices]           # (n_T, m)

    @staticmethod
    def load_nominal(path):
        """Parse Uncertainty_Matrices/<rxn>/nominal.csv -> [ln(A), n, Ea/R]."""
        vals = {}
        with open(path) as f:
            next(f)  # header: Parameter,Value
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k, v = line.split(",")
                vals[k.strip()] = float(v)
        return np.array([np.log(vals["A"]), vals["n"], vals["Ea"]])

    def nominal_curve(self):
        """kappa_0(T) = Theta_full(T) . P0, full 3-parameter nominal curve."""
        return self.Theta_full @ self.P0

    def band(self):
        """f_prior(T) = ||Theta_S(T) . L_r||_2 -- identical quantity
        IdentifiablePRS.reaction_W_fprior computes."""
        W = self.Theta_S @ self.L_r          # (n_T, m)
        return np.linalg.norm(W, axis=1)     # (n_T,)

    def sample_curve(self, zeta_r):
        """delta_kappa(T) for one real design-matrix row's zeta_r (m,)."""
        zeta_r = np.asarray(zeta_r, dtype=float)
        return self.Theta_S @ (self.L_r @ zeta_r)   # (n_T,)

    def violation_ratio_curve(self, zeta_r):
        return np.abs(self.sample_curve(zeta_r)) / self.band()

    def plot(self, valid_samples, invalid_samples, location="Plots", case=None,
              n_valid_show=15, figsize=(12.5, 5.5)):
        """
        valid_samples / invalid_samples: list of (zeta_r, row_id, ratio)
        tuples -- ratio is the row's max-over-T violation ratio (whatever
        the caller already computed via IdentifiablePRS.violation_detail),
        row_id is the design-matrix row index for the legend/annotation.

        Two panels: LEFT is the full range (so the invalid sample's actual
        scale is visible -- it can be 10s of nominal-curve-widths away),
        RIGHT is zoomed to the nominal curve + physical band + valid
        samples only (the invalid curve is off-scale here on purpose --
        an inset arrow/label marks which direction it leaves in), so the
        physical band itself, which is invisible at the full-range scale
        once ratio is ~10-30x, remains readable.
        """
        os.makedirs(location, exist_ok=True)
        K0 = self.nominal_curve()
        fp = self.band()
        invT = 1.0 / self.T

        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=figsize)

        def draw_common(ax, show_legend):
            ax.fill_between(invT, K0 - fp, K0 + fp, color="steelblue", alpha=0.18,
                             label=r"Physical band $\pm f_{\rm prior}(T)$", zorder=1)
            ax.plot(invT, K0, "b-", lw=1.8, label=r"Nominal $\kappa_0(T)$", zorder=2)
            for i, (zeta_r, row_id, ratio) in enumerate(valid_samples[:n_valid_show]):
                dk = self.sample_curve(zeta_r)
                lbl = (f"band-valid samples (n shown={min(n_valid_show, len(valid_samples))})"
                       if (i == 0 and show_legend) else None)
                ax.plot(invT, K0 + dk, color="#888888", lw=0.7, alpha=0.7, zorder=3, label=lbl)
            ax.set_xlabel(r"$1/T\;\mathrm{(K^{-1})}$", fontsize=11)
            ax.set_ylabel(r"$\ln\kappa$ (this reaction's contribution)", fontsize=11)

        draw_common(ax_full, show_legend=True)
        colors = plt.cm.autumn(np.linspace(0, 0.6, max(len(invalid_samples), 1)))
        for i, (zeta_r, row_id, ratio) in enumerate(invalid_samples):
            dk = self.sample_curve(zeta_r)
            ax_full.plot(invT, K0 + dk, color=colors[i], lw=2.2, zorder=4,
                         label=f"row {row_id}: ratio={ratio:.1f}$\\times$ band")
        ax_full.set_title("Full range: nominal, band, and the band-invalid sample", fontsize=10)
        ax_full.legend(fontsize=7.5, loc="best")

        draw_common(ax_zoom, show_legend=False)
        y_zoom_lo = min((K0 - fp).min(), *[
            (K0 + self.sample_curve(z)).min() for z, *_ in valid_samples[:n_valid_show]])
        y_zoom_hi = max((K0 + fp).max(), *[
            (K0 + self.sample_curve(z)).max() for z, *_ in valid_samples[:n_valid_show]])
        pad = 0.1 * (y_zoom_hi - y_zoom_lo)
        ax_zoom.set_ylim(y_zoom_lo - pad, y_zoom_hi + pad)
        for i, (zeta_r, row_id, ratio) in enumerate(invalid_samples):
            dk = self.sample_curve(zeta_r)
            curve = K0 + dk
            direction = "above" if curve.mean() > y_zoom_hi else "below"
            y_edge = y_zoom_hi if direction == "above" else y_zoom_lo
            x_mid = invT[len(invT) // 2]
            ax_zoom.annotate(f"row {row_id} leaves the panel {direction}\n(ratio={ratio:.1f}$\\times$ band)",
                              xy=(x_mid, y_edge), xytext=(x_mid, y_edge - pad * (0.6 if direction == "above" else -0.6)),
                              ha="center", fontsize=7.5, color=colors[i],
                              arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2))
        ax_zoom.set_title("Zoom: physical band + band-valid samples only", fontsize=10)

        title = self.rxn if case is None else f"Case {case}: {self.rxn}"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        safe = str(self.rxn).replace("/", "_").replace(" ", "_")
        tag = f"_case{case}" if case is not None else ""
        out_path = os.path.join(location, f"violation_{safe}{tag}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Resampling-fix figures (2026-08-19 investigation, THREE-PARAM-MSO-LLNL-N-
# HEPTANE_159_CORRELATION_AWARE): the class-B m=3 sampler bug (root cause 1)
# and the reduced-vs-full band-validity metric bug (root cause 2), before and
# after the fix, plus the correlation-aware parameter-selection ablation.
# Every axis is labelled in plain, standard terms -- "perturbation size /
# stated uncertainty limit", "simulated ignition delay", "predicted ignition
# delay" -- never an internal code name (no "zeta", "PRS", "band_fprior" on
# an axis; those stay in code comments and captions only).
# ═══════════════════════════════════════════════════════════════════════════

class BandValidityComparisonPlotter:
    """
    Histogram of "how far outside its stated uncertainty limit a sample
    lands" (violation ratio = 1.0 means exactly on the limit; <1 means
    inside it), before vs after fixing the class-B (m=3) sampler.

    Data: CASE_STUDIES/.../classB_before_after.json (written by a one-off
    diagnostic script during the 2026-08-19 investigation: "before" is the
    ACTUAL on-disk sample pool the buggy sampler produced for one reaction;
    "after" is a fresh draw through the same code path with the fix
    applied). Both ratios are computed identically (full 3-parameter
    physical band, see IdentifiablePRS.reaction_W_fprior's docstring), so
    the two histograms are directly comparable.
    """

    def __init__(self, before_ratio, after_ratio, rxn_label=""):
        self.before = np.asarray(before_ratio, dtype=float)
        self.after = np.asarray(after_ratio, dtype=float)
        self.rxn_label = rxn_label

    def plot(self, location="Plots", figsize=(7.5, 5)):
        os.makedirs(location, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        # Log-x: the "before" distribution spans nearly two orders of
        # magnitude (0.1x-80x the limit) -- a linear axis either hides the
        # bulk of the (small) inside-limit samples or piles the long tail
        # into one misleading spike. A floor avoids log(0) for any exactly-
        # zero perturbation.
        lo = max(1e-2, min(self.before[self.before > 0].min(initial=1e-2),
                            self.after[self.after > 0].min(initial=1e-2)))
        hi = max(self.before.max(), self.after.max())
        bins = np.logspace(np.log10(lo), np.log10(hi * 1.05), 40)
        ax.hist(np.maximum(self.before, lo), bins=bins, alpha=0.55, color="#d62728",
                label=f"Before fix  (n={len(self.before)}, "
                      f"{100*np.mean(self.before <= 1.0):.0f}% inside the limit)")
        ax.hist(np.maximum(self.after, lo), bins=bins, alpha=0.6, color="#2ca02c",
                label=f"After fix  (n={len(self.after)}, "
                      f"{100*np.mean(self.after <= 1.0):.0f}% inside the limit)")
        ax.axvline(1.0, color="k", ls="--", lw=1.3, label="Stated uncertainty limit")
        ax.set_xscale("log")
        ax.set_xlabel("Perturbation size, as a multiple of the reaction's stated uncertainty limit "
                       "(log scale)", fontsize=11)
        ax.set_ylabel("Number of samples", fontsize=11)
        title = "Sample validity before vs after the class-B sampler fix"
        if self.rxn_label:
            title += f"\n{self.rxn_label}"
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
        fig.tight_layout()
        out_path = os.path.join(location, "band_validity_before_after.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class ArrheniusBeforeAfterPlotter:
    """
    Two-panel reaction-rate plot (same geometry/labelling as
    SampleViolationPlotter): nominal rate-constant curve, shaded stated-
    uncertainty band, and a handful of sample curves, LEFT panel before the
    class-B sampler fix (mostly outside the band) vs RIGHT panel after
    (all inside, by construction -- see Uncertainty._psac_class_B_m3_'s
    rejection step added 2026-08-19).

    Data: same classB_before_after.json as BandValidityComparisonPlotter.
    """

    def __init__(self, rxn, L_full, T_min, T_max, n_T=200):
        self.rxn = rxn
        self.L_full = np.asarray(L_full, dtype=float)   # (3,3)
        self.T = np.linspace(T_min, T_max, n_T)
        self.Theta = np.column_stack([np.ones_like(self.T), np.log(self.T), -1.0 / self.T])  # (n_T,3)

    def band(self):
        return np.linalg.norm(self.Theta @ self.L_full, axis=1)

    def curve(self, zeta):
        return self.Theta @ (self.L_full @ np.asarray(zeta, dtype=float))

    def plot(self, before_zetas, after_zetas, location="Plots", n_show=25, figsize=(11, 5)):
        os.makedirs(location, exist_ok=True)
        fp = self.band()
        invT = 1.0 / self.T   # K^-1; NOT sharing a y-axis below so each panel's own
                               # scale stays readable (the "after" band is ~50x
                               # narrower than "before"'s sample spread)
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=figsize)

        def draw(ax, zetas, title, color):
            ax.fill_between(invT, -fp, fp, color="steelblue", alpha=0.35,
                             label="Stated uncertainty limit", zorder=1)
            ax.axhline(0, color="b", lw=1.5, label="Nominal rate constant", zorder=2)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(zetas), size=min(n_show, len(zetas)), replace=False)
            for j, i in enumerate(idx):
                dk = self.curve(zetas[i])
                ax.plot(invT, dk, color=color, lw=0.9, alpha=0.7, zorder=3,
                        label=("Sample perturbations" if j == 0 else None))
            ax.set_xlabel(r"$1/T\;\mathrm{(K^{-1})}$", fontsize=11)
            ax.set_ylabel(r"Change in $\ln(\mathrm{rate\ constant})$ from nominal", fontsize=10.5)
            ax.set_title(title, fontsize=10.5)
            ax.legend(fontsize=8, loc="best")

        draw(ax_b, np.asarray(before_zetas), f"Before fix ({len(before_zetas)} samples shown: "
             f"{min(n_show, len(before_zetas))})", "#d62728")
        draw(ax_a, np.asarray(after_zetas), f"After fix ({min(n_show, len(after_zetas))} of "
             f"{len(after_zetas)} samples shown)", "#2ca02c")

        fig.suptitle(f"{self.rxn}: sampled perturbations vs the stated uncertainty limit", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        safe = str(self.rxn).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(location, f"arrhenius_before_after_{safe}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class ParityBeforeAfterPlotter:
    """
    Predicted vs simulated ignition delay, side by side: LEFT is the
    original (buggy) fit -- reduced-subspace band metric, unscreened
    tolerance search, scored on ALL points including physically-invalid
    ones; RIGHT is the fixed fit -- full physical-band metric, trained and
    scored on band-valid rows only. Both panels show train/validation/hold-
    out points in different markers, with the perfect-prediction diagonal
    and the maximum relative error annotated -- standard parity-plot
    conventions, ready for a journal figure.

    Data: case_parity_dump.py's per-case JSON ("before" / "after" keys,
    each with y_{train,val,hold} / yhat_{train,val,hold}).
    """

    _SPLIT_STYLE = {"train": dict(marker="o", s=14, alpha=0.5, label="Training set"),
                     "val": dict(marker="^", s=22, alpha=0.7, label="Validation set"),
                     "hold": dict(marker="s", s=30, alpha=0.85, label="Untouched hold-out set")}

    @staticmethod
    def _max_rel_err(y, yhat):
        y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
        return float(np.max(np.abs((yhat - y) / y)) * 100)

    def _panel(self, ax, variant, title):
        all_y = []
        for split, style in self._SPLIT_STYLE.items():
            y = np.asarray(variant[f"y_{split}"], dtype=float)
            yhat = np.asarray(variant[f"yhat_{split}"], dtype=float)
            all_y.append(y); all_y.append(yhat)
            ax.scatter(y, yhat, **style, edgecolors="none")
        all_y = np.concatenate(all_y)
        lo, hi = np.nanmin(all_y), np.nanmax(all_y)
        pad = 0.03 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.1, label="Perfect prediction", zorder=0)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        mre_hold = self._max_rel_err(variant["y_hold"], variant["yhat_hold"])
        ax.set_title(f"{title}\nmax relative error, hold-out = {mre_hold:.2f}%", fontsize=10.5)
        ax.set_xlabel(r"Simulated ignition delay, $\ln(\tau)$", fontsize=10.5)
        ax.set_ylabel(r"Predicted ignition delay, $\ln(\tau)$", fontsize=10.5)
        ax.legend(fontsize=7.5, loc="upper left")

    def plot(self, before, after, case=None, location="Plots", figsize=(11.5, 5.2)):
        os.makedirs(location, exist_ok=True)
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=figsize)
        self._panel(ax_b, before, "Before fix (reduced-subspace metric,\nunscreened, scored on all rows)")
        self._panel(ax_a, after, "After fix (full physical-band metric,\ntrained \\& scored on band-valid rows)")
        title = "Predicted vs simulated ignition delay"
        if case is not None:
            title = f"Case {case}: " + title
        fig.suptitle(title, fontsize=12.5)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        tag = f"_case_{case}" if case is not None else ""
        out_path = os.path.join(location, f"parity_before_after{tag}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class CorrelationAwareParityPlotter:
    """
    Same parity-plot machinery as ParityBeforeAfterPlotter, but for the
    correlation-aware ablation: LEFT is "aware" (the regressor sees every
    selected parameter, including the ones cross-target correlation
    analysis rescued below their own sensitivity threshold), RIGHT is
    "unaware" (those specific columns hidden from the regressor, same rows,
    same hyperparameters -- see case_parity_dump.py / case_band_valid_fit.
    rescued_columns). Use on a case with a high rescued-parameter count
    (correlation_ablation_kfold.csv, sorted by delta) to show where
    correlation-aware selection earns its keep.
    """

    def __init__(self):
        self._base = ParityBeforeAfterPlotter()

    def plot(self, aware, unaware, case=None, n_rescued=None, location="Plots", figsize=(11.5, 5.2)):
        os.makedirs(location, exist_ok=True)
        fig, (ax_a, ax_u) = plt.subplots(1, 2, figsize=figsize)
        self._base._panel(ax_a, aware, "Correlation-aware\n(every selected parameter visible)")
        self._base._panel(ax_u, unaware, "Correlation-unaware\n(rescued parameters hidden)")
        title = "Predicted vs simulated ignition delay: effect of correlation-aware selection"
        if case is not None:
            title = f"Case {case} " + (f"({n_rescued} rescued parameters): " if n_rescued else ": ") + title
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        tag = f"_case_{case}" if case is not None else ""
        out_path = os.path.join(location, f"parity_correlation_aware{tag}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class CaseErrorSummaryPlotter:
    """
    Per-case maximum relative error, before vs after the resampling fix,
    as a horizontal bar pair per case with the 2% acceptance line marked.
    Data: case_before_fit.py's and case_band_valid_fit.py's output CSVs,
    joined on Case_ID. "Before" uses the all-points hold-out number (what
    Prediction_errors.csv historically reported); "after" uses the band-
    valid hold-out number (what the pipeline's PASS/FAIL gate actually
    uses) -- see run_script.py's per-case loop for why these are the two
    numbers that matter, not the two happening to be smallest.
    """

    def plot(self, cases, before_mre, after_mre, criteria=2.0, location="Plots",
             figsize=None, cap_display=50.0):
        os.makedirs(location, exist_ok=True)
        cases = list(cases)
        n = len(cases)
        order = np.argsort(after_mre)[::-1]
        cases = [cases[i] for i in order]
        before_mre = np.asarray(before_mre, dtype=float)[order]
        after_mre = np.asarray(after_mre, dtype=float)[order]
        figsize = figsize or (8.5, max(4.0, 0.19 * n))
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(n)
        h = 0.38
        before_capped = np.minimum(before_mre, cap_display)
        ax.barh(y + h / 2, before_capped, height=h, color="#d62728", alpha=0.75, label="Before fix (all-points)")
        ax.barh(y - h / 2, after_mre, height=h, color="#2ca02c", alpha=0.85, label="After fix (band-valid)")
        for i, v in enumerate(before_mre):
            if v > cap_display:
                ax.annotate(f"{v:.0f}%", xy=(cap_display, y[i] + h / 2), va="center", fontsize=6.5,
                            xytext=(2, 0), textcoords="offset points")
        ax.axvline(criteria, color="k", ls="--", lw=1.2, label=f"{criteria:.0f}% acceptance criterion")
        ax.set_yticks(y); ax.set_yticklabels([f"Case {c}" for c in cases], fontsize=7)
        ax.set_xlabel("Maximum relative error, hold-out set (%)", fontsize=11)
        ax.set_xlim(0, cap_display * 108 / 100)
        ax.set_title("Hold-out error per case, before vs after the resampling fix", fontsize=11.5)
        ax.legend(fontsize=8.5, loc="lower right")
        fig.tight_layout()
        out_path = os.path.join(location, "case_error_summary_before_after.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path


class CorrelationAwareErrorBarPlotter:
    """
    Per-case worst-of-10-fold MRE, correlation-aware vs correlation-
    unaware, bars ordered by how many parameters correlation-aware
    selection rescued for that case -- makes a "more rescued parameters ->
    bigger correlation-aware benefit" trend (if present) visible directly,
    rather than asserting it. Data: correlation_ablation_kfold.py's output
    CSV (the ROBUST, worst-of-10-fold ablation -- see that script's
    docstring for why the single-split version was replaced).
    """

    def plot(self, cases, n_rescued, aware_worst, unaware_worst, criteria=2.0,
             location="Plots", figsize=(8.5, 5.5), min_rescued=1, outlier_cases=()):
        """
        One point per case: x = how many parameters correlation-aware
        selection rescued for that case, y = how much WORSE the worst-of-
        10-fold error got once those parameters were hidden from the
        regressor (unaware - aware, percentage points; above the y=0 line
        means correlation-aware selection helped). A scatter, not one bar
        pair per case -- with up to ~50 cases, individual bars are
        unreadable (overlapping labels); the scatter shows the same
        information (and the y=0 trend line) at a glance.

        outlier_cases: case IDs whose delta is numerically unstable (a
        near-singular ablated fold, not a modest accuracy change) --
        plotted as a separate marker at a capped y-value with a value
        label, instead of stretching the axis to fit them and hiding
        everything else (see correlation_ablation_kfold.py's docstring on
        why hiding correlated columns can occasionally break a fold
        outright, not just degrade it).
        """
        os.makedirs(location, exist_ok=True)
        rows = [(c, n, u - a) for c, n, a, u in zip(cases, n_rescued, aware_worst, unaware_worst)
                if n >= min_rescued and c not in outlier_cases]
        out_rows = [(c, n, u - a) for c, n, a, u in zip(cases, n_rescued, aware_worst, unaware_worst)
                    if n >= min_rescued and c in outlier_cases]

        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0.0, color="k", lw=1.2, zorder=1)
        rng = np.random.default_rng(0)
        for c, n, delta in rows:
            jitter = rng.uniform(-0.12, 0.12)
            color = "#2ca02c" if delta > 0 else "#d62728"
            ax.scatter(n + jitter, delta, color=color, s=32, alpha=0.75, zorder=3, edgecolors="none")

        y_cap = max([d for _, _, d in rows], default=1.0) * 1.15 if rows else 1.0
        for c, n, delta in out_rows:
            ax.scatter(n, y_cap, marker="^", s=90, color="#9467bd", zorder=4,
                       label="Numerically unstable fold (off-scale)" if c == out_rows[0][0] else None)
            ax.annotate(f"Case {c}: +{delta:.0f} pp", xy=(n, y_cap), xytext=(n, y_cap * 1.04),
                        fontsize=7.5, ha="center", color="#9467bd")

        # per-n_rescued mean, connected, to show the trend the individual points imply
        by_n = {}
        for c, n, delta in rows:
            by_n.setdefault(n, []).append(delta)
        ns = sorted(by_n)
        means = [np.mean(by_n[n]) for n in ns]
        ax.plot(ns, means, "o-", color="#1f77b4", lw=1.8, ms=6, zorder=5, label="Mean at each rescued-parameter count")

        ax.set_xlabel("Number of parameters correlation-aware selection rescued for that case", fontsize=11)
        ax.set_ylabel("Worst-of-10-fold error, correlation-unaware minus\ncorrelation-aware (percentage points)",
                       fontsize=10.5)
        ax.set_xticks(sorted(set(n for _, n, _ in rows)))
        ax.set_title("Effect of hiding correlation-aware-rescued parameters\n"
                      "(above the line = correlation-aware selection helped)", fontsize=11.5)
        ax.legend(fontsize=8.5, loc="upper left")
        fig.tight_layout()
        out_path = os.path.join(location, "correlation_aware_error_bars.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out_path




# ═══════════════════════════════════════════════════════════════════════════
# Three-parameter p-PRS curve/video study (2026-08-19,
# TEST_THREE_PARAM_pPRS/): reproduces the MB BENCHMARK study's three curve
# figures and search-video machinery, driven entirely by the production
# samplers in Uncertainty.UncertaintyExtractor rather than the BENCHMARK
# scripts' standalone duplicates. Shared math (theta_full/theta_S/f_prior/
# delta_kappa) matches Uncertainty.py's _psac_* primitives exactly -- see
# each class's docstring for which primitive it mirrors.
# ═══════════════════════════════════════════════════════════════════════════

def _pprs_theta_full(T):
    T = np.asarray(T, dtype=float)
    return np.column_stack([np.ones_like(T), np.log(T), -1.0 / T])  # (n_T, 3)


def _pprs_theta_S(T, indices):
    return _pprs_theta_full(T)[:, list(indices)]


def _pprs_f_prior(T, L, indices=None):
    """||Theta_{S or full}(T) . L||_2 row-wise. Pass indices=None for the
    full 3-parameter band (L must be (3,3)); pass indices for a reduced
    band (L must be (m,m)) -- mirrors Uncertainty._f_prior /
    _psac_fp_S exactly."""
    Th = _pprs_theta_full(T) if indices is None else _pprs_theta_S(T, indices)
    return np.linalg.norm(Th @ L, axis=1)


def _pprs_delta_kappa(T, L_r, zeta_r, indices):
    return _pprs_theta_S(T, indices) @ (np.asarray(L_r, dtype=float) @ np.asarray(zeta_r, dtype=float))


def _pprs_has_sign_change(curve):
    curve = np.asarray(curve, dtype=float)
    return bool(np.sign(curve.min()) != np.sign(curve.max()))


class PPRSCurvePlotter:
    """
    Reproduces the three BENCHMARK reference figures
    (test/MUQ-SAC/BENCHMARK/Generate_Class_{A,B,C,D}_curves.py's
    plot_delta_kappa_standalone / plot_curves_for_subfolder /
    plot_curves_plain_blue) for one (reaction, parameter-subset)
    combination, but reads its samples from a driver-written pickle (see
    .from_pickle) instead of holding sampler state itself -- this class
    only plots.

    Pickle schema (dict), written by the TEST_THREE_PARAM_pPRS/ driver:
      {"rxn": str, "class_label": "A"|"B"|"C"|"D", "m": 1|2|3,
       "indices": tuple[int,...], "param_labels": list[str],
       "T_min": float, "T_max": float, "n_T": int,
       "L_full": (3,3) ndarray, "L_r": (m,m) ndarray,
       "zeta_samples": (n_samples, m) ndarray,
       "nominal": (3,) ndarray or None,   # only if kappa(T) curves wanted
       "n_samples_requested": int, "n_samples_generated": int,
       "wall_time_s": float}
    """

    def __init__(self, rxn, class_label, m, indices, T_min, T_max, L_full, L_r,
                 zeta_samples, nominal=None, n_T=200):
        self.rxn = rxn
        self.class_label = class_label
        self.m = m
        self.indices = list(indices)
        self.T = np.linspace(T_min, T_max, n_T)
        self.L_full = np.asarray(L_full, dtype=float)
        self.L_r = np.asarray(L_r, dtype=float)
        self.zeta_samples = np.asarray(zeta_samples, dtype=float)   # (n, m)
        self.nominal = None if nominal is None else np.asarray(nominal, dtype=float)

    @classmethod
    def from_pickle(cls, path):
        import pickle
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["rxn"], d.get("class_label", ""), d.get("m", d["L_r"].shape[0]),
                    d["indices"], d["T_min"], d["T_max"], d["L_full"], d["L_r"],
                    d["zeta_samples"], d.get("nominal"), d.get("n_T", 200))

    def _dk_all(self):
        if len(self.zeta_samples) == 0:
            return np.zeros((0, len(self.T)))
        return np.array([_pprs_delta_kappa(self.T, self.L_r, z, self.indices)
                          for z in self.zeta_samples])   # (n, n_T)

    def _fp_S(self):
        return _pprs_f_prior(self.T, self.L_r, self.indices)

    def plot_delta_kappa(self, location="Plots", figsize=(7, 5)):
        """delta_kappa.pdf: Delta-kappa_S(T) vs 1000/T, all samples plus the
        +-f_prior_S(T) envelope -- same geometry as the reference's
        plot_delta_kappa_standalone."""
        os.makedirs(location, exist_ok=True)
        invT = 1000.0 / self.T
        dk_all = self._dk_all()
        fp = self._fp_S()
        fig, ax = plt.subplots(figsize=figsize)
        for i, row in enumerate(dk_all):
            ax.plot(invT, row, color="red", alpha=0.35, lw=1.1,
                    label="Samples" if i == 0 else None)
        ax.plot(invT, fp, "k--", lw=1.2, label=r"$+f_{\rm prior,S}(T)$")
        ax.plot(invT, -fp, "k--", lw=1.2, label=r"$-f_{\rm prior,S}(T)$")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel(r"$1000/T\;\mathrm{(K^{-1})}$", fontsize=11)
        ax.set_ylabel(r"$\Delta\kappa_S(T)$", fontsize=11)
        ax.set_title(f"{self.rxn}\nClass {self.class_label}, m={self.m} "
                      f"({','.join(str(i) for i in self.indices)})", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = os.path.join(location, "delta_kappa.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out

    def plot_curves(self, location="Plots", figsize_per_axis=(7, 5)):
        """curves.pdf: two-panel Delta-kappa (left) + kappa(T) (right, only
        if self.nominal is set) with a plasma sample-index colorbar --
        matches the reference's plot_curves_for_subfolder."""
        os.makedirs(location, exist_ok=True)
        invT = 1000.0 / self.T
        dk_all = self._dk_all()
        fp = self._fp_S()
        n = len(dk_all)
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(n, 1)))
        has_kappa = self.nominal is not None
        n_axes = 2 if has_kappa else 1
        fig, axes = plt.subplots(1, n_axes, figsize=(figsize_per_axis[0] * n_axes, figsize_per_axis[1]))
        axes = np.atleast_1d(axes)
        ax_dk = axes[0]
        for i, row in enumerate(dk_all):
            ax_dk.plot(invT, row, color=colors[i], alpha=0.45, lw=0.7)
        ax_dk.plot(invT, fp, "k--", lw=1.2)
        ax_dk.plot(invT, -fp, "k--", lw=1.2)
        ax_dk.set_xlabel(r"$1000/T\;\mathrm{(K^{-1})}$"); ax_dk.set_ylabel(r"$\Delta\kappa_S(T)$")
        ax_dk.set_title(r"$\Delta\kappa_S(T)$")
        if has_kappa:
            kappa0 = _pprs_theta_full(self.T) @ self.nominal
            kappa_all = kappa0[None, :] + dk_all
            ax_k = axes[1]
            for i, row in enumerate(kappa_all):
                ax_k.plot(invT, row, color=colors[i], alpha=0.45, lw=0.7)
            ax_k.plot(invT, kappa_all.mean(axis=0), "k-", lw=1.6, label="Mean")
            ax_k.set_xlabel(r"$1000/T\;\mathrm{(K^{-1})}$"); ax_k.set_ylabel(r"$\kappa(T)$")
            ax_k.set_title(r"$\kappa(T) = \kappa_0(T) + \Delta\kappa_S(T)$")
            ax_k.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(1, max(n, 1)))
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes), label="Sample index", fraction=0.03, pad=0.02)
        fig.suptitle(f"{self.rxn}: Class {self.class_label}, m={self.m}", fontsize=11)
        out = os.path.join(location, "curves.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out

    def plot_curves_plain(self, location="Plots", figsize=(7, 5)):
        """curves_plain.pdf: kappa(T) overlay only (solid red + black mean),
        matches the reference's plot_curves_plain_blue -- skipped (returns
        None) if no nominal was provided, same as the reference skipping
        when kappa_curves.npy is absent."""
        if self.nominal is None:
            return None
        os.makedirs(location, exist_ok=True)
        invT = 1000.0 / self.T
        dk_all = self._dk_all()
        kappa0 = _pprs_theta_full(self.T) @ self.nominal
        kappa_all = kappa0[None, :] + dk_all
        fig, ax = plt.subplots(figsize=figsize)
        for i, row in enumerate(kappa_all):
            ax.plot(invT, row, color="red", alpha=0.35, lw=1.1,
                    label="Samples" if i == 0 else None)
        ax.plot(invT, kappa_all.mean(axis=0), "k-", lw=1.8, label="Mean")
        ax.set_xlabel(r"$1000/T\;\mathrm{(K^{-1})}$"); ax.set_ylabel(r"$\kappa(T)$")
        ax.set_title(f"{self.rxn}: Class {self.class_label}, m={self.m}", fontsize=10.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(location, "curves_plain.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out

    def plot_all(self, location="Plots"):
        return {"delta_kappa": self.plot_delta_kappa(location),
                "curves": self.plot_curves(location),
                "curves_plain": self.plot_curves_plain(location)}


class PPRSSubsetGridPlotter:
    """
    7-panel per-reaction summary, one panel per M_CONFIG parameter subset
    (m1: A, n, Ea; m2: A_n, A_Ea, n_Ea; m3: A_n_Ea), each panel showing
    that subset's Delta-kappa_S(T) samples + +-f_prior_S(T) envelope --
    the figure the paper's Updated_SAC.tex figures use. Reuses
    PPRSCurvePlotter's per-subset math so a panel here and a standalone
    delta_kappa.pdf for the same subset are numerically identical.
    """

    SUBSET_ORDER = ["A", "n", "Ea", "A_n", "A_Ea", "n_Ea", "A_n_Ea"]

    def __init__(self, rxn, subset_plotters):
        """subset_plotters: dict[str -> PPRSCurvePlotter], keyed by subset
        label matching SUBSET_ORDER (a missing subset leaves a blank
        panel annotated 'no samples generated', e.g. Class-B's m=1/m=2,
        which are infeasible by the paper's constraint-count argument --
        see PLAN.md Part 4)."""
        self.rxn = rxn
        self.subset_plotters = subset_plotters

    def plot(self, location="Plots", figsize=(18, 9)):
        os.makedirs(location, exist_ok=True)
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        for ax, label in zip(axes, self.SUBSET_ORDER):
            plotter = self.subset_plotters.get(label)
            if plotter is None:
                ax.text(0.5, 0.5, "no samples\ngenerated", ha="center", va="center",
                        fontsize=10, color="gray", transform=ax.transAxes)
                ax.set_title(label, fontsize=10)
                continue
            invT = 1000.0 / plotter.T
            dk_all = plotter._dk_all()
            fp = plotter._fp_S()
            for row in dk_all:
                ax.plot(invT, row, color="red", alpha=0.3, lw=0.8)
            ax.plot(invT, fp, "k--", lw=1.0)
            ax.plot(invT, -fp, "k--", lw=1.0)
            ax.set_title(f"{label} (n={len(dk_all)})", fontsize=10)
            ax.set_xlabel(r"$1000/T$", fontsize=8)
            ax.set_ylabel(r"$\Delta\kappa_S$", fontsize=8)
        for ax in axes[len(self.SUBSET_ORDER):]:
            ax.axis("off")
        fig.suptitle(f"{self.rxn}: all parameter subsets", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe = str(self.rxn).replace("/", "_").replace(" ", "_")
        out = os.path.join(location, f"subset_grid_{safe}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out


class PPRSRejectionPlotter:
    """
    "Why samples get rejected" figure: accepted vs rejected Delta-kappa
    curves in contrasting colors with the violating temperature marked, a
    per-subset acceptance-rate bar chart, and a histogram of the
    violation ratio max_T|Delta-kappa(T)|/f_prior(T) -- matches
    IdentifiablePRS.violation_ratio's definition exactly, so a value here
    printed >1.0 means the same thing "band-invalid" means everywhere
    else in this codebase.
    """

    def __init__(self, rxn):
        self.rxn = rxn

    def plot(self, T, fp, accepted_dk, rejected_dk, rejected_T_violating,
             subset_labels, n_accepted_by_subset, n_rejected_by_subset,
             violation_ratios_by_subset, location="Plots", figsize=(16, 5)):
        """
        T, fp                 : (n_T,) temperature grid / band envelope for the curve panel
        accepted_dk            : list of (n_T,) accepted Delta-kappa curves (one subset)
        rejected_dk             : list of (n_T,) rejected Delta-kappa curves (same subset)
        rejected_T_violating    : list of float|None, T where each rejected curve first left the band
        subset_labels           : list[str], the subset labels for the acceptance bar chart
        n_accepted_by_subset, n_rejected_by_subset : list[int], aligned with subset_labels
        violation_ratios_by_subset : dict[str -> array-like], all-attempts ratios per subset
        """
        os.makedirs(location, exist_ok=True)
        fig, (ax_curve, ax_bar, ax_hist) = plt.subplots(1, 3, figsize=figsize)

        ax_curve.fill_between(T, -fp, fp, color="steelblue", alpha=0.2, label=r"$\pm f_{\rm prior}(T)$")
        for i, dk in enumerate(accepted_dk):
            ax_curve.plot(T, dk, color="#2ca02c", alpha=0.5, lw=0.9,
                          label="Accepted" if i == 0 else None)
        for i, (dk, Tv) in enumerate(zip(rejected_dk, rejected_T_violating)):
            ax_curve.plot(T, dk, color="#d62728", alpha=0.6, lw=1.1,
                          label="Rejected" if i == 0 else None)
            if Tv is not None:
                ax_curve.plot(Tv, np.interp(Tv, T, dk), "x", color="#d62728", ms=7)
        ax_curve.set_xlabel("T (K)"); ax_curve.set_ylabel(r"$\Delta\kappa(T)$")
        ax_curve.set_title(f"{self.rxn}: accepted vs rejected samples", fontsize=10)
        ax_curve.legend(fontsize=8)

        y = np.arange(len(subset_labels))
        total = np.maximum(np.asarray(n_accepted_by_subset) + np.asarray(n_rejected_by_subset), 1)
        rate = 100.0 * np.asarray(n_accepted_by_subset) / total
        ax_bar.barh(y, rate, color="#1f77b4")
        ax_bar.set_yticks(y); ax_bar.set_yticklabels(subset_labels, fontsize=8)
        ax_bar.set_xlabel("Acceptance rate (%)")
        ax_bar.set_title("Acceptance rate by parameter subset", fontsize=10)
        ax_bar.set_xlim(0, 100)

        for label in subset_labels:
            ratios = violation_ratios_by_subset.get(label)
            if ratios is None or len(ratios) == 0:
                continue
            ax_hist.hist(ratios, bins=30, alpha=0.5, label=label)
        ax_hist.axvline(1.0, color="k", ls="--", lw=1.2, label="band_tol")
        ax_hist.set_xlabel(r"Violation ratio, $\max_T|\Delta\kappa(T)|/f_{\rm prior}(T)$")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Violation ratio, all attempts", fontsize=10)
        ax_hist.legend(fontsize=6.5, ncol=2)

        fig.tight_layout()
        safe = str(self.rxn).replace("/", "_").replace(" ", "_")
        out = os.path.join(location, f"rejection_{safe}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out


class PPRSSearchRecorder:
    """
    Video/GIF engine for a p-PRS constrained sample search (Class-A/B/C/D),
    decoupled from any specific sampler: pass an instance as `recorder=`
    to Uncertainty._psac_class_A / _psac_class_C / _psac_class_D_m2 /
    _psac_class_B_m3_ and it collects a frame-by-frame record of every
    attempt via .record()/.mark_restart()/.mark_rejection(), independent
    of which sampler produced them. Modeled on the BENCHMARK reference's
    ClassBSearchViz (test/MUQ-SAC/BENCHMARK/Generate_Class_B_curves.py),
    but records history into plain lists/dicts (picklable) so a video can
    be rebuilt at plot time from a saved history without re-running the
    search -- see .to_pickle()/.from_pickle().

    Unlike the reference, the production _psac_class_B_m3_ sampler has no
    SLSQP/SHGO objective to trace (it is a closed-form 3x3 solve per
    candidate T_u, ">10^4x faster" than the original SHGO/SLSQP path per
    its own docstring) -- so "objective" here is the band-violation ratio
    max_T|Delta-kappa(T)|/f_prior(T) (<=band_tol is a pass), and the
    "constraint residual" panel shows the sign-change test result instead
    of the paper's C1-C4 SLSQP residuals, since that IS this sampler's
    actual acceptance test.
    """

    def __init__(self, T_arr, f_prior_curve, nominal_curve=None, band_tol=1.0, title=""):
        self.T_arr = np.asarray(T_arr, dtype=float)
        self.f_prior_curve = np.asarray(f_prior_curve, dtype=float)
        self.nominal_curve = None if nominal_curve is None else np.asarray(nominal_curve, dtype=float)
        self.band_tol = band_tol
        self.title = title
        self.frames = []          # list of dict: eval, z, obj, accepted, dk, restart_count, n_rejections
        self.restart_evals = []   # eval indices where a restart happened
        self.rejections = []      # list of (eval_idx, reason)
        self._eval = 0
        self._restart_count = 0

    def mark_restart(self):
        self._restart_count += 1
        self.restart_evals.append(self._eval)

    def mark_rejection(self, reason):
        self.rejections.append((self._eval, reason))

    def record(self, z, obj, meta):
        self._eval += 1
        dk = meta.get("dk")
        self.frames.append({
            "eval": self._eval,
            "z": None if z is None else np.asarray(z, dtype=float).tolist(),
            "obj": obj,
            "accepted": bool(meta.get("accepted", False)),
            "dk": None if dk is None else np.asarray(dk, dtype=float).tolist(),
            "restart_count": self._restart_count,
            "n_rejections": len(self.rejections),
        })

    def to_pickle(self, path):
        import pickle
        payload = {
            "T_arr": self.T_arr, "f_prior_curve": self.f_prior_curve,
            "nominal_curve": self.nominal_curve, "band_tol": self.band_tol,
            "title": self.title, "frames": self.frames,
            "restart_evals": self.restart_evals, "rejections": self.rejections,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def from_pickle(cls, path):
        import pickle
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls(payload["T_arr"], payload["f_prior_curve"], payload["nominal_curve"],
                   payload["band_tol"], payload["title"])
        obj.frames = payload["frames"]
        obj.restart_evals = payload["restart_evals"]
        obj.rejections = payload["rejections"]
        obj._eval = len(obj.frames)
        return obj

    def build_and_save(self, path, fps=15, dpi=100, every=1):
        """
        Render the recorded history to `path` (.mp4, via FFMpegWriter),
        plus a matching normal-speed .gif (Pillow), then a 10x-sped-up
        copy of both (matching PLAN.md's "MP4 + 10x GIF" deliverable) via
        .speed_up_animation, reusing the reference's post-hoc-speedup
        pattern rather than hand-resampling frames at record time.
        Returns dict {"mp4", "gif", "mp4_10x", "gif_10x"} (any entry may
        be None if that writer isn't available in this environment).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.animation as animation

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames = self.frames[::every]
        if not frames:
            raise ValueError("PPRSSearchRecorder has no recorded frames to render")

        fig, (ax_dk, ax_obj, ax_res) = plt.subplots(1, 3, figsize=(15, 4.5))
        rejection_idx = {e for e, _ in self.rejections}

        def draw(frame_i):
            for ax in (ax_dk, ax_obj, ax_res):
                ax.clear()
            fr = frames[frame_i]

            ax_dk.fill_between(self.T_arr, -self.f_prior_curve, self.f_prior_curve,
                                color="steelblue", alpha=0.25, label=r"$\pm f_{\rm prior}(T)$")
            ax_dk.axhline(0, color="k", lw=0.8)
            if fr["dk"] is not None:
                color = "#2ca02c" if fr["accepted"] else "#d62728"
                ax_dk.plot(self.T_arr, fr["dk"], color=color, lw=1.8,
                           label="accepted" if fr["accepted"] else "rejected candidate")
            ax_dk.set_xlabel("T (K)"); ax_dk.set_ylabel(r"$\Delta\kappa_S(T)$")
            ax_dk.set_title(self.title, fontsize=9)
            ax_dk.legend(fontsize=7, loc="upper right")

            evals = [f["eval"] for f in frames[:frame_i + 1]]
            objs = [f["obj"] if f["obj"] is not None else np.nan for f in frames[:frame_i + 1]]
            ax_obj.plot(evals, objs, "-", color="#1f77b4", lw=1.2)
            ax_obj.axhline(self.band_tol, color="k", ls="--", lw=1.0, label="band_tol")
            for r in self.restart_evals:
                if r <= fr["eval"]:
                    ax_obj.axvline(r, color="orange", ls=":", lw=0.9)
            ax_obj.set_xlabel("Attempt #"); ax_obj.set_ylabel("Violation ratio")
            ax_obj.set_title("Objective trace (restarts = orange)", fontsize=9)
            ax_obj.legend(fontsize=7)

            sign_ok = 1.0 if (fr["dk"] is not None and _pprs_has_sign_change(fr["dk"])) else 0.0
            band_ok = 1.0 if (fr["obj"] is not None and fr["obj"] <= self.band_tol) else 0.0
            ax_res.bar(["sign change", "band"], [sign_ok, band_ok],
                       color=["#2ca02c" if sign_ok else "#d62728",
                              "#2ca02c" if band_ok else "#d62728"])
            ax_res.set_ylim(0, 1.3)
            is_rej = fr["eval"] in rejection_idx
            ax_res.set_title(f"rejections so far: {fr['n_rejections']}", fontsize=9,
                              color="red" if is_rej else "black")
            ax_res.set_facecolor("#ffe5e5" if is_rej else "white")
            fig.tight_layout()
            return ()

        ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000.0 / fps, blit=False)

        mp4_path = path if path.endswith(".mp4") else path + ".mp4"
        gif_path = mp4_path[:-4] + ".gif"
        out = {"mp4": None, "gif": None, "mp4_10x": None, "gif_10x": None}
        try:
            ani.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="mpeg4"))
            out["mp4"] = mp4_path
        except Exception as e:
            print(f"[PPRSSearchRecorder] FFMpegWriter failed ({e}); mp4 not written")
        try:
            ani.save(gif_path, writer="pillow", fps=max(1, fps // 2), dpi=dpi)
            out["gif"] = gif_path
        except Exception as e:
            print(f"[PPRSSearchRecorder] Pillow gif writer failed ({e}); gif not written")
        plt.close(fig)

        if out["mp4"]:
            out["mp4_10x"] = self.speed_up_animation(out["mp4"], speedup=10)
        if out["gif"]:
            out["gif_10x"] = self.speed_up_animation(out["gif"], speedup=10)
        return out

    @staticmethod
    def speed_up_animation(input_path, speedup=10):
        """
        Post-hoc N-x speedup of an existing .mp4/.gif, ported from the
        BENCHMARK reference's speed_up_animation
        (Generate_Class_B_curves.py) -- .gif re-times frame durations via
        Pillow (floored at 20ms, duration // speedup); .mp4 uses ffmpeg's
        setpts filter with audio stripped (-an). Returns the new path
        {base}_{speedup}x{ext}, or the original input_path if the
        speedup step itself fails (still leaves a usable, if not sped-up,
        artifact behind rather than raising).
        """
        import subprocess
        base, ext = os.path.splitext(input_path)
        out_path = f"{base}_{speedup}x{ext}"
        try:
            if ext.lower() == ".gif":
                from PIL import Image, ImageSequence
                im = Image.open(input_path)
                frames, durations = [], []
                for frame in ImageSequence.Iterator(im):
                    frames.append(frame.copy())
                    durations.append(max(20, frame.info.get("duration", 100) // speedup))
                frames[0].save(out_path, save_all=True, append_images=frames[1:],
                                duration=durations, loop=0)
            elif ext.lower() == ".mp4":
                subprocess.run(
                    ["ffmpeg", "-y", "-i", input_path, "-filter:v",
                     f"setpts=(1/{speedup})*PTS", "-an", out_path],
                    check=True, capture_output=True)
            else:
                return input_path
            return out_path
        except Exception as e:
            print(f"[PPRSSearchRecorder.speed_up_animation] failed on {input_path}: {e}")
            return input_path


# ═══════════════════════════════════════════════════════════════════════════
# MSO-compare (Step 6, /data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/
# MSO_A_FACTOR/p_PRS/JOURNAL_RESULTS/, per its BEAMER_HANDOFF.md): Method 1
# (Sobol'-PRS) vs Method 2 (GA+bisection), 3 fuels x 2 MRE criteria. Headline
# numbers come ONLY from report_prs_hyperopt_<fuel>.csv (Mean WMRE %) -- never
# from *_prs_verify's raw objective sums, a different, non-comparable unit
# (BEAMER_HANDOFF.md guard rail). Unresolved numbers (1_3_Butadiene's nan
# crit.6 headline, unverified 155.4% crit.2) are drawn as hatched bars, never
# fabricated.
# ═══════════════════════════════════════════════════════════════════════════

class MSOCompareWMREPlotter:
    """
    Headline Mean WMRE% comparison, Method 1 (Sobol'-PRS) vs Method 2
    (GA+bisection), grouped by fuel and MRE criterion.
    """

    def plot(self, records, location="Plots", figsize=(10, 5.5)):
        """records: list of dict{fuel, criterion(2|6), method('Method 1'|
        'Method 2'), mean_wmre: float or None (None -> unresolved, drawn
        hatched rather than fabricated)}."""
        os.makedirs(location, exist_ok=True)
        fuels = sorted(set(r["fuel"] for r in records))
        criteria = sorted(set(r["criterion"] for r in records))
        methods = sorted(set(r["method"] for r in records))
        fig, axes = plt.subplots(1, len(criteria), figsize=figsize, sharey=True)
        axes = np.atleast_1d(axes)
        width = 0.8 / max(len(methods), 1)
        colors = {"Method 1": "#1f77b4", "Method 2": "#ff7f0e"}
        all_vals = [r["mean_wmre"] for r in records if r["mean_wmre"] is not None]
        y_max = max(all_vals) if all_vals else 1.0
        placeholder_h = 0.06 * y_max   # small but visible, so "unresolved" never reads as "zero"
        for ax, crit in zip(axes, criteria):
            x = np.arange(len(fuels))
            for j, method in enumerate(methods):
                vals, hatched = [], []
                for fuel in fuels:
                    match = [r for r in records
                             if r["fuel"] == fuel and r["criterion"] == crit and r["method"] == method]
                    v = match[0]["mean_wmre"] if match else None
                    vals.append(placeholder_h if v is None else v)
                    hatched.append(v is None)
                bars = ax.bar(x + j * width, vals, width=width, label=method,
                               color=colors.get(method, None))
                for xi, bar, h in zip(x, bars, hatched):
                    if h:
                        bar.set_hatch("//"); bar.set_facecolor("none"); bar.set_edgecolor("red")
                        ax.annotate("unresolved", xy=(xi + j * width, placeholder_h), xytext=(0, 4),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=6.5, color="red", rotation=90)
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(fuels, rotation=20, ha="right", fontsize=8)
            ax.set_title(f"Criterion {crit}", fontsize=10.5)
            ax.set_ylabel("Mean WMRE (%)")
        axes[0].legend(fontsize=8.5)
        fig.suptitle("PRS hyperparameter-optimization accuracy: Method 1 vs Method 2\n"
                      "(hatched = unresolved / not yet verified, see BEAMER_HANDOFF.md)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out = os.path.join(location, "mso_compare_wmre.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out


class MSOCompareCostPlotter:
    """
    Compute-cost / retry-count comparison per (fuel, criterion), from
    *_timing_failure_summary.json -- quotable as-is per BEAMER_HANDOFF.md
    (no re-derivation needed): compute-days spent and total retry
    attempts, side by side so a study needing far more compute (1,3-
    Butadiene, per the handoff) is visible at a glance rather than buried
    in a table.
    """

    def plot(self, summaries, location="Plots", figsize=(9, 5)):
        """summaries: list of dict, each a *_timing_failure_summary.json's
        top-level content plus a 'label' key (e.g. 'MB crit.6')."""
        os.makedirs(location, exist_ok=True)
        labels = [s["label"] for s in summaries]
        compute_days = [s.get("compute_days_sum", 0.0) for s in summaries]
        retries = [s.get("total_retry_attempts", 0) for s in summaries]
        y = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.barh(y, compute_days, color="#9467bd")
        ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel("Compute (days)")
        ax1.set_title("Total compute spent", fontsize=10.5)
        ax2.barh(y, retries, color="#8c564b")
        ax2.set_yticks(y); ax2.set_yticklabels([])
        ax2.set_xlabel("Retry attempts")
        ax2.set_title("Total retry attempts", fontsize=10.5)
        fig.suptitle("MSO-compare study cost", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out = os.path.join(location, "mso_compare_cost.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out


class MSOCompareParityPlotter:
    """
    Predicted vs confirmed-real WMRE parity for the PRS hyperparameter-
    optimization fit, Method 1 vs Method 2 side by side -- for fuel/
    criterion combos whose *_parity_grid.png doesn't already exist
    (BEAMER_HANDOFF.md notes MB_MRE_2 and n_Heptane_159_MRE_2 lack it),
    built directly from report_prs_hyperopt_<fuel>.csv's predicted_wmre /
    confirm_real_wmre columns (same numbers already in that CSV, no new
    computation) rather than re-deriving from raw *_fit.json.
    """

    def plot(self, method1_rows, method2_rows, fuel="", criterion=None, location="Plots", figsize=(11, 5.2)):
        """method{1,2}_rows: list of dict with 'predicted_wmre' and
        'confirm_real_wmre' (or 'best_real_wmre') keys, one row per
        (objective, optimizer) combo for that method/fuel/criterion."""
        os.makedirs(location, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

        def panel(ax, rows, title):
            pred = np.array([r["predicted_wmre"] for r in rows], dtype=float)
            real = np.array([r.get("confirm_real_wmre", r.get("best_real_wmre")) for r in rows], dtype=float)
            ax.scatter(real, pred, s=40, alpha=0.75, edgecolors="none", color="#1f77b4")
            lo = min(np.nanmin(pred), np.nanmin(real)) if len(pred) else 0.0
            hi = max(np.nanmax(pred), np.nanmax(real)) if len(pred) else 1.0
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, label="Perfect prediction")
            ax.set_xlabel("Confirmed real WMRE (%)"); ax.set_ylabel("Predicted WMRE (%)")
            ax.set_title(title, fontsize=10.5)
            ax.legend(fontsize=8)

        panel(ax1, method1_rows, "Method 1: Sobol'-PRS")
        panel(ax2, method2_rows, "Method 2: GA + bisection")
        title = f"{fuel}" + (f", criterion {criterion}" if criterion is not None else "")
        fig.suptitle(f"Predicted vs confirmed WMRE: {title}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        safe = str(fuel).replace("/", "_").replace(" ", "_")
        tag = f"_crit{criterion}" if criterion is not None else ""
        out = os.path.join(location, f"mso_compare_parity_{safe}{tag}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return out
