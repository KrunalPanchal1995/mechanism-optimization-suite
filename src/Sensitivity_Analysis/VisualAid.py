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

