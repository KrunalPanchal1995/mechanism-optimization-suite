import ast
import sys
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# ── mso_io access (Target Workbench, below) ──────────────────────────────
# MUQ-SAC is a standalone pipeline with its own sys.path conventions (its
# siblings are imported by bare name, e.g. `import Uncertainty`), not part
# of the rmip/mso_io package tree. Append (never insert at 0 -- this repo's
# convention, see feedback_env_isolation) src/ so `mso_io.target_add_io`
# resolves without disturbing MUQ-SAC's own bare-name imports.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.normpath(os.path.join(_this_dir, ".."))
if _src_dir not in sys.path:
    sys.path.append(_src_dir)
try:
    from mso_io.target_add_io import preview_targets as _preview_targets
    _HAS_MSO_IO = True
except ImportError:
    _HAS_MSO_IO = False


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

    def plot(self, location="Plots", figsize=(7.5, 5), filename="band_validity_before_after.pdf"):
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
        out_path = os.path.join(location, filename)
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


class ActiveParameterSVDPlotter:
    """
    Renders the singular-value spectrum IdentifiablePRS.identifiable_directions
    computes per reaction (`build_layout_blocks`'s ReactionBlock.svd_spectrum
    / .svd_ratio_to_max), reading the JSON report written by
    active_parameter_selection/driver_active_parameter_selection.py --
    this class only plots, it does not recompute the SVD.

    JSON schema consumed (case0_active_parameter_report.json):
      {"case": int, "n_blocks": int,
       "reactions": [{"rxn": str, "param_labels": [...], "n_total_directions": int,
                       "n_retained": int, "n_discarded": int,
                       "svd_spectrum": [...], "svd_ratio_to_max": [...],
                       "svd_tol": float}, ...]}
    """

    def __init__(self, report):
        self.report = report

    @classmethod
    def from_json(cls, path):
        import json
        with open(path) as fh:
            return cls(json.load(fh))

    def plot_spectrum(self, outpath="active_parameter_svd_spectrum.pdf", figsize=(11, 5),
                       max_rxn_label=28):
        """
        One point per candidate direction, grouped by reaction on the x-axis,
        y = singular value ratio to that reaction's own max (log scale) --
        the actual retain/discard criterion, since svd_tol in the report is
        a RATIO tolerance, not an absolute one. Retained directions are
        filled blue circles; discarded ones are red X's. The tolerance line
        is common to every reaction (ratio-based), so one dashed horizontal
        line at svd_tol applies to the whole panel.
        """
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        reactions = self.report["reactions"]
        tol = reactions[0]["svd_tol"] if reactions else 1e-6

        fig, ax = plt.subplots(figsize=figsize)
        xt_labels = []
        for i, r in enumerate(reactions):
            ratios = np.asarray(r["svd_ratio_to_max"], dtype=float)
            n_ret = r["n_retained"]
            retained_mask = np.arange(len(ratios)) < n_ret if n_ret else np.zeros(len(ratios), dtype=bool)
            # identifiable_directions keeps directions above tol in descending
            # singular-value order, so the first n_retained entries are kept.
            xs = np.full(len(ratios), i, dtype=float)
            ax.scatter(xs[retained_mask], ratios[retained_mask], marker="o", s=34,
                       color="#1f77b4", zorder=3,
                       label="Retained" if i == 0 and retained_mask.any() else None)
            ax.scatter(xs[~retained_mask], ratios[~retained_mask], marker="x", s=50,
                       color="#d62728", linewidths=1.8, zorder=3,
                       label="Discarded" if i == 0 and (~retained_mask).any() else None)
            label = r["rxn"]
            if len(label) > max_rxn_label:
                label = label[:max_rxn_label - 1] + "…"
            xt_labels.append(f"{label}\n({r['n_retained']}/{r['n_total_directions']})")

        ax.axhline(tol, color="black", ls="--", lw=1.1,
                   label=fr"tolerance ($s/s_{{\max}} = {tol:.0e}$)")
        ax.set_yscale("log")
        ax.set_xticks(range(len(reactions)))
        ax.set_xticklabels(xt_labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel(r"$s_i / s_{\max}$  (singular value ratio, log scale)")
        n_disc_total = sum(r["n_discarded"] for r in reactions)
        n_dir_total = sum(r["n_total_directions"] for r in reactions)
        ax.set_title(f"Case {self.report.get('case', '?')}: SVD spectrum per reaction "
                     f"-- {n_dir_total - n_disc_total}/{n_dir_total} directions retained, "
                     f"{self.report.get('n_blocks', len(reactions))} reactions with a block",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", dpi=150)
        fig.savefig(outpath.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return outpath


# ═══════════════════════════════════════════════════════════════════════════
# MERGED from FINALIZED_MECH_OPT_SUITE (2026-08-26 union merge): target-set
# workbench (TargetRow/TargetSet/TargetSetPlotter/TargetFinding/TargetChecker/
# DummyPointGenerator/TargetPruner/NominalParityAnalyzer), H2SweepPlotter,
# 3D-cube plotting helpers, and the postprocess() CLI. None of these names
# collide with this file's own classes (verified at merge time).
# ═══════════════════════════════════════════════════════════════════════════
def _parse_composition(text):
    """Parse the .target file's 'x-><id_map> = <frac_map>' (multi-species)
    or 'x-><species> = <frac>' (single-species) composition syntax into a
    plain {species: mole_fraction} dict. Mirrors
    mso_io.target_add_io._fuel_multi_repr/_bath_gas_repr's encoding, in
    reverse. Returns {} for missing/unparseable text rather than raising --
    this is a display/check helper, not the authoritative parser (that's
    combustion_target_class.combustion_target.__init__)."""
    if not text or "x->" not in text or "=" not in text:
        return {}
    body = text.split("x->", 1)[1]
    id_part, _, frac_part = body.partition("=")
    id_part, frac_part = id_part.strip(), frac_part.strip()
    try:
        if id_part.startswith("{"):
            id_map = ast.literal_eval(id_part)
            frac_map = ast.literal_eval(frac_part) if frac_part.startswith("{") else {}
            return {sp: float(frac_map.get(k, 0.0)) for k, sp in id_map.items()}
        else:
            return {id_part: float(frac_part)}
    except (ValueError, SyntaxError, TypeError):
        return {}


class TargetRow:
    """One parsed .target line, extending mso_io.target_add_io.TargetPreview
    (display-only: dataset_id/target_type/T/P/phi/solver/runnable) with the
    composition + observation fields TargetChecker/TargetSetPlotter need.
    Built from a TargetPreview's own `.raw` line text -- never re-reads the
    file, and never the authoritative parser (see combustion_target_class
    for that; TargetChecker.check_parses runs it as the write-time gate)."""

    def __init__(self, preview):
        self.line_no = preview.line_no
        self.raw = preview.raw
        self.dataset_id = preview.dataset_id
        self.target_type = preview.target_type
        self.simulation = preview.simulation
        self.temperature_K = preview.temperature_K
        self.pressure_atm = preview.pressure_atm
        self.phi_raw = preview.phi
        self.solver = preview.solver
        self.runnable = preview.runnable
        self.runnable_reason = preview.runnable_reason

        parts = self.raw.split("|")

        def field(key):
            for p in parts:
                kv = p.split("--", 1)
                if len(kv) == 2 and kv[0].strip() == key:
                    return kv[1].strip()
            return None

        self.fuel = _parse_composition(field("Fuel"))
        self.oxidizer = _parse_composition(field("Oxidizer"))
        self.bath_gas = _parse_composition(field("Bath_gas"))

        def to_float(text):
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        self.observed = to_float(field("observed"))
        self.deviation = to_float(field("deviation"))
        weight_text = field("data_weight")
        self.data_weight = to_float(weight_text)
        self.obs_unit = field("obs_unit")

        units_text = field("units")
        try:
            self.units = ast.literal_eval(units_text) if units_text else {}
        except (ValueError, SyntaxError):
            self.units = {}

        self.is_dummy = False
        self.dummy_source = ""

    def composition_sum(self):
        """Total mole fraction across fuel + oxidizer + bath gas -- should
        sum to 1.0 for a physically valid mixture."""
        return (sum(self.fuel.values()) + sum(self.oxidizer.values())
                + sum(self.bath_gas.values()))

    def as_dict(self):
        return {
            "line_no": self.line_no, "dataset_id": self.dataset_id,
            "target_type": self.target_type, "simulation": self.simulation,
            "temperature_K": self.temperature_K, "pressure_atm": self.pressure_atm,
            "phi_raw": self.phi_raw, "solver": self.solver, "runnable": self.runnable,
            "observed": self.observed, "deviation": self.deviation,
            "data_weight": self.data_weight, "obs_unit": self.obs_unit,
            "composition_sum": self.composition_sum(),
            "is_dummy": self.is_dummy, "dummy_source": self.dummy_source,
        }


class TargetSet:
    """Load a .target (+ paired .add) file into TargetRow objects. Purely
    read-only -- the source files are never modified; curation output goes
    through write_curated_set() in mso_io.target_add_io, to a new version."""

    def __init__(self, target_path, add_path=None):
        if not _HAS_MSO_IO:
            raise ImportError(
                "mso_io.target_add_io is required for TargetSet -- src/ "
                "was not importable from this interpreter.")
        self.target_path = target_path
        self.add_path = add_path
        self.rows = [TargetRow(p) for p in _preview_targets(target_path, add_path)]

    def to_dataframe(self):
        return pd.DataFrame([r.as_dict() for r in self.rows])

    def by_line(self, line_no):
        for r in self.rows:
            if r.line_no == line_no:
                return r
        raise KeyError(f"no row with line_no={line_no}")

    def dataset_ids(self):
        seen = []
        for r in self.rows:
            if r.dataset_id not in seen:
                seen.append(r.dataset_id)
        return seen

    def rows_for(self, dataset_id):
        return [r for r in self.rows if r.dataset_id == dataset_id]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class TargetSetPlotter:
    """Plots over a TargetSet -- T/P/phi scatter, ln(tau) vs 1000/T per
    dataset (dummy/dropped points visually distinct), observed-vs-deviation,
    per-dataset condition coverage. All methods return the saved path (or
    None if save_path wasn't given) and always plt.close() their figure."""

    def __init__(self, target_set):
        self.ts = target_set

    def plot_t_p_phi(self, save_path=None, show=False):
        df = self.ts.to_dataframe()
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ids = df["dataset_id"].astype("category")
        sc = ax.scatter(df["temperature_K"], df["pressure_atm"], ids.cat.codes,
                         c=ids.cat.codes, cmap="tab20")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Pressure [atm]")
        ax.set_zlabel("Dataset")
        ax.set_zticks(range(len(ids.cat.categories)))
        ax.set_zticklabels(ids.cat.categories, fontsize=6)
        ax.set_title("T-P-dataset space")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def plot_lntau_vs_invT(self, dataset_id, dummy_line_nos=(), dropped_line_nos=(),
                            save_path=None, show=False):
        rows = [r for r in self.ts.rows_for(dataset_id)
                if r.temperature_K and r.observed]
        if not rows:
            raise ValueError(f"no plottable Tig rows for dataset_id={dataset_id!r}")
        dummy_set = set(dummy_line_nos)
        dropped_set = set(dropped_line_nos)
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in sorted(rows, key=lambda r: r.temperature_K):
            invT = 1000.0 / r.temperature_K
            lntau = np.log(r.observed)
            if r.line_no in dropped_set:
                ax.scatter(invT, lntau, marker="x", color="#999999", s=40, zorder=2)
            elif r.line_no in dummy_set or r.is_dummy:
                ax.scatter(invT, lntau, marker="^", color="#d7301f", s=30, zorder=3)
            else:
                ax.scatter(invT, lntau, marker="o", color="#2c7fb8", s=30, zorder=4)
        ax.set_xlabel("1000/T [1/K]")
        ax.set_ylabel("ln(observed)")
        ax.set_title(f"{dataset_id}: ln(tau) vs 1000/T")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c7fb8", label="original", markersize=8),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#d7301f", label="dummy", markersize=8),
            plt.Line2D([0], [0], marker="x", color="#999999", label="dropped", markersize=8),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def plot_observed_vs_deviation(self, save_path=None, show=False):
        df = self.ts.to_dataframe().dropna(subset=["observed", "deviation"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["observed"], df["deviation"], s=20, alpha=0.7)
        ax.set_xlabel("observed")
        ax.set_ylabel("deviation")
        ax.set_title("Observed vs. deviation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def plot_condition_coverage(self, save_path=None, show=False):
        df = self.ts.to_dataframe()
        fig, ax = plt.subplots(figsize=(8, 5))
        for ds, g in df.groupby("dataset_id"):
            ax.scatter(g["temperature_K"], g["pressure_atm"], label=ds, s=25, alpha=0.75)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Pressure [atm]")
        ax.set_title("Per-dataset condition coverage")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return save_path


class TargetFinding:
    """One TargetChecker finding. severity is 'error' | 'warning' | 'info'."""

    __slots__ = ("severity", "line_no", "dataset_id", "check", "message", "suggested_fix")

    def __init__(self, severity, line_no, dataset_id, check, message, suggested_fix=None):
        self.severity = severity
        self.line_no = line_no
        self.dataset_id = dataset_id
        self.check = check
        self.message = message
        self.suggested_fix = suggested_fix

    def as_dict(self):
        return {"severity": self.severity, "line_no": self.line_no,
                "dataset_id": self.dataset_id, "check": self.check,
                "message": self.message, "suggested_fix": self.suggested_fix}

    def __repr__(self):
        loc = f"line {self.line_no}" if self.line_no else self.dataset_id
        return f"[{self.severity}] {loc}: {self.message}"


class TargetChecker:
    """Runs a battery of sanity checks over a TargetSet. Nothing here is
    written until every 'error'-severity finding from check_parses (the
    authoritative gate) is resolved -- see write_curated_set's own gate."""

    def __init__(self, target_set, add_data=None):
        self.ts = target_set
        self.add_data = add_data or {}

    def run_all(self, composition_tol=0.02, deviation_ratio_tol=5.0):
        findings = []
        findings += self.check_parses()
        findings += self.check_composition_sums(tol=composition_tol)
        findings += self.check_units()
        findings += self.check_deviation_outliers(ratio_tol=deviation_ratio_tol)
        findings += self.check_duplicate_conditions()
        findings += self.check_ntc_crossing()
        return findings

    def check_parses(self):
        """Authoritative gate: instantiate combustion_target_class on every
        row and surface its exception as an 'error' finding. Also reports,
        as 'info', the phi combustion_target actually COMPUTES from
        composition for any row whose raw Phi field wasn't 'N/A' and
        disagrees with it -- __init__ always overwrites self.phi from
        fuel/oxidizer composition, so the raw text field is cosmetic."""
        import combustion_target_class
        from copy import deepcopy
        findings = []
        for r in self.ts.rows:
            try:
                case = combustion_target_class.combustion_target(
                    r.raw, deepcopy(self.add_data), r.line_no - 1)
            except Exception as exc:
                findings.append(TargetFinding(
                    "error", r.line_no, r.dataset_id, "parses",
                    f"combustion_target failed to parse this row: "
                    f"{type(exc).__name__}: {exc}"))
                continue
            raw_phi = (r.phi_raw or "").strip()
            if raw_phi and raw_phi != "N/A" and case.phi is not None:
                try:
                    if abs(float(raw_phi) - float(case.phi)) > 1e-3:
                        findings.append(TargetFinding(
                            "info", r.line_no, r.dataset_id, "phi_backfill",
                            f"stated Phi={raw_phi} but composition computes "
                            f"phi={case.phi:.4g} (the pipeline always uses "
                            f"the computed value)",
                            suggested_fix=f"Phi -- {case.phi:.6g}"))
                except ValueError:
                    pass
        return findings

    def check_composition_sums(self, tol=0.02):
        findings = []
        for r in self.ts.rows:
            total = r.composition_sum()
            if total <= 0:
                continue  # no composition parsed at all -- check_parses covers it
            if abs(total - 1.0) > tol:
                findings.append(TargetFinding(
                    "warning", r.line_no, r.dataset_id, "composition_sum",
                    f"fuel+oxidizer+bath mole fractions sum to {total:.4f}, "
                    f"not 1.0 (tol={tol})"))
        return findings

    def check_units(self):
        findings = []
        for r in self.ts.rows:
            if r.temperature_K is not None and not (150.0 <= r.temperature_K <= 4000.0):
                findings.append(TargetFinding(
                    "warning", r.line_no, r.dataset_id, "units",
                    f"T={r.temperature_K} K is outside a physically plausible "
                    f"combustion range [150, 4000] K"))
            if r.pressure_atm is not None and not (0.0 < r.pressure_atm <= 1000.0):
                findings.append(TargetFinding(
                    "warning", r.line_no, r.dataset_id, "units",
                    f"P={r.pressure_atm} atm is outside a physically plausible "
                    f"range (0, 1000] atm"))
        return findings

    def check_deviation_outliers(self, ratio_tol=5.0):
        findings = []
        ratios = []
        for r in self.ts.rows:
            if r.deviation == 0:
                findings.append(TargetFinding(
                    "warning", r.line_no, r.dataset_id, "deviation",
                    "deviation is exactly 0 -- this row will dominate "
                    "any weighted-least-squares objective"))
                continue
            if r.deviation is None or r.observed in (None, 0):
                continue
            ratios.append((r, abs(r.deviation / r.observed)))
        if not ratios:
            return findings
        vals = np.array([v for _, v in ratios])
        median = float(np.median(vals))
        if median <= 0:
            return findings
        for r, v in ratios:
            if v > ratio_tol * median:
                findings.append(TargetFinding(
                    "warning", r.line_no, r.dataset_id, "deviation",
                    f"deviation/observed ratio {v:.3g} is {v / median:.1f}x "
                    f"the dataset's median ratio ({median:.3g})"))
        return findings

    def check_duplicate_conditions(self, t_round=1, p_round=2):
        findings = []
        seen = {}
        for r in self.ts.rows:
            if r.temperature_K is None or r.pressure_atm is None:
                continue
            key = (r.dataset_id, r.target_type,
                   round(r.temperature_K, t_round), round(r.pressure_atm, p_round))
            seen.setdefault(key, []).append(r)
        for key, rows in seen.items():
            if len(rows) > 1:
                line_nos = [r.line_no for r in rows]
                for r in rows:
                    findings.append(TargetFinding(
                        "warning", r.line_no, r.dataset_id, "duplicate_condition",
                        f"{len(rows)} rows share (target_type, T, P)={key[1:]} "
                        f"(lines {line_nos}) -- their data_weight is being "
                        f"counted {len(rows)}x in the objective"))
        return findings

    def check_ntc_crossing(self, min_points=5):
        """NTC-crossing datasets, via the same fuel-agnostic detector
        DummyPointGenerator uses (Dummy_data_generation.detect_ntc_region).
        Informational only -- an NTC crossing is a real physical feature,
        not a defect, but is useful context before generating dummy points
        or pruning (dropping points inside the NTC well can hide it)."""
        import Dummy_data_generation as ddg
        findings = []
        for ds in self.ts.dataset_ids():
            rows = [r for r in self.ts.rows_for(ds)
                    if r.target_type == "Tig" and r.temperature_K and r.observed]
            if len(rows) < min_points:
                continue
            T = np.array([r.temperature_K for r in rows])
            tau = np.array([r.observed for r in rows])
            try:
                info = ddg.detect_ntc_region(T, tau)
            except Exception:
                continue
            findings.append(TargetFinding(
                "info", None, ds, "ntc_crossing",
                f"dataset crosses an NTC region: {info['ntc_start']:.0f}K - "
                f"{info['ntc_end']:.0f}K"))
        return findings


class DummyPointGenerator:
    """NTC-aware dense dummy-point generation for ONE dataset, reusing
    Dummy_data_generation.py's fuel-agnostic curve-fitting primitives
    (fit_smoothed_logtau_spline, detect_ntc_region, build_equal_zone_grid) --
    the same machinery DATASET/METHYL_BUTANOATE_2026/MB_LTC_DUMMY.target was
    built with, generalized to any single dataset instead of that file's
    hardcoded Ethyl/Butanoate pair."""

    def __init__(self, target_set, dataset_id):
        self.ts = target_set
        self.dataset_id = dataset_id
        rows = [r for r in target_set.rows_for(dataset_id)
                if r.target_type == "Tig" and r.temperature_K and r.observed]
        if len(rows) < 4:
            raise ValueError(
                f"dataset_id={dataset_id!r} has {len(rows)} usable Tig rows "
                f"(need >= 4 to fit a spline)")
        self.rows = sorted(rows, key=lambda r: r.temperature_K)
        self.template_row = self.rows[0]

    def generate(self, points_per_zone=15, smoothing_s=0.05,
                 manual_ntc_bounds=None):
        """Returns a DataFrame of new dummy rows (dataset_ID, Temperature_K,
        observed, obs_unit, is_dummy, dummy_source, generated_zone) spanning
        the dataset's own T range in three equal-count zones (before/inside/
        after its NTC region). manual_ntc_bounds, if given, is (start_K,
        end_K) overriding auto-detection (mirrors Dummy_data_generation.py's
        USE_MANUAL_*_NTC_BOUNDS knobs)."""
        import Dummy_data_generation as ddg

        T = np.array([r.temperature_K for r in self.rows], dtype=float)
        tau = np.array([r.observed for r in self.rows], dtype=float)

        auto_info = ddg.detect_ntc_region(T, tau, smoothing_s=smoothing_s)
        if manual_ntc_bounds is not None:
            info = ddg.apply_manual_bounds(
                auto_info, use_manual=True,
                manual_start=manual_ntc_bounds[0], manual_end=manual_ntc_bounds[1],
                label=self.dataset_id)
        else:
            info = ddg.apply_manual_bounds(auto_info, use_manual=False,
                                            manual_start=0, manual_end=0,
                                            label=self.dataset_id)

        T_grid, zone_labels = ddg.build_equal_zone_grid(
            info["t_min"], info["ntc_start"], info["ntc_end"], info["t_max"],
            points_per_zone)
        y_ln = ddg.evaluate_logtau_with_linear_boundary_extrapolation(
            info["spline"], T_grid, T)
        tau_dense = np.exp(y_ln)

        out = pd.DataFrame({
            "dataset_ID": f"dummy_{self.dataset_id}",
            "Temperature_K": T_grid,
            "observed": tau_dense,
            "obs_unit": self.template_row.obs_unit or "us",
            "is_dummy": True,
            "dummy_source": (
                f"DummyPointGenerator: smoothed ln(tau) spline of "
                f"{self.dataset_id}; NTC bounds mode={info['boundary_mode']}"),
            "generated_zone": zone_labels,
        })
        self._ntc_info = info
        return out


class TargetPruner:
    """Which rows can be dropped without losing much: geometric coverage
    (target_sampling.py's farthest-point diversity sampling over T/P/phi --
    no extra data needed) and, optionally, information-content pruning via
    a caller-supplied basis/sensitivity matrix (e.g. an existing
    DesignMatrix.csv fit for this target set) using the same greedy
    D-optimal det(Gram) scoring AdaptiveResampler.DOptimalResampler uses
    for augmentation, run here in reverse (backward elimination)."""

    def __init__(self, target_set):
        self.ts = target_set

    def coverage_prune(self, n_drop, seed=0):
        """Returns (keep_line_nos, drop_line_nos, coverage_ratio) -- drop
        n_drop rows while maximizing T/P/phi coverage of what remains."""
        import target_sampling as ts_mod
        df = self.ts.to_dataframe()
        df = df.dropna(subset=["temperature_K", "pressure_atm"])
        n = len(df)
        if not (1 <= n_drop < n):
            raise ValueError(f"n_drop must be in [1, {n - 1}], got {n_drop}")
        phi_numeric = pd.to_numeric(df["phi_raw"], errors="coerce").fillna(0.0)
        points = np.column_stack([
            df["temperature_K"].to_numpy(dtype=float),
            df["pressure_atm"].to_numpy(dtype=float),
            phi_numeric.to_numpy(dtype=float),
        ])
        n_keep = n - n_drop
        keep_local = ts_mod.farthest_point_sample(points, n_keep, seed=seed)
        keep_set = set(keep_local)
        line_nos = df["line_no"].tolist()
        keep_line_nos = [line_nos[i] for i in keep_local]
        drop_line_nos = [line_nos[i] for i in range(n) if i not in keep_set]
        coverage = ts_mod.coverage_ratio(points, keep_local)
        return keep_line_nos, drop_line_nos, coverage

    def information_prune(self, basis_matrix, line_numbers, n_drop, ridge_alpha=0.0):
        """Greedy D-optimal backward elimination: repeatedly remove the row
        whose removal shrinks det(Gram) the LEAST, until n_drop rows are
        dropped. basis_matrix rows must align 1:1 with line_numbers (e.g.
        ResponseSurface.build_basis() output from a PRS already fit for
        this target set -- there is no such matrix from the .target file
        alone, since it has no reaction-sensitivity information).

        Returns (keep_line_nos, drop_line_nos, log) where log is a list of
        {line_no, det_ratio} in drop order (det_ratio close to 1.0 means
        that row was nearly redundant; << 1.0 means dropping it cost real
        information)."""
        B = np.asarray(basis_matrix, dtype=float)
        n = B.shape[0]
        if len(line_numbers) != n:
            raise ValueError("basis_matrix rows must align with line_numbers")
        if not (1 <= n_drop < n):
            raise ValueError(f"n_drop must be in [1, {n - 1}], got {n_drop}")

        alive = list(range(n))
        log = []
        for _ in range(n_drop):
            G = B[alive].T @ B[alive]
            if ridge_alpha > 0:
                G = G + ridge_alpha * np.eye(G.shape[0])
            base_det = float(np.linalg.det(G))
            best_i, best_ratio = None, -np.inf
            for j, idx in enumerate(alive):
                trial = [k for k in alive if k != idx]
                Gt = B[trial].T @ B[trial]
                if ridge_alpha > 0:
                    Gt = Gt + ridge_alpha * np.eye(Gt.shape[0])
                det_t = float(np.linalg.det(Gt))
                ratio = det_t / base_det if base_det != 0 else 0.0
                if ratio > best_ratio:
                    best_ratio, best_i = ratio, idx
            alive.remove(best_i)
            log.append({"line_no": line_numbers[best_i], "det_ratio": best_ratio})

        dropped = {e["line_no"] for e in log}
        keep_line_nos = [line_numbers[i] for i in range(n) if line_numbers[i] not in dropped]
        drop_line_nos = [e["line_no"] for e in log]
        return keep_line_nos, drop_line_nos, log


class NominalParityAnalyzer:
    """Model-vs-experiment parity for the nominal (unperturbed) mechanism
    over a target set, plus a per-row leverage score. Deliberately does NOT
    run the nominal simulation itself (that's MUQ-SAC/run_nominal_sim.py's
    job, via nominal_sim_gui.py's subprocess pipeline, per this repo's
    env-isolation convention) -- callers pass in results already produced by
    that pipeline, keyed by line_no."""

    def __init__(self, target_set, nominal_results):
        """nominal_results : dict[int, float] -- line_no -> nominal model
        prediction, in the same units as TargetRow.observed."""
        self.ts = target_set
        self.nominal_results = nominal_results

    def parity_table(self):
        rows = []
        for r in self.ts.rows:
            model = self.nominal_results.get(r.line_no)
            if model is None or r.observed is None:
                continue
            sigma = abs(r.deviation) if r.deviation else None
            leverage = (model - r.observed) / sigma if sigma else None
            rows.append({
                "line_no": r.line_no, "dataset_id": r.dataset_id,
                "observed": r.observed, "model": model,
                "residual": model - r.observed, "leverage_sigma": leverage,
            })
        return pd.DataFrame(rows)

    def flag_leverage(self, n_sigma=3.0):
        """Rows the nominal mechanism already misses by more than n_sigma
        deviations -- these dominate the MSO objective before any
        optimization pass even starts."""
        df = self.parity_table().dropna(subset=["leverage_sigma"])
        return df[df["leverage_sigma"].abs() > n_sigma]

    def plot_parity(self, save_path=None, show=False):
        df = self.parity_table().dropna(subset=["model", "observed"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df["observed"], df["model"], s=25, alpha=0.7)
        lo = min(df["observed"].min(), df["model"].min())
        hi = max(df["observed"].max(), df["model"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Nominal model")
        ax.set_title("Nominal-model parity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return save_path


# =============================================================================
# postprocess() -- "process the results" flag entry point
# =============================================================================
# Everything above this line is the existing plotter/analysis class library
# (untouched -- run_script.py imports TPhiPlotter/ArrheniusPlotter at import
# time and PostOptPlotter mid-run; neither import path changes). What follows
# is new: a standalone way to point this file at an existing (or
# still-running) src/MUQ-SAC/run_script.py run directory and regenerate its
# plots + tables without re-running the optimization, via
# ``mso_io.postprocess_base``'s shared CLI contract (also used by
# Sensitivity_Analysis/VisualAid.py, forward_uq/uq_plots.py, graph_sa/plots.py,
# mso_compare/run_plots.py, automation/prs_report.py, rmip/postprocess.py).
#
# Reconstructs exactly what run_script.py itself already computes right
# before it calls PostOptPlotter (see the "Per-reaction optimisation +
# posterior plots" block, immediately after the ``solution_zeta.save`` /
# ``posterior_cov.pkl`` load) -- from the same three files that block reads,
# so a postprocess() run and the in-line run_script.py plot are the same
# curves. Deliberately does NOT modify PostOptPlotter (or any other class
# above) to do this -- it calls the curve-building methods PostOptPlotter
# already has, the same way any external caller would.

import pickle  # noqa: E402  (kept with the rest of the file's flat imports)

_this_dir_pp = os.path.dirname(os.path.abspath(__file__))
_src_dir_pp = os.path.normpath(os.path.join(_this_dir_pp, ".."))
if _src_dir_pp not in sys.path:
    sys.path.append(_src_dir_pp)
try:
    from mso_io.postprocess_base import (  # noqa: E402
        PostProcessResult, add_postprocess_args, default_out_dir, discover,
        formats_from_args, write_manifest, write_table,
    )
except ImportError:  # pragma: no cover - postprocess_base predates this repo layout
    PostProcessResult = None

try:
    from scipy.interpolate import make_interp_spline as _make_interp_spline
except ImportError:  # pragma: no cover
    _make_interp_spline = None

try:
    from scipy.interpolate import UnivariateSpline as _UnivariateSpline
except ImportError:  # pragma: no cover
    _UnivariateSpline = None

try:
    from scipy.interpolate import PchipInterpolator as _PchipInterpolator
except ImportError:  # pragma: no cover
    _PchipInterpolator = None


def _spline_xy(x, y, n=200, k=3, logy=False, logx=False, denoise=False, shape_preserving=False):
    """Smooth (x,y) onto a dense spline for a line-only (no marker) plot --
    used throughout H2SweepPlotter per the journal-figure convention that a
    deterministic simulation sweep is shown as a smooth curve, not a raw
    polyline through sparse grid points. Falls back to the raw (sorted,
    de-duplicated) points if there aren't enough of them (<4) to fit a
    cubic spline, or if scipy is unavailable -- never raises.

    shape_preserving=True uses a monotone cubic Hermite spline (PCHIP)
    instead of a plain interpolating cubic spline -- needed for a sharply
    peaked-then-decaying physical quantity (e.g. laminar flame speed vs.
    phi) sampled at sparse, unevenly-spaced points, where an ordinary
    cubic spline overshoots into unphysical negative values in the long
    flat tail between distant points. PCHIP never overshoots the data's
    own range.

    denoise=True uses a smoothing (least-squares) spline instead of an
    exact interpolating one -- needed for quantities like the Jacobian
    stiffness ratio, which is genuinely point-to-point noisy (near-
    degenerate eigenvalue selection), where an *interpolating* cubic
    spline through every noisy point just overshoots wildly between them
    (Runge phenomenon) instead of showing the actual trend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if logy:
        good &= (y > 0)
    if logx:
        good &= (x > 0)
    x, y = x[good], y[good]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 1e-12 * max(1.0, np.abs(x).max())))
    x, y = x[keep], y[keep]
    if len(x) < 3 or _make_interp_spline is None:
        return x, y
    xs_fit = np.log10(x) if logx else x
    ys_fit = np.log(y) if logy else y
    kk = min(k, len(x) - 1)
    try:
        if shape_preserving and _PchipInterpolator is not None:
            spl = _PchipInterpolator(xs_fit, ys_fit)
        elif denoise and _UnivariateSpline is not None and len(x) >= 6:
            # Smoothing factor scaled to the data's own variance (standard
            # scipy convention: s ~ m * var(y) for m points is a reasonable
            # amount of smoothing, tuned down since we still want the
            # broad trend, not a near-flat line).
            s = len(ys_fit) * np.var(ys_fit) * 0.5
            spl = _UnivariateSpline(xs_fit, ys_fit, k=min(3, kk), s=s)
        elif _make_interp_spline is not None:
            spl = _make_interp_spline(xs_fit, ys_fit, k=kk)
        else:
            return x, y
    except Exception:
        return x, y
    xs_dense = np.linspace(xs_fit.min(), xs_fit.max(), n)
    ys_dense = spl(xs_dense)
    x_out = 10 ** xs_dense if logx else xs_dense
    y_out = np.exp(ys_dense) if logy else ys_dense
    return x_out, y_out


def _cluster_continuous_branches(points, T_gap_max, logP_gap_max):
    """Greedy nearest-head clustering of (T, P) boundary-crossing points
    into continuous branches, by proximity in (T, log10 P). The raw
    per-T "which transition came first" index from
    explosion_limit_map.py's boundary trace is NOT a continuity label (a
    low-P branch at low T and an unrelated low-P branch reappearing at
    high T can share the same per-T index) -- this reconstructs actual
    continuous curves, which is what makes the classical S/Z explosion-
    limit shape visible as smooth connected lines instead of a
    disconnected jumble.
    """
    pts = sorted(points, key=lambda p: p[0])
    heads = []  # each: [last_T, last_logP, point_list]
    for T, P in pts:
        logP = np.log10(P) if P > 0 else -12.0
        best, best_d = None, None
        for h in heads:
            hT, hlogP, _ = h
            if (T - hT) <= T_gap_max and abs(logP - hlogP) <= logP_gap_max:
                d = abs(logP - hlogP)
                if best is None or d < best_d:
                    best, best_d = h, d
        if best is None:
            heads.append([T, logP, [(T, P)]])
        else:
            best[2].append((T, P))
            best[0], best[1] = T, logP
    return [h[2] for h in heads]


def _cube_verts_edges_faces(xr, yr, zr):
    """8 vertices / 12 edges / 6 faces of an axis-aligned box spanning
    xr=(x0,x1), yr=(y0,y1), zr=(z0,z1) -- shared geometry for the
    wireframe-cube and solid-cube 3D helpers below."""
    x0, x1 = xr
    y0, y1 = yr
    z0, z1 = zr
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    faces_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return verts, edges, faces_idx


def _draw_wireframe_cube(ax, xr, yr, zr, **kwargs):
    """Draw the 12-edge wireframe of an axis-aligned box on a 3D axes --
    used to give a 3D scatter of boundary points an explicit bounding
    envelope (the full explored (T, phi, P) domain) to sit inside."""
    verts, edges, _ = _cube_verts_edges_faces(xr, yr, zr)
    for i, j in edges:
        xs = [verts[i][0], verts[j][0]]
        ys = [verts[i][1], verts[j][1]]
        zs = [verts[i][2], verts[j][2]]
        ax.plot(xs, ys, zs, **kwargs)


def _draw_solid_cube(ax, xr, yr, zr, facecolor, alpha, edgecolor, lw):
    """Draw a small, semi-transparent, solid-faced box on a 3D axes -- used
    for the reference paper's training region so it reads as a distinct,
    easily identifiable sub-volume nested inside the full domain."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts, _, faces_idx = _cube_verts_edges_faces(xr, yr, zr)
    faces = [[verts[i] for i in f] for f in faces_idx]
    poly = Poly3DCollection(faces, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor, linewidths=lw)
    ax.add_collection3d(poly)


#: Short author-year citation for the PC-NODE reference paper (Kumar, Kumar
#: & Pal, Combustion Theory and Modelling 29(3), 2025) -- used everywhere a
#: figure needs to name that paper's training region, instead of the
#: generic "reference paper" (H2/paper/main.tex's \citep{kumar2025pcnode}
#: is the full citation at first use in the text).
PAPER_CITATION = "Kumar et al. (2025)"


def _draw_highlight_box_2d(ax, x_range, y_range_frac, color="crimson", lw=2.2, label=None):
    """A bold dashed-outline rectangle (2D axes-fraction y-range, e.g. from
    a log-scale axhline/axvspan-style computation) -- used in place of a
    flat-filled axvspan for highlighting the reference paper's training
    box, which reads as a much crisper, more identifiable outline than a
    translucent fill (especially layered over a busy curve/heatmap)."""
    from matplotlib.patches import Rectangle
    x0, x1 = x_range
    rect = Rectangle((x0, y_range_frac[0]), x1 - x0, y_range_frac[1] - y_range_frac[0],
                      transform=ax.get_xaxis_transform(), fill=False,
                      edgecolor=color, linewidth=lw, linestyle=(0, (5, 2)), zorder=8,
                      label=label)
    ax.add_patch(rect)
    return rect


class H2SweepPlotter:
    """Plotting for the H2/O2 extreme-condition sweep (see H2/README.md and
    H2/paper/main.tex). Consumes the CSVs written by the pipeline-driven
    nominal-sim runs (``H2/SIM_RUNS/<label>/Plot/Dataset/*/*.csv`` and each
    case's ``time_history.csv``) and by the standalone drivers
    (``H2/results/*.csv``), and produces a small, deliberately CLUBBED set
    of multi-panel figures rather than one figure per condition -- per the
    user's explicit instruction to minimize plot count by grouping similar
    conditions together.

    Conventions match the rest of this file: matplotlib.pyplot only,
    ``fig.savefig(path, dpi=200 or 150, bbox_inches="tight")`` then
    ``plt.close(fig)``, and (where the mso_io postprocess contract applies)
    a CSV sidecar via ``save_figure``/``write_table`` from
    ``mso_io.postprocess_base`` when available.
    """

    def __init__(self, h2_dir):
        self.h2_dir = h2_dir
        self.results_dir = os.path.join(h2_dir, "results")
        self.sim_runs_dir = os.path.join(h2_dir, "SIM_RUNS")
        self.plots_dir = os.path.join(h2_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    # -- small IO helpers -----------------------------------------------
    def _read_csv(self, name):
        path = os.path.join(self.results_dir, name)
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def _save(self, fig, name):
        path = os.path.join(self.plots_dir, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    # -- 1) crossover locus + paper's training box -----------------------
    def plot_crossover_locus(self):
        """The single most load-bearing figure in H2/paper/: the analytic
        2nd-explosion-limit (crossover) locus in the P-T plane, with
        Kumar et al. (2025)'s training envelope (T in [1000,1200] K,
        P=1 atm) overlaid, showing it terminates almost exactly ON the
        crossover curve rather than safely inside either regime.
        Spline-smoothed, line-only (no marker symbols). The training box
        is a bold dashed outline (not a flat grey fill) so it stays
        crisply identifiable against the crossing curves."""
        df = self._read_csv("explosion_limit_crossover_locus.csv")
        if df is None:
            return None
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for dil, grp in df[df["phi"] == 1.0].groupby("diluent"):
            grp = grp.sort_values("P_atm")
            xs, ys = _spline_xy(grp["T_crossover_K"], grp["P_atm"], logy=True)
            ax.plot(xs, ys, lw=2.0, label=rf"{dil} dilution ($\phi$=1)")
        ax.set_yscale("log")
        # axes-fraction y-range for "0.01 to 1 atm" on this log axis, so
        # the highlight box actually spans up to the 1 atm line rather
        # than being squashed into a sliver at the bottom.
        y_lo, y_hi = 0.01, 100.0
        frac_1atm = (np.log10(1.0) - np.log10(y_lo)) / (np.log10(y_hi) - np.log10(y_lo))
        ax.set_ylim(y_lo, y_hi)
        _draw_highlight_box_2d(ax, (1000, 1200), (0, frac_1atm), color="crimson", lw=2.2,
                               label=f"{PAPER_CITATION} training box")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.annotate(f"{PAPER_CITATION}\ntraining box\n(1000-1200 K, 1 atm)",
                    xy=(1200, 1.0), xytext=(1420, 6.0),
                    fontsize=8, color="crimson", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
        ax.set_xlabel("Crossover temperature $T$ [K]")
        ax.set_ylabel("Pressure [atm]")
        ax.set_title(r"H$_2$/O$_2$ 2nd explosion-limit (chain-branching crossover) locus")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        return self._save(fig, "01_crossover_locus")

    # -- 1b) key-temperature comparison (explicit "compare at each key T") --
    def plot_key_temperature_comparison(self):
        """At each of this study's own key temperature design levels
        (1000/1900/2800 K), at Kumar et al. (2025)'s own P=1 atm: is the
        mixture ignitable (10 s bound), is the point inside their training
        box, and how stiff is the chemistry there -- across all three key
        phi values. Bar chart by construction (three discrete temperatures
        on the x-axis, not a continuous sweep), so no line-plot symbol
        rule applies here."""
        df = self._read_csv("key_temperature_comparison.csv")
        if df is None:
            return None
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        T_keys = sorted(df["T_key_K"].unique())
        phis = sorted(df["phi"].unique())
        width = 0.8 / max(len(phis), 1)
        colors = {0.3: "tab:green", 1.0: "tab:red", 6.0: "tab:purple"}
        max_y = 0.0
        for i, phi in enumerate(phis):
            sub = df[df["phi"] == phi].set_index("T_key_K").loc[T_keys]
            x = np.arange(len(T_keys)) + (i - (len(phis) - 1) / 2) * width
            ax.bar(x, sub["stiffness_ratio_nearest"], width=width * 0.9,
                   color=colors.get(phi, "gray"), label=rf"$\phi$={phi:g}")
            for xi, (_, row) in zip(x, sub.iterrows()):
                ign_txt = "ignites" if row["ignites_within_10s"] else "no ignition"
                box_txt = "in training box" if row["in_paper_training_box"] else "outside training box"
                yv = row["stiffness_ratio_nearest"]
                if np.isfinite(yv) and yv > 0:
                    max_y = max(max_y, yv)
                    ax.text(xi, yv * 1.3, f"{ign_txt}\n{box_txt}", ha="center",
                            va="bottom", fontsize=6, rotation=90)
        ax.set_yscale("log")
        if max_y > 0:
            ax.set_ylim(top=max_y * 200)
        ax.set_xticks(range(len(T_keys)))
        ax.set_xticklabels([f"{t:g} K" for t in T_keys])
        ax.set_xlabel("Key temperature (this study's design levels)")
        ax.set_ylabel(r"Jacobian stiffness ratio $|\lambda_{max}|/|\lambda_{min}|$")
        ax.set_title(f"Ignitability and stiffness at each key temperature, "
                     f"P={df['P_atm'].iloc[0]:g} atm (10 s ignition-time bound)\n"
                     f"vs. {PAPER_CITATION} training box")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return self._save(fig, "01b_key_temperature_comparison")

    # -- 2) numeric ignitability map, per key phi, traced S/Z-shaped boundary
    def plot_ignitability_map(self):
        """Per-phi explosion-limit boundary in the P-T plane, TRACED by
        bisection (H2/standalone/explosion_limit_map.py) rather than shown
        as a scatter of individually-classified grid points -- this is what
        actually reveals the classical S/Z-shaped explosion-limit curve.
        Ignite/no-ignite regions are shaded as a filled background from the
        coarse classification grid; the traced boundary itself is
        continuity-clustered into branches and spline-smoothed on top, line
        only, no markers. The legend explicitly names the 10 s ignition-
        time bound used as the criterion."""
        out = []
        for phi in [0.3, 1.0, 6.0]:
            tag = f"{phi:g}".replace(".", "p")
            grid = self._read_csv(f"explosion_limit_numeric_map_phi{tag}.csv")
            bnd = self._read_csv(f"explosion_limit_boundary_phi{tag}.csv")
            if grid is None or grid.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5.5))
            piv = grid.pivot_table(index="P_atm", columns="T0", values="ignites")
            Tv = piv.columns.values.astype(float)
            Pv = piv.index.values.astype(float)
            Z = piv.values.astype(float)
            ax.pcolormesh(Tv, Pv, Z, cmap="RdBu_r", vmin=0, vmax=1,
                          alpha=0.35, shading="nearest")
            ax.set_yscale("log")
            if bnd is not None and not bnd.empty:
                pts = list(zip(bnd["T0"], bnd["P_boundary_atm"]))
                n_T = max(len(grid["T0"].unique()), 1)
                T_span = grid["T0"].max() - grid["T0"].min()
                branches = _cluster_continuous_branches(
                    pts, T_gap_max=max(T_span / n_T * 2.5, 60.0), logP_gap_max=1.0)
                first = True
                for br in sorted(branches, key=len, reverse=True):
                    if len(br) < 2:
                        continue
                    Tb = [p[0] for p in br]
                    Pb = [p[1] for p in br]
                    xs, ys = _spline_xy(Tb, Pb, logy=True)
                    ax.plot(xs, ys, lw=2.2, color="k",
                            label=("traced ignition boundary" if first else None))
                    first = False
            else:
                ax.text(0.5, 0.5, "no ignition within the 10 s bound\nanywhere in "
                        "this $T$-$P$ envelope", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color="tab:blue")
            ax.set_ylim(Pv.min(), Pv.max())
            ax.set_xlabel("Initial temperature $T_0$ [K]")
            ax.set_ylabel("Pressure [atm]")
            ax.set_title(rf"Numeric ignitability map, $\phi$={phi:g}, "
                         r"stoich.-scaled H$_2$/N$_2$ (gas-phase only)")
            handles = [
                plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.35),
                plt.Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.35),
                plt.Line2D([0], [0], color="k", lw=2.2),
            ]
            labels = ["ignites within 10 s", "no ignition within 10 s", "traced boundary (10 s bound)"]
            ax.legend(handles, labels, fontsize=8, loc="lower right")
            ax.grid(alpha=0.25, which="both")
            out.append(self._save(fig, f"02_ignitability_map_phi{tag}"))
        return out

    # -- 2b) 3D (T, P, phi) ignitability boundary surface --------------------
    def plot_ignitability_3d(self):
        """Journal-ready 3D flammability-limit figure. EVERY ignitable grid
        point (not just the traced boundary) is scattered, filling out the
        actual ignitable VOLUME inside a wireframe cube spanning this
        study's full explored (T, phi, P) envelope -- swept over a DENSE
        phi range, 0.3 to 6.0 in steps of 0.3 (20 levels, H2/standalone/
        explosion_limit_map.py's build_numeric_volume_dense()), colored
        continuously by phi (a 20-color discrete legend would be
        unreadable, so a colorbar instead). Kumar et al. (2025)'s training
        region is highlighted with a bold lime wireframe box (a
        translucent 6-face solid cuboid here reads as a muddy smudge once
        its faces overlap in a 3D projection -- an outline is far more
        identifiable) plus a single flat highlighted panel at their fixed
        P=1 atm (a real, if thin, slice of the cube, not a 3D box, since
        the training run itself never varied P); their own actual 30
        discrete training conditions -- their own reported grid, phi=0.5
        to 1.5 in steps of 0.2 (6 levels) x T=1000 to 1200 K in steps of
        50 (5 levels), at their fixed P=1 atm -- are overlaid as bold
        white-on-black markers (chosen for contrast against the full
        purple-to-yellow phi colormap the background points span), at
        their own reported resolution (not resampled) -- a second,
        independent way to see exactly what they covered. Uses the same
        50%-dilution convention as the pipeline's own IDT sweep throughout
        (H2/review/flag.md), so phi=6 correctly shows its real ignitable
        volume."""
        df = self._read_csv("explosion_limit_numeric_volume_dense.csv")
        if df is None or df.empty:
            return None
        ign = df[df["ignites"] == True].copy()  # noqa: E712
        if ign.empty:
            return None

        T_lo, T_hi = float(df["T0"].min()), float(df["T0"].max())
        P_lo, P_hi = float(df["P_atm"].min()), float(df["P_atm"].max())
        phi_lo, phi_hi = float(df["phi"].min()), float(df["phi"].max())
        logP_lo, logP_hi = np.log10(P_lo), np.log10(P_hi)

        # Kumar et al. (2025)'s training region -- their own actual grid,
        # at their own 0.2-phi-increment resolution, their own 5 T levels,
        # and their own fixed P=1 atm.
        paper_T = (1000.0, 1200.0)
        paper_phi = (0.5, 1.5)
        paper_logP = np.log10(1.0)
        paper_T_levels = [1000.0, 1050.0, 1100.0, 1150.0, 1200.0]
        paper_phi_levels = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        fig = plt.figure(figsize=(10.5, 8.6))
        ax = fig.add_subplot(111, projection="3d")

        cmap = plt.get_cmap("plasma")
        norm = plt.Normalize(vmin=phi_lo, vmax=phi_hi)
        sc = ax.scatter(ign["T0"], ign["phi"], np.log10(ign["P_atm"]),
                         c=ign["phi"], cmap=cmap, norm=norm,
                         s=14, alpha=0.35, linewidth=0, depthshade=True)

        # Full explored envelope, as a bounding wireframe cube.
        _draw_wireframe_cube(ax, (T_lo, T_hi), (phi_lo, phi_hi), (logP_lo, logP_hi),
                             color="dimgray", lw=0.8, alpha=0.45, linestyle="--")

        # Kumar et al. (2025)'s training region: a bold, high-contrast
        # wireframe outline (not a translucent solid -- overlapping
        # semi-transparent faces on a small box read as a dark smudge, not
        # a crisp box) plus a single flat highlighted panel at their exact
        # P=1 atm, since they never actually varied P.
        _draw_wireframe_cube(ax, paper_T, paper_phi, (paper_logP - 0.05, paper_logP + 0.05),
                             color="lime", lw=2.6, alpha=0.95, linestyle="-")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        floor = [[(paper_T[0], paper_phi[0], paper_logP), (paper_T[1], paper_phi[0], paper_logP),
                  (paper_T[1], paper_phi[1], paper_logP), (paper_T[0], paper_phi[1], paper_logP)]]
        ax.add_collection3d(Poly3DCollection(floor, facecolor="lime", alpha=0.25,
                                              edgecolor="none"))

        # Their own actual 30 discrete training conditions -- bold
        # white-fill/black-edge markers, chosen for contrast against every
        # color the background phi-colormap can take (dark purple through
        # bright yellow).
        Tg, Pg = np.meshgrid(paper_T_levels, paper_phi_levels)
        ax.scatter(Tg.ravel(), Pg.ravel(), np.full(Tg.size, paper_logP),
                   c="white", edgecolor="black", marker="o", s=55, linewidth=1.3,
                   depthshade=False, zorder=10)

        ax.set_xlabel("Temperature $T$ [K]")
        ax.set_ylabel(r"Equivalence ratio $\phi$")
        ax.set_zlabel("Pressure [atm]")
        zt = np.arange(np.floor(logP_lo), np.ceil(logP_hi) + 1)
        ax.set_zticks(zt)
        ax.set_zticklabels([f"{10.0 ** z:g}" for z in zt])
        ax.set_title(r"Ignitable volume across $T$, $\phi$ (0.3 to 6.0, step 0.3), $P$"
                     f"\n(10 s ignition-time bound) vs. {PAPER_CITATION} training region")
        ax.view_init(elev=18, azim=-60)

        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.65)
        cbar.set_label(r"Equivalence ratio $\phi$ (ignitable points)")

        box_proxy = plt.Line2D([0], [0], color="lime", lw=2.6)
        paper_pts_proxy = plt.Line2D([0], [0], color="black", marker="o", markerfacecolor="white",
                                      linestyle="none", markersize=8, markeredgewidth=1.3)
        ax.legend([box_proxy, paper_pts_proxy],
                  [f"{PAPER_CITATION} training region (explored)",
                   f"{PAPER_CITATION}'s 30 training conditions ($\\phi$=0.5:0.2:1.5, P=1 atm)"],
                  fontsize=7.5, loc="upper left", bbox_to_anchor=(-0.02, 0.98))
        fig.tight_layout()
        return self._save(fig, "02b_ignitability_3d_map")

    # -- 2c) chemical stiffness, heatmap-scattered over the SAME (T,P,phi) --
    #        grid as the ignitable-volume figure above
    def plot_stiffness_3d(self):
        """Companion to plot_ignitability_3d: chemical (Jacobian) stiffness
        shown as a 3D heatmap scatter over EXACTLY the same dense
        (T0, P, phi) grid -- phi=0.3 to 6.0 in steps of 0.3, same T_GRID,
        same P_COARSE_ATM -- as the ignitable-volume figure (H2/standalone/
        explosion_limit_map.py's build_stiffness_volume_dense(), point-for-
        point matched to build_numeric_volume_dense()), per instruction.
        Every grid point is plotted (not just the ignitable ones -- stiffness
        is a property of the chemical state regardless of whether that
        state happens to satisfy the ignition criterion), colored
        continuously by log10(stiffness ratio) via a colorbar. Kumar et al.
        (2025)'s training region is highlighted the same way as in
        plot_ignitability_3d (bold lime wireframe + flat panel at their
        fixed P=1 atm, plus their own 30 training conditions as bold
        white-on-black markers) so the two figures read as a matched pair."""
        df = self._read_csv("stiffness_volume_dense.csv")
        if df is None or df.empty:
            return None
        df = df[np.isfinite(df["stiffness_ratio"]) & (df["stiffness_ratio"] > 0)].copy()
        if df.empty:
            return None
        df["log_stiff"] = np.log10(df["stiffness_ratio"])

        T_lo, T_hi = float(df["T0"].min()), float(df["T0"].max())
        P_lo, P_hi = float(df["P_atm"].min()), float(df["P_atm"].max())
        phi_lo, phi_hi = float(df["phi"].min()), float(df["phi"].max())
        logP_lo, logP_hi = np.log10(P_lo), np.log10(P_hi)

        paper_T = (1000.0, 1200.0)
        paper_phi = (0.5, 1.5)
        paper_logP = np.log10(1.0)
        paper_T_levels = [1000.0, 1050.0, 1100.0, 1150.0, 1200.0]
        paper_phi_levels = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        fig = plt.figure(figsize=(10.5, 8.6))
        ax = fig.add_subplot(111, projection="3d")

        # Diverging blue-white-red, not the plasma/magma sequential ramp
        # plot_ignitability_3d uses for phi -- the two 3D maps used to be
        # indistinguishable at a glance (both dark-purple-to-yellow); this
        # gives stiffness its own, unambiguous visual language.
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=df["log_stiff"].quantile(0.02), vmax=df["log_stiff"].quantile(0.98))
        sc = ax.scatter(df["T0"], df["phi"], np.log10(df["P_atm"]),
                         c=df["log_stiff"], cmap=cmap, norm=norm,
                         s=10, alpha=0.25, linewidth=0, depthshade=True)

        _draw_wireframe_cube(ax, (T_lo, T_hi), (phi_lo, phi_hi), (logP_lo, logP_hi),
                             color="dimgray", lw=0.8, alpha=0.45, linestyle="--")

        _draw_wireframe_cube(ax, paper_T, paper_phi, (paper_logP - 0.05, paper_logP + 0.05),
                             color="lime", lw=2.6, alpha=0.95, linestyle="-")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        floor = [[(paper_T[0], paper_phi[0], paper_logP), (paper_T[1], paper_phi[0], paper_logP),
                  (paper_T[1], paper_phi[1], paper_logP), (paper_T[0], paper_phi[1], paper_logP)]]
        ax.add_collection3d(Poly3DCollection(floor, facecolor="lime", alpha=0.25,
                                              edgecolor="none"))

        Tg, Pg = np.meshgrid(paper_T_levels, paper_phi_levels)
        ax.scatter(Tg.ravel(), Pg.ravel(), np.full(Tg.size, paper_logP),
                   c="white", edgecolor="black", marker="o", s=55, linewidth=1.3,
                   depthshade=False, zorder=10)

        ax.set_xlabel("Temperature $T$ [K]")
        ax.set_ylabel(r"Equivalence ratio $\phi$")
        ax.set_zlabel("Pressure [atm]")
        zt = np.arange(np.floor(logP_lo), np.ceil(logP_hi) + 1)
        ax.set_zticks(zt)
        ax.set_zticklabels([f"{10.0 ** z:g}" for z in zt])
        ax.set_title(r"Chemical stiffness across $T$, $\phi$ (0.3 to 6.0, step 0.3), $P$"
                     f"\n(same grid as the ignitable-volume figure) vs. {PAPER_CITATION} training region")
        ax.view_init(elev=18, azim=-60)

        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.65)
        cbar.set_label(r"log$_{10}$(Jacobian stiffness ratio $|\lambda_{max}|/|\lambda_{min}|$)")

        box_proxy = plt.Line2D([0], [0], color="lime", lw=2.6)
        paper_pts_proxy = plt.Line2D([0], [0], color="black", marker="o", markerfacecolor="white",
                                      linestyle="none", markersize=8, markeredgewidth=1.3)
        ax.legend([box_proxy, paper_pts_proxy],
                  [f"{PAPER_CITATION} training region (explored)",
                   f"{PAPER_CITATION}'s 30 training conditions ($\\phi$=0.5:0.2:1.5, P=1 atm)"],
                  fontsize=7.5, loc="upper left", bbox_to_anchor=(-0.02, 0.98))
        fig.tight_layout()
        return self._save(fig, "02c_stiffness_3d_map")

    # -- 11) explored (paper) vs. unexplored, colored by chemical stiffness --
    def plot_flammability_vs_paper_training(self):
        """Direct explored-vs-not-explored comparison: Kumar et al.
        (2025)'s training box vs. this mechanism's actual 2nd-limit
        crossover locus, with the dense Jacobian-stiffness field
        (H2/standalone/explosion_limit_map.py's build_dense_stiffness_field)
        as a filled background -- directly showing how stiff the chemistry
        gets in the region the NeuralODE was never trained on. Uses the
        same diverging coolwarm scale as plot_stiffness_3d so "stiffness"
        reads as one consistent visual language across both figures,
        distinct from the sequential plasma/phi language of
        plot_ignitability_3d. The training box is a bold lime dashed
        outline (a white hatch fill was hard to read against the heatmap)
        so it reads clearly regardless of what's underneath it."""
        stiff = self._read_csv("stiffness_field_dense.csv")
        cross = self._read_csv("explosion_limit_crossover_locus.csv")
        if stiff is None or stiff.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 6.2))
        piv = stiff.pivot_table(index="P_atm", columns="T0", values="stiffness_ratio")
        Tv = piv.columns.values.astype(float)
        Pv = piv.index.values.astype(float)
        Zvals = piv.values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.log10(Zvals)
        mesh = ax.pcolormesh(Tv, Pv, Z, cmap="coolwarm", shading="nearest")
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(r"log$_{10}$(Jacobian stiffness ratio $|\lambda_{max}|/|\lambda_{min}|$)")
        ax.set_yscale("log")
        if cross is not None and not cross.empty:
            sub = cross[(cross["phi"] == 1.0) & (cross["diluent"] == "N2")].sort_values("P_atm")
            xs, ys = _spline_xy(sub["T_crossover_K"], sub["P_atm"], logy=True)
            ax.plot(xs, ys, color="cyan", lw=2.4,
                    label=r"actual 2nd-limit crossover (analytic, $\phi$=1, N$_2$)")
        y_lo, y_hi = ax.get_ylim()
        frac_1atm = (np.log10(1.0) - np.log10(y_lo)) / (np.log10(y_hi) - np.log10(y_lo))
        _draw_highlight_box_2d(ax, (1000, 1200), (0, frac_1atm), color="lime", lw=2.4,
                               label=f"{PAPER_CITATION} training box (explored)")
        ax.set_xlabel("Temperature $T$ [K]")
        ax.set_ylabel("Pressure [atm]")
        ax.set_title(f"Explored ({PAPER_CITATION}) vs. actual flammability limit,\n"
                     "colored by chemical stiffness -- unexplored region is where "
                     "the NeuralODE was never trained")
        ax.legend(fontsize=8, loc="upper left", facecolor="white", framealpha=0.85)
        return self._save(fig, "11_explored_vs_unexplored_stiffness")

    # -- 3) IDT grid, clubbed: one panel-figure per diluent ---------------
    def _load_idt_rows(self, sim_run_label="nominal_cantera"):
        """Shared loader for pipeline-driven Tig results, used by both
        plot_idt_grids and plot_idt_sensitivity so the two figures are
        built from identically-sourced data. Returns a DataFrame with
        T, P, phi, diluent, dilution_pct (a real percentage, e.g. 50.0 --
        journal convention, never the bare word "dil"), and IDT_us."""
        tig_dir = os.path.join(self.sim_runs_dir, sim_run_label, "Plot", "Dataset", "Tig")
        design_csv = os.path.join(self.h2_dir, "design", "design.csv")
        if not os.path.isdir(tig_dir) or not os.path.exists(design_csv):
            return pd.DataFrame()
        design = pd.read_csv(design_csv).set_index("dataset_id")
        rows = []
        for fn in os.listdir(tig_dir):
            if not fn.endswith(".csv"):
                continue
            ds_id = fn[:-4]
            if ds_id not in design.index:
                continue
            d = pd.read_csv(os.path.join(tig_dir, fn))
            if d.empty or d["Nominal"].iloc[0] in ("N/A", None) or pd.isna(d["Nominal"].iloc[0]):
                continue
            meta = design.loc[ds_id]
            rows.append(dict(T=meta["T"], P=meta["P_atm"], phi=meta["phi"],
                              diluent=meta["diluent"], dilution_level=meta["dilution_level"],
                              dilution_pct=float(meta["dilution_frac"]) * 100.0,
                              IDT_us=float(d["Nominal"].iloc[0])))
        df = pd.DataFrame(rows)
        if not df.empty:
            df["inv1000T"] = 1000.0 / df["T"]
        return df

    def plot_idt_grids(self, sim_run_label="nominal_cantera"):
        """One figure per diluent; each figure is an Arrhenius-style IDT
        plot (ln(tau) vs 1000/T, the standard form throughout the shock-
        tube/IDT literature -- e.g. Petersen & Hanson, Davidson, Hong et al.
        -- since tau = A*P^n*phi^m*exp(Ea/RT) is linear in this space),
        faceted by pressure into panels, with a secondary top axis in T[K]
        for readability. Each (phi, dilution) curve is spline-smoothed and
        drawn as a line only (no marker symbols) -- clubs what would
        otherwise be dozens of separate IDT plots into 3 figures total."""
        df = self._load_idt_rows(sim_run_label)
        if df.empty:
            return []
        out = []
        P_levels = sorted(df["P"].unique())
        for dil, dgrp in df.groupby("diluent"):
            fig, axes = plt.subplots(1, len(P_levels), figsize=(4.6 * len(P_levels), 4.3), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, P in zip(axes, P_levels):
                sub = dgrp[dgrp["P"] == P]
                for (phi, pct), g in sub.groupby(["phi", "dilution_pct"]):
                    g = g.sort_values("inv1000T")
                    if len(g) < 1:
                        continue
                    xs, ys = _spline_xy(g["inv1000T"], g["IDT_us"], logy=True)
                    ax.plot(xs, ys, lw=1.6, label=rf"$\phi$={phi:g}, {pct:.0f}% dilution")
                ax.set_yscale("log")
                ax.set_xlabel(r"1000/T [K$^{-1}$]")
                ax.set_title(f"P = {P:g} atm")
                ax.grid(alpha=0.3, which="both")
                # Secondary top axis in T[K] -- standard companion to an
                # Arrhenius x-axis so readers don't have to mentally invert.
                Tticks = np.array(sorted(df["T"].unique()))
                sec = ax.secondary_xaxis(
                    "top", functions=(lambda x: 1000.0 / np.where(x > 0, x, np.nan),
                                       lambda x: 1000.0 / np.where(x > 0, x, np.nan)))
                sec.set_xticks(1000.0 / Tticks)
                sec.set_xticklabels([f"{int(t)}" for t in Tticks], fontsize=7)
                sec.set_xlabel("$T$ [K]", fontsize=8)
            axes[0].set_ylabel(r"Ignition delay $\tau_{ig}$ [$\mu$s]")
            fig.suptitle(f"IDT (Arrhenius form), {dil} bath gas -- Cantera, h2o2_he.yaml")
            axes[-1].legend(fontsize=6, loc="best", ncol=1)
            fig.tight_layout()
            out.append(self._save(fig, f"03_idt_grid_{dil}"))
        return out

    # -- 3b) IDT sensitivity to P, phi, and dilution -- direct comparison ---
    def plot_idt_sensitivity(self, sim_run_label="nominal_cantera"):
        """Three-panel comparison of how ignition delay changes with each
        of P, phi, and dilution independently, holding the other two (and
        diluent identity) fixed -- the explicit "compare the difference in
        Tig for change in P and phi and dilution" figure. Spline-smoothed
        lines only, no markers; dilution is always shown as a percentage,
        never the bare word "dil"."""
        df = self._load_idt_rows(sim_run_label)
        if df.empty:
            return None
        T_ref = sorted(df["T"].unique())[len(df["T"].unique()) // 2]
        P_ref = sorted(df["P"].unique())[len(df["P"].unique()) // 2]
        phi_ref = 1.0 if 1.0 in df["phi"].unique() else sorted(df["phi"].unique())[len(df["phi"].unique()) // 2]
        dil_ref = "N2" if "N2" in df["diluent"].unique() else sorted(df["diluent"].unique())[0]

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

        # (a) IDT vs P, one line per phi, at T_ref/dil_ref/each dilution pct.
        ax = axes[0]
        sub = df[(df["T"] == T_ref) & (df["diluent"] == dil_ref)]
        for (phi, pct), g in sub.groupby(["phi", "dilution_pct"]):
            g = g.sort_values("P")
            if len(g) < 2:
                continue
            xs, ys = _spline_xy(g["P"], g["IDT_us"], logx=True, logy=True)
            ax.plot(xs, ys, lw=1.8, label=rf"$\phi$={phi:g}, {pct:.0f}% dilution")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pressure $P$ [atm]")
        ax.set_ylabel(r"Ignition delay $\tau_{ig}$ [$\mu$s]")
        ax.set_title(f"Pressure sensitivity, $T$={T_ref:g} K, {dil_ref} bath gas")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3, which="both")

        # (b) IDT vs phi, one line per P, at T_ref/dil_ref.
        ax = axes[1]
        sub = df[(df["T"] == T_ref) & (df["diluent"] == dil_ref)]
        for P, g in sub.groupby("P"):
            g = g.sort_values("phi")
            if len(g) < 2:
                continue
            xs, ys = _spline_xy(g["phi"], g["IDT_us"], logy=True)
            ax.plot(xs, ys, lw=1.8, label=f"P={P:g} atm")
        ax.set_yscale("log")
        ax.set_xlabel(r"Equivalence ratio $\phi$")
        ax.set_title(f"Equivalence-ratio sensitivity, $T$={T_ref:g} K, {dil_ref} bath gas")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

        # (c) IDT vs dilution %, one line per diluent identity, at T_ref/phi_ref.
        ax = axes[2]
        sub = df[(df["T"] == T_ref) & (df["phi"] == phi_ref)]
        for dil, g in sub.groupby("diluent"):
            g = g.sort_values("dilution_pct")
            if len(g) < 2:
                continue
            xs, ys = _spline_xy(g["dilution_pct"], g["IDT_us"], logy=True)
            ax.plot(xs, ys, lw=1.8, label=f"{dil}")
        ax.set_yscale("log")
        ax.set_xlabel("Bath-gas dilution [%]")
        ax.set_title(rf"Dilution sensitivity, $T$={T_ref:g} K, $\phi$={phi_ref:g}")
        ax.legend(fontsize=8, title="diluent")
        ax.grid(alpha=0.3, which="both")

        for ax in axes:
            ax.set_ylabel(r"Ignition delay $\tau_{ig}$ [$\mu$s]")
        fig.suptitle("Ignition-delay sensitivity to pressure, equivalence ratio, "
                     "and dilution -- Cantera, h2o2_he.yaml")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return self._save(fig, "12_idt_sensitivity")

    # -- 3b) IDT vs. temperature, one factor varied per panel -------------
    def plot_idt_vs_T_variation(self, sim_run_label="nominal_cantera"):
        """Companion to plot_idt_sensitivity: that figure holds T fixed and
        varies P, phi, and dilution; this one is explicitly IDT *vs.
        temperature* (Arrhenius form, 1000/T on x, same convention and
        secondary T[K] top axis as plot_idt_grids), with N2 dilution,
        pressure, equivalence ratio, and bath-gas identity each varied one
        at a time across four panels -- the other three held at reference
        levels (P_ref/phi_ref/dil_ref match plot_idt_sensitivity's own
        choice; dilution held at the level closest to 50%, common to all
        three bath gases). Spline-smoothed lines only, no markers;
        dilution always shown as a percentage."""
        df = self._load_idt_rows(sim_run_label)
        if df.empty:
            return None
        P_ref = sorted(df["P"].unique())[len(df["P"].unique()) // 2]
        phi_ref = 1.0 if 1.0 in df["phi"].unique() else sorted(df["phi"].unique())[len(df["phi"].unique()) // 2]
        dil_ref = "N2" if "N2" in df["diluent"].unique() else sorted(df["diluent"].unique())[0]
        pct_levels = sorted(df.loc[df["diluent"] == dil_ref, "dilution_pct"].unique())
        pct_ref = min(pct_levels, key=lambda p: abs(p - 50.0)) if pct_levels else None
        pct_txt = f"{pct_ref:.0f}%" if pct_ref is not None else "?"

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
        Tticks = np.array(sorted(df["T"].unique()))

        def _finish(ax, title):
            ax.set_yscale("log")
            ax.set_xlabel(r"1000/T [K$^{-1}$]")
            ax.set_ylabel(r"Ignition delay $\tau_{ig}$ [$\mu$s]")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3, which="both")
            sec = ax.secondary_xaxis(
                "top", functions=(lambda x: 1000.0 / np.where(x > 0, x, np.nan),
                                   lambda x: 1000.0 / np.where(x > 0, x, np.nan)))
            sec.set_xticks(1000.0 / Tticks)
            sec.set_xticklabels([f"{int(t)}" for t in Tticks], fontsize=7)
            sec.set_xlabel("$T$ [K]", fontsize=8)

        def _lines(ax, sub, group_col, label_fmt):
            for key, g in sub.groupby(group_col):
                g = g.sort_values("inv1000T")
                if len(g) < 2:
                    continue
                xs, ys = _spline_xy(g["inv1000T"], g["IDT_us"], logy=True)
                ax.plot(xs, ys, lw=1.8, label=label_fmt(key))

        # (a) vary N2 dilution %, hold P_ref/phi_ref/dil_ref
        ax = axes[0, 0]
        sub = df[(df["P"] == P_ref) & (df["phi"] == phi_ref) & (df["diluent"] == dil_ref)]
        _lines(ax, sub, "dilution_pct", lambda pct: f"{pct:.0f}% dilution")
        ax.legend(fontsize=7.5, title=f"{dil_ref} dilution")
        _finish(ax, rf"Dilution, $P$={P_ref:g} atm, $\phi$={phi_ref:g}, {dil_ref}")

        # (b) vary P, hold phi_ref/dil_ref/pct_ref
        ax = axes[0, 1]
        sub = df[(df["phi"] == phi_ref) & (df["diluent"] == dil_ref)]
        if pct_ref is not None:
            sub = sub[sub["dilution_pct"] == pct_ref]
        _lines(ax, sub, "P", lambda P: f"P={P:g} atm")
        ax.legend(fontsize=7.5, title="pressure")
        _finish(ax, rf"Pressure, $\phi$={phi_ref:g}, {dil_ref}, {pct_txt} dilution")

        # (c) vary phi, hold P_ref/dil_ref/pct_ref
        ax = axes[1, 0]
        sub = df[(df["P"] == P_ref) & (df["diluent"] == dil_ref)]
        if pct_ref is not None:
            sub = sub[sub["dilution_pct"] == pct_ref]
        _lines(ax, sub, "phi", lambda phi: rf"$\phi$={phi:g}")
        ax.legend(fontsize=7.5, title="equivalence ratio")
        _finish(ax, rf"Equivalence ratio, $P$={P_ref:g} atm, {dil_ref}, {pct_txt} dilution")

        # (d) vary bath-gas identity, hold P_ref/phi_ref/pct_ref
        ax = axes[1, 1]
        sub = df[(df["P"] == P_ref) & (df["phi"] == phi_ref)]
        if pct_ref is not None:
            sub = sub[sub["dilution_pct"] == pct_ref]
        _lines(ax, sub, "diluent", lambda dil: dil)
        ax.legend(fontsize=7.5, title="bath gas")
        _finish(ax, rf"Bath gas, $P$={P_ref:g} atm, $\phi$={phi_ref:g}, {pct_txt} dilution")

        fig.suptitle("Ignition-delay variation with temperature -- Cantera, h2o2_he.yaml", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return self._save(fig, "03b_idt_vs_T_variation")

    # -- 4) species time-history profiles, clubbed by target group -------
    def plot_species_profiles(self, sim_run_label="nominal_cantera", max_examples=6):
        """Representative species-vs-time profiles (from saveAll=True
        time_history.csv dumps). *max_examples* cases are chosen to span
        the design envelope (T, P, phi, bath gas, dilution) as widely as
        possible -- rather than just the first N case dirs on disk, which
        used to all land at the same (T=1000 K, phi=0.3) corner and differ
        only in bath gas -- restricted to cases that actually ignite
        inside the fixed 0-2 ms window every panel here shares, so the
        figures read as genuinely different conditions rather than the
        same S-curve at different time scales. Titles report the physical
        condition (composition, phi, T, P), never the internal TIG_####
        dataset id."""
        nominal_dir = os.path.join(self.sim_runs_dir, sim_run_label, "NOMINAL")
        target_data = os.path.join(self.sim_runs_dir, sim_run_label, "target_data.txt")
        if not os.path.isdir(nominal_dir) or not os.path.exists(target_data):
            return []

        def _fmt_num(v):
            try:
                return f"{float(v):g}"
            except (TypeError, ValueError):
                return str(v)

        # -- parse target_data.txt: ds_id|Tig|{comp}|T_K|P_Pa|phi|... ----
        conds = {}
        with open(target_data) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split("|")
                if len(parts) < 6:
                    continue
                try:
                    comp = ast.literal_eval(parts[2])
                except (ValueError, SyntaxError):
                    comp = {}
                dil_keys = [k for k in comp if k not in ("H2", "O2")]
                dil = dil_keys[0] if dil_keys else None
                conds[idx] = dict(ds_id=parts[0], comp=comp, T=float(parts[3]),
                                   P_atm=float(parts[4]) / 101325.0, phi=float(parts[5]),
                                   diluent=dil, dilution_frac=comp.get(dil, 0.0) if dil else 0.0)

        case_dirs = sorted(
            [d for d in os.listdir(nominal_dir) if d.startswith("case-")],
            key=lambda s: int(s.split("-")[1]),
        )

        species_of_interest = ["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2"]

        def _transition_time_s(d):
            """First time the most-swinging present species crosses 90%
            (or 10%, depending on direction) of its own total swing -- the
            actual visible location of the ignition transient in this
            *saved* trajectory. Deliberately not tau.out's ignition delay:
            tau.out uses a fixed criterion (peak-OH time) that can fall
            far later than the visible transient for very hot, fast-
            igniting conditions -- e.g. a T0=2800 K case whose majors
            finish moving in under 1 microsecond can still report a
            tau.out of ~75 microseconds, which would wrongly pass a
            tau.out-based window check yet show only a flat post-ignition
            plateau across the whole fixed 0-2 ms frame."""
            t_s = np.asarray(d["t"] if "t" in d.columns else d.index, dtype=float)
            present = [s for s in species_of_interest if s in d.columns]
            if not present or len(t_s) < 2:
                return None
            ref = d[present[0]].to_numpy(dtype=float)
            for s in present:
                sv = d[s].to_numpy(dtype=float)
                if (sv.max() - sv.min()) > (ref.max() - ref.min()):
                    ref = sv
            lo, hi = ref.min(), ref.max()
            if not (hi > lo):
                return None
            thresh = lo + 0.9 * (hi - lo) if ref[-1] >= ref[0] else lo + 0.1 * (hi - lo)
            crossed = np.where((ref >= thresh) if ref[-1] >= ref[0] else (ref <= thresh))[0]
            if not len(crossed):
                return None
            return float(t_s[crossed[0]])

        # -- candidates: have a time_history.csv whose actual transition
        #    time (not tau.out's, see above) falls inside a window where
        #    the fixed 0-2 ms frame below is legible -- neither a
        #    near-vertical spike at t~0 nor a flat, already-relaxed line
        #    all the way across. The DataFrame is cached on the candidate
        #    so the plotting loop below doesn't re-read it. -------------
        T_TRANS_LO_S, T_TRANS_HI_S = 50e-6, 1.9e-3
        candidates = []
        for cd in case_dirs:
            th_path = os.path.join(nominal_dir, cd, "time_history.csv")
            if not os.path.exists(th_path):
                continue
            idx = int(cd.split("-")[1])
            meta = conds.get(idx)
            if meta is None or meta["diluent"] is None:
                continue
            try:
                d = pd.read_csv(th_path, index_col=0)
            except Exception:
                continue
            t_trans = _transition_time_s(d)
            if t_trans is None:
                continue
            candidates.append(dict(cd=cd, df=d, t_trans_s=t_trans, **meta))

        pool = [c for c in candidates if T_TRANS_LO_S <= c["t_trans_s"] <= T_TRANS_HI_S]
        widen = 1  # fall back to a wider transition-time window rather than
        while len(pool) < max_examples and widen <= 4 and candidates:  # returning too few examples, if the design under-populates the base window
            lo, hi = T_TRANS_LO_S / (2.0 ** widen), T_TRANS_HI_S * (1.2 ** widen)
            pool = [c for c in candidates if lo <= c["t_trans_s"] <= hi]
            widen += 1
        if not pool:
            pool = candidates
        if not pool:
            return []

        # -- greedy farthest-point selection over the normalized design
        #    tuple (T, P, phi, diluent identity, dilution fraction), so
        #    the chosen cases spread across the envelope instead of
        #    clustering at one corner. Deterministic (no RNG): starts
        #    from the lowest-index candidate and always adds whichever
        #    remaining candidate is furthest from everything chosen so far.
        diluents = sorted({c["diluent"] for c in pool})

        def _feat(c):
            return np.array([
                np.log10(max(c["T"], 1.0)),
                np.log10(max(c["P_atm"], 1e-3)),
                np.log10(max(c["phi"], 1e-3)),
                float(diluents.index(c["diluent"])) * 2.0,
                c["dilution_frac"] * 5.0,
            ])

        feats = {id(c): _feat(c) for c in pool}
        F = np.array(list(feats.values()))
        f_lo, f_hi = F.min(axis=0), F.max(axis=0)
        span = np.where(f_hi > f_lo, f_hi - f_lo, 1.0)
        norm_feats = {k: (v - f_lo) / span for k, v in feats.items()}

        chosen = [pool[0]]
        chosen_ids = {id(pool[0])}
        while len(chosen) < min(max_examples, len(pool)):
            best, best_dist = None, -1.0
            for c in pool:
                if id(c) in chosen_ids:
                    continue
                d = min(np.linalg.norm(norm_feats[id(c)] - norm_feats[id(s)]) for s in chosen)
                if d > best_dist:
                    best, best_dist = c, d
            chosen.append(best)
            chosen_ids.add(id(best))
        chosen.sort(key=lambda c: int(c["cd"].split("-")[1]))

        T_WINDOW_MS = 2.0
        out = []
        for c in chosen:
            d = c["df"]
            present = [s for s in species_of_interest if s in d.columns]
            if not present:
                continue
            # The Cantera SolutionArray dump (make_input_file.py's
            # `timeHistory.to_csv("time_history.csv")`) writes time as the
            # unnamed first/index column -- pd.read_csv(index_col=0) above
            # captures it as the DataFrame index regardless of its header
            # name (empty string, not literally "t"), so just use the index.
            t_s = np.asarray(d["t"] if "t" in d.columns else d.index, dtype=float)
            t_ms = t_s * 1e3  # fixed 0-2 ms window, same for every case (per instruction)

            fig, ax1 = plt.subplots(figsize=(6, 4.2))
            ax2 = ax1.twinx()
            for s in present:
                if s in ("H2", "O2", "H2O"):
                    ax1.plot(t_ms, d[s], label=s)
                else:
                    ax2.plot(t_ms, d[s], "--", label=s, alpha=0.7)
            ax1.set_xlim(0, T_WINDOW_MS)
            ax1.set_xlabel("time [ms]")
            ax1.set_ylabel("Y (major species)")
            ax2.set_ylabel("Y (radicals, dashed)")

            comp = c["comp"]
            comp_keys = [k for k in ("H2", "O2") if k in comp] + ([c["diluent"]] if c["diluent"] else [])
            comp_str = "/".join(comp_keys) + " = " + ":".join(_fmt_num(comp[k]) for k in comp_keys)
            ax1.set_title(
                f"{comp_str} (mole fraction)\n"
                rf"$\phi$={_fmt_num(c['phi'])}, $T$={_fmt_num(c['T'])} K, $P$={_fmt_num(c['P_atm'])} atm",
                fontsize=10,
            )
            l1, lab1 = ax1.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="center right")
            out.append(self._save(fig, f"04_species_profile_{c['cd']}"))
        return out

    # -- 5) flame speed + thickness, clubbed by diluent -------------------
    def plot_flame_speed(self):
        df = self._read_csv("flame_thickness.csv")
        if df is None:
            return []
        out = []
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, (dil, dgrp) in zip(axes, df.groupby("diluent")):
            for P, g in dgrp.groupby("P_atm"):
                g = g[g["converged"] == True].sort_values("phi")  # noqa: E712
                if len(g) < 2:
                    continue
                xs, ys = _spline_xy(g["phi"], g["S_L_cm_s"], shape_preserving=True)
                ax.plot(xs, ys, lw=1.8, label=f"P={P:g} atm")
            ax.set_xlabel(r"Equivalence ratio $\phi$")
            ax.set_title(f"{dil}, 50% dilution")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Laminar flame speed [cm/s]")
        axes[-1].legend(fontsize=8)
        fig.suptitle(r"Laminar flame speed vs. $\phi$ -- Cantera, h2o2_he.yaml")
        out.append(self._save(fig, "05_flame_speed"))

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, (dil, dgrp) in zip(axes2, df.groupby("diluent")):
            for P, g in dgrp.groupby("P_atm"):
                g = g[g["converged"] == True].sort_values("phi")  # noqa: E712
                if len(g) < 2:
                    continue
                xs, ys = _spline_xy(g["phi"], np.array(g["flame_thickness_m"]) * 1e3, logy=True, shape_preserving=True)
                ax.plot(xs, ys, lw=1.8, label=f"P={P:g} atm")
            ax.set_yscale("log")
            ax.set_xlabel(r"Equivalence ratio $\phi$")
            ax.set_title(f"{dil}, 50% dilution")
            ax.grid(alpha=0.3, which="both")
        axes2[0].set_ylabel("Flame thickness [mm]")
        axes2[-1].legend(fontsize=8)
        fig2.suptitle("Laminar flame thickness")
        fig2.tight_layout()
        out.append(self._save(fig2, "05b_flame_thickness"))
        return out

    # -- 6) extinction strain rate, clubbed by diluent --------------------
    def plot_extinction(self):
        df = self._read_csv("extinction_strain_rates.csv")
        if df is None:
            return None
        df = df[df["converged"] == True]  # noqa: E712
        if df.empty:
            return None
        # Not every (P, dilution) combo converged for every diluent, so a
        # per-axes automatic color cycle would assign the same color to
        # different P values across panels (and reuse colors within a
        # panel) purely by which combos happen to be missing -- assign
        # color by P (fixed across all three panels) and linestyle by
        # dilution fraction instead, so the legend means the same thing in
        # every panel regardless of missing combos.
        P_levels = sorted(df["P_atm"].unique())
        cmap = plt.get_cmap("tab10")
        color_by_P = {P: cmap(i % 10) for i, P in enumerate(P_levels)}
        frac_levels = sorted(df["dilution_frac"].unique())
        style_by_frac = {f: ls for f, ls in zip(frac_levels, ["-", "--", ":", "-."])}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        drawn_combos = set()
        for ax, (dil, dgrp) in zip(axes, df.groupby("diluent")):
            for (P, frac), g in dgrp.groupby(["P_atm", "dilution_frac"]):
                g = g.sort_values("phi")
                # A single-phi combo (design has a phi-sweep only at some
                # (P, dilution) corners, with just the centre point phi=1
                # elsewhere) collapses to one point after de-duplication --
                # nothing to draw as a line, and skip it from the legend.
                if g["phi"].nunique() < 2:
                    continue
                xs, ys = _spline_xy(g["phi"], g["a_ext_1_s"], logy=True)
                ax.plot(xs, ys, lw=1.8, color=color_by_P[P], linestyle=style_by_frac[frac],
                         label=f"P={P:g} atm, {frac * 100:.0f}% dilution")
                drawn_combos.add((P, frac))
            ax.set_yscale("log")
            ax.set_xlabel(r"Equivalence ratio $\phi$")
            ax.set_title(f"{dil} dilution")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r"Extinction strain rate $a_{ext}$ [1/s]")
        # Combined legend built from every (P, dilution) combo that was
        # actually drawn in at least one panel (not the full Cartesian
        # product, since coverage differs by diluent and some combos have
        # no phi-sweep at all).
        handles = [plt.Line2D([0], [0], color=color_by_P[P], linestyle=style_by_frac[frac], lw=1.8,
                               label=f"P={P:g} atm, {frac * 100:.0f}% dilution")
                   for P in P_levels for frac in frac_levels if (P, frac) in drawn_combos]
        fig.legend(handles=handles, fontsize=6.5, loc="upper right",
                   bbox_to_anchor=(0.99, 0.98), ncol=1)
        fig.suptitle("Twin premixed counterflow extinction strain rate")
        fig.tight_layout(rect=(0, 0, 0.88, 1))
        return self._save(fig, "06_extinction_strain")

    # -- 7) JSR S-curve bistability, clubbed by diluent --------------------
    def plot_jsr_scurve(self):
        """JSR ignition (light-off) branch: cold-started from the fresh
        feed composition at each T independently -- robust and standard,
        but a single branch, not a hysteresis loop (see H2/review/flag.md
        for why a full bistable-continuation attempt is not shipped: it
        diverged numerically on this reactor/mechanism combination). The
        paper's JSR-bistability claim rests on the separate, independently
        reproduced pipeline convergence failure at the S-curve turning
        point, not on this figure."""
        df = self._read_csv("jsr_scurve_continuation.csv")
        if df is None:
            return []
        df = df[df["converged"] == True]  # noqa: E712
        out = []
        for dil, dgrp in df.groupby("diluent"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
            for ax, (P, pgrp) in zip(axes, dgrp.groupby("P_atm")):
                for phi, g in pgrp.groupby("phi"):
                    g = g.sort_values("T")
                    if len(g) < 2:
                        continue
                    xs, ys = _spline_xy(g["T"], g["X_H2O"], logy=True)
                    ax.plot(xs, ys, lw=1.6, label=rf"$\phi$={phi:g}")
                ax.set_yscale("log")
                ax.set_xlabel("$T$ [K]")
                ax.set_title(f"P={P:g} atm")
                ax.grid(alpha=0.3, which="both")
            frac = dgrp["dilution_frac"].iloc[0] if "dilution_frac" in dgrp.columns else None
            pct_txt = f", {frac * 100:.0f}% dilution" if frac is not None else ""
            axes[0].set_ylabel(r"$X$(H$_2$O) steady state")
            axes[-1].legend(fontsize=8, ncol=1)
            fig.suptitle(f"JSR ignition branch (cold-started per T, $\\tau$=100 ms), {dil}{pct_txt}")
            fig.tight_layout()
            out.append(self._save(fig, f"07_jsr_scurve_{dil}"))
        return out

    # -- 8) stiffness diagnostics ------------------------------------------
    def plot_stiffness(self):
        """Prefers the dense (T,P) stiffness field from
        H2/standalone/explosion_limit_map.py (30x14 grid) over the sparse
        original 10x4 grid, when available, purely for a smoother spline
        fit -- same underlying method (peak-[OH] state, finite-difference
        Jacobian), just more points. Subsamples down to at most 7
        representative pressure levels for plotting (the full 14-level
        dense field is used to FIT each line, just not all 14 drawn) so a
        normal, explicit "each line = one pressure" legend stays readable,
        per instruction, rather than a colorbar."""
        df = self._read_csv("stiffness_field_dense.csv")
        if df is None:
            df = self._read_csv("stiffness_diagnostics.csv")
        if df is None:
            return None
        P_levels_all = sorted(df["P_atm"].unique())
        max_lines = 7
        step = max(1, int(np.ceil(len(P_levels_all) / max_lines)))
        P_levels_plotted = P_levels_all[::step]
        if P_levels_all[-1] not in P_levels_plotted:
            P_levels_plotted.append(P_levels_all[-1])
        cmap = plt.get_cmap("viridis")
        color_by_P = {P: cmap(i / max(1, len(P_levels_plotted) - 1))
                      for i, P in enumerate(P_levels_plotted)}
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for P in P_levels_plotted:
            g = df[df["P_atm"] == P].sort_values("T0")
            if len(g) < 2:
                continue
            xs, ys = _spline_xy(g["T0"], g["stiffness_ratio"], logy=True, denoise=True)
            ax.plot(xs, ys, lw=1.8, color=color_by_P[P], label=f"P = {P:.3g} atm")
        ax.set_yscale("log")
        ax.set_xlabel("Temperature $T$ [K]")
        ax.set_ylabel(r"Jacobian stiffness ratio $|\lambda_{max}|/|\lambda_{min}|$")
        ax.set_title("Chemical stiffness vs. temperature (stoichiometric H$_2$/N$_2$)")
        ax.legend(title="Each line = one pressure", fontsize=8, title_fontsize=8,
                  loc="best", ncol=2 if len(P_levels_plotted) > 4 else 1)
        ax.grid(alpha=0.3)
        return self._save(fig, "08_stiffness_diagnostics")

    # -- 9) ROP dominance spectra, clubbed across representative conditions -
    def plot_rop_spectra(self, top_n=8):
        """Tornado-style rate-of-progress spectra: within each condition,
        bars are normalized by that condition's own max |rate| (raw rates
        span ~1e-9 to ~1e5 mol/m3/s across the sweep -- an absolute-value
        linear axis makes every bar but the single largest invisible,
        which is the standard failure mode this avoids), sorted by
        magnitude, and colored by net direction (forward/reverse) rather
        than a single uninformative color -- the standard "tornado plot"
        convention for rate-of-production/sensitivity spectra."""
        df = self._read_csv("rop_spectra.csv")
        if df is None:
            return None
        conditions = list(df.groupby(["T0", "P_atm", "phi"]).groups.keys())
        n = len(conditions)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 0.45 * top_n + 2.2 + 2.6 * (nrows - 1)))
        axes = np.atleast_2d(axes)
        for i, (T0, P0, phi0) in enumerate(conditions):
            ax = axes[i // ncols][i % ncols]
            sub = df[(df["T0"] == T0) & (df["P_atm"] == P0) & (df["phi"] == phi0)].copy()
            sub["abs_rate"] = sub["net_rate_of_progress"].abs()
            sub = sub.sort_values("abs_rate", ascending=False).head(top_n)
            peak = sub["abs_rate"].max()
            norm = sub["net_rate_of_progress"] / peak if peak > 0 else sub["net_rate_of_progress"] * 0
            colors = ["tab:red" if v < 0 else "tab:blue" for v in norm]
            ypos = range(len(sub) - 1, -1, -1)  # largest at top
            ax.barh(list(ypos), norm, color=colors, edgecolor="k", linewidth=0.4, height=0.7)
            ax.set_yticks(list(ypos))
            labels = [e if len(e) <= 34 else e[:31] + "..." for e in sub["equation"]]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(-1.15, 1.15)
            ax.set_title(rf"T={T0:g} K, P={P0:g} atm, $\phi$={phi0:g}"
                         f"\n(peak |ROP| = {peak:.2e} mol/m$^3$/s)", fontsize=9)
            ax.axvline(0, color="k", lw=0.8)
            ax.grid(axis="x", alpha=0.3)
            ax.set_xlabel("Net ROP / max|ROP| in this condition", fontsize=8)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
                   plt.Rectangle((0, 0), 1, 1, color="tab:red")]
        fig.legend(handles, ["net forward", "net reverse"], loc="upper right",
                   ncol=2, fontsize=9, bbox_to_anchor=(0.99, 1.0))
        fig.suptitle(f"Top-{top_n} rate-of-progress spectra by condition (normalized per panel)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return self._save(fig, "09_rop_spectra")

    # -- 10) equilibrium / adiabatic flame temperature ----------------------
    def plot_equilibrium(self):
        """Standard adiabatic flame temperature vs phi (reactants at a
        fixed 298 K reference, the textbook convention -- Turns/Law -- so
        this is directly comparable to any standard combustion reference
        curve), smooth-sampled and peaking near stoichiometric. Ar and He
        are merged into one "monatomic diluent" panel rather than shown as
        two visually-identical panels: both are chemically inert,
        monatomic ideal gases with the same molar Cp=5/2 R, so at equal
        mole-fraction dilution the constant-HP equilibrium is numerically
        the same problem and their T_ad curves coincide. That coincidence
        is verified at runtime below (not assumed) -- if a future
        mechanism/thermo update ever actually separated them, this falls
        back to three separate panels rather than silently hiding it."""
        df = self._read_csv("equilibrium_flame_temperature.csv")
        if df is None:
            return None
        # H2/standalone/extras.py's DIL_LEVELS -- "low"/"high" map to real
        # bath-gas mole-fraction percentages (journal convention: report the
        # percentage directly, never the bare word "dil").
        level_pct = {"low": 50.0, "high": 90.0, "air": 79.0}

        ARHE_TOL_K = 1.0
        merge_ar_he, ar_he_diff = False, np.nan
        if {"AR", "HE"}.issubset(set(df["diluent"].unique())):
            piv = df.pivot_table(index=["T0", "P_atm", "phi", "dilution_level"],
                                  columns="diluent", values="T_adiabatic_K").dropna(subset=["AR", "HE"])
            if not piv.empty:
                ar_he_diff = float((piv["AR"] - piv["HE"]).abs().max())
                merge_ar_he = ar_he_diff <= ARHE_TOL_K

        if merge_ar_he:
            panels = [("Ar or He (monatomic diluent)", df[df["diluent"] == "AR"]),
                      ("N$_2$ (diatomic diluent)", df[df["diluent"] == "N2"])]
        else:
            panels = [(f"{dil} dilution", dgrp) for dil, dgrp in df.groupby("diluent")]

        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, dgrp) in zip(axes, panels):
            for (P, level), g in dgrp.groupby(["P_atm", "dilution_level"]):
                g = g.sort_values("phi")
                if len(g) < 2:
                    continue
                pct = level_pct.get(level, level)
                xs, ys = _spline_xy(g["phi"], g["T_adiabatic_K"])
                ax.plot(xs, ys, lw=1.6, label=f"P={P:g} atm, {pct:g}% dilution")
            ax.axvline(1.0, color="k", lw=0.6, ls=":")
            ax.set_xlabel(r"Equivalence ratio $\phi$")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r"Adiabatic flame temperature $T_{ad}$ [K] (298 K reactants)")
        axes[-1].legend(fontsize=6)
        if merge_ar_he:
            note = (
                rf"Ar and He coincide to within {ar_he_diff:.1e} K: both are monatomic, chemically"
                "\n"
                r"inert ideal gases with $C_p=\frac{5}{2}R$, so at equal mole-fraction dilution the"
                "\n"
                r"constant-HP equilibrium is numerically the same problem. N$_2$ is diatomic"
                "\n"
                r"($C_p=\frac{7}{2}R$ + vibrational modes) and absorbs more enthalpy, lowering $T_{ad}$."
            )
            fig.suptitle("Equilibrium (HP) adiabatic flame temperature", y=0.99)
            fig.text(0.5, 0.89, note, ha="center", va="top", fontsize=7.5, style="italic")
            fig.tight_layout(rect=(0, 0, 1, 0.78))
        else:
            fig.suptitle("Equilibrium (HP) adiabatic flame temperature")
            fig.tight_layout()
        return self._save(fig, "10_equilibrium_Tad")

    # -- run everything -----------------------------------------------------
    def run_all(self, sim_run_label="nominal_cantera"):
        """Generate the full clubbed figure set. Returns a dict of
        {figure_key: path_or_list}. Any single plot's failure is caught and
        recorded rather than aborting the rest (same defensiveness as
        postprocess() below)."""
        results = {}
        for key, fn, kwargs in [
            ("crossover_locus", self.plot_crossover_locus, {}),
            ("key_temperature_comparison", self.plot_key_temperature_comparison, {}),
            ("ignitability_map", self.plot_ignitability_map, {}),
            ("ignitability_3d", self.plot_ignitability_3d, {}),
            ("stiffness_3d", self.plot_stiffness_3d, {}),
            ("idt_grids", self.plot_idt_grids, {"sim_run_label": sim_run_label}),
            ("idt_sensitivity", self.plot_idt_sensitivity, {"sim_run_label": sim_run_label}),
            ("idt_vs_T_variation", self.plot_idt_vs_T_variation, {"sim_run_label": sim_run_label}),
            ("species_profiles", self.plot_species_profiles, {"sim_run_label": sim_run_label}),
            ("flame_speed", self.plot_flame_speed, {}),
            ("extinction", self.plot_extinction, {}),
            ("jsr_scurve", self.plot_jsr_scurve, {}),
            ("stiffness", self.plot_stiffness, {}),
            ("rop_spectra", self.plot_rop_spectra, {}),
            ("equilibrium", self.plot_equilibrium, {}),
            ("flammability_vs_paper_training", self.plot_flammability_vs_paper_training, {}),
        ]:
            try:
                results[key] = fn(**kwargs)
            except Exception as exc:
                results[key] = None
                results[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
        return results


#: Ea/R -> Ea conversion factor (kJ/mol), matching this repo's own convention
#: in Input_file_reader.py (``R = 8.314E-03``, Ea in kJ/mol).
_R_KJ = 8.314e-3

_MUQSAC_PATTERNS = {
    "unsrt": "**/unsrt.pkl",
    "solution_zeta": "**/solution_zeta.save",
    "posterior_cov": "**/posterior_cov.pkl",
    "optimized_mech": "**/Optimized_mech.yaml",
    "prediction_errors": "**/Prediction_errors.csv",
    "prediction_errors_detailed": "**/Prediction_errors_detailed.csv",
    "target_files": "**/*.target",
}


def _load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_solution_zeta(path):
    lines = open(path, "r").readlines()
    return [float(line.split("=")[1].strip()) for line in lines if "=" in line]


def _per_reaction_zeta_and_cov(unsrt_data, opt_zeta, posterior_cov):
    """Same inline reconstruction run_script.py does right before building
    each PostOptPlotter (see VisualAid.py's module docstring above) --
    duplicated here rather than imported since it is a ~10-line loop in
    run_script.py's own module scope, not a function it exposes."""
    opt_zeta_arr = np.asarray(opt_zeta, dtype=float)
    rxn_slices = posterior_cov["rxn_slices"]
    Sigma_p = posterior_cov["Sigma_p"]
    out = {}
    offset = 0
    for rxn in unsrt_data:
        m_j = len(unsrt_data[rxn].activeParameters)
        z_rxn = opt_zeta_arr[offset: offset + m_j]
        offset += m_j

        sl = rxn_slices[rxn]
        Sp_rxn = Sigma_p[sl, sl]

        z_full = np.zeros(3, dtype=float)
        sel = np.asarray(unsrt_data[rxn].selection, dtype=int)
        act = [i for i, s in enumerate(sel) if s == 1]
        for local_i, global_i in enumerate(act):
            if local_i < len(z_rxn):
                z_full[global_i] = z_rxn[local_i]
        out[rxn] = (z_full, Sp_rxn)
    return out


def _arrhenius_table_row(rxn, u, z_full):
    """One row of the --journal Arrhenius parameter table: prior vs.
    optimised A/n/Ea for one reaction, straight from the same p0/L/zeta_opt
    PostOptPlotter's curves are built from."""
    p0 = np.asarray(u.nominal, dtype=float)
    L = np.asarray(u.cholskyDeCorrelateMat, dtype=float)
    p_opt = p0 + L @ z_full
    return {
        "reaction": rxn,
        "prior_lnA": p0[0], "prior_n": p0[1], "prior_Ea_kJ_per_mol": p0[2] * _R_KJ,
        "optimized_lnA": p_opt[0], "optimized_n": p_opt[1],
        "optimized_Ea_kJ_per_mol": p_opt[2] * _R_KJ,
        "delta_lnA": p_opt[0] - p0[0], "delta_n": p_opt[1] - p0[1],
        "delta_Ea_kJ_per_mol": (p_opt[2] - p0[2]) * _R_KJ,
    }


def postprocess(run_dir, out_dir=None, journal=False, csv_only=False,
                 formats=("png",), intermediate=False):
    """Regenerate MUQ-SAC optimisation plots/tables from an existing (or
    still-running, with ``intermediate=True``) ``run_script.py`` run
    directory. Never raises on missing files -- whatever inputs aren't
    present yet (a run still in its PRS-fitting stage, before optimisation
    has produced ``solution_zeta.save``) are recorded in
    ``result.skipped`` rather than failing the whole call, so this is safe
    to call repeatedly against a run directory while it's still going.

    Returns a :class:`mso_io.postprocess_base.PostProcessResult`.
    """
    if PostProcessResult is None:
        raise ImportError("mso_io.postprocess_base not importable -- is src/ on sys.path?")

    run_dir = os.path.abspath(run_dir)
    out_dir = str(default_out_dir(run_dir, out_dir))
    found = discover(run_dir, _MUQSAC_PATTERNS)
    result = PostProcessResult(run_dir=run_dir, out_dir=out_dir, partial=bool(intermediate))

    figures_dir = os.path.join(out_dir, "figures")
    tables_dir = os.path.join(out_dir, "tables")

    # -- Per-reaction posterior plots (unsrt.pkl + solution_zeta.save + posterior_cov.pkl) --
    have_posterior_inputs = (found["unsrt"] and found["solution_zeta"] and found["posterior_cov"])
    if have_posterior_inputs:
        try:
            unsrt_data = _load_pickle(str(found["unsrt"][0]))
            opt_zeta = _load_solution_zeta(str(found["solution_zeta"][0]))
            posterior_cov = _load_pickle(str(found["posterior_cov"][0]))
            per_rxn = _per_reaction_zeta_and_cov(unsrt_data, opt_zeta, posterior_cov)

            arrhenius_rows = []
            for rxn, (z_full, Sp_rxn) in per_rxn.items():
                try:
                    plotter = PostOptPlotter(unsrt_data=unsrt_data, rxn=rxn,
                                              zeta_opt=z_full, Sigma_p_rxn=Sp_rxn, n_sigma=1)
                    if not csv_only:
                        fig_path = plotter.plot(location=figures_dir)
                        result.figures.append(fig_path)

                    # Curve data -> CSV, unconditionally (the class's own
                    # curve-building methods, called as any external caller
                    # of this public-ish API would -- PostOptPlotter itself
                    # is not modified).
                    x = 1e3 / plotter.T
                    k_nom = plotter._nominal_curve()
                    k_prior_lo, k_prior_hi = plotter._prior_limits()
                    k_opt = plotter._optimised_curve()
                    k_post_lo, k_post_hi = plotter._posterior_band()
                    safe_rxn = str(rxn).replace("/", "_").replace(" ", "_")
                    df = pd.DataFrame({
                        "T_K": plotter.T, "inv1000T": x,
                        "log10k_nominal": k_nom,
                        "log10k_prior_lo": k_prior_lo, "log10k_prior_hi": k_prior_hi,
                        "log10k_optimized": k_opt,
                        "log10k_posterior_lo": k_post_lo, "log10k_posterior_hi": k_post_hi,
                    })
                    result.tables.append(write_table(df, f"{safe_rxn}_posterior", tables_dir))

                    if journal:
                        arrhenius_rows.append(_arrhenius_table_row(rxn, unsrt_data[rxn], z_full))
                except Exception as exc:  # a single bad reaction shouldn't abort the run
                    result.skipped.append(f"reaction {rxn}: {type(exc).__name__}: {exc}")

            if journal and arrhenius_rows:
                journal_dir = os.path.join(out_dir, "journal_data")
                result.journal_tables.append(
                    write_table(pd.DataFrame(arrhenius_rows), "arrhenius_parameters", journal_dir))
        except Exception as exc:
            result.skipped.append(f"posterior plots: {type(exc).__name__}: {exc}")
    else:
        missing = [k for k in ("unsrt", "solution_zeta", "posterior_cov") if not found[k]]
        result.skipped.append(
            f"posterior plots: optimisation not yet at that stage (missing {', '.join(missing)})"
            if intermediate else
            f"posterior plots: no {', '.join(missing)} found under {run_dir}")

    # -- Cost / prediction-error CSVs (already CSVs -- just relocate + always tracked) --
    for key, label in (("prediction_errors", "prediction_errors"),
                        ("prediction_errors_detailed", "prediction_errors_detailed")):
        for src_path in found[key]:
            df = pd.read_csv(src_path)
            result.tables.append(write_table(df, label, tables_dir))
            if journal:
                result.journal_tables.append(
                    write_table(df, f"summary_{label}", os.path.join(out_dir, "journal_data")))

    # -- TargetSet plots, if a .target file is discoverable --
    if found["target_files"]:
        try:
            target_path = str(found["target_files"][0])
            add_path = os.path.splitext(target_path)[0] + ".add"
            target_set = TargetSet(target_path, add_path if os.path.exists(add_path) else None)
            tsp = TargetSetPlotter(target_set)
            if not csv_only:
                os.makedirs(figures_dir, exist_ok=True)
                p1 = tsp.plot_condition_coverage(
                    save_path=os.path.join(figures_dir, "condition_coverage.png"))
                p2 = tsp.plot_observed_vs_deviation(
                    save_path=os.path.join(figures_dir, "observed_vs_deviation.png"))
                result.figures.extend([p for p in (p1, p2) if p])
            result.tables.append(write_table(target_set.to_dataframe(), "target_set", tables_dir))
        except Exception as exc:
            result.skipped.append(f"target set plots: {type(exc).__name__}: {exc}")
    else:
        result.skipped.append(
            f"target set plots: no .target file found under {run_dir} "
            "(NominalParityAnalyzer additionally needs nominal_results, which "
            "run_script.py does not persist to disk -- not attempted)")

    write_manifest(result, out_dir)
    return result


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MUQ-SAC VisualAid: standalone plotter class library, "
                     "plus --postprocess to regenerate plots/tables from an "
                     "existing run_script.py run directory.")
    add_postprocess_args(parser)
    args = parser.parse_args()
    if not args.postprocess:
        parser.error("nothing to do -- pass --postprocess RUN_DIR "
                      "(this file is a library when imported; run_script.py "
                      "does that. This CLI is only for post-processing.)")
    result = postprocess(
        args.postprocess, out_dir=args.out, journal=args.journal,
        csv_only=args.csv_only, formats=formats_from_args(args),
        intermediate=args.intermediate,
    )
    print(f"[VisualAid.postprocess] {len(result.figures)} figure(s), "
          f"{len(result.tables)} table(s), {len(result.journal_tables)} journal table(s) "
          f"-> {result.out_dir}")
    for note in result.skipped:
        print(f"  skipped: {note}")


if __name__ == "__main__":
    _main()

