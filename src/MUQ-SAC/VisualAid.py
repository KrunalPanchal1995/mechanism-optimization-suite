import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
		func = [(i.T.dot(L.dot(z))) for i in Theta.T]
		return np.asarray(func)
	
	def getPerturbed_n_curve(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		z = np.array([0,1,0])
		func = [(i.T.dot(L.dot(z))) for i in Theta.T]
		return np.asarray(func)
	
	def getPerturbed_Ea_curve(self):
		L = self.getCholeskyCovariance()
		Theta =   self.getTheta()
		z = np.array([0,0,100])
		func = [(i.T.dot(L.dot(z))) for i in Theta.T]
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
		fig = plt.figure()
		T = self.getTemperatures()
		Kappa_o = self.getNominalCurve()
		Kappa_max = self.getZetaUnsrtFunc()
		UQ_limit = self.getUncertFunc()
		Z_a = self.getPerturbed_A_curve()
		Z_n = self.getPerturbed_n_curve()
		Z_e = self.getPerturbed_Ea_curve()
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
		plt.savefig(location+f"/{self.rxn}.pdf",bbox_inches="tight")		

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
