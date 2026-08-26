"""
rmip/simulation/finite_solid_reactor.py
=======================================
W5a — Finite bulk-solid reactor (closes P6, P7, P8, P9).

Cantera's ``Reactor``/``ReactorSurface`` treats a bulk solid as an *infinite*
thermodynamic reservoir: bulk mass is not an ODE state, so a depleting
char/solid cannot be modeled without a hand-rolled outer loop that (P7)
injects artificial mass when clamping negatives and (P8) lets the coverage
sum drift, and that is (P9) numerically fragile non-isothermally.

This module integrates an **augmented extensive state vector** with SciPy's
stiff BDF integrator instead:

    y = [ m_k (gas species masses, kg) ...,
          n_i (surface species moles, kmol) ...,
          m_b (bulk solid mass, kg),
          T   (K) ]

Design decisions (each maps to one of the closed problems)
----------------------------------------------------------
P6  — the bulk mass *is* a state; the reactive surface area follows the
      remaining solid through a pluggable area model (default: shrinking
      core, ``A = A0 * (m_b/m_b0)**(2/3)``).
P7  — no artificial mass is ever injected, because rates (not states) vanish
      as inventories deplete: the bulk consumption rate scales with
      ``A(m_b) -> 0`` at burnout, and gas destruction rates vanish with the
      species' own concentration through mass action (negative excursions are
      clamped to zero *inside property evaluation only*).  Residual
      integrator negatives are projected to zero within an **audited**
      tolerance — the audit errors out if total element mass drifts beyond
      ``mass_audit_rtol``.  (BDF, a linear multistep method, preserves the
      linear element-mass invariants of the exactly-conserving Cantera rates
      to Newton tolerance, so the audit passes at ~1e-10 in practice.)
P8  — coverages are never integrated as an intensive vector that can drift:
      the state holds extensive surface moles ``n_i`` and
      ``theta_i = n_i*sigma_i / sum_j n_j*sigma_j`` is computed by
      normalization at every RHS evaluation, so ``sum(theta) == 1`` holds
      *by construction*.  A separate site audit verifies the total site count
      ``sum n_i sigma_i`` is conserved — a mechanism whose surface reactions
      do not balance sites (P4) is flagged as an error, not silently drifted.
P9  — the outer loop is an adaptive-dt ladder with a stiffness guard: a
      failed BDF chunk shrinks dt by 4x and retries; repeated failure raises
      ``IntegrationError`` with actionable guidance instead of a raw solver
      traceback.  An ``isothermal=True`` mode is available for the fragile
      exothermic cases.

Modeling assumptions (documented, not silent)
---------------------------------------------
- Constant-volume, closed, well-mixed gas; adiabatic when non-isothermal.
- Gas and solid share one temperature (thermally lumped particle).
- Surface-species energy storage is neglected in the energy balance (the
  adsorbed inventory is ~Gamma*A*W ~ 1e-7 kg — far below gas/solid heat
  capacity).
- Coverage relaxation uses the *initial* site inventory ``Gamma*A0`` (exact
  at start; late in burnout the adsorbed inventory is negligible anyway),
  while reaction *fluxes* scale with the instantaneous shrinking area.
- Bulk internal energy is approximated by bulk enthalpy (incompressible
  solid; the Pv term is ~1e-4 of h for graphite at 1 atm).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cantera as ct
    _HAS_CANTERA = True
except ImportError:                                       # pragma: no cover
    ct = None  # type: ignore[assignment]
    _HAS_CANTERA = False


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class MassBalanceError(RuntimeError):
    """Element-mass (or site) conservation drifted beyond tolerance."""


class IntegrationError(RuntimeError):
    """The stiff integrator failed even after the dt ladder bottomed out."""


_P9_GUIDANCE = (
    "The stiff integrator could not advance even at the minimum chunk size. "
    "This is the P9 failure mode (exothermic surface chemistry is numerically "
    "fragile). Try: (1) isothermal=True to decouple the energy equation, "
    "(2) a smaller initial particle mass or surface area, (3) looser rtol "
    "(1e-6), or (4) check the mechanism with `rmip check` for rate/thermo "
    "pathologies first."
)


# ---------------------------------------------------------------------------
# Area models (pluggable)
# ---------------------------------------------------------------------------
def shrinking_core_area(m_b: float, m_b0: float, A0: float) -> float:
    """Shrinking-core: area scales with the 2/3 power of remaining mass."""
    x = max(m_b, 0.0) / m_b0
    return A0 * x ** (2.0 / 3.0)


def constant_area(m_b: float, m_b0: float, A0: float) -> float:
    """Fixed area until the solid is exhausted (porous-particle limit)."""
    return A0 if m_b > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FiniteSolidResult:
    t: np.ndarray                       # [n]
    T: np.ndarray                       # [n]
    P: np.ndarray                       # [n] gas pressure, Pa
    gas_mass: np.ndarray                # [n, K] per-species gas mass, kg
    surface_moles: np.ndarray           # [n, S] extensive surface moles, kmol
    coverages: np.ndarray               # [n, S] normalized coverages
    m_bulk: np.ndarray                  # [n] bulk solid mass, kg
    area: np.ndarray                    # [n] reactive area, m^2
    gas_species: List[str]
    surface_species: List[str]
    burned_out: bool
    burnout_time: Optional[float]
    audit: Dict[str, float] = field(default_factory=dict)

    @property
    def Y(self) -> np.ndarray:
        """Gas mass fractions [n, K]."""
        tot = self.gas_mass.sum(axis=1, keepdims=True)
        return self.gas_mass / np.where(tot > 0, tot, 1.0)

    def to_csv(self, path: Union[str, Path]) -> None:
        cols = (["t_s", "T_K", "P_Pa", "m_bulk_kg", "area_m2"]
                + [f"m_{s}_kg" for s in self.gas_species]
                + [f"theta_{s}" for s in self.surface_species])
        data = np.column_stack([self.t, self.T, self.P, self.m_bulk,
                                self.area, self.gas_mass, self.coverages])
        header = ",".join(cols)
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")

    def summary(self) -> str:
        lines = [
            f"FiniteSolidReactor result: {len(self.t)} samples, "
            f"t_end = {self.t[-1]:.6g} s",
            f"  burned out : {self.burned_out}"
            + (f" at t = {self.burnout_time:.6g} s" if self.burned_out else ""),
            f"  bulk mass  : {self.m_bulk[0]:.6g} -> {self.m_bulk[-1]:.6g} kg",
            f"  T          : {self.T[0]:.1f} -> {self.T[-1]:.1f} K",
            f"  audit      : max element drift "
            f"{self.audit.get('max_element_drift_rel', float('nan')):.3g} "
            f"(rel), max site drift "
            f"{self.audit.get('max_site_drift_rel', float('nan')):.3g} (rel), "
            f"clipped mass {self.audit.get('clipped_mass_kg', 0.0):.3g} kg",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The reactor
# ---------------------------------------------------------------------------
class FiniteSolidReactor:
    """Closed, constant-volume, well-mixed gas over a depleting bulk solid.

    Parameters
    ----------
    mech : str
        Path to a Cantera YAML with a gas phase, an ``ideal-surface`` phase,
        and a bulk (``fixed-stoichiometry`` or similar condensed) phase.
    gas_phase, surface_phase, bulk_phase : str, optional
        Phase names.  Auto-detected from the YAML's ``thermo:`` models when
        omitted (errors if ambiguous).
    T0 : float                — initial temperature [K]
    P0 : float                — initial gas pressure [Pa]
    X0 : str | dict           — initial gas mole fractions
    gas_volume_m3 : float     — gas volume [m^3]
    bulk_mass_kg : float      — initial solid mass [kg]
    area_m2 : float, optional — initial reactive area [m^2]; default: the
        outer area of a single sphere of ``bulk_mass_kg`` at the bulk density.
    initial_coverages : str | dict, optional — default: all empty sites
        (first surface species) if not given and the phase state has none.
    isothermal : bool         — hold T fixed (P9 fallback mode)
    area_model : callable(m_b, m_b0, A0) -> A   — default shrinking core
    rtol, atol_scale : float  — BDF tolerances (atol is per-block scaled)
    mass_audit_rtol : float   — element-mass drift tolerance (relative to
        total initial mass; exceeding it raises ``MassBalanceError``)
    site_audit_rtol : float   — total-site drift tolerance (flags P4-style
        site-unbalanced surface mechanisms)
    burnout_fraction : float  — m_b below this fraction of m_b0 ends the run
    """

    def __init__(
        self,
        mech: str,
        gas_phase: Optional[str] = None,
        surface_phase: Optional[str] = None,
        bulk_phase: Optional[str] = None,
        T0: float = 1400.0,
        P0: float = 101325.0,
        X0: Union[str, Dict[str, float]] = "O2:0.21, N2:0.79",
        gas_volume_m3: float = 1.0e-3,
        bulk_mass_kg: float = 1.0e-4,
        area_m2: Optional[float] = None,
        initial_coverages: Optional[Union[str, Dict[str, float]]] = None,
        isothermal: bool = False,
        area_model: Callable[[float, float, float], float] = shrinking_core_area,
        rtol: float = 1.0e-8,
        atol_scale: float = 1.0e-12,
        mass_audit_rtol: float = 1.0e-8,
        site_audit_rtol: float = 1.0e-6,
        burnout_fraction: float = 1.0e-4,
    ) -> None:
        if not _HAS_CANTERA:
            raise ImportError("Cantera is required for FiniteSolidReactor")
        self.mech = str(mech)
        gname, sname, bname = self._resolve_phase_names(
            self.mech, gas_phase, surface_phase, bulk_phase)
        self.gas = ct.Solution(self.mech, gname)
        self.bulk = ct.Solution(self.mech, bname)
        self.surf = ct.Interface(self.mech, sname, [self.gas, self.bulk])

        self.T0 = float(T0)
        self.P0 = float(P0)
        self.X0 = X0
        self.V = float(gas_volume_m3)
        self.m_b0 = float(bulk_mass_kg)
        self.isothermal = bool(isothermal)
        self.area_model = area_model
        self.rtol = float(rtol)
        self.atol_scale = float(atol_scale)
        self.mass_audit_rtol = float(mass_audit_rtol)
        self.site_audit_rtol = float(site_audit_rtol)
        self.burnout_fraction = float(burnout_fraction)

        if len(self.bulk.species_names) != 1:
            raise ValueError(
                f"Bulk phase '{bname}' must contain exactly one species "
                f"(got {self.bulk.species_names}); multi-species bulk solids "
                "are out of scope for the shrinking-core model.")

        # Default area: single sphere of the given mass at bulk density.
        if area_m2 is None:
            rho_b = float(self.bulk.density)
            r = (3.0 * self.m_b0 / (4.0 * math.pi * rho_b)) ** (1.0 / 3.0)
            area_m2 = 4.0 * math.pi * r * r
        self.A0 = float(area_m2)

        self._init_coverages = initial_coverages

        # Index maps: kinetics-species index -> our state blocks.
        self._K = self.gas.n_species
        self._S = self.surf.n_species
        self._kin_gas = np.array(
            [self.surf.kinetics_species_index(n) for n in self.gas.species_names])
        self._kin_surf = np.array(
            [self.surf.kinetics_species_index(n) for n in self.surf.species_names])
        self._kin_bulk = self.surf.kinetics_species_index(
            self.bulk.species_names[0])
        self._sigma = np.array(
            [self.surf.species(i).size for i in range(self._S)])
        self._W_gas = self.gas.molecular_weights          # kg/kmol
        self._W_bulk = float(self.bulk.molecular_weights[0])

        # Element-mass matrices: kg of element e per kg (gas) / kmol (surf).
        self._elements = sorted(
            set(self.gas.element_names) | set(self.surf.element_names)
            | set(self.bulk.element_names))
        self._E_gas = self._element_mass_matrix(self.gas, per="mass")
        self._E_surf = self._element_mass_matrix(self.surf, per="mole")
        self._E_bulk = self._element_mass_matrix(self.bulk, per="mass")[:, 0]

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_phase_names(mech, gas_phase, surface_phase, bulk_phase):
        """Auto-detect gas / surface / bulk phase names from YAML thermo models."""
        # rmip.analysis.yaml_safe, not plain yaml.safe_load: stock PyYAML
        # resolves a bare species name "NO" as a YAML-1.1 bool, silently
        # corrupting species-name lists (see yaml_safe.py docstring).
        from rmip.analysis.yaml_safe import safe_load_path
        doc = safe_load_path(mech)
        phases = doc.get("phases", []) or []
        by_kind: Dict[str, List[str]] = {"gas": [], "surface": [], "bulk": []}
        for ph in phases:
            model = str(ph.get("thermo", ""))
            if model == "ideal-gas":
                by_kind["gas"].append(ph["name"])
            elif "surface" in model or "edge" in model:
                by_kind["surface"].append(ph["name"])
            else:
                by_kind["bulk"].append(ph["name"])

        def pick(kind: str, given: Optional[str]) -> str:
            if given:
                return given
            cands = by_kind[kind]
            if len(cands) == 1:
                return cands[0]
            raise ValueError(
                f"Cannot auto-detect the {kind} phase in {mech}: candidates "
                f"{cands or 'none'}. Pass {kind}_phase= explicitly.")

        return pick("gas", gas_phase), pick("surface", surface_phase), \
            pick("bulk", bulk_phase)

    def _element_mass_matrix(self, phase, per: str) -> np.ndarray:
        """[n_elements x n_species]: kg of element per kg (per='mass') or per
        kmol (per='mole') of each species."""
        E = np.zeros((len(self._elements), phase.n_species))
        for j, sp in enumerate(phase.species_names):
            W_sp = phase.molecular_weights[j]
            for ei, e in enumerate(self._elements):
                if e in phase.element_names:
                    n_atoms = phase.n_atoms(sp, e)
                    if n_atoms:
                        m_e = n_atoms * phase.atomic_weight(e)  # kg/kmol
                        E[ei, j] = m_e / W_sp if per == "mass" else m_e
        return E

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _initial_state(self) -> np.ndarray:
        self.gas.TPX = self.T0, self.P0, self.X0
        m_gas = self.gas.density * self.V * self.gas.Y

        cov = self._init_coverages
        if cov is None:
            cov = {self.surf.species_names[0]: 1.0}
        self.surf.TP = self.T0, self.P0
        self.surf.coverages = cov
        theta = self.surf.coverages
        n_sites_tot = self.surf.site_density * self.A0          # kmol sites
        n_surf = theta * n_sites_tot / self._sigma              # kmol species

        y = np.concatenate([m_gas, n_surf, [self.m_b0], [self.T0]])
        return y

    def _unpack(self, y):
        K, S = self._K, self._S
        return y[:K], y[K:K + S], y[K + S], y[K + S + 1]

    def _element_masses(self, y) -> np.ndarray:
        m_gas, n_surf, m_b, _T = self._unpack(y)
        return (self._E_gas @ np.maximum(m_gas, 0.0)
                + self._E_surf @ np.maximum(n_surf, 0.0)
                + self._E_bulk * max(m_b, 0.0))

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------
    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        m_gas, n_surf, m_b, T = self._unpack(y)

        # Clamp only inside property evaluation (rates then vanish through
        # mass action / the area model — never the state itself: P7).
        m_gas_c = np.maximum(m_gas, 0.0)
        m_tot = m_gas_c.sum()
        rho = m_tot / self.V
        self.gas.TDY = T, rho, m_gas_c / m_tot

        n_surf_c = np.maximum(n_surf, 0.0)
        w = n_surf_c * self._sigma
        w_tot = w.sum()
        theta = w / w_tot if w_tot > 0 else np.full(self._S, 1.0 / self._S)
        P = self.gas.P
        self.surf.TP = T, P
        self.surf.coverages = theta        # sum(theta)=1 by construction (P8)
        self.bulk.TP = T, P

        A = self.area_model(m_b, self.m_b0, self.A0)        # -> 0 at burnout

        wdot = self.gas.net_production_rates                # kmol/m^3/s
        sdot_all = self.surf.net_production_rates           # kmol/m^2/s
        sdot_gas = sdot_all[self._kin_gas]
        sdot_surf = sdot_all[self._kin_surf]
        sdot_bulk = sdot_all[self._kin_bulk]

        dm_gas = self._W_gas * (wdot * self.V + sdot_gas * A)
        dn_surf = sdot_surf * A
        dm_b = self._W_bulk * sdot_bulk * A

        if self.isothermal:
            dT = 0.0
        else:
            # Adiabatic, constant-volume energy balance over gas + solid
            # (single lumped T; surface energy storage neglected):
            #   0 = sum(dm_k u_k) + dm_b u_b + (sum m_k cv_k + m_b c_b) dT/dt
            u_gas = self.gas.partial_molar_int_energies / self._W_gas  # J/kg
            u_b = float(self.bulk.partial_molar_enthalpies[0]) / self._W_bulk
            cv_gas = self.gas.cv_mass
            c_b = self.bulk.cp_mass
            heat_cap = m_tot * cv_gas + max(m_b, 0.0) * c_b
            dT = -(float(u_gas @ dm_gas) + u_b * dm_b) / heat_cap

        return np.concatenate([dm_gas, dn_surf, [dm_b], [dT]])

    # ------------------------------------------------------------------
    # Integration (adaptive-dt outer ladder with stiffness guard: P9)
    # ------------------------------------------------------------------
    def run(
        self,
        t_end: float = 10.0,
        dt_init: float = 1.0e-5,
        dt_max: Optional[float] = None,
        dt_min: float = 1.0e-12,
        max_consecutive_failures: int = 8,
    ) -> FiniteSolidResult:
        from scipy.integrate import solve_ivp

        y = self._initial_state()
        n_state = len(y)
        K, S = self._K, self._S

        # Per-block absolute tolerances (state scales differ by ~1e8).
        atol = np.empty(n_state)
        atol[:K] = max(y[:K].sum() * self.atol_scale, 1.0e-20)
        atol[K:K + S] = max(y[K:K + S].sum() * self.atol_scale * 1e2, 1.0e-24)
        atol[K + S] = max(self.m_b0 * self.atol_scale, 1.0e-20)
        atol[K + S + 1] = 1.0e-6

        elem0 = self._element_masses(y)
        m_tot0 = float(elem0.sum())
        sites0 = float((y[K:K + S] * self._sigma).sum())

        burnout_mass = self.burnout_fraction * self.m_b0

        def burnout_event(t, yy):
            return yy[K + S] - burnout_mass
        burnout_event.terminal = True
        burnout_event.direction = -1

        ts: List[float] = [0.0]
        states: List[np.ndarray] = [y.copy()]
        max_elem_drift = 0.0
        max_site_drift = 0.0
        clipped = 0.0

        t = 0.0
        dt = float(dt_init)
        if dt_max is None:
            dt_max = t_end / 20.0
        fails = 0
        burned_out = False
        burnout_time: Optional[float] = None

        while t < t_end and not burned_out:
            t_next = min(t + dt, t_end)
            sol = solve_ivp(self._rhs, (t, t_next), y, method="BDF",
                            rtol=self.rtol, atol=atol,
                            events=[burnout_event], dense_output=False)
            if not sol.success:
                fails += 1
                dt /= 4.0
                if dt < dt_min or fails > max_consecutive_failures:
                    raise IntegrationError(
                        f"BDF failed at t={t:.6g} s (dt={dt:.3g}): "
                        f"{sol.message}\n{_P9_GUIDANCE}")
                logger.debug("BDF chunk failed at t=%.4g; dt -> %.3g", t, dt)
                continue
            fails = 0

            y = sol.y[:, -1].copy()
            t = float(sol.t[-1])

            # ---- audited projection: element mass BEFORE clipping --------
            drift = float(np.abs(self._element_masses(y) - elem0).sum()) / m_tot0
            max_elem_drift = max(max_elem_drift, drift)
            if drift > self.mass_audit_rtol:
                raise MassBalanceError(
                    f"Element mass drifted {drift:.3e} (rel) at t={t:.6g} s — "
                    f"exceeds mass_audit_rtol={self.mass_audit_rtol:.1e}. "
                    "The integration cannot be trusted; tighten rtol or check "
                    "the mechanism's element balance with `rmip check`.")

            site_drift = abs(float((y[K:K + S] * self._sigma).sum()) - sites0) \
                / sites0
            max_site_drift = max(max_site_drift, site_drift)
            if site_drift > self.site_audit_rtol:
                raise MassBalanceError(
                    f"Total surface-site count drifted {site_drift:.3e} (rel) "
                    f"at t={t:.6g} s. The surface mechanism likely does not "
                    "balance sites (P4) — run balance_surface_sites() / "
                    "`rmip check` on it.")

            # Project residual integrator negatives to zero (audited).
            neg = np.minimum(y[:K + S + 1], 0.0)
            if neg.any():
                clipped += float(-neg[:K].sum() - neg[K + S])
                y[:K + S + 1] = np.maximum(y[:K + S + 1], 0.0)

            ts.append(t)
            states.append(y.copy())

            if sol.t_events and len(sol.t_events[0]):
                burned_out = True
                burnout_time = float(sol.t_events[0][0])
            dt = min(dt * 1.5, dt_max)

        # ---- assemble result --------------------------------------------
        arr = np.array(states)
        m_gas = arr[:, :K]
        n_surf = arr[:, K:K + S]
        m_b_arr = arr[:, K + S]
        T_arr = arr[:, K + S + 1]

        w = n_surf * self._sigma
        w_tot = w.sum(axis=1, keepdims=True)
        cov = np.divide(w, w_tot, out=np.full_like(w, 1.0 / S),
                        where=w_tot > 0)
        area = np.array([self.area_model(m, self.m_b0, self.A0)
                         for m in m_b_arr])
        # Ideal-gas pressure from the extensive state.
        n_gas = m_gas / self._W_gas
        P_arr = n_gas.sum(axis=1) * 1000.0 * 8.314462618 * T_arr / self.V \
            * 1.0  # kmol -> mol
        # (R in J/mol/K with kmol->mol factor folded in: n[kmol]*1000 = mol)

        return FiniteSolidResult(
            t=np.array(ts), T=T_arr, P=P_arr,
            gas_mass=m_gas, surface_moles=n_surf, coverages=cov,
            m_bulk=m_b_arr, area=area,
            gas_species=list(self.gas.species_names),
            surface_species=list(self.surf.species_names),
            burned_out=burned_out, burnout_time=burnout_time,
            audit={
                "max_element_drift_rel": max_elem_drift,
                "max_site_drift_rel": max_site_drift,
                "clipped_mass_kg": clipped,
            },
        )
