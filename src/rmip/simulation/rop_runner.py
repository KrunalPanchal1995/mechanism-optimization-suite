"""
rmip/simulation/rop_runner.py
=============================
Module: First-class Rate-of-Production / reaction-path-flux diagnostic.

Physics
-------
A **native, in-process Cantera** diagnostic evaluated at one (T, P, phi)
condition.  Unlike the two pre-existing "flux" things in the codebase —

  * ``flux_score`` (a *normalized graph node attribute* in
    ``analysis/sensitivity.py``, fed from an external Chemkin-Pro wrapper), and
  * the ``flux`` CLI command, which shells out to the **MUQ-SAC subprocess**
    pipeline (``simulation/muqsac_bridge.py``),

this reads Cantera's raw ``net_rates_of_progress`` directly, so it is cheap and
gives a genuine, condition-specific rate-of-production plus an atom-mapped
element-flux (reaction-path / Sankey) view.

Two products per call, packaged in a :class:`~rmip.types.FluxResult`:
  * ``rop_by_reaction`` : {DB canonical_key (or Cantera equation) -> signed net
    rate of progress [kmol/m3/s]}.
  * ``element_flux``    : list of Sankey edges ``{"source", "target",
    "reaction", "value"}`` giving the flow of one element (C/H/O) from reactant
    to product species, apportioned across each reaction's rate of progress by
    the number of element atoms carried.

Evaluation state (``at``)
-------------------------
- ``"ignition"``    : integrate a constant-volume adiabatic batch reactor
                      (same build as ``idt_runner.py``) to the max-dT/dt point,
                      then read rates there.  This is where the chemistry is
                      most active and is the most useful default.
- ``"time:<s>"``    : advance to a specific time [s].
- ``"initial"``     : read rates on the unburned mixture (t=0).
- ``"equilibrium"`` : ``equilibrate("UV")`` then read rates (near-zero net ROP;
                      mostly a sanity endpoint).

Usage (Python API)
------------------
>>> runner = ROPRunner(mechanism_id=1, db_path="project.db")
>>> res = runner.run_single(T_K=1200, P_atm=13.5, phi=1.0,
...                         fuel="NC7H16:1.0", element="C", at="ignition")
>>> res.rop_by_reaction        # {canonical_key: net_rop}
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

import numpy as np

from rmip.simulation.base import AbstractSimRunner
from rmip.types import FluxResult, SimCondition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure, Cantera-only helpers (no DB) — reusable and unit-testable in isolation
# ---------------------------------------------------------------------------

def _stoich_matrices(gas):
    """
    Return (reactant, product) stoichiometric-coefficient matrices as dense
    numpy arrays of shape (n_species, n_reactions).

    Cantera renamed these properties across the 2.6 -> 3.0 transition
    (``*_stoich_coeffs3`` on 2.6, back to ``*_stoich_coeffs`` on 3.0) and 3.x
    may return a SciPy sparse matrix — normalise both away here.
    """
    def _get(kind: str):
        if hasattr(gas, f"{kind}_stoich_coeffs"):
            m = getattr(gas, f"{kind}_stoich_coeffs")
        else:  # Cantera 2.6 transitional name
            m = getattr(gas, f"{kind}_stoich_coeffs3")
        if hasattr(m, "toarray"):      # scipy sparse
            m = m.toarray()
        return np.asarray(m, dtype=float)

    return _get("reactant"), _get("product")


def build_rop_by_reaction(
    gas,
    net_rates: np.ndarray,
    formula_to_key: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Map Cantera reaction index -> signed net rate of progress, keyed by the DB
    ``canonical_key`` when ``formula_to_key`` resolves the Cantera equation,
    else by the raw Cantera equation string.

    When several Cantera reactions share the same identity key (DUPLICATE
    reactions, or distinct reactions that canonicalise together), their net
    rates of progress are **summed** — the total ROP attributed to that reaction
    identity — rather than overwriting.
    """
    rop: Dict[str, float] = {}
    for i, rate in enumerate(net_rates):
        eq = gas.reaction(i).equation
        key = formula_to_key.get(eq, eq) if formula_to_key else eq
        rop[key] = rop.get(key, 0.0) + float(rate)
    return rop


def build_element_flux(
    gas,
    net_rates: np.ndarray,
    element: str,
    formula_to_key: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build atom-mapped element-flux Sankey edges for one element.

    For reaction *r* with signed net rate of progress ``q_r``, the flow of
    ``element`` from a source species *i* to a target species *j* is

        flux(i->j) = |q_r| * (a_i * a_j) / sum_k a_k

    where ``a_s`` is the number of ``element`` atoms species *s* contributes on
    the *source* side of the (sign-oriented) reaction.  When ``q_r > 0`` the
    source side is the reactants and the target side is the products; when
    ``q_r < 0`` they swap.  This is the standard proportional atom-apportioning
    used by reaction-path analysis and conserves the element by construction.
    Species carrying zero ``element`` atoms produce no edges (so ``element="C"``
    yields only the carbon skeleton, etc.).
    """
    edges: List[Dict[str, Any]] = []
    if element not in gas.element_names:
        return edges  # phase has no such element -> no flux to trace

    species = gas.species_names
    n_atoms = np.array(
        [gas.n_atoms(s, element) for s in species], dtype=float
    )
    if not n_atoms.any():
        return edges

    rsc, psc = _stoich_matrices(gas)  # (n_species, n_reactions)

    for i, q in enumerate(net_rates):
        if q == 0.0:
            continue
        r_atoms = rsc[:, i] * n_atoms
        p_atoms = psc[:, i] * n_atoms
        if q > 0.0:
            src_atoms, tgt_atoms, val = r_atoms, p_atoms, float(q)
        else:
            src_atoms, tgt_atoms, val = p_atoms, r_atoms, float(-q)

        total = src_atoms.sum()
        if total <= 0.0:
            continue

        eq = gas.reaction(i).equation
        rxn_key = formula_to_key.get(eq, eq) if formula_to_key else eq
        src_idx = np.nonzero(src_atoms)[0]
        tgt_idx = np.nonzero(tgt_atoms)[0]
        for si in src_idx:
            for ti in tgt_idx:
                if si == ti:
                    continue  # self-edge carries no net element transfer
                flux = val * (src_atoms[si] * tgt_atoms[ti]) / total
                edges.append({
                    "source":   species[si],
                    "target":   species[ti],
                    "reaction": rxn_key,
                    "value":    float(flux),
                })
    return edges


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_VALID_AT = ("ignition", "initial", "equilibrium")  # plus "time:<s>"


class ROPRunner(AbstractSimRunner):
    """
    Native Cantera rate-of-production / reaction-path-flux runner for one
    mechanism.  Reuses :class:`~rmip.simulation.base.MechanismLoader` (via the
    lazy ``gas`` property) and the ``_fresh_gas`` copy pattern.

    Parameters
    ----------
    mechanism_id : int
        DB id of the mechanism to analyse.
    db_path : str
        Path to the RMIP SQLite database.
    t_end_s / n_steps / rtol / atol :
        Reactor-network controls for the ``at="ignition"`` / ``at="time:..."``
        integration (mirrors ``IDTRunner`` defaults).
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path:      str,
        t_end_s:      float = 10.0,
        n_steps:      int   = 2000,
        rtol:         float = 1e-9,
        atol:         float = 1e-15,
        cache_dir:    Optional[str] = None,
    ) -> None:
        super().__init__(mechanism_id, db_path, cache_dir)
        self._t_end_s = t_end_s
        self._n_steps = n_steps
        self._rtol    = rtol
        self._atol    = atol
        self._formula_to_key_cache: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        T_K:      float,
        P_atm:    float,
        phi:      float,
        fuel:     str,
        oxidizer: str = "O2:1.0, N2:3.76",
        element:  str = "C",
        at:       str = "ignition",
    ) -> FluxResult:
        """Run one ROP diagnostic and return a :class:`FluxResult`."""
        cond = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi,
                            fuel=fuel, oxidizer=oxidizer)
        t0 = time.perf_counter()
        try:
            gas = self._fresh_gas()
            self._stage(gas, T_K, P_atm, phi, fuel, oxidizer, at)

            net_rates = gas.net_rates_of_progress
            f2k = self._formula_to_key()
            rop   = build_rop_by_reaction(gas, net_rates, f2k)
            edges = build_element_flux(gas, net_rates, element, f2k)
            cpu = time.perf_counter() - t0

            logger.debug("ROP: %d reactions, %d %s-flux edges (at=%s)",
                         len(rop), len(edges), element, at)
            return FluxResult(
                mechanism_id=self.mechanism_id, condition=cond,
                solver="cantera", element=element,
                rop_by_reaction=rop, element_flux=edges,
                converged=True, cpu_time_s=cpu,
            )
        except Exception as exc:
            cpu = time.perf_counter() - t0
            logger.warning("ROP diagnostic failed: T=%g K — %s", T_K, exc)
            self.reset_gas()
            return FluxResult(
                mechanism_id=self.mechanism_id, condition=cond,
                solver="cantera", element=element,
                rop_by_reaction={}, element_flux=[],
                converged=False, cpu_time_s=cpu, error_msg=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_gas(self):
        """Return a fresh gas copy with reset state (mirrors IDTRunner)."""
        import cantera as ct
        return ct.Solution(
            thermo=self.gas.thermo_model,
            kinetics=self.gas.kinetics_model,
            transport=None,
            species=self.gas.species(),
            reactions=self.gas.reactions(),
        )

    def _stage(self, gas, T_K, P_atm, phi, fuel, oxidizer, at: str) -> None:
        """Set composition/state on ``gas`` and advance to the eval state."""
        import cantera as ct

        gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
        gas.TP = T_K, P_atm * ct.one_atm

        if at == "initial":
            return
        if at == "equilibrium":
            gas.equilibrate("UV")
            return

        if at == "ignition":
            t_target = self._ignition_time(T_K, P_atm, phi, fuel, oxidizer)
        elif at.startswith("time:"):
            try:
                t_target = float(at.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid time spec '{at}' (want 'time:<seconds>')") from exc
        else:
            raise ValueError(
                f"Unknown evaluation state '{at}'. "
                f"Expected one of {_VALID_AT} or 'time:<seconds>'."
            )

        # IdealGasReactor shares `gas` as its working fluid, so after advancing
        # the network the `gas` object reflects the reactor state at t_target.
        r   = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([r])
        net.rtol = self._rtol
        net.atol = self._atol
        net.advance(t_target)

    def _ignition_time(self, T_K, P_atm, phi, fuel, oxidizer) -> float:
        """
        Find the time of maximum dT/dt in a constant-volume adiabatic batch
        reactor on a throwaway gas copy (so the caller's `gas` is untouched).
        Falls back to the T_init + 400 K crossing, then to t_end.
        """
        import cantera as ct

        g = self._fresh_gas()
        g.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
        g.TP = T_K, P_atm * ct.one_atm
        r   = ct.IdealGasReactor(g)
        net = ct.ReactorNet([r])
        net.rtol = self._rtol
        net.atol = self._atol

        t_arr = [0.0]
        T_arr = [r.T]
        dt_fine   = self._t_end_s * 0.001 / max(self._n_steps // 4, 1)
        dt_coarse = self._t_end_s / self._n_steps
        t_switch  = self._t_end_s * 0.001
        t = 0.0
        while t < self._t_end_s:
            dt = dt_fine if t < t_switch else dt_coarse
            t  = min(t + dt, self._t_end_s)
            net.advance(t)
            t_arr.append(t)
            T_arr.append(r.T)
            if T_arr[-1] - T_arr[0] > 700:
                break

        times = np.asarray(t_arr)
        temps = np.asarray(T_arr)
        if len(times) >= 3:
            dTdt = np.gradient(temps, times)
            peak = int(np.argmax(dTdt))
            if temps[peak] - temps[0] > 100 and dTdt[peak] > 0:
                return float(times[peak])
            idx = np.where(temps >= T_K + 400.0)[0]
            if len(idx) > 0:
                return float(times[idx[0]])
        return float(times[-1])

    def _formula_to_key(self) -> Dict[str, str]:
        """
        Build {Cantera equation / DB formula_string -> canonical_key} for this
        mechanism, so ROP is keyed by the stable ``canonical_key`` used across
        the suite.  Falls back to the formula string itself when a reaction has
        no canonical_key.  Cached after first build; empty dict if the DB has no
        reaction rows (e.g. a bare-Cantera path), in which case callers key by
        the Cantera equation string.
        """
        if self._formula_to_key_cache is not None:
            return self._formula_to_key_cache
        mapping: Dict[str, str] = {}
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT formula_string, canonical_key FROM reactions "
                    "WHERE mechanism_id=?",
                    (self.mechanism_id,),
                ).fetchall()
            finally:
                conn.close()
            for r in rows:
                fs = r["formula_string"]
                key = r["canonical_key"] or fs
                # M6 follow-on: formula_string collisions used to be silent
                # (last write wins) -- this is exactly the failure mode that
                # hid the generic-M falloff collider-tag bug (see
                # _chemkin_parser.py::_build_formula_string) for months: two
                # physically distinct reactions sharing one formula_string
                # meant ROP results for one silently overwrote the other's
                # bucket. Loud now so any future formula_string regression
                # surfaces immediately instead of silently mis-keying ROP.
                if fs in mapping and mapping[fs] != key:
                    logger.warning(
                        "formula_string collision in mechanism_id=%s: "
                        "%r maps to both canonical_key=%r and %r -- ROP "
                        "results for these reactions will be merged into "
                        "one bucket. Likely an unrelated-reaction "
                        "formula_string collision (see M6 in bug.md).",
                        self.mechanism_id, fs, mapping[fs], key,
                    )
                mapping[fs] = key
        except sqlite3.Error as exc:
            logger.debug("Could not load reaction canonical_keys (%s); "
                         "keying ROP by Cantera equation string", exc)
        self._formula_to_key_cache = mapping
        return mapping
