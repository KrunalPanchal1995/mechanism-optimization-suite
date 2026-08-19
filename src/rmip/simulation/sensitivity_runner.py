"""
rmip/simulation/sensitivity_runner.py
======================================
Cantera-native sensitivity analysis runner (Phase 5A).

Wraps Cantera's built-in SA to produce the same ``sensitivity_data`` dict
format as KP's Chemkin-Pro wrapper (CLAUDE.md §10), so the output can be
passed directly to ``SensitivityFluxIntegrator.attach()`` or saved as JSON
for the ``rmip visualize --sa-json`` CLI flag.

Output format (CLAUDE.md §10)::

    {
        "observable"  : "IDT",
        "conditions"  : {"T_K": 1100, "P_atm": 10, "phi": 1.0},
        "coefficients": {
            "H + O2 <=> OH + O" : 0.85,
            ...
        }
    }

The formula strings in ``coefficients`` match the reaction ``equation``
property returned by Cantera (which may differ slightly from the
``formula_string`` stored in the RMIP DB for the same reaction).
KP should verify that the formula strings match before running
``rmip sensitivity`` or ``rmip visualize --sa-json``.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("rmip.simulation.sensitivity_runner")

_IDT_THRESHOLD = 0.1   # temperature rise fraction to call ignition (10 % of ΔT_ad)
_MAX_TIME      = 0.1   # seconds — safety cap for integration


class CanteraSensitivityRunner:
    """
    Cantera-native sensitivity runner.

    Parameters
    ----------
    rtol, atol : float
        ODE solver tolerances.
    max_steps : int
        Maximum integration steps.
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-12,
        max_steps: int = 100_000,
    ) -> None:
        self._rtol      = rtol
        self._atol      = atol
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_idt(
        self,
        mechanism_path: str,
        T_K:     float,
        P_atm:   float,
        phi:     float,
        fuel:    str,
        oxidizer: str = "O2",
        n_top:   Optional[int] = None,
    ) -> Dict:
        """
        Run IDT sensitivity analysis using Cantera's IdealGasReactor SA.

        Parameters
        ----------
        mechanism_path : str
            Path to CHEMKIN .inp (requires ck2yaml conversion first) or
            Cantera .yaml / .cti mechanism file.
        T_K : float
            Initial temperature [K].
        P_atm : float
            Pressure [atm].
        phi : float
            Equivalence ratio.
        fuel : str
            Fuel species name as used in the mechanism (e.g. ``"H2"``).
        oxidizer : str
            Oxidizer species name (default ``"O2"``).
        n_top : int, optional
            If given, keep only the top-N reactions by |coefficient|.
            Default: keep all.

        Returns
        -------
        dict
            ``sensitivity_data`` in CLAUDE.md §10 format.
        """
        try:
            import cantera as ct
        except ImportError:
            raise ImportError("CanteraSensitivityRunner requires Cantera: pip install cantera")

        gas = ct.Solution(mechanism_path)

        # Set mixture composition
        phi_str = f"{fuel}:1"
        # Stoichiometric O2 moles for CH_n fuels: rough estimate via formula
        # For hydrogen: H2 + 0.5 O2; for general: let Cantera handle via phi
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        gas.TP = T_K, P_atm * ct.one_atm

        # Build constant-volume adiabatic reactor with sensitivity analysis
        r   = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([r])
        sim.rtol_sensitivity = self._rtol
        sim.atol_sensitivity = self._atol

        for i in range(gas.n_reactions):
            r.add_sensitivity_reaction(i)

        # ── Integrate until ignition ──
        T_init = gas.T
        T_ad   = self._estimate_T_ad(gas, fuel, oxidizer, phi)
        T_ign  = T_init + _IDT_THRESHOLD * (T_ad - T_init)

        t         = 0.0
        dt        = 1e-7
        idt       = None
        step      = 0
        max_steps = self._max_steps

        while t < _MAX_TIME and step < max_steps:
            t = sim.step()
            step += 1
            if r.thermo.T >= T_ign:
                idt = t
                break

        if idt is None:
            log.warning(
                "IDT not detected within %.2f s at T=%.0f K, P=%.1f atm, phi=%.2f",
                _MAX_TIME, T_K, P_atm, phi,
            )

        # ── Collect sensitivity coefficients ──
        coefficients: Dict[str, float] = {}
        for i, rxn in enumerate(gas.reactions()):
            s = sim.sensitivity(r.T, i)
            if math.isfinite(s):
                coefficients[rxn.equation] = s

        if n_top is not None:
            top = sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)
            coefficients = dict(top[:n_top])

        log.info(
            "Cantera SA: %d reaction coefficients at T=%.0f K, P=%.1f atm, phi=%.2f "
            "(IDT=%.3e s)",
            len(coefficients), T_K, P_atm, phi, idt or float("nan"),
        )

        return {
            "observable":   "IDT",
            "conditions":   {"T_K": T_K, "P_atm": P_atm, "phi": phi},
            "coefficients": coefficients,
        }

    def save(self, sensitivity_data: Dict, output_path: str) -> None:
        """Write sensitivity_data dict to a JSON file (CLAUDE.md §10 format)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(sensitivity_data, fh, indent=2)
        log.info("Sensitivity data written to %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_T_ad(gas, fuel: str, oxidizer: str, phi: float) -> float:
        """Estimate adiabatic flame temperature for ignition threshold."""
        try:
            import cantera as ct
            g2 = gas.__class__(gas.source)
            g2.set_equivalence_ratio(phi, fuel, oxidizer)
            g2.TP = gas.T, gas.P
            g2.equilibrate("HP")
            return g2.T
        except Exception:
            return gas.T + 1500.0   # rough fallback
