"""
rmip/simulation/__init__.py
============================
Simulation sub-package: Cantera-based IDT and LFS runners.

Both runners share the same ``MechanismLoader`` infrastructure from
``simulation.base`` which handles CHEMKIN → YAML conversion and caching.

Usage::

    from rmip.simulation import IDTRunner, LFSRunner

    runner = IDTRunner(mechanism_id=1, db_path="project.db")
    result = runner.run_single(T_K=1200, P_atm=13.5, phi=1.0,
                               fuel="NC7H16:1.0")
    df = runner.run_batch(conditions_list, n_workers=4)
"""
from rmip.simulation.base       import AbstractSimRunner, MechanismLoader
from rmip.simulation.idt_runner import IDTRunner
from rmip.simulation.lfs_runner import LFSRunner
from rmip.simulation.muqsac_bridge import MuqsacBridge
from rmip.simulation.jsr_runner import JSRRunner
from rmip.simulation.pfr_runner import PFRRunner

__all__ = [
    "AbstractSimRunner",
    "MechanismLoader",
    "IDTRunner",
    "LFSRunner",
    "MuqsacBridge",
    "JSRRunner",
    "PFRRunner",
]
