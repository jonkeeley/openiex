import numpy as np
from typing import Dict, Any, Optional
import warnings
from .system import ExchangeSystem

def initialize_state(
    initial_conditions: Dict[str, Dict[str, Any]],
    system: ExchangeSystem
) -> np.ndarray:
    """
    Build the initial state vector y0 for the solver, defining initial
    mobile phase concentrations (C) and bound (Q).

    Example: AEX column equilibrated with 2 mM MgCl2 - binding sites fully occupied with Cl-.
        initial_conditions = {
            "Cl-": {"C": 0.004, "Q": system.config.Lambda},
            "Mg2+": {"C": 0.002}
        }
    """
    Nz = system.config.Nz
    species_list = list(system.species.keys())

    for s in initial_conditions:
        if s not in species_list:
            raise KeyError(f"Unknown species '{s}' in initial_conditions")

    C_init = {s: np.zeros(Nz) for s in species_list}
    Q_init = {s: np.zeros(Nz) for s in species_list}

    for s, cond in initial_conditions.items():
        if "C" in cond:
            C_init[s][:] = cond["C"]
        if "Q" in cond:
            Q_init[s][:] = cond["Q"]

    return np.concatenate([
        C_init[s] for s in species_list
    ] + [
        Q_init[s] for s in species_list
    ])

def load_state(
    prev_result: Any,
    system: ExchangeSystem,
    t_start: Optional[float] = None
) -> np.ndarray:
    """
    Extract a packed y0 from prev_result at (or just before) t_start.

    - prev_result: the SimulationResult to extract state from
    - system:      new ExchangeSystem that must include all species from prev_result.system
    - t_start:     absolute time in prev_result.t to resume from; defaults to last time

    Returns the packed y0 ready for run_simulation.
    """
    from .solver import SimulationResult

    if not isinstance(prev_result, SimulationResult):
        raise TypeError("prev_result must be a SimulationResult")

    t_old, y_old, old_system, _ = (
        prev_result.t,
        prev_result.y,
        prev_result.system,
        prev_result.method
    )

    if t_start is None:
        t_start = t_old[-1]

    Nz_old = old_system.config.Nz
    Nz_new = system.config.Nz
    if Nz_old != Nz_new:
        raise ValueError(f"Nz mismatch: old {Nz_old}, new {Nz_new}")

    for name, old_sp in old_system.species.items():
        if name not in system.species:
            raise KeyError(f"Species '{name}' missing in new system")
        new_sp = system.species[name]
        if (old_sp.D, old_sp.K_d, old_sp.unit) != (new_sp.D, new_sp.K_d, new_sp.unit):
            warnings.warn(f"Species '{name}' parameters differ")
        if hasattr(old_sp, "sigma"):
            if (old_sp.sigma, old_sp.nu) != (new_sp.sigma, new_sp.nu):
                warnings.warn(f"Protein '{name}' sigma/nu differ")

    idx = np.searchsorted(t_old, t_start, side="right") - 1
    if idx < 0:
        raise ValueError(f"t_start={t_start} is before trajectory start")

    y_slice = y_old[:, idx]
    C_old, Q_old = unpack_state(y_slice, old_system)

    C_new: Dict[str, np.ndarray] = {}
    Q_new: Dict[str, np.ndarray] = {}
    for name in system.species:
        if name in C_old:
            C_new[name] = C_old[name].copy()
            Q_new[name] = Q_old[name].copy()
        else:
            C_new[name] = np.zeros(Nz_new)
            Q_new[name] = np.zeros(Nz_new)

    return pack_state(C_new, Q_new, system)

def unpack_state(y, system):
    Nz = system.config.Nz
    species_list = list(system.species.keys())
    num_species = len(species_list)
    expected_len = 2 * num_species * Nz

    if len(y) != expected_len:
        raise ValueError(
            f"State array has length {len(y)}, expected {expected_len} "
            f"({num_species} species x 2 (C + Q) x {Nz} segments)"
        )

    split = num_species * Nz
    C_arrs = y[:split]
    Q_arrs = y[split:]

    C = {s: C_arrs[i * Nz:(i + 1) * Nz] for i, s in enumerate(species_list)}
    Q = {s: Q_arrs[i * Nz:(i + 1) * Nz] for i, s in enumerate(species_list)}

    return C, Q

def pack_state(C, Q, system):
    species_list = list(system.species.keys())
    Nz = system.config.Nz

    # Check for missing or misaligned species
    for s in species_list:
        if s not in C.keys():
            raise KeyError(f"Species '{s}' missing from C.")
        if s not in Q.keys():
            raise KeyError(f"Species '{s}' missing from Q.")

        if len(C[s]) != Nz:
            raise ValueError(f"C[{s}] has length {len(C[s])}, expected {Nz}")
        if len(Q[s]) != Nz:
            raise ValueError(f"Q[{s}] has length {len(Q[s])}, expected {Nz}")

    # Pack flat array
    return np.concatenate([C[s] for s in species_list] +
                          [Q[s] for s in species_list])