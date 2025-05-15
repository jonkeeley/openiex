import numpy as np
from scipy.interpolate import interp1d
from typing import Any
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from .state import unpack_state, pack_state
from .physics import calc_Qbar, calc_dQdt, calc_dCdt
from .method import Method, get_feed
from .system import ExchangeSystem
try:
    from tqdm.notebook import tqdm
except ImportError:
    # Otherwise, use the standard console one
    from tqdm import tqdm
from typing import Tuple

@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    system: ExchangeSystem
    method: Method

class SimulationTracker:
    def __init__(self, t_final):
        self.bar = tqdm(total=t_final, desc="Simulating", unit="s")
        self.last_t = 0.0

    def update(self, t):
        if t > self.last_t:
            self.bar.update(t - self.last_t)
            self.last_t = t

    def close(self):
        self.bar.close()

def method_duration(method: Method, system):
    vol = system.config.vol_column
    return sum(
        block["duration_CV"] * vol / (block["flow_rate_mL_min"] * 1.667e-8)
        for block in method.blocks
    )

class ODEFunction:
    def __init__(self, chromat_method: Method, system: ExchangeSystem):
        self.method  = chromat_method
        self.system  = system
        t_final      = method_duration(chromat_method, system)
        self.tracker = SimulationTracker(t_final)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        self.tracker.update(t)

        C, Q    = unpack_state(y, self.system)
        feed    = get_feed(t, self.method, self.system)
        Qbar    = calc_Qbar(Q, self.system)
        dQdt    = calc_dQdt(C, Q, Qbar, feed, self.system)
        dCdt    = calc_dCdt(C, dQdt, feed, self.system)
        return pack_state(dCdt, dQdt, self.system)

    def close(self):
        self.tracker.close()


def run_simulation(
    y0: np.ndarray,
    t_eval: np.ndarray,
    chromat_method: Method,
    system: ExchangeSystem,
    integrator: str = "BDF",
    **ivp_kwargs
) -> SimulationResult:
    ode_fn = ODEFunction(chromat_method, system)
    sol = solve_ivp(
        fun=ode_fn,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method=integrator,
        **ivp_kwargs
    )
    ode_fn.close()
    return SimulationResult(sol.t, sol.y, system, chromat_method)

def resume_simulation(
    prev_result: SimulationResult,
    t_eval: np.ndarray,
    chromat_method: Method,
    system: ExchangeSystem,
    *,
    t_start: float = None,
    integrator: str = "BDF",
    **ivp_kwargs: Any
) -> SimulationResult:
    """
    Resume a previous SimulationResult at (or just before) t_start,
    keeping all earlier blocks, truncating the current block there,
    then appending the new chromat_method.blocks. Returns a new
    SimulationResult with combined t, y, system, and method.
    """
    t_old, y_old, old_sys, old_method = (
        prev_result.t,
        prev_result.y,
        prev_result.system,
        prev_result.method,
    )

    if system.to_math_dict() != old_sys.to_math_dict():
        raise ValueError("System definitions (config + binding params) must match prev_result.")
    if chromat_method.buffers != old_method.buffers:
        raise ValueError("Buffers of the new method must match old_method.buffers.")

    if t_start is None:
        t_start = t_old[-1]
    if t_start > t_old[-1]:
        raise ValueError(f"t_start={t_start:.2f} exceeds original duration of {t_old[-1]:.2f} s")

    interp = interp1d(t_old, y_old, axis=1, bounds_error=True)
    y0 = interp(t_start)

    elapsed = 0.0
    truncated = None
    Vcol = system.config.vol_column  # total bed volume (m^3)
    block_index = None

    for idx_blk, blk in enumerate(old_method.blocks):
        flow_m3_s = blk["flow_rate_mL_min"] * 1.667e-8
        dur_s     = blk["duration_CV"] * Vcol / flow_m3_s
        end       = elapsed + dur_s

        if elapsed <= t_start <= end:
            dt_in_block = t_start - elapsed
            # exact CV elapsed in this block
            cv_part = dt_in_block * flow_m3_s / Vcol
            # compute fraction for %B interpolation
            frac = dt_in_block / dur_s if dur_s > 0 else 1.0
            startB, endB = blk["start_B"], blk["end_B"]
            endB_part = startB + frac * (endB - startB)

            truncated = {
                "buffer_A": blk["buffer_A"],
                "buffer_B": blk["buffer_B"],
                "start_B": startB,
                "end_B": endB_part,
                "duration_CV": cv_part,
                "flow_rate_mL_min": blk["flow_rate_mL_min"],
            }
            block_index = idx_blk
            break

        elapsed = end

    if truncated is None or block_index is None:
        raise ValueError(f"t_start={t_start} s is not inside any old block.")

    combined_blocks = (
        old_method.blocks[:block_index]
        + [truncated]
        + chromat_method.blocks
    )
    method_combined = Method(chromat_method.buffers, combined_blocks)

    t_abs   = t_start + t_eval
    new_res = run_simulation(y0, t_abs, method_combined, system,
                              integrator=integrator, **ivp_kwargs)

    mask    = t_old < t_start
    t_comb  = np.concatenate([t_old[mask], new_res.t])
    y_comb  = np.concatenate([y_old[:, mask], new_res.y], axis=1)

    return SimulationResult(
        t=t_comb,
        y=y_comb,
        system=system,
        method=method_combined
    )