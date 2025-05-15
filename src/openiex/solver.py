import numpy as np
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
    """Encapsulates your ODE + progress bar."""
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
    Resume a previous SimulationResult at time t_start (absolute),
    then run a new method over t_eval (which is relative, from 0→new_t_final),
    stitch the trajectories, and return a new SimulationResult.

    Args:
      prev_result:     the original SimulationResult
      t_eval:          1D array of new times from 0 → new_t_final
      chromat_method:  Method for the continuation (buffers must match prev_result.method.buffers)
      system:          your ExchangeSystem instance (must equal prev_result.system)
    Keyword Args:
      t_start:         absolute time in prev_result.t to resume from;
                       if None, defaults to prev_result.t[-1]
      integrator:      scipy integrator name (e.g. "BDF", "RK45")
      **ivp_kwargs:    passed through to run_simulation (rtol, atol, max_step…)

    Returns:
      SimulationResult with:
        - t:      concatenated absolute times [0…t_start…t_start+t_eval[-1]]
        - y:      concatenated state array
        - system: same ExchangeSystem
        - method: a Method whose blocks = [truncated_old_block] + chromat_method.blocks
    """
    # unpack the old result
    t_old, y_old, old_system, old_method = (
        prev_result.t,
        prev_result.y,
        prev_result.system,
        prev_result.method,
    )

    # 1) sanity checks
    if system != old_system:
        raise ValueError("The provided system must match prev_result.system.")
    if chromat_method.buffers != old_method.buffers:
        raise ValueError("Buffers of the new method must match the old method.")

    # 2) pick t_start
    if t_start is None:
        t_start = t_old[-1]

    # 3) find which block covers t_start and truncate it
    block_start = 0.0
    truncated = None
    for blk in old_method.blocks:
        flow = blk["flow_rate_mL_min"] * 1.667e-8
        dur  = blk["duration_CV"] * system.config.vol_column / flow
        block_end = block_start + dur

        # allow t_start == block_end for the last block
        if block_start <= t_start <= block_end:
            frac = (t_start - block_start) / dur if dur>0 else 1.0
            truncated = {
                "buffer_A":    blk["buffer_A"],
                "buffer_B":    blk["buffer_B"],
                "start_B":     blk["start_B"],
                "end_B":       blk["start_B"] + frac*(blk["end_B"] - blk["start_B"]),
                "duration_CV": blk["duration_CV"] * frac,
                "flow_rate_mL_min": blk["flow_rate_mL_min"],
            }
            break
        block_start = block_end

    if truncated is None:
        raise ValueError(f"t_start={t_start} is outside the old method blocks.")

    # 4) build the combined Method
    combined_blocks = [truncated] + chromat_method.blocks
    method_combined = Method(chromat_method.buffers, combined_blocks)

    # 5) extract the state at t_start
    idx = np.searchsorted(t_old, t_start, side="left")
    if idx==len(t_old) or not np.isclose(t_old[idx], t_start):
        raise ValueError("t_start must exactly match a point in prev_result.t")
    y0 = y_old[:, idx]

    # 6) shift t_eval into absolute time and rerun
    t_abs = t_start + t_eval
    res_new = run_simulation(y0, t_abs, method_combined, system,
                             integrator=integrator, **ivp_kwargs)

    # 7) stitch the old and new segments (dropping duplicate at idx)
    t_comb = np.concatenate([t_old[:idx], res_new.t])
    y_comb = np.concatenate([y_old[:, :idx], res_new.y], axis=1)

    return SimulationResult(t=t_comb, y=y_comb,
                            system=system, method=method_combined)