import numpy as np
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
    chromat_method: Method,
    system: ExchangeSystem,
    y0: np.ndarray,
    t_eval: np.ndarray,
    integrator: str = "BDF",
    **ivp_kwargs
):
    """
    High‑level helper: wraps ODEFunction + solve_ivp + progress bar.
    
    Args:
      chromat_method: your Method(buffers, blocks)
      system:           ExchangeSystem(...)
      y0:               initial state vector
      t_eval:           times at which to store solution
      integrator:       one of scipy's methods, e.g. "RK45" or "BDF"
      **ivp_kwargs:     other solve_ivp args (rtol, atol, max_step...)
    
    Returns the SolveResult with .t, .y, .nfev, etc.
    """
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
    return sol