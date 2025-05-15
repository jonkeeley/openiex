import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from .state    import unpack_state
from .solver   import SimulationResult
from .method   import get_feed


def plot_single_species(
    result: SimulationResult,
    species: str,
    bound: bool = False,
    z_location: Union[str,int] = "outlet",
    include_feed: bool = False,
    x_axis: str = "time",                         # one of "time","volume","CV"
    time_window: Optional[Tuple[float,float]] = None
):
    """
    Plot a single species over time at a given z-location.

    Args:
      result      : SimulationResult from run_simulation or resume_simulation
      species     : name of the species to plot
      bound       : if True, plot the bound (Q) profile; else mobile (C)
      z_location  : "inlet", "outlet", or integer index for axial segment
      include_feed: if True, overlay the inlet feed concentration
      x_axis      : "time"      -> plot vs. t (s)
                    "volume"    -> cumulative mL passed
                    "CV"        -> CV units (fraction of bed volume)
      time_window : (min,max) in **seconds** to zoom in (applies before x-axis conversion)
    """
    # unpack
    t   = result.t
    C, Q = unpack_state(result.y, result.system)

    # pick which profile dict
    profiles = Q if bound else C

    # determine z-index
    if z_location == "inlet":
        z_idx = 0
        loc_str = "Inlet"
    elif z_location == "outlet":
        z_idx = -1
        loc_str = "Outlet"
    elif isinstance(z_location, int):
        z_idx = z_location
        loc_str = f"z={z_location}"
    else:
        raise ValueError("z_location must be 'inlet','outlet', or int")

    # possibly apply time_window mask first
    mask = np.ones_like(t, dtype=bool)
    if time_window is not None:
        tmin, tmax = time_window
        mask = (t >= tmin) & (t <= tmax)

    t_plot = t[mask]
    y_plot = profiles[species][z_idx][mask]

    # compute x-axis
    if x_axis == "time":
        x = t_plot
        xlabel = "Time (s)"
    else:
        # get flow_rate at each t for cumulative volume
        feeds = [get_feed(ti, result.method, result.system)[1] for ti in t_plot]
        # flow_rate is in m3/s; convert to mL/s
        flow_mL = np.array(feeds) * 1e6
        # cumulative trapezoidal integration to get mL
        vol_mL = np.concatenate([[0], np.cumsum((flow_mL[1:]+flow_mL[:-1])/2 * np.diff(t_plot))])
        if x_axis == "volume":
            x     = vol_mL
            xlabel = "Injected Volume (mL)"
        elif x_axis == "CV":
            Vbed_m3 = result.system.config.vol_column
            Vbed_mL = Vbed_m3 * 1e6
            x     = vol_mL / Vbed_mL
            xlabel = "Column Volumes (CV)"
        else:
            raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # start plotting
    fig, ax = plt.subplots()
    ax.plot(x, y_plot, label=f"{species} ({loc_str})")

    # overlay feed if requested
    if include_feed:
        feed_conc = []
        for ti in t_plot:
            feed, _ = get_feed(ti, result.method, result.system)
            feed_conc.append(feed[species])
        ax.plot(x, feed_conc, "--", label=f"{species} feed")

    ax.set_xlabel(xlabel)
    ylabel = ("Bound" if bound else "Mobile") + " Concentration (M)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} at {loc_str}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_chromatogram(
    result: SimulationResult,
    x_axis: str = "time",
    time_window: Optional[Tuple[float, float]] = None,
    plot_uv260: bool = True,
    plot_uv280: bool = True,
    plot_conductivity: bool = True,
    plot_percent_B: bool = True,
    highlight_time: Optional[float] = None,
    annotate: bool = True
):
    """
    Plot chromatogram traces (UV260, UV280, conductivity, %B) with flexible x-axis.

    Args:
      result           : SimulationResult
      x_axis           : "time", "volume", or "CV"
      time_window      : (t_min, t_max) in seconds to restrict the plot
      plot_uv260       : whether to plot A260
      plot_uv280       : whether to plot A280
      plot_conductivity: whether to plot conductivity
      plot_percent_B   : whether to plot %B
      highlight_time   : if given, draw a vertical line and annotate values at nearest t
      annotate         : whether to show annotation box
    """
    t = result.t  # seconds
    C, _ = unpack_state(result.y, result.system)

    # Apply time window mask
    mask = np.ones_like(t, dtype=bool)
    if time_window is not None:
        tmin, tmax = time_window
        mask = (t >= tmin) & (t <= tmax)
    t_plot = t[mask]

    # Compute UV and conductivity
    A260 = np.zeros_like(t_plot)
    A280 = np.zeros_like(t_plot)
    cond = np.zeros_like(t_plot)

    for i, ti in enumerate(t_plot):
        # outlet index = -1
        out_idx = -1
        # UV
        for name, conc_profile in C.items():
            species = result.system.species[name]
            conc = conc_profile[out_idx, np.where(t == ti)[0][0]]
            A260[i] += species.ext_coeff_260 * conc
            A280[i] += species.ext_coeff_280 * conc
        # conductivity
        cond[i] = sum(
            result.system.species[name].mol_cond * conc_profile[out_idx, np.where(t == ti)[0][0]]
            for name, conc_profile in C.items()
        )

    # Compute %B
    percB = []
    Vcol = result.system.config.vol_column
    for ti in t_plot:
        # reuse get_feed logic for fraction
        comp, _ = get_feed(ti, result.method, result.system)
        # assume percent_B = fraction of B in feed method
        # find %B by comparing a dummy species in buffers, so better to recompute via blocks:
        block_start = 0.0
        for blk in result.method.blocks:
            flow = blk["flow_rate_mL_min"] * 1.667e-8
            dur   = blk["duration_CV"] * Vcol / flow
            block_end = block_start + dur
            if block_start <= ti < block_end:
                frac = (ti - block_start) / dur
                percent_B = blk["start_B"] + frac * (blk["end_B"] - blk["start_B"])
                percB.append(percent_B * 100)
                break
            block_start = block_end
        else:
            percB.append(result.method.blocks[-1]["end_B"] * 100)
    percB = np.array(percB)

    # Compute x-axis
    if x_axis == "time":
        x = t_plot
        xlabel = "Time (s)"
    else:
        # get flow rates at each plotted t
        flows = np.array([get_feed(ti, result.method, result.system)[1] for ti in t_plot])
        flow_ml_s = flows * 1e6  # mL/s
        # cumulative volume by trapezoidal rule
        # vol[0]=0, then integrate over t_plot
        vol = np.concatenate((
            [0.0],
            np.cumsum((flow_ml_s[1:]+flow_ml_s[:-1]) / 2 * np.diff(t_plot))
        ))
        flow_ml_s = flows * 1e6
        # cumulative volume (mL)
        
        if x_axis == "volume":
            x = vol
            xlabel = "Injected Volume (mL)"
        elif x_axis == "CV":
            Vbed_ml = Vcol * 1e6
            x = vol / Vbed_ml
            xlabel = "Column Volumes (CV)"
        else:
            raise ValueError("x_axis must be 'time','volume', or 'CV'")
            raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # Begin plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'uv260': 'purple',
        'uv280': 'blue',
        'cond': 'orange',
        '%B': 'green'
    }
    lines = []
    if plot_uv260:
        lines += ax.plot(x, A260, label='A260', color=colors['uv260'])
    if plot_uv280:
        lines += ax.plot(x, A280, label='A280', color=colors['uv280'])
    if plot_conductivity:
        ax.set_ylabel('Signal')  # placeholder
        lines += ax.plot(x, cond, label='Conductivity', color=colors['cond'])
    if plot_percent_B:
        lines += ax.plot(x, percB, label='%B', color=colors['%B'])

    ax.set_xlabel(xlabel)
    ax.grid(True)
    ax.legend()
    plt.title('Chromatogram')

    # highlight a time point
    if highlight_time is not None:
        # find nearest index in t_plot
        idx = np.abs(t_plot - highlight_time).argmin()
        xt = x[idx]
        ax.axvline(xt, color='gray', linestyle='--')
        if annotate:
            vals = {
                't': t_plot[idx],
                'A260': A260[idx],
                'A280': A280[idx],
                'Conductivity': cond[idx],
                '%B': percB[idx]
            }
            text = '\n'.join(f"{k}: {v:.3g}" for k, v in vals.items())
            ax.text(0.02, 0.95, text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()