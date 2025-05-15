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
    y_axis: str = "UV",
    y_window: Optional[Tuple[float, float]] = None,
    t_data_points: Optional[float] = None,
    annotate: bool = True
):
    """
    Plot a single chromatogram trace based on y_axis selection.

    Args:
      result      : SimulationResult
      x_axis      : "time", "volume", or "CV"
      time_window : (t_min, t_max) in seconds to restrict the plot
      y_axis      : which signal to plot: "UV", "cond", or "percent_B";
                    default "UV" plots both A260 & A280
      y_window    : (y_min, y_max) to zoom y-axis in selected units
      t_data_points: time to highlight with vertical line and annotation
      annotate    : whether to show annotation box at highlight
    """
    # raw time & concentrations
    t = result.t
    C, _ = unpack_state(result.y, result.system)

    # apply time window
    mask = np.ones_like(t, dtype=bool)
    if time_window:
        tmin, tmax = time_window
        mask = (t >= tmin) & (t <= tmax)
    t_plot = t[mask]

    # pre-allocate
    A260 = A280 = cond = percB = None

    # compute based on y_axis
    if y_axis.upper() == "UV":
        A260 = np.zeros_like(t_plot)
        A280 = np.zeros_like(t_plot)
        for i, ti in enumerate(t_plot):
            idx = np.where(t == ti)[0][0]
            out = -1
            for name, conc in C.items():
                sp = result.system.species[name]
                c = conc[out, idx]
                A260[i] += sp.ext_coeff_260 * c
                A280[i] += sp.ext_coeff_280 * c
        ylabel = "Absorbance (AU)"

    elif y_axis.lower() == "cond":
        # conductivity: λ (S·cm²/mol) * C (M) gives mS/cm directly
        cond = np.zeros_like(t_plot)
        for i, ti in enumerate(t_plot):
            idx = np.where(t == ti)[0][0]
            out = -1
            for name, conc in C.items():
                sp = result.system.species[name]
                c = conc[out, idx]
                cond[i] += sp.mol_cond * c
        ylabel = "Conductivity (mS/cm)"

    elif y_axis == "%B" or y_axis.lower() == "percent_B":
        percB = []
        Vcol = result.system.config.vol_column
        for ti in t_plot:
            bs = 0.0
            for blk in result.method.blocks:
                flow = blk["flow_rate_mL_min"] * 1.667e-8
                dur = blk["duration_CV"] * Vcol / flow
                if bs <= ti < bs + dur:
                    frac = (ti - bs) / dur
                    pB = blk["start_B"] + frac*(blk["end_B"]-blk["start_B"])
                    percB.append(pB*100)
                    break
                bs += dur
            else:
                percB.append(result.method.blocks[-1]["end_B"]*100)
        percB = np.array(percB)
        ylabel = "%B"

    else:
        raise ValueError("y_axis must be 'UV', 'cond', or '%B'/'percent_B'")

    # build x-axis
    if x_axis == "time":
        x = t_plot
        xlabel = "Time (s)"
    else:
        flows = np.array([get_feed(ti, result.method, result.system)[1] for ti in t_plot])
        flow_ml_s = flows * 1e6
        vol = np.concatenate((
            [0.0], np.cumsum((flow_ml_s[1:]+flow_ml_s[:-1])/2 * np.diff(t_plot))
        ))
        if x_axis == "volume":
            x = vol
            xlabel = "Injected Volume (mL)"
        elif x_axis == "CV":
            Vbed_ml = result.system.config.vol_column * 1e6
            x = vol / Vbed_ml
            xlabel = "Column Volumes (CV)"
        else:
            raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    colors = {'UV260':'purple','UV280':'blue','cond':'orange','%B':'green'}
    if y_axis.upper() == "UV":
        ax.plot(x, A260, label='A260', color=colors['UV260'])
        ax.plot(x, A280, label='A280', color=colors['UV280'])
    elif y_axis.lower() == 'cond':
        ax.plot(x, cond, label='Conductivity', color=colors['cond'])
    else:
        ax.plot(x, percB, label='%B', color=colors['%B'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    # apply y-window if given
    if y_window:
        ymin, ymax = y_window
        ax.set_ylim(ymin, ymax)

    # highlight
    if t_data_points is not None:
        idx = np.abs(t_plot - t_data_points).argmin()
        xt = x[idx]
        ax.axvline(xt, color='gray', linestyle='--')
        if annotate:
            vals = {}
            if A260 is not None:
                vals['A260'] = A260[idx]
                vals['A280'] = A280[idx]
            if cond is not None:
                vals['Conductivity'] = cond[idx]
            if percB is not None:
                vals['%B'] = percB[idx]
            vals['t'] = t_plot[idx]
            txt = '\n'.join(f"{k}: {v:.3g}" for k, v in vals.items())
            ax.text(0.02,0.95,txt,transform=ax.transAxes,
                    fontsize=9,verticalalignment='top',
                    bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))

    plt.tight_layout()
    plt.show()

