import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from typing import Union, Tuple, Optional, List, Literal
import pandas as pd
from ipywidgets import interact, FloatSlider
from .state    import unpack_state
from .solver   import SimulationResult
from .method   import get_feed, convert_units
from .analysis import compute_chromatogram, generate_fraction_dataframe


def plot_fraction_table(df: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Render a pandas DataFrame as a table on the given Matplotlib Axes,
    formatting numeric entries as before and auto‐sizing columns so
    headers don’t overflow.
    """
    # 1) Build the display text (unchanged)
    display_vals = []
    for row in df.itertuples(index=False):
        disp_row = []
        for col_name, val in zip(df.columns, row):
            if isinstance(val, (int, float)):
                if col_name in ("Mean Concentration", "Total Amount"):
                    disp_row.append(f"{val:.4e}")
                else:
                    disp_row.append(f"{val:.3f}")
            else:
                disp_row.append(str(val))
        display_vals.append(disp_row)

    ax.axis("off")

    # 2) Create the table
    table = ax.table(
        cellText=display_vals,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    # 3) Styling and font
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # 4) Auto‐set each column’s width so header fits
    ncols = len(df.columns)
    table.auto_set_column_width(col=list(range(ncols)))

    # 5) Slightly increase vertical spacing
    table.scale(1, 1.5)

_NA = 6.022e23

def plot_single_species(
    result: SimulationResult,
    species: str,
    bound: bool = False,
    z_location: Union[str, int] = "outlet",
    include_feed: bool = False,
    x_axis: Literal["time", "volume", "CV"] = "time",
    x_window: Optional[Tuple[float, float]] = None,
    y_axis: Optional[Literal["M", "particles/mL"]] = None,
    y_window: Optional[Tuple[float, float]] = None,
    data_point: Optional[float] = None,
    annotate: bool = True,
    fractions: Optional[List[Tuple[float, float]]] = None
):
    """
    Plot one species’ profile (mobile or bound) with:
      - default to the species’ declared unit (None)
      - or override to "M" or "particles/mL"
      - optional feed overlay
      - optional shading for fraction intervals
      - optional data-point annotation
      - optional fraction‐summary table below
    """
    # 1) unpack profiles
    C, Q = unpack_state(result.y, result.system)
    prof = Q if bound else C

    # 2) resolve z-location
    if z_location == "inlet":
        z_idx, loc_str = 0, "Inlet"
    elif z_location == "outlet":
        z_idx, loc_str = -1, "Outlet"
    elif isinstance(z_location, int):
        z_idx, loc_str = z_location, f"z={z_location}"
    else:
        raise ValueError("z_location must be 'inlet','outlet', or int")

    # 3) compute axes
    d    = compute_chromatogram(result)
    t    = d["t"];    vol  = d["vol"];   cv   = d["cv"]

    # 4) pick x-axis
    if x_axis == "time":
        x, xlabel = t, "Time (s)"
    elif x_axis == "volume":
        x, xlabel = vol, "Injected Volume (mL)"
    elif x_axis == "CV":
        x, xlabel = cv, "Column Volumes (CV)"
    else:
        raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # 5) apply x_window
    mask = np.ones_like(x, bool)
    if x_window:
        xmin, xmax = x_window
        mask = (x >= xmin) & (x <= xmax)

    x_p   = x[mask]
    y_mol = prof[species][z_idx][mask]   # always in M

    # 6) determine target unit
    target = y_axis if y_axis else result.system.species[species].unit
    if target == "M":
        y_plot, ylabel = y_mol, "Concentration (M)"
    elif target == "particles/mL":
        # M → particles/mL
        y_plot = y_mol * _NA / 1e3
        ylabel = "Concentration (particles/mL)"
    else:
        raise ValueError("y_axis must be 'M','particles/mL', or None")

    # 7) set up figure (+ table if needed)
    if fractions:
        fig = plt.figure(constrained_layout=True, figsize=(8,6))
        gs  = GridSpec(2, 1, height_ratios=[3,1], figure=fig)
        ax  = fig.add_subplot(gs[0])
        ax_t= fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(8,5))
        ax_t = None

    # 8) plot the main trace
    ax.plot(x_p, y_plot, color="tab:blue", lw=1.5, label=species)

    # 9) overlay feed
    if include_feed:
        feed_vals = []
        for ti in t[mask]:
            feed_c, _ = get_feed(ti, result.method, result.system)
            val = feed_c[species]  # in M
            if target == "particles/mL":
                val = val * _NA / 1e3
            feed_vals.append(val)
        ax.plot(x_p, feed_vals, "--", color="gray", label=f"{species} feed")

    # 10) shade fractions
    if fractions:
        for start, stop in fractions:
            m_f = (x_p >= start) & (x_p <= stop)
            ax.fill_between(x_p, 0, y_plot, where=m_f,
                            color="lightgrey", alpha=0.3)

    # 11) highlight data_point
    if data_point is not None:
        idx = np.abs(x_p - data_point).argmin()
        xv, yv = x_p[idx], y_plot[idx]
        ax.axvline(xv, color="gray", linestyle="--")
        if annotate:
            ax.scatter([xv], [yv], color="red", zorder=3)
            lbl = {"time":"Time","volume":"Volume","CV":"CV"}[x_axis]
            txt = f"{lbl}: {xv:.3g}\n{species}: {yv:.3g}"
            ax.text(xv, yv, txt, ha="left", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 12) finalize axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_window: ax.set_xlim(x_window)
    if y_window: ax.set_ylim(y_window)
    ax.set_title(f"{species} ({'Bound' if bound else 'Mobile'}) at {loc_str}")
    ax.grid(True)
    ax.legend()

    # 13) fraction summary table
    if fractions and ax_t:
        df = generate_fraction_dataframe(
            result, x_axis, fractions, species=[species]
        )
        plot_fraction_table(df, ax_t)

    plt.show()

def plot_chromatogram(
    result: SimulationResult,
    x_axis: str = "time",
    x_window: Optional[Tuple[float, float]] = None,
    y_axis: str = "UV",
    y_window: Optional[Tuple[float, float]] = None,
    data_point: Optional[float] = None,
    annotate: bool = True,
    plot_uv260: bool = True,
    plot_uv280: bool = True,
    plot_conductivity: bool = True,
    plot_percent_B: bool = True,
    fractions: Optional[List[Tuple[float, float]]] = None,
    frac_species: Optional[List[str]] = None,
    show=True,
):
    """
    Plot overlaid chromatogram traces and optional fraction table,
    with vertical lines marking each fraction interval.

    Args:
      result            : SimulationResult
      x_axis            : "time", "volume", or "CV"
      x_window          : (min, max) in chosen x units for zoom
      y_axis            : baseline scale: "UV", "cond", or "%B"/"percent_B"
      y_window          : (min, max) in chosen y units for zoom
      data_point        : x-axis value to highlight
      annotate          : show boxed annotation at data_point
      plot_uv260        : include A260
      plot_uv280        : include A280
      plot_conductivity : include conductivity
      plot_percent_B    : include %B
      fractions         : list of (start,stop) windows in x units;
                           draws lines and shows a table below
    """
    # 1) compute all traces & axes
    d    = compute_chromatogram(result)
    t    = d["t"];    vol  = d["vol"];   cv   = d["cv"]
    A260 = d["A260"]; A280 = d["A280"]; cond = d["cond"]; B    = d["percent_B"]

    # 2) pick x-axis
    if x_axis == "time":
        x, xlabel = t, "Time (s)"
    elif x_axis == "volume":
        x, xlabel = vol, "Injected Volume (mL)"
    elif x_axis == "CV":
        x, xlabel = cv, "Column Volumes (CV)"
    else:
        raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # 3) mask by x_window
    mask = np.ones_like(x, dtype=bool)
    if x_window:
        xmin, xmax = x_window
        mask = (x >= xmin) & (x <= xmax)
    x_p  = x[mask]
    A260_p = A260[mask]; A280_p = A280[mask]
    cond_p = cond[mask]; B_p     = B[mask]

    # 4) scale onto y_axis baseline
    if y_axis.upper() == "UV":
        base = max(A260.max(), A280.max()) or 1.0
        cond_s = cond / (cond.max() or 1.0) * base
        B_s    = B    / 100                   * base
        traces = [
            ("A260", A260_p, plot_uv260, "purple"),
            ("A280", A280_p, plot_uv280, "blue"),
            ("Cond", cond_s[mask], plot_conductivity, "orange"),
            ("%B",   B_s[mask],    plot_percent_B, "green"),
        ]
        ylabel = "Absorbance (mAU)"

    elif y_axis.lower() == "cond":
        base  = cond.max() or 1.0
        A260_s = A260 / (max(A260.max(), A280.max()) or 1.0) * base
        A280_s = A280 / (max(A260.max(), A280.max()) or 1.0) * base
        B_s     = B    / 100                   * base
        traces = [
            ("A260", A260_s[mask], plot_uv260, "purple"),
            ("A280", A280_s[mask], plot_uv280, "blue"),
            ("Cond", cond_p,        plot_conductivity, "orange"),
            ("%B",   B_s[mask],     plot_percent_B, "green"),
        ]
        ylabel = "Conductivity (mS/cm)"

    elif y_axis in ("%B","percent_B"):
        base   = 100.0
        A260_s = A260 / (max(A260.max(), A280.max()) or 1.0) * base
        A280_s = A280 / (max(A260.max(), A280.max()) or 1.0) * base
        cond_s = cond / (cond.max()   or 1.0) * base
        traces = [
            ("A260", A260_s[mask], plot_uv260, "purple"),
            ("A280", A280_s[mask], plot_uv280, "blue"),
            ("Cond", cond_s[mask], plot_conductivity, "orange"),
            ("%B",   B_p,           plot_percent_B, "green"),
        ]
        ylabel = "%B"

    else:
        raise ValueError("y_axis must be 'UV','cond', or '%B'/'percent_B'")

    # 5) create figure + axes (with space for table)
    if fractions:
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        gs  = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
        ax  = fig.add_subplot(gs[0])
        ax_t= fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_t = None

    # 6) plot each trace
    for label, arr, show, color in traces:
        if show:
            ax.plot(x_p, arr, label=label, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    # 7) horizontal & vertical zoom
    if y_window:
        ax.set_ylim(y_window)

    # 8) draw fraction lines & labels
    if fractions:
        # get current y‐limits so spans fill the whole height
        ymin, ymax = ax.get_ylim()
        for i, (start, stop) in enumerate(fractions, 1):
            # light grey shading between start and stop
            ax.axvspan(start, stop, ymin=0, ymax=1,
                       color="lightgrey", alpha=0.2)
            # boundary lines
            ax.axvline(start, color="gray", linestyle="--", linewidth=1)
            ax.axvline(stop,  color="gray", linestyle="--", linewidth=1)
            # label in the middle, just above the top of the plot
            mid = 0.5*(start + stop)
            ax.text(mid, ymax, str(i),
                    ha="center", va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 9) highlight single data_point
    if data_point is not None:
        idx = np.abs(x_p - data_point).argmin()
        xv  = x_p[idx]
        ax.axvline(xv, color="black", linestyle=":")
        if annotate:
            unitlbl = {"time":"Time","volume":"Volume","CV":"CV"}[x_axis]
            lines = [f"{unitlbl}: {xv:.3g}"]
            if plot_uv260:        lines.append(f"A260: {A260_p[idx]:.3g}")
            if plot_uv280:        lines.append(f"A280: {A280_p[idx]:.3g}")
            if plot_conductivity: lines.append(f"Cond: {cond_p[idx]:.3g}")
            if plot_percent_B:    lines.append(f"%B:   {B_p[idx]:.3g}")
            ax.text(0.02, 0.95, "\n".join(lines),
                    transform=ax.transAxes,
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 10) plot fraction summary table
    if fractions and ax_t:
        df_frac = generate_fraction_dataframe(result,
                                              x_axis,
                                              fractions,
                                              species=frac_species)
        plot_fraction_table(df_frac, ax_t)
        
    if show:
        plt.show()
    else:
        return img

def plot_column_snapshot(
    result,
    species: str,
    bound: bool = False,
    x_axis: str = "time",
    x_window: Optional[Tuple[float, float]] = None,
    y_axis: Optional[str] = None,
    data_point: Optional[float] = None
):
    """
    Interactive column cross-section snapshot for one species.

    Args:
        result      : SimulationResult from run_simulation
        species     : species name (e.g. "Cl-", "em")
        bound       : if True, plot bound-phase Q; else mobile-phase C
        x_axis      : 'time', 'volume', or 'CV' for slider axis
        x_window    : optional (min, max) to restrict slider range
        y_axis      : 'M' or 'particles/mL'; if None, uses species.unit
        data_point  : initial slider value on chosen axis
    """
    # 1) Unpack profiles and system
    C_profiles, Q_profiles = unpack_state(result.y, result.system)
    profiles = Q_profiles[species] if bound else C_profiles[species]
    Nz, Nt = profiles.shape

    # 2) Chromatogram axes
    d    = compute_chromatogram(result)
    t    = d['t']; vol = d['vol']; cv = d['cv']
    if x_axis == 'time':
        x, xlabel = t, 'Time (s)'
    elif x_axis == 'volume':
        x, xlabel = vol, 'Injected Volume (mL)'
    elif x_axis == 'CV':
        x, xlabel = cv, 'Column Volumes (CV)'
    else:
        raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # 3) Apply x_window mask
    mask = np.ones_like(x, dtype=bool)
    if x_window is not None:
        xmin, xmax = x_window
        mask = (x >= xmin) & (x <= xmax)
    x_p = x[mask]
    prof_p = profiles[:, mask]

    # 4) Unit conversion for concentration
    unit = y_axis if y_axis else result.system.species[species].unit
    if unit == 'M':
        prof_plot = prof_p
        cbar_label = 'Concentration (M)'
    elif unit == 'particles/mL':
        prof_plot = prof_p * _NA / 1e3
        cbar_label = 'Concentration (particles/mL)'
    else:
        raise ValueError("y_axis must be 'M','particles/mL', or None")

    # 5) Physical height extent
    bed_h = result.system.config.bed_height

    # 6) Frame plotting function
    def _plot_frame(x_val):
        idx = int(np.argmin(np.abs(x_p - x_val)))
        data = prof_plot[::-1, idx][:, None]  # flip so z=0 at top
        fig, ax = plt.subplots(figsize=(2, 5), constrained_layout=True)
        im = ax.imshow(
            data,
            aspect='auto',
            cmap=cm.viridis,
            extent=[0, 1, 0, bed_h],
            vmin=prof_plot.min(),
            vmax=prof_plot.max()
        )
        phase = 'Bound' if bound else 'Mobile'
        ax.set_title(f"{phase} Phase — {species}\n{xlabel} = {x_p[idx]:.2f}")
        ax.set_xticks([])
        ax.set_ylabel('Column Height (m)')
        ax.invert_yaxis()  # z=0 at top
        fig.colorbar(im, ax=ax, label=cbar_label)
        plt.show()

    # 7) Slider initialization
    init = x_p[0] if data_point is None else np.clip(data_point, x_p[0], x_p[-1])
    step = (x_p[-1] - x_p[0]) / max(1, len(x_p)-1)

    interact(
        _plot_frame,
        x_val=FloatSlider(
            value=init,
            min=x_p[0],
            max=x_p[-1],
            step=step,
            description=xlabel,
            continuous_update=False
        )
    )