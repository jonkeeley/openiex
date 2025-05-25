import pandas as pd
import numpy as np
from typing import Literal, Dict, List, Tuple, Any, Optional
from .state    import unpack_state
from .solver   import SimulationResult
from .method   import get_feed

def compute_chromatogram(result) -> Dict[str, np.ndarray]:
    """
    For a SimulationResult, compute and return a dict with:
      - "t"     : time (s)
      - "vol"   : injected volume (mL)
      - "cv"    : column volumes (unitless)
      - "A260"  : Abs260 (mAU)
      - "A280"  : Abs280 (mAU)
      - "cond"  : conductivity (mS/cm)
      - "percent_B": %B
    """
    t      = result.t
    C, _   = unpack_state(result.y, result.system)
    Vcol   = result.system.config.vol_column

    # Cumulative volume
    flows    = np.array([get_feed(ti, result.method, result.system)[1] for ti in t])
    flow_ml  = flows * 1e6
    vol_ml   = np.concatenate(([0.0],
                 np.cumsum((flow_ml[1:]+flow_ml[:-1])/2 * np.diff(t))))
    cv       = vol_ml / (Vcol * 1e6)

    # Allocate signals
    A260     = np.zeros_like(t)
    A280     = np.zeros_like(t)
    cond     = np.zeros_like(t)
    percent_B= np.zeros_like(t)

    # Fill signals
    for i, ti in enumerate(t):
        # UV + cond at outlet
        for name, conc in C.items():
            sp = result.system.species[name]
            c_val = conc[-1, i]
            A260[i]      += sp.ext_coeff_260 * c_val * 1e3
            A280[i]      += sp.ext_coeff_280 * c_val * 1e3
            cond[i]      += sp.mol_cond      * c_val

        # %B
        elapsed = 0.0
        for blk in result.method.blocks:
            flow_blk = blk["flow_rate_mL_min"] * 1.667e-8
            dur_blk  = blk["duration_CV"] * Vcol / flow_blk
            if elapsed <= ti < elapsed + dur_blk:
                frac       = (ti - elapsed) / dur_blk
                percent_B[i] = (blk["start_B"] + frac*(blk["end_B"]-blk["start_B"])) * 100
                break
            elapsed += dur_blk
        else:
            percent_B[i] = result.method.blocks[-1]["end_B"] * 100

    return {
        "t":          t,
        "vol":        vol_ml,
        "cv":         cv,
        "A260":       A260,
        "A280":       A280,
        "cond":       cond,
        "percent_B":  percent_B,
    }

def export_chromatogram_csv(
    result: SimulationResult,
    file_path: str
) -> str:
    """
    Compute chromatogram data and write it to a CSV file.

    The CSV will have columns:
      - Time (s)
      - Volume (mL)
      - CV
      - A280 (mAU)
      - A260 (mAU)
      - Conductivity (mS/cm)
      - %B

    Returns the path to the written file.
    """
    d = compute_chromatogram(result)
    df = pd.DataFrame({
        "Time (s)"            : d["t"],
        "Volume (mL)"         : d["vol"],
        "CV"                  : d["cv"],
        "A280 (mAU)"          : d["A280"],
        "A260 (mAU)"          : d["A260"],
        "Conductivity (mS/cm)": d["cond"],
        "%B"                  : d["percent_B"],
    })
    df.to_csv(file_path, index=False)
    return file_path

def analyze_fraction(
    result: Any,
    x_axis: Literal["time", "volume", "CV"],
    x_start: float,
    x_stop:  float,
    species: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Slice out the window [x_start, x_stop] (in chosen x-axis units),
    compute for each species:
      - actual Start and Stop (x-axis) values
      - Fraction Volume (mL)
      - Mean mobile-phase concentration
      - Total amount (mol or particles)
    
    If `species` is given, only those names are returned.
    """
    # Grab aggregated axes from compute_chromatogram
    d   = compute_chromatogram(result)
    t   = d["t"]
    vol = d["vol"]       # mL
    cv  = d["cv"]        # unitless

    # Pick x array
    if x_axis == "time":
        x = t
    elif x_axis == "volume":
        x = vol
    elif x_axis == "CV":
        x = cv
    else:
        raise ValueError("x_axis must be 'time','volume', or 'CV'")

    # Mask window
    mask = (x >= x_start) & (x <= x_stop)
    if not mask.any():
        raise ValueError(f"No data in [{x_start}, {x_stop}] {x_axis}")

    # Actual bounds & fraction volume
    actual_start = x[mask][0]
    actual_stop  = x[mask][-1]
    vol_start    = vol[mask][0]
    vol_stop     = vol[mask][-1]
    fraction_vol_ml = vol_stop - vol_start

    # Pull out species profiles
    C, _ = unpack_state(result.y, result.system)
    recs = []
    for name, conc_profile in C.items():
        if species and name not in species:
            continue

        # Only outlet segment
        vals = conc_profile[-1, mask]
        mean_c = np.mean(vals)
        unit   = result.system.species[name].unit

        if unit == "M":
            # Mean mol/L × L → moles
            total = mean_c * (fraction_vol_ml / 1e3)
        else:
            # Particles/mL × mL → particles
            total = mean_c * fraction_vol_ml

        recs.append({
            "Species"               : name,
            "Unit"                  : unit,
            "Start"                 : actual_start,
            "Stop"                  : actual_stop,
            "Fraction Volume (mL)"  : fraction_vol_ml,
            "Mean Concentration"    : mean_c,
            "Total Amount"          : total
        })

    return pd.DataFrame.from_records(recs)


def generate_fraction_dataframe(
    result: Any,
    x_axis: Literal["time", "volume", "CV"],
    fractions: List[Tuple[float, float]],
    species: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For each (start, stop) in `fractions`, call `analyze_fraction`,
    prepend a 'Fraction' column, and concatenate all results.
    """
    dfs = []
    for idx, (start, stop) in enumerate(fractions, start=1):
        df = analyze_fraction(result, x_axis, start, stop, species)
        df.insert(0, "Fraction", f"Fraction {idx}")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)