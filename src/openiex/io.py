import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from .system import ExchangeSystem
from .method import Method
from .solver import SimulationResult

# Figure out the project root by walking up from here:
# src/chromatography_sim/io.py → chromatography_sim → src → <project root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_simulation(rel_path: str, result: SimulationResult):
    # unpack
    t, y, system, method = result.t, result.y, result.system, result.method
    """
    Save t, y, system, method under PROJECT_ROOT/rel_path(.npz/.json).
    rel_path is interpreted relative to the project root.
    """
    base      = _PROJECT_ROOT / rel_path
    npz_path  = base.with_suffix('.npz')
    json_path = base.with_suffix('.json')

    # ensure the directory exists
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # abort if either file already exists
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"Won’t overwrite {npz_path} or {json_path}")

    # save the arrays
    np.savez(npz_path, t=t, y=y)

    # save minimal metadata
    meta = {
        "system": system.to_dict(),
        "method": {
            "buffers": method.buffers,
            "blocks":  method.blocks,
        }
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_simulation(rel_path: str) -> SimulationResult:
    """
    Load a saved SimulationResult from
      PROJECT_ROOT/rel_path.npz  (arrays)
      PROJECT_ROOT/rel_path.json (metadata)
    Returns a SimulationResult(t, y, system, method).
    """
    base      = _PROJECT_ROOT / rel_path
    npz_path  = base.with_suffix('.npz')
    json_path = base.with_suffix('.json')

    # load arrays
    data = np.load(npz_path)
    t = data["t"]
    y = data["y"]

    # load metadata
    with open(json_path, 'r') as f:
        meta = json.load(f)

    # reconstruct system
    system = ExchangeSystem.from_dict(meta["system"])

    # reconstruct method
    m = meta["method"]
    method = Method(m["buffers"], m["blocks"])

    return SimulationResult(t=t, y=y, system=system, method=method)