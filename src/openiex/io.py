# src/chromatography_sim/io.py
import numpy as np
import json
from pathlib import Path
from typing import Tuple
from .system import ExchangeSystem
from .method import Method

def save_simulation(
    path: str,
    t: np.ndarray,
    y: np.ndarray,
    system: ExchangeSystem,
    method: Method,
):
    """
    Save only the essentials:
      - t: 1D time vector
      - y: 2D state array
      - system: ExchangeSystem (we call .to_dict())
      - method: Method (we pull buffers & blocks)
    Fails if files already exist.
    """
    base      = Path(path)
    npz_path  = base.with_suffix('.npz')
    json_path = base.with_suffix('.json')

    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"Won’t overwrite {npz_path} or {json_path}")

    # 1) numeric arrays
    np.savez(npz_path, t=t, y=y)

    # 2) metadata from your objects
    meta = {
        "system": system.to_dict(),
        "method": {
            "buffers": method.buffers,
            "blocks":  method.blocks
        }
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_simulation(path: str) -> Tuple[np.ndarray, np.ndarray, ExchangeSystem, Method]:
    """
    Returns (t, y, system, method).
    Rebuilds ExchangeSystem via from_dict, and Method by passing buffers/blocks back in.
    """
    base = Path(path)
    data = np.load(base.with_suffix('.npz'))
    with open(base.with_suffix('.json')) as f:
        meta = json.load(f)

    # Reconstruct system & method
    system = ExchangeSystem.from_dict(meta["system"])
    meth_meta = meta["method"]
    method = Method(meth_meta["buffers"], meth_meta["blocks"])

    return data['t'], data['y'], system, method