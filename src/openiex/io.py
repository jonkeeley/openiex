import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Union

from .system import ExchangeSystem
from .method import Method
from .solver import SimulationResult

# Default directory for saving results
DEFAULT_RESULTS_DIR = Path(os.getcwd()) / "results"


def save_simulation(
    name: str,
    result: SimulationResult,
    results_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> None:
    """
    Save a SimulationResult to disk under a 'results' folder.

    Args:
        name: Base filename (no suffix) for saving.
        result: The SimulationResult object to save.
        results_dir: Directory in which to save (default: './results').
        overwrite: If False, will error if files already exist.

    Outputs:
        Creates '<results_dir>/<name>.npz' and '<results_dir>/<name>.json'.
    """
    # Determine target directory
    base_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    npz_path = base_dir / f"{name}.npz"
    json_path = base_dir / f"{name}.json"

    # Prevent accidental overwrite
    if not overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing files: {npz_path}, {json_path}"
        )

    # Save numeric arrays
    np.savez(npz_path, t=result.t, y=result.y)

    # Save metadata (system + method parameters)
    meta = {
        "system": result.system.to_dict(),
        "method": {
            "buffers": result.method.buffers,
            "blocks": result.method.blocks,
        }
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Simulation saved: {npz_path} and {json_path}")


def load_simulation(
    name: str,
    results_dir: Optional[Union[str, Path]] = None
) -> SimulationResult:
    """
    Load a SimulationResult previously saved with save_simulation().

    Args:
        name: Base filename (no suffix) to load.
        results_dir: Directory where files were saved (default: './results').

    Returns:
        A SimulationResult(t, y, system, method).
    """
    base_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    npz_path = base_dir / f"{name}.npz"
    json_path = base_dir / f"{name}.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz file: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing json file: {json_path}")

    # Load arrays
    data = np.load(npz_path)
    t = data["t"]
    y = data["y"]

    # Load metadata
    with open(json_path, 'r') as f:
        meta = json.load(f)

    # Reconstruct system and method
    system = ExchangeSystem.from_dict(meta["system"])
    method = Method(
        meta["method"]["buffers"],
        meta["method"]["blocks"]
    )

    return SimulationResult(t=t, y=y, system=system, method=method)
