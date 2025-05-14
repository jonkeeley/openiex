import numpy as np

class Method:
    def __init__(self, buffers, blocks):
        self.buffers = buffers
        self.blocks = blocks

def validate_method(method: Method, system):
    buffers = method.buffers
    blocks = method.blocks
    species_list = list(system.species.keys())
    # Check each buffer has correct species
    for name, buffer in buffers.items():
        if set(buffer.keys()) != set(species_list):
            raise ValueError(f"Buffer '{name}' species mismatch. Missing or extra entries.")

    # Check blocks only uses defined buffers
    all_buffer_names = set(buffers.keys())
    for idx, step in enumerate(blocks):
        if step["buffer_A"] not in all_buffer_names:
            raise ValueError(f"Block {idx} uses undefined buffer_A '{step['buffer_A']}'")
        if step["buffer_B"] not in all_buffer_names:
            raise ValueError(f"Block {idx} uses undefined buffer_B '{step['buffer_B']}'")
   
def convert_units(data_dict, system, direction="to_M"):
    """
    Convert a dict of values between M and user-specified units.

    Args:
        data_dict: dict[str, scalar or np.ndarray]
            Values per species (buffer, C, Q, or result arrays)
        units_dict: dict[str, "M" or "particles/mL"]
            Units associated with each species
        direction: "to_M" or "from_M"
            "to_M": convert from user-specified units to M
            "from_M": convert from M to user-specified units

    Returns:
        dict[str, converted values]
    """
    factor = 1e3 / 6.022e23  # particles/mL ↔ M
    converted = {}

    for s, val in data_dict.items():

        unit = system.species[s].unit
        val = np.asarray(val)  # Handles scalar, 1D, or 2D

        if unit not in {"M", "particles/mL"}:
            raise ValueError(f"Unknown unit '{unit}' for species '{s}'")

        if direction == "to_M":
            if unit == "M":
                converted[s] = val
            elif unit == "particles/mL":
                converted[s] = val * factor

        elif direction == "from_M":
            if unit == "M":
                converted[s] = val
            elif unit == "particles/mL":
                converted[s] = val / factor

        else:
            raise ValueError(f"Invalid direction '{direction}'. Use 'to_M' or 'from_M'.")

    return converted
      
def get_feed(t, method: Method, system):
    buffers = method.buffers
    blocks  = method.blocks
    species_list = list(system.species.keys())
    vol_column = system.config.vol_column

    block_start = 0.0
    for block in blocks:
        flow_rate_m3_s = block["flow_rate_mL_min"] * 1.667e-8
        duration_s     = block["duration_CV"] * vol_column / flow_rate_m3_s
        block_end     = block_start + duration_s

        if block_start <= t < block_end:
            frac     = (t - block_start) / duration_s
            percent_B = block["start_B"] + frac * (block["end_B"] - block["start_B"])
            comp_A   = buffers[block["buffer_A"]]
            comp_B   = buffers[block["buffer_B"]]
            composition = {
                s: (1 - percent_B)*comp_A[s] + percent_B*comp_B[s]
                for s in species_list
            }
            return convert_units(composition, system, "to_M"), flow_rate_m3_s

        block_start = block_end

    # If we get here, t is ≥ end of last block: just return the last block’s endpoint
    last = blocks[-1]
    flow_rate_m3_s = last["flow_rate_mL_min"] * 1.667e-8
    comp_A = buffers[last["buffer_A"]]
    comp_B = buffers[last["buffer_B"]]
    # use end_B for the composition
    composition = {
        s: (1 - last["end_B"])*comp_A[s] + last["end_B"]*comp_B[s]
        for s in species_list
    }
    return convert_units(composition, system, "to_M"), flow_rate_m3_s

