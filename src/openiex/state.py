import numpy as np

def initialize_profiles(initial_conditions, system):
    Nz = system.config.Nz
    species_list = list(system.species.keys())
    C_init = {s: np.zeros(Nz) for s in species_list}
    Q_init = {s: np.zeros(Nz) for s in species_list}

    for s in species_list:
        if s in initial_conditions.keys():
            if "C" in initial_conditions[s].keys():
                C_init[s][:] = initial_conditions[s]["C"]
            if "Q" in initial_conditions[s].keys():
                Q_init[s][:] = initial_conditions[s]["Q"]
    return np.concatenate([C_init[s] for s in species_list] +
                          [Q_init[s] for s in species_list])

def load_profiles():
    pass

def unpack_state(y, system):
    Nz = system.config.Nz
    species_list = list(system.species.keys())
    num_species = len(species_list)
    expected_len = 2 * num_species * Nz

    if len(y) != expected_len:
        raise ValueError(
            f"State array has length {len(y)}, expected {expected_len} "
            f"({num_species} species x 2 (C + Q) x {Nz} segments)"
        )

    split = num_species * Nz
    C_arrs = y[:split]
    Q_arrs = y[split:]

    C = {s: C_arrs[i * Nz:(i + 1) * Nz] for i, s in enumerate(species_list)}
    Q = {s: Q_arrs[i * Nz:(i + 1) * Nz] for i, s in enumerate(species_list)}

    return C, Q

def pack_state(C, Q, system):
    species_list = list(system.species.keys())
    Nz = system.config.Nz

    # Check for missing or misaligned species
    for s in species_list:
        if s not in C.keys():
            raise KeyError(f"Species '{s}' missing from C.")
        if s not in Q.keys():
            raise KeyError(f"Species '{s}' missing from Q.")

        if len(C[s]) != Nz:
            raise ValueError(f"C[{s}] has length {len(C[s])}, expected {Nz}")
        if len(Q[s]) != Nz:
            raise ValueError(f"Q[{s}] has length {len(Q[s])}, expected {Nz}")

    # Pack flat array
    return np.concatenate([C[s] for s in species_list] +
                          [Q[s] for s in species_list])