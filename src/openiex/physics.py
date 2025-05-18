import numpy as np

def calc_Qbar(Q, system):
    Nz = system.config.Nz
    Lambda = system.config.Lambda
    Qbar = {i: np.zeros(Nz) for i in system.ions}
    for i in system.ions.keys():
        for z_idx in range(Nz):
            sum_Q_other_ions = sum(
            Q[k][z_idx] for k in system.ions.keys() if k != i
            )
            blocked = sum(
                (p.nu + p.sigma) * Q[p.name][z_idx] for p in system.proteins.values()
            )
            if Q[i][z_idx] > 1e-12:
                Qbar[i][z_idx] = (Lambda - blocked) / (1 + sum_Q_other_ions / Q[i][z_idx])
            else:
                Qbar[i][z_idx] = 0.0
    return Qbar

def calc_Qstar(C, Qbar, system):
    """
    Returns:
      Qstar: dict[j][i] = np.ndarray of length Nz
    """
    Qstar = {j: {} for j in system.proteins}
    for j, prot in system.proteins.items():
        nu = prot.nu
        for i in system.ions:
            # elementwise over z
            Qstar[j][i] = (
                system.K_eq[(j,i)]
                * (Qbar[i]**nu)
                * C[j]
                / (C[i]**nu)
            )
    return Qstar

def calc_dQdt(C, Q, Qbar, Qstar, feed, system):
    """
    Compute dQ/dt using:
      - SMA (only) for ion–ion
      - SMA or LDF for protein–ion, with weights from Qstar
    """
    Nz = system.config.Nz
    _, flow_rate = feed
    t_res = system.config.vol_interstitial / flow_rate

    # initialize output
    dQdt = {s: np.zeros(Nz) for s in system.species}

    # outer loop over z
    for z in range(Nz):
        # 1) protein–ion rates rji[j][i]
        rji = {j: {} for j in system.proteins}

        for j, prot in system.proteins.items():
            nu = prot.nu
            sum_r = 0.0

            # pre-sum Qstar[j][i][z] for normalization (avoid zero-div)
            denom = sum(Qstar[j][i][z] for i in system.ions) or 1e-12

            for i in system.ions:
                model = system.pair_kinetic_model.get((j, i), system.kinetic_model)

                if model == "SMA":
                    # mass-action
                    r = (
                        system.k_ads[(j, i)] * C[j][z] * (Qbar[i][z] ** nu)
                        - system.k_des[(j, i)] * Q[j][z] * (C[i][z] ** nu)
                    )
                else:
                    # LDF toward Qstar
                    w = Qstar[j][i][z] / denom
                    Qji = w * Q[j][z]
                    r   = system.k_ldf[(j, i)] * (Qstar[j][i][z] - Qji)

                rji[j][i] = r
                sum_r     += r

            # protein balance
            dQdt[j][z] = sum_r / t_res
        # 2) ion balances
        for i in system.ions:
            acc = 0.0
            # ion–ion (SMA only)
            for k in system.ions:
                if k == i:
                    continue
                acc += (
                    system.k_ads[(i, k)] * C[i][z] * Q[k][z]
                    - system.k_des[(i, k)] * Q[i][z] * C[k][z]
                )
            # subtract each protein displacement
            for j, prot in system.proteins.items():
                acc -= prot.nu * rji[j][i]
            dQdt[i][z] = acc / t_res
    return dQdt

def calc_dCdt(C, dQdt, feed, system):
    Nz = system.config.Nz
    dz = system.config.dz
    C_feed, flow_rate = feed
    epsilon_p = system.config.epsilon_p
    epsilon_i = system.config.epsilon_i
    A = system.config.A
    v_interstitial = flow_rate / (A * epsilon_i)
    dCdt = {s: np.zeros(Nz) for s in system.species}
    for s in system.species.keys():
        D = system.species[s].D
        Kd = system.species[s].Kd
        for z_idx in range(Nz):
            if z_idx == 0:
                # INLET (one-sided forward difference)
                dCdz = (C[s][0] - C_feed[s]) / dz
                d2Cdz2 = (C[s][1] - 2 * C[s][0] + C_feed[s]) / dz**2
            elif z_idx == system.config.Nz - 1:
                # OUTLET (backward difference)
                dCdz = (C[s][-1] - C[s][-2]) / dz
                d2Cdz2 = (C[s][-1] - 2 * C[s][-2] + C[s][-3]) / dz**2
            else:
                # INTERIOR (central difference for diffusion, backward for advection)
                dCdz = (C[s][z_idx] - C[s][z_idx - 1]) / dz
                d2Cdz2 = (C[s][z_idx + 1] - 2 * C[s][z_idx] + C[s][z_idx - 1]) / dz**2
            numerator = -v_interstitial * dCdz + D * d2Cdz2 - epsilon_p * Kd * dQdt[s][z_idx]
            denominator = epsilon_i + epsilon_p * Kd
            dCdt[s][z_idx] = numerator / denominator
    return dCdt