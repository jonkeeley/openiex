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

def calc_dQdt(C, Q, Qbar, feed, system, eps=1e-30):
    Nz = system.config.Nz
    _, flow_rate = feed
    vol_interstitial = system.config.vol_interstitial
    t_res = vol_interstitial / flow_rate

    dQdt = { s: np.zeros(Nz) for s in system.species }

    for i in system.ions:
        for z in range(Nz):
            # Ion–ion exchange (log-space)
            for k in system.ions:
                if k != i:
                    expo_ads = system.ln_k_ads[(i, k)] + np.log(np.maximum(C[i][z], eps)) + np.log(np.maximum(Q[k][z], eps))
                    expo_des = system.ln_k_des[(i, k)] + np.log(np.maximum(Q[i][z], eps)) + np.log(np.maximum(C[k][z], eps))
                    dQdt[i][z] += np.exp(expo_ads)
                    dQdt[i][z] -= np.exp(expo_des)

            # Protein–ion exchange (log-space)
            for j in system.proteins:
                nu_j = system.proteins[j].nu
                expo_ads = system.ln_k_ads[(i, j)] + nu_j * np.log(np.maximum(C[i][z], eps)) + np.log(np.maximum(Q[j][z], eps))
                expo_des = system.ln_k_des[(i, j)] + np.log(np.maximum(Q[i][z], eps)) + nu_j * np.log(np.maximum(C[j][z], eps))
                dQdt[i][z] += np.exp(expo_ads)
                dQdt[i][z] -= np.exp(expo_des)

    for j in system.proteins:
        for z in range(Nz):
            nu_j = system.proteins[j].nu
            for i in system.ions:
                expo_ads = system.ln_k_ads[(j, i)] + np.log(np.maximum(C[j][z], eps)) + nu_j * np.log(np.maximum(Qbar[i][z], eps))
                expo_des = system.ln_k_des[(j, i)] + np.log(np.maximum(Q[j][z], eps)) + nu_j * np.log(np.maximum(C[i][z], eps))
                dQdt[j][z] += np.exp(expo_ads)
                dQdt[j][z] -= np.exp(expo_des)

    # Scale by residence time
    for s in dQdt:
        dQdt[s] /= t_res

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