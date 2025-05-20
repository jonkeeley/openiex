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

def calc_dQdt(C, Q, Qbar, feed, system, eps=1e-30, max_step=0.1):
    Nz = system.config.Nz
    _, flow_rate     = feed
    vol_interstitial = system.config.vol_interstitial
    t_res            = vol_interstitial / flow_rate

    # 1) Precompute log‐clamped arrays once per species
    logC    = { s: np.log(np.maximum(C[s],    eps)) for s in system.species }
    logQ    = { s: np.log(np.maximum(Q[s],    eps)) for s in system.species }
    logQbar = { i: np.log(np.maximum(Qbar[i], eps)) for i in system.ions }

    # 2) Initialize output
    dQdt = { s: np.zeros(Nz) for s in system.species }

    # 3) Vectorized ion–ion & protein–ion exchange for ions
    for i in system.ions:
        Li = logC[i]    # shape (Nz,)
        Qi = logQ[i]

        # --- ion–ion exchange ---
        for k in system.ions:
            if k == i:
                continue
            Lk = logC[k]
            Qk = logQ[k]
            ads_exp = system.ln_k_ads[(i, k)] + Li + Qk
            des_exp = system.ln_k_des[(i, k)] + Qi + Lk
            dQdt[i] += np.exp(ads_exp) - np.exp(des_exp)

        # --- protein–ion exchange ---
        for j, pj in system.proteins.items():
            nu = pj.nu
            Qj      = logQ[j]
            Cj_log  = logC[j]
            ads_exp = system.ln_k_ads[(i, j)] + nu * Li + Qj
            des_exp = system.ln_k_des[(i, j)] + nu * Qi + Cj_log
            dQdt[i] += np.exp(ads_exp) - np.exp(des_exp)

    # 4) Protein rows: ion–protein exchange
    for j, pj in system.proteins.items():
        Lj = logQ[j]
        nu = pj.nu
        Cj = logC[j]
        for i in system.ions:
            Qbari = logQbar[i]
            Ci    = logC[i]
            ads_exp = system.ln_k_ads[(j, i)] + Cj + nu * Qbari
            des_exp = system.ln_k_des[(j, i)] + Lj + nu * Ci
            dQdt[j] += np.exp(ads_exp) - np.exp(des_exp)

    # 5) Scale by residence time
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