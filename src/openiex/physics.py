import numpy as np
from scipy.special import logsumexp

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

def calc_lnQstar(C, Qbar, system, eps=1e-15):
    """
    Compute ln(Q*_{j,i}(z)) for all proteins j and ions i, vectorized over z.

    Q*_{j,i}(z) = K_eq[(j,i)] * Qbar[i,z]^nu_j * C[j,z] / C[i,z]^nu_j
    We return lnQstar[j][i] = log(Q*_{j,i}(z)), clipped to avoid overflow.
    """
    Nz = system.config.Nz
    lnQstar = {j: {} for j in system.proteins}
    
    for j, prot in system.proteins.items():
        nu = prot.nu
        # for each ion, compute ln(Q*)
        for i in system.ions:
            # logs of each term
            lnK   = np.log(system.K_eq[(j, i)])
            lnQb  = nu * np.log(Qbar[i] + eps)
            lnCj  = np.log(C[j] + eps)
            lnCi  = nu * np.log(C[i] + eps)
            
            # ln(Q*) = lnK + lnQb + lnCj - lnCi
            ln_val = lnK + lnQb + lnCj - lnCi
            
            # clip to safe range
            ln_val = np.clip(ln_val, -700, 700)
            
            lnQstar[j][i] = ln_val
    
    return lnQstar

def calc_dQdt(C, Q, Qbar, lnQstar, feed, system):
    """
    Compute dQ/dt using:
      - SMA mass-action for ion–ion
      - SMA or LDF for protein–ion, using lnQstar for weights
    """
    Nz = system.config.Nz
    _, flow_rate = feed
    t_res = system.config.vol_interstitial / flow_rate
    dQdt = {s: np.zeros(Nz) for s in system.species}
    ion_list = list(system.ions.keys())

    for z in range(Nz):
        # 1) protein–ion
        rji = {j: {} for j in system.proteins}
        for j, prot in system.proteins.items():
            nu = prot.nu
            # build per-z log-vector for this j
            ln_vals = np.array([lnQstar[j][i][z] for i in ion_list])
            # log-sum-exp for normalization
            ln_denom = logsumexp(ln_vals)
            sum_r = 0.0

            for idx, i in enumerate(ion_list):
                model = system.pair_kinetic_model.get((j, i), system.kinetic_model)
                if model == "SMA":
                    r = (
                        system.k_ads[(j, i)] * C[j][z] * Qbar[i][z]**nu
                        - system.k_des[(j, i)] * Q[j][z] * C[i][z]**nu
                    )
                else:
                    # LDF: weight from ln-space
                    w = np.exp(ln_vals[idx] - ln_denom)
                    Qji = w * Q[j][z]
                    Qstar_val = np.exp(ln_vals[idx])
                    r = system.k_ldf[(j, i)] * (Qstar_val - Qji)

                rji[j][i] = r
                sum_r += r

            dQdt[j][z] = sum_r / t_res

        # 2) ion–ion + protein sink
        for i in ion_list:
            acc = 0.0
            # ion–ion SMA only
            for k in ion_list:
                if k == i:
                    continue
                acc += (
                    system.k_ads[(i, k)] * C[i][z] * Q[k][z]
                    - system.k_des[(i, k)] * Q[i][z] * C[k][z]
                )
            # subtract protein displacement
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