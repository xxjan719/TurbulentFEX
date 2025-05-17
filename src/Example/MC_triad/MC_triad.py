import sys
import os
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",'..')))
from utils.helper import Buu, compute_third_order_moments

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def params_init(case_name = None)->dict:
    # initializing the settings
    params = {
        'MC': int(1e4),  # number of Monte Carlo simulations
        'Dt':1e-3, # Time step size
        'tstep': 10, # output every tstep steps
        'T':10,  # total simulation time
        'Nt': int(round(10 / 1e-3)),  # number of time steps
      }
    if case_name=='equipart': # epipartition of energy
        # System matrices
        params['L'] = np.array([[0, 1, -2], [-1, 0, -3], [2, 3, 0]])
        params['G'] = np.diag([0.2, 0.1, 0.1])
        params['B'] = np.array([1, -0.6, -0.4])
        # Noise settings
        params['req'] = 2.5
        params['SS'] = params['req']*np.sqrt(2*params['G'])
        params['SSt'] = np.zeros((3,3))

        # Time-dependent forcing and noise scaling (tmM and tmS)
        params['tmS'] = np.zeros(params['Nt'])
        params['tmM'] = np.zeros((params['Nt'],3))

        params['namefig'] = 'equipart'
    elif case_name == 'cascade': # energy cascade
        # System matrices
        params['L'] = np.zeros((3,3))
        params['G'] = np.diag([1,2,2])
        params['B'] = np.array([2,-1,-1])
        # Noise settings
        params['SS'] = np.diag([np.sqrt(10), np.sqrt(10**(-2)), np.sqrt(10**(-2))])
        params['SSt'] = np.zeros((3,3))

        # Time-dependent forcing and noise scaling (tmM and tmS)
        params['tmS'] = np.zeros(params['Nt'])
        params['tmM'] = np.zeros((params['Nt'],3))

        params['namefig'] = 'cascade'
    
    elif case_name == 'dual_cascade': # dual energy cascade
        # System matrices
        params['L'] = np.array([[0, 0.03, 0.06], [-0.03,0,-0.09],[-0.06,0.09,0]])
        params['G'] = np.diag([1,2,2])
        params['B'] = np.array([2,-1,-1])
        # Noise settings
        params['SS'] = np.diag([np.sqrt(10**(-2)), np.sqrt(10**(-1)), np.sqrt(10**(-1))])
        params['SSt'] = np.zeros((3,3))

        # Time-dependent forcing and noise scaling (tmM and tmS)
        params['tmS'] = np.zeros(params['Nt'])
        params['tmM'] = np.tile(np.array([0, -1, 1]), (params['Nt'], 1))
        params['namefig'] = 'dual_cascade'

    
    elif case_name == 'periodic_cascade':
        # System matrices
        params['L'] = np.zeros((3,3))
        params['G'] = np.diag([1,2,2])
        params['B'] = np.array([2,-1,-1])
        # Noise settings
        params['SS'] = np.diag([np.sqrt(10), np.sqrt(10**(-2)), np.sqrt(10**(-2))])
        params['SSt'] = np.diag([np.sqrt(1),np.sqrt(2),np.sqrt(2)])

        params['fr'] = 2*np.pi/8 # frequency of the forcing
        j_array = np.arange(params['Nt']) 
        params['tmS'] = np.zeros(params['Nt'])
        sin_wave = np.sin(params['fr']*j_array*params['Dt']) 
        params['tmM'] = np.stack([sin_wave, sin_wave, sin_wave], axis=1) # forcing term
        params['namefig'] = f"periodic_cascade{period:.4f}"
    
    elif case_name == 'random_cascade': # random oscillation between 1-2
        # System matrices
        params['L'] = np.zeros((3,3))
        params['G'] = np.diag([1,2,2])
        params['B'] = np.array([2,-1,-1])
        # Noise settings
        params['SS'] = np.diag([np.sqrt(10), np.sqrt(10**(-2)), np.sqrt(10**(-2))])
        params['SSt'] = np.diag([np.sqrt(1),np.sqrt(2),np.sqrt(2)])

        params['fr'] = 2*np.pi/2 # frequency of the forcing
        theta = params['fr']/(2*np.pi) 
        sigma = np.sqrt(2*theta)

        tmS = np.zeros(params['Nt']+1)
        tmM = np.zeros((params['Nt']+1,3))
        tmt = 0.0

        for j in range(params['Nt']):
            dW1 = np.sqrt(params['Dt']/4)*np.random.randn(4)
            Winc = np.sum(dW1)
            tmS[j+1] = tmS[j]-theta*tmS[j]*params['Dt']+sigma*Winc

            dW1= np.sqrt(params['Dt']/4)*np.random.randn(4)
            Winc = np.sum(dW1)
            tmt = tmt - theta*tmt*params['Dt'] + sigma*Winc
            tmM[j+1,:] = tmt*np.array([1, 1, 1])
        
        params['tmS'] = 0.8*tmS
        params['tmM'] = tmM
        # Figure title
        period = (1/params['fr'])*2*np.pi
        params['namefig'] = f"random_{period:.4f}"
    else:
        raise ValueError(f"Unknown case name: {case_name}")

    return params 







def MC_triad_direct(params, m0, var0, method = 'RK4'):
    MC = params['MC']
    Dt = params['Dt'] # time step size
    tstep = params['tstep']
    Nt = int(round(params['T'] / params['Dt']))  # number of time steps
    L = params['L']
    G = params['G']
    B = params['B']

    # === Initial condition: u ~ N(m0, var0) for each component ===
    u = np.random.normal(loc=m0, scale=np.sqrt(var0), size=(MC, 3))    

    # === Initialize containers ===
    u_all = np.zeros((MC, 3, Nt+1))
    mean_MC_all = np.zeros((3, Nt+1))
    cov_MC_all = np.zeros((3, 3, Nt+1))
    moment3_MC_all = np.zeros((3, 3, 3,Nt+1))
    moment3_MC_norm_all = np.zeros((3, 3, 3,Nt+1))
    Energy_MC_all = np.zeros((4, Nt+1))  # here 0 is all energy. 1,2,3 are energy for each dimension
    Energy_dyn = np.zeros((4, Nt+1)) 
    Energy_update = np.zeros(4)  # here 0 is all energy. 1,2,3 are energy for each dimension


    # === t = 0 statistics ===
    u_all[:, :, 0] = u
    mean_u = np.mean(u, axis=0)
    cov_u = np.cov(u, rowvar=False)
    moment3_MC_all[:, :, :, 0], moment3_MC_norm_all[:,:,:,0] = compute_third_order_moments(u)
    mean_MC_all[:, 0] = mean_u
    cov_MC_all[:, :, 0] = cov_u
    # == Energy at t = 0 ===
    Energy_MC_all[0, 0] = 0.5 * np.sum(mean_u ** 2) + 0.5 * np.trace(cov_u)
    Energy_MC_all[1, 0] = 0.5 * (mean_u[0] ** 2 + cov_u[0, 0])
    Energy_MC_all[2, 0] = 0.5 * (mean_u[1] ** 2 + cov_u[1, 1])
    Energy_MC_all[3, 0] = 0.5 * (mean_u[2] ** 2 + cov_u[2, 2])


    Energy_dyn[:, 0] = Energy_MC_all[:, 0]
    Energy_update[:] = Energy_MC_all[:, 0]

        
    # double_check_energy(mean_MC, cov_MC)
    for i  in range(1,Nt+1):
        print("Time step:", i)
        t = i* Dt
        if method == 'Euler':
            u = u + Dt * ((L @ u.T).T - u @ G + Buu(B, u, u) + np.ones((MC, 1)) * params['tmM'][i - 1, :])
        elif method == 'RK4':
            k1 = (L @ u.T).T - u @ G + Buu(B, u, u) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
            u1 = u + 0.5 * Dt * k1
            k2 = (L @ u1.T).T - u1 @ G + Buu(B, u1, u1) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
            u2 = u + 0.5 * Dt * k2
            k3 = (L @ u2.T).T - u2 @ G + Buu(B, u2, u2) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
            u3 = u + Dt * k3
            k4 = (L @ u3.T).T - u3 @ G + Buu(B, u3, u3) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
            u = u + Dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        else:
            raise ValueError("Unknown method: {}".format(method))
        
        # noise term
        SS = params['SS'] + params['tmS'][i - 1] ** 2 * (params['SSt'] - params['SS'])
        Winc = np.random.randn(MC, 3)  # shape (MC, 3)
        u = u + np.sqrt(Dt) * (Winc @ SS)  # (MC,3) @ (3,3) → (MC,3)
        u_all[:, :, i] = u
        # Energy update
        mean_u = np.mean(u, axis=0)           # shape (3,)
        cov_u = np.cov(u, rowvar=False)       # shape (3, 3)
        diag_G = np.diag(G)
        SS_sq_diag = np.diag(SS @ SS.T)

        #  Full dynamics energy evolution
        Energy_update[0] += Dt * (
            -np.sum(diag_G * (mean_u ** 2 + np.diag(cov_u))) +
             np.sum(params['tmM'][i - 1, :] * mean_u) +
             0.5 * np.sum(SS_sq_diag)
        )
        damp1 = np.max(diag_G)
        damp2 = max(np.min(diag_G), 0)
        damp3 = np.mean(diag_G)
        Energy_update[1] += Dt * (-2 * damp1 * Energy_update[1] + np.sum(params['tmM'][i - 1, :] * mean_u) + 0.5 * np.sum(SS_sq_diag))
        Energy_update[2] += Dt * (-2 * damp2 * Energy_update[2] + np.sum(params['tmM'][i - 1, :] * mean_u) + 0.5 * np.sum(SS_sq_diag))
        Energy_update[3] += Dt * (-2 * damp3 * Energy_update[3] + np.sum(params['tmM'][i - 1, :] * mean_u) + 0.5 * np.sum(SS_sq_diag))

        #  Save every tstep

        if i % tstep == 0:
            mean_MC_all[:, i] = mean_u
            cov_MC_all[:, :, i] = cov_u
            moment3_MC_all[:, :, :, i], moment3_MC_norm_all[:,:,:,i] = compute_third_order_moments(u)
            
            Energy_MC_all[0, i] = 0.5 * np.sum(mean_u ** 2) + 0.5 * np.trace(cov_u)
            Energy_MC_all[1, i] = 0.5 * (mean_u[0] ** 2 + cov_u[0, 0])
            Energy_MC_all[2, i] = 0.5 * (mean_u[1] ** 2 + cov_u[1, 1])
            Energy_MC_all[3, i] = 0.5 * (mean_u[2] ** 2 + cov_u[2, 2])

            Energy_dyn[0,i] = Energy_update[0]
            Energy_dyn[1,i] = Energy_update[1]
            Energy_dyn[2,i] = Energy_update[2]
            Energy_dyn[3,i] = Energy_update[3]
            print(f"MC iter = {i}: E_true = {0.5 * (np.sum(mean_u ** 2) + np.trace(cov_u)):.4f}, E_dyn = {Energy_update[0]:.4f}")
        
    return u_all, mean_MC_all, cov_MC_all, moment3_MC_all, moment3_MC_norm_all, Energy_MC_all, Energy_dyn










if __name__ == "__main__":
    m0 = np.array([-1,0.5,-0.5])
    var0 = np.array([0.52, 0.2, 0.12])
    params = params_init('equipart')
    print(params['SS'])
    # u_all, mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn = MC_triad_direct(params, m0, var0)
