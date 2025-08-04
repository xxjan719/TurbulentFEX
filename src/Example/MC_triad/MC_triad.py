import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import random
import torch
from pathlib import Path
import os
import sympy as sp
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",'..')))
from utils.helper import Buu, compute_third_order_moments
from utils.FEX import FEX
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def params_init(case_name = None,
               sample:int=10000)->dict:
    # initializing the settings
    params = {
        'MC': sample,  # number of Monte Carlo simulations (50,000)
        'Dt':1e-2, # Time step size
        'tstep': 10, # output every tstep steps
        'T':10,  # total simulation time
        'Nt': int(round(10 / 1e-2)),  # number of time steps
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







def MC_triad_direct(params, m0, var0, method = 'RK4', noise_level = 1.0):
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
        print(SS)
        Winc = np.random.randn(MC, 3)  # shape (MC, 3)
        u = u + np.sqrt(Dt) * noise_level * (Winc @ SS)  # (MC,3) @ (3,3) → (MC,3)
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


def MC_triad_initial_value():
    m0 = np.array([-1,0.5,-0.5])
    var0 = np.array([0.52, 0.2, 0.12])
    return m0, var0



def get_matrix_coefficients_from_FEX(path):
    """
    Extract matrix coefficients from FEX models for different dimensions and construct matrices.
    
    Args:
        path (Path): Path to the directory containing saved models
        
    Returns:
        tuple: (L, G, B) matrices where:
            - L: 3x3 matrix for linear terms (x1,x2,x3)
            - G: 3x3 diagonal matrix for damping terms
            - B: 3-element vector for quadratic terms (x2x3,x1x3,x1x2)
    """
    path = Path(path)
    # Load operator sequences
    op_seq_file_1 = np.load(os.path.join(path, 'optimal_idx_1.npy'))
    op_seq_file_2 = np.load(os.path.join(path, 'optimal_idx_2.npy'))
    op_seq_file_3 = np.load(os.path.join(path, 'optimal_idx_3.npy'))
    
    # Initialize matrices
    L = np.zeros((3, 3))  # Linear coefficients
    G = np.zeros(3)       # Diagonal damping terms
    B = np.zeros(3)       # Quadratic interaction terms vector
    
    # Variables to track coefficients
    coeffs = {
        'x1': np.zeros(3),
        'x2': np.zeros(3),
        'x3': np.zeros(3),
        'x1x2': 0.0,
        'x2x3': 0.0,
        'x1x3': 0.0
    }
    
    for dim in range(1, 4):  # dimensions 1, 2, 3
        model_file = path / f'FEX_dim_{dim}.pth'
        if model_file.exists():
            print(f"\nProcessing dimension {dim}:")
            if dim == 1:
                op_seq = op_seq_file_1
            elif dim == 2:
                op_seq = op_seq_file_2
            elif dim == 3:
                op_seq = op_seq_file_3
            else:
                raise ValueError(f"Unknown dimension: {dim}")
            # Create FEX model with dim=3 since all models were trained with 3D data
            FEX_model = FEX(torch.tensor(op_seq), dim=3)
            FEX_model.load_state_dict(torch.load(str(model_file), weights_only=True))
            expr = FEX_model.expression_visualize_simplified()
            print(f"Expression for dimension {dim}:")
            print(expr)
            
            # Convert expression to sympy and expand
            expr_sympy = sp.expand(expr)
            print(f"Expanded expression:")
            print(expr_sympy)
            
            # Create symbolic variables
            x1, x2, x3 = sp.symbols('x1 x2 x3')
            
            def get_numeric_coeff(expr, term):
                """Helper function to safely extract numeric coefficient"""
                try:
                    # Get the coefficient dictionary
                    coeff_dict = expr.as_coefficients_dict()
                    # Look for the term in the dictionary
                    if term in coeff_dict:
                        return float(coeff_dict[term].evalf())
                    return 0.0
                except:
                    return 0.0
            
            # Extract coefficients for linear terms
            if dim == 1:  # Looking for x2, x3, x2*x3
                coeffs['x2'][0] = get_numeric_coeff(expr_sympy, x2)
                coeffs['x3'][0] = get_numeric_coeff(expr_sympy, x3)
                coeffs['x2x3'] = get_numeric_coeff(expr_sympy, x2*x3)
                G[0] = -get_numeric_coeff(expr_sympy, x1)  # Damping term for x1
                print(f"Dim 1 coefficients: x2={coeffs['x2'][0]}, x3={coeffs['x3'][0]}, x2x3={coeffs['x2x3']}, G[0]={G[0]}")
                
            elif dim == 2:  # Looking for x1, x3, x1*x3
                coeffs['x1'][1] = get_numeric_coeff(expr_sympy, x1)
                coeffs['x3'][1] = get_numeric_coeff(expr_sympy, x3)
                coeffs['x1x3'] = get_numeric_coeff(expr_sympy, x1*x3)
                G[1] = -get_numeric_coeff(expr_sympy, x2)  # Damping term for x2
                print(f"Dim 2 coefficients: x1={coeffs['x1'][1]}, x3={coeffs['x3'][1]}, x1x3={coeffs['x1x3']}, G[1]={G[1]}")
                
            elif dim == 3:  # Looking for x1, x2, x1*x2
                coeffs['x1'][2] = get_numeric_coeff(expr_sympy, x1)
                coeffs['x2'][2] = get_numeric_coeff(expr_sympy, x2)
                coeffs['x1x2'] = get_numeric_coeff(expr_sympy, x1*x2)
                G[2] = -get_numeric_coeff(expr_sympy, x3)  # Damping term for x3
                print(f"Dim 3 coefficients: x1={coeffs['x1'][2]}, x2={coeffs['x2'][2]}, x1x2={coeffs['x1x2']}, G[2]={G[2]}")
    
    # Construct L matrix (linear terms)
    # For dim 1: x2, x3 coefficients go to first row
    # For dim 2: x1, x3 coefficients go to second row
    # For dim 3: x1, x2 coefficients go to third row
    L[0, 1] = coeffs['x2'][0]  # x2 coefficient in first equation
    L[0, 2] = coeffs['x3'][0]  # x3 coefficient in first equation
    L[1, 0] = coeffs['x1'][1]  # x1 coefficient in second equation
    L[1, 2] = coeffs['x3'][1]  # x3 coefficient in second equation
    L[2, 0] = coeffs['x1'][2]  # x1 coefficient in third equation
    L[2, 1] = coeffs['x2'][2]  # x2 coefficient in third equation
    
    # Construct B vector (quadratic terms)
    B[0] = coeffs['x2x3']  # x2x3 term in first equation
    B[1] = coeffs['x1x3']  # x1x3 term in second equation
    B[2] = coeffs['x1x2']  # x1x2 term in third equation
    
    print("\nExtracted Matrices:")
    print("\nL matrix (linear terms):")
    print(np.array2string(L, precision=4, suppress_small=True))
    print("\nG vector (damping terms):")
    print(np.array2string(G, precision=4, suppress_small=True))
    print("\nB vector (quadratic terms):")
    print(np.array2string(B, precision=4, suppress_small=True))
    
    return L, G, B


def plot_latex_formula(params, path):
    # Configure LaTeX rendering
    plt.rcParams.update({
         "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
    })

    # Extract parameters
    L = params['L']
    G = params['G']
    B = params['B']

    L_learned, G_learned, B_learned = get_matrix_coefficients_from_FEX(path)

    # Extract individual values from matrices
    print(L)
    L1 = L[2, 0]  # u_2 -> u_3
    L2 = L[0, 2]  # u_3 -> u_1
    L3 = L[0, 1]  # u_1 -> u_2
    d1 = G[0, 0]
    d2 = G[1, 1]
    d3 = G[2, 2]
    B1, B2, B3 = B[0], B[1], B[2]

    d1_learned = G_learned[0]
    d2_learned = G_learned[1]
    d3_learned = G_learned[2]
    B1_learned, B2_learned, B3_learned = B_learned[0], B_learned[1], B_learned[2]

    # Define LaTeX equations with formatted coefficients
    ground_truth = (
        r"\textbf{Ground Truth:}\\[0.3em]"
        r"\begin{aligned}"
        fr"\frac{{du_1}}{{dt}} &= {L2}u_3 - {L3}u_2 - {d1:.3f}u_1 + {B1:.3f}u_2 u_3 + F_1 + \text{{noise}} \\"
        fr"\frac{{du_2}}{{dt}} &= {L[1,0]}u_1 - {-L[1,2]}u_3 - {d2:.3f}u_2 + {B2:.3f}u_3 u_1 + F_2 + \text{{noise}} \\"
        fr"\frac{{du_3}}{{dt}} &= {L[2,1]}u_2 - {L[2,0]}u_1 - {d3:.3f}u_3 + {B3:.3f}u_1 u_2 + F_3 + \text{{noise}}"
        r"\end{aligned}"
    )

    FEX_expression = (
        r"\textbf{FEX:}\\[0.3em]"
        r"\begin{aligned}"
        fr"FEX_1 &= {L_learned[0,2]}u_3 - {L_learned[0,1]}u_2 - {d1_learned:.3f}u_1 + {B1_learned:.3f}u_2 u_3 + \text{{residual term}} \\"
        fr"FEX_2 &= {L_learned[1,0]}u_1 - {-L_learned[1,2]}u_3 - {d2_learned:.3f}u_2 + {B2_learned:.3f}u_3 u_1 + \text{{residual term}} \\"
        fr"FEX_3 &= {L_learned[2,1]}u_2 - {L_learned[2,0]}u_1 - {d3_learned:.3f}u_3 + {B3_learned:.3f}u_1 u_2+ \text{{residual term}}"
        r"\end{aligned}"
    )

    # Combine all lines into a LaTeX aligned environment
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].text(0.05, 0.5, f"${ground_truth}$", fontsize=14, va='center', ha='left')
    axs[0].axis('off')

    axs[1].text(0.05, 0.5, f"${FEX_expression}$", fontsize=14, va='center', ha='left')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()






if __name__ == "__main__":
    
    #model_path = Path(os.path.join(os.path.dirname(__file__), 'Results', 'equipart'))
    # Get matrix coefficients for each dimension
    # get_matrix_coefficients_from_FEX(model_path)
    #params = params_init('equipart')
    #plot_latex_formula(params, model_path)
    # #print("Matrix coefficients extraction completed.")
    noise_level = 2.0
    m0, var0 = MC_triad_initial_value()
    params = params_init('equipart')
    data_save_path = Path(os.path.join(os.path.dirname(__file__), 'Results', 'equipart', f'noise_{noise_level}',f'simulation_results_noise_{noise_level}.npz'))
    dataset, mean_MC, cov_MC, moment3_MC, moment3_MC_norm,Energy_MC, Energy_dyn = MC_triad_direct(params, m0, var0, noise_level=noise_level)
    print(dataset.shape)
    np.savez(
    data_save_path,
    dataset=dataset,
    mean_MC=mean_MC,
    cov_MC=cov_MC,
    moment3_MC=moment3_MC,
    moment3_MC_norm=moment3_MC_norm,
    Energy_MC=Energy_MC,
    Energy_dyn=Energy_dyn
    )