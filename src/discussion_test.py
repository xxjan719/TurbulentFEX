import config
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import torch
import matplotlib.pyplot as plt
from config import DIR_EXAMPLE, create_main_parser
from utils.plot import (
    plot_cross_term_vs_noise,
    plot_cross_term_vs_sample,
    plot_mean_covariance_grid_ind_dep,
    plot_log10_error_mean_covariance_grid_ind_dep,
    plot_third_order_moments_2x2,
    plot_third_order_moments_ind_dep_2x2,
    plot_triad_3d_time_grid_matplotlib_cloud_3x3_times,
    plot_discussion_choice3_composite,
)
from utils.FEX import FEX_model_learned
from utils.FEX_with_force import FEX_with_force_model_learned
from utils.helper import get_coefficients, get_coefficients_from_finak_sample_file
from utils import (
    FN_Net,
    Buu,
    compute_third_order_moments,
    simple_step_update,
    FN_multi_update,
)

from Example.MC_triad.MC_triad import params_init, MC_triad_initial_value
import config

# Hardcode DIR_TRIAD for this test
DIR_TRIAD = "Example/MC_triad"

parser = create_main_parser()
args = parser.parse_args()

# Set device to cpu to avoid torch issues
print(f"Using device: {args.DEVICE}")

if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results','Results1','Results')
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results')

# Some parts of this script historically used a lower-case `device` variable.
# Keep it as an alias for compatibility.
device = DEVICE


    
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'

print("DIR_TRIAD being used:", DIR_TRIAD)
print("args.DEVICE:", args.DEVICE)
print("args.Model:", args.Model)


# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Discussion 2: different noise levels test")
print("2. Discussion 3: different sample sizes test")
print("3. Discussion 4: TIME INDEPENDENT CASE vs TIME DEPENDENT CASE")

while True:
    # choice = '1'  # uncomment for debugging
    choice = input("\nChoose option (1 or 2 or 3): ").strip()
    if choice in ['1', '2', '3']:
        break
    print("Please enter '1' or '2' or '3'.")

if choice == '1':
    print("=" * 60)
    # Get coefficients for all three regimes (equipart, cascade, dual_cascade)
    coefficients_equipart = get_coefficients(
        load_dir="Example/MC_triad", model_name="equipart", DEVICE=DEVICE
    )
    coefficients_cascade = get_coefficients(
        load_dir="Example/MC_triad", model_name="cascade", DEVICE=DEVICE
    )
    coefficients_dual_cascade = get_coefficients(
        load_dir="Example/MC_triad", model_name="dual_cascade", DEVICE=DEVICE
    )

    print("\n=== Coefficient Values by Noise Level (equipart) ===")
    for dim_key, dim_data in coefficients_equipart.items():
        print(f"\n{dim_key}:")
        for term_key, coeff_list in dim_data.items():
            print(f"  {term_key}: {coeff_list}")

    # Create save directory if it doesn't exist
    os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)

    # Common noise levels used across plots
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    # Plot cross-term coefficient errors: left=equipart, center=cascade, right=dual_cascade
    plot_cross_term_vs_noise(
        [coefficients_equipart, coefficients_cascade, coefficients_dual_cascade],
        noise_levels=noise_levels,
        save_dir=args.LOG_SAVE_PATH,
        panel_labels=['Equipart', 'Forward Cascade', 'Dual Cascade'],
    )

    # Optional: plot the original energy-conservation diagnostic
    # plot_energy_conservation(coefficients_equipart, noise_levels=noise_levels, save_dir=args.LOG_SAVE_PATH)

elif choice == '2':
    print("=" * 60)
    # Nonlinear terms graph from FINAK_EXPR_SAMPLE.txt (same plot style, x-axis = sample size)
    # Paths: Results/equipart, Results/cascade, Results/dual_cascade
    results_dir = os.path.join(DIR_EXAMPLE, args.Model, 'Results')
    finak_paths = [
        os.path.join(results_dir, 'equipart', 'FINAK_EXPR_SAMPLE.txt'),
        os.path.join(results_dir, 'cascade', 'FINAK_EXPR_SAMPLE.txt'),
        os.path.join(results_dir, 'dual_cascade', 'FINAK_EXPR_SAMPLE.txt'),
    ]
    coeff_equipart_sample, sample_sizes_equipart = get_coefficients_from_finak_sample_file(finak_paths[0])
    coeff_cascade_sample, sample_sizes_cascade = get_coefficients_from_finak_sample_file(finak_paths[1])
    coeff_dual_sample, sample_sizes_dual = get_coefficients_from_finak_sample_file(finak_paths[2])
    sample_sizes = (
        sample_sizes_equipart
        or sample_sizes_cascade
        or sample_sizes_dual
        or [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )

    plot_cross_term_vs_sample(
        [coeff_equipart_sample, coeff_cascade_sample, coeff_dual_sample],
        sample_sizes=sample_sizes,
        save_dir=results_dir,
        filename='cross_terms_vs_sample.pdf',
        panel_labels=['Equipart', 'Forward Cascade', 'Dual Cascade'],
    )

    # Copy the same nonlinear terms graph into each regime folder
    pdf_src = os.path.join(results_dir, 'cross_terms_vs_sample.pdf')
    for subdir in ('equipart', 'cascade', 'dual_cascade'):
        save_subdir = os.path.join(results_dir, subdir)
        os.makedirs(save_subdir, exist_ok=True)
        if os.path.exists(pdf_src):
            shutil.copy(pdf_src, os.path.join(save_subdir, 'cross_terms_vs_sample.pdf'))
            print(f"Copied nonlinear terms graph to {save_subdir}")

elif choice == '3':
    print("=" * 60)
    print("[INFO] Choice 3: generate test samples (skip training) and plot mean+cov grid")
    print("[INFO] Independent: t=0..20 (from independent run). Dependent: t=0..10 (dependent run only).")
    print("=" * 60)

    m0,var0 = MC_triad_initial_value()
    params = params_init(args.params_name)
    # Choose the correct model based on params_name
    if args.params_name in ['equipart', 'cascade', 'dual_cascade']:
        FEX_model_check = FEX_model_learned
    elif args.params_name in ['periodic_cascade', 'random_cascade_deterministic']:
        FEX_model_check = FEX_with_force_model_learned
    L = params['L']
    G = params['G']
    B = params['B']
    
    TIME_AMOUNT = 20
    dt = 0.01
    NPATH = 5000
    initial_state = np.random.normal(loc=m0, scale=np.sqrt(var0), size=(NPATH, 3))    
    x_pred_initial = torch.ones(NPATH, 3).to(device,dtype=torch.float32) * torch.tensor(m0).to(device,dtype=torch.float32)
    scaler = args.DIFF_SCALE
    
    Nt_eval = int(TIME_AMOUNT / dt)
    # Deterministic forcing/noise scaling must match `params_init()`.
    # Previously these were hard-coded to zero, which can bias mean_state.
    tmM = np.zeros((Nt_eval, 3), dtype=np.float32)
    tmS = np.zeros(Nt_eval, dtype=np.float32)
    if 'tmM' in params and params['tmM'] is not None:
        tmM_src = np.asarray(params['tmM'], dtype=np.float32)
        if tmM_src.shape[0] >= Nt_eval:
            tmM = tmM_src[:Nt_eval, :]
        else:
            reps = int(np.ceil(Nt_eval / tmM_src.shape[0]))
            tmM = np.tile(tmM_src, (reps, 1))[:Nt_eval, :]
    if 'tmS' in params and params['tmS'] is not None:
        tmS_src = np.asarray(params['tmS'], dtype=np.float32)
        if tmS_src.shape[0] >= Nt_eval:
            tmS = tmS_src[:Nt_eval]
        else:
            reps = int(np.ceil(Nt_eval / tmS_src.shape[0]))
            tmS = np.tile(tmS_src, reps)[:Nt_eval]
    mean_state_pred = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record[:, 0] = np.mean(initial_state, axis=0)
    mean_state_pred[:, 0] = np.mean(initial_state, axis=0)

    # Add separate mean arrays for single and ensemble
    mean_state_single = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_single[:, 0] = np.mean(initial_state, axis=0)
    mean_state_ensemble = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_ensemble[:, 0] = np.mean(initial_state, axis=0)

    cov_state_pred = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_pred[:, :, 0] = np.cov(initial_state, rowvar=False)

    # Add separate covariance arrays for single and ensemble
    cov_state_single = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_single[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_ensemble = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_ensemble[:, :, 0] = np.cov(initial_state, rowvar=False)

    u_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_all[:,:,0] = initial_state
    u_pred_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_all[:,:,0] = initial_state

    # Add separate arrays for single and ensemble predictions
    u_pred_single = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_single[:,:,0] = initial_state
    u_pred_ensemble = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_ensemble[:,:,0] = initial_state

    # Dependent (time-dependent) prediction trajectory samples for t<=10.
    u_pred_dependent = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_dependent[:, :, 0] = initial_state

    moment3_state_record = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_state_pred = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_first,_ = compute_third_order_moments(initial_state)
    moment3_state_record[:,:,:,0] = moment3_first
    moment3_state_pred[:,:,:,0] = moment3_first

    Energy_MC_all = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_pred = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)

    current_state = initial_state
    current_pred_state = initial_state

    Energy_update_record = np.zeros(4, dtype=np.float32)
    Energy_update_pred = np.zeros(4, dtype=np.float32)
    Energy_dyn_record = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_dyn_pred = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)

    # At t=0
    Energy_update_pred[:] = [
        0.5 * np.sum(mean_state_pred[:, 0] ** 2) + 0.5 * np.trace(cov_state_pred[:, :, 0]),
        0.5 * (mean_state_pred[0, 0] ** 2 + cov_state_pred[0, 0, 0]),
        0.5 * (mean_state_pred[1, 0] ** 2 + cov_state_pred[1, 1, 0]),
        0.5 * (mean_state_pred[2, 0] ** 2 + cov_state_pred[2, 2, 0]),
    ]
    Energy_dyn_pred[:, 0] = Energy_update_pred

    Energy_update_record[:] = [
        0.5 * np.sum(mean_state_record[:, 0] ** 2) + 0.5 * np.trace(cov_state_record[:, :, 0]),
        0.5 * (mean_state_record[0, 0] ** 2 + cov_state_record[0, 0, 0]),
        0.5 * (mean_state_record[1, 0] ** 2 + cov_state_record[1, 1, 0]),
        0.5 * (mean_state_record[2, 0] ** 2 + cov_state_record[2, 2, 0]),
    ]
    Energy_dyn_record[:, 0] = Energy_update_record

    # -----------------------------
    # Independent/Dependent setup (shared loop over t=0..TIME_AMOUNT)
    # -----------------------------
    Nt_ind = int(TIME_AMOUNT / dt)
    TIME_DEP_AMOUNT = 10.0
    Nt_dep = int(TIME_DEP_AMOUNT / dt)

    # Make explicit independent aliases (the rest of the existing code uses
    # the shorter variable names `mean_state_record`, `mean_state_pred`, etc.)
    mean_state_record_independent = mean_state_record
    cov_state_record_independent = cov_state_record
    mean_state_pred_independent = mean_state_pred
    cov_state_pred_independent = cov_state_pred

    tmM_dep = np.zeros((Nt_dep, 3), dtype=np.float32)
    tmS_dep = np.zeros(Nt_dep, dtype=np.float32)
    if 'tmM' in params and params['tmM'] is not None:
        if params['tmM'].shape[0] >= Nt_dep:
            tmM_dep[:] = params['tmM'][:Nt_dep, :].astype(np.float32)
        else:
            rep = int(np.ceil(Nt_dep / params['tmM'].shape[0]))
            tmM_dep[:] = np.tile(params['tmM'].astype(np.float32), (rep, 1))[:Nt_dep, :]
    if 'tmS' in params and params['tmS'] is not None:
        if params['tmS'].shape[0] >= Nt_dep:
            tmS_dep[:] = params['tmS'][:Nt_dep].astype(np.float32)
        else:
            rep_s = int(np.ceil(Nt_dep / params['tmS'].shape[0]))
            tmS_dep[:] = np.tile(params['tmS'].astype(np.float32), rep_s)[:Nt_dep]

    # Dependent arrays (only meaningful for t <= TIME_DEP_AMOUNT)
    mean_state_record_dependent = np.zeros((3, Nt_dep + 1), dtype=np.float32)
    cov_state_record_dependent = np.zeros((3, 3, Nt_dep + 1), dtype=np.float32)
    mean_state_pred_dependent = np.zeros((3, Nt_dep + 1), dtype=np.float32)
    cov_state_pred_dependent = np.zeros((3, 3, Nt_dep + 1), dtype=np.float32)

    moment3_state_pred_dependent = np.zeros((3, 3, 3, Nt_dep + 1), dtype=np.float32)
    moment3_state_pred_dependent[:, :, :, 0] = moment3_first

    mean_state_record_dependent[:, 0] = np.mean(initial_state, axis=0)
    cov_state_record_dependent[:, :, 0] = np.cov(initial_state, rowvar=False)
    mean_state_pred_dependent[:, 0] = np.mean(initial_state, axis=0)
    cov_state_pred_dependent[:, :, 0] = np.cov(initial_state, rowvar=False)

    current_state_dependent = initial_state.copy()
    current_pred_state_dependent = initial_state.copy()

    # Energy from (mean, covariance) for dependent prediction (only defined/updated for t <= 10)
    Energy_MC_pred_dependent = np.zeros((4, Nt_dep + 1), dtype=np.float32)
    Energy_MC_pred_dependent[:, 0] = [
        0.5 * np.sum(mean_state_pred_dependent[:, 0] ** 2) + 0.5 * np.trace(cov_state_pred_dependent[:, :, 0]),
        0.5 * (mean_state_pred_dependent[0, 0] ** 2 + cov_state_pred_dependent[0, 0, 0]),
        0.5 * (mean_state_pred_dependent[1, 0] ** 2 + cov_state_pred_dependent[1, 1, 0]),
        0.5 * (mean_state_pred_dependent[2, 0] ** 2 + cov_state_pred_dependent[2, 2, 0]),
    ]

    # Load neural network models once at the beginning
    print("Loading neural network models...")
    single_models = {}
    single_norms = {}
    ensemble_models = {}
    ensemble_norms = {}

    # Use args.NOISE_LEVEL so CPU and GPU use same paths and same FEX expressions
    noise_str = f'noise_{args.NOISE_LEVEL}'
    if str(device) == 'cuda:0':
        save_dir = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/{noise_str}/second_stage_10000_constant'
        independent_save_dir = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/{noise_str}/second_stage_10000_independent'
    else:
        save_dir = f'../src/Example/MC_triad/Results/{args.params_name}/{noise_str}/second_stage_10000_constant'
        independent_save_dir = f'../src/Example/MC_triad/Results/{args.params_name}/{noise_str}/second_stage_10000_independent'

    # Dependent (time-dependent) 2nd-stage model directories (indexed by time step)
    # Used only for updating predictions up to `t <= TIME_DEP_AMOUNT`.
    if str(device) == 'cuda:0':
        save_dir_single_dep = (
            f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/{noise_str}/'
            f'deter1000/second_stage_{args.TRAIN_SIZE}_single'
        )
        save_dir_ensemble_dep = (
            f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/{noise_str}/'
            f'deter1000/second_stage_{args.TRAIN_SIZE}'
        )
        alt_ensemble = (
            f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/{noise_str}/'
            f'deter1000/ssecond_stage_{args.TRAIN_SIZE}'
        )
    else:
        save_dir_single_dep = (
            f'../src/Example/MC_triad/Results/{args.params_name}/{noise_str}/'
            f'deter1000/second_stage_{args.TRAIN_SIZE}_single'
        )
        save_dir_ensemble_dep = (
            f'../src/Example/MC_triad/Results/{args.params_name}/{noise_str}/'
            f'deter1000/second_stage_{args.TRAIN_SIZE}'
        )
        alt_ensemble = (
            f'../src/Example/MC_triad/Results/{args.params_name}/{noise_str}/'
            f'deter1000/ssecond_stage_{args.TRAIN_SIZE}'
        )
    if not os.path.exists(save_dir_ensemble_dep) and os.path.exists(alt_ensemble):
        save_dir_ensemble_dep = alt_ensemble
    if not os.path.exists(save_dir_single_dep):
        print(f"[WARNING] Dependent model folder not found: {save_dir_single_dep}")
        print("          Dependent prediction will fall back to simple Gaussian noise.")

    dataname = os.path.join(independent_save_dir,'data_inference.pt')
    data_inference = torch.load(dataname, map_location=device)
    ZT_mean = data_inference['ZT_mean'].to(device)
    ZT_std = data_inference['ZT_std'].to(device)
    ODE_mean = data_inference['ODE_mean'].to(device)
    ODE_std = data_inference['ODE_std'].to(device)
    # Use scale from training (diff_scale) for stoch_update, not current args.DIFF_SCALE
    diff_scale = data_inference['diff_scale']
    if torch.is_tensor(diff_scale):
        diff_scale = diff_scale.item()
    # Load NN once (same as time_dependent loads models per step from disk; here we have a single model)
    Neural_Network = FN_Net(3, 3, 50).to(device)
    Neural_Network.load_state_dict(torch.load(os.path.join(save_dir, 'Neural_Network.pth'), map_location=device))
    Neural_Network.eval()
    tM = np.zeros((int(TIME_AMOUNT/dt),3), dtype=np.float32)
    for idx in range(1, Nt_ind + 1):
        # RK4 integration
        k1 = (L @ current_state.T).T - current_state @ G + Buu(B, current_state, current_state) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        u1 = current_state + dt * k1
        k2 = (L @ u1.T).T - u1 @ G + Buu(B, u1, u1) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        next_state = current_state + dt * (k1 + k2) / 2
        SS = params['SS'] + tmS[idx - 1] ** 2 * (params['SSt'] - params['SS'])
        Winc = np.random.randn(NPATH, 3)  # shape (MC, 3)
        next_state = next_state + np.sqrt(dt) * (Winc @ SS)  # (MC,3) @ (3,3) → (MC,3)
        
        # Dependent (time-dependent) ground truth update for t <= 10
        if idx <= Nt_dep:
            k1_dep = (
                (L @ current_state_dependent.T).T
                - current_state_dependent @ G
                + Buu(B, current_state_dependent, current_state_dependent)
                + np.ones((NPATH, 1)) * tmM_dep[idx - 1, :]
            )
            u1_dep = current_state_dependent + dt * k1_dep
            k2_dep = (
                (L @ u1_dep.T).T
                - u1_dep @ G
                + Buu(B, u1_dep, u1_dep)
                + np.ones((NPATH, 1)) * tmM_dep[idx - 1, :]
            )
            next_state_dependent = current_state_dependent + dt * (k1_dep + k2_dep) / 2

            SS_dep_step = params['SS'] + tmS_dep[idx - 1] ** 2 * (params['SSt'] - params['SS'])
            next_state_dependent = next_state_dependent + np.sqrt(dt) * (Winc @ SS_dep_step) * args.NOISE_LEVEL

            mean_state_record_dependent[:, idx] = np.mean(next_state_dependent, axis=0)
            cov_state_record_dependent[:, :, idx] = np.cov(next_state_dependent, rowvar=False)
            current_state_dependent = next_state_dependent
        u_all[:, :, idx] = next_state

    
        mean_state_record[:,idx] = np.mean(next_state, axis=0)
        cov_state_record[:,:,idx] = np.cov(next_state, rowvar=False)
        moment3_state_record[:,:,:,idx],_ = compute_third_order_moments(next_state)
        Energy_MC_all[0, idx] = 0.5 * np.sum(mean_state_record[:,idx] ** 2) + 0.5 * np.trace(cov_state_record[:,:,idx])
        Energy_MC_all[1, idx] = 0.5 * (mean_state_record[0,idx] ** 2 + cov_state_record[0,0,idx])
        Energy_MC_all[2, idx] = 0.5 * (mean_state_record[1,idx] ** 2 + cov_state_record[1,1,idx])
        Energy_MC_all[3, idx] = 0.5 * (mean_state_record[2,idx] ** 2 + cov_state_record[2,2,idx])
        
      
        diag_G = np.diag(G)
        damp1 = np.max(diag_G)
        damp2 = max(np.min(diag_G), 0)
        damp3 = np.mean(diag_G)
        SS_sq_diag = np.diag(SS @ SS.T)
        
      
        Energy_update_record[0] += dt * (
            -np.sum(diag_G * (mean_state_record[:, idx] ** 2 + np.diag(cov_state_record[:, :, idx]))) +
             np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) +
             0.5 * np.sum(SS_sq_diag)
        )
        Energy_update_record[1] += dt * (-2 * damp1 * Energy_update_record[1] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
        Energy_update_record[2] += dt * (-2 * damp2 * Energy_update_record[2] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
        Energy_update_record[3] += dt * (-2 * damp3 * Energy_update_record[3] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
    
        # u_pred_all[:,:,idx] = current_pred_state
        current_state = next_state

        current_tensor = torch.tensor(current_pred_state, dtype=torch.float32).to(device)
    
        # RK4 for the deterministic part (FEX model)
        # Step 1 / Step 2: for periodic/random cascades, `FEX_with_force_model_learned`
        # expects an extra time column as the last feature.
        if args.params_name in ['periodic_cascade', 'random_cascade_deterministic']:
            current_time = idx * dt
            time_column = torch.full(
                (current_tensor.shape[0], 1),
                current_time,
                dtype=torch.float32,
                device=DEVICE,
            )
            current_tensor_with_time = torch.cat([current_tensor, time_column], dim=1)

            k1_det = FEX_model_check(
                current_tensor_with_time,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            k1_det_np = k1_det.cpu().detach().numpy()
            u1 = current_tensor + k1_det

            u1_with_time = torch.cat([u1, time_column], dim=1)
            k2_det = FEX_model_check(
                u1_with_time,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            k2_det_np = k2_det.cpu().detach().numpy()
        else:
            k1_det = FEX_model_check(
                current_tensor,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            k1_det_np = k1_det.cpu().detach().numpy()
            u1 = current_tensor + k1_det

            k2_det = FEX_model_check(
                u1,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            k2_det_np = k2_det.cpu().detach().numpy()

        # Final RK4 update for deterministic part
        det_update = (k1_det_np + k2_det_np) / 2
    
        # Generate stochastic component (just once per step)
        # NN outputs normalized ODE; denormalize (pred = NN*ODE_std + ODE_mean) then divide by diff_scale
        # to match training (residuals were scaled by scaler before ODE solver; use saved diff_scale).
        Npath = current_pred_state.shape[0]
        dim = current_pred_state.shape[1]
        Winc_tensor_raw = torch.tensor(Winc, dtype=torch.float32).to(device)
        Winc_tensor = (Winc_tensor_raw - ZT_mean) / ZT_std
        with torch.no_grad():
            pred = Neural_Network(Winc_tensor) * ODE_std + ODE_mean
            stoch_update = (pred / diff_scale).cpu().detach().numpy()
    
        # Simple noise for comparison (and optional rescaling reference)
        simple_noise = np.sqrt(dt) * (Winc @ SS)
        # Rescale NN output to match simple_noise scale so result is similar to simple noise.
        # Training target (ODE_Solution) is from long ODE integration, not one-step increment, so NN std is often smaller.
        std_nn = np.std(stoch_update, axis=0)
        std_simple = np.std(simple_noise, axis=0)
        std_simple = np.maximum(std_simple, 1e-12)

        # Match both mean and std to the "simple noise" reference.
        # Without mean correction, NN-predicted stochastic increments can have a bias,
        # which shifts the independent mean trajectory.
        mean_nn = np.mean(stoch_update, axis=0)
        mean_simple = np.mean(simple_noise, axis=0)

        scale_match = np.where(std_nn > 1e-12, std_simple / std_nn, 1.0)
        stoch_update = (stoch_update - mean_nn) * scale_match + mean_simple

        # Dependent (time-dependent) stochastic update for t <= 10
        stoch_update_dependent_current = None
        if idx <= Nt_dep:
            stoch_update_dependent = None
            # Recompute SS_dep_step for consistent SS scaling in prediction.
            SS_dep_step = params['SS'] + tmS_dep[idx - 1] ** 2 * (params['SSt'] - params['SS'])
            simple_noise_dependent = np.sqrt(dt) * (Winc @ SS_dep_step) * args.NOISE_LEVEL

            if os.path.exists(save_dir_single_dep):
                if args.params_name in ['equipart', 'cascade']:
                    stoch_update_dependent = simple_step_update(
                        Winc_tensor=Winc_tensor_raw,
                        device=DEVICE,
                        idx=idx,
                        save_dir_single=save_dir_single_dep,
                        save_dir_ensemble=save_dir_ensemble_dep,
                        model_type='single',
                        dim=3,
                        scaler=scaler,
                    )
                else:
                    stoch_update_dependent = FN_multi_update(
                        Winc_tensor=Winc_tensor_raw,
                        device=DEVICE,
                        idx=idx,
                        save_dir_single=save_dir_single_dep,
                        dim=3,
                        scaler=scaler,
                    )

            if stoch_update_dependent is None:
                stoch_update_dependent = simple_noise_dependent

            stoch_update_dependent_current = stoch_update_dependent

            # Deterministic RK4 update for the dependent predictor (must be computed
            # from `current_pred_state_dependent`, same as in `2stage_stochastic_time_dependent.py`).
            current_tensor_dep = torch.tensor(
                current_pred_state_dependent, dtype=torch.float32
            ).to(device)
            if args.params_name in ['periodic_cascade', 'random_cascade_deterministic']:
                current_time = idx * dt
                time_column_dep = torch.full(
                    (current_tensor_dep.shape[0], 1),
                    current_time,
                    dtype=torch.float32,
                ).to(device)
                current_tensor_with_time_dep = torch.cat(
                    [current_tensor_dep, time_column_dep],
                    dim=1,
                )
                k1_det_dep = (
                    FEX_model_check(
                        current_tensor_with_time_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k1_det_dep_np = k1_det_dep.cpu().detach().numpy()
                u1_dep = current_tensor_dep + k1_det_dep

                u1_dep_with_time = torch.cat([u1_dep, time_column_dep], dim=1)
                k2_det_dep = (
                    FEX_model_check(
                        u1_dep_with_time,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k2_det_dep_np = k2_det_dep.cpu().detach().numpy()
            else:
                k1_det_dep = (
                    FEX_model_check(
                        current_tensor_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k1_det_dep_np = k1_det_dep.cpu().detach().numpy()
                u1_dep = current_tensor_dep + k1_det_dep

                k2_det_dep = (
                    FEX_model_check(
                        u1_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k2_det_dep_np = k2_det_dep.cpu().detach().numpy()

            det_update_dep = (k1_det_dep_np + k2_det_dep_np) / 2.0
            next_pred_state_dependent = (
                current_pred_state_dependent + det_update_dep + stoch_update_dependent
            )
            # Store dependent prediction samples for 3D phase-space clouds (t <= 10).
            u_pred_dependent[:, :, idx] = next_pred_state_dependent
            current_pred_state_dependent = next_pred_state_dependent
            mean_state_pred_dependent[:, idx] = np.mean(next_pred_state_dependent, axis=0)
            cov_state_pred_dependent[:, :, idx] = np.cov(next_pred_state_dependent, rowvar=False)

            # Third-order moments for dependent prediction
            moment3_pred_dep, _ = compute_third_order_moments(next_pred_state_dependent)
            moment3_state_pred_dependent[:, :, :, idx] = moment3_pred_dep

            # Dependent prediction energy (optional but kept consistent with mean/cov definition)
            Energy_MC_pred_dependent[0, idx] = (
                0.5 * np.sum(mean_state_pred_dependent[:, idx] ** 2)
                + 0.5 * np.trace(cov_state_pred_dependent[:, :, idx])
            )
            Energy_MC_pred_dependent[1, idx] = 0.5 * (
                mean_state_pred_dependent[0, idx] ** 2 + cov_state_pred_dependent[0, 0, idx]
            )
            Energy_MC_pred_dependent[2, idx] = 0.5 * (
                mean_state_pred_dependent[1, idx] ** 2 + cov_state_pred_dependent[1, 1, idx]
            )
            Energy_MC_pred_dependent[3, idx] = 0.5 * (
                mean_state_pred_dependent[2, idx] ** 2 + cov_state_pred_dependent[2, 2, idx]
            )
    
        # Print comparison every 50 steps
        if idx % 50 == 0 or idx in (Nt_dep, Nt_dep + 1):
            t_now = idx * dt
            print(f"\nStep {idx} (t={t_now:.2f}): Single NN stochastic update stats")
            print("=" * 70)

            dep_str = "not available (t > 10)" if idx > Nt_dep else ""
            if idx <= Nt_dep and stoch_update_dependent_current is not None:
                print(f"single NN dependent mean: {np.mean(stoch_update_dependent_current, axis=0)}")
                print(f"single NN dependent std : {np.std(stoch_update_dependent_current, axis=0)}")
            else:
                print(f"single NN dependent mean: {dep_str}")
                print(f"single NN dependent std : {dep_str}")

            # Independent is always available for all t in this loop.
            if stoch_update is not None:
                print(f"single NN independent mean: {np.mean(stoch_update, axis=0)}")
                print(f"single NN independent std : {np.std(stoch_update, axis=0)}")
            else:
                print("single NN independent mean: not available")
                print("single NN independent std : not available")
            print("=" * 70)
    
        
    
        # Compute both single and ensemble predictions
        if stoch_update is not None:
            next_pred_single = current_pred_state + det_update + stoch_update
        else:
            next_pred_single = current_pred_state + det_update + simple_noise
        
        
    
        # Use the selected model for the main prediction (for backward compatibility)
        next_pred_state = current_pred_state + det_update + stoch_update
    
        # Store results for all three predictions
        u_pred_all[:,:,idx] = next_pred_state
        u_pred_single[:,:,idx] = next_pred_single
    
        # Update statistics for all three predictions
        mean_state_pred[:,idx] = np.mean(next_pred_state, axis=0)
        mean_state_single[:,idx] = np.mean(next_pred_single, axis=0)
       
    
        cov_state_pred[:,:,idx] = np.cov(next_pred_state, rowvar=False)
        cov_state_single[:,:,idx] = np.cov(next_pred_single, rowvar=False)

        # Calculate energy directly from mean and covariance (same as ground truth)
        Energy_MC_pred[0, idx] = 0.5 * np.sum(mean_state_pred[:, idx] ** 2) + 0.5 * np.trace(cov_state_pred[:, :, idx])
        Energy_MC_pred[1, idx] = 0.5 * (mean_state_pred[0, idx] ** 2 + cov_state_pred[0, 0, idx])
        Energy_MC_pred[2, idx] = 0.5 * (mean_state_pred[1, idx] ** 2 + cov_state_pred[1, 1, idx])
        Energy_MC_pred[3, idx] = 0.5 * (mean_state_pred[2, idx] ** 2 + cov_state_pred[2, 2, idx])
    
        # Calculate third-order moments for prediction
        moment3_pred, _ = compute_third_order_moments(next_pred_state)
        moment3_state_pred[:, :, :, idx] = moment3_pred
    
        # Update current state
        current_pred_state = next_pred_state
    
    np.random.seed(0)
    # Physical time 0 to TIME_AMOUNT (e.g. 0 to 50)
    Time_record = np.arange(int(TIME_AMOUNT/dt)+1) * dt

    Time_dep = np.arange(Nt_dep + 1) * dt

    # Plot: independent (t up to 20) vs dependent (t up to 10)
    os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)
    # save_path = os.path.join(args.LOG_SAVE_PATH, 'mean_covariance_grid_ind_vs_dep.pdf')
    # plot_mean_covariance_grid_ind_dep(
    #     Time_ind=Time_record,
    #     mean_gt_ind=mean_state_record_independent,
    #     cov_gt_ind=cov_state_record_independent,
    #     mean_pred_ind=mean_state_pred_independent,
    #     cov_pred_ind=cov_state_pred_independent,
    #     Time_dep=Time_dep,
    #     mean_pred_dep=mean_state_pred_dependent,
    #     cov_pred_dep=cov_state_pred_dependent,
    #     t_ind_max=TIME_AMOUNT,
    #     t_dep_max=TIME_DEP_AMOUNT,
    #     ground_truth_label="Ground Truth",
    #     dependent_label="ASD-FEX-TFDM-dependent",
    #     legend_title=args.params_name.capitalize(),
    #     save_path=save_path,
    # )

    # # Log10 absolute error grid (independent vs dependent), no ground truth curve
    # save_path_err = os.path.join(
    #     args.LOG_SAVE_PATH, "mean_covariance_log10_error_ind_vs_dep.pdf"
    # )
    # plot_log10_error_mean_covariance_grid_ind_dep(
    #     Time_ind=Time_record,
    #     mean_gt_ind=mean_state_record_independent,
    #     cov_gt_ind=cov_state_record_independent,
    #     mean_pred_ind=mean_state_pred_independent,
    #     cov_pred_ind=cov_state_pred_independent,
    #     Time_dep=Time_dep,
    #     mean_pred_dep=mean_state_pred_dependent,
    #     cov_pred_dep=cov_state_pred_dependent,
    #     t_ind_max=TIME_AMOUNT,
    #     t_dep_max=TIME_DEP_AMOUNT,
    #     independent_label="Independent error",
    #     dependent_label="Dependent error",
    #     legend_title=None,
    #     save_path=save_path_err,
    # )

    # # Third-order moments in 2x2 grid (independent + dependent)
    # save_path_mom3 = os.path.join(
    #     args.LOG_SAVE_PATH, "third_order_moments_over_time_ind_vs_dep_2x2.pdf"
    # )
    # plot_third_order_moments_ind_dep_2x2(
    #     moment3_state_record_ind=moment3_state_record,
    #     moment3_state_pred_ind=moment3_state_pred,
    #     moment3_state_pred_dep=moment3_state_pred_dependent,
    #     Time_ind=Time_record,
    #     Time_dep=Time_dep,
    #     save_path=save_path_mom3,
    #     title_suffix=args.params_name.capitalize(),
    #     legend_title=args.params_name.capitalize(),
    #     ground_truth_label="Ground Truth",
    #     independent_label="ASD-FEX-TFDM-independent",
    #     dependent_label="ASD-FEX-TFDM-dependent",
    # )

    # Composite figure: t0_20 (ind), t0_10_dep (dep), mean_cov, third_order;
    # legend center header, Equipart center bottom, font 20; orange box "independent", blue box "dependent".
    save_path_composite = os.path.join(
        args.LOG_SAVE_PATH, "discussion_choice3_composite.pdf"
    )
    plot_discussion_choice3_composite(
        save_path=save_path_composite,
        u_all=u_all,
        u_pred_all=u_pred_all,
        u_pred_dependent=u_pred_dependent,
        dt=dt,
        Time_record=Time_record,
        Time_dep=Time_dep,
        mean_state_record_independent=mean_state_record_independent,
        cov_state_record_independent=cov_state_record_independent,
        mean_state_pred_independent=mean_state_pred_independent,
        cov_state_pred_independent=cov_state_pred_independent,
        mean_state_pred_dependent=mean_state_pred_dependent,
        cov_state_pred_dependent=cov_state_pred_dependent,
        moment3_state_record=moment3_state_record,
        moment3_state_pred=moment3_state_pred,
        moment3_state_pred_dependent=moment3_state_pred_dependent,
        TIME_AMOUNT=TIME_AMOUNT,
        TIME_DEP_AMOUNT=TIME_DEP_AMOUNT,
        params_name=args.params_name.capitalize(),
        font_size=20,
    )

    # (Individual plots commented out; use composite above.)
    # save_path_3d_0_10 = os.path.join(args.LOG_SAVE_PATH, "triad_3d_clouds_t0_10_gt_ind_dep.pdf")
    # save_path_3d_0_10_dep = os.path.join(args.LOG_SAVE_PATH, "triad_3d_clouds_t0_10_dep.pdf")
    # save_path_3d_0_20 = os.path.join(args.LOG_SAVE_PATH, "triad_3d_clouds_t0_20_gt_ind.pdf")
    # plot_triad_3d_time_grid_matplotlib_cloud_3x3_times(...)
    # plot_triad_3d_time_grid_matplotlib_cloud_3x3_times(..., time_points=[0,4,8], grid_rows=1, grid_cols=3, ...)

