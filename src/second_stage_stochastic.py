import numpy as np
import os
import sys
from pathlib import Path
# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from utils.ODEParser import (
    generate_rk4_residue, 
    generate_second_step, 
    generate_mean_and_std, 
    train_FN_ensemble,
    FEX_model_check,
    predict_ensemble_residual_covariance
)

from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
import config

# Import specific functions from ODE Parser
args = config.parse_args()
torch.manual_seed(args.SEED)
np.random.seed(args.SEED)

# Set device
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    device = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU instead")

#===========================Path part==============================================
print("\n"+ "="*60)
print("\n[INFO] Setting up the path...")
if args.Model == 'MC_triad':
    model_PATH = config.DIR_EQUIPART
    if os.path.exists(model_PATH):
        save_dir = os.path.join( model_PATH,f'case_{args.NUM_SAMPLES}')
        print('[INFO] Right now we use our own workspace path.') 
    else:
        model_PATH = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results', 'equipart'))
        save_dir = os.path.join(config.DIR_TRIAD,'Results', 'Results','equipart',f'case_{args.NUM_SAMPLES}')
        print('[INFO] Right now we use hipergator workspace path.')
os.makedirs(save_dir,exist_ok=True)
print(f'[INFO] The save directory is {save_dir}')
print("="*60)
#=================================================================================


# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train everything in second stage")
print("2. Skip to calculate the measurements")
print("="*60)

while True:
    choice = input("\nChoose option (1 or 2):").strip()
    if choice in ['1','2']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    # Option 1: Train everything in second stage
    print("\n[INFO] Training everything in second stage...")
    #=========================data generation part====================================
    print(f'SAMPLE size for each time step is {args.NUM_SAMPLES}')
    params = params_init('equipart',sample=args.NUM_SAMPLES)
    m0, var0 = MC_triad_initial_value()
    if os.path.exists(os.path.join(save_dir,'simulation_data.npz')):
        print('[SUCCESS] data has already been generated in this folder,you just need to train the following.')
        pass
    else:
        dataset, mean_MC, cov_MC, moment3_MC, moment3_MC_norm,Energy_MC, Energy_dyn = MC_triad_direct(params, m0, var0,method = 'RK4',noise_level=args.NOISE_LEVEL)
        np.savez(os.path.join(save_dir,'simulation_data.npz'),
        dataset=dataset,
        mean_MC=mean_MC,
        cov_MC=cov_MC,
        moment3_MC=moment3_MC,
        moment3_MC_norm=moment3_MC_norm,
        Energy_MC=Energy_MC,
        Energy_dyn=Energy_dyn
        )
        print('[SUCCESS] data generation process finished. Now you can train the following.')
    data = np.load(os.path.join(save_dir,'simulation_data.npz')) 
    dt = 0.01
    residuals,u_current,residual_cov_truth = generate_rk4_residue(FEX_model_check, data, dt)
    np.save(os.path.join(save_dir,'residual_cov_truth.npy'), residual_cov_truth)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = args.TRAIN_SIZE
    #===================================================================================
    ODE_Solution,ZT_Solution = generate_second_step(u_current,residuals,scaler,dt,train_size,device)
    print(f'[INFO] the ODE solution shape is: {ODE_Solution.shape}')
    mean_value, std_value = generate_mean_and_std(ODE_Solution)
    print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
    # Train ensemble models for better accuracy
    train_FN_ensemble(ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir)
    
    print("\n"+ "="*60)
    print("\n[INFO] Testing ensemble predictions...")
    residual_cov_pred = predict_ensemble_residual_covariance(
        residuals=residuals,
        save_dir=save_dir,
        dt=dt,
        scaler=scaler,
        train_size=train_size,
        n_models=5,
        device=device,
        residual_cov_truth=residual_cov_truth
    )
    print('[SUCCESS] Ensemble prediction completed.')
    print("="*60)
    print('[SUCCESS] training process finished.')




else:
    print("\n[INFO] Skip training and calculate the measurements...")
    # Load the data
    data = np.load(os.path.join(save_dir,'simulation_data.npz'))
    dt = 0.01
    residuals, u_current, residual_cov_truth = generate_rk4_residue(FEX_model_check, data, dt)
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = args.TRAIN_SIZE
    
    residual_cov_pred = predict_ensemble_residual_covariance(
        residuals=residuals,
        save_dir=save_dir,
        dt=dt,
        scaler=scaler,
        train_size=train_size,
        n_models=5,
        device=device,
        residual_cov_truth=residual_cov_truth
    )
    print('[SUCCESS] Ensemble prediction completed.')
















    
    