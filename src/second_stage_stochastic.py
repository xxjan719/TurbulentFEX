import numpy as np
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from utils.ODEParser import *
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
import config

# Import specific functions from ODE Parser
from utils.ODEParser import (
    generate_rk4_residue, 
    generate_second_step, 
    generate_mean_and_std, 
    train_FN_ensemble,
    FEX_model_check
)

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




#===========================Path part==============================================
model_PATH = Path(os.path.join( 'Results', 'equipart'))
if os.path.exists(model_PATH):
    print(model_PATH)
    save_dir = os.path.join( 'Results', 'equipart',f'case_{args.NUM_SAMPLES}')
    print('Right now we use our own workspace path.')
else:
    model_PATH = Path(os.path.join( 'Results', 'Results', 'equipart'))
    save_dir = os.path.join('Results', 'Results','equipart',f'case_{args.NUM_SAMPLES}')
    print('Right now we use hipergator workspace path.')
os.makedirs(save_dir,exist_ok=True)
FN_SAMPLE_PATH = os.path.join(save_dir,'FN_model')
os.makedirs(save_dir,exist_ok=True)
#=================================================================================

#=========================data generation part====================================
m0, var0 = MC_triad_initial_value()
print(f'SAMPLE size for each time step is {args.NUM_SAMPLES}')
params = params_init('equipart',sample=args.NUM_SAMPLES)
if os.path.exists(os.path.join(save_dir,'simulation_data.npz')):
    print('data has already been generated in this folder,you just need to train the following.')
    pass
else:
    dataset, mean_MC, cov_MC, moment3_MC, moment3_MC_norm,Energy_MC, Energy_dyn = MC_triad_direct(params, m0, var0)
    np.savez(os.path.join(save_dir,'simulation_data.npz'),
    dataset=dataset,
    mean_MC=mean_MC,
    cov_MC=cov_MC,
    moment3_MC=moment3_MC,
    moment3_MC_norm=moment3_MC_norm,
    Energy_MC=Energy_MC,
    Energy_dyn=Energy_dyn
    )
    print('data generation process finished. Now you can train the following.')
data = np.load(os.path.join(save_dir,'simulation_data.npz')) 
dt = 0.01
residuals,u_current,residual_cov_truth = generate_rk4_residue(FEX_model_check, data, dt)
np.save(os.path.join(save_dir,'residual_cov_truth.npy'), residual_cov_truth)
print(f'the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
scaler = np.array([20,20,20])
train_size = 10000
#===================================================================================



ODE_Solution,ZT_Solution = generate_second_step(u_current,residuals,scaler,dt,train_size,device)
print('the ODE solution shape is:',ODE_Solution.shape)
print('=============================================')
mean_value, std_value = generate_mean_and_std(ODE_Solution)
print('============this is print for mean and std=====')
print(mean_value.shape, std_value.shape)
print(mean_value[0:2,:],std_value[0:2,:])
    
# Train ensemble models for better accuracy
train_FN_ensemble(ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir)
    
#  Test predictions with ensemble
    
time_step = residuals.shape[2]  
residual_cov_pred = np.zeros((time_step, 3))
print("\n=== Ensemble Prediction Results ===")
# Load and use ensemble predictions for each dimension
for t in range(time_step):
    z_test = np.random.randn(train_size,3)
    z_test_tensor = torch.tensor(z_test, dtype=torch.float32).to(device)
    for dim in range(1, 4):
        # Load normalization parameters
        norm_params = np.load(os.path.join(save_dir,f'norm_params_dim{dim}_t{t}.npy'), allow_pickle=True).item()
        y_mean = norm_params['mean']
        y_std = norm_params['std']
        
        # Ensemble prediction
        ensemble_predictions = []
        n_models = 5  # Number of models in ensemble
            
        for model_idx in range(n_models):
            FN_dim = FN_Net(1,1,100).to(device)
            FN_dim.load_state_dict(torch.load(os.path.join(save_dir,f'FN_dim{dim}_t0_model{model_idx}.pth'), weights_only=True))
                
            # Make prediction
            pred = (FN_dim(z_test_tensor[:,dim-1:dim].reshape(-1,1))).cpu().detach().numpy()
            ensemble_predictions.append(pred)
            
        # Average ensemble predictions
        pred = np.mean(ensemble_predictions, axis=0)
            
        # Denormalize: pred * y_std + y_mean
        pred = pred * y_std + y_mean
            
        # Scale back by scaler
        pred = pred / scaler[dim-1]
        residual_cov_pred[t, dim-1] = np.std(pred) / np.sqrt(dt)    
        print(f"Dimension {dim}: {np.std(pred)/np.sqrt(dt):.6f}")
        print(f"Comparison with Ground Truth:",residual_cov_truth[t, dim-1])
    
    print("\nExpected values should be close to original")
    print("Ensemble method should provide more accurate results!")
np.save(os.path.join(save_dir,'residual_cov_pred.npy'), residual_cov_pred)



