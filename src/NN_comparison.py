import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path
# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append("../src/Example/MC_triad")
from utils import *

from Example.MC_triad.MC_triad import params_init, MC_triad_initial_value
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

learning_rate = 0.001
n_iter = 5000
#===========================Path part==============================================
print("\n"+ "="*60)
print("\n[INFO] Setting up the path...")
if str(device) == 'cpu':
    model_PATH =Path(os.path.join(config.DIR_TRIAD, 'Results', args.params_name))
    save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', 'deter1000', f'second_stage_{args.TRAIN_SIZE}_NN_comparison')
    os.makedirs(save_dir,exist_ok=True)
    # Default save directory (will be updated based on method choice)
    print('[INFO] Right now we use our own workspace path.') 
else:
    model_PATH = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results1', 'Results', args.params_name))
    save_dir = os.path.join(config.DIR_TRIAD, 'Results', 'Results1', 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', 'deter1000', f'second_stage_{args.TRAIN_SIZE}_NN_comparison')
    os.makedirs(save_dir,exist_ok=True)
    print(f'[INFO] The save directory is set up successfully')
print("="*60)
#=================================================================================
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train to learn stochastic part in noise level and num samples")
print("2. Skip Training and generate the prediction results")

print("="*60)
while True:
# choice = '1' #
    choice = input("\nChoose option (1 or 2 ):").strip()
    if choice in ['1','2','3']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    print("\n[INFO] Training NN comparison...")
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
    print(f"MODEL SELECTION for {args.params_name}")
    print("="*60)
    data = np.load(os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', 'deter1000', f'simulation_results_noise_{args.NOISE_LEVEL}.npz'))
    dt = 0.01

    def learned_model_wrapper(x):
        return FEX_model_learned(x, 
                             model_name = args.Model,
                             params_name = args.params_name,
                             noise_level = args.NOISE_LEVEL,
                             device=device)
    
    residuals, u_current, residual_cov_truth = generate_euler_residue(learned_model_wrapper, data, dt)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')

    scaler = np.ones(3) * args.DIFF_SCALE
    scaler = scaler.reshape(1, 3, 1)
    scaled_residuals = residuals * scaler
    scaled_residuals_mean = np.mean(scaled_residuals, axis=0)  # Shape: (3, 1000)
    scaled_residuals_std = np.std(scaled_residuals, axis=0)    # Shape: (3, 1000)
    print(f'[INFO] the residual shape is {residuals.shape}')
    print(f'[INFO] the scaled residual shape is {scaled_residuals.shape}')



    ZT_Solution = np.zeros((residuals.shape[0], 3, residuals.shape[2]))
    for t_idx in range(residuals.shape[2]):
        print(f'[INFO] this is {t_idx+1} times / overall {residuals.shape[2]} times')
        ZT_Solution[:,:,t_idx] = torch.randn(residuals.shape[0],3).to(device)
        print(f'[INFO] the ZT_Solution shape is {ZT_Solution[:,:,t_idx].shape}')
        for dim in range(1,3+1):
            print(f'[INFO] this is {dim} dimension / overall {3} dimensions')
            FN_dim = FN_Net(1,1,100).to(device)  # Increased hidden size from 50 to 100
            optimizer = optim.Adam(FN_dim.parameters(),lr = learning_rate,weight_decay = 1e-5)  # Reduced weight decay
            
            # Add scheduler for learning rate decay
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
            
            # Keep best model parameters
            best_loss = float('inf')
            best_model_state = None

            y_data = scaled_residuals[:,dim-1,t_idx]  # Shape: (49991,)

            # Fix: Use proper input data instead of random
            # ZT_Solution should be Wiener increments or proper input features
            xTrain_normal = torch.tensor(ZT_Solution[:,dim-1,t_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            yTrain_normal = torch.tensor((y_data - scaled_residuals_mean[dim-1, t_idx]) / scaled_residuals_std[dim-1, t_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            
            # Add data splitting for proper validation
            train_size = int(0.8 * len(xTrain_normal))
            x_train, x_valid = xTrain_normal[:train_size], xTrain_normal[train_size:]
            y_train, y_valid = yTrain_normal[:train_size], yTrain_normal[train_size:]
            
            for i in range(n_iter):
                FN_dim.zero_grad()
                y_pred = FN_dim(x_train)  # Use training data
                loss = nn.functional.mse_loss(y_pred, y_train)
                loss.backward()
                optimizer.step()
                
                # Validation loss for scheduler
                with torch.no_grad():
                    y_pred_valid = FN_dim(x_valid)
                    valid_loss = nn.functional.mse_loss(y_pred_valid, y_valid)
                
                # Update scheduler with validation loss
                scheduler.step(valid_loss.item())
                
                # Keep track of best model using FN_Net's built-in method
                current_loss = valid_loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    FN_dim.update_best()  # Use FN_Net's built-in method
                
                # Print progress every 500 iterations
                if (i + 1) % 500 == 0:
                    print(f'[INFO] Dim {dim}, t_idx {t_idx}, Iteration {i+1}/{n_iter}, Train Loss: {loss.item():.6f}, Valid Loss: {current_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Load best model state using FN_Net's built-in method
            FN_dim.final_update()
            
            print(f'[INFO] Final best loss for dim {dim}, t_idx {t_idx}: {best_loss:.6f}')

            np.save(os.path.join(save_dir, f'FN_dim_{dim}_t_{t_idx}.npy'), FN_dim.state_dict())
            print(f'[INFO] the FN_dim_{dim}_t_{t_idx} is saved')
    print(f'[INFO] the NN comparison is saved')
    print("\n")
    print("[SUCCESS] you finished the NN training. Now  you can test the NN comparison for prediction.")

elif choice == '2':
    print("\n[INFO] Skipping Training and generating the prediction results...")
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
    print(f"MODEL SELECTION for {args.params_name}")
    print("="*60)

    m0,var0 = MC_triad_initial_value()
    params = params_init(args.params_name)
    FEX_model_check = FEX_model_learned

    L = params['L']
    G = params['G']
    B = params['B']
    
    TIME_AMOUNT = 10
    dt = 0.01
    NPATH = 5000
    initial_state = np.random.normal(loc=m0, scale=np.sqrt(var0), size=(NPATH, 3))    
    x_pred_initial = torch.ones(NPATH, 3).to(device,dtype=torch.float32) * torch.tensor(m0).to(device,dtype=torch.float32)
    scaler = args.DIFF_SCALE
    
    tmM = np.zeros((int(TIME_AMOUNT/dt),3), dtype=np.float32)
    tmS = np.zeros(int(TIME_AMOUNT/dt), dtype=np.float32)
    mean_state_pred = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record[:, 0] = np.mean(initial_state, axis=0)
    mean_state_pred[:, 0] = np.mean(initial_state, axis=0)



    cov_state_pred = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_pred[:, :, 0] = np.cov(initial_state, rowvar=False)

    

    u_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_all[:,:,0] = initial_state
    u_pred_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_all[:,:,0] = initial_state



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

    # Load neural network models once at the beginning
    print("Loading neural network models...")
    single_models = {}
    single_norms = {}
    ensemble_models = {}
    ensemble_norms = {}

    if str(device) == 'cuda:0':
        save_dir_comparison = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_{args.NOISE_LEVEL}/deter1000/second_stage_{args.TRAIN_SIZE}_NN_comparison'
    else:
        save_dir_comparison = f'../src/Example/MC_triad/Results/{args.params_name}/noise_{args.NOISE_LEVEL}/deter1000/second_stage_{args.TRAIN_SIZE}_NN_comparison'


    tM = np.zeros((int(TIME_AMOUNT/dt),3), dtype=np.float32)
    for idx in range(1,int(TIME_AMOUNT/dt)+1):
        # RK4 integration
        k1 = (L @ current_state.T).T - current_state @ G + Buu(B, current_state, current_state) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        u1 = current_state + dt * k1
        k2 = (L @ u1.T).T - u1 @ G + Buu(B, u1, u1) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        next_state = current_state + dt * (k1 + k2) / 2
        SS = params['SS'] + tmS[idx - 1] ** 2 * (params['SSt'] - params['SS'])
        Winc = np.random.randn(NPATH, 3)  # shape (MC, 3)
        next_state = next_state + np.sqrt(dt) * (Winc @ SS)*args.NOISE_LEVEL  # (MC,3) @ (3,3) → (MC,3)
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

        current_tensor = torch.tensor(current_pred_state, dtype=torch.float32)
    
        # RK4 for the deterministic part (FEX model)
        # Step 1
        # Step 1
        k1_det = FEX_model_check(current_tensor,params_name=args.params_name,noise_level = args.NOISE_LEVEL,device =device) * dt
        k1_det_np = k1_det.cpu().detach().numpy()
        u1 = current_tensor +  k1_det

      

         # Step 2
        k2_det = FEX_model_check(u1,params_name=args.params_name,noise_level = args.NOISE_LEVEL,device =device) * dt
        k2_det_np = k2_det.cpu().detach().numpy()
        u2 = current_tensor +  k2_det
    
        # Final RK4 update
    
    
        # RK4 update for deterministic part
        det_update = (k1_det_np+k2_det_np)/2
    
        # Generate stochastic component (just once per step)
        Npath = current_pred_state.shape[0]
        dim = current_pred_state.shape[1]
        Winc_tensor = torch.Tensor(Winc).to(device, dtype=torch.float32)
    
        
        
        # Simple noise for comparison
        simple_noise = np.sqrt(dt) * (Winc @ SS)*args.NOISE_LEVEL
        
        stoch_update_single_dim1 = FN_Net(1,1,100).to(device)  
        stoch_update_single_dim1.load_state_dict(torch.load(os.path.join(save_dir_comparison, f'FN_dim_1_t_{idx}.npy')))
        stoch_update_single_dim1.eval()
        stoch_update_single_dim2 = FN_Net(1,1,100).to(device)  
        stoch_update_single_dim2.load_state_dict(torch.load(os.path.join(save_dir_comparison, f'FN_dim_2_t_{idx}.npy')))
        stoch_update_single_dim2.eval()
        stoch_update_single_dim3 = FN_Net(1,1,100).to(device)  
        stoch_update_single_dim3.load_state_dict(torch.load(os.path.join(save_dir_comparison, f'FN_dim_3_t_{idx}.npy')))
        stoch_update_single_dim3.eval()

        stoch_update_single_dim1 = stoch_update_single_dim1(Winc_tensor[:,0:1])
        stoch_update_single_dim2 = stoch_update_single_dim2(Winc_tensor[:,1:2])
        stoch_update_single_dim3 = stoch_update_single_dim3(Winc_tensor[:,2:3])
        
        stoch_update_single = np.concatenate([stoch_update_single_dim1, stoch_update_single_dim2, stoch_update_single_dim3], axis=1)
        
        # Print comparison every 50 steps
        if idx % 50 == 0:
            print(f"\nStep {idx}: Model Comparison")
            print("=" * 50)
        
            if stoch_update_single is not None:
                print(f"Single NN - Mean: {np.mean(stoch_update_single, axis=0)}")
                print(f"Single NN - Std:  {np.std(stoch_update_single, axis=0)}")
            else:
                print("Single NN - Not available")
            
           
            
            print(f"Simple Noise - Mean: {np.mean(simple_noise, axis=0)}")
           
            print("=" * 50)
    

       
    
        # Use the selected model for the main prediction (for backward compatibility)
        next_pred_state = current_pred_state + det_update + stoch_update_single
    
        # Store results for all three predictions
        u_pred_all[:,:,idx] = next_pred_state        
    
        # Update statistics for all three predictions
        mean_state_pred[:,idx] = np.mean(next_pred_state, axis=0)
    
        cov_state_pred[:,:,idx] = np.cov(next_pred_state, rowvar=False)
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
    Time_record = np.arange(int(TIME_AMOUNT/dt)+1)

    # Print final summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Model used for stochastic component: NN")
    print(f"Simulation time: {TIME_AMOUNT}")
    print(f"Time step: {dt}")
    print(f"Number of paths: {NPATH}")
    print(f"Total steps: {int(TIME_AMOUNT/dt)}")

    print("="*80)
    

