import numpy as np
import os
import sys
from pathlib import Path
# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append("../src/Example/MC_triad")
import torch
from utils import *

from Example.MC_triad.MC_triad import params_init, MC_triad_initial_value
import config

# Import specific functions from ODE Parser
args = config.parse_args()
torch.manual_seed(args.SEED)
np.random.seed(args.SEED)
print("\n"+ "="*60)
print("\n[INFO] Setting up the device andpath...")
# Set device
#===========================Path part==============================================
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    device = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    model_PATH = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results1', 'Results', args.params_name))
    save_dir = os.path.join(config.DIR_TRIAD,'Results', 'Results1', 'Results',args.params_name,f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_constant')
    print('[INFO] Right now we use hipergator workspace path.')
    os.makedirs(save_dir,exist_ok=True)
    print(f'[INFO] The save directory is set up successfully')
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    model_PATH =Path(os.path.join(config.DIR_TRIAD, 'Results', args.params_name))
    # Default save directory (will be updated based on method choice)
    save_dir = os.path.join( model_PATH, f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_constant')
    os.makedirs(save_dir,exist_ok=True)
    print('[INFO] Right now we use our own workspace path.') 
print("="*60)


#=================================================================================
# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train to learn stochastic part in time independent case")
print("2. Skip Training and generate the prediction results")
print("="*60)

while True:
#choice = '1' #
    choice = input("\nChoose option (1 or 2 ):").strip()
    if choice in ['1','2']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    print("\n[INFO] Training everything in second stage...")
    
    # Add noise level selection
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
    # Set up common directory for shared files
    independent_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_independent')
    os.makedirs(independent_save_dir, exist_ok=True)
    print(f'[INFO] Using independent save directory for independent files: {independent_save_dir}')
   
    if not os.path.exists(os.path.join(independent_save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz')):
        raise RuntimeError('[ERROR] data has not been generated, you should run the first_stage_deterministic.py first')
    else:
        print('[SUCCESS] data has already been generated in this folder. Now you can train the following.')
        pass
    data = np.load(os.path.join(independent_save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz'))
    dt = 0.01
    def learned_model_wrapper(x):
        return FEX_model_learned(x, 
                                 model_name = args.Model,
                                 params_name = args.params_name,
                                 noise_level = args.NOISE_LEVEL,
                                 device=device)
    
    residuals, u_current, residual_cov_truth = generate_euler_residue(learned_model_wrapper, data, dt)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    np.save(os.path.join(independent_save_dir,'residual_cov_truth.npy'), residual_cov_truth)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    
    # Use original data structure to preserve proper indexing
    residuals_current_train = residuals[:100,:,:]  # Shape: (100, 3, 1000)
    u_current_train = u_current[:100,:,:]  # Shape: (100, 3, 1000)
    
    # Flatten while preserving trajectory structure
    residuals_train_flat = residuals_current_train.reshape(-1, residuals_current_train.shape[1])  # Shape: (100000, 3)
    u_current_train_flat = u_current_train.reshape(-1, u_current_train.shape[1])  # Shape: (100000, 3)
    
    print(f'[INFO] the residual shape is {residuals_train_flat.shape},the state of dyamics is {u_current_train_flat.shape}')
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = 20000
    short_size = 2048
    it_size_utrain = 2000
    
    it_n_index = train_size // it_size_utrain
    print(f'[INFO] the train size is {train_size}; the short size is {short_size}; the it_size_utrain is {it_size_utrain}; the it_n_index is {it_n_index}')
    select_row_indices = np.random.permutation(residuals_train_flat.shape[0])[:train_size]
    u_train = u_current_train_flat[select_row_indices]
    residuals_train = residuals_train_flat[select_row_indices]
    print(f'[INFO] u_train shape is {u_train.shape}')
    # Use u_train as the reference set to ensure proper indexing
    indices = process_chunk_faiss_cpu(it_n_index, it_size_utrain, short_size, u_current_train_flat, u_train, train_size, 3)
    print(indices)
    u_short = u_current_train_flat[indices]
    z_short = residuals_train_flat[indices]
    print(f'[INFO] u_short shape is {u_short.shape}, z_short shape is {z_short.shape}')
    #===================================================================================
    if not os.path.exists(os.path.join(independent_save_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(independent_save_dir,'ZT_Solution.npy')):
        ZT_Solution = np.random.randn(train_size,3)
        ODE_Solution = np.zeros((train_size,3))
        it_size = min(train_size,60000)
        it_n = int(train_size/it_size)
        ODEsolver_time_steps = 2000
        torch.cuda.empty_cache()
        for jj in range(it_n):
            start_indx = jj*it_size
            end_idx = min((jj+1)*it_size,train_size)
            print(f'[INFO] the start index is {start_indx}, the end index is {end_idx}')
            it_ZT =torch.tensor(ZT_Solution[start_indx:end_idx,:],dtype=torch.float32,device=device)
            it_u0 = torch.tensor(u_train[start_indx:end_idx,:],dtype=torch.float32,device=device)

            u_mini_batch = torch.tensor(u_short[start_indx:end_idx]).to(device)
            z_mini_batch = torch.tensor(z_short[start_indx:end_idx]).to(device)
            ODE_Solution[start_indx:end_idx,:] = ODE_solver(it_ZT,u_mini_batch,z_mini_batch,it_u0,ODEsolver_time_steps).to('cpu').detach().numpy()
            if jj % 5==0:
                print(f'[INFO] the {jj}th iteration is done')
            
        print(f'[INFO] the ODE solution shape is: {ODE_Solution.shape}')
        np.save(os.path.join(independent_save_dir, "ODE_Solution.npy"), ODE_Solution)
        np.save(os.path.join(independent_save_dir, "ZT_Solution.npy"), ZT_Solution)
        np.save(os.path.join(independent_save_dir, "u_short.npy"), u_short)
        np.save(os.path.join(independent_save_dir, "residuals_short.npy"), z_short)
        np.save(os.path.join(independent_save_dir, "select_row_indices.npy"), select_row_indices)
        np.save(os.path.join(independent_save_dir, "short_indices.npy"), indices)
    else:
        print('[INFO] the ODE solution has already been generated, skip the generation process.')
        ODE_Solution = np.load(os.path.join(independent_save_dir, "ODE_Solution.npy"))
        ZT_Solution = np.load(os.path.join(independent_save_dir, "ZT_Solution.npy"))
    
    is_finite_ODE_Solution = np.isfinite(ODE_Solution) &~np.isnan(ODE_Solution)
    print(f'[INFO] the number of finite ODE solution is {np.sum(is_finite_ODE_Solution)}')
    ZT_filtered = ZT_Solution[is_finite_ODE_Solution.all(axis=1)]
    ODE_filtered = ODE_Solution[is_finite_ODE_Solution.all(axis=1)]
    print(f'[INFO] the shape of filtered ZT solution is {ZT_filtered.shape}, the shape of filtered ODE solution is {ODE_filtered.shape}')
    print("\n")
    indices_filtered = np.random.permutation(ZT_filtered.shape[0])
    ZT_shuffled = ZT_filtered[indices_filtered]
    ODE_shuffled = ODE_filtered[indices_filtered]
    print(f'[INFO] the shape of shuffled ZT solution is {ZT_shuffled.shape}, the shape of shuffled ODE solution is {ODE_shuffled.shape}')
    ZT_mean = np.mean(ZT_shuffled,axis=0,keepdims=True)
    ZT_std = np.std(ZT_shuffled,axis=0,keepdims=True)
    ODE_mean = np.mean(ODE_shuffled,axis=0,keepdims=True)
    ODE_std = np.std(ODE_shuffled,axis=0,keepdims=True)

    ZT_normalized = (ZT_shuffled - ZT_mean) / ZT_std
    ODE_normalized = (ODE_shuffled - ODE_mean) / ODE_std
    # convert data to a tensor
    ZT_normalized = torch.tensor(ZT_normalized,dtype=torch.float32,device=device)
    ODE_normalized = torch.tensor(ODE_normalized,dtype=torch.float32,device=device)
    ZT_mean = torch.tensor(ZT_mean,dtype=torch.float32,device=device)
    ZT_std = torch.tensor(ZT_std,dtype=torch.float32,device=device)
    ODE_mean = torch.tensor(ODE_mean,dtype=torch.float32,device=device)
    ODE_std = torch.tensor(ODE_std,dtype=torch.float32,device=device)
    dataname = os.path.join(independent_save_dir,'data_inference.pt')
    torch.save({'ZT_mean':ZT_mean,
                 'ZT_std':ZT_std,
                 'ODE_mean':ODE_mean,
                 'ODE_std':ODE_std,
                 'diff_scale':args.DIFF_SCALE,
    },dataname)
    NTrain = int(ZT_filtered.shape[0]*0.8)
    NTest = int(ZT_filtered.shape[0]*0.2)
    ZT_train_normal = ZT_normalized[:NTrain,:]
    ODE_train_normal = ODE_normalized[:NTrain,:]
    ZT_test_normal = ZT_normalized[NTrain:,:]
    ODE_test_normal = ODE_normalized[NTrain:,:]
    learning_rate = 0.01
    Neural_Network = FN_Net(3,3,50).to(device)
    Neural_Network.zero_grad()
    optimizer = torch.optim.Adam(Neural_Network.parameters(),lr=learning_rate, weight_decay = 1e-5)
    criterion = torch.nn.MSELoss()
    best_valid_err = 5.0
    n_iter = 2000
    for j in range(n_iter):
        optimizer.zero_grad()
        pred = Neural_Network(ZT_train_normal)
        loss = criterion(pred,ODE_train_normal)
        loss.backward()
        optimizer.step()
        pred1 = Neural_Network(ZT_test_normal)
        valid_loss = criterion(pred1,ODE_test_normal)
        if valid_loss < best_valid_err:
            Neural_Network.update_best()
            best_valid_err = valid_loss
        if j%100 == 0:
            print(f'epoch is {j+1}; loss is {loss}; valid loss is {valid_loss}')

    Neural_Network.final_update()

    Neural_Network_path = os.path.join(save_dir,'Neural_Network.pth')
    torch.save(Neural_Network.state_dict(),Neural_Network_path)
    print("[SUCCESS] the Neural_Network has been trained successfully")
    print("[SUCESS] you may run the choice 2 to generate the prediction results.")
    