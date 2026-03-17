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
# choice = '1' #
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
    residuals_current_train = residuals[:300,:,:]  # Shape: (MC_samples, 3, 1000)
    u_current_train = u_current[:300,:,:]  # Shape: (MC_samples, 3, 1000)
    
    # Flatten while preserving trajectory structure
    residuals_train_flat = residuals_current_train.reshape(-1, residuals_current_train.shape[1])  # Shape: (MC_samples*1000, 3)
    residuals_train_flat = residuals_train_flat*args.DIFF_SCALE
    u_current_train_flat = u_current_train.reshape(-1, u_current_train.shape[1])  # Shape: (MC_samples*1000, 3)
    
    print(f'[INFO] the residual shape is {residuals_train_flat.shape},the state of dyamics is {u_current_train_flat.shape}')
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = 100000
    short_size = 2048
    it_size_utrain = 2000
    
    it_n_index = train_size // it_size_utrain
    print(f'[INFO] the train size is {train_size}; the short size is {short_size}; the it_size_utrain is {it_size_utrain}; the it_n_index is {it_n_index}')
    select_row_indices = np.random.permutation(residuals_train_flat.shape[0])[:train_size]
    u_train = u_current_train_flat[select_row_indices]
    residuals_train = residuals_train_flat[select_row_indices]
    print(f'[INFO] u_train shape is {u_train.shape}')
    # Use u_train as the reference set to ensure proper indexing
    if not os.path.exists(os.path.join(independent_save_dir,'indices_uint32.npy')):
        indices = process_chunk_faiss_cpu(it_n_index, it_size_utrain, short_size, u_current_train_flat, u_train, train_size,3)
        print(indices)
        #indices = indices.astype(np.uint32)
        #np.save(os.path.join(independent_save_dir, "indices_uint32.npy"), indices)
        #print("[INFO] indices saved:", indices.shape, indices.dtype)
    #else:
    #    indices = np.load(os.path.join(independent_save_dir, "indices_uint32.npy"))
    n_train, k = indices.shape
    u_dim = u_current_train_flat.shape[1]

    if residuals_train_flat.ndim == 1:
        z_short = np.empty((n_train, k), dtype=residuals_train_flat.dtype)
    else:
        z_dim = residuals_train_flat.shape[1]
        z_short = np.empty((n_train, k, z_dim), dtype=residuals_train_flat.dtype)

    u_short = np.empty((n_train, k, u_dim), dtype=u_current_train_flat.dtype)

    batch_rows = 10   # try 50 / 100 / 200

    for start in range(0, n_train, batch_rows):
        end = min(start + batch_rows, n_train)
        idx_batch = indices[start:end]

        u_short_batch = u_current_train_flat[idx_batch]
        z_short_batch = residuals_train_flat[idx_batch]

        u_short[start:end] = u_short_batch
        z_short[start:end] = z_short_batch

        print(f"[INFO] saved batch {start}:{end}, "
          f"u_short_batch shape is {u_short_batch.shape}, "
          f"z_short_batch shape is {z_short_batch.shape}")

        del u_short_batch, z_short_batch

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
            ODE_Solution[start_indx:end_idx,:] = ODE_solver_chunk(it_ZT,u_mini_batch,z_mini_batch,it_u0,ODEsolver_time_steps).to('cpu').detach().numpy()
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
    NTrain = int(ZT_filtered.shape[0]*0.8)
    NTest = int(ZT_filtered.shape[0]*0.2)
    if not os.path.exists(os.path.join(independent_save_dir,'data_inference.pt')):
        
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
    else:
        print('[INFO] the data_inference has already been generated, skip the generation process.')
        dataname = os.path.join(independent_save_dir,'data_inference.pt')
        data_inference = torch.load(dataname)
        ZT_mean = data_inference['ZT_mean']
        ZT_std = data_inference['ZT_std']
        ODE_mean = data_inference['ODE_mean']
        ODE_std = data_inference['ODE_std']
        diff_scale = data_inference['diff_scale']
        ZT_shuffled = torch.tensor(ZT_shuffled,dtype=torch.float32,device=device)
        ODE_shuffled = torch.tensor(ODE_shuffled,dtype=torch.float32,device=device)
        ZT_normalized = (ZT_shuffled - ZT_mean) / ZT_std
        ODE_normalized = (ODE_shuffled - ODE_mean) / ODE_std
        
       
    ZT_train_normal = ZT_normalized[:NTrain,:]
    ODE_train_normal = ODE_normalized[:NTrain,:]
    ZT_test_normal = ZT_normalized[NTrain:,:]
    ODE_test_normal = ODE_normalized[NTrain:,:]

    if not os.path.exists(os.path.join(save_dir,'Neural_Network.pth')):
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
    else:
        print('[INFO] the Neural_Network has already been trained, skip the training process.')
        Neural_Network_path = os.path.join(save_dir,'Neural_Network.pth')
        Neural_Network = FN_Net(3,3,50).to(device)
        Neural_Network.load_state_dict(torch.load(Neural_Network_path))
        print("[SUCCESS] the Neural_Network has been loaded successfully")
        
    print("[SUCCESS] the Neural_Network has been trained successfully")
    print("[SUCESS] you may run the choice 2 to generate the prediction results.")
else:
    print("\n[INFO] Skipping training and generating prediction results...")
     # Add comprehensive testing section
    print("\n" + "="*60)
    print("COMPREHENSIVE TRAJECTORY TESTING FOR BOTH MODELS")
    print("="*60)
    
    print("\n[INFO] Running comprehensive trajectory testing...")
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
        save_dir = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_1.0/second_stage_10000_constant'
        independent_save_dir = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_1.0/second_stage_10000_independent'
    else:
        save_dir = f'../src/Example/MC_triad/Results/{args.params_name}/noise_1.0/second_stage_10000_constant'
        independent_save_dir = f'../src/Example/MC_triad/Results/{args.params_name}/noise_1.0/second_stage_10000_independent'

    dataname = os.path.join(independent_save_dir,'data_inference.pt')
    data_inference = torch.load(dataname)
    ZT_mean = data_inference['ZT_mean']
    ZT_std = data_inference['ZT_std']
    ODE_mean = data_inference['ODE_mean']
    ODE_std = data_inference['ODE_std']
    diff_scale = data_inference['diff_scale']
    tM = np.zeros((int(TIME_AMOUNT/dt),3), dtype=np.float32)
    for idx in range(1,int(TIME_AMOUNT/dt)+1):
        # RK4 integration
        k1 = (L @ current_state.T).T - current_state @ G + Buu(B, current_state, current_state) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        u1 = current_state + dt * k1
        k2 = (L @ u1.T).T - u1 @ G + Buu(B, u1, u1) + np.ones((NPATH, 1)) * tmM[idx - 1, :]
        next_state = current_state + dt * (k1 + k2) / 2
        SS = params['SS'] + tmS[idx - 1] ** 2 * (params['SSt'] - params['SS'])
        Winc = np.random.randn(NPATH, 3)  # shape (MC, 3)
        next_state = next_state + np.sqrt(dt) * (Winc @ SS)  # (MC,3) @ (3,3) → (MC,3)
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
        k1_det = FEX_model_check(current_tensor,params_name=args.params_name,device =device) * dt
        k1_det_np = k1_det.cpu().detach().numpy()
        u1 = current_tensor +  k1_det

      

         # Step 2
        k2_det = FEX_model_check(u1,params_name=args.params_name,device =device) * dt
        k2_det_np = k2_det.cpu().detach().numpy()
        u2 = current_tensor +  k2_det
    
        # Final RK4 update
    
    
        # RK4 update for deterministic part
        det_update = (k1_det_np+k2_det_np)/2
    
        # Generate stochastic component (just once per step)
        # Use scaler like 2stage_stochastic_time_dependent.py: denormalize NN output then scale back by scaler
        Npath = current_pred_state.shape[0]
        dim = current_pred_state.shape[1]
        Winc_tensor = torch.Tensor(Winc).to(device, dtype=torch.float32)
        Winc_tensor = (Winc_tensor - ZT_mean) / ZT_std
       
        Neural_Network = FN_Net(3,3,50).to(device)
        Neural_Network_path = os.path.join(save_dir,'Neural_Network.pth')
        Neural_Network.load_state_dict(torch.load(Neural_Network_path))
        with torch.no_grad():
            pred = Neural_Network(Winc_tensor) * ODE_std + ODE_mean
            stoch_update = (pred / scaler).cpu().detach().numpy()
    
        
    
        # Simple noise for comparison
        simple_noise = np.sqrt(dt) * (Winc @ SS)
    
        # Print comparison every 50 steps
        if idx % 50 == 0:
            print(f"\nStep {idx}: Model Comparison")
            print("=" * 50)
        
            if stoch_update is not None:
                print(f"Single NN - Mean: {np.mean(stoch_update, axis=0)}")
                print(f"Single NN - Std:  {np.std(stoch_update, axis=0)}")
            else:
                print("Single NN - Not available")
                 
            
            print(f"Simple Noise - Mean: {np.mean(simple_noise, axis=0)}")
            print(f"Simple Noise - Std:  {np.std(simple_noise, axis=0)}")
            print("=" * 50)
    
        
    
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
    Time_record = np.arange(int(TIME_AMOUNT/dt)+1)
    plot_mean_comparison(mean_state_record, mean_state_single, Time_record, 
                    save_path=save_dir, title_suffix=" - FEX-framework")
    plot_covariance_comparison(cov_state_record, cov_state_single, Time_record, 
                          save_path=save_dir, title_suffix=" - FEX-framework")

    # Plot energy comparison
    plot_energy_comparison(Energy_MC_all, Energy_MC_pred, Time_record, 
                      save_path=save_dir, title_suffix=" - FEX-framework")

    # Plot third-order moments
    plot_third_order_moments(moment3_state_record, moment3_state_pred, Time_record, 
                        save_path=save_dir, title_suffix=" - FEX-framework")

    # Plot probability distributions
    plot_probability_distributions(u_all, u_pred_all, Time_record, 
                              save_path=save_dir, title_suffix=" - FEX-framework")
    
    