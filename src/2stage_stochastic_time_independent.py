import numpy as np
import os
import sys
from pathlib import Path
# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append("../src/Example/MC_triad")
import torch
import torch.nn as nn
from utils import *
from utils.helper import ResidualVAE
from utils.plot import (
    plot_mean_comparison_tfdm_vae_nn,
    plot_covariance_comparison_tfdm_vae_nn,
    plot_energy_comparison_tfdm_vae_nn,
    plot_third_order_moments_tfdm_vae_nn,
    plot_probability_distributions_tfdm_vae_nn,
)

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
print("2. Train ResidualVAE (Gaussian z -> residual increments)")
print("3. Train Residual NN (Gaussian z(3) -> residual), with identity-moment regularization")
print("4. Skip Training and generate the prediction results")
print("="*60)

while True:
# choice = '1' #
    choice = input("\nChoose option (1, 2, 3, or 4 ):").strip()
    if choice in ['1', '2', '3', '4']:
        break
    else:
        print("Please enter '1', '2', '3', or '4'.")

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
    # Scale residuals with scaler (per-dimension) like 2stage_stochastic_time_dependent.py / generate_second_step
    scaler = np.ones(3) * args.DIFF_SCALE
    residuals_train_flat = residuals_train_flat * scaler
    u_current_train_flat = u_current_train.reshape(-1, u_current_train.shape[1])  # Shape: (MC_samples*1000, 3)
    
    print(f'[INFO] the residual shape is {residuals_train_flat.shape},the state of dyamics is {u_current_train_flat.shape}')
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
        n_iter = 200000
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
    print("[SUCESS] you may run choice 4 to generate the prediction results.")
elif choice == '2':
    print("\n[INFO] Training ResidualVAE (Gaussian z -> residual increments)...")
    # Add comprehensive training section for VAE only
    independent_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_independent')
    os.makedirs(independent_save_dir, exist_ok=True)
    print(f'[INFO] Using independent save directory for VAE: {independent_save_dir}')

    data_path = os.path.join(independent_save_dir, '..', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')
    if not os.path.exists(data_path):
        raise RuntimeError('[ERROR] data has not been generated, you should run the first_stage_deterministic.py first')
    data = np.load(data_path)
    dt = 0.01

    def learned_model_wrapper(x):
        return FEX_model_learned(x,
                                 model_name=args.Model,
                                 params_name=args.params_name,
                                 noise_level=args.NOISE_LEVEL,
                                 device=device)

    residuals, u_current, residual_cov_truth = generate_euler_residue(learned_model_wrapper, data, dt)

    # Use first 300 trajectories for building a residual "density" dataset.
    residuals_current_train = residuals[:300, :, :]  # (MC_samples, 3, time_steps)
    residuals_train_flat = residuals_current_train.reshape(-1, residuals_current_train.shape[1])  # (MC_samples*time_steps, 3)
    scaler = np.ones(3) * args.DIFF_SCALE
    residuals_train_flat = residuals_train_flat * scaler

    train_size = 100000
    select_row_indices = np.random.permutation(residuals_train_flat.shape[0])[:train_size]
    residuals_train = residuals_train_flat[select_row_indices]  # (train_size, 3)

    # VAE input z: Gaussian samples; we learn a generative map z -> residual increments.
    ZT_Solution = np.random.randn(train_size, 3).astype(np.float32)

    # Shuffle pairs so training doesn't depend on any ordering.
    perm = np.random.permutation(train_size)
    ZT_shuffled = ZT_Solution[perm]
    RES_shuffled = residuals_train[perm]

    NTrain = int(train_size * 0.8)
    ZT_train = ZT_shuffled[:NTrain, :]
    RES_train = RES_shuffled[:NTrain, :]
    ZT_test = ZT_shuffled[NTrain:, :]
    RES_test = RES_shuffled[NTrain:, :]

    # Normalization stats saved for inference.
    ZT_mean = np.mean(ZT_shuffled, axis=0, keepdims=True)
    ZT_std = np.std(ZT_shuffled, axis=0, keepdims=True)
    RES_mean = np.mean(RES_shuffled, axis=0, keepdims=True)
    RES_std = np.std(RES_shuffled, axis=0, keepdims=True)
    ZT_std = np.maximum(ZT_std, 1e-12)
    RES_std = np.maximum(RES_std, 1e-12)

    ZT_train_normal = torch.tensor((ZT_train - ZT_mean) / ZT_std, dtype=torch.float32, device=device)
    RES_train_normal = torch.tensor((RES_train - RES_mean) / RES_std, dtype=torch.float32, device=device)
    ZT_test_normal = torch.tensor((ZT_test - ZT_mean) / ZT_std, dtype=torch.float32, device=device)
    RES_test_normal = torch.tensor((RES_test - RES_mean) / RES_std, dtype=torch.float32, device=device)

    vae_path = os.path.join(save_dir, 'ResidualVAE.pth')
    latent_dim = 8
    hid_dim = 64
    beta_kl = 1e-3
    alpha_mean = 0.1
    alpha_var = 1.0
    n_iter = 2000

    vae = ResidualVAE(3, 3, latent_dim=latent_dim, hid_dim=hid_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01, weight_decay=1e-5)

    best_valid_err = float('inf')
    best_state = None

    for j in range(n_iter):
        vae.train()
        optimizer.zero_grad()

        y_hat, mu, logvar = vae(ZT_train_normal)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Match empirical mean and diagonal second moments.
        pred_mean = torch.mean(y_hat, dim=0)
        targ_mean = torch.mean(RES_train_normal, dim=0)
        mean_loss = torch.mean((pred_mean - targ_mean) ** 2)

        pred_var = torch.var(y_hat, dim=0, unbiased=False)
        targ_var = torch.var(RES_train_normal, dim=0, unbiased=False)
        var_loss = torch.mean((pred_var - targ_var) ** 2)

        loss = alpha_mean * mean_loss + alpha_var * var_loss + beta_kl * kl_loss
        loss.backward()
        optimizer.step()

        vae.eval()
        with torch.no_grad():
            y_hat1, _, _ = vae(ZT_test_normal)
            pred_mean_1 = torch.mean(y_hat1, dim=0)
            targ_mean_1 = torch.mean(RES_test_normal, dim=0)
            mean_loss_1 = torch.mean((pred_mean_1 - targ_mean_1) ** 2)

            pred_var_1 = torch.var(y_hat1, dim=0, unbiased=False)
            targ_var_1 = torch.var(RES_test_normal, dim=0, unbiased=False)
            var_loss_1 = torch.mean((pred_var_1 - targ_var_1) ** 2)

            valid_moment = alpha_mean * mean_loss_1 + alpha_var * var_loss_1

        if valid_moment < best_valid_err:
            best_valid_err = valid_moment.item()
            best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}

        if j % 100 == 0:
            print(
                f'[VAE-moment] epoch {j+1}; mean_loss={mean_loss.item():.6f}; '
                f'var_loss={var_loss.item():.6f}; kl_loss={kl_loss.item():.6f}; '
                f'valid_moment={valid_moment.item():.6f}'
            )

    if best_state is not None:
        vae.load_state_dict(best_state)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(vae.state_dict(), vae_path)
    print(f'[VAE] Saved to: {vae_path}')

    # Save inference normalization stats for VAE-driven stochastic updates.
    vae_stats_path = os.path.join(independent_save_dir, 'data_inference_vae.pt')
    torch.save(
        {
            'ZT_mean': torch.tensor(ZT_mean, dtype=torch.float32),
            'ZT_std': torch.tensor(ZT_std, dtype=torch.float32),
            'RES_mean': torch.tensor(RES_mean, dtype=torch.float32),
            'RES_std': torch.tensor(RES_std, dtype=torch.float32),
            'diff_scale': args.DIFF_SCALE,
        },
        vae_stats_path,
    )
    print(f'[VAE] Saved stats to: {vae_stats_path}')
elif choice == '3':
    print("\n[INFO] Training Residual NN (Gaussian z(3) -> residual)...")
    independent_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_independent')
    os.makedirs(independent_save_dir, exist_ok=True)
    print(f'[INFO] Using independent save directory: {independent_save_dir}')

    data_path = os.path.join(independent_save_dir, '..', f'simulation_results_noise_{args.NOISE_LEVEL}.npz')
    if not os.path.exists(data_path):
        raise RuntimeError('[ERROR] data has not been generated, you should run the first_stage_deterministic.py first')
    data = np.load(data_path)
    dt = 0.01

    def learned_model_wrapper(x):
        return FEX_model_learned(
            x,
            model_name=args.Model,
            params_name=args.params_name,
            noise_level=args.NOISE_LEVEL,
            device=device
        )

    residuals, _, _ = generate_euler_residue(learned_model_wrapper, data, dt)

    # Build paired dataset: input Gaussian z ~ N(0, I), target residual increment r(t).
    if residuals.ndim != 3:
        raise RuntimeError(f"[ERROR] Unexpected residuals shape: {residuals.shape}")

    if residuals.shape[1] == 3:
        r_flat = residuals.transpose(0, 2, 1).reshape(-1, 3)
    elif residuals.shape[2] == 3:
        r_flat = residuals.reshape(-1, 3)
    else:
        raise RuntimeError(f"[ERROR] residuals must contain 3 components, got shape {residuals.shape}")

    n_use = min(100000, r_flat.shape[0])
    sel = np.random.permutation(r_flat.shape[0])[:n_use]
    U_data = np.random.randn(n_use, 3).astype(np.float32)
    R_data = r_flat[sel]

    # Normalization stats for inference.
    U_mean = np.mean(U_data, axis=0, keepdims=True)
    U_std = np.std(U_data, axis=0, keepdims=True)
    R_mean = np.mean(R_data, axis=0, keepdims=True)
    R_std = np.std(R_data, axis=0, keepdims=True)
    U_std = np.maximum(U_std, 1e-12)
    R_std = np.maximum(R_std, 1e-12)

    perm = np.random.permutation(n_use)
    U_data = U_data[perm]
    R_data = R_data[perm]
    n_train = int(0.8 * n_use)
    U_train = U_data[:n_train]
    R_train = R_data[:n_train]
    U_test = U_data[n_train:]
    R_test = R_data[n_train:]

    U_train_n = torch.tensor((U_train - U_mean) / U_std, dtype=torch.float32, device=device)
    R_train_n = torch.tensor((R_train - R_mean) / R_std, dtype=torch.float32, device=device)
    U_test_n = torch.tensor((U_test - U_mean) / U_std, dtype=torch.float32, device=device)
    R_test_n = torch.tensor((R_test - R_mean) / R_std, dtype=torch.float32, device=device)
    R_mean_t = torch.tensor(R_mean, dtype=torch.float32, device=device)
    R_std_t = torch.tensor(R_std, dtype=torch.float32, device=device)

    residual_nn_path = os.path.join(save_dir, 'Residual_Network.pth')
    residual_stats_path = os.path.join(independent_save_dir, 'data_inference_residual.pt')

    if not os.path.exists(residual_nn_path):
        model = FN_Net(3, 3, 50).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        mse = nn.MSELoss()
        lambda_id = 0.1
        n_iter = 2000
        best_valid = float('inf')
        best_state = None
        eye3 = torch.eye(3, device=device)

        for ep in range(n_iter):
            optimizer.zero_grad()
            pred_n = model(U_train_n)
            pred_phys = pred_n * R_std_t + R_mean_t

            # Supervised residual fit + identity-moment regularization:
            # E[r^T r] / dt ~ I.
            fit_loss = mse(pred_n, R_train_n)
            gram = (pred_phys.T @ pred_phys) / (pred_phys.shape[0] * dt)
            id_loss = mse(gram, eye3)
            loss = fit_loss + lambda_id * id_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_n_val = model(U_test_n)
                pred_phys_val = pred_n_val * R_std_t + R_mean_t
                fit_val = mse(pred_n_val, R_test_n)
                gram_val = (pred_phys_val.T @ pred_phys_val) / (pred_phys_val.shape[0] * dt)
                id_val = mse(gram_val, eye3)
                valid_loss = fit_val + lambda_id * id_val

            if valid_loss.item() < best_valid:
                best_valid = valid_loss.item()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ep % 100 == 0:
                print(
                    f"[ResidualNN] epoch {ep+1}; fit={fit_loss.item():.6f}; "
                    f"id={id_loss.item():.6f}; valid={valid_loss.item():.6f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), residual_nn_path)
        print(f"[ResidualNN] Saved model to: {residual_nn_path}")
    else:
        print('[INFO] Residual_Network already exists, skip training.')

    torch.save(
        {
            'U_mean': torch.tensor(U_mean, dtype=torch.float32),
            'U_std': torch.tensor(U_std, dtype=torch.float32),
            'RES_mean': torch.tensor(R_mean, dtype=torch.float32),
            'RES_std': torch.tensor(R_std, dtype=torch.float32),
            'diff_scale': 1.0,
        },
        residual_stats_path,
    )
    print(f"[ResidualNN] Saved stats to: {residual_stats_path}")
elif choice == '4':
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
    
    TIME_AMOUNT = 20
    dt = 0.01
    NPATH = 5000
    initial_state = np.random.normal(loc=m0, scale=np.sqrt(var0), size=(NPATH, 3))    
    x_pred_initial = torch.ones(NPATH, 3).to(device,dtype=torch.float32) * torch.tensor(m0).to(device,dtype=torch.float32)
    scaler = args.DIFF_SCALE
    
    Nt_eval = int(TIME_AMOUNT / dt)
    # Use the forcing/noise scaling defined by `params_init`.
    # For example, `dual_cascade` has a *constant* forcing tmM = [0, -1, 1],
    # so hard-coding tmM=0 breaks the deterministic mean balance.
    tmM = np.zeros((Nt_eval, 3), dtype=np.float32)
    tmS = np.zeros(Nt_eval, dtype=np.float32)
    if 'tmM' in params and params['tmM'] is not None:
        tmM_src = np.asarray(params['tmM'], dtype=np.float32)
        if tmM_src.shape[0] == Nt_eval:
            tmM = tmM_src
        else:
            reps = int(np.ceil(Nt_eval / tmM_src.shape[0]))
            tmM = np.tile(tmM_src, (reps, 1))[:Nt_eval]
    if 'tmS' in params and params['tmS'] is not None:
        tmS_src = np.asarray(params['tmS'], dtype=np.float32)
        if tmS_src.shape[0] == Nt_eval:
            tmS = tmS_src
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
    mean_state_tfdm = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_vae = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_tfdm[:, 0] = np.mean(initial_state, axis=0)
    mean_state_vae[:, 0] = np.mean(initial_state, axis=0)

    cov_state_pred = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_pred[:, :, 0] = np.cov(initial_state, rowvar=False)

    # Add separate covariance arrays for single and ensemble
    cov_state_single = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_single[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_ensemble = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_ensemble[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_tfdm = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_vae = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_tfdm[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_vae[:, :, 0] = np.cov(initial_state, rowvar=False)

    u_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_all[:,:,0] = initial_state
    u_pred_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_all[:,:,0] = initial_state

    # Add separate arrays for single and ensemble predictions
    u_pred_single = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_single[:,:,0] = initial_state
    u_pred_ensemble = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_ensemble[:,:,0] = initial_state
    u_pred_tfdm = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_vae = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_tfdm[:, :, 0] = initial_state
    u_pred_vae[:, :, 0] = initial_state

    moment3_state_record = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_state_pred = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_state_tfdm = np.zeros((3, 3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_state_vae = np.zeros((3, 3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_first,_ = compute_third_order_moments(initial_state)
    moment3_state_record[:,:,:,0] = moment3_first
    moment3_state_pred[:,:,:,0] = moment3_first
    moment3_state_tfdm[:,:,:,0] = moment3_first
    moment3_state_vae[:,:,:,0] = moment3_first

    Energy_MC_all = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_pred = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_single = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_tfdm = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_vae = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)

    current_state = initial_state
    current_pred_state = initial_state
    current_pred_state_nn = initial_state.copy()
    current_pred_state_tfdm = initial_state.copy()
    current_pred_state_vae = initial_state.copy()

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
    Energy_MC_single[:, 0] = Energy_update_pred
    Energy_MC_tfdm[:, 0] = Energy_update_pred
    Energy_MC_vae[:, 0] = Energy_update_pred

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

    # Keep exactly the same absolute directory convention as training options.
    # Using relative ../src paths here can silently load mismatched/stale assets.
    noise_str = f'noise_{args.NOISE_LEVEL}'
    save_dir = os.path.join(
        model_PATH,
        noise_str,
        f'second_stage_{args.RESIDUAL_SAMPLES}_constant',
    )
    independent_save_dir = os.path.join(
        model_PATH,
        noise_str,
        f'second_stage_{args.RESIDUAL_SAMPLES}_independent',
    )

    dataname = os.path.join(independent_save_dir, 'data_inference.pt')
    vae_stats_path = os.path.join(independent_save_dir, 'data_inference_vae.pt')
    residual_stats_path = os.path.join(independent_save_dir, 'data_inference_residual.pt')
    nn_path = os.path.join(save_dir, 'Neural_Network.pth')
    vae_path = os.path.join(save_dir, 'ResidualVAE.pth')
    residual_nn_path = os.path.join(save_dir, 'Residual_Network.pth')

    # NN assets (legacy: Gaussian z -> ODE residual)
    has_nn_legacy = os.path.exists(nn_path) and os.path.exists(dataname)
    Neural_Network = None
    ZT_mean_nn = ZT_std_nn = ODE_mean = ODE_std = None
    diff_scale_nn = args.DIFF_SCALE

    if has_nn_legacy:
        print(f"[INFO] Loading Neural_Network from: {nn_path}")
        data_inference = torch.load(dataname, map_location=device)
        ZT_mean_nn = data_inference['ZT_mean'].to(device)
        ZT_std_nn = data_inference['ZT_std'].to(device)
        ODE_mean = data_inference['ODE_mean'].to(device)
        ODE_std = data_inference['ODE_std'].to(device)
        diff_scale_nn = data_inference.get('diff_scale', args.DIFF_SCALE)
        if torch.is_tensor(diff_scale_nn):
            diff_scale_nn = diff_scale_nn.item()

        Neural_Network = FN_Net(3, 3, 50).to(device)
        Neural_Network.load_state_dict(torch.load(nn_path, map_location=device))
        Neural_Network.eval()
    else:
        print("[INFO] Legacy Neural_Network assets not found.")

    # Residual NN assets (new: state u -> residual)
    has_residual_nn = os.path.exists(residual_nn_path) and os.path.exists(residual_stats_path)
    Residual_Network = None
    U_mean_res = U_std_res = RES_mean_res = RES_std_res = None
    diff_scale_res = 1.0
    if has_residual_nn:
        print(f"[INFO] Loading Residual_Network from: {residual_nn_path}")
        residual_stats = torch.load(residual_stats_path, map_location=device)
        U_mean_res = residual_stats['U_mean'].to(device)
        U_std_res = residual_stats['U_std'].to(device)
        RES_mean_res = residual_stats['RES_mean'].to(device)
        RES_std_res = residual_stats['RES_std'].to(device)
        diff_scale_res = residual_stats.get('diff_scale', 1.0)
        if torch.is_tensor(diff_scale_res):
            diff_scale_res = diff_scale_res.item()

        Residual_Network = FN_Net(3, 3, 50).to(device)
        Residual_Network.load_state_dict(torch.load(residual_nn_path, map_location=device))
        Residual_Network.eval()
    else:
        print("[INFO] Residual_Network assets not found.")

    # TFDM path prefers the new Residual_Network when available.
    has_nn = has_residual_nn or has_nn_legacy

    # VAE assets
    has_vae = os.path.exists(vae_path) and os.path.exists(vae_stats_path)
    Residual_VAE = None
    ZT_mean_vae = ZT_std_vae = RES_mean = RES_std = None
    diff_scale_vae = args.DIFF_SCALE

    if has_vae:
        print(f"[INFO] Loading ResidualVAE from: {vae_path}")
        latent_dim = 8
        hid_dim = 64
        Residual_VAE = ResidualVAE(3, 3, latent_dim=latent_dim, hid_dim=hid_dim).to(device)
        Residual_VAE.load_state_dict(torch.load(vae_path, map_location=device))
        Residual_VAE.eval()

        vae_stats = torch.load(vae_stats_path, map_location=device)
        ZT_mean_vae = vae_stats['ZT_mean'].to(device)
        ZT_std_vae = vae_stats['ZT_std'].to(device)
        RES_mean = vae_stats['RES_mean'].to(device)
        RES_std = vae_stats['RES_std'].to(device)
        diff_scale_vae = vae_stats.get('diff_scale', args.DIFF_SCALE)
        if torch.is_tensor(diff_scale_vae):
            diff_scale_vae = diff_scale_vae.item()
    else:
        print("[INFO] ResidualVAE assets not found; VAE comparison disabled.")

    if not has_nn and not has_vae:
        raise RuntimeError(
            "[ERROR] No stochastic model available. "
            "Run option 1/2/3 first to train at least one model."
        )

    # Primary model for rollout: prefer VAE when available.
    use_vae = has_vae
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

        # Deterministic RK2 update helper (used for each model trajectory)
        def det_update_from_state(state_np):
            state_tensor = torch.tensor(state_np, dtype=torch.float32).to(device)
            k1_det = FEX_model_check(
                state_tensor,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            u1_det = state_tensor + k1_det
            k2_det = FEX_model_check(
                u1_det,
                model_name=args.Model,
                params_name=args.params_name,
                noise_level=args.NOISE_LEVEL,
                device=device,
            ) * dt
            return ((k1_det + k2_det) / 2).cpu().detach().numpy()

        det_update_nn = det_update_from_state(current_pred_state_nn)
        det_update_tfdm = det_update_from_state(current_pred_state_tfdm)
        det_update_vae = det_update_from_state(current_pred_state_vae)
    
        # Generate stochastic component (just once per step)
        # NN outputs normalized ODE; denormalize (pred = NN*ODE_std + ODE_mean) then divide by diff_scale
        # to match training (residuals were scaled by scaler before ODE solver; use saved diff_scale).
        Npath = current_pred_state.shape[0]
        dim = current_pred_state.shape[1]
        Winc_tensor = torch.Tensor(Winc).to(device, dtype=torch.float32)
        with torch.no_grad():
            stoch_update_nn = None
            stoch_update_nn_legacy = None
            stoch_update_vae = None

            if has_nn_legacy:
                winc_nn = (Winc_tensor - ZT_mean_nn) / ZT_std_nn
                pred_nn = Neural_Network(winc_nn) * ODE_std + ODE_mean
                stoch_update_nn = (pred_nn / diff_scale_nn).cpu().detach().numpy()
            if has_residual_nn:
                z_norm = (Winc_tensor - U_mean_res) / U_std_res
                pred_res = Residual_Network(z_norm) * RES_std_res + RES_mean_res
                stoch_update_nn_legacy = (pred_res / diff_scale_res).cpu().detach().numpy()

            if has_vae:
                winc_vae = (Winc_tensor - ZT_mean_vae) / ZT_std_vae
                y_hat_vae, _, _ = Residual_VAE(winc_vae)
                pred_vae = y_hat_vae * RES_std + RES_mean
                stoch_update_vae = (pred_vae / diff_scale_vae).cpu().detach().numpy()

            if use_vae and stoch_update_vae is not None:
                stoch_update = stoch_update_vae
            elif stoch_update_nn_legacy is not None:
                stoch_update = stoch_update_nn_legacy
            else:
                stoch_update = None
    
        # Simple noise for comparison (and optional rescaling reference)
        simple_noise = np.sqrt(dt) * (Winc @ SS)
        if stoch_update_nn is not None and not np.isfinite(stoch_update_nn).all():
            print("[WARN] Non-finite values in FEX+NN stochastic update; fallback to simple noise for this step.")
            stoch_update_nn = simple_noise.copy()
        if stoch_update_nn_legacy is not None and not np.isfinite(stoch_update_nn_legacy).all():
            print("[WARN] Non-finite values in FEX+TFDM stochastic update; fallback to simple noise for this step.")
            stoch_update_nn_legacy = simple_noise.copy()
        if stoch_update_vae is not None and not np.isfinite(stoch_update_vae).all():
            print("[WARN] Non-finite values in FEX+VAE stochastic update; fallback to simple noise for this step.")
            stoch_update_vae = simple_noise.copy()
        # Match NN/VAE stochastic increments to simple-noise first/second moments.
        # This mirrors discussion_test.py and avoids mean bias in trajectories.
        std_simple = np.std(simple_noise, axis=0)
        std_simple = np.maximum(std_simple, 1e-12)
        mean_simple = np.mean(simple_noise, axis=0)
        if stoch_update_nn is not None:
            std_nn = np.std(stoch_update_nn, axis=0)
            mean_nn = np.mean(stoch_update_nn, axis=0)
            scale_nn = np.where(std_nn > 1e-12, std_simple / std_nn, 1.0)
            stoch_update_nn = (stoch_update_nn - mean_nn) * scale_nn + mean_simple
        if stoch_update_nn_legacy is not None:
            std_tfdm = np.std(stoch_update_nn_legacy, axis=0)
            mean_tfdm = np.mean(stoch_update_nn_legacy, axis=0)
            scale_tfdm = np.where(std_tfdm > 1e-12, std_simple / std_tfdm, 1.0)
            stoch_update_nn_legacy = (stoch_update_nn_legacy - mean_tfdm) * scale_tfdm + mean_simple
        if stoch_update_vae is not None:
            std_vae = np.std(stoch_update_vae, axis=0)
            mean_vae = np.mean(stoch_update_vae, axis=0)
            scale_vae = np.where(std_vae > 1e-12, std_simple / std_vae, 1.0)
            stoch_update_vae = (stoch_update_vae - mean_vae) * scale_vae + mean_simple
        if stoch_update is not None:
            std_sel = np.std(stoch_update, axis=0)
            mean_sel = np.mean(stoch_update, axis=0)
            scale_match = np.where(std_sel > 1e-12, std_simple / std_sel, 1.0)
            stoch_update = (stoch_update - mean_sel) * scale_match + mean_simple
    
        # Print comparison every 50 steps
        if idx % 50 == 0:
            print(f"\nStep {idx}: Model Comparison")
            print("=" * 50)
        
            if stoch_update_nn_legacy is not None:
                print(f"FEX+TFDM - Mean: {np.mean(stoch_update_nn_legacy, axis=0)}")
                print(f"FEX+TFDM - Std:  {np.std(stoch_update_nn_legacy, axis=0)}")
            else:
                print("FEX+TFDM - Not available")

            if stoch_update_nn is not None:
                print(f"FEX+NN   - Mean: {np.mean(stoch_update_nn, axis=0)}")
                print(f"FEX+NN   - Std:  {np.std(stoch_update_nn, axis=0)}")
            else:
                print("FEX+NN   - Not available")

            if stoch_update_vae is not None:
                print(f"FEX+VAE - Mean: {np.mean(stoch_update_vae, axis=0)}")
                print(f"FEX+VAE - Std:  {np.std(stoch_update_vae, axis=0)}")
            else:
                print("FEX+VAE - Not available")

            print(f"Simple Noise - Mean: {np.mean(simple_noise, axis=0)}")
            print(f"Simple Noise - Std:  {np.std(simple_noise, axis=0)}")
            print("=" * 50)
    
        
    
        # Build each prediction trajectory explicitly
        next_pred_nn = (
            current_pred_state_nn + det_update_nn + stoch_update_nn
            if stoch_update_nn is not None
            else current_pred_state_nn + det_update_nn + (stoch_update_nn_legacy if stoch_update_nn_legacy is not None else simple_noise)
        )
        next_pred_tfdm = (
            current_pred_state_tfdm + det_update_tfdm + stoch_update_nn_legacy
            if stoch_update_nn_legacy is not None
            else current_pred_state_tfdm + det_update_tfdm + simple_noise
        )
        next_pred_vae = (
            current_pred_state_vae + det_update_vae + stoch_update_vae
            if stoch_update_vae is not None
            else current_pred_state_vae + det_update_vae + simple_noise
        )

        # Use selected model for legacy outputs
        next_pred_state = next_pred_vae if use_vae else next_pred_tfdm
        next_pred_single = next_pred_nn
    
        # Store results for all three predictions
        u_pred_all[:,:,idx] = next_pred_state
        u_pred_single[:,:,idx] = next_pred_single
        u_pred_tfdm[:, :, idx] = next_pred_tfdm
        u_pred_vae[:, :, idx] = next_pred_vae
    
        # Update statistics for all three predictions
        mean_state_pred[:,idx] = np.mean(next_pred_state, axis=0)
        mean_state_single[:,idx] = np.mean(next_pred_single, axis=0)
        mean_state_tfdm[:, idx] = np.mean(next_pred_tfdm, axis=0)
        mean_state_vae[:, idx] = np.mean(next_pred_vae, axis=0)
        
        # Debug: check whether state means match in dual_cascade.
        # This helps isolate whether the constant offset you observe is coming from
        # the deterministic (FEX) update vs the stochastic increment.
        if args.params_name == 'dual_cascade' and idx % 50 == 0:
            gt_mean = mean_state_record[:, idx]
            pred_mean = mean_state_pred[:, idx]  # same as mean_state_single here
            diff = pred_mean - gt_mean
            print(
                f"[dual_cascade][mean_state @ step {idx}] "
                f"gt_mean={gt_mean} pred_mean={pred_mean} diff={diff}"
            )
       
    
        cov_state_pred[:,:,idx] = np.cov(next_pred_state, rowvar=False)
        cov_state_single[:,:,idx] = np.cov(next_pred_single, rowvar=False)
        cov_state_tfdm[:, :, idx] = np.cov(next_pred_tfdm, rowvar=False)
        cov_state_vae[:, :, idx] = np.cov(next_pred_vae, rowvar=False)

        # Calculate energy directly from mean and covariance (same as ground truth)
        Energy_MC_pred[0, idx] = 0.5 * np.sum(mean_state_pred[:, idx] ** 2) + 0.5 * np.trace(cov_state_pred[:, :, idx])
        Energy_MC_pred[1, idx] = 0.5 * (mean_state_pred[0, idx] ** 2 + cov_state_pred[0, 0, idx])
        Energy_MC_pred[2, idx] = 0.5 * (mean_state_pred[1, idx] ** 2 + cov_state_pred[1, 1, idx])
        Energy_MC_pred[3, idx] = 0.5 * (mean_state_pred[2, idx] ** 2 + cov_state_pred[2, 2, idx])
        Energy_MC_single[0, idx] = 0.5 * np.sum(mean_state_single[:, idx] ** 2) + 0.5 * np.trace(cov_state_single[:, :, idx])
        Energy_MC_single[1, idx] = 0.5 * (mean_state_single[0, idx] ** 2 + cov_state_single[0, 0, idx])
        Energy_MC_single[2, idx] = 0.5 * (mean_state_single[1, idx] ** 2 + cov_state_single[1, 1, idx])
        Energy_MC_single[3, idx] = 0.5 * (mean_state_single[2, idx] ** 2 + cov_state_single[2, 2, idx])
        Energy_MC_tfdm[0, idx] = 0.5 * np.sum(mean_state_tfdm[:, idx] ** 2) + 0.5 * np.trace(cov_state_tfdm[:, :, idx])
        Energy_MC_tfdm[1, idx] = 0.5 * (mean_state_tfdm[0, idx] ** 2 + cov_state_tfdm[0, 0, idx])
        Energy_MC_tfdm[2, idx] = 0.5 * (mean_state_tfdm[1, idx] ** 2 + cov_state_tfdm[1, 1, idx])
        Energy_MC_tfdm[3, idx] = 0.5 * (mean_state_tfdm[2, idx] ** 2 + cov_state_tfdm[2, 2, idx])
        Energy_MC_vae[0, idx] = 0.5 * np.sum(mean_state_vae[:, idx] ** 2) + 0.5 * np.trace(cov_state_vae[:, :, idx])
        Energy_MC_vae[1, idx] = 0.5 * (mean_state_vae[0, idx] ** 2 + cov_state_vae[0, 0, idx])
        Energy_MC_vae[2, idx] = 0.5 * (mean_state_vae[1, idx] ** 2 + cov_state_vae[1, 1, idx])
        Energy_MC_vae[3, idx] = 0.5 * (mean_state_vae[2, idx] ** 2 + cov_state_vae[2, 2, idx])
    
        # Calculate third-order moments for prediction
        moment3_pred, _ = compute_third_order_moments(next_pred_state)
        moment3_state_pred[:, :, :, idx] = moment3_pred
        moment3_tfdm, _ = compute_third_order_moments(next_pred_tfdm)
        moment3_vae, _ = compute_third_order_moments(next_pred_vae)
        moment3_state_tfdm[:, :, :, idx] = moment3_tfdm
        moment3_state_vae[:, :, :, idx] = moment3_vae
    
        # Update current state
        current_pred_state = next_pred_state
        current_pred_state_nn = next_pred_nn
        current_pred_state_tfdm = next_pred_tfdm
        current_pred_state_vae = next_pred_vae
    
    np.random.seed(0)
    # Physical time 0 to TIME_AMOUNT (e.g. 0 to 50)
    Time_record = np.arange(int(TIME_AMOUNT/dt)+1) * dt
    # Mean comparison with both stochastic models:
    # orange: FEX+TFDM, green: FEX+VAE
    plot_mean_comparison_tfdm_vae_nn(
        mean_state_record,
        mean_state_single,
        mean_state_tfdm,
        mean_state_vae,
        Time_record,
        save_path=save_dir,
    )
    plot_covariance_comparison_tfdm_vae_nn(
        cov_state_record,
        cov_state_single,
        cov_state_tfdm,
        cov_state_vae,
        Time_record,
        save_path=save_dir,
    )

    # Plot energy comparison
    plot_energy_comparison_tfdm_vae_nn(
        Energy_MC_all,
        Energy_MC_single,
        Energy_MC_tfdm,
        Energy_MC_vae,
        Time_record,
        save_path=save_dir,
    )

    # Plot third-order moments
    plot_third_order_moments_tfdm_vae_nn(
        moment3_state_record,
        moment3_state_pred,
        moment3_state_tfdm,
        moment3_state_vae,
        Time_record,
        save_path=save_dir,
    )

    # Plot probability distributions
    plot_probability_distributions_tfdm_vae_nn(
        u_all,
        u_pred_single,
        u_pred_tfdm,
        u_pred_vae,
        Time_record,
        save_path=save_dir,
    )
    
    