import torch
import torch.nn as nn
import numpy as np
import os

import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
import torch.optim as optim

# Set environment variable to handle OpenMP runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from .FEX import FEX_model_ground_truth_equipart,FEX_model_learned
except:
    from FEX import FEX_model_ground_truth_equipart,FEX_model_learned
# Add FAISS imports for CPU-based nearest neighbor search
import faiss
    # FAISS_AVAILABLE = True
    # print("FAISS successfully imported for CPU-based nearest neighbor search")
# except ImportError:
    # print("Warning: FAISS not available. Install with: pip install faiss-cpu")
    # FAISS_AVAILABLE = False

def cond_alpha(t,dt): # in the training paper: it should be related to  b(\tau) in formula (3.1)
    return 1-t+dt

def cond_sigma2(t,dt):
    return t+dt

def f(t,dt):
    alpha_t = cond_alpha(t,dt)
    f_t = -1.0/(alpha_t)
    return f_t

def g2(t,dt):
    dsigma2_dt = 1.0
    f_t = f(t,dt)
    sigma2_t = cond_sigma2(t,dt)
    g2 = dsigma2_dt - 2*f_t*sigma2_t
    return g2
def g(t,dt):
    return (g2(t,dt))**0.5



def ODE_solver(zt,x_sample,z_sample,x0_test,
               ODESOLVER_TIME_STEPS:int=2000):
    t_vec = torch.linspace(1.0,0.0,ODESOLVER_TIME_STEPS+1)
    log_weight_likelihood = -1.0* torch.sum( (x0_test[:,None,:]-x_sample)**2/2 , axis = 2, keepdims= False)
    weight_likelihood =torch.exp(log_weight_likelihood)
    for j in range(ODESOLVER_TIME_STEPS): 
        if j% 100 == 0:
            print(f'this is {j} times / overall {ODESOLVER_TIME_STEPS} times')
        t = t_vec[j+1]
        dt = t_vec[j] - t_vec[j+1]
        #print()
        score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t,dt)*z_sample)/cond_sigma2(t,dt)

        log_weight_gauss= -1.0* torch.sum( (zt[:,None,:]-cond_alpha(t,dt)*z_sample)**2/(2*cond_sigma2(t,dt)) , axis =2, keepdims= False)
        weight_temp = torch.exp( log_weight_gauss )
        weight_temp = weight_temp*weight_likelihood
        weight = weight_temp/ torch.sum(weight_temp,axis=1, keepdims=True)
        score = torch.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False)  
        ## score is followed by the formula 3.11
        
        zt= zt - (f(t,dt)*zt-0.5*g2(t, dt)*score) *dt
    return zt


class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)  # Additional layer
        self.output = nn.Linear(self.hid_size, self.output_dim)
        
        # Initialize weights with better initialization
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Additional activation
        x = self.output(x)
        return x

    def update_best(self):
        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_fc2_weight = torch.clone(self.fc2.weight.data)
        self.best_fc2_bias = torch.clone(self.fc2.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):
        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.fc2.weight.data = self.best_fc2_weight
        self.fc2.bias.data = self.best_fc2_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias

def generate_euler_residue(func, data, dt):
    # Extract data dimensions - data shape is (MC_samples, 3, time_steps+1)
    dataset = data['dataset']  # Shape: (MC_samples, 3, time_steps+1)
    MC_samples, _, time_steps_plus_1 = dataset.shape
    time_steps = time_steps_plus_1 - 1
    
    # Filter out trajectories with NaN values
    print(f"[INFO] Original dataset shape: {dataset.shape}")
    print(f"[INFO] Checking for NaN values in trajectories...")
    
    # Find trajectories that contain any NaN values
    nan_trajectories = np.any(np.isnan(dataset), axis=(1, 2))
    valid_trajectories = ~nan_trajectories
    
    print(f"[INFO] Found {np.sum(nan_trajectories)} trajectories with NaN values")
    print(f"[INFO] Using {np.sum(valid_trajectories)} valid trajectories")
    
    if np.sum(valid_trajectories) == 0:
        raise RuntimeError("No valid trajectories found! All trajectories contain NaN values.")
    
    # Filter the dataset to only include valid trajectories
    dataset = dataset[valid_trajectories]
    MC_samples = dataset.shape[0]
    print(f"[INFO] Filtered dataset shape: {dataset.shape}")
    
    # Initialize output arrays
    residuals = np.zeros((MC_samples, 3, time_steps))
    u_current_reshaped = np.zeros((MC_samples, 3, time_steps))
    
    # Process each time step individually to avoid memory issues
    for t in range(time_steps):
        if t % 100 == 0:
            print(f'Processing time step {t}/{time_steps}')
        
        # Extract current and next states for this time step
        u_current = dataset[:, :, t]      # (MC_samples, 3)
        u_next = dataset[:, :, t + 1]     # (MC_samples, 3)
        
        # Store current state for output
        u_current_reshaped[:, :, t] = u_current
        
        # Euler prediction
        func_output = func(u_current)
        u_euler_pred = u_current + dt * func_output
        
        # Calculate residuals for this time step
        residuals[:, :, t] = u_next - u_euler_pred
    
    # Calculate residual covariance for each time step
    residual_cov_time = np.zeros((time_steps, 3))
    
    for t in range(time_steps):
        # Calculate standard deviations
        std_0 = np.std(residuals[:, 0, t])
        std_1 = np.std(residuals[:, 1, t])
        std_2 = np.std(residuals[:, 2, t])
        
        # Calculate residual covariance
        residual_cov_time[t, 0] = std_0 / np.sqrt(dt)
        residual_cov_time[t, 1] = std_1 / np.sqrt(dt)
        residual_cov_time[t, 2] = std_2 / np.sqrt(dt)
        
        if t % 100 == 0:
            print(f"Time {t}: {residual_cov_time[t, 0]:.6f}, {residual_cov_time[t, 1]:.6f}, {residual_cov_time[t, 2]:.6f}")

    print("Residual covariance shape:", residual_cov_time.shape)
    print("First time step covariance:", residual_cov_time[0, :])
    return residuals, u_current_reshaped,residual_cov_time



def process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim, batch_size=256, sample_batch_size=100):
    """
    A function to perform vector similarity search with large `x_sample` processed in batches.

    Parameters:
    - it_n_index: Number of iterations for chunks.
    - it_size_x0train: Size of each chunk.
    - short_size: Number of nearest neighbors to find.
    - x_sample: Vectors to search against (reference vectors).
    - x0_train: Input vectors to be searched (query vectors).
    - train_size: Total number of query vectors.
    - batch_size: Number of query vectors processed at a time to prevent memory overflow.
    - sample_batch_size: Number of reference vectors (`x_sample`) processed at a time.
    
    Returns:
    - x0_train_index_initial: Indices of the nearest neighbors for each query vector.
    """
    # Ensure x_sample and x0_train are PyTorch tensors for GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_sample = torch.tensor(x_sample, dtype=torch.float32, device=device)
    x0_train = torch.tensor(x0_train, dtype=torch.float32, device=device)

    # Prepare the output array
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)

    for jj in range(it_n_index):
        print(f'This is {jj} time')
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        print(f'start_idx is {start_idx}; end_idx is {end_idx}')

        # Extract chunk of query vectors
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Process query vectors in smaller batches to avoid memory overflow
        for batch_start in range(0, x0_train_chunk.size(0), batch_size):
            batch_end = min(batch_start + batch_size, x0_train_chunk.size(0))
            batch = x0_train_chunk[batch_start:batch_end]

            # Prepare temporary storage for distances and indices
            batch_distances = []
            batch_indices = []

            # Process `x_sample` in smaller batches
            for sample_start in range(0, x_sample.size(0), sample_batch_size):
                # print('this is first batch size', sample_start)
                sample_end = min(sample_start + sample_batch_size, x_sample.size(0))
                sample_batch = x_sample[sample_start:sample_end]

                # Compute pairwise distances between the query batch and `x_sample` batch
                distances = torch.cdist(batch, sample_batch, p=2)

                # Track distances and adjust indices for the chunk
                batch_distances.append(distances)
                batch_indices.append(
                    torch.arange(sample_start, sample_end, device=device).unsqueeze(0).repeat(batch.size(0), 1)
                )

            # Concatenate distances and indices across all `x_sample` batches
            batch_distances = torch.cat(batch_distances, dim=1)
            batch_indices = torch.cat(batch_indices, dim=1)

            # Get the `short_size` nearest neighbors
            _, topk_indices = torch.topk(batch_distances, k=short_size, largest=False, dim=1)

            # Map global indices
            topk_global_indices = torch.gather(batch_indices, 1, topk_indices)

            # Store results in the output array
            x0_train_index_initial[start_idx + batch_start:start_idx + batch_end, :] = topk_global_indices.cpu().numpy()

        if jj % 500 == 0:
            print('Find index iteration', jj, it_size_x0train)

    return x0_train_index_initial
        
 





def process_chunk(it_n_index, it_size_x0train, short_size,x_sample, x0_train, train_size,x_dim):
    x0_train_index_initial = np.empty((train_size, short_size ), dtype=int)
    gpu = faiss.StandardGpuResources()  # Initialize GPU resources each time
    index = faiss.IndexFlatL2(x_dim)  # Create a FAISS index for exact searches
    gpu_index = faiss.index_cpu_to_gpu(gpu, 0, index)
    gpu_index.add(x_sample)  # Add the chunk of x_sample to the index
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Perform the search
        _, index_initial = gpu_index.search(x0_train_chunk, short_size)
        x0_train_index_initial[start_idx:end_idx,:] = index_initial 

        if jj % 500 == 0:
            print('find indx iteration:', jj, it_size_x0train)
    # Cleanup resources
    del gpu_index
    del index
    del gpu
    return x0_train_index_initial

def generate_second_step(u_current:np.ndarray,
                          residuals:np.ndarray,
                          scaler:np.ndarray,
                          dt:float,
                          train_size:int=10000,
                          device:str='cpu',
                          ODESOLVER_TIME_STEPS:int=2000,
                          num_time_points:int=None):
    
    total_time_steps = residuals.shape[2]
    size = int(residuals.shape[0])
    odeslover_time_steps = ODESOLVER_TIME_STEPS
    
    # Select time points to process
    if num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Processing {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
        time_step = len(selected_indices)
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
        time_step = total_time_steps
    
    # Ensure train_size doesn't exceed the actual size
    train_size = min(train_size, size)
    
    #short index:
    short_size = 2048
    
    it_size_x0train = train_size
    it_n_index = train_size // it_size_x0train

    # Batch processing parameters
    it_size = min(60000, size)
    it_n = int(size / it_size)
    
    # Initialize output array
    ODE_Solution = np.zeros((size, 3, time_step))
    ZT_Solution = np.zeros((size, 3, time_step))
    # Debug: Show scaler values
    print(f"Scaler values: {scaler}")
    print(f"Original residual std at t=0: {np.std(residuals[:, :, 0], axis=0)}")
    print(f"Using train_size: {train_size} out of total size: {size}")
    
    for t_idx, t in enumerate(time_indices):
        print('-'.center(100, '-'))
        print(f'this is {t_idx+1} times / overall {time_step} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        print(np.std(residuals[:, 0, t].T)/np.sqrt(dt), np.std(residuals[:, 1, t].T)/np.sqrt(dt), np.std(residuals[:, 2, t].T)/np.sqrt(dt))
        print('-'.center(100, '-'))
        u_sample = u_current[:,:,t]
        u_train = u_sample[:train_size]
        
        short_indx = process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, u_sample, u_train, train_size, u_current.shape[1])
        print('short indx is',short_indx)
        u_short = u_sample[short_indx]
        
        # Scale residuals for this time step
        scaled_residuals = residuals[:, :, t] * scaler
        z_short = scaled_residuals[short_indx]
        ZT_Solution[:,:,t_idx] = np.random.randn(size,3)
        # Debug: Show scaled residual std
        print(f"Scaled residual std at t={t}: {np.std(scaled_residuals, axis=0)}")
        
        # Process in mini-batches
        for jj in range(it_n):
            start_idx = jj * it_size
            end_idx = min((jj + 1) * it_size, size)
            print(f'start_idx is {start_idx}; end_idx is {end_idx}')
            
            # Extract mini-batch
            it_residuals = scaled_residuals[start_idx:end_idx]
            
            # Generate random noise for this batch
            z_T = ZT_Solution[start_idx:end_idx,:,t_idx]
            
            # Convert to tensors (assuming CPU processing, adjust device as needed)
            it_zt = torch.tensor(z_T, dtype=torch.float32).to(device)
            it_x0 = torch.tensor(u_sample[start_idx:end_idx], dtype=torch.float32).to(device)
            
            x_mini_batch = torch.tensor(u_short[start_idx:end_idx],dtype =torch.float32).to(device)
            z_mini_batch = torch.tensor(z_short[start_idx:end_idx],dtype = torch.float32).to(device)
            # Call ODE solver for this mini-batch
            y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odeslover_time_steps)
            
            # Store results
            ODE_Solution[start_idx:end_idx, :, t_idx] = y_temp.cpu().detach().numpy()
        
    
        print(f'this is {t_idx+1} times which has already done.')
    
    return ODE_Solution,ZT_Solution

def generate_mean_and_std(ODE_Solution:np.ndarray):
    mean_value = np.zeros((ODE_Solution.shape[2], ODE_Solution.shape[1]))  
    std_value = np.zeros((ODE_Solution.shape[2], ODE_Solution.shape[1]))   
    
    for t in range(ODE_Solution.shape[2]): 
        for dim in range(ODE_Solution.shape[1]):  
            dim_data = ODE_Solution[:, dim, t]  
            mean_value[t, dim] = np.mean(dim_data)  
            std_value[t, dim] = np.std(dim_data)   
    return mean_value, std_value

def train_FN_each_dimension(ODE_Solution:np.ndarray,
                             ZT_Solution:np.ndarray,
                             dim:int=3,
                             device:str='cpu',
                             learning_rate:float=0.001,  # Reduced learning rate
                             n_iter:int=5000,  # More iterations
                             best_valid_err:float=5.0,
                             save_dir:str=None,
                             num_time_points:int=None,
                             time_range:tuple=None,  # New parameter: (start_idx, end_idx)
                             dt:float=0.01):
    total_time_steps = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        # Use specific time range
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'this is {t_idx+1} times / overall {len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        NTrain = int(size* 0.8)
        for x_dim in range(1,dim+1):
            print(f'this is {x_dim} dimension / overall {dim} dimensions')
            FN_dim = FN_Net(1,1,100).to(device)  # Increased hidden size from 50 to 100
            FN_dim.zero_grad()
            optimizer = optim.Adam(FN_dim.parameters(),lr = learning_rate,weight_decay = 1e-5)  # Reduced weight decay
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
            criterion = nn.MSELoss()
            
            # Get the mean and std for normalization
            y_data = ODE_Solution[0:NTrain,x_dim-1,t_idx]
            y_mean = np.mean(y_data)
            y_std = np.std(y_data)
            
            # Reshape data for neural network (needs to be 2D: [samples, features])
            xTrain_normal = torch.tensor(ZT_Solution[0:NTrain,x_dim-1,t_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            yTrain_normal = torch.tensor((y_data - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1).to(device)
            
            y_valid_data = ODE_Solution[NTrain:size,x_dim-1,t_idx]
            xValid_normal = torch.tensor(ZT_Solution[NTrain:size,x_dim-1,t_idx], dtype=torch.float32).reshape(-1, 1).to(device)
            yValid_normal = torch.tensor((y_valid_data - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1).to(device)
            
            best_valid_loss = float('inf')
            patience_counter = 0
            patience_limit = 500  # Early stopping patience
            
            for it in range(n_iter):
                optimizer.zero_grad()
                pred = FN_dim(xTrain_normal)
                loss = criterion(pred,yTrain_normal)
                loss.backward()
                optimizer.step()
                
                pred1 = FN_dim(xValid_normal)
                valid_loss = criterion(pred1,yValid_normal)
                
                # Learning rate scheduling
                scheduler.step(valid_loss)
                
                # Early stopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    FN_dim.update_best()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    print(f"Early stopping at iteration {it}")
                    break
        
                if it%500 == 0:
                    print(f'epoch is {it+1}; loss is {loss:.6f}; valid loss is {valid_loss:.6f}')

            FN_dim.final_update()
            if save_dir is not None:
                FN_path = os.path.join(save_dir,f'FN_dim{x_dim}_t{t}.pth')
                # Save model parameters in CPU format regardless of training device
                state_dict_cpu = {k: v.cpu() for k, v in FN_dim.state_dict().items()}
                torch.save(state_dict_cpu, FN_path)
                print(f'[SAVE] Saved model to: {FN_path}')
                # Save normalization parameters
                norm_params = {'mean': y_mean, 'std': y_std}
                norm_path = os.path.join(save_dir,f'norm_params_dim{x_dim}_t{t}.npy')
                np.save(norm_path, norm_params)
                print(f'[SAVE] Saved normalization params to: {norm_path}')
            else:
                print(f'[WARNING] save_dir is None, not saving model for dim{x_dim}_t{t}')


def train_FN_multi(ODE_Solution:np.ndarray,
                   ZT_Solution:np.ndarray,
                   dim:int=3,
                   device:str='cpu',
                   learning_rate:float=0.001,
                   n_iter:int=5000,
                   best_valid_err:float=5.0,
                   save_dir:str=None,
                   num_time_points:int=None,
                   time_range:tuple=None,
                   dt:float=0.01):
    """
    Train a multi-output neural network (3→3) for joint prediction
    This preserves correlation structure between dimensions
    """
    total_time_steps = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        # Use specific time range
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training 3→3 network on time range {start_idx}-{min(end_idx, total_time_steps)} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training 3→3 network on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'Training 3→3 network for {t_idx+1}/{len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        
        # Check if 3→3 model already exists
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_3to3_t{t}.pth')
            norm_path = os.path.join(save_dir, f'norm_params_3to3_t{t}.npy')
            
            if os.path.exists(FN_path) and os.path.exists(norm_path):
                print(f'[INFO] 3→3 model for time step {t} already exists. Skipping...')
                continue
        
        NTrain = int(size * 0.8)
        
        # Create 3→3 neural network (3 inputs, 3 outputs)
        FN_multi = FN_Net(3, 3, 100).to(device)  # 3 inputs, 3 outputs
        FN_multi.zero_grad()
        optimizer = optim.Adam(FN_multi.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
        criterion = nn.MSELoss()
        
        # Prepare data for ALL dimensions at once
        # Input: All 3 Wiener increments [W1, W2, W3]
        xTrain_normal = torch.tensor(ZT_Solution[0:NTrain, :, t_idx], dtype=torch.float32).to(device)  # (N, 3)
        # Output: All 3 ODE solutions [dim1, dim2, dim3]
        yTrain_normal = torch.tensor(ODE_Solution[0:NTrain, :, t_idx], dtype=torch.float32).to(device)  # (N, 3)
        
        # Validation data
        xValid_normal = torch.tensor(ZT_Solution[NTrain:size, :, t_idx], dtype=torch.float32).to(device)  # (N, 3)
        yValid_normal = torch.tensor(ODE_Solution[NTrain:size, :, t_idx], dtype=torch.float32).to(device)  # (N, 3)
        
        # Calculate normalization parameters for all dimensions
        y_mean = np.mean(ODE_Solution[0:NTrain, :, t_idx], axis=0)  # (3,)
        y_std = np.std(ODE_Solution[0:NTrain, :, t_idx], axis=0)     # (3,)
        
        # Normalize the data
        yTrain_normal = (yTrain_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        yValid_normal = (yValid_normal - torch.tensor(y_mean, dtype=torch.float32).to(device)) / torch.tensor(y_std, dtype=torch.float32).to(device)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        patience_limit = 500  # Early stopping patience
        
        print(f'[INFO] Training 3→3 network for time step {t}...')
        print(f'[INFO] Input shape: {xTrain_normal.shape}, Output shape: {yTrain_normal.shape}')
        
        for it in range(n_iter):
            optimizer.zero_grad()
            
            # Forward pass: predict all 3 dimensions together
            pred = FN_multi(xTrain_normal)  # (N, 3)
            loss = criterion(pred, yTrain_normal)
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                pred_valid = FN_multi(xValid_normal)
                valid_loss = criterion(pred_valid, yValid_normal)
            
            # Learning rate scheduling
            scheduler.step(valid_loss.item())
            
            # Early stopping
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                FN_multi.update_best()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"Early stopping at iteration {it}")
                break
        
            if it % 500 == 0:
                print(f'[INFO] Epoch {it+1}/{n_iter}; Train Loss: {loss.item():.6f}; Valid Loss: {valid_loss.item():.6f}')

        # Load best model
        FN_multi.final_update()
        
        # Save the 3→3 network
        if save_dir is not None:
            FN_path = os.path.join(save_dir, f'FN_3to3_t{t}.pth')
            # Save model parameters in CPU format regardless of training device
            state_dict_cpu = {k: v.cpu() for k, v in FN_multi.state_dict().items()}
            torch.save(state_dict_cpu, FN_path)
            print(f'[SAVE] Saved 3→3 model to: {FN_path}')
            
            # Save normalization parameters for all dimensions
            norm_params = {'mean': y_mean, 'std': y_std}
            norm_path = os.path.join(save_dir, f'norm_params_3to3_t{t}.npy')
            np.save(norm_path, norm_params)
            print(f'[SAVE] Saved normalization params to: {norm_path}')
        else:
            print(f'[WARNING] save_dir is None, not saving 3→3 model for t{t}')
    
    print(f"[INFO] 3→3 neural network training completed!")



def train_FN_ensemble(ODE_Solution:np.ndarray,
                      ZT_Solution:np.ndarray,
                      dim:int=3,
                      device:str='cpu',
                      n_models:int=5,  # Number of ensemble models
                      save_dir:str=None,
                      num_time_points:int=None,
                      time_range:tuple=None,  # New parameter: (start_idx, end_idx)
                      dt:float=0.01):
    """Train an ensemble of neural networks to reduce approximation error"""
    total_time_steps = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    # Select time points to train on
    if time_range is not None:
        # Use specific time range
        start_idx, end_idx = time_range
        time_indices = range(start_idx, min(end_idx, total_time_steps))
        selected_times = np.array([t * dt for t in time_indices])
        print(f"Training on time range {start_idx}-{end_idx} ({len(time_indices)} time points)")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    elif num_time_points is not None:
        selected_indices, selected_times = select_time_points(
            total_time_steps, dt, num_time_points
        )
        print(f"Training on {len(selected_indices)} time points out of {total_time_steps} total")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        time_indices = selected_indices
    else:
        time_indices = range(total_time_steps)
        selected_times = np.arange(total_time_steps) * dt
    
    for t_idx, t in enumerate(time_indices):
        print(f'this is {t_idx+1} times / overall {len(time_indices)} times (time step {t}, t={selected_times[t_idx]:.2f}s)')
        NTrain = int(size* 0.8)
        
        for x_dim in range(1,dim+1):
            print(f'this is {x_dim} dimension / overall {dim} dimensions')
            
            # Get the mean and std for normalization
            y_data = ODE_Solution[0:NTrain,x_dim-1,t]
            y_mean = np.mean(y_data)
            y_std = np.std(y_data)
            
            # Prepare data
            xTrain_normal = torch.tensor(ZT_Solution[0:NTrain,x_dim-1,t], dtype=torch.float32).reshape(-1, 1).to(device)
            yTrain_normal = torch.tensor((y_data - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1).to(device)
            
            y_valid_data = ODE_Solution[NTrain:size,x_dim-1,t]
            xValid_normal = torch.tensor(ZT_Solution[NTrain:size,x_dim-1,t], dtype=torch.float32).reshape(-1, 1).to(device)
            yValid_normal = torch.tensor((y_valid_data - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1).to(device)
            
            # Train ensemble of models
            for model_idx in range(n_models):
                print(f'Training model {model_idx+1}/{n_models}')
                
                # Set different random seeds for each model
                torch.manual_seed(1234 + model_idx)
                
                FN_dim = FN_Net(1,1,100).to(device)
                FN_dim.zero_grad()
                optimizer = optim.Adam(FN_dim.parameters(), lr=0.001, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
                criterion = nn.MSELoss()
                
                best_valid_loss = float('inf')
                patience_counter = 0
                patience_limit = 500
                
                for it in range(5000):
                    optimizer.zero_grad()
                    pred = FN_dim(xTrain_normal)
                    loss = criterion(pred, yTrain_normal)
                    loss.backward()
                    optimizer.step()
                    
                    pred1 = FN_dim(xValid_normal)
                    valid_loss = criterion(pred1, yValid_normal)
                    
                    scheduler.step(valid_loss)
                    
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        FN_dim.update_best()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience_limit:
                        break
                
                FN_dim.final_update()
                
                # Save each model in ensemble
                if save_dir is not None:
                    FN_path = os.path.join(save_dir, f'FN_dim{x_dim}_t{t}_model{model_idx}.pth')
                    torch.save(FN_dim.state_dict(), FN_path)
            
            # Save normalization parameters (same for all models in ensemble)
            if save_dir is not None:
                norm_params = {'mean': y_mean, 'std': y_std}
                np.save(os.path.join(save_dir, f'norm_params_dim{x_dim}_t{t}.npy'), norm_params)



def select_time_points(total_time_steps: int, dt: float, num_points: int = 100):
    """
    Select a subset of time points with regular spacing.
    
    Args:
        total_time_steps (int): Total number of time steps available
        dt (float): Time step size
        num_points (int): Number of points to select
        
    Returns:
        tuple: (selected_indices, selected_times)
    """
    # Calculate total simulation time
    total_time = total_time_steps * dt
    
    # Create regularly spaced time points
    selected_times = np.linspace(0, total_time, num_points)
    
    # Convert times to indices
    selected_indices = np.round(selected_times / dt).astype(int)
    
    # Ensure indices are within bounds
    selected_indices = np.clip(selected_indices, 0, total_time_steps - 1)
    
    # Remove duplicates while preserving order
    unique_indices = []
    unique_times = []
    for idx, time in zip(selected_indices, selected_times):
        if idx not in unique_indices:
            unique_indices.append(idx)
            unique_times.append(time)
    
    return np.array(unique_indices), np.array(unique_times)



def FN_single_update(Winc_tensor:torch.Tensor,
                     device:str,
                     idx:int,
                     save_dir_single:str,
                     dim:int=3,
                     scaler:float=20.0):
    """
    Simple step update function for single neural network model

    """
    stoch_update = np.zeros((Winc_tensor.shape[0], dim), dtype=np.float32)
    neural_network_used = False
    
    for dim_idx in range(1, dim+1):
        # Load normalization parameters
        norm_params_path = os.path.join(save_dir_single, f'norm_params_dim{dim_idx}_t{idx-1}.npy')
        if not os.path.exists(norm_params_path):
            continue
                
        norm_params = np.load(norm_params_path, allow_pickle=True).item()
        y_mean = norm_params['mean']
        y_std = norm_params['std']
            
        # Load single model
        model_path = os.path.join(save_dir_single, f'FN_dim{dim_idx}_t{idx-1}.pth')
        if not os.path.exists(model_path):
            continue
                
        FN_dim = FN_Net(1, 1, 100).to(device)
            
        # Load the CPU-saved model and move to target device
        state_dict = torch.load(model_path, map_location=device)
        FN_dim.load_state_dict(state_dict)
        FN_dim.eval()
            
        # Make prediction
        with torch.no_grad():
            pred = (FN_dim(Winc_tensor[:, dim_idx-1:dim_idx].reshape(-1, 1))).cpu().detach().numpy()
            
        # Denormalize: pred * y_std + y_mean
        pred = pred * y_std + y_mean
            
        # Scale back by scaler
        pred = pred / scaler
        
        # Use neural network prediction as stochastic update for this dimension
        stoch_update[:, dim_idx-1] = pred.flatten()
        neural_network_used = True
    
    return stoch_update if neural_network_used else None


def FN_ensemble_update(Winc_tensor:torch.Tensor,
                       device:str,
                       idx:int,
                       save_dir_ensemble:str,
                       dim:int=3,
                       scaler:float=20.0,
                       n_models:int=5):
    """
    Simple step update function for ensemble neural network models

    """
    stoch_update = np.zeros((Winc_tensor.shape[0], dim), dtype=np.float32)
    neural_network_used = False
    
    for dim_idx in range(1, dim+1):
        ensemble_predictions = []
        
        # Try to load all ensemble models for this dimension
        for model_idx in range(n_models):
            # Load normalization parameters
            norm_params_path = os.path.join(save_dir_ensemble, f'norm_params_dim{dim_idx}_t{idx-1}.npy')
            if not os.path.exists(norm_params_path):
                continue
                    
            norm_params = np.load(norm_params_path, allow_pickle=True).item()
            y_mean = norm_params['mean']
            y_std = norm_params['std']
                
            # Load ensemble model
            model_path = os.path.join(save_dir_ensemble, f'FN_dim{dim_idx}_t{idx-1}_model{model_idx}.pth')
            if not os.path.exists(model_path):
                continue
                    
            FN_dim = FN_Net(1, 1, 100).to(device)
                
            # Load the CPU-saved model and move to target device
            state_dict = torch.load(model_path, map_location=device)
            FN_dim.load_state_dict(state_dict)
            FN_dim.eval()
                
            # Make prediction
            with torch.no_grad():
                pred = (FN_dim(Winc_tensor[:, dim_idx-1:dim_idx].reshape(-1, 1))).cpu().detach().numpy()
                
            # Denormalize: pred * y_std + y_mean
            pred = pred * y_std + y_mean
                
            # Scale back by scaler
            pred = pred / scaler
            
            ensemble_predictions.append(pred.flatten())
        
        # Average ensemble predictions if any models were loaded
        if ensemble_predictions:
            avg_pred = np.mean(ensemble_predictions, axis=0)
            stoch_update[:, dim_idx-1] = avg_pred
            neural_network_used = True
    
    return stoch_update if neural_network_used else None


def FN_multi_update(Winc_tensor:torch.Tensor,
                    device:str,
                    idx:int,
                    save_dir_single:str,
                    dim:int=3,
                    scaler:float=20.0):
    """
    Load and use 3→3 neural network for joint prediction
    """
    stoch_update = None
    neural_network_used = False
    
    # Load normalization parameters for 3→3 model
    norm_params_path = os.path.join(save_dir_single, f'norm_params_3to3_t{idx-1}.npy')
    if not os.path.exists(norm_params_path):
        return None
        
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    y_mean = norm_params['mean']  # (3,)
    y_std = norm_params['std']    # (3,)
        
    # Load 3→3 model
    model_path = os.path.join(save_dir_single, f'FN_3to3_t{idx-1}.pth')
    if not os.path.exists(model_path):
        return None
            
    FN_3to3 = FN_Net(3, 3, 100).to(device)
    
    # Load the CPU-saved model and move to target device
    state_dict = torch.load(model_path, map_location=device)
    FN_3to3.load_state_dict(state_dict)
    FN_3to3.eval()
        
    # Make prediction for all dimensions at once
    with torch.no_grad():
        pred = FN_3to3(Winc_tensor)  # (N, 3)
        
    # Denormalize: pred * y_std + y_mean
    pred = pred * torch.tensor(y_std, dtype=torch.float32).to(device) + torch.tensor(y_mean, dtype=torch.float32).to(device)
        
    # Scale back by scaler
    pred = pred / scaler
    
    # Convert to numpy
    stoch_update = pred.cpu().detach().numpy()
    neural_network_used = True
    
    return stoch_update if neural_network_used else None


def simple_step_update(Winc_tensor:torch.Tensor,
                      device:str,
                      idx:int,
                      save_dir_single:str,
                      save_dir_ensemble:str,
                      model_type:str='single',
                      dim:int=3,
                      scaler:float=20.0,
                      n_models:int=5):
    """
    Simple step update function that can use either single or ensemble models
    
    Args:
        Winc_tensor: Input noise tensor (NPATH, dim)
        device: Device to run on ('cpu' or 'cuda')
        idx: Current time step index
        save_dir_single: Directory containing single model files
        save_dir_ensemble: Directory containing ensemble model files
        model_type: 'single' or 'ensemble' (default 'single')
        dim: Number of dimensions (default 3)
        scaler: Scaling factor for denormalization (default 20.0)
        n_models: Number of ensemble models (default 5)
    
    Returns:
        stoch_update: Stochastic update array (NPATH, dim) or None if models not found
    """
    if model_type.lower() == 'single':
        return FN_single_update(Winc_tensor, device, idx, save_dir_single, dim, scaler)
    elif model_type.lower() == 'ensemble':
        return FN_ensemble_update(Winc_tensor, device, idx, save_dir_ensemble, dim, scaler, n_models)
    elif model_type.lower() == 'multi':
        return FN_multi_update(Winc_tensor, device, idx, save_dir_single, dim, scaler)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'single', 'ensemble', or 'multi'")




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1234)
    np.random.seed(1234)
    model_path = Path(os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart'))
    data = np.load(os.path.join(model_path, 'equipart.npz')) 
    dt = 0.01
    save_dir = os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart','FN_model')
    os.makedirs(save_dir,exist_ok=True)
    
    residuals,u_current = generate_euler_residue(FEX_model_ground_truth_equipart, data, dt)
    print(residuals.shape,u_current.shape)
    
    scaler = np.array([20,20,20])
    ODE_Solution,ZT_Solution = generate_second_step(
        u_current, residuals, scaler, dt, device=device,
        num_time_points=100  # Only process 100 time points
    )
    print(ODE_Solution.shape)
    
    mean_value, std_value = generate_mean_and_std(ODE_Solution)
    print(mean_value.shape, std_value.shape)
    print(mean_value[0:2,:],std_value[0:2,:])
    
    # Train ensemble models for better accuracy (only on selected time points)
    train_FN_ensemble(
        ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir,
        num_time_points=100,  # Only train on 100 time points
        dt=dt
    )