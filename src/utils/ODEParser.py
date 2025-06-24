import torch
import torch.nn as nn
import numpy as np
import os
import torch.multiprocessing as mp
from functools import partial

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
        
        self.input = nn.Linear(self.input_dim,self.hid_size)
        self.fc1 = nn.Linear(self.hid_size,self.hid_size)
        self.output = nn.Linear(self.hid_size,self.output_dim)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

    def update_best(self):

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):

        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias

def process_single_chunk(chunk_idx, u_train, train_size, x_dim, chunk_size, odesolver_time_steps, save_dir, gpu_id):
    """
    Process a single chunk on a specific GPU.
    """
    # Set device for this process
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    chunk_start_idx = chunk_idx * chunk_size
    chunk_end_idx = min((chunk_idx + 1) * chunk_size, train_size)
    
    print(f'GPU {gpu_id}: Chunk {chunk_idx + 1}: {chunk_start_idx} to {chunk_end_idx}')
    
    # Generate ZT for this chunk
    zT_chunk = np.random.randn(chunk_end_idx - chunk_start_idx, x_dim).astype(np.float32)
    u_train_chunk = u_train[chunk_start_idx:chunk_end_idx]
    
    # Load x_short and z_short for this chunk
    x_short_file = os.path.join(save_dir, 'chunks', f'x_short_{chunk_start_idx}_{chunk_end_idx}.npy')
    z_short_file = os.path.join(save_dir, 'chunks', f'z_short_{chunk_start_idx}_{chunk_end_idx}.npy')
    
    x_short_chunk = np.load(x_short_file)
    z_short_chunk = np.load(z_short_file)
    
    # Convert to tensors
    it_zt = torch.tensor(zT_chunk, device=device, dtype=torch.float32)
    it_x0 = torch.tensor(u_train_chunk, device=device, dtype=torch.float32)
    x_mini_batch = torch.tensor(x_short_chunk, device=device, dtype=torch.float32)
    z_mini_batch = torch.tensor(z_short_chunk, device=device, dtype=torch.float32)
    
    # Solve ODE
    y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odesolver_time_steps)
    
    # Save ZT and yTrain for this chunk
    np.save(os.path.join(save_dir, f'zT_chunk_{chunk_idx}.npy'), zT_chunk)
    np.save(os.path.join(save_dir, f'yTrain_chunk_{chunk_idx}.npy'), y_temp.to('cpu').detach().numpy())
    
    print(f'GPU {gpu_id}: Saved chunk {chunk_idx + 1}')
    
    # Clear memory
    del it_zt, it_x0, x_mini_batch, z_mini_batch, y_temp
    del zT_chunk, u_train_chunk, x_short_chunk, z_short_chunk
    torch.cuda.empty_cache()

def process_ode_training_parallel(u_train, train_size, x_dim, chunk_size=5000, 
                                 odesolver_time_steps=1000, save_dir=None, num_gpus=3):
    """
    Parallel ODE training across multiple GPUs.
    """
    num_chunks = int(np.ceil(train_size / chunk_size))
    
    print(f"Parallel ODE training: {num_chunks} chunks across {num_gpus} GPUs")
    
    # Create list of chunk indices
    chunk_indices = list(range(num_chunks))
    
    # Distribute chunks across GPUs
    chunks_per_gpu = num_chunks // num_gpus
    remainder = num_chunks % num_gpus
    
    # Create process pool
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=num_gpus)
    
    # Distribute work
    start_idx = 0
    for gpu_id in range(num_gpus):
        # Calculate chunks for this GPU
        if gpu_id < remainder:
            num_chunks_this_gpu = chunks_per_gpu + 1
        else:
            num_chunks_this_gpu = chunks_per_gpu
        
        end_idx = start_idx + num_chunks_this_gpu
        gpu_chunks = chunk_indices[start_idx:end_idx]
        
        # Submit work for this GPU
        for chunk_idx in gpu_chunks:
            pool.apply_async(process_single_chunk, 
                           args=(chunk_idx, u_train, train_size, x_dim, chunk_size, 
                                 odesolver_time_steps, save_dir, gpu_id))
        
        start_idx = end_idx
    
    # Wait for all processes to complete
    pool.close()
    pool.join()
    
    print("All chunks processed in parallel!")

def process_ode_training_simple(u_train, train_size, x_dim, device, chunk_size=5000, 
                               odesolver_time_steps=1000, save_dir=None):
    """
    Simple ODE training: generate ZT for each chunk, process, and save both ZT and yTrain.
    """
    num_chunks = int(np.ceil(train_size / chunk_size))
    
    print(f"Simple ODE training: {num_chunks} chunks of {chunk_size}")
    
    # Fix device type checking
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for chunk_idx in range(num_chunks):
        chunk_start_idx = chunk_idx * chunk_size
        chunk_end_idx = min((chunk_idx + 1) * chunk_size, train_size)
        
        print(f'Chunk {chunk_idx + 1}/{num_chunks}: {chunk_start_idx} to {chunk_end_idx}')
        
        # Generate ZT for this chunk
        zT_chunk = np.random.randn(chunk_end_idx - chunk_start_idx, x_dim).astype(np.float32)
        u_train_chunk = u_train[chunk_start_idx:chunk_end_idx]
        
        # Load x_short and z_short for this chunk
        x_short_file = os.path.join(save_dir, 'chunks', f'x_short_{chunk_start_idx}_{chunk_end_idx}.npy')
        z_short_file = os.path.join(save_dir, 'chunks', f'z_short_{chunk_start_idx}_{chunk_end_idx}.npy')
        
        x_short_chunk = np.load(x_short_file)
        z_short_chunk = np.load(z_short_file)
        
        # Convert to tensors
        it_zt = torch.tensor(zT_chunk, device=device, dtype=torch.float32)
        it_x0 = torch.tensor(u_train_chunk, device=device, dtype=torch.float32)
        x_mini_batch = torch.tensor(x_short_chunk, device=device, dtype=torch.float32)
        z_mini_batch = torch.tensor(z_short_chunk, device=device, dtype=torch.float32)
        
        # Solve ODE
        y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odesolver_time_steps)
        
        # Save ZT and yTrain for this chunk
        np.save(os.path.join(save_dir, f'zT_chunk_{chunk_idx}.npy'), zT_chunk)
        np.save(os.path.join(save_dir, f'yTrain_chunk_{chunk_idx}.npy'), y_temp.to('cpu').detach().numpy())
        
        print(f'Saved chunk {chunk_idx + 1}')
        
        # Clear memory
        del it_zt, it_x0, x_mini_batch, z_mini_batch, y_temp
        del zT_chunk, u_train_chunk, x_short_chunk, z_short_chunk
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("All chunks processed!")

def load_and_combine_chunks(train_size, x_dim, save_dir):
    """
    Load and combine all saved chunks into a single yTrain array.
    """
    yTrain = np.zeros((train_size, x_dim), dtype=np.float32)
    
    # Find all chunk files
    chunk_files = [f for f in os.listdir(save_dir) if f.startswith('yTrain_chunk_') and f.endswith('.npy')]
    chunk_files.sort()  # Sort to ensure correct order
    
    print(f"Loading {len(chunk_files)} chunks...")
    
    for chunk_file in chunk_files:
        # Extract indices from filename
        parts = chunk_file.replace('yTrain_chunk_', '').replace('.npy', '').split('_')
        start_idx = int(parts[0])
        end_idx = int(parts[1])
        
        # Load chunk
        chunk_path = os.path.join(save_dir, chunk_file)
        chunk_data = np.load(chunk_path)
        
        # Store in yTrain
        yTrain[start_idx:end_idx] = chunk_data
        
        print(f"Loaded chunk {start_idx}-{end_idx}: {chunk_data.shape}")
    
    return yTrain