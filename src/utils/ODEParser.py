import torch
import torch.nn as nn
import numpy as np
import os
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

def process_ode_training_chunked(zT, u_train, train_size, x_dim, device, chunk_size=5000, 
                                batch_size=4000, odesolver_time_steps=1000, chunk_save_dir=None):
    """
    Process ODE training with chunked data loaded from disk.
    
    Parameters:
    - zT: Random noise tensor
    - u_train: Training data
    - train_size: Total size of training data
    - x_dim: Dimension of the data
    - device: Device to run on (CPU/GPU)
    - chunk_size: Size of each chunk (should match the chunking used earlier)
    - batch_size: Size of each batch for ODE solving
    - odesolver_time_steps: Number of ODE solver time steps
    - chunk_save_dir: Directory where chunks are saved
    """
    yTrain = np.zeros((train_size, x_dim), dtype=np.float32)
    
    # Calculate number of chunks and batches
    num_chunks = int(np.ceil(train_size / chunk_size))
    it_n = int(np.ceil(train_size / batch_size))
    
    print(f"ODE Training with chunks:")
    print(f"- Train size: {train_size}")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of chunks: {num_chunks}")
    print(f"- Number of batches: {it_n}")
    print("=" * 50)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for jj in range(it_n):
        start_idx = jj * batch_size
        end_idx = min((jj + 1) * batch_size, train_size)
        
        if jj % 100 == 0:
            print(f'Processing batch {jj}: start_idx={start_idx}, end_idx={end_idx}')
        
        # Load the corresponding chunks for this batch
        x_short_batch = []
        z_short_batch = []
        
        # Determine which chunks we need for this batch
        chunk_start = start_idx // chunk_size
        chunk_end = (end_idx - 1) // chunk_size
        
        for chunk_idx in range(chunk_start, chunk_end + 1):
            chunk_start_idx = chunk_idx * chunk_size
            chunk_end_idx = min((chunk_idx + 1) * chunk_size, train_size)
            
            # Load chunk data
            x_short_file = os.path.join(chunk_save_dir, f'x_short_{chunk_start_idx}_{chunk_end_idx}.npy')
            z_short_file = os.path.join(chunk_save_dir, f'z_short_{chunk_start_idx}_{chunk_end_idx}.npy')
            
            x_short_chunk = np.load(x_short_file)
            z_short_chunk = np.load(z_short_file)
            
            # Determine the portion of this chunk we need
            if chunk_idx == chunk_start:
                # First chunk: take from start_idx to end of chunk
                local_start = start_idx - chunk_start_idx
                x_short_batch.append(x_short_chunk[local_start:])
                z_short_batch.append(z_short_chunk[local_start:])
            elif chunk_idx == chunk_end:
                # Last chunk: take from start of chunk to end_idx
                local_end = end_idx - chunk_start_idx
                x_short_batch.append(x_short_chunk[:local_end])
                z_short_batch.append(z_short_chunk[:local_end])
            else:
                # Middle chunk: take all
                x_short_batch.append(x_short_chunk)
                z_short_batch.append(z_short_chunk)
        
        # Concatenate all chunks for this batch
        x_short_combined = np.concatenate(x_short_batch, axis=0)
        z_short_combined = np.concatenate(z_short_batch, axis=0)
        
        # Convert data to tensors and move to device
        it_zt = torch.tensor(zT[start_idx:end_idx], device=device, dtype=torch.float32)
        it_x0 = torch.tensor(u_train[start_idx:end_idx], device=device, dtype=torch.float32)
        x_mini_batch = torch.tensor(x_short_combined, device=device, dtype=torch.float32)
        z_mini_batch = torch.tensor(z_short_combined, device=device, dtype=torch.float32)
        
        # Solve ODE for this batch
        y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odesolver_time_steps)
        
        # Store results
        yTrain[start_idx:end_idx, :x_dim] = y_temp.to('cpu').detach().numpy()
        
        # Clear intermediate tensors and variables to free memory
        del it_zt, it_x0, x_mini_batch, z_mini_batch, y_temp
        del x_short_batch, z_short_batch, x_short_combined, z_short_combined
        
        if jj % 100 == 0:
            print(f'Completed batch {jj}/{it_n} ({100*jj/it_n:.1f}%)')
    
    print(f"ODE Training completed! Final yTrain shape: {yTrain.shape}")
    return yTrain

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