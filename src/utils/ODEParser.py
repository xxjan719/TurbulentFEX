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

# Add FAISS imports for CPU-based nearest neighbor search
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS successfully imported for CPU-based nearest neighbor search")
except ImportError:
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

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

def generate_rk4_residue(func, data, dt):
    # Extract data dimensions - data shape is (MC_samples, 3, time_steps+1)
    dataset = data['dataset']  # Shape: (MC_samples, 3, time_steps+1)
    MC_samples, _, time_steps_plus_1 = dataset.shape
    time_steps = time_steps_plus_1 - 1
    
    # Extract individual variables and reshape like in small_test.py
    u1 = dataset[:, 0, :]  # (MC_samples, time_steps+1)
    u2 = dataset[:, 1, :]  # (MC_samples, time_steps+1)
    u3 = dataset[:, 2, :]  # (MC_samples, time_steps+1)
    
    # Reshape for RK4 computation
    u1_next = u1[:, 1:].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    u2_next = u2[:, 1:].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    u3_next = u3[:, 1:].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    u1_current = u1[:, :-1].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    u2_current = u2[:, :-1].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    u3_current = u3[:, :-1].reshape(-1, 1)  # (MC_samples * time_steps, 1)
    
    # Concatenate for function evaluation
    u_current = np.concatenate([u1_current, u2_current, u3_current], axis=1)  # (MC_samples * time_steps, 3)
    u_next = np.concatenate([u1_next, u2_next, u3_next], axis=1)  # (MC_samples * time_steps, 3)
    
    # RK4 steps
    k1 = func(u_current)
    k2 = func(u_current + 0.5 * dt * k1)
    k3 = func(u_current + 0.5 * dt * k2)
    k4 = func(u_current + dt * k3)
    
    # RK4 prediction
    u_rk4_pred = u_current + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
    
    # Reshape back to original format for residual calculation
    u1_next_reshaped = u1_next.reshape(MC_samples, time_steps)
    u2_next_reshaped = u2_next.reshape(MC_samples, time_steps)
    u3_next_reshaped = u3_next.reshape(MC_samples, time_steps)
    u_pred_reshaped = u_rk4_pred.reshape(MC_samples, time_steps, 3)
    
    u_current_reshaped = u_current.reshape(MC_samples, 3, time_steps)

    # Calculate residuals for each time step
    residuals = np.zeros((MC_samples, 3, time_steps))
    for t in range(time_steps):
        residuals[:, 0, t] = u1_next_reshaped[:, t] - u_pred_reshaped[:, t, 0]
        residuals[:, 1, t] = u2_next_reshaped[:, t] - u_pred_reshaped[:, t, 1]
        residuals[:, 2, t] = u3_next_reshaped[:, t] - u_pred_reshaped[:, t, 2]
    
    residual_cov_time = np.zeros((time_steps, 3))
    
    for t in range(time_steps):
        residual_cov_time[t,0] = np.std(residuals[:, 0, t].T) / np.sqrt((dt))
        residual_cov_time[t,1] = np.std(residuals[:, 1, t].T) / np.sqrt((dt))
        residual_cov_time[t,2] = np.std(residuals[:, 2, t].T) / np.sqrt((dt))
        if t% 100 == 0:
            print(residual_cov_time[t,0],residual_cov_time[t,1],residual_cov_time[t,2])

    print("Residual covariance shape:", residual_cov_time.shape)
    print("First time step covariance:")
    return residuals,u_current_reshaped



def process_chunk_faiss_cpu(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim):
    """
    CPU version of process_chunk using FAISS for efficient nearest neighbor search.
    
    Args:
        it_n_index: Number of iterations
        it_size_x0train: Size of each training chunk
        short_size: Number of nearest neighbors to find
        x_sample: Reference points for nearest neighbor search
        x0_train: Training points to find neighbors for
        train_size: Total number of training points
        x_dim: Dimension of the data points
    
    Returns:
        x0_train_index_initial: Array of nearest neighbor indices
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is required for this function. Install with: pip install faiss-cpu")
    
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    
    # Create a FAISS index for exact L2 distance searches on CPU
    index = faiss.IndexFlatL2(x_dim)
    
    # Add the reference points to the index
    index.add(x_sample.astype(np.float32))
    
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Perform the search on CPU
        _, index_initial = index.search(x0_train_chunk.astype(np.float32), short_size)
        x0_train_index_initial[start_idx:end_idx, :] = index_initial 

        if jj % 500 == 0:
            print('find index iteration:', jj, it_size_x0train)
    
    # Cleanup resources
    del index
    
    return x0_train_index_initial





def FEX_model1(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return -0.2*x1 + 1*x2*x3 + 1*x2 + -2*x3 

def FEX_model2(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return  -0.6*x1*x3 + -1*x1 + -0.1*x2 + -3*x3 

def FEX_model3(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return  -0.4*x1*x2 + 2*x1 + 3*x2 + -0.1*x3 

def FEX_model_check(x):
    return np.stack([FEX_model1(x), FEX_model2(x), FEX_model3(x)], axis=1)


def generate_second_step(u_current:np.ndarray,
                          residuals:np.ndarray,
                          scaler:np.ndarray,
                          dt:float,
                          device:str='cpu',
                          ODESOLVER_TIME_STEPS:int=2000):
    
    time_step = residuals.shape[2]
    size = residuals.shape[0]
    odeslover_time_steps = ODESOLVER_TIME_STEPS
    
    
    #short index:
    short_size = 2048
    it_size_x0train = 4000 
    it_n_index = size//it_size_x0train

    # Batch processing parameters
    it_size = min(60000, size)
    it_n = int(size / it_size)
    
    # Initialize output array
    ODE_Solution = np.zeros((size, 3, time_step))
    ZT_Solution = np.zeros((size, 3, time_step))
    # Debug: Show scaler values
    print(f"Scaler values: {scaler}")
    print(f"Original residual std at t=0: {np.std(residuals[:, :, 0], axis=0)}")
    
    for t in range(2):#time_step):
        print('-'.center(100, '-'))
        print(f'this is {t} times / overall {time_step} times')
        print(np.std(residuals[:, 0, t].T)/np.sqrt(dt), np.std(residuals[:, 1, t].T)/np.sqrt(dt), np.std(residuals[:, 2, t].T)/np.sqrt(dt))
        print('-'.center(100, '-'))
        u_sample = u_current[:,:,t]
        short_indx = process_chunk_faiss_cpu(it_n_index,it_size_x0train,short_size,u_sample,u_sample,size,u_current.shape[1])
        u_short = u_sample[short_indx]
        
        # Scale residuals for this time step
        scaled_residuals = residuals[:, :, t] * scaler
        z_short = scaled_residuals[short_indx]
        ZT_Solution[:,:,t] = np.random.randn(size,3)
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
            z_T = ZT_Solution[start_idx:end_idx,:,t]
            
            # Convert to tensors (assuming CPU processing, adjust device as needed)
            it_zt = torch.tensor(z_T, dtype=torch.float32).to(device)
            it_x0 = torch.tensor(u_sample[start_idx:end_idx], dtype=torch.float32).to(device)
            
            x_mini_batch = torch.tensor(u_short[start_idx:end_idx],dtype =torch.float32).to(device)
            z_mini_batch = torch.tensor(z_short[start_idx:end_idx],dtype = torch.float32).to(device)
            # Call ODE solver for this mini-batch
            y_temp = ODE_solver(it_zt, x_mini_batch, z_mini_batch, it_x0, odeslover_time_steps)
            
            # Store results
            ODE_Solution[start_idx:end_idx, :, t] = y_temp.cpu().detach().numpy()
            
        
        print(f'this is {t} times which has already done.')
    
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
                             save_dir:str=None):
    time_step = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    for t in range(2):#time_step):
        print(f'this is {t} times / overall {time_step} times')
        NTrain = int(size* 0.8)
        for x_dim in range(1,dim+1):
            print(f'this is {x_dim} dimension / overall {dim} dimensions')
            FN_dim = FN_Net(1,1,100).to(device)  # Increased hidden size from 50 to 100
            FN_dim.zero_grad()
            optimizer = optim.Adam(FN_dim.parameters(),lr = learning_rate,weight_decay = 1e-5)  # Reduced weight decay
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
            criterion = nn.MSELoss()
            
            # Get the mean and std for normalization
            y_data = ODE_Solution[0:NTrain,x_dim-1,t]
            y_mean = np.mean(y_data)
            y_std = np.std(y_data)
            
            # Reshape data for neural network (needs to be 2D: [samples, features])
            xTrain_normal = torch.tensor(ZT_Solution[0:NTrain,x_dim-1,t], dtype=torch.float32).reshape(-1, 1).to(device)
            yTrain_normal = torch.tensor((y_data - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1).to(device)
            
            y_valid_data = ODE_Solution[NTrain:size,x_dim-1,t]
            xValid_normal = torch.tensor(ZT_Solution[NTrain:size,x_dim-1,t], dtype=torch.float32).reshape(-1, 1).to(device)
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
                torch.save(FN_dim.state_dict(),FN_path)
                # Save normalization parameters
                norm_params = {'mean': y_mean, 'std': y_std}
                np.save(os.path.join(save_dir,f'norm_params_dim{x_dim}_t{t}.npy'), norm_params)

def train_FN_ensemble(ODE_Solution:np.ndarray,
                      ZT_Solution:np.ndarray,
                      dim:int=3,
                      device:str='cpu',
                      n_models:int=5,  # Number of ensemble models
                      save_dir:str=None):
    """Train an ensemble of neural networks to reduce approximation error"""
    time_step = ODE_Solution.shape[2]
    size = ODE_Solution.shape[0]
    
    for t in range(2):#time_step):
        print(f'this is {t} times / overall {time_step} times')
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
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=False)
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







if __name__ == "__main__":
    print(os.getcwd())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1234)
    np.random.seed(1234)
    model_path = Path(os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart'))
    data = np.load(os.path.join(model_path, 'equipart.npz')) 
    dt = 0.01
    save_dir = os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart','FN_model')
    os.makedirs(save_dir,exist_ok=True)
    
    residuals,u_current = generate_rk4_residue(FEX_model_check, data, dt)
    print(residuals.shape,u_current.shape)
    
    scaler = np.array([20,20,20])
    ODE_Solution,ZT_Solution = generate_second_step(u_current,residuals,scaler,dt,device)
    print(ODE_Solution.shape)
    
    mean_value, std_value = generate_mean_and_std(ODE_Solution)
    print(mean_value.shape, std_value.shape)
    print(mean_value[0:2,:],std_value[0:2,:])
    
    # Train ensemble models for better accuracy
    train_FN_ensemble(ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir)
    
    # Test predictions with ensemble
    z_test = np.random.randn(1000,3)
    z_test_tensor = torch.tensor(z_test, dtype=torch.float32).to(device)
    
    print("\n=== Ensemble Prediction Results ===")
    # Load and use ensemble predictions for each dimension
    for dim in range(1, 4):
        # Load normalization parameters
        norm_params = np.load(os.path.join(save_dir,f'norm_params_dim{dim}_t0.npy'), allow_pickle=True).item()
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
        
        print(f"Dimension {dim}: {np.std(pred)/np.sqrt(dt):.6f}")
    
    print("\nExpected values should be close to original")
    print("Ensemble method should provide more accurate results!")