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
    residuals = residuals.reshape(residuals.shape[0]*residuals.shape[2],residuals.shape[1])
    u_current = u_current.reshape(u_current.shape[0]*u_current.shape[2],u_current.shape[1])
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = 20000
    short_size = 2048
    it_size_utrain = 2000
    it_n_index = train_size // it_size_utrain
    print(f'[INFO] the train size is {train_size}; the short size is {short_size}; the it_size_utrain is {it_size_utrain}; the it_n_index is {it_n_index}')
    select_row_indices = np.random.permutation(residuals.shape[0])[:train_size]
    u_train = u_current[select_row_indices]
    print(f'[INFO] u_train shape is {u_train.shape}')
    process_chunk_faiss_cpu(it_n_index, it_size_utrain, short_size, u_current, u_train, train_size, 3)
    #===================================================================================
    