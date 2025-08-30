import sympy as sp
import torch
import numpy as np
import os
import sys



# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and setup
from config import create_main_parser
from utils.FEX import FEX
from utils import *
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value

# Parse arguments
parser = create_main_parser()
args = parser.parse_args()

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    base_path = os.path.join('Example', args.Model, 'Results', 'Results1', 'Results')
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    base_path = os.path.join('Example', args.Model, 'Results')

# Set up paths
if args.DATA_SAVE_PATH is None:
    args.DATA_SAVE_PATH = f'{base_path}/{args.params_name}/noise_{args.NOISE_LEVEL}/simulation_results_noise_{args.NOISE_LEVEL}.npz'
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'
if args.FIGURE_SAVE_PATH is None:
    args.FIGURE_SAVE_PATH = f'{base_path}/{args.params_name}'

# Create necessary directories
os.makedirs(os.path.dirname(args.DATA_SAVE_PATH), exist_ok=True)
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)
os.makedirs(args.FIGURE_SAVE_PATH, exist_ok=True)

# Set up constants
SEED = args.SEED
FEX_LR = args.FEX_LR
TRAIN_EPOCHS_SECOND = args.TRAIN_EPOCHS_SECOND
INTEGRATOR_METHOD = args.INTEGRATOR_METHOD

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize parameters and get initial values
m0, var0 = MC_triad_initial_value()
params = params_init(args.params_name, sample=10000)

# Load or generate dataset
data_file = args.DATA_SAVE_PATH
if os.path.exists(data_file):
    print("\n"+"="*60)
    print(f'[INFO] Data has already generated, just using for the first stage training:FEX'.center(60, '='))
    data = np.load(data_file)
    dataset_full = data['dataset']
    mean_MC = data['mean_MC']
    cov_MC = data['cov_MC']
    moment3_MC = data['moment3_MC']
    moment3_MC_norm = data['moment3_MC_norm']
    Energy_MC = data['Energy_MC']
    Energy_dyn = data['Energy_dyn']
    
    # Select 1000 trajectories for training
    print(f'[INFO] Full dataset shape: {dataset_full.shape}')
    print(f'[INFO] Selecting 1000 trajectories for training...')
    np.random.seed(SEED)
    selected_indices = np.random.choice(dataset_full.shape[0], size=1000, replace=False)
    dataset = dataset_full[selected_indices]
    print(f'[INFO] Selected dataset shape: {dataset.shape}')



# Convert to tensor and set up integrator
dataset_tensor = torch.from_numpy(dataset).float().to(DEVICE)
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'])
integrator = Body4TrainIntegrator(integratorParams, method=INTEGRATOR_METHOD)

print("\n"+"="*60)
print("[INFO] Loading FEX models from previous stage...")
# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+"="*60)
print("Find the candidate operator sequence from training")
    
# print(f"[INFO] get the picture of how the single dimension FEX model works")
# coefficients = get_coefficients(load_dir= DIR_TRIAD, DEVICE=args.DEVICE)
# plot_NOISE_LEVEL_EFFECT(coefficients,save_dir=args.LOG_SAVE_PATH)
# print(f"the coefficients are {coefficients}")

# Initialize coefficient history for tracking
coefficents_history = {1: {}, 2: {}, 3: {}}
for dim in range(1, 4):
    coefficents_history[dim] = {
        'linear_a': [], 'linear_b': [], 
        'nonlinear_a_0': [], 'nonlinear_b_0': [],
        'nonlinear_a_1': [], 'nonlinear_b_1': [],
        'nonlinear_a_2': [], 'nonlinear_b_2': []
    }

op_seqs_all = {}
models = {}
symbols = [sp.symbols(f'x{i+1}') for i in range(3)]
print(f'[INFO] the noise level is {args.NOISE_LEVEL}')
    
# Let user select operator sequences for each dimension
print("\n" + "="*60)
print("Selecting operator sequences for each dimension...")
print("="*60)
    
for dim in range(1, 4):
    print(f'\nSelecting for dimension {dim}...')
    file_path = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}", f'best_candidates_pool_summary_{dim}.txt')
    selected_sequence = select_operator_sequence(file_path, dim)
    if selected_sequence is None:
        print(f"[ERROR] Failed to get sequence for dimension {dim}")
        sys.exit(1)
            
    op_seqs = torch.tensor(selected_sequence, device=DEVICE)
    op_seqs_all[dim] = op_seqs
    print(f"[INFO] {dim} dimension data found. Now let us train integrated FEX model")
    print("\n")
    model = FEX(op_seqs, dim=3).to(DEVICE)  # Fixed: use 3 instead of dimension
    model.apply(weights_init)
    models[str(dim)] = model
        
    # Show initial expression before training
    print(f"Initial expression for Dimension {dim}:")
    print(f"  Full expression: {model.expression_visualize()}")
    print(f"  Simplified expression: {model.expression_visualize_simplified()}")
    print("-" * 60)
        

    print("="*60)
  
loss_history = []
mse = torch.nn.MSELoss()
# # Create optimizer for all parameters
all_params = []
for model in models.values():
    all_params.extend(model.parameters())
model_optim = torch.optim.Adam(all_params, lr=FEX_LR)

print(f"the dataset_tensor is {dataset_tensor.shape}")





current_state = dataset_tensor[:, :, :-1]
print(f"the current_state is {current_state.shape}")









# second_moment_loss = torch.mean((torch.var(state_next_step_collection,dim=0) - torch.var(next_state,dim=0))**2)
# print(second_moment_loss)


# # Training loop
for train_idx in range(TRAIN_EPOCHS_SECOND):  # Changed from 1 to 10 epochs for testing
    #adjust_learning_rate(model_optim, train_idx, FEX_LR, TRAIN_EPOCHS_SECOND)
    model_optim.zero_grad()
    total_pred_loss = 0
    total_moment_loss = 0

    # FIXED: Properly collect predictions from all models with gradient tracking
    # Don't use requires_grad=True on the collection tensor - gradients will flow through the model outputs
    state_next_step_collection = torch.zeros_like(current_state)
    
#     for t_idx in range(current_state.shape[2]):
#         current_tidx = current_state[:,:,t_idx]
#         current_tidx_next_step = torch.zeros_like(current_tidx)
        
#         # FIXED: Properly collect outputs from each model
#         for dim in range(1, 4):
#             model = models[str(dim)]
#             # Each model should take input of shape (batch_size, 3) and output (batch_size, 1)
#             test_output = model(current_tidx)
#             #print(f"Model {dim} input shape: {current_tidx.shape}, output shape: {test_output.shape}")
#             # FIXED: Store output in the correct dimension
#             current_tidx_next_step[:,dim-1] = test_output.squeeze(-1)
        
#         # FIXED: Store the complete next step prediction
#         state_next_step_collection[:,:,t_idx] = current_tidx_next_step

#     #print(f"the state_next_step_collection is {state_next_step_collection.shape}")
#     next_state = dataset_tensor[:,:,1:]
#     #print(f"the next_state is {next_state.shape}")
    
#     # FIXED: Compute moment losses with proper gradient tracking
#     mean_next_state_pred = torch.mean(state_next_step_collection,dim=0)
#     #print(f"the mean_next_state_pred is {mean_next_state_pred.shape}")
#     mean_next_state_target = torch.mean(next_state,dim=0)
#     #print(f"the mean_next_state_target is {mean_state_target.shape}")
#     first_moment_loss = torch.mean((mean_next_state_pred - mean_next_state_target)**2)
#     #print(first_moment_loss)

#     var_next_state_pred = torch.var(state_next_step_collection,dim=0)
#     #print(f"the var_next_state_pred is {var_next_state_pred.shape}")
#     var_next_state_target = torch.var(next_state,dim=0)
#     #print(f"the var_next_state_target is {var_next_state_target.shape}")
#     second_moment_loss = torch.mean((var_next_state_pred - var_next_state_target)**2)
#     #print(second_moment_loss)

#     third_moment_pred = torch.mean((state_next_step_collection - mean_next_state_pred)**3,dim=0)
#     third_moment_target = torch.mean((next_state - mean_next_state_target)**3,dim=0)
#     third_moment_loss = torch.mean((third_moment_pred - third_moment_target)**2)
#     #print(third_moment_loss)

    # FIXED: Add individual prediction losses for each dimension to ensure gradient flow
    individual_pred_losses = 0
    for dim in range(1, 4):
        model = models[str(dim)]
        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
            
        u_pred, u_target = integrator.integrate(integration_args)
        dim_loss = mse(u_pred, u_target)
        individual_pred_losses += dim_loss

    # Combine moment losses with individual prediction losses
    total_pred_loss = individual_pred_losses#+first_moment_loss #+ second_moment_loss + third_moment_loss
    
    # FIXED: Debug gradient flow
    # print(f"\n[DEBUG] Epoch {train_idx} - Gradient check before backward:")
    # for dim in range(1, 4):
    #     model = models[str(dim)]
    #     print(f"  Model {dim} linear_a grad: {model.linear_a.grad}")
    #     print(f"  Model {dim} linear_b grad: {model.linear_b.grad}")
    
    #print(f"[INFO] In {train_idx} epoch, the total_pred_loss is {total_pred_loss.item()}; the first_moment_loss is {first_moment_loss.item()}; the second_moment_loss is {second_moment_loss.item()}; the third_moment_loss is {third_moment_loss.item()};")

    total_pred_loss.backward(retain_graph=True)
    
    # # FIXED: Debug gradient flow after backward
    # print(f"[DEBUG] Epoch {train_idx} - Gradient check after backward:")
    # for dim in range(1, 4):
    #     model = models[str(dim)]
    #     print(f"  Model {dim} linear_a grad: {model.linear_a.grad}")
    #     print(f"  Model {dim} linear_b grad: {model.linear_b.grad}")
    
    model_optim.step()
    
    # FIXED: Debug parameter updates and track changes
    # print(f"[DEBUG] Epoch {train_idx} - Parameter check after step:")
    # for dim in range(1, 4):
    #     model = models[str(dim)]
    #     print(f"  Model {dim} linear_a: {model.linear_a.data}")
    #     print(f"  Model {dim} linear_b: {model.linear_b.data}")
        
    #     # Track parameter changes over epochs
    #     if train_idx == 0:
    #         # Store initial values for comparison
    #         if not hasattr(model, 'initial_linear_a'):
    #             model.initial_linear_a = model.linear_a.data.clone()
    #             model.initial_linear_b = model.linear_b.data.clone()
    #     else:
    #         # Show how much parameters have changed from initial values
    #         linear_a_change = torch.abs(model.linear_a.data - model.initial_linear_a)
    #         linear_b_change = torch.abs(model.linear_b.data - model.initial_linear_b)
    #         print(f"    Model {dim} linear_a change from initial: {linear_a_change}")
    #         print(f"    Model {dim} linear_b change from initial: {linear_b_change}")

        
    with torch.no_grad():
        if train_idx % 1 == 0:  # Show progress every 5 epochs
            print("\n"+"="*60)
            print(f"Training index: {train_idx}")
            print(f"Total Loss: {total_pred_loss.item():.6f}")
            print(f"Individual Pred Loss: {individual_pred_losses.item():.6f}")
            # print(f"First Moment Loss: {first_moment_loss.item():.6f}")
            # print(f"Second Moment Loss: {second_moment_loss.item():.6f}")
            # print(f"Third Moment Loss: {third_moment_loss.item():.6f}")
            
            # Print expressions for each dimension to show coefficient changes
            expressions = {}
            for dim in range(1, 4):  # Fixed: use 4 instead of dimension+1
                expressions[f'Dimension {dim}'] = models[str(dim)].expression_visualize_simplified()
            print(f"Expression: {expressions}")
            print("="*60)

