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

# Set device
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    device = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU instead")

#===========================Path part==============================================
print("\n"+ "="*60)
print("\n[INFO] Setting up the path...")
if str(device) == 'cpu':
    model_PATH =Path(os.path.join(config.DIR_TRIAD, 'Results', args.params_name))
    # Default save directory (will be updated based on method choice)
    save_dir = os.path.join( model_PATH, f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_single')
    os.makedirs(save_dir,exist_ok=True)
    print('[INFO] Right now we use our own workspace path.') 
else:
    model_PATH = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results1', 'Results', args.params_name))
    save_dir = os.path.join(config.DIR_TRIAD,'Results', 'Results1', 'Results',args.params_name,f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_single')
    print('[INFO] Right now we use hipergator workspace path.')
    os.makedirs(save_dir,exist_ok=True)
    print(f'[INFO] The save directory is set up successfully')
print("="*60)
#=================================================================================


# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train to learn stochastic part in noise level and num samples")
print("2. Skip Training and generate the prediction results")

print("="*60)

while True:
#choice = '1' #
    choice = input("\nChoose option (1 or 2 ):").strip()
    if choice in ['1','2','3']:
        break
    else:
        print("Please enter '1' or '2'.")

if choice == '1':
    # Option 1: Train everything in second stage
    print("\n[INFO] Training everything in second stage...")
    
    # Add noise level selection
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
   
    # Set up common directory for shared files
    common_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_common')
    os.makedirs(common_save_dir, exist_ok=True)
    print(f'[INFO] Using common save directory for shared files: {common_save_dir}')
   
    if not os.path.exists(os.path.join(common_save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz')):
        raise RuntimeError('[ERROR] data has not been generated, you should run the first_stage_deterministic.py first')
    else:
        print('[SUCCESS] data has already been generated in this folder. Now you can train the following.')
        pass
    data = np.load(os.path.join(common_save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz'))
    dt = 0.01
    # ground truth
    # residuals, u_current, residual_cov_truth = generate_euler_residue(FEX_model_ground_truth_equipart, data, dt)
    def learned_model_wrapper(x):
        return FEX_model_learned(x, 
                                 model_name = args.Model,
                                 params_name = args.params_name,
                                 noise_level = args.NOISE_LEVEL,
                                 device=device)
    
    residuals, u_current, residual_cov_truth = generate_euler_residue(learned_model_wrapper, data, dt)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    np.save(os.path.join(common_save_dir,'residual_cov_truth.npy'), residual_cov_truth)
    print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
    scaler = np.ones(3) * args.DIFF_SCALE
    train_size = args.RESIDUAL_SAMPLES
    #===================================================================================
    if not os.path.exists(os.path.join(common_save_dir,'ODE_Solution.npy')) and not os.path.exists(os.path.join(common_save_dir,'ZT_Solution.npy')):
        ODE_Solution,ZT_Solution = generate_second_step(
        u_current, residuals, scaler, dt, train_size, device,
        num_time_points=1001  # Only process 100 time points
    )
        print(f'[INFO] the ODE solution shape is: {ODE_Solution.shape}')
        mean_value, std_value = generate_mean_and_std(ODE_Solution)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        np.save(os.path.join(common_save_dir, "ODE_Solution.npy"), ODE_Solution)
        np.save(os.path.join(common_save_dir, "ZT_Solution.npy"), ZT_Solution)
    else:
        print('[INFO] the ODE solution has already been generated, skip the generation process.')
        ODE_Solution = np.load(os.path.join(common_save_dir, "ODE_Solution.npy"))
        mean_value, std_value = generate_mean_and_std(ODE_Solution)
        print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
        ZT_Solution = np.load(os.path.join(common_save_dir, "ZT_Solution.npy"))
        
    # Initialize training method
    training_method = 'unknown'  # Will be detected from file patterns
    
    # Scan both directories to detect existing models and method
    ensemble_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}')
    single_dir = os.path.join(model_PATH,  f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_single')
    os.makedirs(ensemble_dir, exist_ok=True)
    print(f'[DEBUG] Scanning ensemble directory: {ensemble_dir}')
    print(f'[DEBUG] Scanning single directory: {single_dir}')
    
    # Get all model files from both directories
    model_files = []
    if os.path.exists(ensemble_dir):
        ensemble_files = [f for f in os.listdir(ensemble_dir) if f.startswith('FN_dim') and f.endswith('.pth')]
        print(f'[DEBUG] Found {len(ensemble_files)} ensemble model files')
        print(f'[DEBUG] First 10 ensemble files: {ensemble_files[:10]}')
        for file in ensemble_files:
            model_files.append(os.path.join(ensemble_dir, file))
    else:
        print(f'[DEBUG] Ensemble directory does not exist: {ensemble_dir}')
        
    if os.path.exists(single_dir):
        single_files = [f for f in os.listdir(single_dir) if f.startswith('FN_dim') and f.endswith('.pth')]
        print(f'[DEBUG] Found {len(single_files)} single model files')
        print(f'[DEBUG] First 10 single files: {single_files[:10]}')
        for file in single_files:
            model_files.append(os.path.join(single_dir, file))
    else:
        print(f'[DEBUG] Single directory does not exist: {single_dir}')
    
    print(f'[DEBUG] Total model files found: {len(model_files)}')
    
    # Check if there are any model files at all
    if len(model_files) == 0:
        print('[WARNING] No model files found in either directory!')
        print('[INFO] Checking normalization files to find maximum trained time step...')
        
        # Look for normalization files to determine maximum trained time step
        max_trained_step = -1
        if os.path.exists(ensemble_dir):
            norm_files = [f for f in os.listdir(ensemble_dir) if f.startswith('norm_params_dim') and f.endswith('.npy')]
            print(f'[DEBUG] Found {len(norm_files)} normalization files in ensemble directory')
            
            for file in norm_files:
                # Parse filename like "norm_params_dim3_t99.npy"
                parts = file.replace('.npy', '').split('_')
                if len(parts) >= 4 and parts[2] == 't':
                    try:
                        time_step = int(parts[3])
                        max_trained_step = max(max_trained_step, time_step)
                    except ValueError:
                        continue
        
        if max_trained_step >= 0:
            print(f'[INFO] Found normalization files up to time step {max_trained_step}')
            print(f'[INFO] Will continue training from time step {max_trained_step + 1}')
            time_steps = set(range(max_trained_step + 1))
            training_method = 'ensemble'  # Assume ensemble since we found ensemble norm files
        else:
            print('[INFO] No normalization files found either. Starting from beginning.')
            time_steps = set()
            training_method = 'unknown'
            
            # Choose training method (always ask, regardless of existing models)
            print("\n" + "="*60)
            print("NEURAL NETWORK TRAINING METHOD SELECTION")
            print("="*60)
            print("1. Single Neural Network (train_FN_each_dimension)")
            print("2. Ensemble Method (train_FN_ensemble)")
            print("="*60)
            
            while True:
                method_choice = input("\nChoose training method (1 or 2): ").strip()
                if method_choice in ['1', '2']:
                    break
                else:
                    print("Please enter '1' or '2'.")
            
            # Update save directory based on method choice
            if method_choice == '1':
                # Single neural network method
                model_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_single')
                chosen_method = 'single'
            else:
                # Ensemble method
                model_save_dir = os.path.join(model_PATH, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}')
                chosen_method = 'ensemble'
            
            # Create model directory if it doesn't exist
            os.makedirs(model_save_dir, exist_ok=True)
            print(f'[INFO] Using model save directory: {model_save_dir}')
            
            # Create time ranges for starting from beginning
            total_time_steps = ODE_Solution.shape[2]
            steps_per_range = total_time_steps // 4
            extra_steps = total_time_steps % 4
            
            time_ranges = []
            current_start = 0
            
            for i in range(4):
                # Add extra steps to first few ranges if needed
                current_steps = steps_per_range + (1 if i < extra_steps else 0)
                current_end = min(current_start + current_steps, total_time_steps)
                time_ranges.append((current_start, current_end))
                current_start = current_end
                
                # Stop if we've reached the end
                if current_end >= total_time_steps:
                    break
            
            print(f'[INFO] No existing models found, created 4 initial ranges: {time_ranges}')
            print(f'[INFO] Will start training from time step 0 to {total_time_steps}')
            
            # Since we're starting from beginning, all ranges are incomplete
            incomplete_ranges = time_ranges.copy()
            fn_models_exist = False
            
            print(f'[INFO] All ranges are incomplete, will train: {incomplete_ranges}')
            print(f'[DEBUG] fn_models_exist: {fn_models_exist}')
            
            # Train the models for all ranges
            print('[INFO] Training FN models for all time ranges...')
            
            # Save time range configuration
            time_range_config_path = os.path.join(model_save_dir, 'time_range_config.npy')
            time_range_config = {
                'time_ranges': time_ranges,
                'total_time_steps': ODE_Solution.shape[2],
                'dt': dt,
                'num_models_per_ensemble': 5 if method_choice == '2' else 1,
                'num_dimensions': 3,
                'training_method': 'ensemble' if method_choice == '2' else 'single'
            }
            np.save(time_range_config_path, time_range_config)
            print(f'[INFO] Time range configuration saved to: {time_range_config_path}')
            
            # Train all ranges
            for start_idx, end_idx in incomplete_ranges:
                print(f"\n[INFO] Training for time range {start_idx}-{end_idx}...")
                
                if method_choice == '1':
                    # Single neural network method
                    print(f"[INFO] Using single neural network method...")
                    # For single method, we need to calculate num_time_points from the range
                    num_time_points = end_idx - start_idx
                    train_FN_each_dimension(
                        ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=model_save_dir,
                        time_range=(start_idx, end_idx),  # Use specific time range
                        dt=dt
                    )
                else:
                    # Ensemble method
                    print(f"[INFO] Using ensemble method...")
                    train_FN_ensemble(
                        ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=model_save_dir,
                        time_range=(start_idx, end_idx),  # Use specific time range
                        dt=dt
                    )
            
            print('\n[INFO] Training completed successfully!')
            print("\n"+ "="*60)
            print('[SUCCESS] Training process completed successfully!')
            print("="*60)
    else:
        # Extract time steps from model files and detect method
        time_steps = set()
        training_method = 'unknown'  # Will be detected from file patterns
        
        for file_path in model_files:
            file = os.path.basename(file_path)  # Get just the filename
            # Parse filename like "FN_dim1_t123_model0.pth" (ensemble) or "FN_dim1_t123.pth" (single)
            parts = file.replace('.pth', '').split('_')
            if len(parts) >= 3 and parts[1].startswith('dim') and parts[2].startswith('t'):
                try:
                    time_step = int(parts[2][1:])  # Extract number after 't'
                    time_steps.add(time_step)
                    
                    # Detect training method based on filename pattern
                    if len(parts) >= 4 and parts[3].startswith('model'):
                        training_method = 'ensemble'
                    else:
                        training_method = 'single'
                except ValueError:
                    continue
        
        print(f'[DEBUG] Extracted time steps: {sorted(list(time_steps))[:20]}...' + (f' (and {len(time_steps)-20} more)' if len(time_steps) > 20 else ''))
        print(f'[DEBUG] Min time step: {min(time_steps) if time_steps else "None"}')
        print(f'[DEBUG] Max time step: {max(time_steps) if time_steps else "None"}')
        print(f'[DEBUG] Detected training method: {training_method}')
        
        # Determine which directory to use for scanning based on detected method
        if training_method == 'ensemble':
            scan_dir = ensemble_dir
        elif training_method == 'single':
            scan_dir = single_dir
        else:
            # If no method detected, use the original save_dir
            scan_dir = save_dir
        
        print(f'[INFO] Scanning directory for existing models: {scan_dir}')
        print(f'[INFO] Detected training method: {training_method}')
        
        # Sort time steps and create ranges
        if time_steps:
            time_steps = sorted(list(time_steps))
            print(f'[INFO] Found models for time steps: {time_steps[:10]}...' + (f' (and {len(time_steps)-10} more)' if len(time_steps) > 10 else ''))
            
            # Find the maximum time step that has been trained
            max_trained_step = max(time_steps)
            print(f'[INFO] Maximum trained time step: {max_trained_step}')
            
            # Create 4 ranges starting from the next untrained step
            start_from = max_trained_step + 1
            total_time_steps = ODE_Solution.shape[2]
            remaining_steps = total_time_steps - start_from
            
            if remaining_steps > 0:
                # Divide remaining steps into 4 parts
                steps_per_range = remaining_steps // 4
                extra_steps = remaining_steps % 4  # Distribute extra steps to first ranges
                
                time_ranges = []
                current_start = start_from
                
                for i in range(4):
                    # Add extra steps to first few ranges if needed
                    current_steps = steps_per_range + (1 if i < extra_steps else 0)
                    current_end = min(current_start + current_steps, total_time_steps)
                    time_ranges.append((current_start, current_end))
                    current_start = current_end
                    
                    # Stop if we've reached the end
                    if current_end >= total_time_steps:
                        break
                
                print(f'[INFO] Created 4 continuation ranges: {time_ranges}')
                print(f'[INFO] Will continue training from time step {start_from} to {total_time_steps}')
            else:
                # All steps are already trained
                time_ranges = []
                print(f'[INFO] All time steps (0-{total_time_steps-1}) are already trained!')
        else:
            # No existing models, create 4 ranges starting from 0
            total_time_steps = ODE_Solution.shape[2]
            steps_per_range = total_time_steps // 4
            extra_steps = total_time_steps % 4
            
            time_ranges = []
            current_start = 0
            
            for i in range(4):
                # Add extra steps to first few ranges if needed
                current_steps = steps_per_range + (1 if i < extra_steps else 0)
                current_end = min(current_start + current_steps, total_time_steps)
                time_ranges.append((current_start, current_end))
                current_start = current_end
                
                # Stop if we've reached the end
                if current_end >= total_time_steps:
                    break
            
            print(f'[INFO] No existing models found, created 4 initial ranges: {time_ranges}')
            print(f'[INFO] Will start training from time step 0 to {total_time_steps}')
        
        # If no time ranges were created, use default ranges
        if len(time_ranges) == 0:
            time_ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
            print(f'[WARNING] No time ranges detected, using default ranges: {time_ranges}')
        
        # Check which time ranges are already completed
        completed_ranges = []
        incomplete_ranges = []
        
        for start_idx, end_idx in time_ranges:
            range_complete = True
            for dim in range(1, 4):  # dimensions 1, 2, 3
                for t in range(start_idx, end_idx):
                    if training_method == 'ensemble':
                        # Check for ensemble models (5 models per time point)
                        for model_idx in range(5):
                            fn_path = os.path.join(scan_dir, f"FN_dim{dim}_t{t}_model{model_idx}.pth")
                            if not os.path.exists(fn_path):
                                range_complete = False
                                break
                        if not range_complete:
                            break
                    else:
                        # Check for single model
                        fn_path = os.path.join(scan_dir, f"FN_dim{dim}_t{t}.pth")
                        if not os.path.exists(fn_path):
                            range_complete = False
                            break
                if not range_complete:
                    break
            
            if range_complete:
                completed_ranges.append((start_idx, end_idx))
            else:
                incomplete_ranges.append((start_idx, end_idx))
        
        # Check if all ranges are complete
        fn_models_exist = len(incomplete_ranges) == 0
        
        # Print status
        if completed_ranges:
            print(f'[INFO] Completed time ranges: {completed_ranges}')
        if incomplete_ranges:
            print(f'[INFO] Incomplete time ranges: {incomplete_ranges}')
        
        # Show detailed progress for each range
        print('\n[INFO] Detailed progress by time range:')
        print(f'[INFO] Detected training method: {training_method}')
        
        for start_idx, end_idx in time_ranges:
            range_name = f'{start_idx}-{end_idx}'
            if (start_idx, end_idx) in completed_ranges:
                print(f'  ✓ Range {range_name}: COMPLETED')
            else:
                # Count how many models exist for this range
                existing_models = 0
                total_models = 0
                for dim in range(1, 4):
                    for t in range(start_idx, end_idx):
                        if training_method == 'ensemble':
                            for model_idx in range(5):
                                total_models += 1
                                fn_path = os.path.join(scan_dir, f"FN_dim{dim}_t{t}_model{model_idx}.pth")
                                if os.path.exists(fn_path):
                                    existing_models += 1
                        else:
                            total_models += 1
                            fn_path = os.path.join(scan_dir, f"FN_dim{dim}_t{t}.pth")
                            if os.path.exists(fn_path):
                                existing_models += 1
                
                progress = (existing_models / total_models) * 100 if total_models > 0 else 0
                print(f'  ○ Range {range_name}: {existing_models}/{total_models} models ({progress:.1f}%)')
        
        # Choose training method (always ask, regardless of existing models)
        print("\n" + "="*60)
        print("NEURAL NETWORK TRAINING METHOD SELECTION")
        print("="*60)
        print("1. Single Neural Network (train_FN_each_dimension)")
        print("2. Ensemble Method (train_FN_ensemble)")
        print("="*60)
        
        while True:
            method_choice = input("\nChoose training method (1 or 2): ").strip()
            if method_choice in ['1', '2']:
                break
            else:
                print("Please enter '1' or '2'.")
        
        # Update save directory based on method choice
        if method_choice == '1':
            # Single neural network method
            model_save_dir = os.path.join(model_PATH, f'second_stage_{args.RESIDUAL_SAMPLES}_single')
            chosen_method = 'single'
        else:
            # Ensemble method
            model_save_dir = os.path.join(model_PATH, f'second_stage_{args.RESIDUAL_SAMPLES}')
            chosen_method = 'ensemble'
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        print(f'[INFO] Using model save directory: {model_save_dir}')
        
        # Re-check completion based on chosen method
        print(f'\n[INFO] Re-checking completion status for {chosen_method} method...')
        
        # First, detect what ensemble models actually exist in the chosen directory
        if chosen_method == 'ensemble':
            print(f'[DEBUG] Scanning ensemble directory for existing models: {model_save_dir}')
            ensemble_model_files = []
            if os.path.exists(model_save_dir):
                ensemble_model_files = [f for f in os.listdir(model_save_dir) if f.startswith('FN_dim') and f.endswith('.pth')]
            
            print(f'[DEBUG] Found {len(ensemble_model_files)} ensemble model files')
            
            # Extract time steps from ensemble models
            ensemble_time_steps = set()
            for file in ensemble_model_files:
                parts = file.replace('.pth', '').split('_')
                if len(parts) >= 3 and parts[1].startswith('dim') and parts[2].startswith('t'):
                    try:
                        time_step = int(parts[2][1:])
                        ensemble_time_steps.add(time_step)
                    except ValueError:
                        continue
            
            if ensemble_time_steps:
                ensemble_time_steps = sorted(list(ensemble_time_steps))
                max_ensemble_step = max(ensemble_time_steps)
                print(f'[DEBUG] Ensemble models exist for time steps: {ensemble_time_steps[:10]}...' + (f' (and {len(ensemble_time_steps)-10} more)' if len(ensemble_time_steps) > 10 else ''))
                print(f'[DEBUG] Maximum ensemble time step: {max_ensemble_step}')
                
                # Create new time ranges starting from max_ensemble_step + 1
                start_from = max_ensemble_step + 1
                total_time_steps = ODE_Solution.shape[2]
                remaining_steps = total_time_steps - start_from
                
                if remaining_steps > 0:
                    # Divide remaining steps into 4 parts
                    steps_per_range = remaining_steps // 4
                    extra_steps = remaining_steps % 4
                    
                    time_ranges = []
                    current_start = start_from
                    
                    for i in range(4):
                        current_steps = steps_per_range + (1 if i < extra_steps else 0)
                        current_end = min(current_start + current_steps, total_time_steps)
                        time_ranges.append((current_start, current_end))
                        current_start = current_end
                        
                        if current_end >= total_time_steps:
                            break
                    
                    print(f'[INFO] Created ensemble continuation ranges: {time_ranges}')
                    print(f'[INFO] Will continue ensemble training from time step {start_from} to {total_time_steps}')
                else:
                    time_ranges = []
                    print(f'[INFO] All ensemble time steps (0-{total_time_steps-1}) are already trained!')
            else:
                # No ensemble models exist, start from 0
                total_time_steps = ODE_Solution.shape[2]
                steps_per_range = total_time_steps // 4
                extra_steps = total_time_steps % 4
                
                time_ranges = []
                current_start = 0
                
                for i in range(4):
                    current_steps = steps_per_range + (1 if i < extra_steps else 0)
                    current_end = min(current_start + current_steps, total_time_steps)
                    time_ranges.append((current_start, current_end))
                    current_start = current_end
                    
                    if current_end >= total_time_steps:
                        break
                
                print(f'[INFO] No ensemble models found, created initial ranges: {time_ranges}')
        
        # Now check completion for the chosen method
        completed_ranges = []
        incomplete_ranges = []
        
        for start_idx, end_idx in time_ranges:
            range_complete = True
            for dim in range(1, 4):  # dimensions 1, 2, 3
                for t in range(start_idx, end_idx):
                    if chosen_method == 'ensemble':
                        # Check for ensemble models (5 models per time point)
                        for model_idx in range(5):
                            fn_path = os.path.join(model_save_dir, f"FN_dim{dim}_t{t}_model{model_idx}.pth")
                            if not os.path.exists(fn_path):
                                range_complete = False
                                break
                        if not range_complete:
                            break
                    else:
                        # Check for single model
                        fn_path = os.path.join(model_save_dir, f"FN_dim{dim}_t{t}.pth")
                        if not os.path.exists(fn_path):
                            range_complete = False
                            break
                if not range_complete:
                    break
            
            if range_complete:
                completed_ranges.append((start_idx, end_idx))
            else:
                incomplete_ranges.append((start_idx, end_idx))
        
        # Check if all ranges are complete
        fn_models_exist = len(incomplete_ranges) == 0
        
        # Print updated status
        if completed_ranges:
            print(f'[INFO] Completed time ranges ({chosen_method}): {completed_ranges}')
        if incomplete_ranges:
            print(f'[INFO] Incomplete time ranges ({chosen_method}): {incomplete_ranges}')
        
        print(f'[DEBUG] Total ranges: {len(time_ranges)}')
        print(f'[DEBUG] Completed ranges: {len(completed_ranges)}')
        print(f'[DEBUG] Incomplete ranges: {len(incomplete_ranges)}')
        print(f'[DEBUG] fn_models_exist: {fn_models_exist}')
        
        if not fn_models_exist:
            print('[INFO] Training FN models for incomplete time ranges...')
            
            # Save time range configuration if it doesn't exist
            time_range_config_path = os.path.join(model_save_dir, 'time_range_config.npy')
            if not os.path.exists(time_range_config_path):
                time_range_config = {
                    'time_ranges': time_ranges,
                    'total_time_steps': ODE_Solution.shape[2],
                    'dt': dt,
                    'num_models_per_ensemble': 5 if method_choice == '2' else 1,
                    'num_dimensions': 3,
                    'training_method': 'ensemble' if method_choice == '2' else 'single'
                }
                np.save(time_range_config_path, time_range_config)
                print(f'[INFO] Time range configuration saved to: {time_range_config_path}')
            
            # Train only the incomplete ranges
            for start_idx, end_idx in incomplete_ranges:
                print(f"\n[INFO] Training for time range {start_idx}-{end_idx}...")
                
                if method_choice == '1':
                    # Single neural network method
                    print(f"[INFO] Using single neural network method...")
                    train_FN_each_dimension(
                        ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=model_save_dir,
                        time_range=(start_idx, end_idx),  # Use specific time range
                        dt=dt
                    )
                else:
                    # Ensemble method
                    print(f"[INFO] Using ensemble method...")
                    train_FN_ensemble(
                        ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=model_save_dir,
                        time_range=(start_idx, end_idx),  # Use specific time range
                        dt=dt
                    )
            
            # Final status update
            print('\n[INFO] Training completed. Final status:')
            final_completed = []
            final_incomplete = []
            
            # Re-detect time ranges after training
            model_files = []
            for file in os.listdir(model_save_dir):
                if file.startswith('FN_dim') and file.endswith('.pth'):
                    model_files.append(file)
            
            time_steps = set()
            final_training_method = 'unknown'
            for file in model_files:
                parts = file.replace('.pth', '').split('_')
                if len(parts) >= 3 and parts[1].startswith('dim') and parts[2].startswith('t'):
                    try:
                        time_step = int(parts[2][1:])
                        time_steps.add(time_step)
                        
                        # Detect training method
                        if len(parts) >= 4 and parts[3].startswith('model'):
                            final_training_method = 'ensemble'
                        else:
                            final_training_method = 'single'
                    except ValueError:
                        continue
            
            if time_steps:
                time_steps = sorted(list(time_steps))
                # Group consecutive time steps into ranges
                final_time_ranges = []
                start_idx = time_steps[0]
                prev_step = time_steps[0]
                
                for step in time_steps[1:]:
                    if step != prev_step + 1:
                        final_time_ranges.append((start_idx, prev_step + 1))
                        start_idx = step
                    prev_step = step
                
                final_time_ranges.append((start_idx, prev_step + 1))
                
                # Check completion status for each range
                for start_idx, end_idx in final_time_ranges:
                    range_complete = True
                    for dim in range(1, 4):
                        for t in range(start_idx, end_idx):
                            if final_training_method == 'ensemble':
                                for model_idx in range(5):
                                    fn_path = os.path.join(model_save_dir, f"FN_dim{dim}_t{t}_model{model_idx}.pth")
                                    if not os.path.exists(fn_path):
                                        range_complete = False
                                        break
                                if not range_complete:
                                    break
                            else:
                                fn_path = os.path.join(model_save_dir, f"FN_dim{dim}_t{t}.pth")
                                if not os.path.exists(fn_path):
                                    range_complete = False
                                    break
                        if not range_complete:
                            break
                    
                    if range_complete:
                        final_completed.append((start_idx, end_idx))
                    else:
                        final_incomplete.append((start_idx, end_idx))
            
            if final_completed:
                print(f'  ✓ Completed ranges: {final_completed}')
            if final_incomplete:
                print(f'  ○ Incomplete ranges: {final_incomplete}')
            else:
                print('  ✓ All detected time ranges completed successfully!')
        else:
            print('[INFO] All FN models already exist, skipping training process.')
            print(f'[INFO] Found models for dimensions 1-3,')
            
            # Load time range configuration
            time_range_config_path = os.path.join(save_dir, 'time_range_config.npy')
            if os.path.exists(time_range_config_path):
                time_range_config = np.load(time_range_config_path, allow_pickle=True).item()
                time_ranges = time_range_config['time_ranges']
                print(f'[INFO] Loaded time range configuration:')
                print(f'  - Time ranges: {time_ranges}')
                print(f'  - Total time steps: {time_range_config["total_time_steps"]}')
                print(f'  - Models per ensemble: {time_range_config["num_models_per_ensemble"]}')
            else:
                print('[WARNING] Time range configuration file not found')  
    
    print("\n"+ "="*60)
    print("\n[INFO] Testing predictions in choice 2...")
    
    
    print("="*60)
    print('[SUCCESS] training process finished.')

elif choice == '2':
    print("\n[INFO] Skip training and doing the prediction...")
    
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
        save_dir_single = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_1.0/second_stage_10000_single'
        save_dir_ensemble = f'../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_1.0/second_stage_10000'
    else:
        save_dir_single = f'../src/Example/MC_triad/Results/{args.params_name}/noise_1.0/second_stage_10000_single'
        save_dir_ensemble = f'../src/Example/MC_triad/Results/{args.params_name}/noise_1.0/second_stage_10000'

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
        Npath = current_pred_state.shape[0]
        dim = current_pred_state.shape[1]
        Winc_tensor = torch.Tensor(Winc).to(device, dtype=torch.float32)
    
        # Use the simple step update function for neural networks
        from utils.ODEParser import simple_step_update
    
        # Try both single and ensemble neural networks
        stoch_update_single = simple_step_update(
        Winc_tensor=Winc_tensor,
        device=device,
        idx=idx,
        save_dir_single=save_dir_single,
        save_dir_ensemble=save_dir_ensemble,
        model_type='single',
        scaler=scaler
        )
    
        stoch_update_ensemble = simple_step_update(
        Winc_tensor=Winc_tensor,
        device=device,
        idx=idx,
        save_dir_single=save_dir_single,
        save_dir_ensemble=save_dir_ensemble,
        model_type='ensemble',
        scaler=scaler
        )
    
        # Simple noise for comparison
        simple_noise = np.sqrt(dt) * (Winc @ SS)
    
        # Print comparison every 50 steps
        if idx % 50 == 0:
            print(f"\nStep {idx}: Model Comparison")
            print("=" * 50)
        
            if stoch_update_single is not None:
                print(f"Single NN - Mean: {np.mean(stoch_update_single, axis=0)}")
                print(f"Single NN - Std:  {np.std(stoch_update_single, axis=0)}")
            else:
                print("Single NN - Not available")
            
            if stoch_update_ensemble is not None:
                print(f"Ensemble NN - Mean: {np.mean(stoch_update_ensemble, axis=0)}")
                print(f"Ensemble NN - Std:  {np.std(stoch_update_ensemble, axis=0)}")
            else:
                print("Ensemble NN - Not available")
            
            print(f"Simple Noise - Mean: {np.mean(simple_noise, axis=0)}")
            print(f"Simple Noise - Std:  {np.std(simple_noise, axis=0)}")
            print("=" * 50)
    
        # Choose which model to use (priority: ensemble > single > simple noise)
        if stoch_update_ensemble is not None:
            stoch_update = stoch_update_ensemble
            model_used = "Ensemble"
        elif stoch_update_single is not None:
            stoch_update = stoch_update_single
            model_used = "Single"
        else:
            print("Single NN - Not available")
            
        # Fallback to simple noise
        stoch_update = simple_noise
        model_used = "Simple"
    
        # Compute both single and ensemble predictions
        if stoch_update_single is not None:
            next_pred_single = current_pred_state + det_update + stoch_update_single
        else:
            next_pred_single = current_pred_state + det_update + simple_noise
        
        if stoch_update_ensemble is not None:
            next_pred_ensemble = current_pred_state + det_update + stoch_update_ensemble
        else:
            next_pred_ensemble = current_pred_state + det_update + simple_noise
    
        # Use the selected model for the main prediction (for backward compatibility)
        next_pred_state = current_pred_state + det_update + stoch_update
    
        # Store results for all three predictions
        u_pred_all[:,:,idx] = next_pred_state
        u_pred_single[:,:,idx] = next_pred_single
        u_pred_ensemble[:,:,idx] = next_pred_ensemble
    
        # Update statistics for all three predictions
        mean_state_pred[:,idx] = np.mean(next_pred_state, axis=0)
        mean_state_single[:,idx] = np.mean(next_pred_single, axis=0)
        mean_state_ensemble[:,idx] = np.mean(next_pred_ensemble, axis=0)
    
        cov_state_pred[:,:,idx] = np.cov(next_pred_state, rowvar=False)
        cov_state_single[:,:,idx] = np.cov(next_pred_single, rowvar=False)
        cov_state_ensemble[:,:,idx] = np.cov(next_pred_ensemble, rowvar=False)

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
    print(f"Model used for stochastic component: {model_used}")
    print(f"Simulation time: {TIME_AMOUNT}")
    print(f"Time step: {dt}")
    print(f"Number of paths: {NPATH}")
    print(f"Total steps: {int(TIME_AMOUNT/dt)}")

    print("="*80)

    # Generate plots
    print("\nGenerating comparison plots...")

    # Create save directory for plots
    import os
    if str(device)=="cuda:0":
        save_dir = f"../src/Example/MC_triad/Results/Results1/Results/{args.params_name}/noise_{args.NOISE_LEVEL}/plots"
    else:
        save_dir = f"../src/Example/MC_triad/Results/{args.params_name}/noise_{args.NOISE_LEVEL}/plots"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}")

    # Import plotting functions
    from utils.plot import plot_mean_comparison, plot_covariance_comparison, plot_energy_comparison, plot_third_order_moments, plot_probability_distributions

    # Plot mean and covariance comparisons
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

    print("[INFO] All plots saved successfully!")
    print("\n")
    print("[SUCCESS] have already finished prediction! Now you finish the work!")
    
    
    
    
    