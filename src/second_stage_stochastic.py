import numpy as np
import os
import sys
from pathlib import Path
# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from utils.ODEParser import (
    generate_euler_residue, 
    generate_second_step, 
    generate_mean_and_std, 
    train_FN_each_dimension,
    train_FN_ensemble,
    predict_single_model_residual_covariance,
    predict_ensemble_residual_covariance
)
from utils.FEX import FEX_model_ground_truth,FEX_model_learned
from utils.plot import (
    plot_residual_covariance_comparison,
    plot_log10_error,
    plot_multiple_residual_covariance,
    plot_multiple_log10_error,
    plot_time_selection_info
)

from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
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
if args.Model == 'MC_triad':
    model_PATH = config.DIR_TRIAD
    # Default save directory (will be updated based on method choice)
    save_dir = os.path.join( model_PATH,'Results',args.params_name,f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}_single')
    if os.path.exists(save_dir):   
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
    common_save_dir = os.path.join(model_PATH, 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_common')
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
    # residuals, u_current, residual_cov_truth = generate_euler_residue(FEX_model_ground_truth, data, dt)
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
    ensemble_dir = os.path.join(model_PATH, 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}')
    single_dir = os.path.join(model_PATH, 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_single')
    
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
            model_save_dir = os.path.join(model_PATH, 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}_single')
            chosen_method = 'single'
        else:
            # Ensemble method
            model_save_dir = os.path.join(model_PATH, 'Results', args.params_name, f'noise_{args.NOISE_LEVEL}', f'second_stage_{args.RESIDUAL_SAMPLES}')
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
    print("\n[INFO] Testing predictions...")
    
    if chosen_method == 'ensemble':
        print("[INFO] Using ensemble prediction method...")
        residual_cov_pred, selected_times = predict_ensemble_residual_covariance(
            residuals=residuals,
            save_dir=model_save_dir,  # Use the model save directory
            dt=dt,
            scaler=scaler,
            u_current=u_current,
            fex_model_func=learned_model_wrapper,
            train_size=train_size,
            n_models=5,
            device=device,
            residual_cov_truth=residual_cov_truth,
            num_time_points=1001  # Only predict on 1001 time points
        )
        print('[SUCCESS] Ensemble prediction completed.')
    else:
        print("[INFO] Using single model prediction method...")
        predict_single_model_residual_covariance(
            residuals=residuals,
            save_dir=model_save_dir,  # Use the model save directory
            dt=dt,
            scaler=scaler,
            u_current=u_current,
            fex_model_func=learned_model_wrapper,
            train_size=train_size,
            device=device,
            residual_cov_truth=residual_cov_truth,
            num_time_points=1001  # Only predict on 1001 time points
        )
        print('[SUCCESS] Single model prediction completed.')
    
    print("="*60)
    print('[SUCCESS] training process finished.')

elif choice == '2':
    print("\n[INFO] Skip training and deducing the performances...")
    # Load the data
#     data = np.load(os.path.join(save_dir,'simulation_data.npz'))
#     dt = 0.01
#     residuals, u_current, residual_cov_truth = generate_rk4_residue(FEX_model_check, data, dt)
#     scaler = np.ones(3) * args.DIFF_SCALE
#     train_size = args.TRAIN_SIZE
    
#     residual_cov_pred, selected_times = predict_ensemble_residual_covariance(
#         residuals=residuals,
#         save_dir=save_dir,
#         dt=dt,
#         scaler=scaler,
#         train_size=train_size,
#         n_models=5,
#         device=device,
#         residual_cov_truth=residual_cov_truth,
#         num_time_points=101  # Only predict on 100 time points
#     )
#     print('[SUCCESS] Ensemble prediction completed.')
    


#===========================Plotting Section==============================================
# print("\n"+ "="*60)
# print("[INFO] Creating plots and visualizations...")
# print("="*60)

# # Auto-detect existing cases in equipart directory
# equipart_dir = config.DIR_EQUIPART
# if not os.path.exists(equipart_dir):
#     equipart_dir = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results1', 'Results', 'equipart'))

# print(f"[INFO] Looking for cases in: {equipart_dir}")

# # Find all case directories
# case_dirs = []
# sample_sizes = []
# for item in os.listdir(equipart_dir):
#     if item.startswith('case_') and os.path.isdir(os.path.join(equipart_dir, item)):
#         try:
#             sample_size = int(item.split('_')[1])
#             case_dirs.append(item)
#             sample_sizes.append(sample_size)
#             print(f"[INFO] Found case: {item} (sample size: {sample_size})")
#         except (ValueError, IndexError):
#             continue

# if not case_dirs:
#     print("[WARNING] No case directories found!")
#     case_dirs = [f'case_{args.NUM_SAMPLES}']
#     sample_sizes = [args.NUM_SAMPLES]

# # Sort by sample size
# sorted_cases = sorted(zip(sample_sizes, case_dirs), key=lambda x: x[0])
# sample_sizes, case_dirs = zip(*sorted_cases)

# print(f"[INFO] Found {len(case_dirs)} cases: {case_dirs}")

# # Load data from each case
# data_list = []
# selected_times_list = []
# labels = []

# for sample_size, case_dir in zip(sample_sizes, case_dirs):
#     case_path = os.path.join(equipart_dir, case_dir)
    
#     # Check if residual data exists
#     residual_cov_truth_path = os.path.join(case_path, 'residual_cov_truth.npy')
#     residual_cov_pred_path = os.path.join(case_path, 'residual_cov_pred.npy')
    
#     if os.path.exists(residual_cov_truth_path) and os.path.exists(residual_cov_pred_path):
#         print(f"[INFO] Loading data from {case_dir}...")
        
#         # Load residual covariance data
#         residual_cov_truth = np.load(residual_cov_truth_path)
#         residual_cov_pred_data = np.load(residual_cov_pred_path, allow_pickle=True).item()
#         residual_cov_pred = residual_cov_pred_data['residual_cov_pred']
#         selected_times = residual_cov_pred_data['selected_times']
        
#         print(f"  - Ground truth shape: {residual_cov_truth.shape}")
#         print(f"  - Predicted shape: {residual_cov_pred.shape}")
#         print(f"  - Time points: {len(selected_times)}")
        
#         # Add to data lists
#         data_list.append({
#             'residual_cov_pred': residual_cov_pred,
#             'residual_cov_truth': residual_cov_truth,
#             'selected_times': selected_times,
#             'sample_size': sample_size,
#             'case_dir': case_dir
#         })
#         selected_times_list.append(selected_times)
#         labels.append(f'Sample Size {sample_size}')
        
#     else:
#         print(f"[WARNING] Missing residual data in {case_dir}")
#         if not os.path.exists(residual_cov_truth_path):
#             print(f"  - Missing: {residual_cov_truth_path}")
#         if not os.path.exists(residual_cov_pred_path):
#             print(f"  - Missing: {residual_cov_pred_path}")

# if not data_list:
#     print("[ERROR] No valid residual data found in any case!")
#     print("[INFO] Creating plots with current data only...")
    
#     # Fallback to current data if available
#     if 'residual_cov_pred' in locals() and 'residual_cov_truth' in locals():
#         data_list = [{
#             'residual_cov_pred': residual_cov_pred,
#             'residual_cov_truth': residual_cov_truth,
#             'selected_times': selected_times,
#             'sample_size': args.NUM_SAMPLES,
#             'case_dir': f'case_{args.NUM_SAMPLES}'
#         }]
#         selected_times_list = [selected_times]
#         labels = ['Current Method']
#     else:
#         print("[ERROR] No data available for plotting!")
#         exit(1)

# # Create plots directory
# plots_dir = os.path.join(save_dir, 'plots')
# os.makedirs(plots_dir, exist_ok=True)

# print(f"\n[INFO] Creating plots with {len(data_list)} datasets: {labels}")

# # 1. Multiple dataset comparison (3×1 Layout)
# print("\n[INFO] Creating multiple dataset comparison plots...")

# # Plot multiple residual covariance comparison
# plot_multiple_residual_covariance(
#     data_list, selected_times_list, plots_dir, labels
# )

# # Plot multiple log10 error comparison
# plot_multiple_log10_error(
#     data_list, selected_times_list, plots_dir, labels
# )

# # 2. Individual plots for each dataset
# print("\n[INFO] Creating individual comparison plots...")
# for i, (data, label) in enumerate(zip(data_list, labels)):
#     print(f"[INFO] Creating plots for {label}...")
    
#     # Create subdirectory for this dataset
#     dataset_plots_dir = os.path.join(plots_dir, f'dataset_{i+1}_{label.replace(" ", "_")}')
#     os.makedirs(dataset_plots_dir, exist_ok=True)
    
#     # Basic residual covariance comparison
#     plot_residual_covariance_comparison(
#         data['residual_cov_pred'], data['residual_cov_truth'], 
#         data['selected_times'], dataset_plots_dir
#     )
    
#     # Log10 error plot
#     plot_log10_error(
#         data['residual_cov_pred'], data['residual_cov_truth'], 
#         data['selected_times'], dataset_plots_dir
#     )

# # 3. Time selection information (use first dataset as reference)
# print("\n[INFO] Creating time selection information plot...")
# if data_list:
#     first_data = data_list[0]
#     selected_times = first_data['selected_times']
#     dt = selected_times[1] - selected_times[0] if len(selected_times) > 1 else 0.01
#     total_time_steps = int(selected_times[-1] / dt) + 1
#     selected_indices = np.round(selected_times / dt).astype(int)
    
#     plot_time_selection_info(
#         total_time_steps, selected_indices, selected_times, dt, plots_dir
#     )

# # 4. Save comprehensive results summary
# print("\n[INFO] Saving comprehensive results summary...")
# results_summary = {
#     'datasets': data_list,
#     'labels': labels,
#     'sample_sizes': sample_sizes,
#     'case_dirs': case_dirs
# }

# # Calculate error statistics for each dataset
# error_stats_list = []
# for i, (data, label) in enumerate(zip(data_list, labels)):
#     if data['residual_cov_truth'] is not None:
#         truth_data = data['residual_cov_truth']
#         selected_times = data['selected_times']
#         N_truth = truth_data.shape[0]
#         truth_times = np.linspace(selected_times[0], selected_times[-1], N_truth)
#         indices = np.searchsorted(np.round(truth_times, 8), np.round(selected_times, 8))
#         truth_selected = truth_data[indices, :]
#         error = np.abs(data['residual_cov_pred'] - truth_selected)
#         log10_error = np.log10(error + 1e-12)
#         error_stats = {
#             'dataset': label,
#             'sample_size': sample_sizes[i],
#             'mean_error': np.mean(error, axis=0),
#             'std_error': np.std(error, axis=0),
#             'mean_log10_error': np.mean(log10_error, axis=0),
#             'std_log10_error': np.std(log10_error, axis=0),
#             'max_error': np.max(error, axis=0),
#             'min_error': np.min(error, axis=0)
#         }
#         error_stats_list.append(error_stats)
#         print(f"\n=== Error Statistics for {label} ===")
#         dimensions = ['x1', 'x2', 'x3']
#         for dim in range(3):
#             print(f"Dimension {dim+1} ({dimensions[dim]}):")
#             print(f"  Mean Error: {error_stats['mean_error'][dim]:.6f}")
#             print(f"  Std Error: {error_stats['std_error'][dim]:.6f}")
#             print(f"  Mean Log10 Error: {error_stats['mean_log10_error'][dim]:.3f}")
#             print(f"  Max Error: {error_stats['max_error'][dim]:.6f}")

# results_summary['error_stats'] = error_stats_list

# np.save(os.path.join(plots_dir, 'comprehensive_results_summary.npy'), results_summary)

# print("\n"+ "="*60)
# print("[SUCCESS] All plots and analysis completed!")
# print(f"[INFO] Results saved in: {plots_dir}")
# print(f"[INFO] Created {len(data_list)} dataset comparisons")
# print("="*60)
    
    