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
    train_FN_ensemble,
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
    if os.path.exists(model_PATH):
        save_dir = os.path.join( model_PATH,'Results',args.params_name,f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}')
        print('[INFO] Right now we use our own workspace path.') 
    else:
        model_PATH = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results', args.params_name))
        save_dir = os.path.join(config.DIR_TRIAD,'Results', 'Results',args.params_name,f'noise_{args.NOISE_LEVEL}',f'second_stage_{args.RESIDUAL_SAMPLES}')
        print('[INFO] Right now we use hipergator workspace path.')
os.makedirs(save_dir,exist_ok=True)
print(f'[INFO] The save directory is {save_dir}')
print("="*60)
#=================================================================================


# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Train to learn stochastic part in noise level and num samples")
print("2. Skip to calculate the measurements")
print("3. Skip all and plot the results")
print("="*60)

while True:
# #choice = '1' #
    choice = input("\nChoose option (1 or 2 or 3):").strip()
    if choice in ['1','2','3']:
        break
    else:
        print("Please enter '1' or '2' or '3'.")

if choice == '1':
    # Option 1: Train everything in second stage
    print("\n[INFO] Training everything in second stage...")
    
    # Add noise level selection
    print("\n" + "="*60)
    print(f"NOISE LEVEL SELECTION for {args.NOISE_LEVEL}")
    print("="*60)
   
   
    if not os.path.exists(os.path.join(save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz')):
        raise RuntimeError('[ERROR] data has not been generated, you should run the first_stage_deterministic.py first')
    else:
        print('[SUCCESS] data has already been generated in this folder. Now you can train the following.')
        pass
    data = np.load(os.path.join(save_dir,'..',f'simulation_results_noise_{args.NOISE_LEVEL}.npz'))
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

#     np.save(os.path.join(save_dir,'residual_cov_truth.npy'), residual_cov_truth)
#     print(f'[INFO] the residual shape is {residuals.shape},the state of dyamics is {u_current.shape}')
#     scaler = np.ones(3) * args.DIFF_SCALE
#     train_size = args.NUM_SAMPLES
#     #===================================================================================
#     ODE_Solution,ZT_Solution = generate_second_step(
#         u_current, residuals, scaler, dt, train_size, device,
#         num_time_points=101  # Only process 100 time points
#     )
#     print(f'[INFO] the ODE solution shape is: {ODE_Solution.shape}')
#     mean_value, std_value = generate_mean_and_std(ODE_Solution)
#     print(f'[INFO] this is print for mean and std: {mean_value.shape} {std_value.shape}')
#     # Train ensemble models for better accuracy (only on selected time points)
#     train_FN_ensemble(
#         ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir,
#         num_time_points=101,  # Only train on 100 time points
#         dt=dt
#     )
    
#     print("\n"+ "="*60)
#     print("\n[INFO] Testing ensemble predictions...")
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
#     print("="*60)
#     print('[SUCCESS] training process finished.')

# elif choice == '2':
#     print("\n[INFO] Skip training and calculate the measurements...")
#     # Load the data
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
    
# elif choice == '3':
#     print("\n[INFO] Skip all and plot the results...")
#     pass

# #===========================Plotting Section==============================================
# print("\n"+ "="*60)
# print("[INFO] Creating plots and visualizations...")
# print("="*60)

# # Auto-detect existing cases in equipart directory
# equipart_dir = config.DIR_EQUIPART
# if not os.path.exists(equipart_dir):
#     equipart_dir = Path(os.path.join(config.DIR_TRIAD, 'Results', 'Results', 'equipart'))

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
    
    