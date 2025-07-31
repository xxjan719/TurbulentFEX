import config
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from utils.ODEParser import select_time_points, FN_Net
import torch
from config import DIR_EXAMPLE, DIR_TRIAD,create_main_parser
from utils.plot import plot_NOISE_LEVEL_EFFECT
from utils import *
parser = create_main_parser()
args = parser.parse_args()
# Check if CUDA is available and set device accordingly
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results','Results')
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results')
    
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'
    
    
coefficients = get_coefficients(load_dir= DIR_TRIAD, DEVICE=args.DEVICE)
plot_NOISE_LEVEL_EFFECT(coefficients,save_dir=args.LOG_SAVE_PATH)




# selected_indices, selected_times = select_time_points(1000, 0.01, 101)
# selected_indices = np.arange(101)

# save_dir = 'Example/MC_triad/Results/equipart/case_20000'
# residual_cov_truth = np.load(os.path.join(save_dir, 'residual_cov_truth.npy'))
# dt = 0.01
# scaler = np.array([20,20,20])
# print(f"\n=== Ensemble Prediction Results ===")
# print(f"Processing {len(selected_indices)} time points out of {1000} total")
# print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
    
# residual_cov_pred = np.zeros((len(selected_indices), 3))
# n_models = 5
# device = 'cpu'

# # Load and use ensemble predictions for each selected time point
# for i, t in enumerate(selected_indices):
#     z_test = np.random.randn(1000, 3)
#     z_test_tensor = torch.tensor(z_test, dtype=torch.float32).to(device)
        
#     for dim in range(1, 4):
#         # Load normalization parameters
#         norm_params_path = os.path.join(save_dir, f'norm_params_dim{dim}_t{t}.npy')
#         if not os.path.exists(norm_params_path):
#             print(f"Warning: Normalization parameters not found for dim{dim}_t{t}")
#             continue
                
#         norm_params = np.load(norm_params_path, allow_pickle=True).item()
#         y_mean = norm_params['mean']
#         y_std = norm_params['std']
            
#         # Ensemble prediction
#         ensemble_predictions = []
            
#         for model_idx in range(n_models):
#             model_path = os.path.join(save_dir, f'FN_dim{dim}_t{t}_model{model_idx}.pth')
#             if not os.path.exists(model_path):
#                 print(f"Warning: Model not found: {model_path}")
#                 continue
                    
#             FN_dim = FN_Net(1, 1, 100).to(device)
#             FN_dim.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
                
#             # Make prediction
#             pred = (FN_dim(z_test_tensor[:, dim-1:dim].reshape(-1, 1))).cpu().detach().numpy()
#             ensemble_predictions.append(pred)
            
#         if not ensemble_predictions:
#             print(f"Warning: No valid predictions for dim{dim}_t{t}")
#             continue
                
#         # Average ensemble predictions
#         pred = np.mean(ensemble_predictions, axis=0)
            
#         # Denormalize: pred * y_std + y_mean
#         pred = pred * y_std + y_mean
            
#         # Scale back by scaler
#         pred = pred / scaler[dim-1]
#         residual_cov_pred[i, dim-1] = np.std(pred) / np.sqrt(dt)
            
#         print(f"Time {selected_times[i]:.2f}s, Dimension {dim}: {np.std(pred)/np.sqrt(dt):.6f}")
            
#         if residual_cov_truth is not None:
#             print(f"Comparison with Ground Truth: {residual_cov_truth[t, dim-1]:.6f}")
    
#     print("\nExpected values should be close to original")
#     print("Ensemble method should provide more accurate results!")

# np.save(os.path.join(save_dir, 'residual_cov_pred.npy'), residual_cov_pred)




#--- CASE 1000 ---
# selected_indices_1000, selected_times_1000 = select_time_points(1000, 0.01, 101)
# print(selected_indices_1000)
# print(selected_times_1000)
# cov_truth_1000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_1000','residual_cov_truth.npy'))
# print(cov_truth_1000.shape)
# cov_truth_slice_1000 = cov_truth_1000[selected_indices_1000]
# print(cov_truth_slice_1000.shape)

# cov_pred_1000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_1000','residual_cov_pred.npy'))

# # --- CASE 5000 ---
# selected_indices_5000, selected_times_5000 = select_time_points(1000, 0.01, 101)
# cov_truth_5000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_5000','residual_cov_truth.npy'))
# selected_indices_5000 = np.clip(selected_indices_5000, 0, cov_truth_5000.shape[0] - 1)
# cov_truth_slice_5000 = cov_truth_5000[selected_indices_5000]
# cov_pred_5000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_5000','residual_cov_pred.npy'))

# # --- CASE 10000 ---
# selected_indices_10000, selected_times_10000 = select_time_points(1000, 0.01, 101)
# cov_truth_10000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_10000','residual_cov_truth.npy'))
# selected_indices_10000 = np.clip(selected_indices_10000, 0, cov_truth_10000.shape[0] - 1)
# cov_truth_slice_10000 = cov_truth_10000[selected_indices_10000]
# cov_pred_10000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_10000','residual_cov_pred.npy'))

# # --- CASE 20000 ---
# selected_indices_20000, selected_times_20000 = select_time_points(1000, 0.01, 101)
# cov_truth_20000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_20000','residual_cov_truth.npy'))
# selected_indices_20000 = np.clip(selected_indices_20000, 0, cov_truth_20000.shape[0] - 1)
# cov_truth_slice_20000 = cov_truth_20000[selected_indices_20000]
# cov_pred_20000 = np.load(os.path.join(config.DIR_EQUIPART, 'case_20000','residual_cov_pred.npy'))

# data_list = [{
#             'residual_cov_pred': cov_pred_1000,
#             'residual_cov_truth': cov_truth_1000,
#             'selected_indices': selected_indices_1000,
#             'sample_size': 1000,
#             'case_dir': f'case_1000'
#         }]

# selected_times_list = [selected_indices_1000]
# labels = ['1000']

# N = 101  # Number of points to plot

# fig, axes = plt.subplots(3, 1, figsize=(12, 10))
# dimensions = ['u1', 'u2', 'u3']
# colors = ['blue', 'red', 'green', 'purple']
# labels = ['case_1000', 'case_5000', 'case_10000', 'case_20000']

# for dim in range(3):
#     ax = axes[dim]
#     # Case 1000
#     pred_1000 = cov_pred_1000[:, dim]
#     truth_1000 = cov_truth_slice_1000[:, dim]
#     log10_error_1000 = np.log10(np.abs(pred_1000 - truth_1000) + 1e-12)
#     ax.plot(selected_times_1000[:N], log10_error_1000[:N], 'o-', color=colors[0], alpha=0.8, linewidth=2, markersize=4, label=labels[0])
#     # Case 5000
#     pred_5000 = cov_pred_5000[:, dim]
#     truth_5000 = cov_truth_slice_5000[:, dim]
#     log10_error_5000 = np.log10(np.abs(pred_5000 - truth_5000) + 1e-12)
#     ax.plot(selected_times_5000[:N], log10_error_5000[:N], 's--', color=colors[1], alpha=0.8, linewidth=2, markersize=4, label=labels[1])
#     # Case 10000
#     pred_10000 = cov_pred_10000[:, dim]
#     truth_10000 = cov_truth_slice_10000[:, dim]
#     log10_error_10000 = np.log10(np.abs(pred_10000 - truth_10000) + 1e-12)
#     ax.plot(selected_times_10000[:N], log10_error_10000[:N], 'd-.', color=colors[2], alpha=0.8, linewidth=2, markersize=4, label=labels[2])
#     # Case 20000
#     pred_20000 = cov_pred_20000[:, dim]
#     truth_20000 = cov_truth_slice_20000[:, dim]
#     log10_error_20000 = np.log10(np.abs(pred_20000 - truth_20000) + 1e-12)
#     ax.plot(selected_times_20000[:N], log10_error_20000[:N], 'x:', color=colors[3], alpha=0.8, linewidth=2, markersize=4, label=labels[3])
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel(f'log10(|Error|) - {dimensions[dim]}')
#     ax.set_title(f'Dimension {dim+1} ({dimensions[dim]}) - Log10 Error (First {N} Points)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()






# # Parse the log file to extract ground truth values
# log_file_path = 'slurm/QIDIFEX_SAMPLE5000.out'
# ground_truth_data = {'times': [], 'dim1': [], 'dim2': [], 'dim3': []}

# if os.path.exists(log_file_path):
#     with open(log_file_path, 'r') as f:
#         content = f.read()
    
#     # Find the section with ground truth comparisons
#     pattern = r'Time ([\d.]+)s, Dimension (\d): [\d.]+[\s\S]*?Comparison with Ground Truth: ([\d.]+)'
#     matches = re.findall(pattern, content)
    
#     current_time = None
#     for match in matches:
#         time, dim, gt_value = match
#         time = float(time)
#         dim = int(dim)
#         gt_value = float(gt_value)
        
#         if time != current_time:
#             current_time = time
#             ground_truth_data['times'].append(time)
#             ground_truth_data['dim1'].append(None)
#             ground_truth_data['dim2'].append(None)
#             ground_truth_data['dim3'].append(None)
        
#         ground_truth_data[f'dim{dim}'][-1] = gt_value
    
#     # Convert to numpy arrays
#     ground_truth_data['times'] = np.array(ground_truth_data['times'])
#     ground_truth_data['dim1'] = np.array(ground_truth_data['dim1'])
#     ground_truth_data['dim2'] = np.array(ground_truth_data['dim2'])
#     ground_truth_data['dim3'] = np.array(ground_truth_data['dim3'])
    
#     print(f"\nExtracted {len(ground_truth_data['times'])} time points from log")
#     print("Ground truth values from log:")
#     for i in range(min(10, len(ground_truth_data['times']))):
#         print(f"Time {ground_truth_data['times'][i]:.2f}s: "
#               f"Dim1={ground_truth_data['dim1'][i]:.6f}, "
#               f"Dim2={ground_truth_data['dim2'][i]:.6f}, "
#               f"Dim3={ground_truth_data['dim3'][i]:.6f}")
    
#     # Compare with sliced data
#     print(f"\nComparison with sliced data (first 10 points):")
#     for i in range(min(10, len(ground_truth_data['times']))):
#         log_idx = i
#         slice_idx = i
#         if slice_idx < len(cov_truth_slice_1000):
#             print(f"Time {ground_truth_data['times'][log_idx]:.2f}s:")
#             print(f"  Log GT:     Dim1={ground_truth_data['dim1'][log_idx]:.6f}, "
#                   f"Dim2={ground_truth_data['dim2'][log_idx]:.6f}, "
#                   f"Dim3={ground_truth_data['dim3'][log_idx]:.6f}")
#             print(f"  Slice data: Dim1={cov_truth_slice_1000[slice_idx, 0]:.6f}, "
#                   f"Dim2={cov_truth_slice_1000[slice_idx, 1]:.6f}, "
#                   f"Dim3={cov_truth_slice_1000[slice_idx, 2]:.6f}")
#             print()
# else:
#     print(f"Log file not found: {log_file_path}")







