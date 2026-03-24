from ast import Dict
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D
import os

def set_figure_position(x=100, y=100, width=800, height=600):
    """Set the position and size of the current figure window (only if supported)."""
    try:
        manager = plt.get_current_fig_manager()
        if manager is not None and hasattr(manager, 'window'):
            manager.window.setGeometry(x, y, width, height)  # type: ignore
    except Exception as e:
        print(f"Figure positioning skipped: {e}")

def plot_stats(TT_MC, mean_MC, cov_MC, M3_MC, Ene_MC, Ene_dyn, save_path=None):
    # Set publication style
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 100,
    })

    # Color definitions
    line_colors = {
        'u1': '#2171B5',
        'u2': '#D94801',
        'u3': '#238B45',
        'total': '#000000',
        'extra1': '#CC79A7',
        'extra2': '#E69F00',
        'extra3': '#56B4E9',
        'gray': '#888888'
    }

    fig, axs = plt.subplots(3, 2, figsize=(13, 11))
    set_figure_position(x=100, y=100, width=1300, height=1100)  # Set figure position
    axs = axs.flatten()

    std_MC = np.sqrt(np.clip(np.diagonal(cov_MC, axis1=0, axis2=1).T, a_min=0, a_max=None))

    # Mean
    for i, key in enumerate(['u1', 'u2', 'u3']):
        mask = mean_MC[i] != 0
        axs[0].scatter(TT_MC[mask], mean_MC[i][mask], s=9, color=line_colors[key], label=fr'$\langle u_{i+1} \rangle$')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mean')
    axs[0].legend(loc='upper right', frameon=False)

    # Variance
    for i, key in enumerate(['u1', 'u2', 'u3']):
        mask = cov_MC[i, i] != 0
        axs[1].scatter(TT_MC[mask], cov_MC[i, i][mask], s=9, color=line_colors[key], label=fr'$\mathrm{{Var}}(u_{i+1})$')
    total_var = np.sum(std_MC**2, axis=0)
    mask_total = total_var != 0
    axs[1].scatter(TT_MC[mask_total], total_var[mask_total], s=9, color=line_colors['total'], label='Total Var')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Variance')
    axs[1].legend(loc='upper left', frameon=False)

    # Cross-Covariance
    mask_01 = cov_MC[0, 1] != 0
    mask_02 = cov_MC[0, 2] != 0
    mask_12 = cov_MC[1, 2] != 0
    axs[2].scatter(TT_MC[mask_01], cov_MC[0, 1][mask_01], s=9, color=line_colors['extra1'], label=r'$\mathrm{Cov}(u_1, u_2)$')
    axs[2].scatter(TT_MC[mask_02], cov_MC[0, 2][mask_02], s=9, color=line_colors['extra2'], label=r'$\mathrm{Cov}(u_1, u_3)$')
    axs[2].scatter(TT_MC[mask_12], cov_MC[1, 2][mask_12], s=9, color=line_colors['extra3'], label=r'$\mathrm{Cov}(u_2, u_3)$')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Cross-Covariance')
    axs[2].legend(loc='lower left', frameon=False)

    # 3rd-Order Moments
    mask_012 = M3_MC[0, 1, 2] != 0
    mask_011 = M3_MC[0, 1, 1] != 0
    mask_022 = M3_MC[0, 2, 2] != 0
    mask_112 = M3_MC[1, 1, 2] != 0
    axs[3].scatter(TT_MC[mask_012], M3_MC[0, 1, 2][mask_012], s=9, color=line_colors['total'], label=r'$\langle M_{123} \rangle$')
    axs[3].scatter(TT_MC[mask_011], M3_MC[0, 1, 1][mask_011], s=9, color=line_colors['u1'], label=r'$\langle M_{122} \rangle$')
    axs[3].scatter(TT_MC[mask_022], M3_MC[0, 2, 2][mask_022], s=9, color=line_colors['u2'], label=r'$\langle M_{133} \rangle$')
    axs[3].scatter(TT_MC[mask_112], M3_MC[1, 1, 2][mask_112], s=9, color=line_colors['u3'], label=r'$\langle M_{223} \rangle$')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('3rd Moment')
    axs[3].legend(loc='upper right', frameon=False)

    # Energy (Truth)
    for i, key in enumerate(['total', 'u1', 'u2', 'u3']):
        mask = Ene_MC[i] != 0
        axs[4].scatter(TT_MC[mask], Ene_MC[i][mask], s=9, color=line_colors[key], label=['Total', 'Mode 1', 'Mode 2', 'Mode 3'][i])
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Energy (Truth)')
    axs[4].legend(loc='upper left', frameon=False)

    # Energy (Dynamics)
    ene_dyn_labels = ['Truth', 'Full Eqn.', 'Lower Bound', 'Upper Bound', 'Mean']
    ene_dyn_keys = ['total', 'extra1', 'extra2', 'extra3', 'gray']

    for i in range(Ene_dyn.shape[0]):
        key = ene_dyn_keys[i]
        label = ene_dyn_labels[i]
        mask = Ene_dyn[i] != 0
        axs[5].scatter(TT_MC[mask], Ene_dyn[i][mask], s=9, color=line_colors[key], label=label)
    axs[5].set_xlabel('Time')
    axs[5].set_ylabel('Energy (Dynamics)')
    axs[5].legend(loc='lower right', frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

def plot_third_order_moments(TT_MC, M3_MC, save_path = None):
    T = len(TT_MC)
    M112 = M3_MC[0,0,1,:].reshape(T)
    M113 = M3_MC[0,0,2,:].reshape(T)
    M233 = M3_MC[1,2,2,:].reshape(T)
    M111 = M3_MC[0,0,0,:].reshape(T)

    plt.figure(figsize=(8, 5))
    set_figure_position(x=100, y=100, width=800, height=500)  # Set figure position
    mask_112 = M112 != 0
    mask_113 = M113 != 0
    mask_233 = M233 != 0
    mask_111 = M111 != 0
    plt.scatter(TT_MC[mask_112], M112[mask_112],s=9, label=r'$\langle M_{112} \rangle$')
    plt.scatter(TT_MC[mask_113], M113[mask_113],s=9, label=r'$\langle M_{113} \rangle$')
    plt.scatter(TT_MC[mask_233], M233[mask_233], s=9,label=r'$\langle M_{233} \rangle$')
    plt.scatter(TT_MC[mask_111], M111[mask_111], s=9,label=r'$\langle M_{111} \rangle$')

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('3rd Order Central Moments', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Legend
    plt.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    

def plot_deviation_subplots(TT_MC, cov_MC, M3_MC_norm,save_path=None):
    cov_total = cov_MC[0, 0, :] + cov_MC[1, 1, :] + cov_MC[2, 2, :]
    trace_dev = (cov_total - cov_total[-1]) / cov_total[-1]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    set_figure_position(x=100, y=100, width=1800, height=500)  # Set figure position

    tol = 1e-8  # Tolerance for detecting constant lines

    # --- Subplot 1: Deviation in variance ---
    for i, color in zip(range(3), ['C0', 'C1', 'C2']):
        dev = (cov_MC[i, i, :] - cov_MC[i, i, -1]) / cov_MC[i, i, -1]
        mask = dev != -1
        axs[0].scatter(TT_MC[mask], dev[mask], s=9, color=color, label=fr'$u_{i+1}$')
    mask = trace_dev != -1
    axs[0].scatter(TT_MC[mask], trace_dev[mask], s=9, color='k', label='trace(R)')
    axs[0].set_xlabel('time')
    axs[0].set_title('deviation in variance')
    axs[0].legend()

    # --- Subplot 2: Deviation in cross-covariance ---
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [r'$(u_1,u_2)$', r'$(u_1,u_3)$', r'$(u_2,u_3)$']
    colors = ['C0', 'C1', 'C2']
    for (i, j), label, color in zip(pairs, labels, colors):
        num = cov_MC[i, j, :] - cov_MC[i, j, -1]
        denom = np.sqrt(cov_MC[i, i, :] * cov_MC[j, j, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.where(denom != 0, num / denom, 0.0)
            mask = dev != 0
            axs[1].scatter(TT_MC[mask], dev[mask], s=9, color=color, label=label)
    mask = trace_dev != -1
    axs[1].scatter(TT_MC[mask], trace_dev[mask], s=9,color='k',label='trace(R)')  # also plot
    axs[1].set_xlabel('time')
    axs[1].set_title('deviation in cross-covariance')
    axs[1].legend()

    # --- Subplot 3: Deviation in 3rd order central moments ---
    moments = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    labels = [r'$\langle M_{123} \rangle$', r'$\langle M_{122} \rangle$',
              r'$\langle M_{133} \rangle$', r'$\langle M_{223} \rangle$']
    colors = ['C0', 'C1', 'C2', 'C3']
    for (i, j, k), label, color in zip(moments, labels, colors):
        dev = M3_MC_norm[i, j, k, :] - M3_MC_norm[i, j, k, -1]
        # unique_vals = np.unique(np.round(dev, 8))  # round for numerical stability
        # print(f"{label}: unique y-values = {unique_vals}")
        axs[2].scatter(TT_MC, dev, s=9, color=color, label=label)
    axs[2].set_xlabel('time')
    axs[2].set_title('deviation in 3rd order central moments')
    axs[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_residual_covariance_comparison(residual_cov_pred, residual_cov_truth, selected_times, save_dir=None):
    """
    Plot comparison between predicted and ground truth residual covariance.
    
    Args:
        residual_cov_pred (np.ndarray): Predicted residual covariance
        residual_cov_truth (np.ndarray): Ground truth residual covariance
        selected_times (np.ndarray): Time points used for prediction
        save_dir (str): Directory to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dimensions = ['x1', 'x2', 'x3']
    colors = ['blue', 'red', 'green']
    
    for dim in range(3):
        ax = axes[dim]
        
        # Plot ground truth (if available)
        if residual_cov_truth is not None:
            # Interpolate ground truth to selected time points
            total_time_steps = residual_cov_truth.shape[0]
            dt = selected_times[1] - selected_times[0] if len(selected_times) > 1 else 0.01
            truth_times = np.arange(total_time_steps) * dt
            
            # Find indices for selected times in ground truth
            truth_indices = np.round(selected_times / dt).astype(int)
            truth_indices = np.clip(truth_indices, 0, total_time_steps - 1)
            
            ax.plot(truth_times[truth_indices], residual_cov_truth[truth_indices, dim], 
                   'o-', color='red', alpha=0.7, label='Ground Truth', linewidth=2)
        
        # Plot predictions
        ax.plot(selected_times, residual_cov_pred[:, dim], 
               's-', color='blue', alpha=0.8, label='Predicted', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Residual Covariance - {dimensions[dim]}')
        ax.set_title(f'Dimension {dim+1} ({dimensions[dim]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'residual_covariance_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(save_dir, 'residual_covariance_comparison.pdf')}")
    
    plt.show()

def plot_time_selection_info(total_time_steps, selected_indices, selected_times, dt, save_dir=None):
    """
    Plot information about time point selection.
    
    Args:
        total_time_steps (int): Total number of time steps
        selected_indices (np.ndarray): Selected time indices
        selected_times (np.ndarray): Selected time points
        dt (float): Time step size
        save_dir (str): Directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time point distribution
    all_times = np.arange(total_time_steps) * dt
    ax1.plot(all_times, np.ones_like(all_times), 'k-', alpha=0.3, label='All time points')
    ax1.plot(selected_times, np.ones_like(selected_times), 'ro', markersize=8, label='Selected time points')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Selection')
    ax1.set_title(f'Time Point Selection: {len(selected_indices)}/{total_time_steps} points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time intervals
    intervals = np.diff(selected_times)
    ax2.plot(selected_times[1:], intervals, 'bo-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Time Interval (s)')
    ax2.set_title('Time Intervals Between Selected Points')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'time_selection_info.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(save_dir, 'time_selection_info.png')}")
    
    plt.show()

def load_and_plot_results(save_dir):
    """
    Load saved results and create plots.
    
    Args:
        save_dir (str): Directory containing saved results
    """
    # Load results
    results_path = os.path.join(save_dir, 'residual_cov_pred.npy')
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        residual_cov_pred = results['residual_cov_pred']
        selected_times = results['selected_times']
        selected_indices = results['selected_indices']
        
        print(f"Loaded results: {len(selected_indices)} time points")
        print(f"Time range: {selected_times[0]:.2f}s to {selected_times[-1]:.2f}s")
        
        # Load ground truth if available
        residual_cov_truth = None
        truth_path = os.path.join(save_dir, 'residual_cov_truth.npy')
        if os.path.exists(truth_path):
            residual_cov_truth = np.load(truth_path)
            print("Ground truth loaded")
        
        # Create plots
        plot_residual_covariance_comparison(residual_cov_pred, residual_cov_truth, selected_times, save_dir)
        
        # Plot log10 error if ground truth is available
        if residual_cov_truth is not None:
            plot_log10_error(residual_cov_pred, residual_cov_truth, selected_times, save_dir)
        
        # Calculate time step info
        dt = selected_times[1] - selected_times[0] if len(selected_times) > 1 else 0.01
        total_time_steps = int(selected_times[-1] / dt) + 1
        
        plot_time_selection_info(total_time_steps, selected_indices, selected_times, dt, save_dir)
        
    else:
        print(f"Results file not found: {results_path}")

def plot_log10_error(residual_cov_pred, residual_cov_truth, selected_times, save_dir=None):
    """
    Plot log10 of absolute error between predicted and ground truth residual covariance for each dimension.
    
    Args:
        residual_cov_pred (np.ndarray): Predicted residual covariance
        residual_cov_truth (np.ndarray): Ground truth residual covariance
        selected_times (np.ndarray): Time points used for prediction
        save_dir (str): Directory to save the plot
    """
    if residual_cov_truth is None:
        print("Warning: Ground truth not available, cannot compute error")
        return
    
    # Calculate absolute error
    error = np.abs(residual_cov_pred - residual_cov_truth)
    
    # Calculate log10 of error (handle zeros by adding small epsilon)
    epsilon = 1e-12  # Small value to avoid log(0)
    log10_error = np.log10(error + epsilon)
    
    # Create subplots for each dimension
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dimensions = ['x1', 'x2', 'x3']
    colors = ['blue', 'red', 'green']
    
    for dim in range(3):
        ax = axes[dim]
        
        # Plot log10 error for this dimension
        ax.plot(selected_times, log10_error[:, dim], 
               'o-', color=colors[dim], alpha=0.8, linewidth=2, markersize=4)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'log10(|Error|) - {dimensions[dim]}')
        ax.set_title(f'Dimension {dim+1} ({dimensions[dim]}) - Log10 Error')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(log10_error[:, dim])
        std_error = np.std(log10_error[:, dim])
        min_error = np.min(log10_error[:, dim])
        max_error = np.max(log10_error[:, dim])
        
        stats_text = f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}\nMin: {min_error:.3f}\nMax: {max_error:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'log10_error_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        print(f"Log10 error plot saved to {os.path.join(save_dir, 'log10_error_comparison.pdf')}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Log10 Error Summary ===")
    for dim in range(3):
        mean_error = np.mean(log10_error[:, dim])
        std_error = np.std(log10_error[:, dim])
        print(f"Dimension {dim+1} ({dimensions[dim]}): Mean={mean_error:.3f}, Std={std_error:.3f}")

def plot_multiple_residual_covariance(data_list, selected_times_list, save_dir=None, labels=None):
    """
    Plot multiple residual covariance predictions and ground truth values in a 3×1 subplot layout.
    
    Args:
        data_list (list): List of dictionaries, each containing:
            - 'residual_cov_pred': Predicted residual covariance
            - 'residual_cov_truth': Ground truth residual covariance (optional)
            - 'selected_times': Time points for this dataset
        selected_times_list (list): List of time arrays for each dataset
        save_dir (str): Directory to save the plot
        labels (list): Labels for each dataset (optional)
    """
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(data_list))]
    
    # Create 3×1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    dimensions = ['x1', 'x2', 'x3']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for dim in range(3):
        ax = axes[dim]
        
        for i, (data, selected_times, label) in enumerate(zip(data_list, selected_times_list, labels)):
            color = colors[i % len(colors)]
            
            # Plot predictions
            if 'residual_cov_pred' in data and data['residual_cov_pred'] is not None:
                ax.plot(selected_times, data['residual_cov_pred'][:, dim], 
                       'o-', color=color, alpha=0.8, linewidth=2, markersize=4, 
                       label=f'{label} (Pred)')
            
            # Plot ground truth (if available)
            if 'residual_cov_truth' in data and data['residual_cov_truth'] is not None:
                truth_data = data['residual_cov_truth']
                N_truth = truth_data.shape[0]
                truth_times = np.linspace(selected_times[0], selected_times[-1], N_truth)
                indices = np.searchsorted(np.round(truth_times, 8), np.round(selected_times, 8))
                truth_selected = truth_data[indices, dim]
                # Use dashed line for ground truth
                ax.plot(selected_times, truth_selected, 
                       's--', color=color, alpha=0.6, linewidth=2, markersize=4, 
                       label=f'{label} (Truth)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Residual Covariance - {dimensions[dim]}')
        ax.set_title(f'Dimension {dim+1} ({dimensions[dim]}) - Multiple Datasets')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multiple_residual_covariance.pdf'), 
                   dpi=300, bbox_inches='tight')
        print(f"Multiple residual covariance plot saved to {os.path.join(save_dir, 'multiple_residual_covariance.pdf')}")
    
    plt.show()

def plot_multiple_log10_error(data_list, selected_times_list, save_dir=None, labels=None):
    """
    Plot log10 of absolute error for multiple datasets in a 3×1 subplot layout.
    
    Args:
        data_list (list): List of dictionaries, each containing:
            - 'residual_cov_pred': Predicted residual covariance
            - 'residual_cov_truth': Ground truth residual covariance
            - 'selected_times': Time points for this dataset
        selected_times_list (list): List of time arrays for each dataset
        save_dir (str): Directory to save the plot
        labels (list): Labels for each dataset (optional)
    """
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(data_list))]
    
    # Create 3×1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    dimensions = ['x1', 'x2', 'x3']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for dim in range(3):
        ax = axes[dim]
        
        for i, (data, selected_times, label) in enumerate(zip(data_list, selected_times_list, labels)):
            # Check if both pred and truth are available
            if ('residual_cov_pred' in data and data['residual_cov_pred'] is not None and
                'residual_cov_truth' in data and data['residual_cov_truth'] is not None):
                
                truth_data = data['residual_cov_truth']
                N_truth = truth_data.shape[0]
                truth_times = np.linspace(selected_times[0], selected_times[-1], N_truth)
                indices = np.searchsorted(np.round(truth_times, 8), np.round(selected_times, 8))
                truth_selected = truth_data[indices, dim]
                
                # Calculate absolute error
                error = np.abs(data['residual_cov_pred'][:, dim] - truth_selected)
                
                # Calculate log10 of error (handle zeros by adding small epsilon)
                epsilon = 1e-12
                log10_error = np.log10(error + epsilon)
                color = colors[i % len(colors)]
                ax.plot(selected_times, log10_error, 
                       'o-', color=color, alpha=0.8, linewidth=2, markersize=4, 
                       label=f'{label}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'log10(|Error|) - {dimensions[dim]}')
        ax.set_title(f'Dimension {dim+1} ({dimensions[dim]}) - Log10 Error')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'multiple_log10_error.pdf'), 
                   dpi=300, bbox_inches='tight')
        print(f"Multiple log10 error plot saved to {os.path.join(save_dir, 'multiple_log10_error.pdf')}")
    
    plt.show()


def plot_NOISE_LEVEL_EFFECT(coeff: dict, noise_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], save_dir: str = ""):
    """
    Plot the effect of noise level on coefficients across dimensions.
    
    Args:
        coeff (dict): Dictionary containing coefficients for each dimension
        noise_levels (list): List of noise levels corresponding to the data
        save_dir (str): Directory to save the plot
    """
    # Define the terms to plot for each dimension
    terms_config = {
        'dim_1': ['x1', 'x2', 'x3', 'x2x3'],
        'dim_2': ['x1', 'x2', 'x3', 'x1x3'], 
        'dim_3': ['x1', 'x2', 'x3', 'x1x2']
    }
    
    # Ground truth coefficients for each dimension
    ground_truth_coeffs = {
        'dim_1': {'x1': -0.2, 'x2': 1.0, 'x3': 2.0, 'x2x3': 1.0},
        'dim_2': {'x1': 1.0, 'x2': 0.1, 'x3': 3.0, 'x1x3': -0.6},
        'dim_3': {'x1': 2.0, 'x2': 3.0, 'x3': 0.1, 'x1x2': -0.4},
    }
    # Define row labels for the 4 rows
    row_labels = ['x1', 'x2', 'x3', 'Cross-term']
    
    # Create figure with 3 columns (dimensions) and 4 rows (terms), make it tall
    fig, axes = plt.subplots(4, 3, figsize=(20, 10), constrained_layout=True)
    fig.suptitle("Log10(Error) vs Noise Level for Each Dimension", fontsize=20)
    set_figure_position(x=100, y=100, width=1500, height=1600)
    
    # Color palette for different terms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each dimension in columns
    for col, (dim_key, terms) in enumerate(terms_config.items()):
        dim_data = coeff[dim_key]
        # Plot each term in rows
        for row, term in enumerate(terms):
            ax = axes[row, col]
            # Get the cross-term name for this dimension
            cross_term_name = {
                'dim_1': 'x2x3',
                'dim_2': 'x1x3', 
                'dim_3': 'x1x2'
            }[dim_key]
            # Check if we have data for this term
            has_data = term in dim_data and dim_data[term] and len(dim_data[term]) > 0
            if has_data:
                coeff_values = dim_data[term]
                plot_noise_levels = noise_levels[:len(coeff_values)]
                
                # Calculate errors and convert to log10
                ground_truth = ground_truth_coeffs[dim_key][term]
                errors = [abs(learned - ground_truth) for learned in coeff_values]
                log_errors = [np.log10(max(error, 1e-16)) for error in errors]  # Avoid log(0)
                
                ax.plot(plot_noise_levels, log_errors, 'o-', 
                        color=colors[row], linewidth=2, markersize=6, 
                        label=f'{term}')
                #ax.grid(True, alpha=0.3)
                ax.set_xlabel('Noise Level')
                ax.set_ylabel('Log10(Error)')
                # Use only the term as the subplot title
                if row < 3:
                    ax.set_title(f'{term}')
                else:
                    ax.set_title(f'{cross_term_name}')
                for i, (x, y) in enumerate(zip(plot_noise_levels, log_errors)):
                    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {term}', 
                        ha='center', va='center', transform=ax.transAxes)
                if row < 3:
                    ax.set_title(f'{term}')
                else:
                    ax.set_title(f'{cross_term_name}')
    # Remove tight_layout (constrained_layout handles it)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'noise_level_effect.pdf'), 
                    dpi=300, bbox_inches='tight')
        print(f"Noise level effect plot saved to {os.path.join(save_dir, 'noise_level_effect.pdf')}")
    plt.show()
    
    

def plot_cross_term_vs_noise(coeff,
                             noise_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                             save_dir: str = "",
                             filename: str = "cross_terms_vs_noise.pdf",
                             panel_labels: list = None,
                             ground_truths_list: list = None):
    """
    Plot log10-error of cross-term coefficients vs noise in a 1x3 layout.
    Left: equipart, center: cascade, right: dual_cascade (when coeff is a list of 3).
    If coeff is a single dict, the same data is drawn in all three panels (backward compat).

    Args:
        coeff: Single dict or list of 3 dicts [equipart, cascade, dual_cascade].
        noise_levels: Noise levels for x-axis.
        save_dir: Directory to save the plot.
        filename: Output PDF filename.
        panel_labels: Optional list of 3 strings for subplot titles.
        ground_truths_list: Optional list of 3 tuples (B1, B2, B3) per panel.
    """
    if isinstance(coeff, dict):
        coeff_list = [coeff, coeff, coeff]
        ground_truths_list = ground_truths_list or [(1.0, -0.6, -0.4)] * 3
    else:
        coeff_list = list(coeff)
        if len(coeff_list) != 3:
            raise ValueError("coeff must be a dict or a list of 3 coefficient dicts")
        ground_truths_list = ground_truths_list or [
            (1.0, -0.6, -0.4),   # equipart
            (2.0, -1.0, -1.0),  # cascade
            (2.0, -1.0, -1.0),  # dual_cascade
        ]
    if panel_labels is None:
        panel_labels = ['equipart', 'cascade', 'dual cascade']

    term_names = {'dim_1': 'x2x3', 'dim_2': 'x1x3', 'dim_3': 'x1x2'}
    styles = {
        'dim_1': {'color': '#1f77b4', 'marker': 'o'},
        'dim_2': {'color': '#ff7f0e', 'marker': 's'},
        'dim_3': {'color': '#2ca02c', 'marker': 'D'},
    }
    legend_labels = [
        r'$\log_{10}(|\hat{B}_1-B_1|)$',
        r'$\log_{10}(|\hat{B}_2-B_2|)$',
        r'$\log_{10}(|\hat{B}_3-B_3|)$',
        r'$\log_{10}(|\hat{B}_1+\hat{B}_2+\hat{B}_3-0|)$',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    set_figure_position(x=100, y=100, width=1400, height=600)
    legend_handles = []

    for idx, ax in enumerate(axes):
        c = coeff_list[idx]
        gt_B1, gt_B2, gt_B3 = ground_truths_list[idx]
        gts = {'dim_1': gt_B1, 'dim_2': gt_B2, 'dim_3': gt_B3}

        for dim_key in ['dim_1', 'dim_2', 'dim_3']:
            term_name = term_names[dim_key]
            if dim_key not in c or term_name not in c[dim_key] or not c[dim_key][term_name]:
                continue
            values = c[dim_key][term_name]
            x = noise_levels[:len(values)]
            err = np.abs(np.array(values) - gts[dim_key])
            err = np.maximum(err, 1e-16)
            y = np.log10(err)
            style = styles[dim_key]
            h = ax.plot(x, y, linestyle='-', marker=style['marker'], color=style['color'],
                        linewidth=2, markersize=5)
            if idx == 0:
                legend_handles.append(h[0])

        x2x3 = c.get('dim_1', {}).get('x2x3', [])
        x1x3 = c.get('dim_2', {}).get('x1x3', [])
        x1x2 = c.get('dim_3', {}).get('x1x2', [])
        n_pts = min(len(x2x3), len(x1x3), len(x1x2))
        if n_pts > 0:
            summed = np.array(x2x3[:n_pts]) + np.array(x1x3[:n_pts]) + np.array(x1x2[:n_pts])
            err_sum = np.maximum(np.abs(summed - 0.0), 1e-16)
            y_sum = np.log10(err_sum)
            h_sum = ax.plot(noise_levels[:n_pts], y_sum, linestyle='-', marker='^', color='#d62728',
                            linewidth=2, markersize=5)
            if idx == 0:
                legend_handles.append(h_sum[0])

        if idx == 0:
            ax.set_ylabel(r'semi-log scale error', fontsize=18)
        ax.set_xlabel('Noise Level', fontsize=18)
        ax.set_title(panel_labels[idx], fontsize=18)
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(axis='both', labelsize=18)

    # Overall layout first (leave a bit more headroom)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    # Shared legend as a single row just above the axes area, with a bigger box
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(legend_handles),
            frameon=True,
            fancybox=True,
            borderaxespad=0.6,
            title=None,
            fontsize=18,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-term vs noise plot saved to {save_path}")

    plt.show()


def plot_cross_term_vs_sample(coeff,
                              sample_sizes: list = None,
                              save_dir: str = "",
                              filename: str = "cross_terms_vs_sample.pdf",
                              panel_labels: list = None,
                              ground_truths_list: list = None):
    """
    Plot log10-error of cross-term coefficients vs sample size in a 1x3 layout.
    Same style as plot_cross_term_vs_noise but x-axis is sample size (from FINAK_EXPR_SAMPLE.txt).
    Left: equipart, center: cascade, right: dual_cascade (when coeff is a list of 3).

    Args:
        coeff: Single dict or list of 3 dicts [equipart, cascade, dual_cascade].
        sample_sizes: List of sample sizes for x-axis (e.g. [1000, 2000, ..., 10000]).
        save_dir: Directory to save the plot.
        filename: Output PDF filename.
        panel_labels: Optional list of 3 strings for subplot titles.
        ground_truths_list: Optional list of 3 tuples (B1, B2, B3) per panel.
    """
    if isinstance(coeff, dict):
        coeff_list = [coeff, coeff, coeff]
        ground_truths_list = ground_truths_list or [(1.0, -0.6, -0.4)] * 3
    else:
        coeff_list = list(coeff)
        if len(coeff_list) != 3:
            raise ValueError("coeff must be a dict or a list of 3 coefficient dicts")
        ground_truths_list = ground_truths_list or [
            (1.0, -0.6, -0.4),   # equipart
            (2.0, -1.0, -1.0),   # cascade
            (2.0, -1.0, -1.0),   # dual_cascade
        ]
    if panel_labels is None:
        panel_labels = ['Equipart', 'Forward Cascade', 'Dual Cascade']
    if sample_sizes is None:
        sample_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    term_names = {'dim_1': 'x2x3', 'dim_2': 'x1x3', 'dim_3': 'x1x2'}
    styles = {
        'dim_1': {'color': '#1f77b4', 'marker': 'o'},
        'dim_2': {'color': '#ff7f0e', 'marker': 's'},
        'dim_3': {'color': '#2ca02c', 'marker': 'D'},
    }
    legend_labels = [
        r'$\log_{10}(|\hat{B}_1-B_1|)$',
        r'$\log_{10}(|\hat{B}_2-B_2|)$',
        r'$\log_{10}(|\hat{B}_3-B_3|)$',
        r'$\log_{10}(|\hat{B}_1+\hat{B}_2+\hat{B}_3-0|)$',
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    set_figure_position(x=100, y=100, width=1400, height=600)
    legend_handles = []

    for idx, ax in enumerate(axes):
        c = coeff_list[idx]
        gt_B1, gt_B2, gt_B3 = ground_truths_list[idx]
        gts = {'dim_1': gt_B1, 'dim_2': gt_B2, 'dim_3': gt_B3}

        for dim_key in ['dim_1', 'dim_2', 'dim_3']:
            term_name = term_names[dim_key]
            if dim_key not in c or term_name not in c[dim_key] or not c[dim_key][term_name]:
                continue
            values = c[dim_key][term_name]
            n_vals = len(values)
            x = sample_sizes[:n_vals]
            err = np.abs(np.array(values) - gts[dim_key])
            err = np.maximum(err, 1e-16)
            y = np.log10(err)
            style = styles[dim_key]
            h = ax.plot(x, y, linestyle='-', marker=style['marker'], color=style['color'],
                        linewidth=2, markersize=5)
            if idx == 0:
                legend_handles.append(h[0])

        x2x3 = c.get('dim_1', {}).get('x2x3', [])
        x1x3 = c.get('dim_2', {}).get('x1x3', [])
        x1x2 = c.get('dim_3', {}).get('x1x2', [])
        n_pts = min(len(x2x3), len(x1x3), len(x1x2))
        if n_pts > 0:
            summed = np.array(x2x3[:n_pts]) + np.array(x1x3[:n_pts]) + np.array(x1x2[:n_pts])
            err_sum = np.maximum(np.abs(summed - 0.0), 1e-16)
            y_sum = np.log10(err_sum)
            h_sum = ax.plot(sample_sizes[:n_pts], y_sum, linestyle='-', marker='^', color='#d62728',
                           linewidth=2, markersize=5)
            if idx == 0:
                legend_handles.append(h_sum[0])

        if idx == 0:
            ax.set_ylabel(r'semi-log scale error', fontsize=18)
        ax.set_xlabel('Sample size', fontsize=18)
        ax.set_title(panel_labels[idx], fontsize=18)
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(axis='both', labelsize=18)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(legend_handles),
            frameon=True,
            fancybox=True,
            borderaxespad=0.6,
            fontsize=18,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-term vs sample plot saved to {save_path}")
    plt.show()


def to_latex(expr):
    expr = str(expr)
    expr = expr.replace('**', '^')
    expr = expr.replace('*', r'\cdot ')
    return expr

def plot_training_progress_grid(loss_history, coeff_history, final_expr, noise_level,
                                 save_dir=None):
    """
    Plots a 3x5 grid: 3 rows (dimensions), 5 columns (loss, x1, x2, x3, cross-term).
    Each subplot: y=quantity, x=epoch. Title includes noise_level.
    Final expression is printed as LaTeX on the leftmost plot of each row.
    Uses a color palette for each term.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    dims = [1, 2, 3]
    terms = {
        1: ['loss', 'x1', 'x2', 'x3', 'x2x3'],
        2: ['loss', 'x1', 'x2', 'x3', 'x1x3'],
        3: ['loss', 'x1', 'x2', 'x3', 'x1x2'],
    }
    ground_truth_coeffs = {
        1: {'x1': -0.2, 'x2': 1.0, 'x3': 2.0, 'x2x3': 1.0},
        2: {'x1': 1.0, 'x2': 0.1, 'x3': 3.0, 'x1x3': -0.6},
        3: {'x1': 2.0, 'x2': 3.0, 'x3': 0.1, 'x1x2': -0.4},
    }
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']  # You can change these to any matplotlib color names
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), constrained_layout=True)
    print_interval = 50
    epochs = np.arange(len(next(iter(loss_history.values())))) * print_interval
    for i, dim in enumerate(dims):
        for j, term in enumerate(terms[dim]):
            ax = axes[i, j]
            color = colors[j % len(colors)]
            if term == 'loss':
                ax.plot(epochs, loss_history[dim], label='Loss', color=color)
                ax.set_ylabel(f'Dim {dim}')
                ax.set_title(f'Loss (Noise={noise_level})')
            else:
                # Calculate log10 of absolute error from ground truth
                epsilon = 1e-12  # Small value to avoid log(0)
                coeff_values = np.array(coeff_history[dim][term])
                ground_truth_value = ground_truth_coeffs[dim][term]
                error = np.abs(coeff_values - ground_truth_value)
                log10_error = np.log10(error + epsilon)
                ax.plot(epochs, log10_error, label=f'log10(|{term}-gt|)', color=color)
                ax.set_title(f'log10(|{term}-gt|) (Noise={noise_level})')
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        # Print final expression as LaTeX on the leftmost plot of each row
        expr = final_expr[dim]
        if not isinstance(expr, str):
            expr = str(expr)
        axes[i, 0].text(0.05, 0.95, f'${to_latex(expr)}$', transform=axes[i, 0].transAxes,
                        fontsize=10, va='top', ha='left', color='purple', bbox=dict(facecolor='white', alpha=0.7))
    plt.suptitle(f'Training Progress (Noise Level: {noise_level})', fontsize=18)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'training_progress_grid_noise_{noise_level}.pdf'), 
                    dpi=300, bbox_inches='tight')
        print(f"Training progress grid plot saved to {os.path.join(save_dir, f'training_progress_grid_noise_{noise_level}.pdf')}")
    plt.show()
    
    

def plot_trajectory_comparison(mean_state_record, mean_state_pred, Time_record, save_path, 
                              method_name="FEX-framework", component_names=None):
    """
    Create comprehensive trajectory comparison plots.
    
    Args:
        mean_state_record (np.ndarray): Ground truth mean states (3, time_steps)
        mean_state_pred (np.ndarray): Predicted mean states (3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        method_name (str): Name of the prediction method (e.g., "Single Model", "Ensemble")
        component_names (list): Names for the components (default: ['u1', 'u2', 'u3'])
    """
    if component_names is None:
        component_names = ['u1', 'u2', 'u3']
    
    # Color & Style Setup
    colors = {'Ground-Truth': 'black', 'Prediction': 'orange'}
    linestyles = {'Ground-Truth': ':', 'Prediction': '-'}
    markers = {'Prediction': 'o'}
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    for i in range(3):
        # True values
        mask_true = mean_state_record[i] != 0
        axs[i].plot(Time_record[mask_true], mean_state_record[i][mask_true], 
                    linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], 
                    linewidth=3, label=fr'Ground Truth $\langle u_{i+1} \rangle$')
        
        # Predicted values
        mask_pred = mean_state_pred[i] != 0
        axs[i].plot(Time_record[mask_pred], mean_state_pred[i][mask_pred], 
                   linestyle=linestyles['Prediction'], color=colors['Prediction'], 
                   linewidth=3, marker=markers['Prediction'], markersize=5, alpha=0.7,
                   label=fr'{method_name} $\langle u_{i+1} \rangle$')
        
        axs[i].set_ylabel(fr'Mean $u_{i+1}$', fontsize=15)
        axs[i].set_title(f'Component {i+1}', fontsize=18)
        axs[i].legend(loc='upper right', frameon=False, fontsize=12)
        axs[i].tick_params(axis='both', labelsize=12)
    
    axs[2].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle(f'Mean Values of Components Over Time - {method_name}', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.9)
    
    # Save and show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_covariance_comparison(cov_state_record, cov_state_pred, Time_record, save_path=None, 
                              title_suffix="FEX-framework"):
    """
    Plot covariance comparison between ground truth and prediction.
    
    Args:
        cov_state_record (np.ndarray): Ground truth covariance (3, 3, time_steps)
        cov_state_pred (np.ndarray): Predicted covariance (3, 3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    colors = {'Ground-Truth': 'black', 'Prediction': 'orange'}
    linestyles = {'Ground-Truth': ':', 'Prediction': '-'}
    markers = {'Prediction': 'o'}
    
    for i in range(3):
        for j in range(3):
            # Ground truth
            #mask_true = cov_state_record[i, j] != 0
            #axs[i, j].plot(Time_record[mask_true], cov_state_record[i, j][mask_true], 
            axs[i, j].plot(Time_record, cov_state_record[i, j],
                          linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], 
                          linewidth=2, label='Ground Truth')
            
            # # Prediction
            #mask_pred = cov_state_pred[i, j] != 0
            #axs[i, j].plot(Time_record[mask_pred], cov_state_pred[i, j][mask_pred],
            axs[i, j].plot(Time_record, cov_state_pred[i, j],
                         linestyle=linestyles['Prediction'], color=colors['Prediction'], 
                         linewidth=2, marker=markers['Prediction'], markersize=4, alpha=0.7,
                         label=f'Prediction {title_suffix}')
            
            axs[i, j].set_title(f'Cov(u{i+1}, u{j+1})', fontsize=12)
            axs[i, j].set_xlabel('Time', fontsize=10)
            axs[i, j].set_ylabel('Covariance', fontsize=10)
            axs[i, j].legend(frameon=False, fontsize=8)
            axs[i, j].tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Covariance Comparison {title_suffix}', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    if save_path:
        plt.savefig(os.path.join(save_path, 'covariance_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_mean_covariance_grid_ind_dep(Time_ind,
                                       mean_gt_ind,
                                       cov_gt_ind,
                                       mean_pred_ind,
                                       cov_pred_ind,
                                       Time_dep,
                                       mean_pred_dep,
                                       cov_pred_dep,
                                       t_ind_max: float = 20.0,
                                       t_dep_max: float = 10.0,
                                       save_path: str = None,
                                       legend_title: str = None,
                                       independent_label: str = "ASD-FEX-TFDM-independent",
                                       dependent_label: str = "ASD-FEX-TFDM-dependent",
                                       ground_truth_label: str = "Ground Truth",
                                       show_legend: bool = True,
                                       font_size: int = 18):
    """
    Plot a 3x4 grid:
      rows: u1, u2, u3
      col0: mean(u_i)
      col1..3: cov(u_i, u_j) with j = 1..3

    Shows:
      - ground truth (black) on independent time grid (Time_ind)
      - independent prediction (orange dashed) on Time_ind
      - dependent prediction (blue dash-dot) on Time_dep, plotted only up to t_dep_max
    """
    title_fs = label_fs = legend_fs = font_size

    mask_ind = Time_ind <= (t_ind_max + 1e-9)
    mask_dep = Time_dep <= (t_dep_max + 1e-9)
    Time_ind_plot = Time_ind[mask_ind]
    Time_dep_plot = Time_dep[mask_dep]

    fig, axs = plt.subplots(3, 4, figsize=(22, 12), sharex=True)

    row_names = ["u1", "u2", "u3"]
    col_titles = ["Mean", "Cov(·, u1)", "Cov(·, u2)", "Cov(·, u3)"]
    for c in range(4):
        axs[0, c].set_title(col_titles[c], fontsize=title_fs)

    colors = {"gt": "black", "ind": "#ff7f0e", "dep": "#1f77b4"}
    # Use a solid line for the independent prediction (dashed rendering can look
    # like "gaps" even when all points exist). Add markers every few points.
    linestyles = {"ind": "-", "dep": "-."}
    ind_marker = "o"
    # Roughly ~10-15 markers across the time axis (avoid heavy clutter).
    ind_markevery = max(1, len(Time_ind_plot) // 12)

    handles = []
    labels = []

    for i in range(3):
        # Mean
        ax = axs[i, 0]
        h_gt = ax.plot(Time_ind_plot, mean_gt_ind[i, mask_ind], color=colors["gt"], lw=2.0, label=ground_truth_label)[0]
        h_ind = ax.plot(
            Time_ind_plot,
            mean_pred_ind[i, mask_ind],
            color=colors["ind"],
            lw=2.0,
            ls=linestyles["ind"],
            marker=ind_marker,
            markersize=3.5,
            markevery=ind_markevery,
            label=independent_label,
        )[0]
        h_dep = ax.plot(
            Time_dep_plot,
            mean_pred_dep[i, mask_dep],
            color=colors["dep"],
            lw=2.0,
            ls=linestyles["dep"],
            label=dependent_label,
        )[0]
        ax.set_ylabel(row_names[i], fontsize=label_fs)
        ax.grid(True, alpha=0.25)
        if i == 0:
            handles = [h_gt, h_ind, h_dep]
            labels = [ground_truth_label, independent_label, dependent_label]

        # Covariance matrix elements
        for j in range(3):
            k = j  # cov(u_i, u_{k+1})
            ax = axs[i, j + 1]
            ax.plot(Time_ind_plot, cov_gt_ind[i, k, mask_ind], color=colors["gt"], lw=1.8, label=None)
            ax.plot(
                Time_ind_plot,
                cov_pred_ind[i, k, mask_ind],
                color=colors["ind"],
                lw=1.8,
                ls=linestyles["ind"],
                marker=ind_marker,
                markersize=2.8,
                markevery=ind_markevery,
                label=None,
            )
            ax.plot(Time_dep_plot, cov_pred_dep[i, k, mask_dep], color=colors["dep"], lw=1.8, ls=linestyles["dep"], label=None)
            ax.grid(True, alpha=0.25)

    for ax in axs.flatten():
        ax.tick_params(axis="both", labelsize=font_size)

    axs[2, 0].set_xlabel("Time (s)", fontsize=label_fs)
    for c in range(1, 4):
        axs[2, c].set_xlabel("Time (s)", fontsize=label_fs)

    # Leave a bit of headroom for a shared legend at the top (optional).
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if show_legend and handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(handles),
            frameon=False,
            fancybox=True,
            title=legend_title,
            title_fontsize=legend_fs,
            fontsize=legend_fs,
            handlelength=2.2,
            columnspacing=1.4,
        )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    plt.show()
    return fig


def plot_log10_error_mean_covariance_grid_ind_dep(
    Time_ind,
    mean_gt_ind,
    cov_gt_ind,
    mean_pred_ind,
    cov_pred_ind,
    Time_dep,
    mean_pred_dep,
    cov_pred_dep,
    t_ind_max: float = 20.0,
    t_dep_max: float = 10.0,
    save_path: str = None,
    independent_label: str = "ASD-FEX-TFDM-independent (error)",
    dependent_label: str = "ASD-FEX-TFDM-dependent (error)",
    legend_title: str = None,
    eps: float = 1e-16,
):
    """
    Plot a 3x4 grid of log10 absolute errors:
      rows: u1, u2, u3
      col0: |mean_pred - mean_gt|
      col1..3: |cov_pred - cov_gt|

    Only two curves are shown: independent (orange) and dependent (blue).
    """
    # Discussion figures: force a single fixed fontsize everywhere.
    font_size = 18
    title_fs = 18
    label_fs = 18
    legend_fs = 18

    mask_ind = Time_ind <= (t_ind_max + 1e-9)
    mask_dep = Time_dep <= (t_dep_max + 1e-9)
    # Drop the first point (t=0) for nicer visualization (often large transients)
    Time_ind_masked = Time_ind[mask_ind]
    Time_dep_masked = Time_dep[mask_dep]
    Time_ind_plot = Time_ind_masked[1:]
    Time_dep_plot = Time_dep_masked[1:]
    n_dep_plot = len(Time_dep_plot)  # equals (Nt_dep + 1) - 1

    fig, axs = plt.subplots(3, 4, figsize=(22, 12), sharex=True)

    row_names = ["u1", "u2", "u3"]
    col_titles = ["log10|mean error|", "log10|cov(·, u1) error|", "log10|cov(·, u2) error|", "log10|cov(·, u3) error|"]
    for c in range(4):
        axs[0, c].set_title(col_titles[c], fontsize=title_fs)

    colors = {"ind": "#ff7f0e", "dep": "#1f77b4"}
    linestyles = {"ind": "-", "dep": "-."}
    ind_marker = "o"
    ind_markevery = max(1, len(Time_ind_plot) // 12)

    handles = []
    labels = []

    # Dependent ground truth slice should match `mean_pred_dep[..., mask_dep]` BEFORE dropping t=0.
    dep_len = np.sum(mask_dep)
    mean_gt_dep = mean_gt_ind[:, :dep_len]  # includes t=0
    cov_gt_dep = cov_gt_ind[:, :, :dep_len]  # includes t=0

    for i in range(3):
        ax = axs[i, 0]

        err_ind = np.abs(mean_pred_ind[i, mask_ind][1:] - mean_gt_ind[i, mask_ind][1:])
        y_ind = np.log10(np.maximum(err_ind, eps))
        h_ind = ax.plot(
            Time_ind_plot,
            y_ind,
            color=colors["ind"],
            lw=2.0,
            ls=linestyles["ind"],
            marker=ind_marker,
            markersize=3.5,
            markevery=ind_markevery,
            label=independent_label,
        )[0]

        err_dep = np.abs(mean_pred_dep[i, mask_dep][1:] - mean_gt_dep[i, 1:])
        y_dep = np.log10(np.maximum(err_dep, eps))
        h_dep = ax.plot(
            Time_dep_plot,
            y_dep,
            color=colors["dep"],
            lw=2.0,
            ls=linestyles["dep"],
            label=dependent_label,
        )[0]

        ax.set_ylabel(row_names[i], fontsize=label_fs)
        ax.grid(True, alpha=0.25)

        if i == 0:
            handles = [h_ind, h_dep]
            labels = [independent_label, dependent_label]

        # Covariance elements errors
        for j in range(3):
            k = j
            ax = axs[i, j + 1]

            err_ind_cov = np.abs(
                cov_pred_ind[i, k, mask_ind][1:] - cov_gt_ind[i, k, mask_ind][1:]
            )
            y_ind_cov = np.log10(np.maximum(err_ind_cov, eps))
            ax.plot(
                Time_ind_plot,
                y_ind_cov,
                color=colors["ind"],
                lw=1.8,
                ls=linestyles["ind"],
                marker=ind_marker,
                markersize=2.8,
                markevery=ind_markevery,
                label=None,
            )

            err_dep_cov = np.abs(cov_pred_dep[i, k, mask_dep][1:] - cov_gt_dep[i, k, 1:])
            y_dep_cov = np.log10(np.maximum(err_dep_cov, eps))
            ax.plot(
                Time_dep_plot,
                y_dep_cov,
                color=colors["dep"],
                lw=1.8,
                ls=linestyles["dep"],
                label=None,
            )
            ax.grid(True, alpha=0.25)

    for ax in axs.flatten():
        ax.tick_params(axis="both", labelsize=font_size)

    axs[2, 0].set_xlabel("Time (s)", fontsize=label_fs)
    for c in range(1, 4):
        axs[2, c].set_xlabel("Time (s)", fontsize=label_fs)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(handles),
            frameon=False,
            fancybox=True,
            title=legend_title,
            title_fontsize=18,
            fontsize=legend_fs,
            handlelength=2.2,
            columnspacing=1.4,
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    plt.show()
    return fig
    
def plot_mean_comparison(mean_state_record, mean_state_pred, Time_record, save_path=None, 
                              title_suffix="FEX-framework"):
    """
    Plot mean comparison between ground truth and prediction.
    
    Args:
        mean_state_record (np.ndarray): Ground truth mean (3, time_steps)
        mean_state_pred (np.ndarray): Predicted mean (3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    
    # Plot each component separately
    component_names = ['u1', 'u2', 'u3']
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Color & Style Setup
    colors = {'Ground-Truth': 'black', 'Prediction': 'orange'}
    linestyles = {'Ground-Truth': ':', 'Prediction': '-'}
    markers = {'Prediction': 'o'}

    for i in range(3):
        # True values
        mask_true = mean_state_record[i] != 0
        axs[i].plot(Time_record[mask_true], mean_state_record[i][mask_true], 
                linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], linewidth=3,
                label=fr'Ground Truth $\langle u_{i+1} \rangle$')
    
        # Predicted values
        mask_pred = mean_state_pred[i] != 0
        axs[i].plot(Time_record[mask_pred], mean_state_pred[i][mask_pred], 
               linestyle=linestyles['Prediction'], color=colors['Prediction'], 
               linewidth=3, marker=markers['Prediction'], markersize=5, alpha=0.7,
               label=fr'Prediction {title_suffix} $\langle u_{i+1} \rangle$')
    
        axs[i].set_ylabel(fr'Mean $u_{i+1}$', fontsize=15)
        axs[i].set_title(f'Component {i+1}', fontsize=18)
        axs[i].legend(loc='upper right', frameon=False, fontsize=12)
        axs[i].tick_params(axis='both', labelsize=12)

    axs[2].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle(f'Mean Values of Components Over Time {title_suffix}', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.9)  # Adjust top to make room for the title
    if save_path:
        plt.savefig(os.path.join(save_path, 'mean_components_over_time.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig
    

def plot_mean_comparison_ensemble(mean_state_record, mean_state_pred_single, mean_state_pred_ensemble, Time_record, save_path=None, 
                              title_suffix="FEX-framework"):
    """
    Plot mean comparison between ground truth, single NN, and ensemble NN predictions together.
    
    Args:
        mean_state_record (np.ndarray): Ground truth mean (3, time_steps)
        mean_state_pred_single (np.ndarray): Single NN predicted mean (3, time_steps)
        mean_state_pred_ensemble (np.ndarray): Ensemble NN predicted mean (3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    
    # Plot each component separately
    component_names = ['u1', 'u2', 'u3']
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Color & Style Setup - using the requested color scheme
    colors = {'Ground-Truth': 'black', 'Single Neural Network': 'orange', 'Ensemble Neural Network': 'blue'}
    linestyles = {'Ground-Truth': ':', 'Single Neural Network': '-', 'Ensemble Neural Network': '-'}
    markers = {'Single Neural Network': 'o', 'Ensemble Neural Network': 's'}

    for i in range(3):
        # True values
        mask_true = mean_state_record[i] != 0
        axs[i].plot(Time_record[mask_true], mean_state_record[i][mask_true], 
                linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], linewidth=3,
                label=fr'Ground Truth $\langle u_{i+1} \rangle$')
    
        # Single NN predicted values
        mask_pred_single = mean_state_pred_single[i] != 0
        axs[i].plot(Time_record[mask_pred_single], mean_state_pred_single[i][mask_pred_single], 
               linestyle=linestyles['Single Neural Network'], color=colors['Single Neural Network'], 
               linewidth=2, marker=markers['Single Neural Network'], markersize=4, alpha=0.7,
               label=fr'Single NN $\langle u_{i+1} \rangle$')
        
        # Ensemble NN predicted values
        mask_pred_ensemble = mean_state_pred_ensemble[i] != 0
        axs[i].plot(Time_record[mask_pred_ensemble], mean_state_pred_ensemble[i][mask_pred_ensemble], 
               linestyle=linestyles['Ensemble Neural Network'], color=colors['Ensemble Neural Network'], 
               linewidth=2, marker=markers['Ensemble Neural Network'], markersize=4, alpha=0.7,
               label=fr'Ensemble NN $\langle u_{i+1} \rangle$')
    
        axs[i].set_ylabel(fr'Mean $u_{i+1}$', fontsize=15)
        axs[i].set_title(f'Component {i+1}', fontsize=18)
        axs[i].legend(loc='upper right', frameon=False, fontsize=12)
        axs[i].tick_params(axis='both', labelsize=12)

    axs[2].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle(f'Mean Values Comparison {title_suffix}', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.9)  # Adjust top to make room for the title
    if save_path:
        plt.savefig(os.path.join(save_path, 'mean_components_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_covariance_comparison_ensemble(cov_state_record, cov_state_pred_single, cov_state_pred_ensemble, Time_record, save_path=None, 
                              title_suffix="FEX-framework"):
    """
    Plot covariance comparison between ground truth, single NN, and ensemble NN predictions together.
    
    Args:
        cov_state_record (np.ndarray): Ground truth covariance (3, 3, time_steps)
        cov_state_pred_single (np.ndarray): Single NN predicted covariance (3, 3, time_steps)
        cov_state_pred_ensemble (np.ndarray): Ensemble NN predicted covariance (3, 3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    colors = {'Ground-Truth': 'black', 'Single Neural Network': 'orange', 'Ensemble Neural Network': 'blue'}
    linestyles = {'Ground-Truth': ':', 'Single Neural Network': '-', 'Ensemble Neural Network': '-'}
    markers = {'Single Neural Network': 'o', 'Ensemble Neural Network': 's'}
    
    for i in range(3):
        for j in range(3):
            # Ground truth
            mask_true = cov_state_record[i, j] != 0
            axs[i, j].plot(Time_record[mask_true], cov_state_record[i, j][mask_true], 
                          linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], 
                          linewidth=2, label='Ground Truth')
            
            # Single NN prediction
            mask_pred_single = cov_state_pred_single[i, j] != 0
            axs[i, j].plot(Time_record[mask_pred_single], cov_state_pred_single[i, j][mask_pred_single], 
                         linestyle=linestyles['Single Neural Network'], color=colors['Single Neural Network'], 
                         linewidth=2, marker=markers['Single Neural Network'], markersize=3, alpha=0.7,
                         label='Single NN')
            
            # Ensemble NN prediction
            mask_pred_ensemble = cov_state_pred_ensemble[i, j] != 0
            axs[i, j].plot(Time_record[mask_pred_ensemble], cov_state_pred_ensemble[i, j][mask_pred_ensemble], 
                         linestyle=linestyles['Ensemble Neural Network'], color=colors['Ensemble Neural Network'], 
                         linewidth=2, marker=markers['Ensemble Neural Network'], markersize=3, alpha=0.7,
                         label='Ensemble NN')
            
            axs[i, j].set_title(f'Cov(u{i+1}, u{j+1})', fontsize=12)
            axs[i, j].set_xlabel('Time', fontsize=10)
            axs[i, j].set_ylabel('Covariance', fontsize=10)
            axs[i, j].legend(frameon=False, fontsize=8)
            axs[i, j].tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Covariance Comparison {title_suffix}', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'covariance_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
    
    
def plot_energy_comparison(Energy_MC_all, Energy_MC_pred, Time_record, save_path=None, title_suffix="FEX-framework"):
    """
    Plot energy comparison between ground truth and prediction.
    
    Args:
        Energy_MC_all (np.ndarray): Ground truth energy (4, time_steps)
        Energy_MC_pred (np.ndarray): Predicted energy (4, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    energy_labels = ['Total', 'Mode 1', 'Mode 2', 'Mode 3']
    energy_keys = ['total', 'u1', 'u2', 'u3']
    colors = {'Ground-Truth': 'black', 'Prediction': 'orange'}

    for idx, (label, key) in enumerate(zip(energy_labels, energy_keys)):
        ax = axs[idx // 2, idx % 2]
        mask_truth = Energy_MC_all[idx] != 0
        mask_pred = Energy_MC_pred[idx] != 0
        ax.plot(Time_record[mask_truth], Energy_MC_all[idx][mask_truth], 
                color=colors['Ground-Truth'], label=f'{label} (Truth)', linewidth=2)
        ax.plot(Time_record[mask_pred], Energy_MC_pred[idx][mask_pred], 
                color=colors['Prediction'], label=f'{label} (Prediction)', 
                linestyle='--', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.suptitle(f'Energy Comparison {title_suffix}', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'energy_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_third_order_moments(moment3_state_record, moment3_state_pred, Time_record, save_path=None, title_suffix="FEX-framework"):
    """
    Plot third-order moments comparison between ground truth and prediction.
    
    Args:
        moment3_state_record (np.ndarray): Ground truth 3rd moments (3, 3, 3, time_steps)
        moment3_state_pred (np.ndarray): Predicted 3rd moments (3, 3, 3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    moment_indices = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    moment_labels = [r'$\langle M_{123} \rangle$', r'$\langle M_{122} \rangle$', 
                    r'$\langle M_{133} \rangle$', r'$\langle M_{223} \rangle$']

    # Color & Style Setup
    colors = {'Ground-Truth': 'black', 'Prediction': 'orange'}
    linestyles = {'Ground-Truth': ':', 'Prediction': '-'}
    markers = {'Prediction': 'o'}

    for idx, (i, j, k) in enumerate(moment_indices):
        # True 3rd-order moment values
        mask_true = moment3_state_record[i, j, k, :] != 0
        axs[idx].plot(Time_record[mask_true], moment3_state_record[i, j, k, mask_true], 
                     linestyle=linestyles['Ground-Truth'], color=colors['Ground-Truth'], linewidth=3,
                     label=f'Ground Truth {moment_labels[idx]}')
        
        # Predicted 3rd-order moment values
        mask_pred = moment3_state_pred[i, j, k, :] != 0
        axs[idx].plot(Time_record[mask_pred], moment3_state_pred[i, j, k, mask_pred], 
                     linestyle=linestyles['Prediction'], color=colors['Prediction'], 
                     linewidth=3, marker=markers['Prediction'], markersize=5, alpha=0.7,
                     label=f'{title_suffix} {moment_labels[idx]}')
        
        axs[idx].set_ylabel(f'3rd Moment', fontsize=15)
        axs[idx].set_title(f'{moment_labels[idx]}', fontsize=18)
        axs[idx].legend(loc='upper right', frameon=False, fontsize=12)
        axs[idx].tick_params(axis='both', labelsize=12)

    axs[3].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle(f'Third-Order Moments Over Time {title_suffix}', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'third_order_moments_over_time.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_third_order_moments_2x2(
    moment3_state_record,
    moment3_state_pred,
    Time_record,
    save_path=None,
    title_suffix="FEX-framework",
):
    """
    Plot selected 3rd-order moments in a 2x2 grid.
    Uses the same moment indices as `plot_third_order_moments`.
    """
    # Discussion figures: force a single fixed fontsize everywhere.
    font_size = 18
    label_fs = 18
    title_fs = 18
    legend_fs = 18

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.flatten()

    moment_indices = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    moment_labels = [
        r'$\langle M_{123} \rangle$',
        r'$\langle M_{122} \rangle$',
        r'$\langle M_{133} \rangle$',
        r'$\langle M_{223} \rangle$',
    ]

    colors = {"Ground-Truth": "black", "Prediction": "orange"}
    linestyles = {"Ground-Truth": ":", "Prediction": "-"}
    markers = {"Prediction": "o"}

    for idx_plot, (i, j, k) in enumerate(moment_indices):
        ax = axs[idx_plot]

        mask_true = moment3_state_record[i, j, k, :] != 0
        ax.plot(
            Time_record[mask_true],
            moment3_state_record[i, j, k, mask_true],
            linestyle=linestyles["Ground-Truth"],
            color=colors["Ground-Truth"],
            linewidth=2.5,
            label=f"Ground Truth {moment_labels[idx_plot]}",
        )

        mask_pred = moment3_state_pred[i, j, k, :] != 0
        ax.plot(
            Time_record[mask_pred],
            moment3_state_pred[i, j, k, mask_pred],
            linestyle=linestyles["Prediction"],
            color=colors["Prediction"],
            linewidth=2.5,
            marker=markers["Prediction"],
            markersize=4.5,
            alpha=0.8,
            label=f"{title_suffix} {moment_labels[idx_plot]}",
        )

        ax.set_title(moment_labels[idx_plot], fontsize=title_fs)
        ax.set_ylabel("3rd Moment", fontsize=label_fs)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(frameon=False, fontsize=font_size, loc="best")

    axs[2].set_xlabel("Time", fontsize=label_fs)
    axs[3].set_xlabel("Time", fontsize=label_fs)

    plt.tight_layout()
    plt.suptitle(
        f"Third-Order Moments Over Time (2x2) {title_suffix}",
        fontsize=font_size,
        y=1.02,
    )
    plt.subplots_adjust(top=0.88)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    plt.show()
    return fig


def plot_third_order_moments_ind_dep_2x2(
    moment3_state_record_ind,
    moment3_state_pred_ind,
    moment3_state_pred_dep,
    Time_ind,
    Time_dep,
    save_path=None,
    title_suffix="FEX-framework",
    legend_title=None,
    ground_truth_label="Ground Truth",
    independent_label="Independent",
    dependent_label="Dependent (t<=10)",
    show_legend: bool = True,
):
    """
    2x2 grid of selected third-order moments, showing:
      - ground truth (black) on Time_ind
      - independent prediction (orange) on Time_ind
      - dependent prediction (blue dash-dot) on Time_dep
    """
    # Discussion figures: force a single fixed fontsize everywhere.
    font_size = 18
    label_fs = 18
    title_fs = 18
    legend_fs = 18

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.flatten()
    for ax in axs:
        ax.tick_params(axis="both", labelsize=font_size)

    moment_indices = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    moment_labels = [
        r'$\langle M_{123} \rangle$',
        r'$\langle M_{122} \rangle$',
        r'$\langle M_{133} \rangle$',
        r'$\langle M_{223} \rangle$',
    ]

    colors = {"gt": "black", "ind": "#ff7f0e", "dep": "#1f77b4"}
    # Make ground-truth a solid straight line (matches user expectation).
    linestyles = {"gt": "-", "ind": "-", "dep": "-."}
    marker_ind = "o"
    markevery = max(1, len(Time_ind) // 12)

    handles = []
    labels = []

    for idx_plot, (i, j, k) in enumerate(moment_indices):
        ax = axs[idx_plot]

        # Ground truth
        mask_gt = moment3_state_record_ind[i, j, k, :] != 0
        h_gt = ax.plot(
            Time_ind[mask_gt],
            moment3_state_record_ind[i, j, k, mask_gt],
            linestyle=linestyles["gt"],
            color=colors["gt"],
            linewidth=2.5,
        )[0]

        # Independent prediction
        mask_ind = moment3_state_pred_ind[i, j, k, :] != 0
        h_ind = ax.plot(
            Time_ind[mask_ind],
            moment3_state_pred_ind[i, j, k, mask_ind],
            linestyle=linestyles["ind"],
            color=colors["ind"],
            linewidth=2.5,
            marker=marker_ind,
            markersize=3.5,
            markevery=markevery,
        )[0]

        # Dependent prediction (only first Nt_dep+1 samples)
        mask_dep = moment3_state_pred_dep[i, j, k, :] != 0
        h_dep = ax.plot(
            Time_dep[mask_dep],
            moment3_state_pred_dep[i, j, k, mask_dep],
            linestyle=linestyles["dep"],
            color=colors["dep"],
            linewidth=2.5,
        )[0]

        ax.set_title(moment_labels[idx_plot], fontsize=title_fs)
        ax.set_ylabel("3rd Moment", fontsize=label_fs)
        ax.grid(True, alpha=0.25)

        if idx_plot == 0:
            handles = [h_gt, h_ind, h_dep]
            labels = [ground_truth_label, independent_label, dependent_label]

    axs[2].set_xlabel("Time (s)", fontsize=label_fs)
    axs[3].set_xlabel("Time (s)", fontsize=label_fs)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if show_legend and handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(handles),
            frameon=False,
            fancybox=True,
            title=legend_title,
            fontsize=font_size,
            title_fontsize=font_size if legend_title else None,
            handlelength=2.2,
            columnspacing=1.4,
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {save_path}")
    plt.show()
    return fig


def plot_discussion_choice3_composite(
    save_path: str,
    u_all: np.ndarray,
    u_pred_all: np.ndarray,
    u_pred_dependent: np.ndarray | None,
    dt: float,
    Time_record: np.ndarray,
    Time_dep: np.ndarray,
    mean_state_record_independent: np.ndarray,
    cov_state_record_independent: np.ndarray,
    mean_state_pred_independent: np.ndarray,
    cov_state_pred_independent: np.ndarray,
    mean_state_pred_dependent: np.ndarray,
    cov_state_pred_dependent: np.ndarray,
    moment3_state_record: np.ndarray,
    moment3_state_pred: np.ndarray,
    moment3_state_pred_dependent: np.ndarray,
    TIME_AMOUNT: float = 20.0,
    TIME_DEP_AMOUNT: float = 10.0,
    params_name: str = "Equipart",
    font_size: int = 20,
):
    """
    Build one composite figure from the four panels (order: t0_20, t0_10_dep, mean_cov, third_order).
    Layout: left column = dep (1x3) top, t0_20 (2x3) bottom; right column = mean_cov top, third_order 2x2 bottom.
    Orange box + 'independent' on t0_20; blue box + 'dependent' on t0_10_dep.
    Legend centered in header; 'Equipart' (or params_name) centered in footer. All font 20.
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_comp
    import matplotlib.image as mpimg
    from matplotlib.gridspec import GridSpec

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        dep_png = os.path.join(tmpdir, "t0_10_dep.png")
        ind_png = os.path.join(tmpdir, "t0_20.png")
        mean_cov_png = os.path.join(tmpdir, "mean_cov.png")
        third_png = os.path.join(tmpdir, "third.png")

        # 1) Dependent 1x3 (t=0,4,8)
        plot_triad_3d_time_grid_matplotlib_cloud_3x3_times(
            u_gt=u_all,
            u_ind=u_pred_all,
            u_dep=u_pred_dependent,
            dt=dt,
            time_points=[0, 4, 8],
            save_path=dep_png,
            grid_rows=1,
            grid_cols=3,
            elev=20,
            azim=-60,
            sample_size=15000,
            point_size=6.0,
            opacity=0.55,
            cmap="turbo",
        )
        # 2) Independent 2x3 (t=0,4,8,12,16,20)
        plot_triad_3d_time_grid_matplotlib_cloud_3x3_times(
            u_gt=u_all,
            u_ind=u_pred_all,
            u_dep=u_pred_dependent,
            dt=dt,
            time_points=[0, 4, 8, 12, 16, 20],
            save_path=ind_png,
            grid_rows=2,
            grid_cols=3,
            elev=20,
            azim=-60,
            sample_size=15000,
            point_size=6.0,
            opacity=0.55,
            cmap="turbo",
        )
        plt.close("all")

        # 3) Mean/covariance 3x4 (no internal legend; composite has one in header)
        # Use larger font so labels stay readable when embedded in the composite.
        plot_mean_covariance_grid_ind_dep(
            Time_ind=Time_record,
            mean_gt_ind=mean_state_record_independent,
            cov_gt_ind=cov_state_record_independent,
            mean_pred_ind=mean_state_pred_independent,
            cov_pred_ind=cov_state_pred_independent,
            Time_dep=Time_dep,
            mean_pred_dep=mean_state_pred_dependent,
            cov_pred_dep=cov_state_pred_dependent,
            t_ind_max=TIME_AMOUNT,
            t_dep_max=TIME_DEP_AMOUNT,
            save_path=mean_cov_png,
            legend_title=None,
            ground_truth_label="Ground Truth",
            independent_label="ASD-FEX-TFDM-independent",
            dependent_label="ASD-FEX-TFDM-dependent",
            show_legend=False,
            font_size=26,
        )
        plt.close("all")

        # 4) Third-order 2x2 (no internal legend)
        plot_third_order_moments_ind_dep_2x2(
            moment3_state_record_ind=moment3_state_record,
            moment3_state_pred_ind=moment3_state_pred,
            moment3_state_pred_dep=moment3_state_pred_dependent,
            Time_ind=Time_record,
            Time_dep=Time_dep,
            save_path=third_png,
            title_suffix=params_name,
            legend_title=None,
            ground_truth_label="Ground Truth",
            independent_label="ASD-FEX-TFDM-independent",
            dependent_label="ASD-FEX-TFDM-dependent",
            show_legend=False,
        )
        plt.close("all")

        # Load images
        img_dep = mpimg.imread(dep_png)
        img_ind = mpimg.imread(ind_png)
        img_mean_cov = mpimg.imread(mean_cov_png)
        img_third = mpimg.imread(third_png)

    # Composite figure: header legend + 2x2 content + footer (Equipart)
    fig = plt_comp.figure(figsize=(20, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.12, 1, 1, 0.08], hspace=0.25, wspace=0.12)

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    # Centered legend in header.
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], color="black", lw=3, label="Ground Truth"),
        Line2D([0], [0], color="#ff7f0e", lw=3, label="ASD-FEX-TFDM-independent"),
        Line2D([0], [0], color="#1f77b4", lw=3, label="ASD-FEX-TFDM-dependent"),
    ]
    ax_legend.legend(
        handles=leg_handles,
        loc="center",
        ncol=3,
        fontsize=font_size,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )

    ax_dep = fig.add_subplot(gs[1, 0])
    ax_dep.imshow(img_dep)
    ax_dep.axis("off")
    # Blue box around dependent panel
    for spine in ax_dep.spines.values():
        spine.set_visible(True)
    ax_dep.spines["top"].set_color("blue")
    ax_dep.spines["top"].set_linewidth(4)
    ax_dep.spines["bottom"].set_color("blue")
    ax_dep.spines["bottom"].set_linewidth(4)
    ax_dep.spines["left"].set_color("blue")
    ax_dep.spines["left"].set_linewidth(4)
    ax_dep.spines["right"].set_color("blue")
    ax_dep.spines["right"].set_linewidth(4)
    ax_dep.text(0.5, 1.08, "dependent", transform=ax_dep.transAxes, ha="center", fontsize=font_size, color="blue")

    ax_ind = fig.add_subplot(gs[2, 0])
    ax_ind.imshow(img_ind)
    ax_ind.axis("off")
    # Orange box around independent panel
    for spine in ax_ind.spines.values():
        spine.set_visible(True)
    ax_ind.spines["top"].set_color("#ff7f0e")
    ax_ind.spines["top"].set_linewidth(4)
    ax_ind.spines["bottom"].set_color("#ff7f0e")
    ax_ind.spines["bottom"].set_linewidth(4)
    ax_ind.spines["left"].set_color("#ff7f0e")
    ax_ind.spines["left"].set_linewidth(4)
    ax_ind.spines["right"].set_color("#ff7f0e")
    ax_ind.spines["right"].set_linewidth(4)
    ax_ind.text(0.5, 1.08, "independent", transform=ax_ind.transAxes, ha="center", fontsize=font_size, color="#ff7f0e")

    ax_mc = fig.add_subplot(gs[1, 1])
    ax_mc.imshow(img_mean_cov)
    ax_mc.axis("off")
    ax_mc.set_title("Mean & Covariance", fontsize=font_size)

    ax_third = fig.add_subplot(gs[2, 1])
    ax_third.imshow(img_third)
    ax_third.axis("off")
    ax_third.set_title("Third-order moments", fontsize=font_size)

    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.axis("off")
    ax_footer.text(0.5, 0.5, params_name, transform=ax_footer.transAxes, ha="center", va="center", fontsize=font_size)

    plt_comp.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.lower().endswith(".pdf"):
        png_path = save_path[:-4] + ".png"
        plt_comp.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved composite to: {png_path}")
    print(f"[INFO] Saved composite to: {save_path}")
    plt_comp.close(fig)
    return None


def plot_discussion_choice4_triad_grid(
    packs,
    save_path,
    row_labels=("Equipart", "Cascade", "Dual cascade"),
    fs=25,
):
    """
    3×7 panel figure: rows = regimes, columns = cov(u_i,u_i) for i=1,2,3 and
    ⟨M123⟩, ⟨M122⟩, ⟨M133⟩, ⟨M223⟩. Each panel: ground truth (black), ASD-FEX-TFDM
    (orange), ASD-FEX-SRAN (pink), ASD-FEX-VAE (green), all over t∈[0,20].

    `packs` is a length-3 list of dicts from ``discussion_choice5_rollout(..., plot_composite=False)``:
    Time_ind, cov_gt, moment3_gt, cov_pred_tfdm, moment3_pred_tfdm,
    cov_pred_sran, moment3_pred_sran, cov_pred_vae, moment3_pred_vae.
    Falls back to ``cov_pred_ind`` / ``moment3_pred_ind`` if *_tfdm keys are absent.
    """
    from matplotlib.lines import Line2D

    assert len(packs) == len(row_labels) == 3
    moment_idx = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    moment_latex = [
        r"$\langle M_{123} \rangle$",
        r"$\langle M_{122} \rangle$",
        r"$\langle M_{133} \rangle$",
        r"$\langle M_{223} \rangle$",
    ]
    col_labels = [
        r"$\mathrm{cov}(u_1,u_1)$",
        r"$\mathrm{cov}(u_2,u_2)$",
        r"$\mathrm{cov}(u_3,u_3)$",
    ] + moment_latex

    fig, axes = plt.subplots(3, 7, figsize=(28, 10), sharex=False)

    gt_color = "black"
    tfdm_color = "#ff7f0e"
    sran_color = "#e377c2"
    vae_color = "#2ca02c"

    for row, (axrow, pack, rlabel) in enumerate(zip(axes, packs, row_labels)):
        Time_ind = np.asarray(pack["Time_ind"], dtype=float).ravel()
        cov_gt = pack["cov_gt"]
        cov_tfdm = pack.get("cov_pred_tfdm", pack["cov_pred_ind"])
        cov_sran = pack.get("cov_pred_sran", cov_tfdm)
        cov_vae = pack.get("cov_pred_vae", cov_tfdm)
        m_gt = pack["moment3_gt"]
        m_tfdm = pack.get("moment3_pred_tfdm", pack["moment3_pred_ind"])
        m_sran = pack.get("moment3_pred_sran", m_tfdm)
        m_vae = pack.get("moment3_pred_vae", m_tfdm)

        for col in range(7):
            ax = axrow[col]
            if col < 3:
                i = col
                y_gt = cov_gt[i, i, :]
                y_tfdm = cov_tfdm[i, i, :]
                y_sran = cov_sran[i, i, :]
                y_vae = cov_vae[i, i, :]
            else:
                mi, mj, mk = moment_idx[col - 3]
                y_gt = m_gt[mi, mj, mk, :]
                y_tfdm = m_tfdm[mi, mj, mk, :]
                y_sran = m_sran[mi, mj, mk, :]
                y_vae = m_vae[mi, mj, mk, :]

            ax.plot(
                Time_ind,
                y_gt,
                "-",
                color=gt_color,
                linewidth=2.0,
                label="Ground Truth",
            )
            ax.plot(
                Time_ind,
                y_tfdm,
                "-",
                color=tfdm_color,
                linewidth=1.6,
                label="ASD-FEX-TFDM",
            )
            ax.plot(
                Time_ind,
                y_sran,
                "-",
                color=sran_color,
                linewidth=1.6,
                label="ASD-FEX-SRAN",
            )
            ax.plot(
                Time_ind,
                y_vae,
                "-",
                color=vae_color,
                linewidth=1.6,
                label="ASD-FEX-VAE",
            )

            if row == 0:
                ax.set_title(col_labels[col], fontsize=fs)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=fs)
            if row == 2:
                ax.set_xlabel("Time", fontsize=fs)
            ax.tick_params(axis="both", labelsize=fs)
            ax.grid(False)

    handles = [
        Line2D([0], [0], color=gt_color, lw=2.5, linestyle="-", label="Ground Truth"),
        Line2D([0], [0], color=tfdm_color, lw=2.5, linestyle="-", label="ASD-FEX-TFDM"),
        Line2D([0], [0], color=sran_color, lw=2.5, linestyle="-", label="ASD-FEX-SRAN"),
        Line2D([0], [0], color=vae_color, lw=2.5, linestyle="-", label="ASD-FEX-VAE"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=fs,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved discussion choice 4 grid to: {save_path}")


def run_discussion_choice4_triad_grid(
    args,
    base_path: str,
    dir_example: str,
    model_name: str,
    rollout_worker,
    regimes=("equipart", "cascade", "dual_cascade"),
    fs=25,
):
    """
    Run `rollout_worker(plot_composite=False)` per regime (mutates and restores
    ``args.params_name`` and ``args.LOG_SAVE_PATH``), then call
    :func:`plot_discussion_choice4_triad_grid`.

    ``rollout_worker`` must return the pack dict expected by
    :func:`plot_discussion_choice4_triad_grid` (e.g. ``discussion_choice5_rollout``).
    """
    _saved_pn = args.params_name
    _saved_log = args.LOG_SAVE_PATH
    packs = []
    try:
        for regime in regimes:
            args.params_name = regime
            args.LOG_SAVE_PATH = f"{base_path}/{regime}"
            packs.append(rollout_worker(plot_composite=False))
    finally:
        args.params_name = _saved_pn
        args.LOG_SAVE_PATH = _saved_log
    out_dir = os.path.join(dir_example, model_name, "Results")
    os.makedirs(out_dir, exist_ok=True)
    save_pdf = os.path.join(out_dir, "discussion_choice4_cov_moments_grid.pdf")
    plot_discussion_choice4_triad_grid(packs, save_path=save_pdf, fs=fs)


def plot_probability_distributions(u_all, u_pred, Time_record, save_path=None, title_suffix="FEX-framework"):
    """
    Plot probability distributions and joint distributions comparing ground truth and prediction.
    
    Args:
        u_all (np.ndarray): Ground truth trajectories (NPATH, 3, time_steps)
        u_pred (np.ndarray): Predicted trajectories (NPATH, 3, time_steps)
        Time_record (np.ndarray): Time points
        save_path (str): Path to save the plot
        title_suffix (str): Suffix for the title
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select a time point for analysis (e.g., middle of simulation)
    time_idx = len(Time_record) // 2
    print(f"Plotting distributions at time step {time_idx} (t = {Time_record[time_idx]:.2f})")
    
    # Extract data at the selected time point
    u1_truth = u_all[:, 0, time_idx]
    u2_truth = u_all[:, 1, time_idx]
    u3_truth = u_all[:, 2, time_idx]
    
    u1_pred = u_pred[:, 0, time_idx]
    u2_pred = u_pred[:, 1, time_idx]
    u3_pred = u_pred[:, 2, time_idx]
    
    # Top row: Probability distributions for each mode
    mode_data = [(u1_truth, u1_pred, 'u1'), (u2_truth, u2_pred, 'u2'), (u3_truth, u3_pred, 'u3')]
    
    for idx, (truth_data, pred_data, mode_name) in enumerate(mode_data):
        ax = axs[0, idx]
        
        # Create histogram bins
        all_data = np.concatenate([truth_data, pred_data])
        bins = np.linspace(np.min(all_data), np.max(all_data), 50)
        
        # Plot histograms
        hist_truth, bin_edges = np.histogram(truth_data, bins=bins, density=True)
        hist_pred, _ = np.histogram(pred_data, bins=bins, density=True)
        
        # Plot with log scale
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.semilogy(bin_centers, hist_truth, 'b-', linewidth=2, label=f'Ground Truth {mode_name}')
        ax.semilogy(bin_centers, hist_pred, 'k--', linewidth=2, label=f'Prediction {mode_name}')
        
        # Remove grid and clean up styling
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('Probability Density')
        ax.set_title('')
        ax.legend(frameon=False)
        
        # Set y-axis limits similar to the image
        ax.set_ylim(1e-2, 10) # 10 for cascade, 3e-01 for equipart
        ax.set_xlim(-5, 5)
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Bottom row: Joint distributions (scatter plots)
    joint_pairs = [(u1_truth, u2_truth, u1_pred, u2_pred, 'u1', 'u2'),
                   (u1_truth, u3_truth, u1_pred, u3_pred, 'u1', 'u3'),
                   (u2_truth, u3_truth, u2_pred, u3_pred, 'u2', 'u3')]
    
    for idx, (x_truth, y_truth, x_pred, y_pred, x_label, y_label) in enumerate(joint_pairs):
        ax = axs[1, idx]
        
        # Plot ground truth (blue dots only, like in the reference image)
        ax.scatter(x_truth, y_truth, c='blue', s=1, alpha=0.6, label='Ground Truth')
        
        ax.set_xlabel('')
        ax.set_ylabel(y_label)
        ax.set_title('')
        
        # Remove grid and clean up styling
        ax.grid(False)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'probability_distributions.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_mean_comparison_tfdm_vae_nn(mean_state_record, mean_state_nn, mean_state_tfdm, mean_state_vae, Time_record, save_path=None):
    """Plot mean components with Ground Truth / FEX+NN / FEX+TFDM / FEX+VAE."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for i in range(3):
        axs[i].plot(Time_record, mean_state_record[i], linestyle=':', color='black', linewidth=3,
                    label=fr'Ground Truth $\langle u_{i+1} \rangle$')
        axs[i].plot(Time_record, mean_state_nn[i], linestyle='-', color='pink', linewidth=2.5,
                    label=fr'Prediction - FEX+NN $\langle u_{i+1} \rangle$')
        axs[i].plot(Time_record, mean_state_tfdm[i], linestyle='-', color='orange', linewidth=2.5,
                    label=fr'Prediction - FEX+TFDM $\langle u_{i+1} \rangle$')
        axs[i].plot(Time_record, mean_state_vae[i], linestyle='-', color='green', linewidth=2.5,
                    label=fr'Prediction - FEX+VAE $\langle u_{i+1} \rangle$')
        axs[i].set_ylabel(fr'Mean $u_{i+1}$', fontsize=15)
        axs[i].set_title(f'Component {i+1}', fontsize=18)
        axs[i].legend(loc='upper right', frameon=False, fontsize=11)
        axs[i].tick_params(axis='both', labelsize=12)
    axs[2].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle('Mean Values: FEX+NN / FEX+TFDM / FEX+VAE', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.9)
    if save_path:
        plt.savefig(os.path.join(save_path, 'mean_components_over_time.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_covariance_comparison_tfdm_vae_nn(cov_state_record, cov_state_nn, cov_state_tfdm, cov_state_vae, Time_record, save_path=None):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(Time_record, cov_state_record[i, j], linestyle=':', color='black', linewidth=2, label='Ground Truth')
            axs[i, j].plot(Time_record, cov_state_nn[i, j], linestyle='-', color='pink', linewidth=2, label='FEX+NN')
            axs[i, j].plot(Time_record, cov_state_tfdm[i, j], linestyle='-', color='orange', linewidth=2, label='FEX+TFDM')
            axs[i, j].plot(Time_record, cov_state_vae[i, j], linestyle='-', color='green', linewidth=2, label='FEX+VAE')
            axs[i, j].set_title(f'Cov(u{i+1}, u{j+1})', fontsize=12)
            axs[i, j].set_xlabel('Time', fontsize=10)
            axs[i, j].set_ylabel('Covariance', fontsize=10)
            axs[i, j].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.suptitle('Covariance Comparison: FEX+NN / FEX+TFDM / FEX+VAE', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    if save_path:
        plt.savefig(os.path.join(save_path, 'covariance_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_energy_comparison_tfdm_vae_nn(Energy_MC_all, Energy_MC_nn, Energy_MC_tfdm, Energy_MC_vae, Time_record, save_path=None):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    energy_labels = ['Total', 'Mode 1', 'Mode 2', 'Mode 3']
    for idx, label in enumerate(energy_labels):
        ax = axs[idx // 2, idx % 2]
        ax.plot(Time_record, Energy_MC_all[idx], color='black', linestyle=':', linewidth=2, label=f'{label} (Truth)')
        ax.plot(Time_record, Energy_MC_nn[idx], color='pink', linewidth=2, label=f'{label} (FEX+NN)')
        ax.plot(Time_record, Energy_MC_tfdm[idx], color='orange', linewidth=2, label=f'{label} (FEX+TFDM)')
        ax.plot(Time_record, Energy_MC_vae[idx], color='green', linewidth=2, label=f'{label} (FEX+VAE)')
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.suptitle('Energy Comparison: FEX+NN / FEX+TFDM / FEX+VAE', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    if save_path:
        plt.savefig(os.path.join(save_path, 'energy_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_third_order_moments_tfdm_vae_nn(moment3_state_record, moment3_state_nn, moment3_state_tfdm, moment3_state_vae, Time_record, save_path=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    moment_indices = [(0, 1, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2)]
    moment_labels = [r'$\langle M_{123} \rangle$', r'$\langle M_{122} \rangle$', r'$\langle M_{133} \rangle$', r'$\langle M_{223} \rangle$']
    for idx, (i, j, k) in enumerate(moment_indices):
        axs[idx].plot(Time_record, moment3_state_record[i, j, k, :], linestyle=':', color='black', linewidth=3, label=f'Ground Truth {moment_labels[idx]}')
        axs[idx].plot(Time_record, moment3_state_nn[i, j, k, :], linestyle='-', color='pink', linewidth=2.5, label=f'FEX+NN {moment_labels[idx]}')
        axs[idx].plot(Time_record, moment3_state_tfdm[i, j, k, :], linestyle='-', color='orange', linewidth=2.5, label=f'FEX+TFDM {moment_labels[idx]}')
        axs[idx].plot(Time_record, moment3_state_vae[i, j, k, :], linestyle='-', color='green', linewidth=2.5, label=f'FEX+VAE {moment_labels[idx]}')
        axs[idx].set_ylabel('3rd Moment', fontsize=15)
        axs[idx].set_title(moment_labels[idx], fontsize=18)
        axs[idx].legend(loc='upper right', frameon=False, fontsize=11)
    axs[3].set_xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.suptitle('Third-Order Moments: FEX+NN / FEX+TFDM / FEX+VAE', fontsize=20, y=1.02)
    plt.subplots_adjust(top=0.95)
    if save_path:
        plt.savefig(os.path.join(save_path, 'third_order_moments_over_time.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_probability_distributions_tfdm_vae_nn(u_all, u_pred_nn, u_pred_tfdm, u_pred_vae, Time_record, save_path=None):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    time_idx = len(Time_record) // 2
    print(f"Plotting distributions at time step {time_idx} (t = {Time_record[time_idx]:.2f})")

    truth = [u_all[:, 0, time_idx], u_all[:, 1, time_idx], u_all[:, 2, time_idx]]
    nnv = [u_pred_nn[:, 0, time_idx], u_pred_nn[:, 1, time_idx], u_pred_nn[:, 2, time_idx]]
    tfdm = [u_pred_tfdm[:, 0, time_idx], u_pred_tfdm[:, 1, time_idx], u_pred_tfdm[:, 2, time_idx]]
    vae = [u_pred_vae[:, 0, time_idx], u_pred_vae[:, 1, time_idx], u_pred_vae[:, 2, time_idx]]
    names = ['u1', 'u2', 'u3']
    for idx in range(3):
        ax = axs[0, idx]
        all_data = np.concatenate([truth[idx], nnv[idx], tfdm[idx], vae[idx]])
        bins = np.linspace(np.min(all_data), np.max(all_data), 50)
        hc_t, be = np.histogram(truth[idx], bins=bins, density=True)
        hc_n, _ = np.histogram(nnv[idx], bins=bins, density=True)
        hc_f, _ = np.histogram(tfdm[idx], bins=bins, density=True)
        hc_v, _ = np.histogram(vae[idx], bins=bins, density=True)
        centers = (be[:-1] + be[1:]) / 2
        ax.semilogy(centers, hc_t, 'k:', linewidth=2, label=f'Ground Truth {names[idx]}')
        ax.semilogy(centers, hc_n, color='pink', linewidth=2, label=f'FEX+NN {names[idx]}')
        ax.semilogy(centers, hc_f, color='orange', linewidth=2, label=f'FEX+TFDM {names[idx]}')
        ax.semilogy(centers, hc_v, color='green', linewidth=2, label=f'FEX+VAE {names[idx]}')
        ax.legend(frameon=False)
        ax.set_ylabel('Probability Density')

    pairs = [(0, 1), (0, 2), (1, 2)]
    for idx, (a, b) in enumerate(pairs):
        ax = axs[1, idx]
        ax.scatter(truth[a], truth[b], c='blue', s=1, alpha=0.3, label='Ground Truth')
        ax.scatter(nnv[a], nnv[b], c='pink', s=1, alpha=0.2, label='FEX+NN')
        ax.scatter(tfdm[a], tfdm[b], c='orange', s=1, alpha=0.2, label='FEX+TFDM')
        ax.scatter(vae[a], vae[b], c='green', s=1, alpha=0.2, label='FEX+VAE')
        ax.legend(frameon=False, fontsize=8)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'probability_distributions.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return fig





def plot_energy_conservation(coefficients, noise_levels=None, save_dir=None):
    """
    Plot the sum of cross-terms (x1x2 + x2x3 + x1x3) for different noise levels.
    
    Args:
        coefficients (dict): Dictionary containing coefficients for each dimension
        noise_levels (list): List of noise levels (optional, will be inferred from data)
        save_path (str): Path to save the plot
    """
    # Set publication style
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 100,
    })
    
    # Extract cross-terms from each dimension
    x2x3_values = coefficients['dim_1']['x2x3']  # From dimension 1
    x1x3_values = coefficients['dim_2']['x1x3']  # From dimension 2  
    x1x2_values = coefficients['dim_3']['x1x2']  # From dimension 3
    
    # Calculate the sum
    cross_terms_sum = [x2x3 + x1x3 + x1x2 for x2x3, x1x3, x1x2 in zip(x2x3_values, x1x3_values, x1x2_values)]
    
    # If noise_levels not provided, create default range
    if noise_levels is None:
        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8][:len(cross_terms_sum)]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    set_figure_position(x=100, y=100, width=1000, height=600)
    
    # Plot the sum of cross-terms
    ax.plot(noise_levels, cross_terms_sum, 'o-', color='#2171B5', linewidth=3, markersize=8, 
            label='x1x2 + x2x3 + x1x3')
    
   
    
    # Add value annotations
    for i, (x, y) in enumerate(zip(noise_levels, cross_terms_sum)):
        ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('Noise Level', fontsize=14)
    ax.set_ylabel('Cross-terms Sum', fontsize=14)
    ax.set_title('Sum of Cross-terms vs Noise Level', fontsize=16)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits to show full range
    ax.set_xlim(min(noise_levels) - 0.1, max(noise_levels) + 0.1)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Energy_conservation_sum.pdf'), dpi=300, bbox_inches='tight')
        print(f"Cross-terms sum plot saved to {os.path.join(save_dir, 'Energy_conservation_sum.pdf')}")
    
    plt.show()
    
    return fig



def plot_comparative_grid(u_all, u_pred_single, u_pred_ensemble,
                             Energy_MC_all, Energy_MC_single, Energy_MC_ensemble,
                             Time_record, dt,
                             save_path=None, title_suffix="FEX-framework-3D"):
    """
    4x6 comparison figure for 3D triad system:

      Rows 1–3 (3D scatter, one panel = one time snapshot):
        - Row 1: Numerical model (u_all)
        - Row 2: Emulator (single)
        - Row 3: Thermalized (ensemble)

      Row 4:
        - Energy vs time (log scale), up to that column's snapshot time.

    Args
    ----
    u_all, u_pred_single, u_pred_ensemble : np.ndarray
        Shape (NPATH, 3, Nt) or (NPATH, 3, Nt+1).
    Energy_MC_* : np.ndarray
        Shape (4, Nt)  # [total, u1, u2, u3]
    Time_record : np.ndarray
        Time indices (0..Nt-1 or Nt), length Nt.
    dt : float
        Time step.
    """

    # -------- style --------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    })

    Nt = Time_record.shape[0]
    total_steps = Nt
    # target physical times (秒)
    target_times = [0.0, 1.0, 5.0, 7.0, 10.0, min(20.0, (total_steps - 1) * dt)]

    time_indices = []
    time_values = []
    for t in target_times:
        idx = min(int(round(t / dt)), total_steps - 1)
        time_indices.append(idx)
        time_values.append(idx * dt)

    # -------- 全局坐标范围 (u1,u2,u3) 保证所有 3D 面板范围一致 --------
    all_states = np.concatenate([
        u_all.reshape(-1, 3),
        u_pred_single.reshape(-1, 3),
        u_pred_ensemble.reshape(-1, 3)
    ], axis=0)
    u1_min, u1_max = np.percentile(all_states[:, 0], [1, 99])
    u2_min, u2_max = np.percentile(all_states[:, 1], [1, 99])
    u3_min, u3_max = np.percentile(all_states[:, 2], [1, 99])

    # -------- figure 布局 --------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'3D triad: Numerical vs Emulator vs Thermalized – {title_suffix}',
                 fontsize=14, fontweight='bold')

    row_labels = ['Numerical model', 'Emulator', 'Thermalized']
    data_sets = [u_all, u_pred_single, u_pred_ensemble]

    # 固定随机种子，保证截图可复现
    np.random.seed(0)
    NPATH = u_all.shape[0]
    sample_size = min(1000, NPATH)
    sample_idx = np.random.choice(NPATH, sample_size, replace=False)

    # -------- 上三行：3D scatter --------
    for row in range(3):
        U = data_sets[row]  # (NPATH, 3, Nt)

        for col in range(6):
            tidx = time_indices[col]
            snapshot = U[sample_idx, :, tidx]  # (sample_size, 3)

            # subplot index: 4 行 × 6 列
            ax_idx = row * 6 + col + 1
            ax = fig.add_subplot(4, 6, ax_idx, projection='3d')

            ax.scatter(snapshot[:, 0],
                       snapshot[:, 1],
                       snapshot[:, 2],
                       s=2, alpha=0.25, c='tab:purple')

            ax.set_xlim([u1_min, u1_max])
            ax.set_ylim([u2_min, u2_max])
            ax.set_zlim([u3_min, u3_max])

            # 只在左边几列标轴，避免太乱
            if col == 0:
                ax.set_xlabel('u1')
                ax.set_ylabel('u2')
                ax.set_zlabel('u3')
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            # 第一行加时间标题
            if row == 0:
                ax.set_title(f't={time_values[col]:.1f}s\nstep={tidx}',
                             fontsize=9, pad=2)

            # 每行左边加行标题
            if col == 0:
                ax.text2D(-0.25, 0.5, row_labels[row],
                          transform=ax.transAxes,
                          fontsize=11, fontweight='bold',
                          rotation=90, va='center', ha='right')

    # -------- 最后一行：energy vs time (log) --------
    t_full = Time_record * dt

    for col in range(6):
        tidx = time_indices[col]
        ax_idx = 3 * 6 + col + 1
        ax = fig.add_subplot(4, 6, ax_idx)

        t_slice = t_full[:tidx + 1]

        # Numerical
        if Energy_MC_all is not None and Energy_MC_all.shape[1] > tidx:
            ax.plot(t_slice, Energy_MC_all[0, :tidx+1],
                    color='tab:red', lw=1.5, label='Numerical')

        # Emulator
        if Energy_MC_single is not None and Energy_MC_single.shape[1] > tidx:
            ax.plot(t_slice, Energy_MC_single[0, :tidx+1],
                    color='gray', lw=1.0, ls='--', label='Emulator')

        # Thermalized
        if Energy_MC_ensemble is not None and Energy_MC_ensemble.shape[1] > tidx:
            ax.plot(t_slice, Energy_MC_ensemble[0, :tidx+1],
                    color='tab:blue', lw=1.2, ls='-', label='Thermalized')

        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')

        if col == 0:
            ax.set_ylabel('Total kinetic energy')
            ax.legend(frameon=False, fontsize=8)

    fig.text(0.5, 0.03,
             'Energy evolution from different models (log scale)',
             ha='center', fontsize=11)

    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.93])

    # -------- save --------
    if save_path is not None:
        base = f'comparative_grid_3D_{title_suffix.replace(" ", "_")}'
        pdf_path = os.path.join(save_path, base + '.pdf')
        png_path = os.path.join(save_path, base + '.png')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {pdf_path}")
        print(f"[INFO] Saved: {png_path}")

    return fig


def plot_triad_3d_snapshot(u,
                           time_idx,
                           dt,
                           title="Numerical model",
                           cmap="viridis",
                           sample_size=5000,
                           save_path=None):
    """
    u: np.ndarray, shape (NPATH, 3, Nt+1)
       Triad trajectories (can be u_all, u_pred_single, etc.)
    time_idx: int
       Index in time (0..Nt)
    dt: float
       Time step (only for labeling)
    """
    NPATH = u.shape[0]
    Nt = u.shape[2]

    time_idx = min(time_idx, Nt - 1)
    t_val = time_idx * dt

    # sample trajectories if too many
    if sample_size is not None and sample_size < NPATH:
        np.random.seed(0)
        idx = np.random.choice(NPATH, sample_size, replace=False)
        snapshot = u[idx, :, time_idx]
    else:
        snapshot = u[:, :, time_idx]

    x = snapshot[:, 0]
    y = snapshot[:, 1]
    z = snapshot[:, 2]

    # color by radius in state space (like "magnitude")
    r = np.sqrt(x**2 + y**2 + z**2)

    # global bounds to make a nice box
    pad = 0.1 * (np.max(snapshot) - np.min(snapshot) + 1e-8)
    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    zmin, zmax = z.min() - pad, z.max() + pad

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=r, s=2, alpha=0.25, cmap=cmap)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")
    ax.set_zlabel(r"$u_3$")

    ax.set_title(f"{title}\n t = {t_val:.2f}, step = {time_idx}")

    # make it look more "DNS box"-like
    ax.view_init(elev=20, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=r"$|\mathbf{u}|$")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved 3D snapshot to {save_path}")

    return fig


def plot_triad_3d_time_grid_matplotlib_cloud_3x3_times(
    u_gt: np.ndarray,
    u_ind: np.ndarray,
    u_dep=None,
    dt: float = 0.01,
    time_points: list | None = None,
    save_path: str | None = None,
    grid_rows: int = 2,
    grid_cols: int = 3,
    elev: float = 20,
    azim: float = -60,
    opacity: float = 0.55,
    point_size: float = 6.0,
    cmap: str = "turbo",
    sample_size: int = 15000,
):
    """
    Matplotlib-based 3D scatter "cloud" renderer (no mesh, no 3D grid/panes).

    Layout: grid_rows x grid_cols (e.g. 2x3 for 6 panels, 1x3 for 3 panels).
    Default time_points: [0, 4, 8, 12, 16, 20].

    u_* shape: (NPATH, 3, Nt+1)
    """
    if time_points is None:
        time_points = [0, 4, 8, 12, 16, 20]

    # Ensure Matplotlib is only imported if we actually call this function.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Use consistent samples across all panels/datasets.
    NPATH = u_gt.shape[0]
    sample_size = min(sample_size, NPATH)
    np.random.seed(0)
    sample_idx = np.random.choice(NPATH, sample_size, replace=False)

    Nt_gt = u_gt.shape[2]
    max_idx_gt = Nt_gt - 1
    time_indices = [min(int(round(t / dt)), max_idx_gt) for t in time_points]
    time_values = [idx * dt for idx in time_indices]

    # Global bounds computed from GT at requested times (consistent look).
    states = []
    for tidx in time_indices:
        snap = u_gt[sample_idx, :, tidx]  # (sample, 3)
        states.append(snap)
    all_states = np.concatenate(states, axis=0)
    u1_min, u1_max = np.percentile(all_states[:, 0], [1, 99])
    u2_min, u2_max = np.percentile(all_states[:, 1], [1, 99])
    u3_min, u3_max = np.percentile(all_states[:, 2], [1, 99])

    n_rows, n_cols = int(grid_rows), int(grid_cols)
    fig = plt.figure(figsize=(4.0 * n_cols, 3.7 * n_rows))

    n_panels = min(len(time_indices), n_rows * n_cols)
    for panel_idx in range(n_panels):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection="3d")
        ax.set_facecolor("white")
        ax.grid(False)

        tidx = time_indices[panel_idx]
        t_val = time_values[panel_idx]

        points_list = [u_gt[sample_idx, :, tidx], u_ind[sample_idx, :, tidx]]
        if u_dep is not None and tidx < u_dep.shape[2]:
            points_list.append(u_dep[sample_idx, :, tidx])

        points_xyz = np.concatenate(points_list, axis=0)  # (k*sample, 3)
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        ax.scatter(
            x,
            y,
            z,
            c=r,
            cmap=cmap,
            s=point_size,
            alpha=opacity,
            linewidths=0,
            marker="o",
        )

        ax.set_xlim([u1_min, u1_max])
        ax.set_ylim([u2_min, u2_max])
        ax.set_zlim([u3_min, u3_max])

        ax.view_init(elev=elev, azim=azim)

        # Remove any 3D "grid/box" look.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_title(f"t={t_val:g}", fontsize=18, pad=6)

        # Hide panes/borders if available in this Matplotlib version.
        try:
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
        except Exception:
            pass
        ax.xaxis.line.set_alpha(0)
        ax.yaxis.line.set_alpha(0)
        ax.zaxis.line.set_alpha(0)

    # If time_points < 6, hide remaining axes in the 2x3 grid.
    for panel_idx in range(n_panels, n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection="3d")
        ax.set_axis_off()

    if save_path:
        # If the user wants a "image then pdf" workflow, write both PNG and PDF.
        # Matplotlib can render directly to either, but we explicitly produce a PNG too.
        save_kwargs = dict(dpi=300, bbox_inches="tight")
        if save_path.lower().endswith(".pdf"):
            png_path = save_path[:-4] + ".png"
            fig.savefig(png_path, **save_kwargs)
            print(f"[INFO] Saved Matplotlib 3x3 3D cloud grid to: {png_path}")

        fig.savefig(save_path, **save_kwargs)
        print(f"[INFO] Saved Matplotlib 3x3 3D cloud grid to: {save_path}")

    plt.close(fig)
    return None


def plot_triad_3d_grid(u_all, u_pred_single, u_pred_ensemble,
                       Time_record, dt,
                       save_path=None,
                       title_suffix="triad_3D"):
    """
    3x6 grid:
      rows: Numerical / Emulator / Thermalized
      cols: t = 0, 1, 5, 7, 10, 20 (or max available)
    """
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
    })

    Nt = Time_record.shape[0]
    total_steps = Nt
    target_times = [0.0, 1.0, 5.0, 7.0, 10.0, min(20.0, (total_steps - 1) * dt)]

    time_indices = []
    time_values = []
    for t in target_times:
        idx = min(int(round(t / dt)), total_steps - 1)
        time_indices.append(idx)
        time_values.append(idx * dt)

    # global ranges over all models & times
    all_states = np.concatenate([
        u_all.reshape(-1, 3),
        u_pred_single.reshape(-1, 3),
        u_pred_ensemble.reshape(-1, 3)
    ], axis=0)
    u1_min, u1_max = np.percentile(all_states[:, 0], [1, 99])
    u2_min, u2_max = np.percentile(all_states[:, 1], [1, 99])
    u3_min, u3_max = np.percentile(all_states[:, 2], [1, 99])

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Triad phase-space clouds – {title_suffix}",
                 fontsize=14, fontweight="bold")

    row_labels = ['Numerical model', 'Emulator', 'Thermalized']
    data_sets = [u_all, u_pred_single, u_pred_ensemble]

    np.random.seed(0)
    NPATH = u_all.shape[0]
    sample_size = min(2500, NPATH)
    sample_idx = np.random.choice(NPATH, sample_size, replace=False)

    for row in range(3):
        U = data_sets[row]
        for col in range(6):
            tidx = time_indices[col]
            t_val = time_values[col]

            snapshot = U[sample_idx, :, tidx]
            x = snapshot[:, 0]
            y = snapshot[:, 1]
            z = snapshot[:, 2]

            r = np.sqrt(x**2 + y**2 + z**2)

            ax_idx = row * 6 + col + 1
            ax = fig.add_subplot(3, 6, ax_idx, projection="3d")
            sc = ax.scatter(x, y, z, c=r, s=2, alpha=0.25, cmap="viridis")

            ax.set_xlim([u1_min, u1_max])
            ax.set_ylim([u2_min, u2_max])
            ax.set_zlim([u3_min, u3_max])

            if col == 0:
                ax.set_xlabel(r"$u_1$")
                ax.set_ylabel(r"$u_2$")
                ax.set_zlabel(r"$u_3$")
                ax.text2D(-0.22, 0.5, row_labels[row],
                          transform=ax.transAxes,
                          fontsize=10, fontweight="bold",
                          rotation=90, va="center", ha="right")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            if row == 0:
                ax.set_title(f"t={t_val:.1f}s\nstep={tidx}",
                             fontsize=8, pad=2)

            ax.view_init(elev=20, azim=-60)

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    if save_path is not None:
        base = f"triad_3D_grid_{title_suffix.replace(' ', '_')}"
        pdf_path = os.path.join(save_path, base + ".pdf")
        png_path = os.path.join(save_path, base + ".png")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {pdf_path}")
        print(f"[INFO] Saved: {png_path}")

    return fig
