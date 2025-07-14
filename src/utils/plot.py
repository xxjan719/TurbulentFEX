from ast import Dict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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


def plot_NOISE_LEVEL_EFFECT(coeff: dict, noise_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], save_dir: str = ""):
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
    
    # Define row labels for the 4 rows
    row_labels = ['x1', 'x2', 'x3', 'Cross-term']
    
    # Create figure with 3 columns (dimensions) and 4 rows (terms), make it tall
    fig, axes = plt.subplots(4, 3, figsize=(20, 10), constrained_layout=True)
    fig.suptitle("Coefficient vs Noise Level for Each Dimension", fontsize=20)
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
                ax.plot(plot_noise_levels, coeff_values, 'o-', 
                        color=colors[row], linewidth=2, markersize=6, 
                        label=f'{term}')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Noise Level')
                ax.set_ylabel('Coefficient Value')
                # Use only the term as the subplot title
                if row < 3:
                    ax.set_title(f'{term}')
                else:
                    ax.set_title(f'{cross_term_name}')
                for i, (x, y) in enumerate(zip(plot_noise_levels, coeff_values)):
                    ax.annotate(f'{y:.6f}', (x, y), textcoords="offset points", 
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
        plt.savefig(os.path.join(save_dir, 'training_progress_grid.pdf'), 
                    dpi=300, bbox_inches='tight')
        print(f"Training progress grid plot saved to {os.path.join(save_dir, 'training_progress_grid.pdf')}")
    plt.show()
    
    
