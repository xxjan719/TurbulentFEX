import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def set_figure_position(x=100, y=100, width=800, height=600):
    """Set the position and size of the current figure window (only if supported)."""
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            manager.window.setGeometry(x, y, width, height)
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
        plt.savefig(os.path.join(save_dir, 'residual_covariance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(save_dir, 'residual_covariance_comparison.png')}")
    
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
        
        # Calculate time step info
        dt = selected_times[1] - selected_times[0] if len(selected_times) > 1 else 0.01
        total_time_steps = int(selected_times[-1] / dt) + 1
        
        plot_time_selection_info(total_time_steps, selected_indices, selected_times, dt, save_dir)
        
    else:
        print(f"Results file not found: {results_path}")