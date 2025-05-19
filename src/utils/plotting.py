import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def set_figure_position(x=100, y=100, width=800, height=600):
    """Set the position and size of the current figure window."""
    plt.get_current_fig_manager().window.setGeometry(x, y, width, height)

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
        axs[0].scatter(TT_MC, mean_MC[i], s=9, color=line_colors[key], label=fr'$\langle u_{i+1} \rangle$')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mean')
    axs[0].legend(loc='upper right', frameon=False)

    # Variance
    for i, key in enumerate(['u1', 'u2', 'u3']):
        axs[1].scatter(TT_MC, cov_MC[i, i], s=9, color=line_colors[key], label=fr'$\mathrm{{Var}}(u_{i+1})$')
    axs[1].scatter(TT_MC, np.sum(std_MC**2, axis=0), s=9, color=line_colors['total'], label='Total Var')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Variance')
    axs[1].legend(loc='upper left', frameon=False)

    # Cross-Covariance
    axs[2].scatter(TT_MC, cov_MC[0, 1], s=9, color=line_colors['extra1'], label=r'$\mathrm{Cov}(u_1, u_2)$')
    axs[2].scatter(TT_MC, cov_MC[0, 2], s=9, color=line_colors['extra2'], label=r'$\mathrm{Cov}(u_1, u_3)$')
    axs[2].scatter(TT_MC, cov_MC[1, 2], s=9, color=line_colors['extra3'], label=r'$\mathrm{Cov}(u_2, u_3)$')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Cross-Covariance')
    axs[2].legend(loc='lower left', frameon=False)

    # 3rd-Order Moments
    axs[3].scatter(TT_MC, M3_MC[0, 1, 2], s=9, color=line_colors['total'], label=r'$\langle M_{123} \rangle$')
    axs[3].scatter(TT_MC, M3_MC[0, 1, 1], s=9, color=line_colors['u1'], label=r'$\langle M_{122} \rangle$')
    axs[3].scatter(TT_MC, M3_MC[0, 2, 2], s=9, color=line_colors['u2'], label=r'$\langle M_{133} \rangle$')
    axs[3].scatter(TT_MC, M3_MC[1, 1, 2], s=9, color=line_colors['u3'], label=r'$\langle M_{223} \rangle$')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('3rd Moment')
    axs[3].legend(loc='upper right', frameon=False)

    # Energy (Truth)
    axs[4].scatter(TT_MC, Ene_MC[0], s=9, color=line_colors['total'], label='Total')
    axs[4].scatter(TT_MC, Ene_MC[1], s=9, color=line_colors['u1'], label='Mode 1')
    axs[4].scatter(TT_MC, Ene_MC[2], s=9, color=line_colors['u2'], label='Mode 2')
    axs[4].scatter(TT_MC, Ene_MC[3], s=9, color=line_colors['u3'], label='Mode 3')
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Energy (Truth)')
    axs[4].legend(loc='upper left', frameon=False)

    # Energy (Dynamics)
    axs[5].scatter(TT_MC, Ene_MC[0],  s=9, color=line_colors['total'], label='Truth')
    axs[5].scatter(TT_MC, Ene_dyn[0], s=9, color=line_colors['extra1'], label='Full Eqn.')
    axs[5].scatter(TT_MC, Ene_dyn[1], s=9, color=line_colors['extra2'], label='Lower Bound')
    axs[5].scatter(TT_MC, Ene_dyn[2], s=9, color=line_colors['extra3'], label='Upper Bound')
    axs[5].scatter(TT_MC, Ene_dyn[3], s=9, color=line_colors['gray'], label='Mean')
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
    plt.scatter(TT_MC, M112,s=9, label=r'$\langle M_{112} \rangle$')
    plt.scatter(TT_MC, M113,s=9, label=r'$\langle M_{113} \rangle$')
    plt.scatter(TT_MC, M233, s=9,label=r'$\langle M_{233} \rangle$')
    plt.scatter(TT_MC, M111, s=9,label=r'$\langle M_{111} \rangle$')

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

    # --- Subplot 1: Deviation in variance ---
    for i, color in zip(range(3), ['C0', 'C1', 'C2']):
        dev = (cov_MC[i, i, :] - cov_MC[i, i, -1]) / cov_MC[i, i, -1]
        axs[0].scatter(TT_MC, dev, s=9, color=color, label=fr'$u_{i+1}$')
    axs[0].scatter(TT_MC, trace_dev, s=9,color='k', label='trace(R)')  # use plot instead of scatter
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
        axs[1].scatter(TT_MC, dev, s=9, color=color, label=label)
    axs[1].scatter(TT_MC, trace_dev, s=9,color='k',label='trace(R)')  # also plot
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
        axs[2].scatter(TT_MC, dev, s=9, color=color, label=label)
    axs[2].set_xlabel('time')
    axs[2].set_title('deviation in 3rd order central moments')
    axs[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')