import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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
    fill_colors = {
        'u1': '#C6DBEF',     # light blue
        'u2': '#FDD0A2',     # light orange
        'u3': '#C7E9C0',     # light green
        'total': '#D9D9D9',  # light gray
    }
    line_colors = {
        'u1': '#2171B5',     # dark blue
        'u2': '#D94801',     # dark orange
        'u3': '#238B45',     # dark green
        'total': '#000000',  # black
        'extra1': '#CC79A7', # purple
        'extra2': '#E69F00', # orange
        'extra3': '#56B4E9', # sky blue
        'gray': '#888888'    # soft gray
    }

    fig, axs = plt.subplots(3, 2, figsize=(13, 11))
    axs = axs.flatten()

    # Compute std (shape: 3 x T)
    std_MC = np.sqrt(np.clip(np.diagonal(cov_MC, axis1=0, axis2=1).T, a_min=0, a_max=None))

    # --- Mean with ± std fill ---
    for i, key in enumerate(['u1', 'u2', 'u3']):
        axs[0].fill_between(TT_MC, mean_MC[i] - std_MC[i], mean_MC[i] + std_MC[i],
                            color=fill_colors[key], alpha=0.6)
        axs[0].plot(TT_MC, mean_MC[i], color=line_colors[key], lw=1.8, label=fr'$\langle u_{i+1} \rangle$')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mean')
    axs[0].legend(frameon=False, loc='upper left')

    # --- Variance ---
    for i, key in enumerate(['u1', 'u2', 'u3']):
        axs[1].plot(TT_MC, cov_MC[i, i], color=line_colors[key], lw=1.2, label=fr'$\mathrm{{Var}}(u_{i+1})$')
    axs[1].plot(TT_MC, np.sum(std_MC**2, axis=0),
                color=line_colors['total'], lw=1.5, linestyle='--', label='Total Var')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Variance')
    axs[1].legend(frameon=False, loc='upper left')

    # --- Cross-Covariance ---
    axs[2].plot(TT_MC, cov_MC[0, 1], color=line_colors['extra1'], lw=1.2, label=r'$\mathrm{Cov}(u_1, u_2)$')
    axs[2].plot(TT_MC, cov_MC[0, 2], color=line_colors['extra2'], lw=1.2, label=r'$\mathrm{Cov}(u_1, u_3)$')
    axs[2].plot(TT_MC, cov_MC[1, 2], color=line_colors['extra3'], lw=1.2, label=r'$\mathrm{Cov}(u_2, u_3)$')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Cross-Covariance')
    axs[2].legend(frameon=False, loc='upper left')

    # --- 3rd-Order Moments ---
    axs[3].plot(TT_MC, M3_MC[0, 1, 2], color=line_colors['total'], lw=2, label=r'$\langle M_{123} \rangle$')
    axs[3].plot(TT_MC, M3_MC[0, 1, 1], color=line_colors['u1'], lw=1.2, label=r'$\langle M_{122} \rangle$')
    axs[3].plot(TT_MC, M3_MC[0, 2, 2], color=line_colors['u2'], lw=1.2, label=r'$\langle M_{133} \rangle$')
    axs[3].plot(TT_MC, M3_MC[1, 1, 2], color=line_colors['u3'], lw=1.2, label=r'$\langle M_{223} \rangle$')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('3rd Moment')
    axs[3].legend(frameon=False, loc='upper left')

    # --- Energy from Truth ---
    axs[4].plot(TT_MC, Ene_MC[0], color=line_colors['total'], lw=2, label='Total')
    axs[4].plot(TT_MC, Ene_MC[1], color=line_colors['u1'], lw=1.2, label='Mode 1')
    axs[4].plot(TT_MC, Ene_MC[2], color=line_colors['u2'], lw=1.2, label='Mode 2')
    axs[4].plot(TT_MC, Ene_MC[3], color=line_colors['u3'], lw=1.2, label='Mode 3')
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Energy (Truth)')
    axs[4].legend(frameon=False, loc='upper left')

    # --- Energy from Dynamics ---
    axs[5].plot(TT_MC, Ene_MC[0], color=line_colors['total'], lw=2, label='Truth')
    axs[5].plot(TT_MC, Ene_dyn[0], color=line_colors['extra1'], lw=1.2, label='Full Eqn.')
    axs[5].plot(TT_MC, Ene_dyn[1], color=line_colors['extra2'], lw=1.2, label='Lower Bound')
    axs[5].plot(TT_MC, Ene_dyn[2], color=line_colors['extra3'], lw=1.2, label='Upper Bound')
    axs[5].plot(TT_MC, Ene_dyn[3], color=line_colors['gray'], lw=1.2, linestyle='--', label='Mean')
    axs[5].set_xlabel('Time')
    axs[5].set_ylabel('Energy (Dynamics)')
    axs[5].legend(frameon=False, loc='upper left')

    # Final layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()