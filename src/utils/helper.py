import numpy as np
import logging
import math
import matplotlib.pyplot as plt
def Buu(B,u,v):
    '''Compute the Buu operator terms for the triad model.'''
    if len(u.shape) == 1:
        vB = np.zeros(3)
        vB[0] = B[0] * u[1] * v[2]  # corresponds to u(2) * v(3)
        vB[1] = B[1] * u[2] * v[0]  # corresponds to u(3) * v(1)
        vB[2] = B[2] * u[0] * v[1]  # corresponds to u(1) * v(2)
    
    else:
        vB = np.zeros_like(u)
        vB[:,0] = B[0]*u[:,1]*v[:,2]
        vB[:,1] = B[1]*u[:,2]*v[:,0]
        vB[:,2] = B[2]*u[:,0]*v[:,1]   
    return vB


def compute_third_order_moments(u):
    mean_MC = np.mean(u, axis=0)
    cov_MC = np.cov(u, rowvar=False)
    moment3_MC = np.zeros((3, 3, 3))
    moment3_MC_norm = np.zeros((3, 3, 3))
    for k1 in range(3):
        for k2 in range(3):
            for k3 in range(3):
                centered = (
                    (u[:, k1] - mean_MC[k1]) *
                    (u[:, k2] - mean_MC[k2]) *
                    (u[:, k3] - mean_MC[k3])
                )
                moment3_MC[k1, k2, k3] = np.mean(centered)
                denom = np.sqrt(
                    cov_MC[k1, k1] * cov_MC[k2, k2] * cov_MC[k3, k3])
                moment3_MC_norm[k1, k2, k3] = moment3_MC[k1, k2, k3] / denom if denom != 0 else 0.0
    return moment3_MC, moment3_MC_norm


def double_check_energy(mean_MC, cov_MC):
    # Correct total energy computation
    Energy_MC = 0.5 * np.sum(mean_MC**2) + 0.5 * np.trace(cov_MC)
    print("Energy_MC:", Energy_MC)
    # Energy for each individual dimension
    Energy_MC_dimension1 = 0.5 * (mean_MC[0]**2 + cov_MC[0, 0])
    Energy_MC_dimension2 = 0.5 * (mean_MC[1]**2 + cov_MC[1, 1])
    Energy_MC_dimension3 = 0.5 * (mean_MC[2]**2 + cov_MC[2, 2])
    Energy_sum = Energy_MC_dimension1 + Energy_MC_dimension2 + Energy_MC_dimension3
    print("Are summation of each dimension equal to the total energy?", Energy_sum == Energy_MC)


def logprint(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logging.info(message)

def adjust_learning_rate(optimizer, epoch, start_lr, num_iter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * 0.5* (math.cos(math.pi*epoch /num_iter)+1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_latex_formula(params,Formula_1:list, Formula_2:list, Formula_3:list):
    # Configure LaTeX rendering
    plt.rcParams.update({
         "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
    })

    # Extract parameters
    L = params['L']
    G = params['G']
    B = params['B']

    # Extract individual values from matrices
    L1 = L[1, 2]  # u_2 -> u_3
    L2 = L[2, 0]  # u_3 -> u_1
    L3 = L[0, 1]  # u_1 -> u_2
    d1 = G[0, 0]
    d2 = G[1, 1]
    d3 = G[2, 2]
    B1, B2, B3 = B[0], B[1], B[2]

    # Define LaTeX equations with formatted coefficients
    ground_truth = (
        r"\textbf{Ground Truth:}\\[0.3em]"
        r"\begin{aligned}"
        fr"\frac{{du_1}}{{dt}} &= {L2}u_3 - {L3}u_2 - {d1:.1f}u_1 + {B1:.1f}u_2 u_3 + F_1 + \text{{noise}} \\"
        fr"\frac{{du_2}}{{dt}} &= {L3}u_1 - {L1}u_3 - {d2:.1f}u_2 + {B2:.1f}u_3 u_1 + F_2 + \text{{noise}} \\"
        fr"\frac{{du_3}}{{dt}} &= {L1}u_2 - {L2}u_1 - {d3:.1f}u_3 + {B3:.1f}u_1 u_2 + F_3 + \text{{noise}}"
        r"\end{aligned}"
    )

    FEX_expression = (
        r"\textbf{FEX:}\\[0.3em]"
        r"\begin{aligned}"
        fr"F_1 &= {Formula_1[0]}u_3 - {Formula_1[1]}u_2 - {Formula_1[2]:.1f}u_1 + {Formula_1[3]:.1f}u_2 u_3 + residual term\\"
        fr"F_2 &= {Formula_2[0]}u_1 - {Formula_2[1]}u_3 - {Formula_2[2]:.1f}u_2 + {Formula_2[3]:.1f}u_3 u_1 + residual term\\"
        fr"F_3 &= {Formula_3[0]}u_2 - {Formula_3[1]}u_1 - {Formula_3[2]:.1f}u_3 + {Formula_3[3]:.1f}u_1 u_2+ residual term"
        r"\end{aligned}"
    )

    # Combine all lines into a LaTeX aligned environment
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].text(0.05, 0.5, f"${ground_truth}$", fontsize=14, va='center', ha='left')
    axs[0].axis('off')

    axs[1].text(0.05, 0.5, f"${FEX_expression}$", fontsize=14, va='center', ha='left')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()