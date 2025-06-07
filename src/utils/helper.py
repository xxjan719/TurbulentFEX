import numpy as np
import logging
import math
import matplotlib.pyplot as plt
import torch
import faiss
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


def weights_init(m):
    if isinstance(m, torch.nn.Linear):  # or whatever layers you use
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)






def process_chunk_cpu(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim):
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    # Create CPU index
    index = faiss.IndexFlatL2(x_dim)  # Create a FAISS index for exact searches
    index.add(x_sample)  # Add x_sample to the index
        
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

            # Perform the search on CPU
        _, index_initial = index.search(x0_train_chunk, short_size)
        x0_train_index_initial[start_idx:end_idx,:] = index_initial 

        if jj % 500 == 0:
            print('find index iteration:', jj, it_size_x0train)
        
        # Cleanup resources
    del index
    return x0_train_index_initial