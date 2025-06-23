import numpy as np
import logging
import math
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import numba
from numba import jit, prange

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

@jit(nopython=True, parallel=True)
def compute_distances_parallel(x0_train_chunk, x_sample):
    """Compute distances between training chunk and sample points using Numba for speed."""
    n_train = x0_train_chunk.shape[0]
    n_sample = x_sample.shape[0]
    distances = np.zeros((n_train, n_sample))
    
    for i in prange(n_train):
        for j in range(n_sample):
            diff = x0_train_chunk[i] - x_sample[j]
            distances[i, j] = np.sum(diff * diff)
    
    return distances

@jit(nopython=True)
def find_nearest_neighbors(distances, short_size):
    """Find nearest neighbors using Numba for speed."""
    n_train = distances.shape[0]
    indices = np.zeros((n_train, short_size), dtype=np.int32)
    
    for i in range(n_train):
        # Get indices of smallest distances
        sorted_indices = np.argsort(distances[i])
        indices[i] = sorted_indices[:short_size]
    
    return indices

def process_chunk_cpu(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim):
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    
    # Use KDTree for efficient nearest neighbor search
    tree = KDTree(x_sample, leaf_size=40)  # Optimized leaf size for better performance
    
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Perform the search using KDTree
        distances, index_initial = tree.query(x0_train_chunk, k=short_size)
        x0_train_index_initial[start_idx:end_idx, :] = index_initial

        if jj % 500 == 0:
            print('find index iteration:', jj, it_size_x0train)
    
    return x0_train_index_initial

def process_chunk(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim):
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    
    # For GPU-like performance, use optimized CPU implementation with Numba
    # First, compute all distances efficiently
    print("Computing distances...")
    distances = cdist(x0_train, x_sample, metric='sqeuclidean')  # Use squared Euclidean for speed
    
    print("Finding nearest neighbors...")
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        
        # Get distances for this chunk
        chunk_distances = distances[start_idx:end_idx]
        
        # Find nearest neighbors
        index_initial = np.argsort(chunk_distances, axis=1)[:, :short_size]
        x0_train_index_initial[start_idx:end_idx, :] = index_initial

        if jj % 500 == 0:
            print('find index iteration:', jj, it_size_x0train)
    
    return x0_train_index_initial

def process_chunk_optimized(it_n_index, it_size_x0train, short_size, x_sample, x0_train, train_size, x_dim):
    """Optimized version using Numba for maximum speed."""
    x0_train_index_initial = np.empty((train_size, short_size), dtype=int)
    
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Use Numba-optimized distance computation
        distances = compute_distances_parallel(x0_train_chunk, x_sample)
        index_initial = find_nearest_neighbors(distances, short_size)
        x0_train_index_initial[start_idx:end_idx, :] = index_initial

        if jj % 500 == 0:
            print('find index iteration:', jj, it_size_x0train)
    
    return x0_train_index_initial

def process_chunk_cpu_chunked(x_sample, x0_train, start_idx, end_idx, short_size, x_dim, batch_size=5000):
    """
    CPU version of chunked processing with batching for large datasets.
    Similar to the GPU version but optimized for CPU.
    """
    x0_train_index_initial = np.empty((end_idx - start_idx, short_size), dtype=int)
    
    # Use KDTree for efficient nearest neighbor search
    # Build the tree on the entire x_sample for better performance
    print(f"Building KDTree for {x_sample.shape[0]} sample points...")
    tree = KDTree(x_sample, leaf_size=40)
    
    print(f"Indexing complete, now performing searches for data from {start_idx} to {end_idx}.")

    # Perform the search for the current chunk in smaller batches
    for k in range(start_idx, end_idx, batch_size):
        end_k = min(k + batch_size, end_idx)
        search_chunk = x0_train[k:end_k]
        
        # Perform the search using KDTree
        distances, index_initial = tree.query(search_chunk, k=short_size)
        
        # Store results
        local_start = k - start_idx
        local_end = local_start + search_chunk.shape[0]
        x0_train_index_initial[local_start:local_end, :] = index_initial
        
        if (k - start_idx) % (batch_size * 10) == 0:
            print(f"Processed {k - start_idx} out of {end_idx - start_idx} points")

    return x0_train_index_initial




