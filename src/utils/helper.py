import numpy as np
import logging
import math
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import numba
from numba import jit, prange

import os

def check_allowed_terms(expression, dimension):
    """
    Check if expression contains only allowed terms for the given dimension.
                    
    Args:
        expression (str): The expression to check
        dimension (int): Dimension (1, 2, or 3)
                    
    Returns:
        bool: True if expression contains only allowed terms, False otherwise
    """
    # Define allowed terms for each dimension
    allowed_terms = {
    1: ['x1', 'x2', 'x3', 'x2*x3', 'x2x3'],  # x1, x2, x3, x2x3, and constants
    2: ['x1', 'x2', 'x3', 'x1*x3', 'x1x3'],  # x1, x2, x3, x1x3, and constants
    3: ['x1', 'x2', 'x3', 'x1*x2', 'x1x2']   # x1, x2, x3, x1x2, and constants
    }
                    
    # Convert expression to lowercase for easier checking
    expr_lower = expression.lower()
                    
    # Check for disallowed terms (terms that should NOT be present)
    disallowed_terms = {
        1: ['x1*x2', 'x1x2', 'x1*x3', 'x1x3', 'cos', 'sin','exp', '**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x1x3 are not allowed in dim 1
        2: ['x1*x2', 'x1x2', 'x2*x3', 'x2x3', 'cos', 'sin','exp','**2','**3','**4','**5','**6','**7','**8'],  # x1x2 and x2x3 are not allowed in dim 2
        3: ['x1*x3', 'x1x3', 'x2*x3', 'x2x3', 'cos','sin','exp','**2','**3','**4','**5','**6','**7','**8']   # x1x3 and x2x3 are not allowed in dim 3
        }
                    
    # Check if any disallowed terms are present
    for term in disallowed_terms[dimension]:
        if term in expr_lower:
            return False
                    
    # Check if at least one allowed term is present (excluding constants)
    allowed_vars = ['x1', 'x2', 'x3']
    has_allowed_var = any(var in expr_lower for var in allowed_vars)
                    
    return has_allowed_var


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


def get_coefficients(load_dir: str = "", 
                     noise_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     DEVICE: str = "cpu")->dict:
    """
    Plot coefficients of the FEX model.
    
    Args:
        load_dir (str): Directory to load the coefficients
        save_dir (str): Directory to save the plot
        noise_levels (list): List of noise levels to process
        DEVICE (str): Device type for path construction
    """
    import re
    
    # Define noise levels to load
    if DEVICE == "cuda":
        base_path = os.path.join(load_dir, "Results", "Results", "equipart")
    else:
        base_path = os.path.join(load_dir, "Results", "equipart")
    
    # Dictionary to store coefficients for each noise level
    coefficients_data = {"dim_1":{"x1":[], "x2":[], "x3":[], "x2x3":[], "x1x3":[]}, 
        "dim_2":{"x1":[], "x2":[], "x3":[], "x1x3":[]}, 
        "dim_3":{"x1":[], "x2":[], "x3":[], "x1x2":[]}}
    
    for noise_level in noise_levels:
        noise_dir = os.path.join(base_path, f"noise_{noise_level}")
        
        if os.path.exists(noise_dir):
            print(f"Loading coefficients from: {noise_dir}")
            
            for dim in range(1, 4):
                file_path = os.path.join(noise_dir, f"best_candidates_pool_summary_{dim}.txt")
                
                if os.path.exists(file_path):
                    # Read the text file
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Find candidate 5 (index 4 since we start from 0)
                    if len(lines) > 4:
                        candidate_5_line = lines[4]  # Candidate 5 is at index 4
                        
                        # Extract the expression part (after "Expr=")
                        expr_match = re.search(r'Expr=(.+)', candidate_5_line)
                        if expr_match:
                            expression = expr_match.group(1).strip()
                            print(f"  Dimension {dim} - Candidate 5 expression: {expression}")
                            
                            # Parse coefficients from the expression
                            # Look for patterns like: -0.2379*x1, 1.00881760666693*x2*x3, etc.
                            
                            # Parse x1 coefficient
                            x1_match = re.search(r'([+-]?\d+\.?\d*)\*x1', expression)
                            if x1_match:
                                coeff = float(x1_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x1"].append(coeff)
                                print(f"    x1 coefficient: {coeff}")
                            else:
                                print(f"    x1 coefficient: NOT FOUND")
                            
                            # Parse x2 coefficient
                            x2_match = re.search(r'([+-]?\d+\.?\d*)\*x2(?!\*x3)', expression)
                            if x2_match:
                                coeff = float(x2_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x2"].append(coeff)
                                print(f"    x2 coefficient: {coeff}")
                            else:
                                print(f"    x2 coefficient: NOT FOUND")
                            
                            # Parse x3 coefficient
                            x3_match = re.search(r'([+-]?\d+\.?\d*)\*x3(?!\*x[12])', expression)
                            if x3_match:
                                coeff = float(x3_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x3"].append(coeff)
                                print(f"    x3 coefficient: {coeff}")
                            else:
                                print(f"    x3 coefficient: NOT FOUND")
                            
                            
                            # Parse x1*x3 coefficient (for dim_1 and dim_2)
                            if dim == 1:
                                x2x3_match = re.search(r'([+-]?\d+\.?\d*)\*x2\*x3', expression)
                                if x2x3_match:
                                    coeff = float(x2x3_match.group(1))
                                    coefficients_data[f"dim_{dim}"]["x2x3"].append(coeff)
                                    print(f"    x2*x3 coefficient: {coeff}")
                                else:
                                    print(f"    x2*x3 coefficient: NOT FOUND")
                            elif dim == 2:
                                x1x3_match = re.search(r'([+-]?\d+\.?\d*)\*x1\*x3', expression)
                                if x1x3_match:
                                    coeff = float(x1x3_match.group(1))
                                    coefficients_data[f"dim_{dim}"]["x1x3"].append(coeff)
                                    print(f"    x1*x3 coefficient: {coeff}")
                                else:
                                    print(f"    x1*x3 coefficient: NOT FOUND")

                            # Parse x1*x2 coefficient (for dim_3)
                            if dim == 3:
                                x1x2_match = re.search(r'([+-]?\d+\.?\d*)\*x1\*x2', expression)
                                if x1x2_match:
                                    coeff = float(x1x2_match.group(1))
                                    coefficients_data[f"dim_{dim}"]["x1x2"].append(coeff)
                                    print(f"    x1*x2 coefficient: {coeff}")
                                else:
                                    print(f"    x1*x2 coefficient: NOT FOUND")
                        else:
                            print(f"  Warning: Could not find expression in candidate 5 for dimension {dim}")
                else:
                    print(f"  Warning: File not found: {file_path}")
        else:
            print(f"Noise directory not found: {noise_dir}")
    
    # Print summary of loaded coefficients
    print("\n"+"="*60)
    print("Loaded Coefficients Summary ")
    for dim_key, dim_data in coefficients_data.items():
        print(f"\n{dim_key}:")
        for term_key, coeff_list in dim_data.items():
            if coeff_list:
                print(f"  {term_key}: {len(coeff_list)} coefficients loaded")
            else:
                print(f"  {term_key}: No coefficients found")
    print("="*60)
    return coefficients_data


def get_score_expression_from_file(file_path: str) -> dict:
    """
    Get the score, loss, and expression from candidate 5 in the file.
    
    Args:
        file_path (str): Path to the file containing candidate information
        
    Returns:
        dict: Dictionary containing score, loss, and expression
    """
    import re
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) > 4:
        candidate_5_line = lines[4]  # Candidate 5 is at index 4
        
        # Extract score
        score_match = re.search(r'Score=([\d.]+)', candidate_5_line)
        score = float(score_match.group(1)) if score_match else None
        
        # Extract loss
        loss_match = re.search(r'Loss=([\d.]+)', candidate_5_line)
        loss = float(loss_match.group(1)) if loss_match else None
        
        # Extract expression
        expr_match = re.search(r'Expr=(.+)', candidate_5_line)
        expression = expr_match.group(1).strip() if expr_match else None
        
        # Print the information
        print("\n"+"="*60)
        print(f"[INFO] File is saved in: {file_path}")
        print(f"  Score: {score}")
        print(f"  Loss: {loss}")
        print(f"  Expression: {expression}")
        print("="*60)
        
        return {
            'score': score,
            'loss': loss,
            'expression': expression
        }
    else:
        print(f"Warning: File {file_path} has fewer than 5 lines")
        return {'score': None, 'loss': None, 'expression': None}


