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
        dict: Dictionary with 'valid' (bool) and 'terms_present' (list of terms found)
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
            return {'valid': False, 'terms_present': []}
                    
    # Check which allowed terms are present
    terms_present = [term for term in allowed_terms[dimension] if term in expr_lower]
    
    # Check if at least one allowed term is present (excluding constants)
    allowed_vars = ['x1', 'x2', 'x3']
    has_allowed_var = any(var in expr_lower for var in allowed_vars)
                    
    return {'valid': has_allowed_var, 'terms_present': terms_present}


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
    if DEVICE == "cuda:0":
        base_path = os.path.join(load_dir, "Results", "Results", "equipart")
    else:
        base_path = os.path.join(load_dir, "Results", "equipart")
    
    # Dictionary to store coefficients for each noise level
    coefficients_data = {"dim_1":{"x1":[], "x2":[], "x3":[], "x2x3":[]}, 
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
                            float_pattern = r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?|\d+)'
                            # --- x1 ---
                            x1_regex = r'(?<!\*x2)(?<!\*x3)' + float_pattern + r'\s*\*\s*x1(?!\*)'
                            x1_matches = list(re.finditer(x1_regex, expression))
                            print(f"    [DEBUG] All x1 matches: {[m.group(0) for m in x1_matches]}")
                            last_coeff = None
                            for m in x1_matches:
                                last_coeff = float(m.group(1))
                            if last_coeff is not None:
                                coefficients_data[f"dim_{dim}"]["x1"].append(last_coeff)
                                print(f"    x1 coefficient: {last_coeff:.7f}")
                            else:
                                print(f"    x1 coefficient: NOT FOUND")
                            
                            # --- x2 ---
                            x2_regex = r'(?<!\*x1)(?<!\*x3)' + float_pattern + r'\s*\*\s*x2(?!\*)'
                            x2_matches = list(re.finditer(x2_regex, expression))
                            print(f"    [DEBUG] All x2 matches: {[m.group(0) for m in x2_matches]}")
                            last_coeff = None
                            for m in x2_matches:
                                last_coeff = float(m.group(1))
                            if last_coeff is not None:
                                coefficients_data[f"dim_{dim}"]["x2"].append(last_coeff)
                                print(f"    x2 coefficient: {last_coeff:.7f}")
                            else:
                                print(f"    x2 coefficient: NOT FOUND")
                            
                            # --- x3 ---
                            x3_regex = r'(?<!\*x1)(?<!\*x2)' + float_pattern + r'\s*\*\s*x3(?!\*)'
                            x3_matches = list(re.finditer(x3_regex, expression))
                            print(f"    [DEBUG] All x3 matches: {[m.group(0) for m in x3_matches]}")
                            last_coeff = None
                            for m in x3_matches:
                                last_coeff = float(m.group(1))
                            if last_coeff is not None:
                                coefficients_data[f"dim_{dim}"]["x3"].append(last_coeff)
                                print(f"    x3 coefficient: {last_coeff:.7f}")
                            else:
                                print(f"    x3 coefficient: NOT FOUND")
                            
                            # --- x2*x3 ---
                            x2x3_match = re.search(float_pattern + r'\s*\*\s*x2\s*\*\s*x3', expression)
                            if x2x3_match and dim == 1:
                                coeff = float(x2x3_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x2x3"].append(coeff)
                                print(f"    x2*x3 coefficient: {coeff:.7f}")
                            elif dim == 1:
                                print(f"    x2*x3 coefficient: NOT FOUND")
                            
                            # --- x1*x3 ---
                            x1x3_match = re.search(float_pattern + r'\s*\*\s*x1\s*\*\s*x3', expression)
                            if x1x3_match and dim == 2:
                                coeff = float(x1x3_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x1x3"].append(coeff)
                                print(f"    x1*x3 coefficient: {coeff:.7f}")
                            elif dim == 2:
                                print(f"    x1*x3 coefficient: NOT FOUND")
                            
                            # --- x1*x2 ---
                            x1x2_match = re.search(float_pattern + r'\s*\*\s*x1\s*\*\s*x2', expression)
                            if x1x2_match and dim == 3:
                                coeff = float(x1x2_match.group(1))
                                coefficients_data[f"dim_{dim}"]["x1x2"].append(coeff)
                                print(f"    x1*x2 coefficient: {coeff:.7f}")
                            elif dim == 3:
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

    # Manually flip sign for dim_2['x2'] and dim_3['x3']
    coefficients_data['dim_2']['x2'] = [-abs(v) for v in coefficients_data['dim_2']['x2']]
    coefficients_data['dim_3']['x3'] = [-abs(v) for v in coefficients_data['dim_3']['x3']]

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


def get_sequence_from_candidate(file_path: str, candidate_num: int) -> list:
    """
    Extract the sequence (Seq=[...]) from a specific candidate in the given file.
    Args:
        file_path (str): Path to the file containing candidate information
        candidate_num (int): Candidate number to extract (1-based)
    Returns:
        list: Sequence as a list of integers (or empty list if not found)
    """
    import re
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line for the specified candidate
    candidate_line = None
    for line in lines:
        if line.startswith(f'Candidate {candidate_num}:'):
            candidate_line = line
            break
    
    if candidate_line:
        seq_match = re.search(r'Seq=\[([^\]]+)\]', candidate_line)
        if seq_match:
            seq_str = seq_match.group(1)
            seq_list = [int(x.strip()) for x in seq_str.split(',')]
            return seq_list
        else:
            print(f"[WARN] No sequence found in candidate {candidate_num} for {file_path}")
            return []
    else:
        print(f"[WARN] Candidate {candidate_num} not found in {file_path}")
        return []

def select_operator_sequence(file_path: str, dim: int) -> list:
    """
    Display unique candidates and let user select one for the given dimension
    Args:
        file_path (str): Path to the candidate file
        dim (int): Dimension number for display purposes
    Returns:
        list: Selected sequence as a list of integers (or None if failed)
    """
    import re
    import sys
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse all candidates with their expressions
    candidates = []
    for line in lines:
        if line.startswith('Candidate'):
            # Extract candidate number, score, loss, sequence, and expression
            candidate_match = re.search(r'Candidate (\d+): Score=([^,]+), Loss=([^,]+), Seq=\[([^\]]+)\], Expr=(.+)', line)
            if candidate_match:
                candidate_num = int(candidate_match.group(1))
                score = candidate_match.group(2)
                loss = candidate_match.group(3)
                seq_str = candidate_match.group(4)
                expr = candidate_match.group(5).strip()
                seq_list = [int(x.strip()) for x in seq_str.split(',')]
                candidates.append({
                    'num': candidate_num,
                    'score': score,
                    'loss': loss,
                    'seq': seq_list,
                    'expr': expr
                })
    
    if not candidates:
        print(f"[ERROR] No candidates found in {file_path}")
        return None
    
    # Filter to unique sequences only
    unique_candidates = []
    seen_sequences = set()
    
    for candidate in candidates:
        seq_tuple = tuple(candidate['seq'])
        if seq_tuple not in seen_sequences:
            seen_sequences.add(seq_tuple)
            unique_candidates.append(candidate)
    
    # Display unique candidates
    print(f"\n" + "="*80)
    print(f"Unique candidates for Dimension {dim} (showing {len(unique_candidates)} out of {len(candidates)} total):")
    print("="*80)
    
    for i, candidate in enumerate(unique_candidates, 1):
        print(f"Option {i}: Score={candidate['score']}, Loss={candidate['loss']}")
        print(f"  Sequence: {candidate['seq']}")
        print(f"  Expression: {candidate['expr']}")
        print("-" * 80)
    
    # Get user selection
    while True:
        try:
            selection = input(f"Select option for Dimension {dim} (1-{len(unique_candidates)}): ").strip()
            selection_idx = int(selection)
            if 1 <= selection_idx <= len(unique_candidates):
                selected_candidate = unique_candidates[selection_idx - 1]
                print(f"Selected for Dimension {dim}: Option {selection_idx}")
                print(f"  Sequence: {selected_candidate['seq']}")
                print(f"  Expression: {selected_candidate['expr']}")
                return selected_candidate['seq']
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(unique_candidates)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def extract_coefficients_from_expr(expr: str, dim: int) -> dict:
    """
    Extracts the correct coefficients for the given dimension from the expression string.
    Returns a dict with keys for the relevant terms.
    """
    import re
    float_pattern = r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?|\d+)'
    result = {}

    # x1
    x1_regex = r'(?<!\*x2)(?<!\*x3)' + float_pattern + r'\s*\*\s*x1(?!\*)'
    x1_matches = list(re.finditer(x1_regex, expr))
    result['x1'] = float(x1_matches[-1].group(1)) if x1_matches else None

    # x2
    x2_regex = r'(?<!\*x1)(?<!\*x3)' + float_pattern + r'\s*\*\s*x2(?!\*)'
    x2_matches = list(re.finditer(x2_regex, expr))
    result['x2'] = float(x2_matches[-1].group(1)) if x2_matches else None

    # x3
    x3_regex = r'(?<!\*x1)(?<!\*x2)' + float_pattern + r'\s*\*\s*x3(?!\*)'
    x3_matches = list(re.finditer(x3_regex, expr))
    result['x3'] = float(x3_matches[-1].group(1)) if x3_matches else None

    # Cross-term
    if dim == 1:
        x2x3_regex = float_pattern + r'\s*\*\s*x2\s*\*\s*x3'
        x2x3_match = re.search(x2x3_regex, expr)
        result['x2x3'] = float(x2x3_match.group(1)) if x2x3_match else None
    elif dim == 2:
        x1x3_regex = float_pattern + r'\s*\*\s*x1\s*\*\s*x3'
        x1x3_match = re.search(x1x3_regex, expr)
        result['x1x3'] = float(x1x3_match.group(1)) if x1x3_match else None
    elif dim == 3:
        x1x2_regex = float_pattern + r'\s*\*\s*x1\s*\*\s*x2'
        x1x2_match = re.search(x1x2_regex, expr)
        result['x1x2'] = float(x1x2_match.group(1)) if x1x2_match else None

    return result



