import numpy as np
import logging
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def check_allowed_terms_periodic_cascade(expression, dimension):
    """
    Check if expression contains only allowed terms for periodic_cascade case.
    This version allows sin, cos, exp functions for time-dependent forcing.
                    
    Args:
        expression (str): The expression to check
        dimension (int): Dimension (1, 2, or 3)
                    
    Returns:
        dict: Dictionary with 'valid' (bool) and 'terms_present' (list of terms found)
    """
    # Define allowed terms for each dimension (same as regular case)
    allowed_terms = {
    1: ['x1', 'x2', 'x3', 'x2*x3', 'x2x3'],  # x1, x2, x3, x2x3, and constants
    2: ['x1', 'x2', 'x3', 'x1*x3', 'x1x3'],  # x1, x2, x3, x1x3, and constants
    3: ['x1', 'x2', 'x3', 'x1*x2', 'x1x2']   # x1, x2, x3, x1x2, and constants
    }
                    
    # Convert expression to lowercase for easier checking
    expr_lower = expression.lower()
                    
    # Check for disallowed terms (terms that should NOT be present)
    # For periodic_cascade, we allow sin, cos, exp but only for time-dependent forcing
    # We want to exclude: sin(x1), sin(x2), sin(x3), cos(x1), cos(x2), cos(x3), exp(x1), exp(x2), exp(x3)
    # And high powers: x1**2, x1**3, x1**4, x2**2, x2**3, x2**4, x3**2, x3**3, x3**4
    disallowed_terms = {
        1: ['x1*x2', 'x1x2', 'x1*x3', 'x1x3',  # Wrong interaction terms
            'sin(x1)', 'sin(x2)', 'sin(x3)', 'cos(x1)', 'cos(x2)', 'cos(x3)', 'exp(x1)', 'exp(x2)', 'exp(x3)',  # Trig/exp of variables
            'x1**2', 'x1**3', 'x1**4', 'x1**5', 'x1**6', 'x1**7', 'x1**8',  # High powers of x1
            'x2**2', 'x2**3', 'x2**4', 'x2**5', 'x2**6', 'x2**7', 'x2**8',  # High powers of x2  
            'x3**2', 'x3**3', 'x3**4', 'x3**5', 'x3**6', 'x3**7', 'x3**8'], # High powers of x3
        2: ['x1*x2', 'x1x2', 'x2*x3', 'x2x3',  # Wrong interaction terms
            'sin(x1)', 'sin(x2)', 'sin(x3)', 'cos(x1)', 'cos(x2)', 'cos(x3)', 'exp(x1)', 'exp(x2)', 'exp(x3)',  # Trig/exp of variables
            'x1**2', 'x1**3', 'x1**4', 'x1**5', 'x1**6', 'x1**7', 'x1**8',  # High powers of x1
            'x2**2', 'x2**3', 'x2**4', 'x2**5', 'x2**6', 'x2**7', 'x2**8',  # High powers of x2
            'x3**2', 'x3**3', 'x3**4', 'x3**5', 'x3**6', 'x3**7', 'x3**8'], # High powers of x3
        3: ['x1*x3', 'x1x3', 'x2*x3', 'x2x3',  # Wrong interaction terms
            'sin(x1)', 'sin(x2)', 'sin(x3)', 'cos(x1)', 'cos(x2)', 'cos(x3)', 'exp(x1)', 'exp(x2)', 'exp(x3)',  # Trig/exp of variables
            'x1**2', 'x1**3', 'x1**4', 'x1**5', 'x1**6', 'x1**7', 'x1**8',  # High powers of x1
            'x2**2', 'x2**3', 'x2**4', 'x2**5', 'x2**6', 'x2**7', 'x2**8',  # High powers of x2
            'x3**2', 'x3**3', 'x3**4', 'x3**5', 'x3**6', 'x3**7', 'x3**8']  # High powers of x3
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
    
    # For periodic_cascade, check for proper time-dependent forcing
    # Allow: sin(t), cos(t), exp(t) or sin(something*t), cos(something*t), exp(something*t)
    # But NOT: t**2, t**3, t**4, t**5, etc.
    has_sin_t = 'sin(' in expr_lower and 't' in expr_lower
    has_cos_t = 'cos(' in expr_lower and 't' in expr_lower  
    has_exp_t = 'exp(' in expr_lower and 't' in expr_lower
    has_high_power_t = any(f't**{i}' in expr_lower for i in range(2, 9))  # t**2, t**3, ..., t**8
    
    # Valid time-dependent forcing: sin(t), cos(t), or exp(t) but no high powers of t
    has_time_var = (has_sin_t or has_cos_t or has_exp_t) and not has_high_power_t
                    
    return {'valid': has_allowed_var and has_time_var, 'terms_present': terms_present}


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


class ResidualVAE(nn.Module):
    """
    Simple VAE for residual increments.

    Encoder: q(z | zt)
    Decoder: p(y | z) where y is the residual increment (in normalized ODE space).
    """

    def __init__(self, zt_dim: int = 3, y_dim: int = 3, latent_dim: int = 8, hid_dim: int = 64):
        super().__init__()
        self.zt_dim = zt_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(zt_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
        )
        self.enc_mu = nn.Linear(hid_dim, latent_dim)
        self.enc_logvar = nn.Linear(hid_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, y_dim),
        )

    def encode(self, zt: torch.Tensor):
        h = self.encoder(zt)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, zt: torch.Tensor):
        mu, logvar = self.encode(zt)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        return y_hat, mu, logvar

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
                     model_name:str = "equipart",
                     noise_levels: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,2.0],
                     DEVICE: str = "cpu")->dict:
    """
    Extract coefficients from FINAL_EXPR.txt file.
    
    Args:
        load_dir (str): Directory to load the coefficients
        noise_levels (list): List of noise levels to process
        DEVICE (str): Device type for path construction
    """
    import re
    
    print(f"DEBUG: get_coefficients called with load_dir='{load_dir}', DEVICE='{DEVICE}'")
    
    # Define path to FINAL_EXPR.txt (preferred) or FINAL_EXPR_1000.txt (fallback, e.g. equipart)
    if str(DEVICE) == "cuda:0":
        base_dir = os.path.join(load_dir, "Results", "Results1", "Results", model_name)
    else:
        base_dir = os.path.join(load_dir, "Results", model_name)

    file_path = os.path.join(base_dir, "FINAL_EXPR.txt")
    alt_file_path = os.path.join(base_dir, "FINAL_EXPR_1000.txt")
    
    print(f"DEBUG: Looking for file at: {file_path}")
    print(f"DEBUG: File exists: {os.path.exists(file_path)}")
    
    # Dictionary to store coefficients for each noise level
    coefficients_data = {"dim_1":{"x1":[], "x2":[], "x3":[], "x2x3":[]}, 
        "dim_2":{"x1":[], "x2":[], "x3":[], "x1x3":[]}, 
        "dim_3":{"x1":[], "x2":[], "x3":[], "x1x2":[]}}
    
    if not os.path.exists(file_path):
        # Try the alternative FINAL_EXPR_1000.txt if the default is missing
        if os.path.exists(alt_file_path):
            print(f"Warning: FINAL_EXPR.txt not found at {file_path}, using FINAL_EXPR_1000.txt instead.")
            file_path = alt_file_path
        else:
            print(f"Error: Neither FINAL_EXPR.txt nor FINAL_EXPR_1000.txt found in {base_dir}")
            return coefficients_data
    
    print(f"Loading coefficients from: {file_path}")
    
    # Read the FINAL_EXPR file
    with open(file_path, 'r') as f:
        content = f.read()
    print(noise_levels)
    # Process each noise level
    for noise_level in noise_levels:
        # Find the section for this noise level
        if noise_level == 0.0:
            noise_pattern = rf'(?:NOISE|Noise)\s*{noise_level}(?:\s*ODE|\s*total noise)?\s*(?:#=+|=+)\s*(.*?)(?=\n(?:NOISE|Noise)|\Z)'
        elif noise_level == 1.0:
            noise_pattern = rf'(?:NOISE|Noise)\s*{noise_level}(?:\s*ODE|\s*total noise)?\s*(?:#=+|=+)\s*(.*?)(?=\n(?:NOISE|Noise)|\Z)'
        elif noise_level == 1.8:
            # Special case for the last section (1.8); include optional "ODE" text
            noise_pattern = rf'(?:NOISE|Noise)\s*{noise_level}(?:\s*ODE|\s*total noise)?\s*=+\s*(.*?)(?=\Z)'
        else:
            # Handle both formats: with #= and with ==, optionally followed by "ODE" or "total noise".
            # Accept both "NOISE" and "Noise" as in FINAL_EXPR_1000.txt.
            noise_pattern = rf'(?:NOISE|Noise)\s*{noise_level}(?:\s*ODE|\s*total noise)?\s*(?:#=+|=+)\s*(.*?)(?=\n(?:NOISE|Noise)|\Z)'
        
        noise_match = re.search(noise_pattern, content, re.DOTALL)
        if noise_match:
            noise_section = noise_match.group(1).strip()
            print(f"\nProcessing noise level {noise_level}:")
            
            # Extract expressions for each dimension
            for dim in range(1, 4):
                dim_pattern = rf'dimension_{dim}:\s*(.*?)(?=\ndimension_|\Z)'
                dim_match = re.search(dim_pattern, noise_section, re.DOTALL)
                
                if dim_match:
                    expression = dim_match.group(1).strip()
                    print(f"  Dimension {dim} expression: {expression}")
                    
                    # Extract coefficients using the existing helper function
                    coeffs = extract_coefficients_from_expr(expression, dim)
                    
                    # Store coefficients
                    if coeffs['x1'] is not None:
                        coefficients_data[f"dim_{dim}"]["x1"].append(coeffs['x1'])
                        print(f"    x1 coefficient: {coeffs['x1']:.7f}")
                    
                    if coeffs['x2'] is not None:
                        coefficients_data[f"dim_{dim}"]["x2"].append(coeffs['x2'])
                        print(f"    x2 coefficient: {coeffs['x2']:.7f}")
                    
                    if coeffs['x3'] is not None:
                        coefficients_data[f"dim_{dim}"]["x3"].append(coeffs['x3'])
                        print(f"    x3 coefficient: {coeffs['x3']:.7f}")
                    
                    # Cross-term coefficients
                    if dim == 1 and coeffs['x2x3'] is not None:
                        coefficients_data[f"dim_{dim}"]["x2x3"].append(coeffs['x2x3'])
                        print(f"    x2*x3 coefficient: {coeffs['x2x3']:.7f}")
                    elif dim == 2 and coeffs['x1x3'] is not None:
                        coefficients_data[f"dim_{dim}"]["x1x3"].append(coeffs['x1x3'])
                        print(f"    x1*x3 coefficient: {coeffs['x1x3']:.7f}")
                    elif dim == 3 and coeffs['x1x2'] is not None:
                        coefficients_data[f"dim_{dim}"]["x1x2"].append(coeffs['x1x2'])
                        print(f"    x1*x2 coefficient: {coeffs['x1x2']:.7f}")
                else:
                    print(f"  Warning: Could not find expression for dimension {dim}")
        else:
            print(f"Warning: Could not find section for noise level {noise_level}")
    
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


def get_coefficients_from_finak_sample_file(file_path: str, verbose: bool = False) -> tuple:
    """
    Parse a FINAK_EXPR_SAMPLE.txt (or FINAL_EXPR_SAMPLE) file and extract
    coefficients for each sample-size block. Same coefficient structure as
    get_coefficients, but one entry per sample size (1000, 2000, ...).

    Args:
        file_path: Path to FINAK_EXPR_SAMPLE.txt.
        verbose: If True, print per-block info.

    Returns:
        (coefficients_data, sample_sizes): coefficients_data has keys dim_1, dim_2, dim_3
        with x1, x2, x3 and cross-term (x2x3, x1x3, x1x2) lists; sample_sizes is e.g. [1000, 2000, ...].
    """
    import re

    coefficients_data = {
        "dim_1": {"x1": [], "x2": [], "x3": [], "x2x3": []},
        "dim_2": {"x1": [], "x2": [], "x3": [], "x1x3": []},
        "dim_3": {"x1": [], "x2": [], "x3": [], "x1x2": []},
    }
    sample_sizes = []

    if not os.path.exists(file_path):
        if verbose:
            print(f"File not found: {file_path}")
        return coefficients_data, sample_sizes

    with open(file_path, "r") as f:
        content = f.read()

    # Split by "SAMPLE <number>" blocks; content has optional "SAMPLE" header then "SAMPLE 1000", "SAMPLE 2000", ...
    block_pattern = re.compile(r"SAMPLE\s+(\d+)\s*\n(?:[#=]+\s*\n)?(.*?)(?=SAMPLE\s+\d+\s*\n|\Z)", re.DOTALL)
    for match in block_pattern.finditer(content):
        sample_num = int(match.group(1))
        block = match.group(2).strip()
        sample_sizes.append(sample_num)

        for dim in range(1, 4):
            # Match "dimension_N:" or "ddimension_N:" (typo) and expression to next dimension or end
            dim_pattern = re.compile(
                r"d{1,2}imension_" + str(dim) + r":\s*(.*?)(?=\nd{1,2}imension_|\Z)", re.DOTALL
            )
            dim_match = dim_pattern.search(block)
            if dim_match:
                expression = dim_match.group(1).strip()
                coeffs = extract_coefficients_from_expr(expression, dim)
                if coeffs.get("x1") is not None:
                    coefficients_data[f"dim_{dim}"]["x1"].append(coeffs["x1"])
                if coeffs.get("x2") is not None:
                    coefficients_data[f"dim_{dim}"]["x2"].append(coeffs["x2"])
                if coeffs.get("x3") is not None:
                    coefficients_data[f"dim_{dim}"]["x3"].append(coeffs["x3"])
                if dim == 1 and coeffs.get("x2x3") is not None:
                    coefficients_data["dim_1"]["x2x3"].append(coeffs["x2x3"])
                elif dim == 2 and coeffs.get("x1x3") is not None:
                    coefficients_data["dim_2"]["x1x3"].append(coeffs["x1x3"])
                elif dim == 3 and coeffs.get("x1x2") is not None:
                    coefficients_data["dim_3"]["x1x2"].append(coeffs["x1x2"])
                if verbose:
                    print(f"  SAMPLE {sample_num} dim_{dim}: cross-term extracted")

    return coefficients_data, sample_sizes


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




def simplify_expression_for_periodic_cascade(expr: str, dim: int) -> str:
    """
    Simplifies expressions for periodic cascade to show only essential terms:
    x1, x2, x3, x1*x2, x2*x3, x1*x3, and sin(2π/8 * t)
    """
    import re
    import sympy as sp
    
    # Parse the expression with sympy
    try:
        expr_sympy = sp.sympify(expr)
    except:
        return expr  # Return original if parsing fails
    
    # Define symbols
    x1, x2, x3, t = sp.symbols('x1 x2 x3 t')
    
    # Collect coefficients for each term type
    coeffs = {}
    
    # Linear terms
    coeffs['x1'] = expr_sympy.coeff(x1)
    coeffs['x2'] = expr_sympy.coeff(x2) 
    coeffs['x3'] = expr_sympy.coeff(x3)
    
    # Cross terms
    coeffs['x1x2'] = expr_sympy.coeff(x1*x2)
    coeffs['x2x3'] = expr_sympy.coeff(x2*x3)
    coeffs['x1x3'] = expr_sympy.coeff(x1*x3)
    
    # Time-dependent terms - look for sin patterns
    sin_pattern = r'sin\([^)]*t[^)]*\)'
    sin_matches = re.findall(sin_pattern, expr)
    if sin_matches:
        # Extract the sin term and try to simplify it
        sin_term = sin_matches[0]
        # Try to extract frequency from sin(freq*t + phase)
        freq_match = re.search(r'sin\(([^)]*t[^)]*)\)', sin_term)
        if freq_match:
            sin_expr = freq_match.group(1)
            # Try to simplify the frequency to 2π/8 ≈ 0.785
            try:
                sin_expr_sympy = sp.sympify(sin_expr)
                # Check if it's close to 2π/8
                if abs(float(sin_expr_sympy) - 2*3.14159/8) < 0.1:
                    coeffs['sin_term'] = 'sin(2*pi/8*t)'
                else:
                    coeffs['sin_term'] = f'sin({sin_expr_sympy})'
            except:
                coeffs['sin_term'] = sin_term
        else:
            coeffs['sin_term'] = sin_term
    else:
        coeffs['sin_term'] = '0'
    
    # Constant term
    constant_expr = expr_sympy.subs([(x1, 0), (x2, 0), (x3, 0), (t, 0)])
    coeffs['constant'] = float(constant_expr) if constant_expr.is_number else 0
    
    # Build simplified expression
    terms = []
    
    # Add linear terms
    if abs(coeffs['x1']) > 1e-6:
        terms.append(f"{coeffs['x1']:.1f}*x1")
    if abs(coeffs['x2']) > 1e-6:
        terms.append(f"{coeffs['x2']:.1f}*x2")
    if abs(coeffs['x3']) > 1e-6:
        terms.append(f"{coeffs['x3']:.1f}*x3")
    
    # Add cross terms
    if abs(coeffs['x1x2']) > 1e-6:
        terms.append(f"{coeffs['x1x2']:.1f}*x1*x2")
    if abs(coeffs['x2x3']) > 1e-6:
        terms.append(f"{coeffs['x2x3']:.1f}*x2*x3")
    if abs(coeffs['x1x3']) > 1e-6:
        terms.append(f"{coeffs['x1x3']:.1f}*x1*x3")
    
    # Add time-dependent term
    if coeffs['sin_term'] != '0':
        terms.append(coeffs['sin_term'])
    
    # Add constant term
    if abs(coeffs['constant']) > 1e-6:
        terms.append(f"{coeffs['constant']:.1f}")
    
    # Join terms with ' + '
    simplified = ' + '.join(terms)
    
    # Handle negative signs properly
    simplified = simplified.replace(' + -', ' - ')
    
    return simplified



