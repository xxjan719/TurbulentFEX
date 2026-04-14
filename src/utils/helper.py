import numpy as np
import logging
import math
import warnings
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


def random_cascade_diffusion_forcing_for_eval(n_steps: int, Dt: float, seed: int = 42):
    """
    Mean forcing tmM and noise-modulation tmS for ``random_cascade`` (diffusive OU drivers),
    same recursion as MC_triad.params_init('random_cascade'), length n_steps for indices 0..n_steps-1.
    """
    fr = 2 * np.pi / 2
    theta = fr / (2 * np.pi)
    sigma = np.sqrt(2 * theta)
    rng = np.random.RandomState(seed)
    tmS = np.zeros(n_steps + 1, dtype=np.float64)
    tmM = np.zeros((n_steps + 1, 3), dtype=np.float64)
    tmt = 0.0
    for j in range(n_steps):
        dW1 = np.sqrt(Dt / 4) * rng.randn(4)
        winc = np.sum(dW1)
        tmS[j + 1] = tmS[j] - theta * tmS[j] * Dt + sigma * winc
        dW1 = np.sqrt(Dt / 4) * rng.randn(4)
        winc = np.sum(dW1)
        tmt = tmt - theta * tmt * Dt + sigma * winc
        tmM[j + 1, :] = tmt
    tmS_out = (0.8 * tmS[:n_steps]).astype(np.float32)
    tmM_out = tmM[:n_steps].astype(np.float32)
    return tmM_out, tmS_out


def random_cascade_deterministic_tmM_ou(Nt_eval: int, Dt: float, seed: int = 42) -> np.ndarray:
    """OU forcing path; same recursion as MC_triad.params_init('random_cascade_deterministic')."""
    theta = 5.0
    sigma = 0.2
    rng = np.random.RandomState(seed)
    tmM = np.zeros((Nt_eval, 3), dtype=np.float32)
    tmt = 1.5
    for j in range(Nt_eval):
        dW = np.sqrt(Dt) * rng.randn()
        tmt = tmt - theta * tmt * Dt + sigma * dW
        tmM[j, :] = tmt
    return tmM


def build_tmM_eval(Nt_eval: int, params: dict, params_name: str) -> np.ndarray:
    """
    Build (Nt_eval, 3) deterministic mean forcing for moment-closure rollouts.

    - periodic_cascade: sin(fr * k * dt) for k = 0..Nt_eval-1 (no tiling jump at T=10).
    - random_cascade_deterministic: full OU path with RandomState(42), not np.tile of the first 10s
      (tiling would repeat the same OU segment and misalign ground truth vs continuous-time FEX).
    - Otherwise: use params['tmM'] when length matches, else tile (e.g. dual_cascade).
    """
    Dt = float(params["Dt"])
    tmM = np.zeros((Nt_eval, 3), dtype=np.float32)
    raw = params.get("tmM")
    tmM_src = np.asarray(raw, dtype=np.float32) if raw is not None else None

    if params_name == "periodic_cascade" and "fr" in params:
        fr = float(params["fr"])
        k = np.arange(Nt_eval, dtype=np.float32)
        s = np.sin(fr * Dt * k).astype(np.float32)
        tmM[:, 0] = s
        tmM[:, 1] = s
        tmM[:, 2] = s
        return tmM

    if params_name == "random_cascade_deterministic":
        tmM_ou = random_cascade_deterministic_tmM_ou(Nt_eval, Dt, seed=42)
        if tmM_src is not None and tmM_src.shape[0] > 0:
            L = min(int(tmM_src.shape[0]), Nt_eval)
            if not np.allclose(tmM_ou[:L], tmM_src[:L], rtol=1e-4, atol=1e-3):
                warnings.warn(
                    "random_cascade_deterministic: params['tmM'] does not match OU replay (seed=42); "
                    "using the seed=42 OU path for the full evaluation horizon.",
                    stacklevel=2,
                )
        return tmM_ou

    if tmM_src is not None and tmM_src.shape[0] > 0:
        if tmM_src.shape[0] == Nt_eval:
            return tmM_src.copy()
        reps = int(np.ceil(Nt_eval / tmM_src.shape[0]))
        return np.tile(tmM_src, (reps, 1))[:Nt_eval].copy()
    return tmM


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


# Representative multi-indices for 4th–7th central moments (3D state), four traces per order
# (same spirit as the four panels in ``plot_third_order_moments_tfdm_vae_nn``).
MOMENT_4_INDICES = ((0, 1, 2, 0), (0, 1, 1, 1), (0, 2, 2, 2), (1, 1, 2, 2))
MOMENT_5_INDICES = ((0, 1, 2, 0, 1), (0, 1, 1, 1, 2), (0, 2, 2, 2, 0), (1, 1, 2, 2, 1))
MOMENT_6_INDICES = ((0, 1, 2, 0, 1, 2), (0, 1, 1, 1, 2, 2), (0, 2, 2, 2, 0, 1), (1, 1, 2, 2, 0, 0))
MOMENT_7_INDICES = (
    (0, 1, 2, 0, 1, 2, 0),
    (0, 1, 1, 1, 2, 2, 0),
    (0, 2, 2, 2, 0, 1, 2),
    (1, 1, 2, 2, 0, 0, 1),
)


def compute_nth_order_moment_time_series(
    u: np.ndarray,
    index_tuples: tuple,
) -> np.ndarray:
    """
    Central moments along a stored trajectory ``u`` of shape ``(N, 3, Nt)``.

    Returns
    -------
    ndarray, shape ``(len(index_tuples), Nt)``
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 3 or u.shape[1] != 3:
        raise ValueError("u must have shape (N, 3, Nt)")
    nt = u.shape[2]
    out = np.zeros((len(index_tuples), nt), dtype=np.float32)
    for t in range(nt):
        out[:, t] = compute_selected_nth_order_moments(u[:, :, t], index_tuples)
    return out


def compute_selected_nth_order_moments(
    u: np.ndarray,
    index_tuples: tuple,
) -> np.ndarray:
    """
    Central moments :math:`E[\\prod_k (u_{i_k} - \\mu_{i_k})]` for a batch ``u`` of shape (N, 3).

    Parameters
    ----------
    u : ndarray
        Sample paths at one time, shape (N, 3).
    index_tuples : sequence of tuple of int
        Each tuple has length ``order`` with entries in ``{0,1,2}``.

    Returns
    -------
    ndarray, shape (len(index_tuples),)
        Raw central moments (same convention as the leading output of ``compute_third_order_moments``).
    """
    u = np.asarray(u, dtype=np.float64)
    mean_MC = np.mean(u, axis=0)
    out = np.zeros(len(index_tuples), dtype=np.float64)
    for t, idx in enumerate(index_tuples):
        centered = np.ones(u.shape[0], dtype=np.float64)
        for d in range(len(idx)):
            centered *= u[:, idx[d]] - mean_MC[idx[d]]
        out[t] = np.mean(centered)
    return out.astype(np.float32)


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
    Standard VAE for residual increments R in R^{data_dim}.

    Encoder: q_phi(z | R). Decoder: p_theta(R_hat | z).
    Training: pass normalized residuals R; loss = recon + KL + optional second-moment term.
    Rollout / generation: sample z ~ N(0, I) in latent space and call ``decode(z)``.
    """

    def __init__(self, data_dim: int = 3, out_dim: int = 3, latent_dim: int = 8, hid_dim: int = 50):
        super().__init__()
        self.data_dim = data_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hid_dim),
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
            nn.Linear(hid_dim, out_dim),
        )

    def encode(self, r: torch.Tensor):
        h = self.encoder(r)
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

    def forward(self, r: torch.Tensor):
        """Encode residual r, sample latent, decode reconstruction."""
        mu, logvar = self.encode(r)
        z = self.reparameterize(mu, logvar)
        r_hat = self.decode(z)
        return r_hat, mu, logvar

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


def discussion_choice5_rollout(args, device, plot_composite=True):
    """Discussion choice 5: MC rollout (independent t≤20, dependent t≤10) and optional composite PDF."""
    import config
    from Example.MC_triad.MC_triad import params_init, MC_triad_initial_value
    from .FEX import FEX_model_learned
    from .FEX_with_force import FEX_with_force_model_learned
    from .ODEParser import FN_Net, simple_step_update, FN_multi_update

    print("=" * 60)
    print("[INFO] Choice 5 rollout: generate test samples (skip training) and optional composite plot")
    print("[INFO] Independent: t=0..20 (from independent run). Dependent: t=0..10 (dependent run only).")
    print("=" * 60)

    m0,var0 = MC_triad_initial_value()
    params = params_init(args.params_name)
    # Choose the correct model based on params_name
    if args.params_name in ['equipart', 'cascade', 'dual_cascade', 'random_cascade']:
        FEX_model_check = FEX_model_learned
    elif args.params_name in ['periodic_cascade', 'random_cascade_deterministic']:
        FEX_model_check = FEX_with_force_model_learned
    L = params['L']
    G = params['G']
    B = params['B']
    
    TIME_AMOUNT = 20
    dt = 0.01
    # Align MC path count with second-stage sample budget (default 10000 in config).
    NPATH = int(getattr(args, "RESIDUAL_SAMPLES", 10000))
    initial_state = np.random.normal(loc=m0, scale=np.sqrt(var0), size=(NPATH, 3))    
    x_pred_initial = torch.ones(NPATH, 3).to(device,dtype=torch.float32) * torch.tensor(m0).to(device,dtype=torch.float32)
    scaler = args.DIFF_SCALE
    
    Nt_eval = int(TIME_AMOUNT / dt)
    # Forcing / noise scaling must match `params_init()` (or seed-consistent replay for long horizons).
    if args.params_name == "random_cascade":
        tmM, tmS = random_cascade_diffusion_forcing_for_eval(
            Nt_eval, float(params["Dt"]), seed=42
        )
    else:
        tmM = build_tmM_eval(Nt_eval, params, args.params_name)
        tmS = np.zeros(Nt_eval, dtype=np.float32)
        if "tmS" in params and params["tmS"] is not None:
            tmS_src = np.asarray(params["tmS"], dtype=np.float32)
            if tmS_src.shape[0] >= Nt_eval:
                tmS = tmS_src[:Nt_eval]
            else:
                reps = int(np.ceil(Nt_eval / tmS_src.shape[0]))
                tmS = np.tile(tmS_src, reps)[:Nt_eval]
    mean_state_pred = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_record[:, 0] = np.mean(initial_state, axis=0)
    mean_state_pred[:, 0] = np.mean(initial_state, axis=0)

    # Add separate mean arrays for single and ensemble
    mean_state_single = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_single[:, 0] = np.mean(initial_state, axis=0)
    mean_state_ensemble = np.zeros((3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    mean_state_ensemble[:, 0] = np.mean(initial_state, axis=0)

    cov_state_pred = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_record[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_pred[:, :, 0] = np.cov(initial_state, rowvar=False)

    # Add separate covariance arrays for single and ensemble
    cov_state_single = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_single[:, :, 0] = np.cov(initial_state, rowvar=False)
    cov_state_ensemble = np.zeros((3, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    cov_state_ensemble[:, :, 0] = np.cov(initial_state, rowvar=False)

    u_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_all[:,:,0] = initial_state
    u_pred_all = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_all[:,:,0] = initial_state

    # Add separate arrays for single and ensemble predictions
    u_pred_single = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_single[:,:,0] = initial_state
    u_pred_vae = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_vae[:, :, 0] = initial_state
    u_pred_ensemble = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_ensemble[:,:,0] = initial_state

    # Dependent (time-dependent) prediction trajectory samples for t<=10.
    u_pred_dependent = np.zeros((NPATH, 3, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    u_pred_dependent[:, :, 0] = initial_state

    moment3_state_record = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_state_pred = np.zeros((3, 3, 3,int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    moment3_first,_ = compute_third_order_moments(initial_state)
    moment3_state_record[:,:,:,0] = moment3_first
    moment3_state_pred[:,:,:,0] = moment3_first

    moment3_state_nn = np.zeros((3, 3, 3, int(TIME_AMOUNT / dt) + 1), dtype=np.float32)
    moment3_state_tfdm = np.zeros_like(moment3_state_nn)
    moment3_state_vae = np.zeros_like(moment3_state_nn)
    moment3_state_nn[:, :, :, 0] = moment3_first
    moment3_state_tfdm[:, :, :, 0] = moment3_first
    moment3_state_vae[:, :, :, 0] = moment3_first

    mean_state_nn = np.zeros((3, int(TIME_AMOUNT / dt) + 1), dtype=np.float32)
    mean_state_tfdm = np.zeros_like(mean_state_nn)
    mean_state_vae = np.zeros_like(mean_state_nn)
    cov_state_nn = np.zeros((3, 3, int(TIME_AMOUNT / dt) + 1), dtype=np.float32)
    cov_state_tfdm = np.zeros_like(cov_state_nn)
    cov_state_vae = np.zeros_like(cov_state_nn)
    mean_state_nn[:, 0] = mean_state_tfdm[:, 0] = mean_state_vae[:, 0] = np.mean(
        initial_state, axis=0
    )
    c0 = np.cov(initial_state, rowvar=False)
    cov_state_nn[:, :, 0] = cov_state_tfdm[:, :, 0] = cov_state_vae[:, :, 0] = c0

    Energy_MC_all = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_pred = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_sran = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_MC_vae = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)

    current_state = initial_state
    current_pred_state_nn = initial_state.copy()
    current_pred_state_tfdm = initial_state.copy()
    current_pred_state_vae = initial_state.copy()

    Energy_update_record = np.zeros(4, dtype=np.float32)
    Energy_update_pred = np.zeros(4, dtype=np.float32)
    Energy_dyn_record = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)
    Energy_dyn_pred = np.zeros((4, int(TIME_AMOUNT/dt)+1), dtype=np.float32)

    # At t=0
    Energy_update_pred[:] = [
        0.5 * np.sum(mean_state_pred[:, 0] ** 2) + 0.5 * np.trace(cov_state_pred[:, :, 0]),
        0.5 * (mean_state_pred[0, 0] ** 2 + cov_state_pred[0, 0, 0]),
        0.5 * (mean_state_pred[1, 0] ** 2 + cov_state_pred[1, 1, 0]),
        0.5 * (mean_state_pred[2, 0] ** 2 + cov_state_pred[2, 2, 0]),
    ]
    Energy_dyn_pred[:, 0] = Energy_update_pred

    Energy_update_record[:] = [
        0.5 * np.sum(mean_state_record[:, 0] ** 2) + 0.5 * np.trace(cov_state_record[:, :, 0]),
        0.5 * (mean_state_record[0, 0] ** 2 + cov_state_record[0, 0, 0]),
        0.5 * (mean_state_record[1, 0] ** 2 + cov_state_record[1, 1, 0]),
        0.5 * (mean_state_record[2, 0] ** 2 + cov_state_record[2, 2, 0]),
    ]
    Energy_dyn_record[:, 0] = Energy_update_record
    Energy_MC_all[:, 0] = Energy_update_record
    Energy_MC_pred[:, 0] = Energy_update_pred
    Energy_MC_sran[:, 0] = Energy_update_pred
    Energy_MC_vae[:, 0] = Energy_update_pred

    # -----------------------------
    # Independent/Dependent setup (shared loop over t=0..TIME_AMOUNT)
    # -----------------------------
    Nt_ind = int(TIME_AMOUNT / dt)
    TIME_DEP_AMOUNT = 10.0
    Nt_dep = int(TIME_DEP_AMOUNT / dt)

    # Make explicit independent aliases (the rest of the existing code uses
    # the shorter variable names `mean_state_record`, `mean_state_pred`, etc.)
    mean_state_record_independent = mean_state_record
    cov_state_record_independent = cov_state_record
    mean_state_pred_independent = mean_state_pred
    cov_state_pred_independent = cov_state_pred

    tmM_dep = np.zeros((Nt_dep, 3), dtype=np.float32)
    tmS_dep = np.zeros(Nt_dep, dtype=np.float32)
    if args.params_name == "random_cascade":
        tmM_dep[:] = tmM[:Nt_dep, :]
        tmS_dep[:] = tmS[:Nt_dep]
    elif "tmM" in params and params["tmM"] is not None:
        if params["tmM"].shape[0] >= Nt_dep:
            tmM_dep[:] = params["tmM"][:Nt_dep, :].astype(np.float32)
        else:
            rep = int(np.ceil(Nt_dep / params["tmM"].shape[0]))
            tmM_dep[:] = np.tile(params["tmM"].astype(np.float32), (rep, 1))[:Nt_dep, :]
        if "tmS" in params and params["tmS"] is not None:
            if params["tmS"].shape[0] >= Nt_dep:
                tmS_dep[:] = params["tmS"][:Nt_dep].astype(np.float32)
            else:
                rep_s = int(np.ceil(Nt_dep / params["tmS"].shape[0]))
                tmS_dep[:] = np.tile(params["tmS"].astype(np.float32), rep_s)[:Nt_dep]
    elif "tmS" in params and params["tmS"] is not None:
        if params["tmS"].shape[0] >= Nt_dep:
            tmS_dep[:] = params["tmS"][:Nt_dep].astype(np.float32)
        else:
            rep_s = int(np.ceil(Nt_dep / params["tmS"].shape[0]))
            tmS_dep[:] = np.tile(params["tmS"].astype(np.float32), rep_s)[:Nt_dep]

    # Dependent arrays (only meaningful for t <= TIME_DEP_AMOUNT)
    mean_state_record_dependent = np.zeros((3, Nt_dep + 1), dtype=np.float32)
    cov_state_record_dependent = np.zeros((3, 3, Nt_dep + 1), dtype=np.float32)
    mean_state_pred_dependent = np.zeros((3, Nt_dep + 1), dtype=np.float32)
    cov_state_pred_dependent = np.zeros((3, 3, Nt_dep + 1), dtype=np.float32)

    moment3_state_pred_dependent = np.zeros((3, 3, 3, Nt_dep + 1), dtype=np.float32)
    moment3_state_pred_dependent[:, :, :, 0] = moment3_first

    mean_state_record_dependent[:, 0] = np.mean(initial_state, axis=0)
    cov_state_record_dependent[:, :, 0] = np.cov(initial_state, rowvar=False)
    mean_state_pred_dependent[:, 0] = np.mean(initial_state, axis=0)
    cov_state_pred_dependent[:, :, 0] = np.cov(initial_state, rowvar=False)

    current_state_dependent = initial_state.copy()
    current_pred_state_dependent = initial_state.copy()

    # Energy from (mean, covariance) for dependent prediction (only defined/updated for t <= 10)
    Energy_MC_pred_dependent = np.zeros((4, Nt_dep + 1), dtype=np.float32)
    Energy_MC_pred_dependent[:, 0] = [
        0.5 * np.sum(mean_state_pred_dependent[:, 0] ** 2) + 0.5 * np.trace(cov_state_pred_dependent[:, :, 0]),
        0.5 * (mean_state_pred_dependent[0, 0] ** 2 + cov_state_pred_dependent[0, 0, 0]),
        0.5 * (mean_state_pred_dependent[1, 0] ** 2 + cov_state_pred_dependent[1, 1, 0]),
        0.5 * (mean_state_pred_dependent[2, 0] ** 2 + cov_state_pred_dependent[2, 2, 0]),
    ]

    # Paths match `2stage_stochastic_time_independent.py` (DIR_TRIAD + RESIDUAL_SAMPLES).
    print("Loading neural network models...")
    dev_str = getattr(args, "DEVICE", str(device))
    if torch.cuda.is_available() and isinstance(dev_str, str) and dev_str.startswith("cuda"):
        model_PATH = os.path.join(
            config.DIR_TRIAD, "Results", "Results1", "Results", args.params_name
        )
        dep_root = model_PATH
    else:
        model_PATH = os.path.join(config.DIR_TRIAD, "Results", args.params_name)
        dep_root = model_PATH

    residual_samples = int(getattr(args, "RESIDUAL_SAMPLES", 10000))
    noise_str = f"noise_{args.NOISE_LEVEL}"
    save_dir = os.path.join(
        model_PATH, noise_str, f"second_stage_{residual_samples}_constant"
    )
    independent_save_dir = os.path.join(
        model_PATH, noise_str, f"second_stage_{residual_samples}_independent"
    )

    save_dir_single_dep = os.path.join(
        dep_root, noise_str, "deter1000", f"second_stage_{args.TRAIN_SIZE}_single"
    )
    save_dir_ensemble_dep = os.path.join(
        dep_root, noise_str, "deter1000", f"second_stage_{args.TRAIN_SIZE}"
    )
    alt_ensemble = os.path.join(
        dep_root, noise_str, "deter1000", f"ssecond_stage_{args.TRAIN_SIZE}"
    )
    if not os.path.exists(save_dir_ensemble_dep) and os.path.exists(alt_ensemble):
        save_dir_ensemble_dep = alt_ensemble
    if not os.path.exists(save_dir_single_dep):
        print(f"[WARNING] Dependent model folder not found: {save_dir_single_dep}")
        print("          Dependent prediction will fall back to simple Gaussian noise.")

    dataname = os.path.join(independent_save_dir, "data_inference.pt")
    vae_stats_path = os.path.join(independent_save_dir, "data_inference_vae.pt")
    residual_stats_path = os.path.join(independent_save_dir, "data_inference_residual.pt")
    nn_path = os.path.join(save_dir, "Neural_Network.pth")
    vae_path = os.path.join(save_dir, "ResidualVAE.pth")
    residual_nn_path = os.path.join(save_dir, "Residual_Network.pth")

    has_nn_legacy = os.path.exists(nn_path) and os.path.exists(dataname)
    Neural_Network = None
    ZT_mean_nn = ZT_std_nn = ODE_mean = ODE_std = None
    diff_scale_nn = args.DIFF_SCALE
    if has_nn_legacy:
        print(f"[INFO] Loading Neural_Network (FEX+NN) from: {nn_path}")
        data_inference = torch.load(dataname, map_location=device)
        ZT_mean_nn = data_inference["ZT_mean"].to(device)
        ZT_std_nn = data_inference["ZT_std"].to(device)
        ODE_mean = data_inference["ODE_mean"].to(device)
        ODE_std = data_inference["ODE_std"].to(device)
        diff_scale_nn = data_inference.get("diff_scale", args.DIFF_SCALE)
        if torch.is_tensor(diff_scale_nn):
            diff_scale_nn = diff_scale_nn.item()
        Neural_Network = FN_Net(3, 3, 50).to(device)
        Neural_Network.load_state_dict(torch.load(nn_path, map_location=device))
        Neural_Network.eval()
    else:
        print("[INFO] Legacy Neural_Network.pth + data_inference.pt not found (FEX+NN disabled).")

    has_residual_nn = os.path.exists(residual_nn_path) and os.path.exists(residual_stats_path)
    Residual_Network = None
    U_mean_res = U_std_res = RES_mean_res = RES_std_res = None
    diff_scale_res = 1.0
    if has_residual_nn:
        print(f"[INFO] Loading Residual_Network (FEX+TFDM) from: {residual_nn_path}")
        residual_stats = torch.load(residual_stats_path, map_location=device)
        U_mean_res = residual_stats["U_mean"].to(device)
        U_std_res = residual_stats["U_std"].to(device)
        RES_mean_res = residual_stats["RES_mean"].to(device)
        RES_std_res = residual_stats["RES_std"].to(device)
        diff_scale_res = residual_stats.get("diff_scale", 1.0)
        if torch.is_tensor(diff_scale_res):
            diff_scale_res = diff_scale_res.item()
        Residual_Network = FN_Net(3, 3, 50).to(device)
        Residual_Network.load_state_dict(torch.load(residual_nn_path, map_location=device))
        Residual_Network.eval()
    else:
        print("[INFO] Residual_Network.pth not found (FEX+TFDM will use Gaussian fallback).")

    has_vae = os.path.exists(vae_path) and os.path.exists(vae_stats_path)
    Residual_VAE = None
    ZT_mean_vae = ZT_std_vae = RES_mean = RES_std = None
    diff_scale_vae = args.DIFF_SCALE
    vae_format = 1
    vae_latent_dim = 8
    if has_vae:
        print(f"[INFO] Loading ResidualVAE (FEX+VAE) from: {vae_path}")
        vae_stats = torch.load(vae_stats_path, map_location=device)
        vae_format = int(vae_stats.get("vae_format", 1))
        vae_latent_dim = int(vae_stats.get("latent_dim", 8))
        Residual_VAE = ResidualVAE(3, 3, latent_dim=vae_latent_dim, hid_dim=50).to(device)
        Residual_VAE.load_state_dict(torch.load(vae_path, map_location=device))
        Residual_VAE.eval()
        if vae_format < 2:
            ZT_mean_vae = vae_stats["ZT_mean"].to(device)
            ZT_std_vae = vae_stats["ZT_std"].to(device)
        RES_mean = vae_stats["RES_mean"].to(device)
        RES_std = vae_stats["RES_std"].to(device)
        diff_scale_vae = vae_stats.get("diff_scale", args.DIFF_SCALE)
        if torch.is_tensor(diff_scale_vae):
            diff_scale_vae = diff_scale_vae.item()
    else:
        print("[INFO] ResidualVAE assets not found (FEX+VAE disabled).")

    has_nn = has_residual_nn or has_nn_legacy
    if not has_nn and not has_vae:
        print(
            "[WARNING] No second-stage checkpoints for this regime; "
            "independent FEX+NN / FEX+TFDM / FEX+VAE all use matched Gaussian noise. "
            f"Expected under:\n  {save_dir}\n  {independent_save_dir}\n"
            "Train 2stage_stochastic_time_independent.py for this params_name + noise "
            "when you want learned stochastic corrections."
        )

    ones_mc = np.ones((NPATH, 1), dtype=np.float64)
    L64 = np.asarray(L, dtype=np.float64)
    G64 = np.asarray(G, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    dt64 = float(dt)
    nl64 = float(args.NOISE_LEVEL)

    for idx in range(1, Nt_ind + 1):
        # Ground truth: Heun drift + diffusion (same as choice 4 in 2stage_stochastic_time_independent.py)
        tm_row = np.asarray(tmM[idx - 1, :], dtype=np.float64)
        cs = np.asarray(current_state, dtype=np.float64)
        k1 = (L64 @ cs.T).T - cs @ G64 + Buu(B64, cs, cs) + ones_mc * tm_row
        u1 = cs + dt64 * k1
        k2 = (L64 @ u1.T).T - u1 @ G64 + Buu(B64, u1, u1) + ones_mc * tm_row
        next_det = cs + dt64 * (k1 + k2) / 2.0
        SS_nd = np.asarray(
            params["SS"] + tmS[idx - 1] ** 2 * (params["SSt"] - params["SS"]),
            dtype=np.float64,
        )
        Winc = np.random.randn(NPATH, 3).astype(np.float64)
        next_state = next_det + np.sqrt(dt64) * nl64 * (Winc @ SS_nd)
        next_state = np.asarray(next_state, dtype=np.result_type(current_state, np.float32))
        SS = SS_nd.astype(np.float32)
        Winc_f32 = Winc.astype(np.float32)
        
        # Dependent (time-dependent) ground truth update for t <= 10
        if idx <= Nt_dep:
            k1_dep = (
                (L @ current_state_dependent.T).T
                - current_state_dependent @ G
                + Buu(B, current_state_dependent, current_state_dependent)
                + np.ones((NPATH, 1)) * tmM_dep[idx - 1, :]
            )
            u1_dep = current_state_dependent + dt * k1_dep
            k2_dep = (
                (L @ u1_dep.T).T
                - u1_dep @ G
                + Buu(B, u1_dep, u1_dep)
                + np.ones((NPATH, 1)) * tmM_dep[idx - 1, :]
            )
            next_state_dependent = current_state_dependent + dt * (k1_dep + k2_dep) / 2

            SS_dep_step = params['SS'] + tmS_dep[idx - 1] ** 2 * (params['SSt'] - params['SS'])
            next_state_dependent = next_state_dependent + np.sqrt(dt) * (Winc_f32 @ SS_dep_step) * args.NOISE_LEVEL

            mean_state_record_dependent[:, idx] = np.mean(next_state_dependent, axis=0)
            cov_state_record_dependent[:, :, idx] = np.cov(next_state_dependent, rowvar=False)
            current_state_dependent = next_state_dependent
        u_all[:, :, idx] = next_state

    
        mean_state_record[:,idx] = np.mean(next_state, axis=0)
        cov_state_record[:,:,idx] = np.cov(next_state, rowvar=False)
        moment3_state_record[:,:,:,idx],_ = compute_third_order_moments(next_state)
        Energy_MC_all[0, idx] = 0.5 * np.sum(mean_state_record[:,idx] ** 2) + 0.5 * np.trace(cov_state_record[:,:,idx])
        Energy_MC_all[1, idx] = 0.5 * (mean_state_record[0,idx] ** 2 + cov_state_record[0,0,idx])
        Energy_MC_all[2, idx] = 0.5 * (mean_state_record[1,idx] ** 2 + cov_state_record[1,1,idx])
        Energy_MC_all[3, idx] = 0.5 * (mean_state_record[2,idx] ** 2 + cov_state_record[2,2,idx])
        
      
        diag_G = np.diag(G)
        damp1 = np.max(diag_G)
        damp2 = max(np.min(diag_G), 0)
        damp3 = np.mean(diag_G)
        SS_sq_diag = np.diag(SS @ SS.T)
        
      
        Energy_update_record[0] += dt * (
            -np.sum(diag_G * (mean_state_record[:, idx] ** 2 + np.diag(cov_state_record[:, :, idx]))) +
             np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) +
             0.5 * np.sum(SS_sq_diag)
        )
        Energy_update_record[1] += dt * (-2 * damp1 * Energy_update_record[1] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
        Energy_update_record[2] += dt * (-2 * damp2 * Energy_update_record[2] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
        Energy_update_record[3] += dt * (-2 * damp3 * Energy_update_record[3] + np.sum(tmM[idx - 1, :] * mean_state_record[:, idx]) + 0.5 * np.sum(SS_sq_diag))
    
        current_state = next_state

        def det_update_from_state(state_np):
            state_tensor = torch.tensor(state_np, dtype=torch.float32).to(device)
            if args.params_name in [
                "periodic_cascade",
                "random_cascade",
                "random_cascade_deterministic",
            ]:
                current_time = idx * dt
                time_column = torch.full(
                    (state_tensor.shape[0], 1),
                    current_time,
                    dtype=torch.float32,
                    device=device,
                )
                state_with_time = torch.cat([state_tensor, time_column], dim=1)
                k1_det = FEX_model_check(
                    state_with_time,
                    model_name=args.Model,
                    params_name=args.params_name,
                    noise_level=args.NOISE_LEVEL,
                    device=device,
                ) * dt
                u1_det = state_tensor + k1_det
                u1_with_time = torch.cat([u1_det, time_column], dim=1)
                k2_det = FEX_model_check(
                    u1_with_time,
                    model_name=args.Model,
                    params_name=args.params_name,
                    noise_level=args.NOISE_LEVEL,
                    device=device,
                ) * dt
            else:
                k1_det = FEX_model_check(
                    state_tensor,
                    model_name=args.Model,
                    params_name=args.params_name,
                    noise_level=args.NOISE_LEVEL,
                    device=device,
                ) * dt
                u1_det = state_tensor + k1_det
                k2_det = FEX_model_check(
                    u1_det,
                    model_name=args.Model,
                    params_name=args.params_name,
                    noise_level=args.NOISE_LEVEL,
                    device=device,
                ) * dt
            return ((k1_det + k2_det) / 2).cpu().detach().numpy()

        det_update_nn = det_update_from_state(current_pred_state_nn)
        det_update_tfdm = det_update_from_state(current_pred_state_tfdm)
        det_update_vae = det_update_from_state(current_pred_state_vae)

        Npath = current_pred_state_nn.shape[0]
        Winc_tensor = torch.tensor(Winc_f32, dtype=torch.float32).to(device)
        with torch.no_grad():
            stoch_update_nn = None
            stoch_update_tfdm = None
            stoch_update_vae = None
            if has_nn_legacy:
                winc_nn = (Winc_tensor - ZT_mean_nn) / ZT_std_nn
                pred_nn = Neural_Network(winc_nn) * ODE_std + ODE_mean
                stoch_update_nn = (pred_nn / diff_scale_nn).cpu().detach().numpy()
            if has_residual_nn:
                z_norm = (Winc_tensor - U_mean_res) / U_std_res
                pred_res = Residual_Network(z_norm) * RES_std_res + RES_mean_res
                stoch_update_tfdm = (pred_res / diff_scale_res).cpu().detach().numpy()
            if has_vae:
                if vae_format >= 2:
                    z_prior = torch.randn(
                        Npath, vae_latent_dim, device=device, dtype=torch.float32
                    )
                    y_hat_vae = Residual_VAE.decode(z_prior)
                    pred_vae = y_hat_vae * RES_std + RES_mean
                else:
                    winc_vae = (Winc_tensor - ZT_mean_vae) / ZT_std_vae
                    y_hat_vae, _, _ = Residual_VAE(winc_vae)
                    pred_vae = y_hat_vae * RES_std + RES_mean
                stoch_update_vae = (pred_vae / diff_scale_vae).cpu().detach().numpy()

        simple_noise = np.sqrt(dt) * args.NOISE_LEVEL * (Winc_f32 @ SS)
        if stoch_update_nn is not None and not np.isfinite(stoch_update_nn).all():
            stoch_update_nn = simple_noise.copy()
        if stoch_update_tfdm is not None and not np.isfinite(stoch_update_tfdm).all():
            stoch_update_tfdm = simple_noise.copy()
        if stoch_update_vae is not None and not np.isfinite(stoch_update_vae).all():
            stoch_update_vae = simple_noise.copy()

        std_simple = np.std(simple_noise, axis=0)
        std_simple = np.maximum(std_simple, 1e-12)
        mean_simple = np.mean(simple_noise, axis=0)
        if stoch_update_nn is not None:
            std_n = np.std(stoch_update_nn, axis=0)
            mean_n = np.mean(stoch_update_nn, axis=0)
            scale_n = np.where(std_n > 1e-12, std_simple / std_n, 1.0)
            stoch_update_nn = (stoch_update_nn - mean_n) * scale_n + mean_simple
        if stoch_update_tfdm is not None:
            std_t = np.std(stoch_update_tfdm, axis=0)
            mean_t = np.mean(stoch_update_tfdm, axis=0)
            scale_t = np.where(std_t > 1e-12, std_simple / std_t, 1.0)
            stoch_update_tfdm = (stoch_update_tfdm - mean_t) * scale_t + mean_simple
        if stoch_update_vae is not None:
            std_v = np.std(stoch_update_vae, axis=0)
            mean_v = np.mean(stoch_update_vae, axis=0)
            scale_v = np.where(std_v > 1e-12, std_simple / std_v, 1.0)
            stoch_update_vae = (stoch_update_vae - mean_v) * scale_v + mean_simple

        # Dependent (time-dependent) stochastic update for t <= 10
        if idx <= Nt_dep:
            stoch_update_dependent = None
            # Recompute SS_dep_step for consistent SS scaling in prediction.
            SS_dep_step = params['SS'] + tmS_dep[idx - 1] ** 2 * (params['SSt'] - params['SS'])
            simple_noise_dependent = np.sqrt(dt) * (Winc @ SS_dep_step) * args.NOISE_LEVEL

            if os.path.exists(save_dir_single_dep):
                if args.params_name in ['equipart', 'cascade']:
                    stoch_update_dependent = simple_step_update(
                        Winc_tensor=Winc_tensor,
                        device=device,
                        idx=idx,
                        save_dir_single=save_dir_single_dep,
                        save_dir_ensemble=save_dir_ensemble_dep,
                        model_type='single',
                        dim=3,
                        scaler=scaler,
                    )
                else:
                    stoch_update_dependent = FN_multi_update(
                        Winc_tensor=Winc_tensor,
                        device=device,
                        idx=idx,
                        save_dir_single=save_dir_single_dep,
                        dim=3,
                        scaler=scaler,
                    )

            if stoch_update_dependent is None:
                stoch_update_dependent = simple_noise_dependent

            # Deterministic RK4 update for the dependent predictor (must be computed
            # from `current_pred_state_dependent`, same as in `2stage_stochastic_time_dependent.py`).
            current_tensor_dep = torch.tensor(
                current_pred_state_dependent, dtype=torch.float32
            ).to(device)
            if args.params_name in [
                "periodic_cascade",
                "random_cascade",
                "random_cascade_deterministic",
            ]:
                current_time = idx * dt
                time_column_dep = torch.full(
                    (current_tensor_dep.shape[0], 1),
                    current_time,
                    dtype=torch.float32,
                ).to(device)
                current_tensor_with_time_dep = torch.cat(
                    [current_tensor_dep, time_column_dep],
                    dim=1,
                )
                k1_det_dep = (
                    FEX_model_check(
                        current_tensor_with_time_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k1_det_dep_np = k1_det_dep.cpu().detach().numpy()
                u1_dep = current_tensor_dep + k1_det_dep

                u1_dep_with_time = torch.cat([u1_dep, time_column_dep], dim=1)
                k2_det_dep = (
                    FEX_model_check(
                        u1_dep_with_time,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k2_det_dep_np = k2_det_dep.cpu().detach().numpy()
            else:
                k1_det_dep = (
                    FEX_model_check(
                        current_tensor_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k1_det_dep_np = k1_det_dep.cpu().detach().numpy()
                u1_dep = current_tensor_dep + k1_det_dep

                k2_det_dep = (
                    FEX_model_check(
                        u1_dep,
                        model_name=args.Model,
                        params_name=args.params_name,
                        noise_level=args.NOISE_LEVEL,
                        device=device,
                    )
                    * dt
                )
                k2_det_dep_np = k2_det_dep.cpu().detach().numpy()

            det_update_dep = (k1_det_dep_np + k2_det_dep_np) / 2.0
            next_pred_state_dependent = (
                current_pred_state_dependent + det_update_dep + stoch_update_dependent
            )
            # Store dependent prediction samples for 3D phase-space clouds (t <= 10).
            u_pred_dependent[:, :, idx] = next_pred_state_dependent
            current_pred_state_dependent = next_pred_state_dependent
            mean_state_pred_dependent[:, idx] = np.mean(next_pred_state_dependent, axis=0)
            cov_state_pred_dependent[:, :, idx] = np.cov(next_pred_state_dependent, rowvar=False)

            # Third-order moments for dependent prediction
            moment3_pred_dep, _ = compute_third_order_moments(next_pred_state_dependent)
            moment3_state_pred_dependent[:, :, :, idx] = moment3_pred_dep

            # Dependent prediction energy (optional but kept consistent with mean/cov definition)
            Energy_MC_pred_dependent[0, idx] = (
                0.5 * np.sum(mean_state_pred_dependent[:, idx] ** 2)
                + 0.5 * np.trace(cov_state_pred_dependent[:, :, idx])
            )
            Energy_MC_pred_dependent[1, idx] = 0.5 * (
                mean_state_pred_dependent[0, idx] ** 2 + cov_state_pred_dependent[0, 0, idx]
            )
            Energy_MC_pred_dependent[2, idx] = 0.5 * (
                mean_state_pred_dependent[1, idx] ** 2 + cov_state_pred_dependent[1, 1, idx]
            )
            Energy_MC_pred_dependent[3, idx] = 0.5 * (
                mean_state_pred_dependent[2, idx] ** 2 + cov_state_pred_dependent[2, 2, idx]
            )
    
        if idx % 50 == 0:
            t_now = idx * dt
            print(f"\nStep {idx} (t={t_now:.2f}): independent stochastic increments (matched to simple noise)")
            print("=" * 50)
            if stoch_update_tfdm is not None:
                print(f"FEX+TFDM — mean {np.mean(stoch_update_tfdm, axis=0)}  std {np.std(stoch_update_tfdm, axis=0)}")
            else:
                print("FEX+TFDM — not available")
            if stoch_update_nn is not None:
                print(f"FEX+NN   — mean {np.mean(stoch_update_nn, axis=0)}  std {np.std(stoch_update_nn, axis=0)}")
            else:
                print("FEX+NN   — not available")
            if stoch_update_vae is not None:
                print(f"FEX+VAE  — mean {np.mean(stoch_update_vae, axis=0)}  std {np.std(stoch_update_vae, axis=0)}")
            else:
                print("FEX+VAE  — not available")
            print(f"Simple noise — mean {mean_simple}  std {std_simple}")
            print("=" * 50)

        next_pred_nn = (
            current_pred_state_nn + det_update_nn + stoch_update_nn
            if stoch_update_nn is not None
            else current_pred_state_nn
            + det_update_nn
            + (stoch_update_tfdm if stoch_update_tfdm is not None else simple_noise)
        )
        next_pred_tfdm = (
            current_pred_state_tfdm + det_update_tfdm + stoch_update_tfdm
            if stoch_update_tfdm is not None
            else current_pred_state_tfdm + det_update_tfdm + simple_noise
        )
        next_pred_vae = (
            current_pred_state_vae + det_update_vae + stoch_update_vae
            if stoch_update_vae is not None
            else current_pred_state_vae + det_update_vae + simple_noise
        )

        # Independent curves in plots: orange = FEX-TFDM; u_pred_single holds SRAN ensemble for debugging.
        u_pred_all[:, :, idx] = next_pred_tfdm
        u_pred_single[:, :, idx] = next_pred_nn
        u_pred_vae[:, :, idx] = next_pred_vae

        mean_state_nn[:, idx] = np.mean(next_pred_nn, axis=0)
        cov_state_nn[:, :, idx] = np.cov(next_pred_nn, rowvar=False)
        moment3_state_nn[:, :, :, idx], _ = compute_third_order_moments(next_pred_nn)

        mean_state_tfdm[:, idx] = np.mean(next_pred_tfdm, axis=0)
        cov_state_tfdm[:, :, idx] = np.cov(next_pred_tfdm, rowvar=False)
        moment3_state_tfdm[:, :, :, idx], _ = compute_third_order_moments(next_pred_tfdm)

        mean_state_vae[:, idx] = np.mean(next_pred_vae, axis=0)
        cov_state_vae[:, :, idx] = np.cov(next_pred_vae, rowvar=False)
        moment3_state_vae[:, :, :, idx], _ = compute_third_order_moments(next_pred_vae)

        mean_state_pred[:, idx] = mean_state_tfdm[:, idx]
        cov_state_pred[:, :, idx] = cov_state_tfdm[:, :, idx]
        moment3_state_pred[:, :, :, idx] = moment3_state_tfdm[:, :, :, idx]

        mean_state_single[:, idx] = mean_state_nn[:, idx]
        cov_state_single[:, :, idx] = cov_state_nn[:, :, idx]

        Energy_MC_pred[0, idx] = 0.5 * np.sum(mean_state_pred[:, idx] ** 2) + 0.5 * np.trace(cov_state_pred[:, :, idx])
        Energy_MC_pred[1, idx] = 0.5 * (mean_state_pred[0, idx] ** 2 + cov_state_pred[0, 0, idx])
        Energy_MC_pred[2, idx] = 0.5 * (mean_state_pred[1, idx] ** 2 + cov_state_pred[1, 1, idx])
        Energy_MC_pred[3, idx] = 0.5 * (mean_state_pred[2, idx] ** 2 + cov_state_pred[2, 2, idx])

        Energy_MC_sran[0, idx] = 0.5 * np.sum(mean_state_nn[:, idx] ** 2) + 0.5 * np.trace(cov_state_nn[:, :, idx])
        Energy_MC_sran[1, idx] = 0.5 * (mean_state_nn[0, idx] ** 2 + cov_state_nn[0, 0, idx])
        Energy_MC_sran[2, idx] = 0.5 * (mean_state_nn[1, idx] ** 2 + cov_state_nn[1, 1, idx])
        Energy_MC_sran[3, idx] = 0.5 * (mean_state_nn[2, idx] ** 2 + cov_state_nn[2, 2, idx])

        Energy_MC_vae[0, idx] = 0.5 * np.sum(mean_state_vae[:, idx] ** 2) + 0.5 * np.trace(cov_state_vae[:, :, idx])
        Energy_MC_vae[1, idx] = 0.5 * (mean_state_vae[0, idx] ** 2 + cov_state_vae[0, 0, idx])
        Energy_MC_vae[2, idx] = 0.5 * (mean_state_vae[1, idx] ** 2 + cov_state_vae[1, 1, idx])
        Energy_MC_vae[3, idx] = 0.5 * (mean_state_vae[2, idx] ** 2 + cov_state_vae[2, 2, idx])

        current_pred_state_nn = next_pred_nn
        current_pred_state_tfdm = next_pred_tfdm
        current_pred_state_vae = next_pred_vae

    idx_t20 = int(round(TIME_AMOUNT / dt))
    print("\n" + "=" * 60)
    print(
        f"[INFO] t = {TIME_AMOUNT} (time index {idx_t20}): mean, var(u_i), "
        "third moments — GT, FEX-SRAN, FEX-TFDM, FEX-VAE"
    )
    print("=" * 60)
    _snap = [
        ("Ground truth", mean_state_record, cov_state_record, moment3_state_record),
        ("FEX-SRAN", mean_state_nn, cov_state_nn, moment3_state_nn),
        ("FEX-TFDM", mean_state_tfdm, cov_state_tfdm, moment3_state_tfdm),
        ("FEX-VAE", mean_state_vae, cov_state_vae, moment3_state_vae),
    ]
    for label, ms, cs, m3 in _snap:
        md = ms[:, idx_t20]
        vd = np.diag(cs[:, :, idx_t20])
        print(
            f"  {label}: mean={md}  cov_diag={vd}  "
            f"M123={m3[0, 1, 2, idx_t20]:.6g}  M122={m3[0, 1, 1, idx_t20]:.6g}  "
            f"M133={m3[0, 2, 2, idx_t20]:.6g}  M223={m3[1, 1, 2, idx_t20]:.6g}"
        )
    print("=" * 60)

    np.random.seed(0)
    Time_record = np.arange(int(TIME_AMOUNT/dt)+1) * dt
    Time_dep = np.arange(Nt_dep + 1) * dt
    os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)
    if plot_composite:
        from .plot import plot_discussion_choice3_composite
        save_path_composite = os.path.join(
            args.LOG_SAVE_PATH, "discussion_choice3_composite.pdf"
        )
        if args.params_name == "cascade":
            regime_display_title = "Forward Cascade"
        else:
            regime_display_title = args.params_name.capitalize()
        plot_discussion_choice3_composite(
            save_path=save_path_composite,
            u_all=u_all,
            u_pred_all=u_pred_all,
            u_pred_dependent=u_pred_dependent,
            dt=dt,
            Time_record=Time_record,
            Time_dep=Time_dep,
            mean_state_record_independent=mean_state_record_independent,
            cov_state_record_independent=cov_state_record_independent,
            mean_state_pred_independent=mean_state_pred_independent,
            cov_state_pred_independent=cov_state_pred_independent,
            mean_state_pred_dependent=mean_state_pred_dependent,
            cov_state_pred_dependent=cov_state_pred_dependent,
            moment3_state_record=moment3_state_record,
            moment3_state_pred=moment3_state_pred,
            moment3_state_pred_dependent=moment3_state_pred_dependent,
            TIME_AMOUNT=TIME_AMOUNT,
            TIME_DEP_AMOUNT=TIME_DEP_AMOUNT,
            params_name=regime_display_title,
            font_size=20,
        )
    return {
        "Time_ind": Time_record,
        "Time_dep": Time_dep,
        "dt": dt,
        "params": params,
        "u_all_gt": u_all,
        "u_pred_tfdm": u_pred_all,
        "u_pred_sran": u_pred_single,
        "u_pred_vae": u_pred_vae,
        "mean_gt": mean_state_record,
        "mean_pred_tfdm": mean_state_tfdm,
        "mean_pred_sran": mean_state_nn,
        "mean_pred_vae": mean_state_vae,
        "cov_gt": cov_state_record,
        "moment3_gt": moment3_state_record,
        # Discussion 4 grid: TFDM (Residual NN), SRAN (legacy z→residual NN), VAE
        "cov_pred_tfdm": cov_state_tfdm,
        "moment3_pred_tfdm": moment3_state_tfdm,
        "cov_pred_sran": cov_state_nn,
        "moment3_pred_sran": moment3_state_nn,
        "cov_pred_vae": cov_state_vae,
        "moment3_pred_vae": moment3_state_vae,
        # Aliases: independent orange curve in older plots = TFDM track
        "cov_pred_ind": cov_state_pred,
        "moment3_pred_ind": moment3_state_pred,
        "cov_pred_dep": cov_state_pred_dependent,
        "moment3_pred_dep": moment3_state_pred_dependent,
        # Energy: index 0 = total, 1–3 = per-dimension (same convention as energy_comparison.pdf)
        "energy_gt": Energy_MC_all,
        "energy_pred_tfdm": Energy_MC_pred,
        "energy_pred_sran": Energy_MC_sran,
        "energy_pred_vae": Energy_MC_vae,
    }



