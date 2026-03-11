import config
import numpy as np
import os
import re
import torch
import matplotlib.pyplot as plt
from config import DIR_EXAMPLE, create_main_parser
from utils.plot import plot_NOISE_LEVEL_EFFECT, plot_energy_conservation, plot_cross_term_vs_noise
from utils.helper import get_coefficients

# Hardcode DIR_TRIAD for this test
DIR_TRIAD = "Example/MC_triad"

parser = create_main_parser()
args = parser.parse_args()

# Set device to cpu to avoid torch issues
print(f"Using device: {args.DEVICE}")

if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results','Results1','Results')
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results')


    
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'

print("DIR_TRIAD being used:", DIR_TRIAD)
print("args.DEVICE:", args.DEVICE)
print("args.Model:", args.Model)

# Get coefficients for all three regimes (equipart, cascade, dual_cascade)
coefficients_equipart = get_coefficients(load_dir="Example/MC_triad", model_name="equipart", DEVICE=DEVICE)
coefficients_cascade = get_coefficients(load_dir="Example/MC_triad", model_name="cascade", DEVICE=DEVICE)
coefficients_dual_cascade = get_coefficients(load_dir="Example/MC_triad", model_name="dual_cascade", DEVICE=DEVICE)

# Print coefficients for current params_name only (optional: print all three)
coefficients = coefficients_equipart  # for any downstream use
print("\n=== Coefficient Values by Noise Level (equipart) ===")
for dim_key, dim_data in coefficients_equipart.items():
    print(f"\n{dim_key}:")
    for term_key, coeff_list in dim_data.items():
        print(f"  {term_key}: {coeff_list}")

# Create save directory if it doesn't exist
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)

# Common noise levels used across plots
noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# Optional: detailed coefficient error vs noise (per term and dimension)
# plot_NOISE_LEVEL_EFFECT(coefficients, noise_levels=noise_levels, save_dir=args.LOG_SAVE_PATH)

# Plot cross-term coefficient errors: left=equipart, center=cascade, right=dual_cascade
plot_cross_term_vs_noise(
    [coefficients_equipart, coefficients_cascade, coefficients_dual_cascade],
    noise_levels=noise_levels,
    save_dir=args.LOG_SAVE_PATH,
    panel_labels=['Equipart', 'Forward Cascade', 'Dual Cascade'],
)

# Optional: plot the original energy-conservation diagnostic
# plot_energy_conservation(coefficients, noise_levels=noise_levels, save_dir=args.LOG_SAVE_PATH)





