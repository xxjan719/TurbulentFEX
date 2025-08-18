import config
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from config import DIR_EXAMPLE, create_main_parser
from utils.plot import plot_NOISE_LEVEL_EFFECT, plot_energy_conservation
from utils.helper import get_coefficients

# Hardcode DIR_TRIAD for this test
DIR_TRIAD = "Example/MC_triad"

parser = create_main_parser()
args = parser.parse_args()

# Set device to cpu to avoid torch issues
print(f"Using device: {args.DEVICE}")

# Set up base path
if str(args.DEVICE) == 'cpu':
    base_path = os.path.join(DIR_EXAMPLE, args.Model, 'Results')
    print(f"Base path: {base_path}")
elif str(args.DEVICE) == "cuda:0":
    base_path = os.path.join(DIR_TRIAD, "Results","Results1","Results")
    
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'

print("DIR_TRIAD being used:", DIR_TRIAD)
print("args.DEVICE:", args.DEVICE)
print("args.Model:", args.Model)

# Get coefficients
coefficients = get_coefficients(load_dir="Example/MC_triad",model_name=args.params_name,DEVICE=args.DEVICE)

# Print out all coefficients for each dimension and term
print("\n=== Coefficient Values by Noise Level ===")
for dim_key, dim_data in coefficients.items():
    print(f"\n{dim_key}:")
    for term_key, coeff_list in dim_data.items():
        print(f"  {term_key}: {coeff_list}")

# Create save directory if it doesn't exist
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)

# Plot the results
plot_NOISE_LEVEL_EFFECT(coefficients, noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], save_dir=args.LOG_SAVE_PATH)


# Plot the energy conservation
# Plot the sum of cross-terms
plot_energy_conservation(coefficients, noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], save_dir=args.LOG_SAVE_PATH)





