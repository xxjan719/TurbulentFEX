import config
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from config import DIR_EXAMPLE, create_main_parser
from utils.plot import plot_NOISE_LEVEL_EFFECT
from utils.helper import get_coefficients

# Hardcode DIR_TRIAD for this test
DIR_TRIAD = "src/Example/MC_triad"

parser = create_main_parser()
args = parser.parse_args()

# Set device to cpu to avoid torch issues
args.DEVICE = 'cpu'
print(f"Using device: {args.DEVICE}")

# Set up base path
base_path = os.path.join(DIR_EXAMPLE, args.Model, 'Results')
print(f"Base path: {base_path}")
    
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'

print("DIR_TRIAD being used:", DIR_TRIAD)
print("args.DEVICE:", args.DEVICE)
print("args.Model:", args.Model)

# Get coefficients
coefficients = get_coefficients(load_dir=DIR_TRIAD, DEVICE=args.DEVICE)

# Print out all coefficients for each dimension and term
print("\n=== Coefficient Values by Noise Level ===")
for dim_key, dim_data in coefficients.items():
    print(f"\n{dim_key}:")
    for term_key, coeff_list in dim_data.items():
        print(f"  {term_key}: {coeff_list}")

# Create save directory if it doesn't exist
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)

# Plot the results
plot_NOISE_LEVEL_EFFECT(coefficients, save_dir=args.LOG_SAVE_PATH)







