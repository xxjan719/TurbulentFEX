import config
import argparse
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import torch
from config import DIR_EXAMPLE, create_main_parser
from utils.plot import (
    plot_cross_term_vs_noise,
    plot_cross_term_vs_sample,
    plot_mean_covariance_grid_ind_dep,
    plot_log10_error_mean_covariance_grid_ind_dep,
    plot_third_order_moments_2x2,
    plot_third_order_moments_ind_dep_2x2,
    plot_state_projections_3x3,
    plot_state_projections_cases_3x9,
    plot_triad_3d_time_grid_matplotlib_cloud_3x3_times,
    run_discussion_choice4_triad_grid,
    run_discussion_choice5_offdiagonal_cov_grid,
    run_discussion_cov_moments_grid,
)
from utils.helper import (
    get_coefficients,
    get_coefficients_from_finak_sample_file,
    discussion_choice5_rollout,
)

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

# Some parts of this script historically used a lower-case `device` variable.
# Keep it as an alias for compatibility.
device = DEVICE
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'

print("DIR_TRIAD being used:", DIR_TRIAD)
print("args.DEVICE:", args.DEVICE)
print("args.Model:", args.Model)


# Ask user whether to train everything in second stage or skip to calculate the measurements
print("\n"+ "="*60)
print("SECOND STAGE: STOCHASTIC OPTIONS")
print("="*60)
print("1. Discussion 2: different noise levels test")
print("2. Discussion 3: different sample sizes test")
print("3. Discussion 4: covariance diagonals + third moments (3×7 grid: equipart / Forward Cascade / dual_cascade)")
print("4. Discussion 5: off-diagonal covariances (3×3) time indep vs dep")
print("5. Diffusion (random_cascade): cov(u_i,u_i) + all ⟨M⟩ (1×7), fontsize 28")
print("6. Periodic + random deterministic forcing: same columns (2×7), fontsize 28")
print("7. State projections 3×9: three cases × three projections at t=5,10,20")

while True:
    # choice = '1'  # uncomment for debugging
    choice = input("\nChoose option (1–7): ").strip()
    if choice in ["1", "2", "3", "4", "5", "6", "7"]:
        break
    print("Please enter a number from 1 to 7.")

if choice == '1':
    print("=" * 60)
    # Get coefficients for all three regimes (equipart, cascade, dual_cascade)
    coefficients_equipart = get_coefficients(
        load_dir="Example/MC_triad", model_name="equipart", DEVICE=DEVICE
    )
    coefficients_cascade = get_coefficients(
        load_dir="Example/MC_triad", model_name="cascade", DEVICE=DEVICE
    )
    coefficients_dual_cascade = get_coefficients(
        load_dir="Example/MC_triad", model_name="dual_cascade", DEVICE=DEVICE
    )

    print("\n=== Coefficient Values by Noise Level (equipart) ===")
    for dim_key, dim_data in coefficients_equipart.items():
        print(f"\n{dim_key}:")
        for term_key, coeff_list in dim_data.items():
            print(f"  {term_key}: {coeff_list}")

    # Create save directory if it doesn't exist
    os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)

    # Common noise levels used across plots
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    # Plot cross-term coefficient errors: left=equipart, center=cascade, right=dual_cascade
    plot_cross_term_vs_noise(
        [coefficients_equipart, coefficients_cascade, coefficients_dual_cascade],
        noise_levels=noise_levels,
        save_dir=args.LOG_SAVE_PATH,
        panel_labels=['Equipartition', 'Forward Cascade', 'Dual Cascade'],
    )

    # Optional: plot the original energy-conservation diagnostic
    # plot_energy_conservation(coefficients_equipart, noise_levels=noise_levels, save_dir=args.LOG_SAVE_PATH)

elif choice == '2':
    print("=" * 60)
    # Nonlinear terms graph from FINAK_EXPR_SAMPLE.txt (same plot style, x-axis = sample size)
    # Paths: Results/equipart, Results/cascade, Results/dual_cascade
    results_dir = os.path.join(DIR_EXAMPLE, args.Model, 'Results')
    finak_paths = [
        os.path.join(results_dir, 'equipart', 'FINAK_EXPR_SAMPLE.txt'),
        os.path.join(results_dir, 'cascade', 'FINAK_EXPR_SAMPLE.txt'),
        os.path.join(results_dir, 'dual_cascade', 'FINAK_EXPR_SAMPLE.txt'),
    ]
    coeff_equipart_sample, sample_sizes_equipart = get_coefficients_from_finak_sample_file(finak_paths[0])
    coeff_cascade_sample, sample_sizes_cascade = get_coefficients_from_finak_sample_file(finak_paths[1])
    coeff_dual_sample, sample_sizes_dual = get_coefficients_from_finak_sample_file(finak_paths[2])
    sample_sizes = (
        sample_sizes_equipart
        or sample_sizes_cascade
        or sample_sizes_dual
        or [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )

    plot_cross_term_vs_sample(
        [coeff_equipart_sample, coeff_cascade_sample, coeff_dual_sample],
        sample_sizes=sample_sizes,
        save_dir=results_dir,
        filename='cross_terms_vs_sample.pdf',
        panel_labels=['Equipartition', 'Forward Cascade', 'Dual Cascade'],
    )

    # Copy the same nonlinear terms graph into each regime folder
    pdf_src = os.path.join(results_dir, 'cross_terms_vs_sample.pdf')
    for subdir in ('equipart', 'cascade', 'dual_cascade'):
        save_subdir = os.path.join(results_dir, subdir)
        os.makedirs(save_subdir, exist_ok=True)
        if os.path.exists(pdf_src):
            shutil.copy(pdf_src, os.path.join(save_subdir, 'cross_terms_vs_sample.pdf'))
            print(f"Copied nonlinear terms graph to {save_subdir}")

elif choice == '3':
    print("=" * 60)
    print("[INFO] Discussion 4: 3×7 grid — cov(u_i,u_i) and selected ⟨M⟩; three regimes.")
    print("=" * 60)
    run_discussion_choice4_triad_grid(
        args,
        base_path=base_path,
        dir_example=DIR_EXAMPLE,
        model_name=args.Model,
        rollout_worker=lambda plot_composite=False: discussion_choice5_rollout(
            args, device, plot_composite=plot_composite
        ),
        fs=28,
    )

elif choice == '4':
    print("=" * 60)
    print("[INFO] Discussion 5: 3×3 off-diagonal cov (time indep vs dep).")
    print("=" * 60)
    run_discussion_choice5_offdiagonal_cov_grid(
        args,
        base_path=base_path,
        dir_example=DIR_EXAMPLE,
        model_name=args.Model,
        rollout_worker=lambda plot_composite=False: discussion_choice5_rollout(
            args, device, plot_composite=plot_composite
        ),
        fs=20,
    )

elif choice == "5":
    print("=" * 60)
    print("[INFO] Diffusion (random_cascade): 1×7 cov diagonals + third moments, fs=28.")
    print("=" * 60)
    run_discussion_cov_moments_grid(
        args,
        base_path=base_path,
        dir_example=DIR_EXAMPLE,
        model_name=args.Model,
        rollout_worker=lambda plot_composite=False: discussion_choice5_rollout(
            args, device, plot_composite=plot_composite
        ),
        regimes=("random_cascade",),
        row_labels=("Random cascade",),
        save_filename="discussion_diffusion_random_cascade_cov_moments_grid.pdf",
        fs=28,
        log_inset_row_index=None,
        log_label="discussion diffusion 1×7 grid",
    )

elif choice == "6":
    print("=" * 60)
    print(
        "[INFO] Periodic cascade + random cascade (deterministic): 2×7 cov + ⟨M⟩, fs=28."
    )
    print("=" * 60)
    run_discussion_cov_moments_grid(
        args,
        base_path=base_path,
        dir_example=DIR_EXAMPLE,
        model_name=args.Model,
        rollout_worker=lambda plot_composite=False: discussion_choice5_rollout(
            args, device, plot_composite=plot_composite
        ),
        regimes=("periodic_cascade", "random_cascade_deterministic"),
        row_labels=("Periodic cascade", "Random cascade"),
        save_filename="discussion_periodic_random_det_cov_moments_grid.pdf",
        fs=28,
        log_inset_row_index=None,
        log_label="discussion periodic + random det 2×7 grid",
    )

elif choice == "7":
    print("=" * 60)
    print("[INFO] State projections 3×9: Equipartition / Forward Cascade / Dual Cascade.")
    print("=" * 60)
    out_png = os.path.join(
        base_path, "discussion_state_projections_3x9_cases_t5_t10_t20.pdf"
    )
    case_specs = [
        ("Equipartition", "equipart"),
        ("Forward Cascade", "cascade"),
        ("Dual Cascade", "dual_cascade"),
    ]
    case_data = {}
    for display_name, params_name in case_specs:
        args_case = argparse.Namespace(**vars(args))
        args_case.params_name = params_name
        rollout = discussion_choice5_rollout(args_case, device, plot_composite=False)
        case_data[display_name] = rollout["u_all_gt"]

    saved = plot_state_projections_cases_3x9(
        case_data=case_data,
        dt=rollout["dt"],
        times=(5, 10, 20),
        save_path=out_png,
        fs=52,
    )
    print(f"[SAVED] {saved}")
