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
    plot_state_projections_cases_4x9_scatter_and_gt_density,
    plot_discussion_first_moments_5x6,
    plot_discussion_energy_modes_5x6,
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
print(
    "3. Discussion 4: ‖⟨u⟩‖₂ + covariance diagonals + third moments "
    "(3×8 grid: equipart / Forward Cascade / dual_cascade)"
)
print("4. Discussion 5: off-diagonal covariances (3×3) time indep vs dep")
print("5. Diffusion (random_cascade_deterministic): ‖⟨u⟩‖₂ + cov + ⟨M⟩ (1×8), fontsize 28")
print("6. Periodic + random cascade deterministic: same columns (2×8), fontsize 28")
print(
    "7. State projections 4×9: TFDM / SRAN / VAE scatter + GT density at t=20 only"
)
print(
    "8. State projections 4×6: periodic + random cascade det. (same style as opt. 7, t=20)"
)
print(
    "9. First moments 5×6: ⟨u1⟩,⟨u2⟩,⟨u3⟩ + log10|pred−GT| per component, five regimes"
)
print(
    "10. Energy modes 5×6: per-dimension energy (no total) + log10|pred−GT|, five regimes"
)

while True:
    # choice = '1'  # uncomment for debugging
    choice = input("\nChoose option (1–10): ").strip()
    if choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        break
    print("Please enter a number from 1 to 10.")

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
    print(
        "[INFO] Discussion 4: 3×8 grid — ‖⟨u⟩‖₂, cov(u_i,u_i), selected ⟨M⟩; three regimes."
    )
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
    # Same figure builder as option 3: plot_discussion_cov_moments_grid (row labels, ticks, margins).
    print("=" * 60)
    print(
        "[INFO] Diffusion (random_cascade_deterministic): 1×8 ‖⟨u⟩‖₂ + cov diagonals + ⟨M⟩, fs=28."
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
        regimes=("random_cascade_deterministic",),
        row_labels=("Random cascade",),
        save_filename="discussion_diffusion_random_cascade_deterministic_cov_moments_grid.pdf",
        fs=28,
        log_label="discussion diffusion 1×8 grid",
    )

elif choice == "6":
    # Row 2 rollout uses random_cascade_deterministic; y-axis label is short "Random cascade".
    print("=" * 60)
    print(
        "[INFO] Periodic cascade + random_cascade_deterministic: 2×8 ‖⟨u⟩‖₂ + cov + ⟨M⟩, fs=28 "
        '(second row y-label: "Random cascade").'
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
        log_label="discussion periodic + random cascade 2×8 grid",
    )

elif choice == "7":
    print("=" * 60)
    print(
        "[INFO] State projections 4×9 at t=20: fs=40; x-axis per column matches Ground truth row (3 ticks)."
    )
    print("=" * 60)
    out_pdf = os.path.join(
        base_path,
        "discussion_state_projections_4x9_t20_scatter_methods_gt_density.pdf",
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
        case_data[display_name] = {
            "gt": rollout["u_all_gt"],
            "tfdm": rollout["u_pred_tfdm"],
            "sran": rollout["u_pred_sran"],
            "vae": rollout["u_pred_vae"],
        }

    _nplot = int(getattr(args, "RESIDUAL_SAMPLES", 10000))
    _max_pts = max(_nplot, 12000)
    saved = plot_state_projections_cases_4x9_scatter_and_gt_density(
        case_data=case_data,
        dt=rollout["dt"],
        time=20.0,
        save_path=out_pdf,
        fs=40,
        max_points=_max_pts,
        point_size=3.0,
        alpha=0.55,
        case_title_x_shift=0.04,
        case_title_x_shift_forward_dual_extra=0.03,
        cell_side_inches=5.0,
        wspace=0.56,
        hspace=0.50,
        row_label_pad=0.0008,
        xaxis_numticks=3,
    )
    print(f"[SAVED] {saved}")

elif choice == "8":
    print("=" * 60)
    print(
        "[INFO] State projections 4×6: Periodic + random cascade (det.); "
        "same layout as option 7 (4 rows × 6 cols), t=20; optional 3×6 contours t=5,10,20."
    )
    print("=" * 60)

    out_contour = os.path.join(
        base_path,
        "discussion_state_projections_3x6_periodic_random_det_cases_t5_t10_t20.pdf",
    )
    out_scatter_4x6 = os.path.join(
        base_path,
        "discussion_state_projections_4x6_periodic_random_det_t20_scatter_methods_gt_density.pdf",
    )

    case_specs = [
        ("Periodic cascade", "periodic_cascade"),
        ("Random cascade", "random_cascade_deterministic"),
    ]
    case_data = {}
    for display_name, params_name in case_specs:
        args_case = argparse.Namespace(**vars(args))
        args_case.params_name = params_name
        rollout = discussion_choice5_rollout(args_case, device, plot_composite=False)
        case_data[display_name] = {
            "gt": rollout["u_all_gt"],
            "tfdm": rollout["u_pred_tfdm"],
            "sran": rollout["u_pred_sran"],
            "vae": rollout["u_pred_vae"],
        }

    saved = plot_state_projections_cases_3x9(
        case_data={k: v["gt"] for k, v in case_data.items()},
        dt=rollout["dt"],
        times=(5, 10, 20),
        save_path=out_contour,
        fs=56,
    )
    print(f"[SAVED] {saved}")

    _nplot = int(getattr(args, "RESIDUAL_SAMPLES", 10000))
    _max_pts = max(_nplot, 12000)
    saved_s = plot_state_projections_cases_4x9_scatter_and_gt_density(
        case_data=case_data,
        dt=rollout["dt"],
        time=20.0,
        save_path=out_scatter_4x6,
        fs=40,
        max_points=_max_pts,
        point_size=3.0,
        alpha=0.55,
        case_title_x_shift=0.04,
        case_title_x_shift_forward_dual_extra=0.03,
        cell_side_inches=5.0,
        wspace=0.56,
        hspace=0.50,
        row_label_pad=0.0008,
        forward_cascade_yticks=None,
        dual_cascade_yticks=None,
        periodic_cascade_case_name="Periodic cascade",
        periodic_cascade_yticks=(-5.0, 0.0, 5.0),
        random_cascade_case_name="Random cascade",
        random_cascade_yticks=(-0.3, 0.0, 0.3),
    )
    print(f"[SAVED] {saved_s}")

elif choice == "9":
    print("=" * 60)
    print(
        "[INFO] First moments 5×6: means + log10|mean_pred−mean_GT| "
        "(ASD-FEX-TFDM / SRAN / VAE; see rollout print block)."
    )
    print("=" * 60)
    out_pdf = os.path.join(
        base_path, "discussion_first_moments_5x3_five_regimes.pdf"
    )
    regime_specs = [
        ("Equipartition", "equipart"),
        ("Forward cascade", "cascade"),
        ("Dual cascade", "dual_cascade"),
        ("Periodic cascade", "periodic_cascade"),
        ("Random cascade", "random_cascade_deterministic"),
    ]
    regime_series = []
    for row_label, params_name in regime_specs:
        args_case = argparse.Namespace(**vars(args))
        args_case.params_name = params_name
        rollout = discussion_choice5_rollout(args_case, device, plot_composite=False)
        regime_series.append({"row_label": row_label, "rollout": rollout})

    saved = plot_discussion_first_moments_5x6(
        regime_series=regime_series,
        save_path=out_pdf,
        fs=28,
    )
    print(f"[SAVED] {saved}")

elif choice == "10":
    print("=" * 60)
    print(
        "[INFO] Energy modes 5×6: modes 1–3 only (no total) + log10|energy_pred−energy_GT|."
    )
    print("=" * 60)
    out_pdf = os.path.join(
        base_path, "discussion_energy_modes_5x6_five_regimes.pdf"
    )
    regime_specs = [
        ("Equipartition", "equipart"),
        ("Forward cascade", "cascade"),
        ("Dual cascade", "dual_cascade"),
        ("Periodic cascade", "periodic_cascade"),
        ("Random cascade", "random_cascade_deterministic"),
    ]
    regime_series = []
    for row_label, params_name in regime_specs:
        args_case = argparse.Namespace(**vars(args))
        args_case.params_name = params_name
        rollout = discussion_choice5_rollout(args_case, device, plot_composite=False)
        regime_series.append({"row_label": row_label, "rollout": rollout})

    saved = plot_discussion_energy_modes_5x6(
        regime_series=regime_series,
        save_path=out_pdf,
        fs=28,
    )
    print(f"[SAVED] {saved}")
