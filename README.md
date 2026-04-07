# TurbulentFEX: Feature Expression Model for Turbulent Systems

## Description
```bash
TurbulentFEX/
│
├── src/
│   ├── __init__.py                    # Make src a Python package
│   ├── 1stage_deterministic.py        # First stage: FEX learning drift term
│   ├── 2stage_stochastic_time_dependent.py  # Second stage (time-dependent)
│   ├── comparison_wsindy.py           # WSINDy / SINDy vs MC_triad npz bundles
│   ├── config.py                      # Central configuration management
│   ├── utils/
│   │   ├── __init__.py                # Make utils a Python package
│   │   ├── FEX.py                     # Functional Expansion implementation
│   │   ├── ODEParser.py               # ODE solving and integration utilities
│   │   ├── controller.py              # Neural controller for FEX exploration
│   │   ├── Train_Integrator.py        # Training integration utilities
│   │   ├── Pool.py                    # Candidate pool management
│   │   ├── Sampler.py                 # Sampling utilities
│   │   ├── constant.py                # Constants and operations
│   │   ├── helper.py                  # Helper functions
│   │   ├── wsindy.py                  # Weak SINDy (WSINDy)
│   │   └── plot.py                    # Plotting utilities
│   ├── Example/
│   │   └── MC_triad/                  # Example: Monte Carlo triad system
│   │       ├── MC_triad.py            # Main triad system definition
│   │       └── Results/               # Output directories for results
│   │           ├── equipart/          # Equipartition results
│   │           └── cascade/           # Cascade results
│   └── README.md                      # Detailed usage documentation
│
├── turbulentfex_env/                  # Python virtual environment
├── environment.yml                     # Conda environment specification
└── .gitignore                         # Git ignore file
```

## Current File Structure (Actual)

The project has evolved from the original structure. Here's what actually exists:

### Main Scripts
- **`1stage_deterministic.py`**: FEX learning for drift terms (first stage)
- **`2stage_stochastic_time_dependent.py`** / **`2stage_stochastic_time_independent.py`**: Second-stage stochastic training
- **`comparison_wsindy.py`**: WSINDy fits and diagnostics on MC_triad `simulation_results_*.npz` bundles (batch or interactive menu)
- **`config.py`**: Shared CLI defaults and configuration

### Utility Modules (`src/utils/`)
- **`ODEParser.py`**: ODE solving and neural network training (33KB, 801 lines)
- **`plot.py`**: Comprehensive plotting utilities (51KB, 1189 lines)
- **`FEX.py`**: Functional Expansion implementation (15KB, 424 lines)
- **`helper.py`**: Helper functions (18KB, 474 lines)
- **`wsindy.py`**: Weak SINDy (WSINDy) implementation used by `comparison_wsindy.py`
- **`Train_Integrator.py`**: Training integration (3.1KB, 92 lines)
- **`Pool.py`**: Candidate pool management (1.3KB, 51 lines)
- **`Sampler.py`**: Sampling utilities (1.3KB, 37 lines)
- **`controller.py`**: Neural controller (1.2KB, 43 lines)
- **`constant.py`**: Constants and operations (367B, 15 lines)

### Example Implementation (`src/Example/MC_triad/`)
- **`MC_triad.py`**: Monte Carlo triad system definition (18KB, 455 lines)
- **Results structure**:
  - `equipart/`: Equipartition parameter results
  - `cascade/`: Cascade parameter results
    - `noise_0.2/`: Results for 0.2 noise level
    - `noise_1.0/`: Results for 1.0 noise level (main focus)
      - `second_stage_10000/`: Ensemble neural network models
      - `second_stage_10000_single/`: Single neural network models
      - `second_stage_10000_common/`: Shared data and ODE solutions
      - `plots/`: Generated comparison plots
      - `final_expressions.txt`: Learned FEX expressions
      - `simulation_results_noise_1.0.npz`: Simulation data (230MB)

## Recent Updates

### Neural Network Training (Current Focus)
- **Second stage training**: Successfully implemented neural network training for stochastic components
- **Time range training**: Added support for training specific time ranges (0-250, 250-500, 500-750, 750-1000)
- **Method selection**: Support for both single neural network and ensemble methods
- **Model saving**: Fixed issues with model saving and added debug output

### FEX Learning (First Stage)
- **Drift term discovery**: Functional Expansion for learning deterministic dynamics
- **Coefficient optimization**: Two-phase training with LBFGS refinement
- **Expression learning**: Automatic discovery of mathematical expressions

### Code Improvements
- **Modular structure**: Separated training and prediction logic
- **Error handling**: Added comprehensive error checking and debugging
- **Configuration**: Centralized parameter management
- **Documentation**: Enhanced code comments and structure

## Usage

### Environment Setup
```bash
python config.py

```

### WSINDy on MC triad data (`comparison_wsindy.py`)

Fit WSINDy on the **ensemble mean** stored in `src/Example/MC_triad/Results/<case>/noise_<level>/simulation_results_noise_<level>.npz`, compare to PySINDy (unless `--wsindy_only`), and optionally write figures next to that npz.

**Plot diagnostics** (writes PNGs in the same folder as the npz):

| Output | Meaning |
|--------|---------|
| `wsindy_fit_comparison.png` | Ensemble mean vs **full** WSINDy; if `--triad_interaction_only`, also the triad-projected curve on the data window |
| `wsindy_prediction_extended.png` | Deterministic + stochastic Heun rollouts: use **triad-enforced** WSINDy when `--triad_interaction_only`, else the **full** fit (`t ∈ [0, prediction_t_max]`, same `dt` as the npz) |
| `wsindy_truth_paired_tmax.png` | Only if `--paired_truth`: ground-truth resim vs WSINDy with shared noise (enforced WSINDy if `--triad_interaction_only`) |

**Examples**

Single case (e.g. equipart), **triad-enforced** rollouts (`--triad_interaction_only`), plots + optional paired truth:

```bash
cd src
python comparison_wsindy.py --batch --params_name equipart --noise_level 1.0 \
  --wsindy_only --no_standardize --triad_interaction_only \
  --plot_wsindy --paired_truth --prediction_t_max 20
```

**Several cases in one run** (writes PNGs under each case’s `Results/<case>/noise_<level>/`):

```bash
cd src
python comparison_wsindy.py --batch --cases equipart cascade --noise_level 1.0 \
  --wsindy_only --no_standardize --triad_interaction_only --plot_wsindy --prediction_t_max 20
```

Use `--all` instead of `--cases` to run every supported triad case (`equipart`, `cascade`, `dual_cascade`, `periodic_cascade`, `random_cascade`, `random_cascade_deterministic`).

Omit `--triad_interaction_only` to use the **full** WSINDy coefficients for extended / stochastic / paired plots (fit comparison still shows full vs data).

Triad-enforced exports **keep a constant (bias) term by default** (important for affine/forced means, e.g. `dual_cascade`). Use `--no_keep_constant_terms` to drop it.

Useful flags: `--prediction_t_max`, `--stoch_paths`, `--plot_seed`, `--stoch_clip_margin` (default clips stochastic states to avoid polynomial drift overflow; `0` disables), `--paired_seed` (defaults to `--plot_seed` for paired runs).

From Python, `comparison_wsindy.run_wsindy_fit_plots("cascade", ...)` runs the same plot workflow for one case without hard-coding equipart.

Interactive mode (no `--batch`): option **3** writes `finite_expression_wsindy.txt` and `wsindy_expression_{1,2,3}.txt` using **triad-enforced** coefficients; option **4** matches `--plot_wsindy`.

### Two-Stage Training Workflow

1. **First Stage (Drift Learning)**:
   ```bash
   cd src
   python 1stage_deterministic.py --params_name cascade --DEVICE cpu
   ```

2. **Second Stage (Diffusion Learning)**:
   ```bash
   cd src
   python 2stage_stochastic_time_dependent.py
   # Follow the menu (training method, etc.)
   ```

### Testing and Development
```bash
# Example one-off check (adjust imports to what you need)
python -c "import numpy as np; print('ok')"
```

## Current Status

- **First stage**: Complete - FEX learning for drift terms working
- **Second stage**: In progress - Neural network training implemented, needs testing
- **Plotting**: Enhanced with noise level effect analysis and cross-term visualization
- **Structure**: Cleaner, more modular code organization

## Next Steps

1. **Test neural network training** with the restructured second stage
2. **Verify model saving** and loading functionality
3. **Complete prediction pipeline** for trained models
4. **Add comprehensive testing** for all components
5. **Document API** for each module

## Dependencies

Key dependencies (see `environment.yml`):
- **torch**: Deep learning framework
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing
- **Additional**: sympy, pandas, scikit-learn, jupyter