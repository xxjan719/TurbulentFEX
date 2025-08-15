# TurbulentFEX: Feature Expression Model for Turbulent Systems

## Description
```bash
TurbulentFEX/
│
├── src/
│   ├── __init__.py                    # Make src a Python package
│   ├── first_stage_deterministic.py   # First stage: FEX learning drift term
│   ├── second_stage_stochastic.py     # Second stage: Learning diffusion term
│   ├── small_test.py                  # Small test script for testing functions
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
- **`first_stage_deterministic.py`**: FEX learning for drift terms (27KB, 573 lines)
- **`second_stage_stochastic.py`**: Neural network training for diffusion terms (50KB, 1073 lines)
- **`small_test.py`**: Testing script for various functions (1.6KB, 56 lines)
- **`config.py`**: Configuration management (18KB, 448 lines)

### Utility Modules (`src/utils/`)
- **`ODEParser.py`**: ODE solving and neural network training (33KB, 801 lines)
- **`plot.py`**: Comprehensive plotting utilities (51KB, 1189 lines)
- **`FEX.py`**: Functional Expansion implementation (15KB, 424 lines)
- **`helper.py`**: Helper functions (18KB, 474 lines)
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

### Two-Stage Training Workflow

1. **First Stage (Drift Learning)**:
   ```bash
   cd src
   python first_stage_deterministic.py --params_name cascade --DEVICE cpu
   ```

2. **Second Stage (Diffusion Learning)**:
   ```bash
   cd src
   python second_stage_stochastic.py
   # Choose option 1 for training
   # Choose training method (1 for single, 2 for ensemble)
   ```

### Testing and Development
```bash
# Test plotting functions
python src/small_test.py

# Test specific utilities
python -c "from src.utils.plot import plot_NOISE_LEVEL_EFFECT; print('Import successful')"
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