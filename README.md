# TurbulentFEX: Feature Expression Model for Turbulent Systems

## Description
```bash
TurbulentFEX/
│
├── src/
│   ├── __init__.py                    # Make src a Python package
│   ├── first_stage_deterministic.py   # First stage: FEX learning drift term
│   ├── second_stage_stochastic.py     # Second stage: Learning diffusion term
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
│   │   └── helper.py                  # Helper functions
│   ├── Example/
│   │   └── MC_triad/                  # Example: Monte Carlo triad system
│   │       ├── MC_triad.py            # Main triad system definition
│   │       ├── prediction.ipynb       # Prediction and evaluation notebook
│   │       └── Results/               # Output directories for results
│   │           ├── equipart/          # Equipartition results
│   │           └── cascade/           # Cascade results
│   └── README.md                      # Detailed usage documentation
│
└── .gitignore
```

## Issues


## Recent Updates

### File Reorganization (2025-01-XX)
- **Renamed main files for clarity**:
  - `main.py` → `first_stage_deterministic.py` (FEX learning drift)
  - `small_test.py` → `second_stage_stochastic.py` (learning diffusion)
- **Fixed import issues**: Resolved all relative import problems
- **Improved configuration**: Enhanced config.py with lazy loading and better error handling
- **Updated documentation**: Added comprehensive README in src/ directory

### Need to Update (2025-06-15 - 2025-06-22)
- print all the dataset performance in `prediction.py`

### Updates (2025-06-09 - 2025-06-15)
1. **Coefficient Training and Refinement**
   - Implemented a two-phase coefficient training process:
     - Phase 1: Initial FEX training to get approximate coefficients
     - Phase 2: Coefficient acceleration step to refine values closer to targets
   - Added coefficient constraints to prevent drift:
     - Parameters are clamped to be within ±0.1 of target values
     - Strong L1 and L2 penalties for deviation from targets
     - Learning rate scheduling for stable convergence

2. **Drift Term Improvements**
   - Fixed issues with linear, nonlinear, and force terms in drift calculations
   - Implemented separate handling for different term types:
     - Linear terms (x1, x2, x3): Direct coefficient optimization
     - Interaction terms (x1*x2, x1*x3, x2*x3): Special handling for cross-term effects
     - Constant terms: Preserved original values
   - Added coefficient verification to ensure physical consistency

3. **Interaction Term Optimization**
   - Developed new approach for handling interaction terms (cov(u1,u2)):
     - Interaction terms now depend on linear coefficient performance
     - Added coefficient acceleration step to improve accuracy
     - Implemented special handling for terms close to integer values
   - Results show improved accuracy in cross-term predictions

4. **Energy Conservation and Dissipation**
   - Implemented new approach for energy conservation:
     - First train with given dissipation coefficient
     - Then apply coefficient acceleration
     - Results show better stability than direct energy conservation enforcement
   - Added verification of energy conservation properties

5. **Performance Monitoring**
   - Added comprehensive performance tracking:
     - Coefficient convergence monitoring
     - Loss tracking (prediction and coefficient losses)
     - Final expression accuracy verification
   - Results available in prediction.ipynb with detailed metrics

6. **Code Improvements**
   - Enhanced coefficient training process:
     - Added parameter constraints
     - Implemented adaptive learning rates
     - Added early stopping for stable convergence
   - Improved expression formatting and output
   - Added detailed logging of training progress

### Updates (2025-06-08)
- Added `MultiDimFEX` class for unified 3D FEX prediction: loads all three FEX models (one per dimension) and returns a 3D result for input data.
- Improved plotting utilities to filter out nearly-constant lines and only plot meaningful data.
- Enhanced code structure for easier model loading and prediction.


### Updates (2025-06-06 - 2025-06-07)
- Complete implementation of `ODEParser.py` with score-based ODE solver and neural network components
- Implement FEX expression optimization system with best score tracking and expression selection capabilities
- Add comprehensive dataset performance evaluation in `prediction.py` with metrics and visualizations
- Optimize memory usage in batch processing and fix shape mismatch issues
- Add error handling for NaN values in LBFGS optimization

### Updates (2025-06-05)
1. **Matrix Coefficient Extraction**
   - Implement coefficient extraction for L, G, and B terms
   - Add coefficient verification and debugging output

2. **LaTeX Visualization**
   - Add equation visualization with ground truth comparison
   - Improve mathematical notation formatting

3. **Training Process**
   - Add ground truth training support
   - Switch to Integration-based method from Derivative-based
   - Integrate LBFGS into training pipeline
   - Update argument parser and stage control logic

### Updates (2025-06-04)
1. **Optimization**
   - Add two-phase optimization (Adam + LBFGS)
   - Optimize LBFGS parameters for efficiency
   - Add NaN detection and handling

2. **Code Structure**
   - Update paths and simplify visualization code
   - Improve results organization

3. **Scoring System**
   - Implement direct loss-based scoring
   - Use unified scoring formula: 1/(1 + loss)

### Dependencies

All dependencies are specified in `environment.yml` for easy setup. Key dependencies include:

- **torch (2.0.0)**: Deep learning framework
- **sympy (1.12)**: Symbolic mathematics
- **numpy (1.21.0)**: Numerical computations
- **matplotlib (3.7.0)**: Plotting and visualization
- **latex2sympy2 (1.8.3)**: LaTeX parsing
- **pandas (1.3.0)**: Data manipulation
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **jupyter**: Interactive notebooks
- **Additional packages**: tqdm, seaborn, plotly, wandb for enhanced functionality

## Usage

### Environment Setup

1. **Create and activate the conda environment**:
   ```bash
   # Create the environment from the yml file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate turbulentfex
   ```

2. **Alternative: Manual installation** (if conda is not available):
   ```bash
   # Create a virtual environment
   python -m venv turbulentfex_env
   
   # Activate the virtual environment
   # On macOS/Linux:
   source turbulentfex_env/bin/activate
   # On Windows:
   turbulentfex_env\Scripts\activate
   
   # Install dependencies
   pip install torch==2.0.0 torchvision torchaudio
   pip install numpy==1.21.0 matplotlib==3.7.0 pandas==1.3.0
   pip install sympy==1.12 scipy scikit-learn
   pip install latex2sympy2==1.8.3 tqdm seaborn plotly wandb
   pip install jupyter ipykernel
   ```

3. **Verify installation**:
   ```bash
   # Test that PyTorch is working
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   
   # Test that CUDA is available (if you have a GPU)
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Two-Stage Training Workflow

1. **First Stage (Drift Learning)**:
   ```bash
   cd src
   python first_stage_deterministic.py --params_name equipart --DEVICE cuda:0
   ```
   - Trains FEX models to discover the deterministic part of the dynamics
   - Uses Functional Expansion to learn the drift term
   - Outputs trained models and optimal operator sequences

2. **Second Stage (Diffusion Learning)**:
   ```bash
   cd src
   python second_stage_stochastic.py
   ```
   - Trains neural networks to model the stochastic noise component
   - Uses the drift models from first stage
   - Learns the diffusion term for complete stochastic modeling

### Configuration Options

**First Stage Parameters**:
- `--params_name`: Choose between 'equipart' or 'cascade'
- `--DEVICE`: Choose device ('cpu', 'cuda:0', 'auto')
- `--SEED`: Random seed for reproducibility
- `--FEX_LR`: Learning rate for FEX training
- `--TRAIN_EPOCHS_FIRST`: Number of training epochs

**Second Stage Parameters**:
- `--NUM_SAMPLES`: Number of samples for stochastic training
- `--DEVICE`: Choose device for training

### Legacy Usage (for reference)
```python
# Training with ground truth
python src/first_stage_deterministic.py --TRAIN_GROUND_TRUTH True --SECOND_STAGE_OPEN_BOOL False

# Two-stage training
python src/first_stage_deterministic.py --TRAIN_GROUND_TRUTH False --SECOND_STAGE_OPEN_BOOL True
```

## Comments
- Variable names: Follow Python naming conventions
- Code structure: Modular design with clear separation of concerns
- Documentation: Comprehensive docstrings and comments
- Configuration: Centralized argument parsing and model settings
- Error handling: Robust error checking and debugging output
- Results organization: Structured output directory for equations, coefficients, and plots