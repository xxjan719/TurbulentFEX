# TurbulentFEX: Feature Expression Model for Turbulent Systems

## Description
```bash
TurbulentFEX/
│
├── src/
│   ├── __init__.py             # Make src a Python package
│   ├── main.py                 # Main training and evaluation script
│   ├── utils/
│   │   ├── __init__.py         # Make utils a Python package
│   │   ├── lawtrainingstep.py  # Training step implementations
│   │   ├── coefficient_extractor.py  # Matrix coefficient extraction utilities
│   │   ├── visualization.py     # LaTeX and plot generation utilities
│   │   ├── plotting.py          # Plotting utilities for results
│   │   └── expression_handler.py # Symbolic expression manipulation
│   ├── models/
│   │   ├── __init__.py         # Make models a Python package
│   │   ├── fex_model.py        # FEX model implementation
│   │   └── fex_optimizer.py    # Custom optimizers for FEX
│   ├── Example/
│   │   ├── MC_triad/           # Example: Monte Carlo triad system
│   │   └── prediction.ipynb    # Prediction and evaluation notebook
│   └── examples/
│       └── turbulent_cases/    # Example turbulent system cases
│
├── config/
│   ├── __init__.py
│   ├── arg_parser.py           # Command line argument configuration
│   └── model_config.py         # Model hyperparameters and settings
│
├── results/                    # Training results and visualizations
│   ├── equations/              # Generated LaTeX equations
│   ├── coefficients/           # Extracted matrix coefficients
│   └── plots/                  # Performance plots and visualizations
│
└── .gitignore
```

## Issues


## Recent Updates

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
- torch (2.0.0): Deep learning framework
- sympy (1.12): Symbolic mathematics
- numpy (1.21.0): Numerical computations
- matplotlib (3.7.0): Plotting and visualization
- latex2sympy2 (1.8.3): LaTeX parsing
- pandas (1.3.0): Data manipulation

## Usage
```python
# Training with ground truth
python src/main.py --TRAIN_GROUND_TRUTH True --SECOND_STAGE_OPEN_BOOL False

# Two-stage training
python src/main.py --TRAIN_GROUND_TRUTH False --SECOND_STAGE_OPEN_BOOL True

# Coefficient extraction
from src.utils.coefficient_extractor import extract_coefficients

coefficients = extract_coefficients(expression)
L, G, B = coefficients.L, coefficients.G, coefficients.B

# LaTeX visualization
from src.utils.visualization import plot_latex_formula

plot_latex_formula(ground_truth, predicted, save_path="results/equations/comparison.png")
```

## Comments
- Variable names: Follow Python naming conventions
- Code structure: Modular design with clear separation of concerns
- Documentation: Comprehensive docstrings and comments
- Configuration: Centralized argument parsing and model settings
- Error handling: Robust error checking and debugging output
- Results organization: Structured output directory for equations, coefficients, and plots