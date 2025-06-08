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
1. Energy Conservation law might not be working.
2. I don't understand why we need to focus three moments.

## Recent Updates

### Need to Update (2025-06-08 - 2025-06-09)
- print all the dataset performance in `prediction.py`

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