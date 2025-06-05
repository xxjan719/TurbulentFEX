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
│   │   └── expression_handler.py # Symbolic expression manipulation
│   ├── models/
│   │   ├── __init__.py         # Make models a Python package
│   │   ├── fex_model.py        # FEX model implementation
│   │   └── fex_optimizer.py    # Custom optimizers for FEX
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

## Recent Updates

### Need to Update (2025-06-06)
- Finish `ODEParser.py`
- print all the dataset performance in `prediction.py`

### Updates (2025-06-05)
1. **Matrix Coefficient Extraction**
   - Implemented proper extraction of L (linear coupling), G (damping), and B (quadratic) terms
   - Fixed numerical precision issues in coefficient extraction
   - Added better handling of sympy expressions and coefficients
   - Added debugging output for coefficient verification

2. **LaTeX Visualization**
   - Added new `plot_latex_formula` function for equation visualization
   - Implemented side-by-side comparison of ground truth and FEX-learned expressions
   - Improved mathematical notation formatting

3. **Training Process Improvements**
   - Modified argument parser to handle ground truth training
   - Added logic to automatically disable second stage training when using ground truth
   - Change `Derivative-based method` to `Integration-based method` for updating.
   - Change LBFGS into the training update.
   - Changed `get_parser()` to return parsed args instead of parser object
   - Ensured `SECOND_STAGE_OPEN_BOOL` is always False when `TRAIN_GROUND_TRUTH` is True


### Updates (2025-06-04)
1. **Optimization Improvements**
   - Added LBFGS optimization as a second phase after Adam
   - Reduced LBFGS epochs from 50 to 10 for better efficiency
   - Added NaN detection and handling in LBFGS optimization

2. **Code Structure Changes**
   - Updated base path to `src/Example/{args.Model}/Results`
   - Simplified expression visualization code
   - Improved path handling for results and logs

3. **Scoring System Updates**
   - Modified scoring formula to use direct loss values
   - Removed dimension-specific loss thresholds
   - New scoring formula: `1/(1 + loss)` for all dimensions

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