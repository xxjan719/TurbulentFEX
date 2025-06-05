# TurbulentFEX

## Changelog

### 2025-06-05 Updates
1. **Matrix Coefficient Extraction**:
   - Implemented proper coefficient extraction from FEX expressions
   - Added support for extracting L (linear coupling), G (damping), and B (quadratic) terms
   - Fixed numerical precision issues in coefficient extraction

2. **LaTeX Visualization**:
   - Added new `plot_latex_formula` function to visualize equations
   - Side-by-side comparison of ground truth and FEX-learned expressions
   - Improved formatting with proper mathematical notation

3. **Expression Handling**:
   - Fixed issues with sympy expression parsing and coefficient extraction
   - Added better handling of numeric coefficients in expressions
   - Improved debugging output for coefficient extraction

4. **Code Structure**:
   - Reorganized matrix construction for better clarity
   - Updated coefficient storage format (L: 3x3 matrix, G: vector, B: vector)
   - Added detailed documentation and type hints

5. **Bug Fixes**:
   - Fixed sign conventions in matrix L construction
   - Corrected coefficient extraction for quadratic terms
   - Improved error handling in symbolic computations

### 2024-06-04 Updates
1. **Optimization Improvements**:
   - Added LBFGS optimization as a second phase after Adam
   - Reduced LBFGS epochs from 50 to 10 for better efficiency
   - Added NaN detection and handling in LBFGS optimization

2. **Code Structure Changes**:
   - Updated base path to `src/Example/{args.Model}/Results`
   - Simplified expression visualization code
   - Removed simplified expression output to reduce complexity

3. **Scoring System Updates**:
   - Modified scoring formula to use direct loss values
   - Removed dimension-specific loss thresholds
   - New scoring formula: `1/(1 + loss)` for all dimensions

4. **Error Handling**:
   - Added try-catch blocks for LBFGS optimization
   - Improved error reporting during training
   - Added safeguards against NaN values in loss computation

5. **Path Management**:
   - Updated directory structure for better organization
   - Added automatic directory creation for results
   - Improved path handling for data, logs, and figures