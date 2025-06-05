# TurbulentFEX

## Changelog

### 2025-06-05 Updates
- For energy descent, the result is turning to linear case. And it is not good. We can just use single formula to compute the results.
- For the latex results, I have already updated for equipart case.
- My simplified expression in `FEX` class may have some problem when calculating. So I compute it by myself.


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