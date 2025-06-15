from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import sympy as sp
import os
from typing import List, Dict, Optional, Tuple
try:
    from .FEX import FEX
    from .helper import weights_init
except:
    from FEX import FEX
    from helper import weights_init


class MultiDimensionFEX(nn.Module):
    def __init__(self, op_seqs: dict, dimension: int):
        super().__init__()
        # Define operator sequences for each dimension
        self.op_seqs = op_seqs
        self.dim = dimension
        # Create FEX models for each dimension
        self.models = nn.ModuleDict({
            str(dim): FEX(torch.tensor(op_seq), dim=self.dim) 
            for dim, op_seq in self.op_seqs.items()
        })
        
        # Initialize weights for each model
        for model in self.models.values():
            model.apply(weights_init)
        
        # print('✅'*40)
        # print(f"Initialized MultiDimensionFEX with {self.dim} dimensions.")
        # self.get_cross_term_coefficients()
        # print('✅'*40)
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for dim in range(0+1, self.dim+1):
            model = self.models[str(dim)]
            output = model(x)
            outputs.append(output)
        outputs = [output.squeeze(-1) for output in outputs]  # remove last dim
        return torch.stack(outputs, dim=1)  
    

    def expression_visualize(self) -> Dict[str, str]:
        """
        Visualize the expressions for each dimension.
        Returns a dictionary with dimension as key and expression as value.
        """
        exprs = {}
        for idx in range(1, self.dim + 1):
            model = self.models[str(idx)]
            exprs[f'Dimension {idx}'] = model.expression_visualize_simplified()
        return exprs


class MultiDimFEXLoader(nn.Module):
    def __init__(self, model_path: str, dimension: int, device='cpu'):
        super().__init__()
        self.model_path = model_path
        self.dimension = dimension
        self.device = device

        # Load operator sequences
        self.op_seqs = {}
        for idx in range(1, dimension + 1):
            op_file = os.path.join(model_path, f'optimal_idx_{idx}.npy')
            if not os.path.exists(op_file):
                raise FileNotFoundError(f"Missing operator index file: {op_file}")
            self.op_seqs[idx] = torch.tensor(np.load(op_file, allow_pickle=True), dtype=torch.long)

        # Build FEX models
        self.models = nn.ModuleDict()
        for idx, op_seq in self.op_seqs.items():
            model = FEX(op_seq, dim=dimension)
            weight_file = os.path.join(model_path, f'FEX_dim_{idx}.pth')
            if not os.path.exists(weight_file):
                raise FileNotFoundError(f"Missing model weight file: {weight_file}")
            model.load_state_dict(torch.load(weight_file, map_location=device))
            self.models[str(idx)] = model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over all dimensions"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        x = x.to(self.device)

        outputs = []
        for idx in range(1, self.dimension + 1):
            out = self.models[str(idx)](x)
            outputs.append(out.squeeze(-1))  # remove last dim

        return torch.stack(outputs, dim=1)
    
    def expression_visualize(self) -> Dict[str, str]:
        """
        Visualize the expressions for each dimension.
        Returns a dictionary with dimension as key and expression as value.
        """
        exprs = {}
        for idx in range(1, self.dimension + 1):
            model = self.models[str(idx)]
            exprs[f'Dimension {idx}'] = model.expression_visualize_simplified()
        return exprs


def generate_fex_eval_function(model_path, dimension=3):
    """
    Loads FEX expressions and returns:
        - eval_expr(x): rounded expression evaluator
        - eval_change_expr(x): evaluates only the correction term from rounding
        - symbolic_change_exprs: dict of SymPy expressions showing the "lost" rounding delta
    """
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    f_exprs = {}
    change_exprs = {}
    trainable_terms = {}  # Store terms that need further training

    def get_closest_value(coeff):
        """Get the closest common value (integer or decimal) to the coefficient"""
        # Check common decimal values first
        common_values = [-0.8,-0.6, -0.4,-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.6,0.8]
        closest_val = round(float(coeff))
        min_diff = abs(float(coeff) - closest_val)
        
        for val in common_values:
            diff = abs(float(coeff) - val)
            if diff < min_diff:
                min_diff = diff
                closest_val = val
                
        return closest_val

    for dim in range(1, dimension + 1):
        op_file = os.path.join(model_path, f'optimal_idx_{dim}.npy')
        op_seq = torch.tensor(np.load(op_file, allow_pickle=True), dtype=torch.long)

        model = FEX(op_seq, dim=dimension)
        model.load_state_dict(torch.load(os.path.join(model_path, f'FEX_dim_{dim}.pth')))

        expr = model.expression_visualize_simplified()
        expr_sympy = sp.sympify(expr)

        rounded_terms = []
        change_terms = []
        trainable_terms[dim] = []  # List to store trainable terms for this dimension

        for term in expr_sympy.as_ordered_terms():
            coeff, rest = term.as_coeff_Mul()
            rounded_coeff = coeff  # default: no change

            # print(f"rest: {rest}")
            # Check if term is not a constant (has variables)
            if not (rest == 1):
                # Get closest value (integer or common decimal)
                rounded_val = get_closest_value(coeff)
                diff = abs(float(coeff) - rounded_val)
                # print(f"diff from {rounded_val}: {diff}")
                
                # Special cases for ground truth coefficients
                if dim == 1 and rest.has(x1):
                    rounded_coeff = coeff  # Keep original coefficient for x1
                    # print(f"keeping original coefficient for x1: {rounded_coeff}")
                elif dim == 2 and rest.has(x2):
                    rounded_coeff = coeff  # Keep original coefficient for x2
                    # print(f"keeping original coefficient for x2: {rounded_coeff}")
                elif dim == 3 and rest.has(x3):
                    rounded_coeff = coeff  # Keep original coefficient for x3
                    # print(f"keeping original coefficient for x3: {rounded_coeff}")
                # For x2*x3 in dim 1 or x1*x2 in dim 2, if diff < 0.01, keep original coefficient
                elif (dim == 1 and rest.has(x2) and rest.has(x3)) or \
                     (dim == 2 and rest.has(x1) and rest.has(x2)):
                    if diff < 0.01:
                        rounded_coeff = coeff  # Keep original coefficient
                        # print(f"keeping original coefficient for cross term: {rounded_coeff}")
                    else:
                        trainable_terms[dim].append((coeff, rest, rounded_val))
                        # print(f"Adding trainable term: {coeff}*{rest} (diff from {rounded_val}: {diff})")
                # For other terms, if diff > 0.01, mark as trainable
                elif diff > 0.01:
                    trainable_terms[dim].append((coeff, rest, rounded_val))
                    # print(f"Adding trainable term: {coeff}*{rest} (diff from {rounded_val}: {diff})")
                else:
                    rounded_coeff = coeff  # Keep original coefficient for non-trainable terms
                    # print(f"keeping original coefficient (non-trainable): {rounded_coeff}*{rest}")

            # For rounded expression, use the rounded coefficient
            rounded_terms.append(rounded_coeff * rest)
            # For change expression, use original coefficient for terms that need training
            change_terms.append(coeff * rest)

        rounded_expr = sum(rounded_terms)
        change_expr = sum(change_terms)
        change_exprs[dim] = change_expr

        print(f"\nFinal rounded expression (dim={dim}):\n{rounded_expr}")
        print(f"Change expression (dim={dim}):\n{change_expr}\n")
        print(f"Trainable terms (dim={dim}):")
        for coeff, rest, target in trainable_terms[dim]:
            print(f"  {coeff}*{rest} -> target: {target}")

        f_exprs[dim] = sp.lambdify((x1, x2, x3), rounded_expr, modules='numpy')
        change_exprs[dim] = sp.lambdify((x1, x2, x3), change_expr, modules='numpy')

    def train_coefficients(u_current, learning_rate=0.0001, epochs=200):
        """
        Train the coefficients that need further refinement using supervised learning.
        Args:
            u_current: Training data tensor
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
        """
        # Get target values from rounded expression
        target_values = eval_expr(u_current)
        
        # Store final expressions for each dimension
        final_exprs = {}
        
        for dim in range(1, dimension + 1):
            if not trainable_terms[dim]:
                continue

            print(f"\nTraining coefficients for dimension {dim}:")
            
            # Get the original expression for this dimension
            op_file = os.path.join(model_path, f'optimal_idx_{dim}.npy')
            op_seq = torch.tensor(np.load(op_file, allow_pickle=True), dtype=torch.long)
            model = FEX(op_seq, dim=dimension)
            model.load_state_dict(torch.load(os.path.join(model_path, f'FEX_dim_{dim}.pth')))
            dim_expr = sp.sympify(model.expression_visualize_simplified())
            
            # Initialize all parameters for this dimension
            params = []
            terms = []
            targets = []
            for coeff, rest, target in trainable_terms[dim]:
                param = torch.tensor(float(coeff), requires_grad=True)
                params.append(param)
                terms.append(rest)
                targets.append(target)
            
            # Create optimizer for all parameters with a smaller learning rate
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            
            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Get predictions using current parameter values
                pred_terms = []
                for param, rest in zip(params, terms):
                    # Evaluate the term using PyTorch operations
                    if rest == x1:
                        term_pred = param * u_current[:, 0]
                    elif rest == x2:
                        term_pred = param * u_current[:, 1]
                    elif rest == x3:
                        term_pred = param * u_current[:, 2]
                    elif rest == x1 * x2:
                        term_pred = param * (u_current[:, 0] * u_current[:, 1])
                    elif rest == x1 * x3:
                        term_pred = param * (u_current[:, 0] * u_current[:, 2])
                    elif rest == x2 * x3:
                        term_pred = param * (u_current[:, 1] * u_current[:, 2])
                    else:
                        raise ValueError(f"Unknown term: {rest}")
                    pred_terms.append(term_pred)
                
                # Sum all predictions
                pred = sum(pred_terms)
                
                # Calculate two losses:
                # 1. Prediction loss - how well the expression matches the target values
                pred_loss = torch.mean((pred - target_values[:, dim-1]) ** 2)
                
                # 2. Coefficient loss - how close each parameter is to its target value
                coeff_loss = torch.tensor(0.0, requires_grad=True)
                for param, target in zip(params, targets):
                    # Use a stronger penalty for deviation from target
                    diff = param - target
                    coeff_loss = coeff_loss + torch.abs(diff) + 100.0 * (diff ** 2)
                
                # Combine losses with much higher weight on coefficient loss
                loss = pred_loss + 1000.0 * coeff_loss
                
                # Compute gradients
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Project parameters to be within 0.1 of their target values
                with torch.no_grad():
                    for param, target in zip(params, targets):
                        param.clamp_(target - 0.1, target + 0.1)
                
                if epoch % 20 == 0:
                    print(f"  Epoch {epoch}:")
                    for param, rest, target in zip(params, terms, targets):
                        print(f"    Term {rest}: {param.item():.6f} (target: {target})")
                    print(f"    Prediction Loss: {pred_loss.item():.6f}")
                    print(f"    Coefficient Loss: {coeff_loss.item():.6f}")
                    print(f"    Total Loss: {loss.item():.6f}")
            
            print(f"Final parameters:")
            for param, rest, target in zip(params, terms, targets):
                print(f"    Term {rest}: {param.item():.6f} (target: {target})")
            
            # Build final expression for this dimension
            final_terms = []
            for term in dim_expr.as_ordered_terms():
                coeff, rest = term.as_coeff_Mul()
                # Check if this term was trainable
                is_trainable = False
                for train_param, train_rest, _ in zip(params, terms, targets):
                    if rest == train_rest:
                        coeff = train_param.item()
                        is_trainable = True
                        break
                final_terms.append(f"{coeff:.15f}*{rest}")
            
            final_expr = " + ".join(final_terms)
            final_exprs[dim] = final_expr
        
        # Print final expressions
        print('✅'*40)
        print("\n" + "="*50)
        print("Final Expressions After Training:")
        print("="*50)
        for dim in range(1, dimension + 1):
            if dim in final_exprs:
                print(f"\nDimension {dim}:")
                print(f"  {final_exprs[dim]}")
        print("="*50)
        print('✅'*40)

    def eval_expr(x_tensor):
        x_np = x_tensor.cpu().numpy()
        return torch.tensor(np.stack([
            f_exprs[1](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
            f_exprs[2](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
            f_exprs[3](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
        ], axis=1), dtype=torch.float32)

    def eval_change_expr(x_tensor):
        x_np = x_tensor.cpu().numpy()
        return torch.tensor(np.stack([
            change_exprs[1](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
            change_exprs[2](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
            change_exprs[3](x_np[:, 0], x_np[:, 1], x_np[:, 2]),
        ], axis=1), dtype=torch.float32)

    return (eval_expr, eval_change_expr, change_exprs, train_coefficients)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # op_seqs = {
    #     1: [1, 2, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2],
    #     2: [1, 0, 2, 2, 1, 2, 1, 2, 2, 1, 0, 2],
    #     3: [2, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2],
    # }
    # model = MultiDimensionFEX(op_seqs,len(op_seqs)).to(DEVICE)
    model_path = os.path.join('..', 'Example', 'MC_triad', 'Results', 'equipart')
    data = np.load(os.path.join(model_path, 'simulation_results.npz'))
    eval_expr, eval_change_expr, change_exprs, train_coefficients = generate_fex_eval_function(model_path, dimension=3)
    
    # Prepare data
    u1 = data['dataset'][:,0]
    u2 = data['dataset'][:,1]
    u3 = data['dataset'][:,2]
    u1_next = u1[:,1:].reshape(-1,1)
    u2_next = u2[:,1:].reshape(-1,1)
    u3_next = u3[:,1:].reshape(-1,1)
    u1_current = u1[:,:-1].reshape(-1,1)
    u2_current = u2[:,:-1].reshape(-1,1)
    u3_current = u3[:,:-1].reshape(-1,1)
    u_current = np.concatenate([u1_current,u2_current,u3_current],axis=1)
    u_next = np.concatenate([u1_next,u2_next,u3_next],axis=1)
    u_current = torch.tensor(u_current, dtype=torch.float32)
    u_next = torch.tensor(u_next, dtype=torch.float32)
    
    # Train coefficients
    train_coefficients(u_current)
    #print('✅'*40)
