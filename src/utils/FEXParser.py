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

    for dim in range(1, dimension + 1):
        op_file = os.path.join(model_path, f'optimal_idx_{dim}.npy')
        op_seq = torch.tensor(np.load(op_file, allow_pickle=True), dtype=torch.long)

        model = FEX(op_seq, dim=dimension)
        model.load_state_dict(torch.load(os.path.join(model_path, f'FEX_dim_{dim}.pth')))

        expr = model.expression_visualize_simplified()
        expr_sympy = sp.sympify(expr)

        rounded_terms = []
        change_terms = []

        for term in expr_sympy.as_ordered_terms():
            coeff, rest = term.as_coeff_Mul()
            rounded_coeff = coeff  # default: no change

            # Apply rounding and track change
            if dim == 1:
                if (rest.has(x2) and not (rest.has(x1) or rest.has(x3))) or \
                   (rest.has(x3) and not (rest.has(x1) or rest.has(x2))):
                    rounded_coeff = int(round(float(coeff)))
            elif dim == 2:
                if (rest.has(x1) and not (rest.has(x2) or rest.has(x3))) or \
                   (rest.has(x3) and not (rest.has(x1) or rest.has(x2))):
                    rounded_coeff = int(round(float(coeff)))
            elif dim == 3:
                if (rest.has(x1) and not (rest.has(x2) or rest.has(x3))) or \
                   (rest.has(x2) and not (rest.has(x1) or rest.has(x3))):
                    rounded_coeff = int(round(float(coeff)))

            # Append final expression and correction delta
            rounded_terms.append(rounded_coeff * rest)
            change_terms.append((coeff - rounded_coeff) * rest)

        rounded_expr = sum(rounded_terms)
        change_expr = sum(change_terms)
        change_exprs[dim] = change_expr

        print(f"\nFinal rounded expression (dim={dim}):\n{rounded_expr}")
        print(f"Change expression (dim={dim}):\n{change_expr}\n")

        f_exprs[dim] = sp.lambdify((x1, x2, x3), rounded_expr, modules='numpy')
        change_exprs[dim] = sp.lambdify((x1, x2, x3), change_expr, modules='numpy')

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

    return (eval_expr, eval_change_expr, change_exprs)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # op_seqs = {
    #     1: [1, 2, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2],
    #     2: [1, 0, 2, 2, 1, 2, 1, 2, 2, 1, 0, 2],
    #     3: [2, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2],
    # }
    # model = MultiDimensionFEX(op_seqs,len(op_seqs)).to(DEVICE)
    model_path=os.path.join('..','Example','MC_triad','Results','equipart')
    data = np.load(os.path.join(model_path, 'simulation_results.npz')) 
    eval_expr, eval_change_expr, change_exprs = generate_fex_eval_function(model_path, dimension=3)
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
    y_pred = eval_change_expr(u_current)
    print(f"Output: {y_pred}")