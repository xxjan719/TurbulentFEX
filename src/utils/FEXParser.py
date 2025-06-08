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
    

    def get_cross_term_coefficients(self):
        """
        Extract cross-term coefficients and maintain gradients for training.
        Returns a dictionary with cross-term coefficients that maintain gradient information.
        """
        coeffs_dict = {}
        pairs = [(i,j) for i in range(self.dim) for j in range(i+1,self.dim)]
        keys = [f'x{i+1}x{j+1}' for i, j in pairs]
        for key in keys:
            coeffs_dict[key] = []
        
        xs = sp.symbols(' '.join([f'x{i+1}' for i in range(self.dim)]))
        cross_map = {f'x{i+1}x{j+1}': xs[i]*xs[j] for i in range(self.dim) for j in range(i+1, self.dim)}
        
        for dim in range(1, self.dim+1):
            model = self.models[str(dim)]
            simplified_expr = model.expression_visualize_simplified()
            
            # Process each cross term
            for key, sym in cross_map.items():
                coeff = simplified_expr.coeff(sym)
                if coeff != 0:
                    # Extract indices from key (e.g., 'x1x2' -> 0,1)
                    i, j = int(key[1])-1, int(key[3])-1
                    
                    # Calculate coefficient with gradient tracking
                    if dim == 1 and key == 'x2x3':  # For dimension 1, look for x2*x3 term
                        coeffs_dict[key].append(model.nonlinear_a[1][2] * model.nonlinear_a[2][2])
                    elif dim == 2 and key == 'x1x3':  # For dimension 2, look for x1*x3 term
                        coeffs_dict[key].append(model.nonlinear_a[0][2] * model.nonlinear_a[2][2])
                    elif dim == 3 and key == 'x1x2':  # For dimension 3, look for x1*x2 term
                        coeffs_dict[key].append(model.nonlinear_a[0][2] * model.nonlinear_a[1][2])
        
        self.B_coeffs_dict = coeffs_dict
        return coeffs_dict


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


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    op_seqs = {
        1: [1, 2, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2],
        2: [1, 0, 2, 2, 1, 2, 1, 2, 2, 1, 0, 2],
        3: [2, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2],
    }
    model = MultiDimensionFEX(op_seqs,len(op_seqs)).to(DEVICE)
    x = torch.randn(10**4,3).to(DEVICE)
    y_pred = model(x)
    print(f"Output shape: {y_pred.shape}")