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

@dataclass
class JointTrainerParams:
    dt: float
    constraint_weight: float

@dataclass
class JointTrainerArgs:
    y0: torch.Tensor
    op_seqs: Dict[int, List[str]]
    integration_func: callable

@dataclass
class JointTrainer:
    def __init__(self, integratorParams: JointTrainerParams):
        self._integratorparams = integratorParams

    def integrate(self, integrationArgs: JointTrainerArgs) -> Tensor:
        y0 = integrationArgs.y0
        return y0

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

    # def train_step(self, dataset_tensor: Tensor, integrator, mse, FEX_LR: float, TRAIN_EPOCHS: int):
    #     # Create a single optimizer for all models
    #     all_params = []
    #     for model in self.models.values():
    #         all_params.extend(model.parameters())
    #     model_optim = torch.optim.Adam(all_params, lr=FEX_LR)
        
    #     for train_idx in range(TRAIN_EPOCHS):
    #         model_optim.zero_grad()
            
    #         # Calculate loss for each dimension
    #         total_loss = 0
    #         for dim in range(1, 4):
    #             model = self.models[str(dim)]
    #             integration_args = Body4TrainIntegrationArgs(
    #                 y0=dataset_tensor, 
    #                 integration_func=model, 
    #                 index=dim
    #             )
    #             du_pred, du_target = integrator.integrate(integration_args)
    #             dim_loss = mse(du_pred, du_target)
    #             total_loss += dim_loss
            
    #         # Average loss across dimensions
    #         total_loss = total_loss / 3
            
    #         # Add constraint loss for cross terms
    #         coeff_x1x2, coeff_x2x3, coeff_x1x3 = self.get_cross_term_coefficients()
    #         constraint_loss = constraint_weight * (coeff_x1x2 + coeff_x2x3 + coeff_x1x3) ** 2
    #         total_loss += constraint_loss
            
    #         # Backpropagate and update
    #         total_loss.backward()
    #         model_optim.step()
            
    #         if train_idx % 10 == 0:
    #             print(f"Training index: {train_idx}, Total Loss: {total_loss.item():.6f}")
    #             print(f"Cross term coefficients - x1x2: {coeff_x1x2.item():.4f}, x2x3: {coeff_x2x3.item():.4f}, x1x3: {coeff_x1x3.item():.4f}")
    #             print("Expressions:")
    #             for dim in range(1, 4):
    #                 print(f"Dimension {dim}: {self.models[str(dim)].expression_visualize()}")
    #             print("-" * 80)
        
    #     # Get final losses and expressions
    #     losses = {}
    #     expressions = {}
    #     for dim in range(1, 4):
    #         model = self.models[str(dim)]
    #         integration_args = Body4TrainIntegrationArgs(
    #             y0=dataset_tensor, 
    #             integration_func=model, 
    #             index=dim
    #         )
    #         du_pred, du_target = integrator.integrate(integration_args)
    #         loss = mse(du_pred, du_target)
    #         losses[dim] = loss.item()
    #         expressions[dim] = model.expression_visualize()
        
    #     return losses, expressions
    
    # def get_expressions(self) -> dict:
    #     return {
    #         dim: model.expression_visualize() 
    #         for dim, model in self.models.items()
    #     }
    
    # def get_losses(self, dataset_tensor: Tensor, integrator, mse) -> dict:
    #     losses = {}
    #     for dim in range(1, 4):
    #         model = self.models[str(dim)]
    #         integration_args = Body4TrainIntegrationArgs(
    #             y0=dataset_tensor, 
    #             integration_func=model, 
    #             index=dim
    #         )
    #         du_pred, du_target = integrator.integrate(integration_args)
    #         loss = mse(du_pred, du_target)
    #         losses[dim] = loss.item()
    #     return losses
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