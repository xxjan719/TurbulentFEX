from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import sympy as sp
import os
from typing import List, Dict, Optional, Tuple
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
        
        print('✅'*40)
        print(f"Initialized MultiDimensionFEX with {self.dim} dimensions.")
        self.get_cross_term_coefficients()
        print('✅'*40)
    
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
        For each model, print the 4-element op_seq chunk for each dimension,
        and if unary op is 2 (identity), print the corresponding coefficient.
        """

        coeffs_dict = {}
        pairs = [(i,j) for i in range(self.dim) for j in range(i+1,self.dim)]
        keys = [f'x{i+1}x{j+1}' for i, j in pairs]
        for key in keys:
            coeffs_dict[key] = []
        
        xs = sp.symbols(' '.join([f'x{i+1}' for i in range(self.dim)]))
        cross_map = {f'x{i+1}x{j+1}': xs[i]*xs[j] for i in range(self.dim) for j in range(i+1, self.dim)}
        final_coeffs = {key: 0 for key in cross_map}
        for dim in range(1, self.dim+1):
            model = self.models[str(dim)]
            _, simplied_expr = model.expression_visualize()
            # print(simplied_expr)
            coeffs_check = {}
            for key, sym in cross_map.items():
                coeff = simplied_expr.coeff(sym)
                final_coeffs[key] += coeff  # sum across models
            
            op_seq = model.op_seq
            # print(f"Model {dim} op_seq: {op_seq}")
            coefficents = 1.0
            index_set = set()
            for i in range(self.dim):
                
                op_chunk = op_seq[i*4:(i+1)*4]
                # print(f"  Dimension {i+1} op_seq chunk: {op_chunk}")
                unary_0 = op_chunk[0].item() if hasattr(op_chunk[0], 'item') else op_chunk[0]
                unary_2 = op_chunk[2].item() if hasattr(op_chunk[2], 'item') else op_chunk[2]
                unary_3 = op_chunk[3].item() if hasattr(op_chunk[3], 'item') else op_chunk[3]
                binary_0 = op_chunk[1].item() if hasattr(op_chunk[1], 'item') else op_chunk[1]
                coeff = 0
                is_constant_term = False

                if unary_3 == 1:
                    coeff = model.nonlinear_a[i][2]*1+model.nonlinear_b[i][2]
                    is_constant_term = True

                elif unary_3 ==2:
                    if (unary_0 == 2 and unary_2 !=2):
                        if binary_0 == 2 and unary_2 == 1:
                            coeff = model.nonlinear_a[i][0]*model.nonlinear_a[i][1]*model.nonlinear_a[i][2]
                        else:
                            coeff = model.nonlinear_a[i][0]*model.nonlinear_a[i][2]
        
                            
                    elif (unary_0!=2 and unary_2 ==2): 
                        if binary_0 == 2 and unary_0 == 1:
                            coeff = model.nonlinear_a[i][0]*model.nonlinear_a[i][1]*model.nonlinear_a[i][2]
                        elif binary_0 == 1:
                            coeff = -1*model.nonlinear_a[i][1]*model.nonlinear_a[i][2]
                        elif binary_0 == 0:
                            coeff = model.nonlinear_a[i][1]*model.nonlinear_a[i][2]
                            
                    elif (unary_0==2 and unary_2 ==2) and (binary_0 == 0):
                        coeff = (model.nonlinear_a[i][0]+model.nonlinear_a[i][1])*model.nonlinear_a[i][2]
                    
                    elif unary_0==1 and unary_2 == 0:
                        if binary_0 == 0:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])+(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 1:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])-(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 2:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])*(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        is_constant_term = True
                    
                    elif unary_0==1 and unary_2 == 1:
                        if binary_0 == 0:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])+(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 1:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])-(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 2:
                            coeff = (model.nonlinear_a[i][0]*1+model.nonlinear_b[i][0])*(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        is_constant_term = True
                    
                    elif unary_0 ==0 and unary_2 == 0:
                        if binary_0 == 0:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])+(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 1:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])-(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 2:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])*(model.nonlinear_a[i][1]*0+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        is_constant_term = True

                    elif unary_0 == 0 and unary_2 == 1:
                        if binary_0 == 0:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])+(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 1:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])-(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        elif binary_0 == 2:
                            coeff = (model.nonlinear_a[i][0]*0+model.nonlinear_b[i][0])*(model.nonlinear_a[i][1]*1+model.nonlinear_b[i][1])*model.nonlinear_a[i][2]+ model.nonlinear_b[i][2]
                            # print('this is constant term in this dimension')
                        is_constant_term = True

                else:
                    coeff = model.nonlinear_a[i][2]*0+model.nonlinear_b[i][2]
                    is_constant_term = True
                
                if not isinstance(coeff, torch.Tensor):
                    coeff = torch.tensor(coeff, dtype=torch.float32)

                if is_constant_term:
                    print(f"    Term for x{i+1} is constant.")
                else:
                    index_set.add(f"x{i+1}")
                    print(f"    Term for x{i+1} involves variables: {index_set}")

                coefficents *= coeff
                # print(f"    Coefficient after x{i+1}: {coefficents}, requires_grad={coefficents.requires_grad}")

                # print(f"\n==> Final combined coefficient for dimension {dim}: {coefficents}\n")

                if len(index_set) == 2:
                    var_pair = sorted(index_set)
                    key = ''.join(var_pair)  # e.g., 'x1x2'
                    if key in coeffs_dict:
                        coeffs_dict[key].append(coefficents)
                    print(f"==> Final combined coefficient for {key}: {coefficents}")
                
                
                print("\n===== Final Coefficient Dictionary =====")
            # print(coeffs_dict)    
            for key in coeffs_dict:
                unique_values = []
                for v in coeffs_dict[key]:
                    if not any(torch.allclose(v, u) for u in unique_values):
                        unique_values.append(v)
                coeffs_dict[key] = unique_values
            print("==> Final combined cross-term coefficients for symoblic:", final_coeffs)
            print('==> Final combined cross-term coefficients for callback:', coeffs_dict)
            self.B_coeffs_dict = coeffs_dict

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