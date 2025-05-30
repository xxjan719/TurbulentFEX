
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import os
from typing import List, Dict, Optional, Tuple
from .FEX import FEX
from .helper import weights_init
from .trainingstep import Body4TrainIntegrationArgs

@dataclass
class JointTrainerParams:
    lr: float
    constraint_weight: float

@dataclass
class JointTrainerArgs:


class ThreeDimensionFEX(nn.Module):
    def __init__(self, op_seqs: dict):
        super().__init__()
        # Define operator sequences for each dimension
        self.op_seqs = op_seqs
        
        # Create FEX models for each dimension
        self.models = nn.ModuleDict({
            str(dim): FEX(torch.tensor(op_seq)) 
            for dim, op_seq in self.op_seqs.items()
        })
        
        # Initialize weights for each model
        for model in self.models.values():
            model.apply(weights_init)
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for dim in range(1, 4):
            model = self.models[str(dim)]
            output = model(x)
            outputs.append(output)
        return torch.stack(outputs, dim=1)
    
    def train_step(self, dataset_tensor: Tensor, integrator, mse, FEX_LR: float, TRAIN_EPOCHS: int):
        # Create a single optimizer for all models
        all_params = []
        for model in self.models.values():
            all_params.extend(model.parameters())
        model_optim = torch.optim.Adam(all_params, lr=FEX_LR)
        
        for train_idx in range(TRAIN_EPOCHS):
            model_optim.zero_grad()
            
            # Calculate loss for each dimension
            total_loss = 0
            for dim in range(1, 4):
                model = self.models[str(dim)]
                integration_args = Body4TrainIntegrationArgs(
                    y0=dataset_tensor, 
                    integration_func=model, 
                    index=dim
                )
                du_pred, du_target = integrator.integrate(integration_args)
                dim_loss = mse(du_pred, du_target)
                total_loss += dim_loss
            
            # Average loss across dimensions
            total_loss = total_loss / 3
            
            # Backpropagate and update
            total_loss.backward()
            model_optim.step()
            
            if train_idx % 10 == 0:
                print(f"Training index: {train_idx}, Total Loss: {total_loss.item():.6f}")
                print("Expressions:")
                for dim in range(1, 4):
                    print(f"Dimension {dim}: {self.models[str(dim)].expression_visualize()}")
                print("-" * 80)
        
        # Get final losses and expressions
        losses = {}
        expressions = {}
        for dim in range(1, 4):
            model = self.models[str(dim)]
            integration_args = Body4TrainIntegrationArgs(
                y0=dataset_tensor, 
                integration_func=model, 
                index=dim
            )
            du_pred, du_target = integrator.integrate(integration_args)
            loss = mse(du_pred, du_target)
            losses[dim] = loss.item()
            expressions[dim] = model.expression_visualize()
        
        return losses, expressions
    
    def get_expressions(self) -> dict:
        return {
            dim: model.expression_visualize() 
            for dim, model in self.models.items()
        }
    
    def get_losses(self, dataset_tensor: Tensor, integrator, mse) -> dict:
        losses = {}
        for dim in range(1, 4):
            model = self.models[str(dim)]
            integration_args = Body4TrainIntegrationArgs(
                y0=dataset_tensor, 
                integration_func=model, 
                index=dim
            )
            du_pred, du_target = integrator.integrate(integration_args)
            loss = mse(du_pred, du_target)
            losses[dim] = loss.item()
        return losses