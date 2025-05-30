import torch
import torch.nn as nn
from torch import Tensor
from .constant import unary_ops, binary_ops
# from .helper import weights_init
# from .trainingstep import Body4TrainIntegrationArgs
class FEX(nn.Module):
    def __init__(self, operator_sequence: Tensor) -> None:
        super().__init__()
        self.op_seq = operator_sequence
        # Define the linear element
        self.linear_a = nn.Parameter(torch.ones(3))
        self.linear_b = nn.Parameter(torch.zeros(3))
        
        # Define the non-linear element
        self.nonlinear_a = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(3)])
        self.nonlinear_b = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(3)])
        

    def unary(self, op_idx: int, x: Tensor):
        if op_idx == 0:
            return torch.zeros_like(x)
        elif op_idx == 1:
            return torch.ones_like(x)
        elif op_idx == 2:
            return x
        elif op_idx == 3:
            return torch.square(x)
        elif op_idx == 4:
            return torch.pow(x,3)
        elif op_idx == 5:
            return torch.pow(x,4)
        elif op_idx == 6:
            return torch.exp(x)
        elif op_idx == 7:
            return torch.sin(x)
        elif op_idx == 8:
            return torch.cos(x) 
        else:
            raise ValueError(f"Unary operator index {op_idx} is undefined.")
    
    def binary(self, op_idx: int, x: Tensor, y: Tensor):
        if op_idx == 0:
            return torch.add(x,y)
        elif op_idx == 1:
            return torch.sub(x,y)
        elif op_idx == 2:
            return torch.mul(x,y)
        elif op_idx == 3:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    
    def linear(self, x: Tensor) -> Tensor:
        linear_part = self.unary(2,x)
        # Apply the linear operator
        linear_output = (self.linear_a * linear_part+self.linear_b).sum(dim=-1,keepdim = True)
        return linear_output
    
    def nonlinear(self, x: Tensor) -> Tensor:
        nonlinear_outputs = []
        op_ptr = 0
        for i in range(3):
            xi = x[:,i].unsqueeze(-1)
            a = self.nonlinear_a[i]
            b = self.nonlinear_b[i]
            part1 = a[0] * self.unary(self.op_seq[op_ptr], xi) + b[0]
            part2 = a[1] * self.unary(self.op_seq[op_ptr+2], xi) + b[1]
            binary_out = self.binary(self.op_seq[op_ptr+1], part1, part2)
            out = a[2] * self.unary(self.op_seq[op_ptr+3], binary_out) + b[2]
            nonlinear_outputs.append(out)
            op_ptr += 4
        # Combine the non-linear outputs
        nonlinear_output = nonlinear_outputs[0]*nonlinear_outputs[1]*nonlinear_outputs[2]
        return nonlinear_output


    def forward(self, x: Tensor) -> Tensor:
        # linear part
        linear_output = self.linear(x)
        nonlinear_output = self.nonlinear(x)
        return linear_output + nonlinear_output
    
    def expression_visualize(self,) -> str:
        
        # Linear part
        linear_terms = []
        for i in range(3):
            a = self.linear_a[i].item()
            b = self.linear_b[i].item()
            linear_terms.append(f"{a:.4f}*x{i+1}+{b:.4f}")
        linear_expr = "+".join(linear_terms)

        # Non-linear part
        exprs = []
        op_ptr = 0
        for idx in range(3):
            a = self.nonlinear_a[idx]
            b = self.nonlinear_b[idx]
            part1 = f"{a[0].item():.4f}*{unary_ops[self.op_seq[op_ptr]].format(f'x{idx+1}')}" \
                    f"+{b[0].item():.4f}"
            part2 = f"{a[1].item():.4f}*{unary_ops[self.op_seq[op_ptr+2]].format(f'x{idx+1}')}" \
                    f"+{b[1].item():.4f}"
            binary_expr = binary_ops[self.op_seq[op_ptr+1]].format(part1, part2)
            out = f"{a[2].item():.4f}*{unary_ops[self.op_seq[op_ptr+3]].format(binary_expr)}" \
                  f"+{b[2].item():.4f}"
            exprs.append(out)
            op_ptr += 4        
        nonlinear_expr = f"({exprs[0]})*({exprs[1]})*({exprs[2]})"
        return f"({linear_expr}) + ({nonlinear_expr})",f"{linear_expr}",f"{nonlinear_expr}"

# class ThreeDimensionFEX(nn.Module):
#     def __init__(self, op_seqs: dict):
#         super().__init__()
#         # Define operator sequences for each dimension
#         self.op_seqs = op_seqs
        
#         # Create FEX models for each dimension
#         self.models = nn.ModuleDict({
#             str(dim): FEX(torch.tensor(op_seq)) 
#             for dim, op_seq in self.op_seqs.items()
#         })
        
#         # Initialize weights for each model
#         for model in self.models.values():
#             model.apply(weights_init)
    
#     def forward(self, x: Tensor) -> Tensor:
#         outputs = []
#         for dim in range(1, 4):
#             model = self.models[str(dim)]
#             output = model(x)
#             outputs.append(output)
#         return torch.stack(outputs, dim=1)
    
#     def train_step(self, dataset_tensor: Tensor, integrator, mse, FEX_LR: float, TRAIN_EPOCHS: int):
#         # Create a single optimizer for all models
#         all_params = []
#         for model in self.models.values():
#             all_params.extend(model.parameters())
#         model_optim = torch.optim.Adam(all_params, lr=FEX_LR)
        
#         for train_idx in range(TRAIN_EPOCHS):
#             model_optim.zero_grad()
            
#             # Calculate loss for each dimension
#             total_loss = 0
#             for dim in range(1, 4):
#                 model = self.models[str(dim)]
#                 integration_args = Body4TrainIntegrationArgs(
#                     y0=dataset_tensor, 
#                     integration_func=model, 
#                     index=dim
#                 )
#                 du_pred, du_target = integrator.integrate(integration_args)
#                 dim_loss = mse(du_pred, du_target)
#                 total_loss += dim_loss
            
#             # Average loss across dimensions
#             total_loss = total_loss / 3
            
#             # Backpropagate and update
#             total_loss.backward()
#             model_optim.step()
            
#             if train_idx % 10 == 0:
#                 print(f"Training index: {train_idx}, Total Loss: {total_loss.item():.6f}")
#                 print("Expressions:")
#                 for dim in range(1, 4):
#                     print(f"Dimension {dim}: {self.models[str(dim)].expression_visualize()}")
#                 print("-" * 80)
        
#         # Get final losses and expressions
#         losses = {}
#         expressions = {}
#         for dim in range(1, 4):
#             model = self.models[str(dim)]
#             integration_args = Body4TrainIntegrationArgs(
#                 y0=dataset_tensor, 
#                 integration_func=model, 
#                 index=dim
#             )
#             du_pred, du_target = integrator.integrate(integration_args)
#             loss = mse(du_pred, du_target)
#             losses[dim] = loss.item()
#             expressions[dim] = model.expression_visualize()
        
#         return losses, expressions
    
#     def get_expressions(self) -> dict:
#         return {
#             dim: model.expression_visualize() 
#             for dim, model in self.models.items()
#         }
    
#     def get_losses(self, dataset_tensor: Tensor, integrator, mse) -> dict:
#         losses = {}
#         for dim in range(1, 4):
#             model = self.models[str(dim)]
#             integration_args = Body4TrainIntegrationArgs(
#                 y0=dataset_tensor, 
#                 integration_func=model, 
#                 index=dim
#             )
#             du_pred, du_target = integrator.integrate(integration_args)
#             loss = mse(du_pred, du_target)
#             losses[dim] = loss.item()
#         return losses




