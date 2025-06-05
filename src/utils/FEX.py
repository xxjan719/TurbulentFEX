import torch
import torch.nn as nn
from torch import Tensor
import sympy as sp

try:
    from .constant import unary_ops, binary_ops
except:
    from constant import unary_ops, binary_ops
# from .helper import weights_init
# from .trainingstep import Body4TrainIntegrationArgs
class FEX(nn.Module):
    def __init__(self, operator_sequence: Tensor,dim: int) -> None:
        super().__init__()
        self.op_seq = operator_sequence
        self.dim = dim
        # Define the linear element
        self.linear_a = nn.Parameter(torch.ones(dim))
        self.linear_b = nn.Parameter(torch.zeros(dim))
        
        # Define the non-linear element
        self.nonlinear_a = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(dim)])
        self.nonlinear_b = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(dim)])
        

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
        linear_part = x
        linear_output = (self.linear_a.to(x.device) * linear_part + self.linear_b.to(x.device)).sum(dim=-1,keepdim = True)
        return linear_output
    
    def nonlinear(self, x: Tensor) -> Tensor:
        nonlinear_outputs = []
        op_ptr = 0
        for i in range(self.dim):
            xi = x[:, i:i+1]
            a = self.nonlinear_a[i].to(x.device)
            b = self.nonlinear_b[i].to(x.device)
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
        for i in range(self.dim):
            a = self.linear_a[i].item()
            b = self.linear_b[i].item()
            linear_terms.append(f"{a:.4f}*x{i+1}+{b:.4f}")
        self.linear_expr = "+".join(linear_terms)

        # Non-linear part
        exprs = []
        op_ptr = 0
        for idx in range(self.dim):
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
        self.nonlinear_expr = f"({exprs[0]})*({exprs[1]})*({exprs[2]})"

        expr_str = f"({self.linear_expr}) + ({self.nonlinear_expr})"

        
        return expr_str
    
    def expression_visualize_simplified(self,) -> str:
        # Try to split and simplify each part (linear and nonlinear)    
        print(self.nonlinear_expr)

        #return total_expr








