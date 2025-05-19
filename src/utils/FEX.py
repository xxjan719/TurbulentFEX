import torch
import torch.nn as nn
from torch import Tensor
from .constant import unary_ops, binary_ops
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
    
    def expression_visualize(self,x: Tensor) -> str:
        
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
        return f"({linear_expr}) + ({nonlinear_expr})"

    def derivative(self, x: Tensor) -> Tensor:
        """
        Compute the derivative (Jacobian) of the model output with respect to input x.
        Returns a tensor of shape (batch_size, input_dim) if output is scalar per sample.
        """
        x = x.clone().detach().requires_grad_(True)
        y = self.forward(x)
        grads = []
        for i in range(y.shape[0]):
            grad = torch.autograd.grad(y[i], x, retain_graph=True, create_graph=True, allow_unused=True)[0][i]
            grads.append(grad)
        return torch.stack(grads, dim=0)


