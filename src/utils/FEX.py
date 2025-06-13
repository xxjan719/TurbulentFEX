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
        self.linear_b = nn.Parameter(torch.ones(dim))
        
        # Define the non-linear element
        self.nonlinear_a = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(dim)])
        self.nonlinear_b = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(dim)])
        

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
            part1 = f"{a[0].item():.4f}*({unary_ops[self.op_seq[op_ptr]].format(f'x{idx+1}')})" \
                    f"+{b[0].item():.4f}"
            part2 = f"{a[1].item():.4f}*({unary_ops[self.op_seq[op_ptr+2]].format(f'x{idx+1}')})" \
                    f"+{b[1].item():.4f}"
            binary_expr = binary_ops[self.op_seq[op_ptr+1]].format(part1, part2)
            out = f"{a[2].item():.4f}*({unary_ops[self.op_seq[op_ptr+3]].format(binary_expr)})" \
                  f"+{b[2].item():.4f}"
            exprs.append(out)
            op_ptr += 4
        self.exprs_0 = exprs[0]
        self.exprs_1 = exprs[1]
        self.exprs_2 = exprs[2]
        self.nonlinear_expr = f"({exprs[0]})*({exprs[1]})*({exprs[2]})"

        expr_str = f"({self.linear_expr}) + ({self.nonlinear_expr})"
        return expr_str
    
    def expression_visualize_simplified(self,) -> str:
        # Try to split and simplify each part (linear and nonlinear)    
        self.expression_visualize()
        #print("Nonlinear expression:")
        #print(self.nonlinear_expr)
        
        # Remove outer parentheses and split by )*(
        parts = self.nonlinear_expr.strip('()').split(')*(')
        
        # Function to check if expression is constant (no x1,x2,x3)
        def is_constant_expr(expr):
            return all(f'x{i+1}' not in expr for i in range(self.dim))
        
        # Function to evaluate constant expression
        def eval_constant_expr(expr):
            # Replace mathematical operations with Python syntax
            expr = expr.replace('^', '**')
            try:
                return f"{eval(expr):.4f}"
            except:
                return expr

        # Process each expression
        exprs = [self.exprs_0, self.exprs_1, self.exprs_2]
        simplified_exprs = []
        
        #print("\nSimplified parts:")
        for i, expr in enumerate(exprs):
            if is_constant_expr(expr):
                result = eval_constant_expr(expr)
                #print(f"Part {i+1}: {expr} = {result} (constant)")
                simplified_exprs.append(result)
            else:
                #print(f"Part {i+1}: {expr} (contains variables)")
                simplified = sp.expand(expr)
                #print(f"Part {i+1} simplified: {simplified}")
                simplified_exprs.append(simplified)
        
        # Combine the parts
        nonlinear_expr = f"({simplified_exprs[0]})*({simplified_exprs[1]})*({simplified_exprs[2]})"
        # print("\nNonlinear before expansion:")
        # print(nonlinear_expr)
        
        # Convert both expressions to sympy for proper expansion
        nonlinear_sympy = sp.sympify(nonlinear_expr)
        linear_sympy = sp.sympify(self.linear_expr)
        
        # Expand each part separately
        nonlinear_expanded = sp.expand(nonlinear_sympy)
        linear_expanded = sp.expand(linear_sympy)
        
        # print("\nExpanded nonlinear term:")
        # print(nonlinear_expanded)
        
        # Combine and expand final expression
        final_expr = linear_expanded + nonlinear_expanded
        final_expanded = sp.expand(final_expr)
        
        # print("\nFinal expanded expression:")
        # print(final_expanded)  
        return str(final_expanded)
    

    def get_all_linear_nonlinear_coeffs_autograd(self, dim,x_point=None):
        """
        Returns:
            linear_coeffs: list of tensors (requires_grad=True)
            linear_biases: list of tensors (requires_grad=True)
            nonlinear_grads: list of tensors (requires_grad=True), local gradient at x_point
            nonlinear_bias: tensor, nonlinear output at x_point
        """
        if x_point is None:
            x_point = torch.zeros(1, self.dim, requires_grad=True)
        else:
            x_point = x_point.clone().detach().requires_grad_(True)
        # Linear
        linear_coeffs = [self.linear_a[i] for i in range(self.dim)]
        linear_biases = [self.linear_b[i] for i in range(self.dim)]
        # Nonlinear
        nonlinear_output = self.nonlinear(x_point)
        grads = torch.autograd.grad(nonlinear_output, x_point, retain_graph=True, create_graph=True)[0]
        nonlinear_grads = [grads[0, i] for i in range(self.dim)]
        nonlinear_bias = nonlinear_output.item()
        
        coeff_x1 = linear_coeffs[0]+nonlinear_grads[0]
        coeff_x2 = linear_coeffs[1]+nonlinear_grads[1]
        coeff_x3 = linear_coeffs[2]+nonlinear_grads[2]
        # print(f'coeff_x1: {coeff_x1}, coeff_x2: {coeff_x2}, coeff_x3: {coeff_x3}')
        # print('requires_grad: ',coeff_x1.requires_grad,coeff_x2.requires_grad,coeff_x3.requires_grad)
        return coeff_x1,coeff_x2,coeff_x3
    
    def final_simplied_expression(self,):
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        final_expanded = self.expression_visualize_simplified()
        terms = final_expanded.as_coefficients_dict()
        # Build a new expression with rounded linear terms
        rounded_expr = 0
        for term, coeff in terms.items():
            # Check if the term is linear in x1, x2, or x3
            if term == x1 or term == x2 or term == x3:
                rounded_coeff = round(float(coeff), 3)
                rounded_expr += rounded_coeff * term
        else:
            rounded_expr += coeff * term

        print("Rounded expression:", rounded_expr)
       

if __name__ == "__main__":
    import os
    op_seqs = torch.tensor([1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2])
    fex = FEX(op_seqs,3)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "Example", "MC_triad", "Results", "equipart", "FEX_dim_1.pth")
    if os.path.exists(model_path):
        fex.load_state_dict(torch.load(model_path))
    print(fex.expression_visualize())
    print(fex.expression_visualize_simplified())
    # Create an input tensor with requires_grad=True
    fex.get_all_linear_nonlinear_coeffs_autograd(dim=1)

    









