import torch
import torch.nn as nn
from torch import Tensor
import sympy as sp
import numpy as np
try:
    from .constant import unary_ops, binary_ops
except:
    from constant import unary_ops, binary_ops
# from .helper import weights_init
# from .trainingstep import Body4TrainIntegrationArgs
class FEX_with_force(nn.Module):
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
        
        # Define the force element
        self.force_a = nn.Parameter(torch.ones(1))
        self.force_b = nn.Parameter(torch.ones(1))
        
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

    def force(self, t: Tensor) -> Tensor:
        force_output_first = self.force_a.to(t.device) * t + self.force_b.to(t.device)
        force_output_final = self.unary(self.op_seq[-1], force_output_first)
        return force_output_final.unsqueeze(-1)  # Add dimension to match other outputs

    def forward(self, x: Tensor) -> Tensor:
        # linear part
        t_variable = x[:, -1:].squeeze(-1)
        x_variable = x[:, :-1]
        linear_output = self.linear(x_variable)
        nonlinear_output = self.nonlinear(x_variable)
        force_output = self.force(t_variable)
        return linear_output + nonlinear_output + force_output
    
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
        
        # Force part - apply unary operator to the entire force expression
        force_linear = f"{self.force_a.item():.4f}*t + {self.force_b.item():.4f}"
        self.force_expr = unary_ops[self.op_seq[-1]].format(force_linear)
        expr_str = f"({self.linear_expr}) + ({self.nonlinear_expr}) + ({self.force_expr})"
        return expr_str
    
    def expression_visualize_simplified(self,) -> str:
        # Try to split and simplify each part (linear, nonlinear, and force)    
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
        
        # Convert all expressions to sympy for proper expansion
        nonlinear_sympy = sp.sympify(nonlinear_expr)
        linear_sympy = sp.sympify(self.linear_expr)
        force_sympy = sp.sympify(self.force_expr)
        
        # Expand each part separately, but handle exponential functions specially
        nonlinear_expanded = sp.expand(nonlinear_sympy)
        linear_expanded = sp.expand(linear_sympy)
        
        # For force term, don't expand if it contains time variable or exponential functions
        if 't' in str(force_sympy) or 'exp(' in str(force_sympy):
            force_expanded = force_sympy  # Keep time-dependent and exponential terms as is
        else:
            force_expanded = sp.expand(force_sympy)

        # print("\nExpanded nonlinear term:")
        # print(nonlinear_expanded)
        
        # Combine and expand final expression (linear + nonlinear + force)
        final_expr = linear_expanded + nonlinear_expanded + force_expanded
        final_expanded = sp.expand(final_expr)
        
        # print("\nFinal expanded expression:")
        # print(final_expanded)  
        return str(final_expanded)
    

def FEX_with_force_dim1_ground_truth_periodic_cascade(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    t  =  x[:, -1:].squeeze(-1)
    # du1/dt = -G[0,0]*u1 + B[0]*u2*u3 + sin(2π/8 * t)
    # G[0,0] = 1, B[0] = 2
    return -1*x1 + 0*x2+ 0*x3 +2*x2*x3 + np.sin(2*np.pi/8 * t)

def FEX_with_force_dim2_ground_truth_periodic_cascade(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    t  =  x[:, -1:].squeeze(-1)
    # du2/dt = -G[1,1]*u2 + B[1]*u3*u1 + sin(2π/8 * t)
    # G[1,1] = 2, B[1] = -1
    return -2*x2 + 0*x1+ 0*x3 +(-1)*x3*x1 + np.sin(2*np.pi/8 * t)

def FEX_with_force_dim3_ground_truth_periodic_cascade(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    t  =  x[:, -1:].squeeze(-1)
    # du3/dt = -G[2,2]*u3 + B[2]*u1*u2 + sin(2π/8 * t)
    # G[2,2] = 2, B[2] = -1
    return -2*x3 + 0*x1+ 0*x2 +(-1)*x1*x2 + np.sin(2*np.pi/8 * t) 

def FEX_with_force_ground_truth_periodic_cascade(x):
    return np.stack([
        FEX_with_force_dim1_ground_truth_periodic_cascade(x), 
        FEX_with_force_dim2_ground_truth_periodic_cascade(x), 
        FEX_with_force_dim3_ground_truth_periodic_cascade(x)
    ], axis=1)




# Global cache for expressions to avoid reading file multiple times
_expression_cache = {}

def FEX_with_force_model_learned(x, 
             model_name =  'MC_triad',
             params_name = 'equipart',
             noise_level = 1.0,
             device = 'CPU'):
    """
    Create learned FEX_with_force model by reading final expressions from file.
    
    Args:
        x: Input tensor of shape (batch_size, 3)
        noise_level: Noise level to determine which file to read
    
    Returns:
        Output tensor of shape (batch_size, 3) with learned expressions
    """
    import os
    import re
    
    # Extract dimensions from input
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    t  =  x[:, -1:].squeeze(-1)
    
    # Construct path to final_expressions.txt
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if str(device) == 'cuda:0':
        expr_file = os.path.join(base_dir, "Example", model_name, "Results", "Results1", "Results", params_name, f"noise_{noise_level}", "deter1000","final_expressions.txt")
    else:
        expr_file = os.path.join(base_dir, "Example", model_name, "Results", params_name, f"noise_{noise_level}", "deter1000","final_expressions.txt")
    
    # print(device)
    # print(expr_file)
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"Final expressions file not found: {expr_file}")
    
    # Check if expressions are already cached for this configuration
    cache_key = f"{model_name}_{params_name}_noise_{noise_level}"
    if cache_key not in _expression_cache:
        # Read the expressions from file
        expressions = {}
        with open(expr_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\n[INFO] Reading learned expressions from: {expr_file}")
        print("="*60)
        print("LEARNED FEX EXPRESSIONS:")
        print("="*60)
            
        for line in lines:
            if line.startswith('dimension_'):
                # Parse dimension and expression
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    dim_name = parts[0]
                    expr_str = parts[1]
                    expressions[dim_name] = expr_str
                    print(f"{dim_name}: {expr_str}")
        
        print("="*60)
        
        if not expressions:
            raise ValueError(f"No expressions found in {expr_file}")
        
        # Cache the expressions
        _expression_cache[cache_key] = expressions
    else:
        # Use cached expressions (no printout)
        expressions = _expression_cache[cache_key]
    
    # Create the learned model outputs
    outputs = []
    
    # Process each dimension
    for dim in range(1, 4):
        dim_key = f'dimension_{dim}'
        if dim_key not in expressions:
            raise ValueError(f"Expression for {dim_key} not found in file")
        
        expr_str = expressions[dim_key]
        
        # The expressions already use x1, x2, x3, t directly - no need to replace
        # Create local variables for evaluation
        x1_tensor = x1
        x2_tensor = x2
        x3_tensor = x3
        t_tensor = t
        
        # Evaluate the expression
        try:
            # Use numpy operations for compatibility
            x1_np = x1.detach().cpu().numpy() if hasattr(x1, 'detach') else x1
            x2_np = x2.detach().cpu().numpy() if hasattr(x2, 'detach') else x2
            x3_np = x3.detach().cpu().numpy() if hasattr(x3, 'detach') else x3
            t_np = t.detach().cpu().numpy() if hasattr(t, 'detach') else t
            
            # Replace variables in expression using word boundaries to avoid conflicts
            import re
            expr_np = re.sub(r'\bx1\b', 'x1_np', expr_str)
            expr_np = re.sub(r'\bx2\b', 'x2_np', expr_np)
            expr_np = re.sub(r'\bx3\b', 'x3_np', expr_np)
            expr_np = re.sub(r'\bt\b', 't_np', expr_np)
            
            # Import necessary mathematical functions for evaluation
            import numpy as np
            import math
            
            # Create a safe evaluation context with mathematical functions
            safe_dict = {
                'x1_np': x1_np,
                'x2_np': x2_np, 
                'x3_np': x3_np,
                't_np': t_np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e,
                'np': np
            }
            
            # Evaluate the expression with safe context
            result = eval(expr_np, {"__builtins__": {}}, safe_dict)
            
            # Convert back to tensor if needed
            if hasattr(x1, 'detach'):
                result = torch.tensor(result, dtype=x1.dtype, device=x1.device)
            
            outputs.append(result)
            
        except Exception as e:
            print(f"Error evaluating expression for {dim_key}: {expr_str}")
            print(f"Error: {e}")
            raise
    
    # Stack outputs to create (batch_size, 3) tensor
    if hasattr(x1, 'detach'):
        return torch.stack(outputs, dim=1)
    else:
        return np.stack(outputs, axis=1)


if __name__ == "__main__":
    import os
    # the first 12 are the same as FEX, the last one is the force
    op_seqs = torch.tensor([1, 1, 2, 2, 1, 2, 2, 2, 8, 1, 5, 1, 6])
    fex = FEX_with_force(op_seqs,3)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # model_path = os.path.join(base_dir, "Example", "MC_triad", "Results", "equipart", "FEX_dim_1.pth")
    # if os.path.exists(model_path):
    #     fex.load_state_dict(torch.load(model_path))
    print(fex.expression_visualize())
    print(fex.expression_visualize_simplified())