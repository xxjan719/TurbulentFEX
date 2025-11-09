import torch
from torch import Tensor
import torch.nn as nn
import sympy as sp

class BaseFEX(nn.Module):
    """Base class with shared unary and binary operations"""
    
    @staticmethod
    def unary(op_idx: int, x: Tensor):
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
    
    @staticmethod
    def binary(op_idx: int, x: Tensor, y: Tensor):
        if op_idx == 0:
            return torch.add(x,y)
        elif op_idx == 1:
            return torch.sub(x,y)
        elif op_idx == 2:
            return torch.mul(x,y)
        elif op_idx == 3:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    


class ForceFEX(BaseFEX):
    def __init__(self, op_seq: torch.Tensor, hardcode_ou: bool = False)-> None:
        super().__init__()
        self.op_seq = op_seq
        self.hardcode_ou = hardcode_ou  # If True, return -0.5*m directly
        self.force_weight_1 = nn.Parameter(torch.ones(1))
        self.force_weight_2 = nn.Parameter(torch.ones(1))
        self.force_weight_3 = nn.Parameter(torch.ones(1))
        self.force_bias_1 = nn.Parameter(torch.zeros(1))
        self.force_bias_2 = nn.Parameter(torch.zeros(1))
        self.force_bias_3 = nn.Parameter(torch.zeros(1))
        # For expression visualization
        self.linear_expr = ""
        self.nonlinear_expr = ""
        self.exprs_0 = ""
        self.exprs_1 = ""
        self.exprs_2 = ""

    def forward(self, x: Tensor) -> Tensor:
        # If hardcode_ou is True, directly return -0.5*m (known OU process formula)
        if self.hardcode_ou:
            theta = 0.5  # Known OU parameter
            return -theta * x
        
        # Otherwise, learn the formula using weights and biases
        # Apply weights and biases following the pattern from FEX models
        # op_seq[3] -> first unary with weight_1 and bias_1
        first_part = self.force_weight_1.to(x.device) * self.unary(self.op_seq[3], x) + self.force_bias_1.to(x.device)
        # op_seq[2] -> second unary with weight_2 and bias_2
        second_part = self.force_weight_2.to(x.device) * self.unary(self.op_seq[2], x) + self.force_bias_2.to(x.device)
        # op_seq[1] -> binary operation on the two weighted parts
        third_part = self.binary(self.op_seq[1], first_part, second_part)
        # op_seq[0] -> final unary operation
        fourth_part = self.force_weight_3.to(x.device) * self.unary(self.op_seq[0], third_part) + self.force_bias_3.to(x.device)
        return fourth_part
    
    def expression_visualize(self,) -> str:
        """Generate expression string by following the forward pass structure"""
        # If hardcoded, return the known formula
        if self.hardcode_ou:
            return "-0.5*m"
        
        # Otherwise, reconstruct the expression based on the forward pass with weights and biases
        # Forward: weight_1 * unary(op_seq[3], m) + bias_1 -> binary(op_seq[1], ...) -> unary(op_seq[0], ...)
        
        # First unary operation (op_seq[3]) with weight_1 and bias_1
        first_unary = self._op_to_str(self.op_seq[3].item(), "m")
        first = f"{self.force_weight_1.item():.4f}*({first_unary})+{self.force_bias_1.item():.4f}"
        
        # Second unary operation (op_seq[2]) with weight_2 and bias_2
        second_unary = self._op_to_str(self.op_seq[2].item(), "m")
        second = f"{self.force_weight_2.item():.4f}*({second_unary})+{self.force_bias_2.item():.4f}"
        
        # Binary operation (op_seq[1])
        third = self._binary_to_str(self.op_seq[1].item(), first, second)
        
        # Final unary operation (op_seq[0]) with weight_3 and bias_3
        third_unary = self._op_to_str(self.op_seq[0].item(), third)
        fourth = f"{self.force_weight_3.item():.4f}*({third_unary})+{self.force_bias_3.item():.4f}"
        
        return fourth
    
    def _op_to_str(self, op_idx: int, x_str: str) -> str:
        """Convert operator index to string representation"""
        if op_idx == 0:
            return "0"
        elif op_idx == 1:
            return "1"
        elif op_idx == 2:
            return x_str
        elif op_idx == 3:
            return f"({x_str})**2"
        elif op_idx == 4:
            return f"({x_str})**3"
        elif op_idx == 5:
            return f"({x_str})**4"
        elif op_idx == 6:
            return f"exp({x_str})"
        elif op_idx == 7:
            return f"sin({x_str})"
        elif op_idx == 8:
            return f"cos({x_str})"
        else:
            raise ValueError(f"Unary operator index {op_idx} is undefined.")
    
    def _binary_to_str(self, op_idx: int, x_str: str, y_str: str) -> str:
        """Convert binary operator index to string representation"""
        if op_idx == 0:
            return f"({x_str}+{y_str})"
        elif op_idx == 1:
            return f"({x_str}-{y_str})"
        elif op_idx == 2:
            return f"({x_str}*{y_str})"
        elif op_idx == 3:
            return f"({x_str}/{y_str})"
        else:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    
    def expression_visualize_simplified(self,) -> str:
        """Generate and simplify the expression"""
        expr = self.expression_visualize()
        # If hardcoded, return as is
        if self.hardcode_ou:
            return expr
        
        # Otherwise, convert to sympy and expand
        try:
            expr_sympy = sp.sympify(expr)
            expanded = sp.expand(expr_sympy)
            return str(expanded)
        except:
            return expr


class FEX_with_random_force(BaseFEX):
    def __init__(self, op_seq: torch.Tensor, dim: int = 3, Force_FEX: ForceFEX = ForceFEX, hardcode_ou: bool = False)-> None:
        super().__init__()
        # Store full operator sequence for compatibility
        self.op_seq = op_seq  # Full 16 operators (12 state + 4 force)
        # Split operator sequence: first 12 for state, last 4 for force
        self.op_seq_for_state = op_seq[:-4]  # First 12 operators
        self.op_seq_for_force = op_seq[-4:]  # Last 4 operators for force
        self.dim = dim
        self.hardcode_ou = hardcode_ou  # Pass to ForceFEX
        
        # Define the linear element
        self.linear_a = nn.Parameter(torch.ones(dim))
        self.linear_b = nn.Parameter(torch.ones(dim))
        
        # Define the non-linear element
        self.nonlinear_a = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(dim)])
        self.nonlinear_b = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(dim)])

        # Create model for m(t) - the OU process
        # This will be a simple learnable parameter that gets updated during training
        # Initialize with small random value instead of zero to ensure it's trainable
        self.m_t = nn.Parameter(torch.randn(1) * 0.01)  # Initialize m(t) with small random value
        
        # Create ForceFEX instance with last 4 operators
        self.Force_FEX = Force_FEX(self.op_seq_for_force, hardcode_ou=hardcode_ou)
        
    
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
            part1 = a[0] * self.unary(self.op_seq_for_state[op_ptr], xi) + b[0]
            part2 = a[1] * self.unary(self.op_seq_for_state[op_ptr+2], xi) + b[1]
            binary_out = self.binary(self.op_seq_for_state[op_ptr+1], part1, part2)
            out = a[2] * self.unary(self.op_seq_for_state[op_ptr+3], binary_out) + b[2]
            nonlinear_outputs.append(out)
            op_ptr += 4
        # Combine the non-linear outputs
        nonlinear_output = nonlinear_outputs[0]*nonlinear_outputs[1]*nonlinear_outputs[2]
        return nonlinear_output
    
    def forward(self, x: Tensor) -> Tensor:
        # x should have shape (batch_size, dim) where dim is the number of state variables
        batch_size = x.shape[0]
        
        # Compute linear and nonlinear outputs for state variables
        linear_output = self.linear(x)
        nonlinear_output = self.nonlinear(x)
        
        # Get m(t) value from the model parameter and create trainable expanded version
        # m_t_expanded will be a registered parameter where each element can be different and trainable
        m_t_base = self.m_t.to(x.device)  # Shape: (1,)
        
        # Create m_t_expanded as a registered parameter so optimizer can track and update it
        # Check if we already have a parameter registered for this batch size
        param_name = f'_m_t_expanded_{batch_size}'
        if not hasattr(self, param_name) or getattr(self, param_name) is None:
            # Create new parameter initialized from self.m_t with small random variation per element
            # This ensures each element can have different values and is not all zeros
            m_t_init = m_t_base.repeat(batch_size, 1).clone().detach()
            # Add small random variation to each element so they're not all the same
            m_t_init = m_t_init + torch.randn(batch_size, 1, device=x.device) * 0.01
            m_t_expanded_param = nn.Parameter(m_t_init, requires_grad=True)
            # Register it as a parameter so optimizer tracks it
            self.register_parameter(param_name, m_t_expanded_param)
            m_t_expanded = m_t_expanded_param.to(x.device)
        else:
            # Use existing parameter (optimizer is already tracking it)
            m_t_expanded = getattr(self, param_name).to(x.device)
        
        # Now m_t_expanded is a registered parameter where each element can be trained
        # When you train with (m_{t+1} - m_t)/dt loss, gradients will update m_t_expanded
        # The optimizer will automatically track and update this parameter
        
        # Apply ForceFEX to m(t) to learn how it evolves (used for OU loss: dm/dt = Force_FEX(m(t)))
        # But for state dynamics, we add m(t) itself, not dm/dt
        # force_output = self.Force_FEX(m_t_expanded)  # This is dm/dt, NOT what we need here
        
        # Combine: state dynamics + m(t) contribution
        # m(t) is added as a forcing term to each equation (the value of m(t), not its derivative)
        return linear_output + nonlinear_output + m_t_expanded
    
    def _op_to_str(self, op_idx: int, x_str: str) -> str:
        """Convert operator index to string representation"""
        if op_idx == 0:
            return "0"
        elif op_idx == 1:
            return "1"
        elif op_idx == 2:
            return x_str
        elif op_idx == 3:
            return f"({x_str})**2"
        elif op_idx == 4:
            return f"({x_str})**3"
        elif op_idx == 5:
            return f"({x_str})**4"
        elif op_idx == 6:
            return f"exp({x_str})"
        elif op_idx == 7:
            return f"sin({x_str})"
        elif op_idx == 8:
            return f"cos({x_str})"
        else:
            raise ValueError(f"Unary operator index {op_idx} is undefined.")
    
    def _binary_to_str(self, op_idx: int, x_str: str, y_str: str) -> str:
        """Convert binary operator index to string representation"""
        if op_idx == 0:
            return f"({x_str}+{y_str})"
        elif op_idx == 1:
            return f"({x_str}-{y_str})"
        elif op_idx == 2:
            return f"({x_str}*{y_str})"
        elif op_idx == 3:
            return f"({x_str}/{y_str})"
        else:
            raise ValueError(f"Binary operator index {op_idx} is undefined.")
    
    def expression_visualize(self,) -> str:
        # Linear part
        linear_terms = []
        for i in range(self.dim):
            a = self.linear_a[i].item()
            b = self.linear_b[i].item()
            linear_terms.append(f"{a:.4f}*x{i+1}+{b:.4f}")
        self.linear_expr = "+".join(linear_terms)

        # Non-linear part using state operators
        exprs = []
        op_ptr = 0
        for idx in range(self.dim):
            a = self.nonlinear_a[idx]
            b = self.nonlinear_b[idx]
            part1 = f"{a[0].item():.4f}*({self._op_to_str(self.op_seq_for_state[op_ptr].item(), f'x{idx+1}')})" \
                    f"+{b[0].item():.4f}"
            part2 = f"{a[1].item():.4f}*({self._op_to_str(self.op_seq_for_state[op_ptr+2].item(), f'x{idx+1}')})" \
                    f"+{b[1].item():.4f}"
            binary_expr = self._binary_to_str(self.op_seq_for_state[op_ptr+1].item(), part1, part2)
            out = f"{a[2].item():.4f}*({self._op_to_str(self.op_seq_for_state[op_ptr+3].item(), binary_expr)})" \
                  f"+{b[2].item():.4f}"
            exprs.append(out)
            op_ptr += 4
        self.exprs_0 = exprs[0]
        self.exprs_1 = exprs[1]
        self.exprs_2 = exprs[2]
        self.nonlinear_expr = f"({exprs[0]})*({exprs[1]})*({exprs[2]})"
        
        # Force part - m(t) is added directly to state dynamics
        # ForceFEX is used to learn dm/dt = Force_FEX(m(t)) for the OU loss
        force_evolution_expr = self.Force_FEX.expression_visualize()
        self.force_expr = f"m(t)"  # m(t) is the OU process (added directly to state dynamics)
        expr_str = f"({self.linear_expr}) + ({self.nonlinear_expr}) + m(t)  with random process evolution: dm/dt = {force_evolution_expr}"
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
        
        # Expand each part separately
        nonlinear_expanded = sp.expand(nonlinear_sympy)
        linear_expanded = sp.expand(linear_sympy)
        
        # Get m(t) evolution expression from ForceFEX (dm/dt = Force_FEX(m(t)))
        force_evolution_expr = self.Force_FEX.expression_visualize()  # Full expression with weights/biases
        force_evolution_expr_simplified = self.Force_FEX.expression_visualize_simplified()  # Simplified version
        
        # m(t) is a symbolic process (added directly to state dynamics)
        m_symbol = sp.Symbol('m')
        
        # Combine and expand final expression (linear + nonlinear + m(t))
        final_expr = linear_expanded + nonlinear_expanded + m_symbol
        final_expanded = sp.expand(final_expr)
        
        # Separate the forcing part (m(t)) from the state dynamics
        # m(t) is added as a separate term, so we can extract it
        # Collect all terms that don't contain m
        state_terms = []
        for term in sp.Add.make_args(final_expanded):
            if m_symbol not in term.free_symbols:
                state_terms.append(term)
        
        # Reconstruct state dynamics without m(t)
        if state_terms:
            state_dynamics = sp.Add(*state_terms)
        else:
            state_dynamics = sp.Integer(0)
        
        # Format output: state dynamics + forcing part separately
        state_str = str(sp.expand(state_dynamics))
        forcing_str = f"m(t)"  # The forcing term
        
        # print("\nFinal expanded expression:")
        # print(final_expanded)  
        return f"State: {state_str} + Forcing: {forcing_str} | Random process m(t) formula: {force_evolution_expr} | Simplified: {force_evolution_expr_simplified}"


if __name__ == "__main__":
    # Test ForceFEX with 4 operators
    # force_op_seq = torch.tensor([3,2,1,8])
    # force_fex = ForceFEX(force_op_seq)
    # force_x = torch.randn(10,1)
    # print("ForceFEX output:", force_fex(force_x))
    # print("ForceFEX expression:", force_fex.expression_visualize())
    # print("ForceFEX simplified:", force_fex.expression_visualize_simplified())
    
    # Test FEX_with_random_force with 16 operators (12 state + 4 force)
    print("\n" + "="*80)
    print("Testing FEX_with_random_force (state dynamics + m(t) OU process)")
    print("="*80)
    
    full_op_seq = torch.tensor([
        # 12 operators for state variables (4 per dimension: x1, x2, x3)
        1, 1, 2, 2,    # x1: [unary, binary, unary, unary]
        1, 2, 2, 2,    # x2: [unary, binary, unary, unary]  
        8, 1, 5, 1,    # x3: [unary, binary, unary, unary]
        # 4 operators for force/OU process m(t)
        3, 1, 4, 8     # m(t): [unary, binary, unary, unary]
    ])
    
    fex_with_force = FEX_with_random_force(full_op_seq, dim=3)
    
    # Input is just state variables (dim=3)
    x_state = torch.randn(10, 3)
    
    print(f"Full operator sequence: {full_op_seq.tolist()}")
    print(f"State operators (12): {fex_with_force.op_seq_for_state.tolist()}")
    print(f"Force operators (4): {fex_with_force.op_seq_for_force.tolist()}")
    print(f"\nExpression:\n{fex_with_force.expression_visualize()}")
    print(f"\nSimplified expression:\n{fex_with_force.expression_visualize_simplified()}")
    
    # Call forward first to create m_t_expanded parameter
    batch_size = x_state.shape[0]
    output = fex_with_force(x_state)  # This will create _m_t_expanded_{batch_size}
    
    # Now check m_t shape matches x batch size
    param_name = f'_m_t_expanded_{batch_size}'
    if hasattr(fex_with_force, param_name):
        m_t_expanded = getattr(fex_with_force, param_name)
        m_t = fex_with_force.m_t
        
        print(f"\nShape check:")
        print(f"  x_state shape: {x_state.shape}")
        print(f"  m_t (parameter) shape: {m_t.shape}")
        print(f"  m_t_expanded (parameter) shape: {m_t_expanded.shape}")
        print(f"  Batch size match: {m_t_expanded.shape[0] == x_state.shape[0]} (m_t_expanded batch: {m_t_expanded.shape[0]}, x batch: {x_state.shape[0]})")
        assert m_t_expanded.shape[0] == x_state.shape[0], f"m_t_expanded batch size ({m_t_expanded.shape[0]}) must match x batch size ({x_state.shape[0]})"
        print(f"  ✓ m_t_expanded has same batch size as x_state")
        print(f"  ✓ m_t_expanded is a registered parameter: {isinstance(m_t_expanded, nn.Parameter)}")
    else:
        print(f"\nWarning: Parameter {param_name} not found after forward pass")
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sample: {output[:3]}")
