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
    """
    FEX with explicit time-forcing branch.
    - operator_sequence length must be 4*(dim + 1)
      (4 ops per state channel + 4 ops for the time/forcing channel).
    - Output is scalar: linear(x) + nonlinear(x) + forcing(t)
    """
    def __init__(self, operator_sequence: Tensor, dim: int) -> None:
        super().__init__()
        self.op_seq = operator_sequence.tolist()
        self.dim = dim

        # ---- Linear (states only) ----
        self.linear_a = nn.Parameter(torch.ones(dim))
        self.linear_b = nn.Parameter(torch.ones(dim))

        # ---- Nonlinear (per-state): each channel uses 3 'a' and 3 'b' scalars ----
        self.nonlinear_a = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(dim)])
        self.nonlinear_b = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(dim)])

        # ---- Forcing (time branch): also 3 'a' and 3 'b' scalars ----
        self.forcing_a  = nn.Parameter(torch.ones(3))
        self.forcing_b  = nn.Parameter(torch.ones(3))

        # quick sanity: op_seq length
        expected = 4*(dim + 1)
        if len(self.op_seq) != expected:
            raise ValueError(f"op_seq length {len(self.op_seq)} != 4*(dim+1) = {expected}")

    # -------- primitive ops --------
    def unary(self, op_idx: int, x: Tensor):
        if op_idx == 0: return torch.zeros_like(x)
        if op_idx == 1: return torch.ones_like(x)
        if op_idx == 2: return x
        if op_idx == 3: return x**2
        if op_idx == 4: return x**3
        if op_idx == 5: return x**4
        if op_idx == 6: return torch.exp(x)
        if op_idx == 7: return torch.sin(x)
        if op_idx == 8: return torch.cos(x)
        raise ValueError(f"Unary operator index {op_idx} is undefined.")

    def binary(self, op_idx: int, x: Tensor, y: Tensor):
        if op_idx == 0: return x + y
        if op_idx == 1: return x - y
        if op_idx == 2: return x * y
        raise ValueError(f"Binary operator index {op_idx} is undefined.")

    # -------- parts --------
    def linear(self, x: Tensor) -> Tensor:
        # x: [B, dim] -> scalar [B,1]
        a = self.linear_a.to(x.device)
        b = self.linear_b.to(x.device)
        return (a * x + b).sum(dim=-1, keepdim=True)

    def nonlinear(self, x: Tensor) -> Tensor:
        """
        Per-state mini-tree (uses 4 ops/channel), then product over all state channels.
        x: [B, dim] -> [B,1]
        """
        B = x.shape[0]
        outs = []
        op_ptr = 0
        for i in range(self.dim):
            xi = x[:, i:i+1]               # [B,1]
            a = self.nonlinear_a[i].to(x.device)  # [3]
            b = self.nonlinear_b[i].to(x.device)  # [3]

            u1 = self.unary(self.op_seq[op_ptr + 0], xi)
            bop = self.op_seq[op_ptr + 1]
            u2 = self.unary(self.op_seq[op_ptr + 2], xi)
            bin_out = self.binary(bop, a[0]*u1 + b[0], a[1]*u2 + b[1])
            out_i = a[2] * self.unary(self.op_seq[op_ptr + 3], bin_out) + b[2]  # [B,1]
            outs.append(out_i)
            op_ptr += 4

        # product over channels, robust for any dim
        nl = torch.stack(outs, dim=0).prod(dim=0) if outs else torch.ones(B,1, device=x.device)
        return nl

    def forcing_term(self, t: Tensor) -> Tensor:
        """
        Forcing uses the LAST 4 ops in op_seq on scalar time input t: [B,1] -> [B,1].
        """
        if t.dim() == 1: t = t.unsqueeze(-1)
        a = self.forcing_a.to(t.device)  # [3]
        b = self.forcing_b.to(t.device)  # [3]

        op0, op1, op2, op3 = self.op_seq[-4], self.op_seq[-3], self.op_seq[-2], self.op_seq[-1]
        u1 = self.unary(op0, t)
        u2 = self.unary(op2, t)
        bin_out = self.binary(op1, a[0]*u1 + b[0], a[1]*u2 + b[1])
        f = a[2] * self.unary(op3, bin_out) + b[2]  # [B,1]
        return f

    # -------- forward --------
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: [B, dim], t: [B,1] (typically t_omega = ω t for periodic models)
        returns scalar [B,1]
        """
        return self.linear(x) + self.nonlinear(x) + self.forcing_term(t)

    # -------- printers --------
    def expression_visualize(self) -> str:
        # Linear part
        lin_terms = []
        for i in range(self.dim):
            ai = self.linear_a[i].item()
            bi = self.linear_b[i].item()
            lin_terms.append(f"{ai:.4f}*x{i+1}+{bi:.4f}")
        self.linear_expr = "+".join(lin_terms) if lin_terms else "0"

        # Nonlinear part (per state)
        exprs = []
        op_ptr = 0
        for i in range(self.dim):
            a = self.nonlinear_a[i]
            b = self.nonlinear_b[i]
            v = f"x{i+1}"
            part1 = f"{a[0].item():.4f}*({unary_ops[self.op_seq[op_ptr+0]].format(v)})+{b[0].item():.4f}"
            part2 = f"{a[1].item():.4f}*({unary_ops[self.op_seq[op_ptr+2]].format(v)})+{b[1].item():.4f}"
            bin_expr = binary_ops[self.op_seq[op_ptr+1]].format(part1, part2)
            out = f"{a[2].item():.4f}*({unary_ops[self.op_seq[op_ptr+3]].format(bin_expr)})+{b[2].item():.4f}"
            exprs.append(out)
            op_ptr += 4

        self.nonlinear_terms = exprs
        self.nonlinear_expr = "*".join(f"({e})" for e in exprs) if exprs else "1"

        # Forcing (time) pretty string
        op0, op1, op2, op3 = self.op_seq[-4], self.op_seq[-3], self.op_seq[-2], self.op_seq[-1]
        fa = [p.item() for p in self.forcing_a]
        fb = [p.item() for p in self.forcing_b]
        f_p1 = f"{fa[0]:.4f}*({unary_ops[op0].format('t')})+{fb[0]:.4f}"
        f_p2 = f"{fa[1]:.4f}*({unary_ops[op2].format('t')})+{fb[1]:.4f}"
        f_bin = binary_ops[op1].format(f_p1, f_p2)
        f_out = f"{fa[2]:.4f}*({unary_ops[op3].format(f_bin)})+{fb[2]:.4f}"
        self.forcing_expr = f_out

        full = f"({self.linear_expr}) + ({self.nonlinear_expr}) + ({self.forcing_expr})"
        return full

    def expression_visualize_simplified(self) -> str:
        # Build full expression and expand with SymPy (includes forcing!)
        expr_str = self.expression_visualize()
        expr_sym = sp.sympify(expr_str.replace("^", "**"))
        return str(sp.expand(expr_sym))
    

    

if __name__ == "__main__":
    op_seq = torch.tensor([
    8, 1, 5, 1,   # x1 branch (placeholder)
    1, 2, 2, 2,   # x2 branch
    1, 2, 2, 2,   # x3 branch
    0, 0, 7, 2    # time branch
    ], dtype=torch.long)
    
    model = FEX_with_force(op_seq, dim=3)
    # IMPORTANT: feed t as ω t (so the branch actually sees the right frequency)
    omega = 2.0 * torch.pi / 8.0
    t = torch.linspace(0, 8.0, 128).unsqueeze(-1)        # [B,1], physical time
    t_omega = omega * t                                   # pass this to model
    x = torch.zeros(128, 3)                               # if forcing should depend on t only
    y = model(x, t_omega)                                 # [B,1]
    print(model.expression_visualize())
    print(model.expression_visualize_simplified())
    
