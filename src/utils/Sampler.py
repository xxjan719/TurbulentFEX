import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

SAMPLER_EPSILON = 0.2
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pmfs: Tuple[Tensor, ...], output: Tensor):       
        for i, pmf in enumerate(pmfs):
            # print(f'this is {i} times and corresponding pmf is {pmf}')
            u = torch.rand(1, device=pmf.device)
            # print(u<SAMPLER_EPSILON)
            if u < SAMPLER_EPSILON:
                classes: int = pmf.shape[0]
                # print(classes)
                pmf_unif = torch.full((classes,), fill_value=1/classes, device=pmf.device)
                # print(pmf_unif)
                output[i] = torch.multinomial(pmf_unif, 1, replacement=True)
                # print(output)
            else:
                output[i] = torch.multinomial(pmf, 1, replacement=True)
                # print(output)
                
        return output

if __name__ == "__main__":
    pmfs = (
        torch.tensor([0.7, 0.1, 0.1, 0.1]),  # prefers class 0
    torch.tensor([0.1, 0.7, 0.1, 0.1]),  # prefers class 1
    torch.tensor([0.1, 0.1, 0.7, 0.1]),  # prefers class 2
    )
    output = torch.zeros(3, dtype=torch.long)
    sampler = Sampler()
    sampled = sampler(pmfs,output)