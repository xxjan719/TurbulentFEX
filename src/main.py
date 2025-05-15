from utils import FEX
import torch
import numpy as np
import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

op_seqs = [2,0,3,2,
            4,2,5,2,
            6,1,7,2]
model = FEX(op_seqs)
x = torch.randn(224, 3)
# print(x)
y = model(x)
print(y)
print(model.expression_visualize(x))