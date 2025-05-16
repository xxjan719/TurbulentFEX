from utils import FEX
from utils.plotting import plot_stats
from Example.MC_triad.MC_triad import params_init, MC_triad_direct
import torch
import numpy as np
import os
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


m0 = np.array([-1,0.5,-0.5])
var0 = np.array([0.52, 0.2, 0.12])
params = params_init('equipart')
u_all, mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn = MC_triad_direct(params, m0, var0)


op_seqs = [2,0,3,2,
            4,2,5,2,
            6,1,7,2]
model = FEX(op_seqs)
x = torch.randn(224, 3)
# print(x)
y = model(x)
print(y)
print(model.expression_visualize(x))

save_path = 'Example/MC_triad/results/Energy_saving.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)