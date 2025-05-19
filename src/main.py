import sys
sys.path.append("..")  # So you can import from top-level utils

from utils import FEX
from utils.plotting import plot_stats, plot_third_order_moments,plot_deviation_subplots
from Example.MC_triad.MC_triad import params_init, MC_triad_direct
from config.arg_parser import get_parser
from config.paths import ensure_dir_exists
import torch
import numpy as np
import os
import random


parser = get_parser()
args = parser.parse_args()
ensure_dir_exists(args.save_path)

SEED = args.SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


m0 = np.array([-1,0.5,-0.5])
var0 = np.array([0.52, 0.2, 0.12])
params = params_init(args.params_name)
u_all, mean_MC_all, cov_MC_all, moment3_MC_all, moment3_MC_norm_all,Energy_MC_all, Energy_dyn = MC_triad_direct(params, m0, var0)

# Save all results in a single .npz file
np.savez(
    args.data_save_path,
    dataset=u_all,
    mean_MC=mean_MC_all,
    cov_MC=cov_MC_all,
    moment3_MC=moment3_MC_all,
    moment3_MC_norm=moment3_MC_norm_all,
    Energy_MC=Energy_MC_all,
    Energy_dyn=Energy_dyn
)


op_seqs = [2,0,3,2,
            4,2,5,2,
            6,1,7,2]
model = FEX(op_seqs)
x = torch.randn(224, 3)
# print(x)
y = model(x)
print(y)
print(model.expression_visualize(x))

save_path = os.path.join(args.figure_save_path, 'three_comparing.pdf')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)
# plot_third_order_moments(np.arange(params['Nt']+1), moment3_MC_all,save_path)
plot_deviation_subplots(np.arange(params['Nt']+1), cov_MC_all, moment3_MC_norm_all,save_path)

