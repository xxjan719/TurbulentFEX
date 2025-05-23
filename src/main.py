import sys
sys.path.append("..")  # So you can import from top-level utils

from utils import FEX
from utils.plotting import plot_stats, plot_third_order_moments,plot_deviation_subplots
from utils.constant import *
from utils.helper import logprint,adjust_learning_rate
from utils.controller import Controller
from utils.Sampler import Sampler
from utils.Pool import Pool
from utils.trainingstep import Body4TrainIntegrationParams, Body4TrainIntegrationArgs, Body4TrainIntegrator
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
from config.arg_parser import get_parser
import torch
import torch.nn as nn
import math
import numpy as np
import os
import random
import logging

parser = get_parser()
args = parser.parse_args()
base_path = f'Example/{args.Model}/results'
if args.data_save_path is None:
    args.data_save_path = f'{base_path}/simulation_results.npz'
if args.log_save_path is None:
    args.log_save_path = base_path
if args.figure_save_path is None:
    args.figure_save_path = base_path


DEVICE = args.DEVICE
SEED = args.SEED
PMF_SIZES = [len(unary_ops),len(binary_ops),len(unary_ops),len(binary_ops)]*3
NUM_NODES = len(PMF_SIZES)
NUM_TREES = args.NUM_TREES

CONTROLLER_LR = args.CONTROLLER_LR
CONTROLLER_INPUT_SIZE = args.CONTROLLER_INPUT_SIZE
CONTROLLER_TOP_SAMPLES_FRACTION = args.CONTROLLER_TOP_SAMPLES_FRACTION
CONTROLLER_QUANTILE_METHOD = args.CONTROLLER_QUANTILE_METHOD
EXPLORATION_ITERS = args.EXPLORATION_ITERS

FEX_LR = args.FEX_LR
TRAIN_EPOCHS_FIRST = args.TRAIN_EPOCHS_FIRST
TRAIN_EPOCHS_SECOND = args.TRAIN_EPOCHS_SECOND



torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

m0, var0 = MC_triad_initial_value()
params = params_init(args.params_name)
data_file = args.data_save_path

if os.path.exists(data_file):
    print(f'Data has already generated, just using for the first stage training:FEX'.center(60, '='))
    data = np.load(data_file)
    dataset =  data['dataset']
    mean_MC = data['mean_MC']
    cov_MC = data['cov_MC']
    moment3_MC = data['moment3_MC']
    moment3_MC_norm = data['moment3_MC_norm']
    Energy_MC = data['Energy_MC']
    Energy_dyn = data['Energy_dyn']
else:
    print(f'There is no dataset in this environment, it generates automatically'.center(60,'-'))
    dataset, mean_MC, cov_MC, moment3_MC, moment3_MC_norm,Energy_MC, Energy_dyn = MC_triad_direct(params, m0, var0)
    np.savez(
    args.data_save_path,
    dataset=dataset,
    mean_MC=mean_MC,
    cov_MC=cov_MC,
    moment3_MC=moment3_MC,
    moment3_MC_norm=moment3_MC_norm,
    Energy_MC=Energy_MC,
    Energy_dyn=Energy_dyn
    )
    print(f'Right now it is ok for data. We use it for the first stage training: FEX'.center(60,'='))

dataset_tensor = torch.from_numpy(dataset).float()
controller = Controller(pmf_sizes=PMF_SIZES).to(DEVICE)
controller_optim = torch.optim.Adam(controller.parameters(), CONTROLLER_LR)
sampler = Sampler()
mse = nn.MSELoss()
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'],)
integrator = Body4TrainIntegrator(integratorParams)
pool = Pool()
# print(dataset.shape)

for dim in range(0+1,3+1):
    model_save_path = os.path.join(args.log_save_path, f"optimal_FEX_{dim}.pth")
    optimal_idx_path = os.path.join(os.path.dirname(args.data_save_path), f"optimal_idx_{dim}.npy")

    if os.path.exists(model_save_path) and os.path.exists(optimal_idx_path):
        print(f'Model for dimension {dim} has already generated, just using for the second stage training:FEX'.center(60, '='))
        optimal_idx = np.load(optimal_idx_path)
        model_optimal = FEX(optimal_idx)
        model_optimal.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        model_optimal.eval()
    else:
        print(f'There is no model for dimension {dim} in this environment, it generates automatically'.center(60,'-'))
        log_file = os.path.join(args.log_save_path, f'log_dimension_{dim}.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  
            logging.StreamHandler(sys.stdout)])


        for explore_idx in range(EXPLORATION_ITERS):
            logprint(f' Exploration index: {explore_idx} '.center(60, '='))
            controller_optim.zero_grad()
            pmfs = controller(torch.zeros(CONTROLLER_INPUT_SIZE))
            scores = torch.zeros(NUM_TREES)
            op_seqs = torch.zeros(NUM_TREES, NUM_NODES, dtype=int)
            for tree_idx in range(NUM_TREES):
                op_seqs[tree_idx, :] = sampler(pmfs, output=torch.zeros(NUM_NODES, dtype=int))
                # print(op_seqs[tree_idx,:])
                model = FEX(op_seqs[tree_idx,:])
                expression = model.expression_visualize()
                parts = expression.split(') + (')
                nonlinear_expr = parts[1].strip()
                if "x1" not in nonlinear_expr and "x2" not in nonlinear_expr and "x3" not in nonlinear_expr:
                    logprint("❌ Skipping model with trivial nonlinear expression.")
                    continue
                if ("x1" in nonlinear_expr and "x2" not in nonlinear_expr and "x3" not in nonlinear_expr)  or ("x1"not in nonlinear_expr and "x2" in nonlinear_expr and "x3" not in nonlinear_expr) or \
                ("x1"  not in nonlinear_expr and "x2" not in nonlinear_expr and "x3" in nonlinear_expr):
                    logprint("❌ Skipping model with trivial nonlinear expression.")
                    continue
                # print(f'expression: {expression}; nonlinear_expr: {nonlinear_expr}')
                model_optim = torch.optim.Adam(model.parameters(),lr=FEX_LR)
                for train_idx in range(TRAIN_EPOCHS_FIRST):
                    model_optim.zero_grad()
                    integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
                    du_pred,du_target = integrator.integrate(integration_args)
                    loss = mse(du_pred,du_target)
                    loss.backward()
                    model_optim.step()
                    # if train_idx % 10 == 0:
                        # print(f"Training index: {train_idx}, Loss: {loss.item()}")
                        # print(model.expression_visualize())
                if not math.isnan(loss.item()):
                    if dim == 1:
                        scores[tree_idx] = 1/ (1+0.1*(loss-245))
                    elif dim ==2:
                        scores[tree_idx] = 1/(1+0.1*(loss-120))
                    elif dim ==3:
                        scores[tree_idx] = 1/(1+0.1*(loss))
                else:
                    scores[tree_idx] = 0.
                logprint('✅'*40)
                logprint(f'Operator Sequence: {op_seqs[tree_idx,:].tolist()}')
                logprint(f'Final Loss: {loss.item():.6f}')
                logprint(f'Score: {scores[tree_idx]:.6f}')
                logprint(f'Final Expression look like:{model.expression_visualize()}')
                logprint('✅'*40)
                pool.add(scores[tree_idx],model,loss.item(),op_seqs[tree_idx,:].tolist())
                # print(f'Pool summaries'.center(80,'-'))
                # for candidate_ in pool:
                #     print('loss: {:.6f} | operator sequence: {} | formula: {}'.format(
                # candidate_.error,
                # [v for v in candidate_.action],
                # candidate_.expression))
                # print(f''.center(80,'-'))
            # ======= Formula (3.8), (3.9) for controller update==========================
            scores_detached = scores.detach().numpy()
            scores_upper_quantile = np.percentile(scores_detached, q=(1 - CONTROLLER_TOP_SAMPLES_FRACTION), method=CONTROLLER_QUANTILE_METHOD)
            indicator_upper_quantile = (scores_detached >= scores_upper_quantile).astype(int)
            
            sum_log_probs = torch.zeros(NUM_TREES)
            log_pmfs = [torch.log(pmf) for pmf in pmfs]
            for tree_idx, ops in enumerate(op_seqs): # loop over trees
                for pmf_idx, op in enumerate(ops): # loop over nodes
                    log_prob = log_pmfs[pmf_idx][op]
                    sum_log_probs[tree_idx] += log_prob

            scores_detached = torch.from_numpy(scores_detached)
            indicator_upper_quantile = torch.from_numpy(indicator_upper_quantile)
            
            controller_loss = -(1 / CONTROLLER_TOP_SAMPLES_FRACTION) * torch.mean((scores_detached - scores_upper_quantile) * indicator_upper_quantile * sum_log_probs) 
            controller_loss.backward() # only sum_log_probs requires autograd
            controller_optim.step()

            logprint(f'PMFs'.center(60, '-'))
            for i in range(NUM_NODES):
                logprint(f'Node {i}:{np.around(pmfs[i].detach().numpy(),decimals = 4)}')
            
            logprint(f'Final Pool'.center(60, '='))
            for candidate_ in pool:
                    logprint('loss: {:.6f} | operator sequence: {} | formula: {}'.format(
                candidate_.error,
                [v for v in candidate_.action],
                candidate_.expression))
            logprint(f''.center(60,'-'))


        logprint('✅'*40)
        logprint(f' Below is code for training FEX with a set fixed operator sequence. '.center(60, '='))   
        logprint('✅'*40)

        best_candidate = min(pool, key=lambda c: c.error)
        optimal_idx = best_candidate.action
        print(f'Optimal operator sequence: {optimal_idx}')
        model_optimal = FEX(optimal_idx)
        model_optim_optimal = torch.optim.Adam(model_optimal.parameters(),lr=FEX_LR)
        for train_idx in range(TRAIN_EPOCHS_SECOND):
            adjust_learning_rate(model_optim_optimal,train_idx,FEX_LR,TRAIN_EPOCHS_SECOND)
            model_optim_optimal.zero_grad()
            integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model_optimal, index=dim)
            du_pred,du_target = integrator.integrate(integration_args)
            loss = mse(du_pred,du_target)
            loss.backward()
            model_optim_optimal.step()
            if train_idx % 100 == 0:
                logprint(f"Training step {train_idx} | Loss: {loss.item():.6f}")

        np.save(optimal_idx_path, optimal_idx)
        torch.save(model_optimal.state_dict(), model_save_path)
        logprint(f"Model saved to {model_save_path}")
        logprint(f"Optimal operator sequence saved to {optimal_idx_path}")












# save_path = os.path.join(args.figure_save_path, 'three_comparing.pdf')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)
# plot_third_order_moments(np.arange(params['Nt']+1), moment3_MC_all,save_path)
# plot_deviation_subplots(np.arange(params['Nt']+1), cov_MC_all, moment3_MC_norm_all,save_path)

