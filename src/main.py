import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import FEX#, ThreeDimensionFEX
from utils.plotting import plot_stats, plot_third_order_moments,plot_deviation_subplots
from utils.constant import *
from utils.helper import logprint,adjust_learning_rate,plot_latex_formula,weights_init
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
import sympy as sp

parser = get_parser()
args = parser.parse_args()
base_path = f'src/Example/{args.Model}/Results'
if args.data_save_path is None:
    args.data_save_path = f'{base_path}/{args.params_name}/simulation_results.npz'
if args.log_save_path is None:
    args.log_save_path = f'{base_path}/{args.params_name}'
if args.figure_save_path is None:
    args.figure_save_path = f'{base_path}/{args.params_name}'

# Create necessary directories
os.makedirs(os.path.dirname(args.data_save_path), exist_ok=True)
os.makedirs(args.log_save_path, exist_ok=True)
os.makedirs(args.figure_save_path, exist_ok=True)

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")

SEED = args.SEED

NUM_TREES = args.NUM_TREES

CONTROLLER_LR = args.CONTROLLER_LR
CONTROLLER_INPUT_SIZE = args.CONTROLLER_INPUT_SIZE
CONTROLLER_TOP_SAMPLES_FRACTION = args.CONTROLLER_TOP_SAMPLES_FRACTION
CONTROLLER_QUANTILE_METHOD = args.CONTROLLER_QUANTILE_METHOD
EXPLORATION_ITERS = args.EXPLORATION_ITERS

FEX_LR = args.FEX_LR
TRAIN_EPOCHS_FIRST = args.TRAIN_EPOCHS_FIRST
TRAIN_EPOCHS_SECOND = args.TRAIN_EPOCHS_SECOND

INTEGRATOR_METHOD = args.INTEGRATOR_METHOD


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

dataset_tensor = torch.from_numpy(dataset).float().to(DEVICE)
dimension = dataset_tensor.shape[1]  # Assuming the second dimension is the number of features

PMF_SIZES = [len(unary_ops),len(binary_ops),len(unary_ops),len(binary_ops)]*dimension
NUM_NODES = len(PMF_SIZES)


controller = Controller(pmf_sizes=PMF_SIZES).to(DEVICE)
controller_optim = torch.optim.Adam(controller.parameters(), CONTROLLER_LR)
sampler = Sampler()
mse = nn.MSELoss()
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'],)
integrator = Body4TrainIntegrator(integratorParams,method=INTEGRATOR_METHOD)
pool = Pool()
# print(dataset.shape)
if args.TRAIN_GROUND_TRUTH == False:
    for dim in range(0+2,dimension+1):
        # model_save_path = os.path.join(args.log_save_path, f"optimal_FEX_{dim}.pth")
        log_file = os.path.join(args.log_save_path, f'log_dimension_{dim}.txt')
        if os.path.exists(log_file): #os.path.exists(model_save_path) and 
            print(f'Model for dimension {dim} has already generated, just using for the second stage training:FEX'.center(60, '='))
            # optimal_idx = np.load(optimal_idx_path)
            # print(f'dimension:{dim}, operator indx is {optimal_idx}')

        else:
            print(f'There is no model for dimension {dim} in this environment, it generates automatically'.center(60,'-'))
            
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

            # dimension 1 need 10 EXPLORATION_ITERS
            for explore_idx in range(EXPLORATION_ITERS):
                logprint(f' Exploration index: {explore_idx} '.center(60, '='))
                controller_optim.zero_grad()
                pmfs = controller(torch.zeros(CONTROLLER_INPUT_SIZE, device=DEVICE))
                scores = torch.zeros(NUM_TREES, device=DEVICE)
                # For the exploration phase, make sure op_seqs is properly on the device
                op_seqs = torch.zeros(NUM_TREES, NUM_NODES, dtype=int, device=DEVICE)
                for tree_idx in range(NUM_TREES):
                    op_seqs[tree_idx, :] = sampler(pmfs, output=torch.zeros(NUM_NODES, dtype=int, device=DEVICE))
                    # print(op_seqs[tree_idx,:])
                    model = FEX(op_seqs[tree_idx,:], dim=dimension).to(DEVICE)
                    model.apply(weights_init)
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
                        if train_idx % 10 == 0:
                            # Get the expression string from FEX
                            expr_str = model.expression_visualize()
                            print(f'Training index: {train_idx}, Loss: {loss.item()}')
                            print(f'Expression: {expr_str}')

                    # Second phase: LBFGS fine-tuning
                    print("\nStarting LBFGS fine-tuning...")
                    lbfgs_optim = torch.optim.LBFGS(model.parameters(),
                                                  lr=0.1,  # Smaller learning rate for fine-tuning
                                                  max_iter=20,
                                                  max_eval=25,
                                                  tolerance_grad=1e-7,
                                                  tolerance_change=1e-9,
                                                  history_size=50)

                    def lbfgs_closure():
                        lbfgs_optim.zero_grad()
                        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
                        du_pred, du_target = integrator.integrate(integration_args)
                        loss = mse(du_pred, du_target)
                        # Check for NaN in loss
                        if torch.isnan(loss):
                            print("Warning: NaN detected in loss during LBFGS closure")
                            return torch.tensor(1e6, requires_grad=True)  # Return a large value
                        loss.backward()
                        return loss

                    # Run LBFGS for fewer epochs since we're just fine-tuning
                    for train_idx in range(10):  # You can adjust this number
                        try:
                            loss = lbfgs_optim.step(lbfgs_closure)
                            if torch.isnan(loss):
                                print("Warning: NaN detected in loss, stopping LBFGS optimization")
                                break
                                
                            if train_idx % 5 == 0:  # Print more frequently during fine-tuning
                                print('✅'*40)
                                print(f'LBFGS Epoch {train_idx}, Loss: {loss.item()}')
                                try:
                                    expr_str = model.expression_visualize()
                                    print(f"Current expression: {expr_str}")
                                except Exception as e:
                                    print(f"Could not visualize expression: {e}")
                                print('✅'*40)
                        except Exception as e:
                            print(f"Error in LBFGS step: {e}")
                            break

                    if not math.isnan(loss.item()):
                        if dim == 1:
                            scores[tree_idx] = 1/ (1+torch.sqrt(loss))
                        elif dim ==2:
                            scores[tree_idx] = 1/(1+torch.sqrt(loss))
                        elif dim ==3:
                            scores[tree_idx] = 1/(1+torch.sqrt(loss))
                    else:
                        scores[tree_idx] = 0.
                    final_expr= model.expression_visualize()
                    logprint('✅'*40)
                    logprint(f'Operator Sequence: {op_seqs[tree_idx,:].tolist()}')
                    logprint(f'Final Loss: {loss.item():.6f}')
                    logprint(f'Score: {scores[tree_idx]:.6f}')
                    logprint(f'Final Expression look like:{final_expr}')
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

                scores_detached = torch.from_numpy(scores_detached).to(DEVICE)
                indicator_upper_quantile = torch.from_numpy(indicator_upper_quantile).to(DEVICE)
                
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
            # model_optimal = FEX(optimal_idx)
            # model_optim_optimal = torch.optim.Adam(model_optimal.parameters(),lr=FEX_LR)
            # for train_idx in range(TRAIN_EPOCHS_SECOND):
            #     adjust_learning_rate(model_optim_optimal,train_idx,FEX_LR,TRAIN_EPOCHS_SECOND)
            #     model_optim_optimal.zero_grad()
            #     integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model_optimal, index=dim)
            #     du_pred,du_target = integrator.integrate(integration_args)
            #     loss = mse(du_pred,du_target)
            #     loss.backward()
            #     model_optim_optimal.step()
            #     if train_idx % 100 == 0:
            #         logprint(f"Training step {train_idx} | Loss: {loss.item():.6f}")

            # np.save(optimal_idx_path, optimal_idx)
            # torch.save(model_optimal.state_dict(), model_save_path)
            # logprint(f"Model saved to {model_save_path}")
            # logprint(f"Optimal operator sequence saved to {optimal_idx_path}")
else:
    # Replace the hardcoded symbols with a dimension-variable approach
    symbols = [sp.symbols(f'x{i+1}') for i in range(dimension)]
    
    for dim in range(0+1,dimension+1):
        print(f'the dimension is {dim}')
        # In the ground truth training section, convert the list to tensor:
        if dim == 1: 
            op_seqs = torch.tensor([0, 2, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2], device=DEVICE)
        elif dim == 2:
            op_seqs = torch.tensor([1, 0, 2, 2, 0, 2, 1, 2, 2, 1, 0, 2], device=DEVICE)
        elif dim == 3:
            op_seqs = torch.tensor([2, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2], device=DEVICE)
        model = FEX(op_seqs, dim=dimension)
        model.apply(weights_init)
        model_optim = torch.optim.Adam(model.parameters(),lr=FEX_LR)
        for train_idx in range(TRAIN_EPOCHS_FIRST):
            model_optim.zero_grad()
            integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
            du_pred,du_target = integrator.integrate(integration_args)
            expr_str,_ = model.expression_visualize()
            # nonlinear_expr = sp.sympify(nonlinear_str)
            # nonlinear_expanded = sp.expand(nonlinear_expr)
            # if dim == 3:
            #     # Use the symbol variables dynamically
            #     coeff_x1x2 = nonlinear_expanded.coeff(symbols[0] * symbols[1])  # x1 * x2
            #     coeff_x1x2_tensor = torch.tensor(coeff_x1x2, dtype=du_pred.dtype, device=du_pred.device)
            #     loss = mse(du_pred,du_target)+(coeff_x1x2_tensor-0.4)**2
            # else:
            loss = mse(du_pred,du_target)
            loss.backward()
            model_optim.step()
            if train_idx % 10 == 0:
                # Get the expression string from FEX
                expr_str,simplified_expr_str = model.expression_visualize()
                # Try to split and simplify each part (linear and nonlinear)    
                print(f"Training index: {train_idx}, Loss: {loss.item()}")
                print(f'overall expression:{expr_str}')
                print(f'simplified expression:{simplified_expr_str}')



# Formula_1 = [-1.6100, 0.9751, -0.2461, 0.8654] # −0.2461x1+0.9751x2−1.6100x3+0.8654x2x3−0.0229
# Formula_2 = [-0.9674, -2.0017, -0.15087, -0.3720]
# Formula_3 = [1.5229, 1.2813, 0.1577, 1.05]
# plot_latex_formula(params,Formula_1, Formula_2, Formula_3)











# save_path = os.path.join(args.figure_save_path, 'three_comparing.pdf')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)
# plot_third_order_moments(np.arange(params['Nt']+1), moment3_MC_all,save_path)
# plot_deviation_subplots(np.arange(params['Nt']+1), cov_MC_all, moment3_MC_norm_all,save_path)



