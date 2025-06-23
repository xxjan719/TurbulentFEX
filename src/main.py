import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.FEX import FEX#, ThreeDimensionFEX
from utils.ODEParser import ODE_solver,FN_Net
from utils.constant import *
from utils.helper import logprint,adjust_learning_rate,weights_init,process_chunk_cpu,process_chunk_auto
from utils.controller import Controller
from utils.Sampler import Sampler
from utils.Pool import Pool
from utils.FEXParser import MultiDimensionFEX
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

    # Force SECOND_STAGE_OPEN_BOOL to False if TRAIN_GROUND_TRUTH is True
if args.TRAIN_GROUND_TRUTH:
    args.SECOND_STAGE_OPEN_BOOL = False


base_path = f'Example/{args.Model}/Results'
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

FEX_STAGE_OPEN_BOOL = args.FEX_STAGE_OPEN_BOOL
FEX_LR = args.FEX_LR
TRAIN_EPOCHS_FIRST = args.TRAIN_EPOCHS_FIRST
TRAIN_EPOCHS_SECOND = args.TRAIN_EPOCHS_SECOND

INTEGRATOR_METHOD = args.INTEGRATOR_METHOD
SECOND_STAGE_OPEN_BOOL = args.SECOND_STAGE_OPEN_BOOL
SHORT_SIZE = args.SHORT_SIZE

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
sampler = Sampler()
mse = nn.MSELoss()
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'],)
integrator = Body4TrainIntegrator(integratorParams,method=INTEGRATOR_METHOD)
pool = Pool()
if SECOND_STAGE_OPEN_BOOL == False:
    if FEX_STAGE_OPEN_BOOL == False:
        raise ValueError("Either FEX_STAGE_OPEN_BOOL must be True if SECOND_STAGE_OPEN_BOOL is False; \
                         or SECOND_STAGE_OPEN_BOOL must be True if FEX_STAGE_OPEN_BOOL is False; \
                         otherwise, you didn't run the framework")
    else:

        PMF_SIZES = [len(unary_ops),len(binary_ops),len(unary_ops),len(binary_ops)]*dimension
        NUM_NODES = len(PMF_SIZES)


        controller = Controller(pmf_sizes=PMF_SIZES).to(DEVICE)
        controller_optim = torch.optim.Adam(controller.parameters(), CONTROLLER_LR)
        
        # print(dataset.shape)
        if args.TRAIN_GROUND_TRUTH == False:
            for dim in range(0+2,3):#dimension+1):
                model_save_path = os.path.join(args.log_save_path, f"FEX_dim_{dim}.pth")
                log_file = os.path.join(args.log_save_path, f'log_dimension_{dim}.txt')
                if os.path.exists(model_save_path) and os.path.exists(log_file): #os.path.exists(model_save_path) and 
                    print(f'Model for dimension {dim} has already generated, just using for the second stage training:FEX'.center(60, '='))
                    # Load the saved model
                    optimal_idx = np.load(os.path.join(args.log_save_path, f'optimal_idx_{dim}.npy'))
                    print(f'dimension:{dim}, operator indx is {optimal_idx}')
                    model = FEX(torch.tensor(optimal_idx, device=DEVICE), dim=dimension).to(DEVICE)  # Initialize with dummy sequence
                    model.load_state_dict(torch.load(model_save_path))
                    print(f"Loaded model from {model_save_path}")
                    print(f"Model expression: {model.expression_visualize()}")
                    # Store the loaded model's operator sequence
                    optimal_idx = model.op_seqs.tolist()
                    print(f"Loaded operator sequence: {optimal_idx}")

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

                    # Initialize best candidates pool
                    best_candidates_pool = []
                    best_loss = float('inf')
                    MAX_BEST_CANDIDATES = 20

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
                                integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor.to(DEVICE), integration_func=model, index=dim)
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
                                integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor.to(DEVICE), integration_func=model, index=dim)
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
                        scores_detached = scores.cpu().detach().numpy()
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
                        
                        logprint(f'Pool for Exploration {explore_idx}'.center(60, '='))
                        logprint("\nAvailable candidates for this exploration:")
                        logprint("-" * 80)
                        for idx, candidate_ in enumerate(pool):
                            logprint(f"\nCandidate {idx + 1}:")
                            logprint(f"Loss: {candidate_.error:.6f}")
                            logprint(f"Operator sequence: {candidate_.action}")
                            logprint(f"Expression: {candidate_.expression}")
                            logprint("-" * 80)

                            # Update best candidates pool
                            current_loss = candidate_.error
                            current_expr = candidate_.expression
                            
                            # Check if this is a new best loss
                            if current_loss < best_loss:
                                best_loss = current_loss
                                best_candidates_pool = []  # Clear pool for new best loss
                                best_candidates_pool.append(candidate_)  # Add the new best candidate

                                logprint(f"\nAdded new best candidate to pool:")
                                logprint(f"Loss: {current_loss:.6f}")
                                logprint(f"Expression: {current_expr}")
                                logprint(f"Operator sequence: {candidate_.action}")
                                logprint("-" * 80)
                            # If this candidate has a loss very close to the best loss (within 1e-4)
                            elif np.abs(current_loss - best_loss) < 1.0e-4:
                                # Check if this expression is already in the pool
                                is_duplicate = False
                                for existing_candidate in best_candidates_pool:
                                    if current_expr == existing_candidate.expression:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    best_candidates_pool.append(candidate_)
                                    logprint(f"\nAdded candidate with close loss to pool:")
                                    logprint(f"Loss: {current_loss:.6f}")
                                    logprint(f"Expression: {current_expr}")
                                    logprint(f"Operator sequence: {candidate_.action}")
                                    logprint("-" * 80)

                        # Print current best candidates pool
                        logprint(f'\nCurrent Best Candidates Pool (Size: {len(best_candidates_pool)})'.center(60, '='))
                        for idx, candidate_ in enumerate(best_candidates_pool):
                            logprint(f"\nBest Candidate {idx + 1}:")
                            logprint(f"Loss: {candidate_.error:.6f}")
                            logprint(f"Operator sequence: {candidate_.action}")
                            logprint(f"Expression: {candidate_.expression}")
                            logprint("-" * 80)
                        
                    # Ask for user input from best candidates pool
                    logprint("\nSelect from the best candidates pool:")
                    for idx, candidate_ in enumerate(best_candidates_pool):
                        logprint(f"\nCandidate {idx + 1}:")
                        logprint(f"Loss: {candidate_.error:.6f}")
                        logprint(f"Operator sequence: {candidate_.action}")
                        logprint(f"Expression: {candidate_.expression}")
                        logprint("-" * 80)
                    
                    choice = input("\nEnter the candidate number you want to use for second stage training (1, 2, 3, etc.), or 'q' to quit: ")
                    if choice.lower() != 'q':
                        try:
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(best_candidates_pool):
                                best_candidate = best_candidates_pool[choice_idx]
                                optimal_idx = best_candidate.action
                                print(f'\nSelected candidate {choice}:')
                                print(f'Operator sequence: {optimal_idx}')
                                print(f'Expression: {best_candidate.expression}')
                            else:
                                print("Invalid choice. Using best candidate by loss.")
                                best_candidate = min(best_candidates_pool, key=lambda c: c.error)
                                optimal_idx = best_candidate.action
                        except ValueError:
                            print("Invalid input. Using best candidate by loss.")
                            best_candidate = min(best_candidates_pool, key=lambda c: c.error)
                            optimal_idx = best_candidate.action
                    else:
                        print("Exiting without second stage training.")
                        sys.exit(0)

                    logprint('✅'*40)
                    best_candidate = min(best_candidates_pool, key=lambda c: c.error)
                    optimal_idx = best_candidate.action
                    logprint(f'Optimal operator sequence: {optimal_idx}')

                    # Train and save the model
                    model = FEX(torch.tensor(optimal_idx, device=DEVICE), dim=dimension).to(DEVICE)
                    model.apply(weights_init)
                    model_optim = torch.optim.Adam(model.parameters(), lr=FEX_LR)
                    for train_idx in range(TRAIN_EPOCHS_SECOND):
                        adjust_learning_rate(model_optim,train_idx,FEX_LR,TRAIN_EPOCHS_SECOND)
                        model_optim.zero_grad()
                        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
                        du_pred, du_target = integrator.integrate(integration_args)
                        loss = mse(du_pred, du_target)
                        loss.backward()
                        model_optim.step()
                        if train_idx % 100 == 0:
                            logprint('✅'*40)
                            logprint(f"Training index: {train_idx}")
                            logprint(f"Loss: {loss.item():.6f}")
                            logprint(f"Expression: {model.expression_visualize()}")
                            logprint('✅'*40)
                    # Save both the model and its operator sequence
                    save_path = os.path.join(args.log_save_path, f'FEX_dim_{dim}.pth')
                    optimal_idx_path = os.path.join(args.log_save_path, f'optimal_idx_{dim}.npy')
                    torch.save(model.state_dict(), save_path)
                    np.save(optimal_idx_path, optimal_idx)
                    logprint(f"Model for dimension {dim} saved to {save_path}")
                    logprint(f"Optimal operator sequence saved to {optimal_idx_path}")
        else:
            # Replace the hardcoded symbols with a dimension-variable approach
            symbols = [sp.symbols(f'x{i+1}') for i in range(dimension)]
            op_seqs_all = {}
            
            for dim in range(0+1,dimension+1):
                print(f'the dimension is {dim}')
                # In the ground truth training section, convert the list to tensor:
                if dim == 1: 
                    op_seqs = torch.tensor([1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 2, 2], device=DEVICE)
                    
                elif dim == 2:
                    op_seqs = torch.tensor([2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2], device=DEVICE)#torch.tensor([2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2], device=DEVICE)
                elif dim == 3:
                    op_seqs = torch.tensor([0, 0, 2, 2, 2, 0, 2, 2, 5, 0, 7, 1], device=DEVICE)
                op_seqs_all[dim] = op_seqs

            print(f'the op_seqs_all is {op_seqs_all}')
            if args.MULTI_FEX_OPEN == True:
                combined_conservation_law = MultiDimensionFEX(op_seqs_all, dimension).to(DEVICE)
                print(f'data shape is {dataset_tensor.shape}')
                # Initialize all models
                for dim in range(1, dimension+1):
                    model = combined_conservation_law.models[str(dim)]
                    model.apply(weights_init)
                
                # Create optimizer for all parameters
                all_params = []
                for model in combined_conservation_law.models.values():
                    all_params.extend(model.parameters())
                model_optim = torch.optim.Adam(all_params, lr=FEX_LR)
                # Training loop
                for train_idx in range(TRAIN_EPOCHS_SECOND):
                    model_optim.zero_grad()
                    total_pred_loss = 0
                    # E_sum = 0
                       
                    # # Build L and G matrices
                    # L = torch.zeros(dimension, dimension, device=DEVICE)
                    # G = torch.zeros(dimension, dimension, device=DEVICE)
                    # for i in range(dimension):
                    #     coeffs = combined_conservation_law.models[str(i+1)].get_all_linear_nonlinear_coeffs_autograd(dim=i)
                    #     # coeffs is a tuple/list: (coeff_x1, coeff_x2, coeff_x3)
                    #     for j in range(dimension):
                    #         L[i, j] = coeffs[j]
                    # # Diagonal to G, off-diagonal to L
                    # for j in range(dimension):
                    #     G[j, j] = L[j, j]
                    #     L[j, j] = 0  # Zero out diagonal in L
                    # u_pred_all = torch.zeros(dataset_tensor.shape[0], dataset_tensor.shape[1],dataset_tensor.shape[2]-1, device=DEVICE)
                    # u_target_all = torch.zeros(dataset_tensor.shape[0], dataset_tensor.shape[1],dataset_tensor.shape[2]-1, device=DEVICE)
                    # l1_loss = 0
                    # Prediction and extra loss
                    for dim in range(1, dimension+1):
                        model = combined_conservation_law.models[str(dim)]
                        # Step 1: Get coefficients with autograd enabled
                        coeffs = model.get_all_linear_nonlinear_coeffs_autograd(dim=dim-1)
                        # Step 2: Compute rounded values

                        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
                        current_state = dataset_tensor[:, :, :-1]
                        u_current = current_state[:, 0, :]
                        # u_current_flat = u_current.reshape(-1, 1)
                        # u_current_reshaped = u_current_flat.reshape(u_current.shape)
                        u_pred, u_target = integrator.integrate(integration_args)

                        du_pred = torch.gradient(u_pred, dim=0)[0]
                        # dE_dt += torch.sum(u_pred * du_pred, dim=1)
                        loss = mse(u_pred, u_target)
                        if dim == 1:
                            coeffs_3 = round(float(coeffs[2]))
                            coeffs_2 = round(float(coeffs[1]))
                            #l1_loss = torch.abs(coeffs_3) + torch.abs(coeffs_2)
                            extra_loss = torch.abs(model.linear_a[0] + 0.2)**2
                            
                        elif dim == 2:
                            coeffs_3 = round(float(coeffs[2]))
                            coeffs_1 = round(float(coeffs[0]))
                            #l1_loss = torch.abs(coeffs_3) + torch.abs(coeffs_1)
                            extra_loss = torch.abs(model.linear_a[1] + 0.1)**2
                        elif dim == 3:
                            coeffs_2 = round(float(coeffs[1]))
                            coeffs_1 = round(float(coeffs[0]))
                            #l1_loss = torch.abs(coeffs_2) + torch.abs(coeffs_1)
                            extra_loss = torch.abs(model.linear_a[2] + 0.1)**2
                        total_pred_loss += loss+extra_loss
                    #     E_sum += torch.sum(u_pred**2, dim=1)
                    #     u_pred_all[:,dim-1,:] = u_pred.reshape(u_current.shape)
                    #     # print(f'u_pred_all shape is {u_pred_all.shape}')
                    #     u_target_all[:,dim-1,:] = u_target.reshape(u_current.shape)
                    #     #l1_loss += l1_loss
                    # # print(f'u_pred_all shape is {u_pred_all.shape}')
                    # # print(f'u_target_all shape is {u_target_all.shape}')
                    # N, D, T = u_pred_all.shape  # N=1000, D=3, T=1000
                    # cov_pred = torch.zeros((D, D, T))
                    # cov_target = torch.zeros((D, D, T))
                    # for t in range(T):
                    #     # u_pred_all[:, :, t] is shape (N, D), need (D, N)
                    #     cov_pred[:, :, t] = torch.cov(u_pred_all[:, :, t].T)
                    #     cov_target[:, :, t] = torch.cov(u_target_all[:, :, t].T)
                    # covu1u2_pred = cov_pred[0, 1, :]  # shape (T,)
                    # covu1u3_pred = cov_pred[0, 2, :]  # shape (T,)
                    # covu2u3_pred = cov_pred[1, 2, :]  # shape (T,)

                    # covu1u2_target = cov_target[0, 1, :]
                    # covu1u3_target = cov_target[0, 2, :]
                    # covu2u3_target = cov_target[1, 2, :]
                    # cov_pred_stack = torch.stack([covu1u2_pred, covu1u3_pred, covu2u3_pred], dim=1)
                    # cov_target_stack = torch.stack([covu1u2_target, covu1u3_target, covu2u3_target], dim=1)
                    # # cov_loss = mse(cov_pred, cov_target)
                    # # print(f'cov_pred shape is {cov_pred.shape}')
                    # # print(f'cov_target shape is {cov_target.shape}')
                    # # print(f'cov_pred_stack shape is {cov_pred_stack.shape}')
                    # # print(f'cov_target_stack shape is {cov_target_stack.shape}')
                    # cov_loss = mse(cov_pred_stack, cov_target_stack)
                    # # print(f'cov_loss shape is {cov_loss}')
                    
                    # total_pred_loss += cov_loss
                    # Energy conservation loss: minimize dE/dt
                    # derviatve_E_sum = torch.gradient(E_sum, dim=0)[0]
                    # print(f'derviative_E_sum shape is {derviatve_E_sum.shape}')
                    # print(f'L shape is {L.shape}')
                    # print(f'L is {L}')
                    # skew_loss = mse(L, -L.T)
                    #neg_diag_loss = torch.relu(G.diag()).sum()
                    #l1_loss = sum(torch.abs(param).sum() for param in model.parameters())
     
                    total_loss = total_pred_loss#+skew_loss #+ neg_diag_loss
                    # print(f'total_loss is {total_loss}, cov_loss is {cov_loss}, total_pred_loss is {total_pred_loss}, l1_loss is {l1_loss}')
                    total_loss.backward()
                    model_optim.step()

                    with torch.no_grad():
                        
                                
                        if train_idx % 100 == 0:
                            print(f"Training index: {train_idx}")
                            print(f"Loss: {total_loss.item():.6f},")
                            print(f"Expression: {combined_conservation_law.expression_visualize()}")
                

                # After the main Adam training loop, do LBFGS fine-tuning for each model
                for dim in range(1, dimension+1):
                     model = combined_conservation_law.models[str(dim)]
                     # Save individual model after training
                     save_path = os.path.join(args.log_save_path, f'FEX_dim_{dim}.pth')
                     torch.save(model.state_dict(), save_path)
                     optimal_idx_path = os.path.join(args.log_save_path, f'optimal_idx_{dim}.npy')
                     np.save(optimal_idx_path,  op_seqs_all[dim] )
                     print(f"Model for dimension {dim} saved to {save_path}")
            else:
                for dim in range(1, dimension+1):
                    model = FEX(op_seqs_all[dim], dim=dimension).to(DEVICE)
                    model.apply(weights_init)
                    model_optim = torch.optim.Adam(model.parameters(), lr=FEX_LR)
                    for train_idx in range(TRAIN_EPOCHS_SECOND):
                        model_optim.zero_grad()
                        adjust_learning_rate(model_optim,train_idx,FEX_LR,TRAIN_EPOCHS_SECOND)
                        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
                        du_pred, du_target = integrator.integrate(integration_args)
                        loss = mse(du_pred, du_target)
                        loss.backward()
                        model_optim.step()
                        if train_idx % 100 == 0:
                            print(f"Training index: {train_idx}")
                            print(f"Loss: {loss.item():.6f}")
                            print(f"Expression: {model.expression_visualize()}")
                            

                    # Save individual model after training
                    save_path = os.path.join(args.log_save_path, f'FEX_dim_{dim}.pth')
                    torch.save(model.state_dict(), save_path)
                    optimal_idx_path = os.path.join(args.log_save_path, f'optimal_idx_{dim}.npy')
                    np.save(optimal_idx_path,  op_seqs_all[dim] )
                    print(f"Model for dimension {dim} saved to {save_path}")
else:
    if not os.path.exists(os.path.join(args.log_save_path, 'optimal_idx_1.npy')) and not os.path.exists(os.path.join(args.log_save_path, 'optimal_idx_2.npy')) and not os.path.exists(os.path.join(args.log_save_path, 'optimal_idx_3.npy')):
        raise FileNotFoundError(f"optimal_idx_1.npy, optimal_idx_2.npy, optimal_idx_3.npy not found in {args.log_save_path}, you should run the FEX stage first.")
    
    # Load the optimal operator sequences
    op_seq_file_1 = np.load(os.path.join(args.log_save_path, 'optimal_idx_1.npy'), allow_pickle=True)
    op_seq_file_2 = np.load(os.path.join(args.log_save_path, 'optimal_idx_2.npy'), allow_pickle=True)
    op_seq_file_3 = np.load(os.path.join(args.log_save_path, 'optimal_idx_3.npy'), allow_pickle=True)
    op_seqs_all = [op_seq_file_1, op_seq_file_2, op_seq_file_3]
    diff_scale = args.DIFF_SCALE
    print(f'the dataset shape is {dataset.shape}')
    
    batch_size = 50000  # Changed from 4000 to 1000 to match the actual data size
    x_sample = dataset[:,:,:-1].reshape(-1, 3) 
    train_size = int(x_sample.shape[0]/10)
    print(f'train_size is {train_size}')
    # Reshape dataset to get x_sample
    SELECTED_ROW_INDICES = np.random.permutation(x_sample.shape[0])[:train_size]
     # Shape: (1000000, 3)
    X_TRAIN = x_sample[SELECTED_ROW_INDICES]
    print(f'x_sample shape is {x_sample.shape}')
    
    # Calculate z_short
    DIFFEREMCE = np.zeros((x_sample.shape[0], dimension))
    for idx in range(1, dimension+1):
        model_file = os.path.join(args.log_save_path, f'FEX_dim_{idx}.pth')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"FEX_dim_{idx}.pth not found in {args.log_save_path}, you should run the FEX stage first.")
        op_seq = op_seqs_all[idx-1]
        FEX_model = FEX(op_seq, dim=dimension).to(DEVICE)
        FEX_model.load_state_dict(torch.load(str(model_file), weights_only=True))
        integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor.to(DEVICE), integration_func=FEX_model, index=idx)
        du_pred, du_target = integrator.integrate(integration_args)
        difference = (du_target-du_pred)*diff_scale
        DIFFEREMCE[:,idx-1] = np.squeeze(difference.cpu().detach().numpy())
    print('✅'*40)
    print(f'DIFFEREMCE shape is {DIFFEREMCE.shape}')
    print(f'First few x_sample values:\n{x_sample[:5]}')
    print(f'First few DIFFEREMCE values:\n{DIFFEREMCE[:5]}')
    print(DIFFEREMCE)
    print(np.max(DIFFEREMCE, axis=0))
    print(np.min(DIFFEREMCE, axis=0))
    it_n_index = int(np.ceil(train_size / batch_size))
    print(f'it_n_index is {it_n_index}')
    TRAIN_INDEX_INITIAL = process_chunk_auto(
        it_n_index=it_n_index,
        it_size_x0train=batch_size,
        short_size=SHORT_SIZE,
        x_sample=x_sample,
        x0_train=X_TRAIN,  # Using x_sample as both training and query data
        train_size=train_size,
        x_dim=dimension
    )
    print(f'Index search completed. Shape of indices: {TRAIN_INDEX_INITIAL.shape}')
    X_SHORT = x_sample[TRAIN_INDEX_INITIAL]
    Z_SHORT = DIFFEREMCE[TRAIN_INDEX_INITIAL]
    ZT = np.random.randn(train_size, dimension)
    ODE_solution = np.zeros((train_size, dimension))
    
    print('✅'*40)
    EPOCHS_ODE_BATCH = int(min(train_size, 50000)/batch_size)  # Changed from 400000 to 200000 to be more conservative
    print(f'ZT shape is {ZT.shape}; ODE_solution shape is {ODE_solution.shape}, EPOCHS_ODE_BATCH is {EPOCHS_ODE_BATCH}')
    print('✅'*40)
    print('right now, we are going to solve the reverse ODE')
    
    torch.cuda.empty_cache()
    for BATCH_IDX in range(EPOCHS_ODE_BATCH):
        start_idx = BATCH_IDX*batch_size
        end_idx = min((BATCH_IDX+1)*batch_size,x_sample.shape[0])
        print(f'start_idx is {start_idx}; end_idx is {end_idx}')
        ZT_BATCH = torch.tensor(ZT[start_idx:end_idx]).to(DEVICE,dtype = torch.float32)
        INPUT_BATCH = torch.tensor(X_TRAIN[start_idx:end_idx]).to(DEVICE,dtype = torch.float32)
        MEAN_INPUT_BATCH = torch.tensor(X_SHORT[start_idx:end_idx]).to(DEVICE,dtype = torch.float32)
        RESIDUAL_BATCH = torch.tensor(Z_SHORT[start_idx:end_idx]).to(DEVICE,dtype = torch.float32)
        ODE_solution_BATCH = ODE_solver(ZT_BATCH,MEAN_INPUT_BATCH,RESIDUAL_BATCH,INPUT_BATCH)
        ODE_solution[start_idx:end_idx,:] = ODE_solution_BATCH.to('cpu').detach().numpy()
        if BATCH_IDX % 4 == 0:
            print(f'this is {BATCH_IDX} times / overall {EPOCHS_ODE_BATCH} times')
    print(f'ODE_solution shape is {ODE_solution.shape}')
    print('✅'*40)
    print('\nright now, we are going to save the data for second stage training')
    if not os.path.exists(os.path.join(args.log_save_path, 'DATA_TRAINING_X_SHORT.npy')):
        np.save(os.path.join(args.log_save_path, 'DATA_TRAINING_X_SHORT.npy'), X_SHORT)
    if not os.path.exists(os.path.join(args.log_save_path, 'DATA_TRAINING_Z_SHORT.npy')):
        np.save(os.path.join(args.log_save_path, 'DATA_TRAINING_Z_SHORT.npy'), Z_SHORT)
    if not os.path.exists(os.path.join(args.log_save_path, 'DATA_TRAINING_X_TRAIN.npy')):
        np.save(os.path.join(args.log_save_path, 'DATA_TRAINING_X_TRAIN.npy'), X_TRAIN)
    

    SECOND_STAGE_TRAINING_DATA = ZT
    if not os.path.exists(os.path.join(args.log_save_path, 'SECOND_STAGE_TRAINING_DATA.npy')):
        np.save(os.path.join(args.log_save_path, 'SECOND_STAGE_TRAINING_DATA.npy'), SECOND_STAGE_TRAINING_DATA)
    if not os.path.exists(os.path.join(args.log_save_path, 'ODE_REVERSE_SOLUTION.npy')):
        np.save(os.path.join(args.log_save_path, 'ODE_REVERSE_SOLUTION.npy'), ODE_solution)

    IS_FINITE_ODESOLUTION = np.isfinite(ODE_solution) &~np.isnan(ODE_solution)
    SECOND_STAGE_TRAINING_DATA_FILTERED = SECOND_STAGE_TRAINING_DATA[IS_FINITE_ODESOLUTION.all(axis=1)]
    ODE_REVERSE_SOLUTION_FILTERED = ODE_solution[IS_FINITE_ODESOLUTION.all(axis=1)]
    print(f'SECOND_STAGE_TRAINING_DATA_FILTERED shape is {SECOND_STAGE_TRAINING_DATA_FILTERED.shape[0]}')
    INDICES = np.random.permutation(SECOND_STAGE_TRAINING_DATA_FILTERED.shape[0])
    SECOND_STAGE_TRAINING_DATA_SHUFFLED = SECOND_STAGE_TRAINING_DATA_FILTERED[INDICES]
    ODE_REVERSE_SOLUTION_SHUFFLED = ODE_REVERSE_SOLUTION_FILTERED[INDICES]
    print(f'SECOND_STAGE_TRAINING_DATA_SHUFFLED shape is {SECOND_STAGE_TRAINING_DATA_SHUFFLED.shape}')
    print(f'ODE_REVERSE_SOLUTION_SHUFFLED shape is {ODE_REVERSE_SOLUTION_SHUFFLED.shape}')

    SECOND_STAGE_TRAINING_DATA_MEAN = np.mean(SECOND_STAGE_TRAINING_DATA_SHUFFLED, axis=0, keepdims=True)
    SECOND_STAGE_TRAINING_DATA_STD = np.std(SECOND_STAGE_TRAINING_DATA_SHUFFLED, axis=0, keepdims=True)
    SECOND_STAGE_TRAINING_DATA_NEW = (SECOND_STAGE_TRAINING_DATA_SHUFFLED - SECOND_STAGE_TRAINING_DATA_MEAN) / SECOND_STAGE_TRAINING_DATA_STD

    ODE_REVERSE_SOLUTION_MEAN = np.mean(ODE_REVERSE_SOLUTION_SHUFFLED, axis=0, keepdims=True)
    ODE_REVERSE_SOLUTION_STD = np.std(ODE_REVERSE_SOLUTION_SHUFFLED, axis=0, keepdims=True)
    ODE_REVERSE_SOLUTION_NEW = (ODE_REVERSE_SOLUTION_SHUFFLED - ODE_REVERSE_SOLUTION_MEAN) / ODE_REVERSE_SOLUTION_STD

    SECOND_STAGE_TRAINING_DATA_MEAN = torch.tensor(SECOND_STAGE_TRAINING_DATA_MEAN, dtype=torch.float32).to(DEVICE)
    SECOND_STAGE_TRAINING_DATA_STD = torch.tensor(SECOND_STAGE_TRAINING_DATA_STD, dtype=torch.float32).to(DEVICE)
    ODE_REVERSE_SOLUTION_MEAN = torch.tensor(ODE_REVERSE_SOLUTION_MEAN, dtype=torch.float32).to(DEVICE)
    ODE_REVERSE_SOLUTION_STD = torch.tensor(ODE_REVERSE_SOLUTION_STD, dtype=torch.float32).to(DEVICE)

    SECOND_STAGE_TRAINING_DATA_NEW = torch.tensor(SECOND_STAGE_TRAINING_DATA_NEW, dtype=torch.float32).to(DEVICE)
    ODE_REVERSE_SOLUTION_NEW = torch.tensor(ODE_REVERSE_SOLUTION_NEW, dtype=torch.float32).to(DEVICE)

    dataname2 = os.path.join(args.log_save_path, 'data_inf.pt')
    if not os.path.exists(dataname2):
        torch.save({'SECOND_STAGE_TRAINING_DATA_MEAN': SECOND_STAGE_TRAINING_DATA_MEAN,
                'SECOND_STAGE_TRAINING_DATA_STD': SECOND_STAGE_TRAINING_DATA_STD,
                'ODE_REVERSE_SOLUTION_MEAN': ODE_REVERSE_SOLUTION_MEAN,
                'ODE_REVERSE_SOLUTION_STD': ODE_REVERSE_SOLUTION_STD,
                'diff_scale': diff_scale}, dataname2)
        print(f'data saved to {dataname2}')
    else:
        print(f'data already exists in {dataname2}')
    
    print('✅'*40)
    print('SECOND STAGE TRAINING DATA IS SAVED')
    print(f'second stage mean is {SECOND_STAGE_TRAINING_DATA_MEAN}, std is {SECOND_STAGE_TRAINING_DATA_STD}')
    print(f'ODE reverse solution mean is {ODE_REVERSE_SOLUTION_MEAN}, std is {ODE_REVERSE_SOLUTION_STD}')
    print('✅'*40)

    NTrain = int(SECOND_STAGE_TRAINING_DATA_SHUFFLED.shape[0]*0.8)
    NValid = int(SECOND_STAGE_TRAINING_DATA_SHUFFLED.shape[0]*0.2)
    SECOND_STAGE_TRAINING_DATA_NORMAL = SECOND_STAGE_TRAINING_DATA_NEW[:NTrain,:]
    ODE_REVERSE_SOLUTION_NORMAL = ODE_REVERSE_SOLUTION_NEW[:NTrain,:]
    SECOND_STAGE_TRAINING_DATA_VALID = SECOND_STAGE_TRAINING_DATA_NEW[NTrain:,:]
    ODE_REVERSE_SOLUTION_VALID = ODE_REVERSE_SOLUTION_NEW[NTrain:,:]

    FN = FN_Net(dimension,dimension,50).to(DEVICE)
    FN.zero_grad()
    optimizer = torch.optim.Adam(FN.parameters(),lr = args.NN_SOLVER_LR,weight_decay = 1e-6)
    criterion = torch.nn.MSELoss()
    best_valid_err = 5.0
    for j in range(args.NN_SOLVER_EPOCHS):
        optimizer.zero_grad()
        pred = FN(SECOND_STAGE_TRAINING_DATA_NORMAL)
        loss = criterion(pred,ODE_REVERSE_SOLUTION_NORMAL)
        loss.backward()
        optimizer.step()
        pred1 = FN(SECOND_STAGE_TRAINING_DATA_VALID)
        valid_loss = criterion(pred1,ODE_REVERSE_SOLUTION_VALID)
        if valid_loss < best_valid_err:
            FN.update_best()
            best_valid_err = valid_loss
            print(f'best valid loss is {best_valid_err} at iteration {j}')
    FN.final_update()
    torch.save(FN.state_dict(), os.path.join(args.log_save_path, 'FN_Net.pth'))
    print('FN_Net saved to same folder')
    print('✅'*40)
    print('SECOND STAGE TRAINING IS COMPLETED')
    print('✅'*40)
    print('NOW, you can run the prediction file.')


    



    
    
    
    


    
    
    


    





# Formula_1 = [-1.6100, 0.9751, -0.2461, 0.8654] # −0.2461x1+0.9751x2−1.6100x3+0.8654x2x3−0.0229
# Formula_2 = [-0.9674, -2.0017, -0.15087, -0.3720]
# Formula_3 = [1.5229, 1.2813, 0.1577, 1.05]




# [2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2] dimension 2  11 epochs
#  [0, 0, 2, 2, 2, 2, 2, 2, 5, 0, 7, 0] dimension 3 14 epochs
# Selected candidate 4:
# Operator sequence: [8, 2, 2, 2, 5, 2, 1, 0, 2, 2, 7, 2]






# save_path = os.path.join(args.figure_save_path, 'three_comparing.pdf')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)
# plot_third_order_moments(np.arange(params['Nt']+1), moment3_MC_all,save_path)
# plot_deviation_subplots(np.arange(params['Nt']+1), cov_MC_all, moment3_MC_norm_all,save_path)



