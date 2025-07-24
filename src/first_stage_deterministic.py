import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
from config import DIR_EXAMPLE, DIR_TRIAD,create_main_parser
import torch
import torch.nn as nn
import math
import numpy as np
import random
import logging
import sympy as sp

parser = create_main_parser()
args = parser.parse_args()

# Check if CUDA is available and set device accordingly
if torch.cuda.is_available() and args.DEVICE.startswith('cuda'):
    DEVICE = torch.device(args.DEVICE)
    print(f"Using {args.DEVICE}")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results','Results')
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available, using CPU instead")
    base_path = os.path.join(DIR_EXAMPLE,args.Model,'Results')

if args.DATA_SAVE_PATH is None:
    args.DATA_SAVE_PATH = f'{base_path}/{args.params_name}/noise_{args.NOISE_LEVEL}/simulation_results_noise_{args.NOISE_LEVEL}.npz'
if args.LOG_SAVE_PATH is None:
    args.LOG_SAVE_PATH = f'{base_path}/{args.params_name}'
if args.FIGURE_SAVE_PATH is None:
    args.FIGURE_SAVE_PATH = f'{base_path}/{args.params_name}'

# Create necessary directories
os.makedirs(os.path.dirname(args.DATA_SAVE_PATH), exist_ok=True)
os.makedirs(args.LOG_SAVE_PATH, exist_ok=True)
os.makedirs(args.FIGURE_SAVE_PATH, exist_ok=True)



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
params = params_init(args.params_name,sample=1000)
data_file = args.DATA_SAVE_PATH

if os.path.exists(data_file):
    print("\n"+"="*60)
    print(f'[INFO] Data has already generated, just using for the first stage training:FEX'.center(60, '='))
    data = np.load(data_file)
    dataset =  data['dataset']
    mean_MC = data['mean_MC']
    cov_MC = data['cov_MC']
    moment3_MC = data['moment3_MC']
    moment3_MC_norm = data['moment3_MC_norm']
    Energy_MC = data['Energy_MC']
    Energy_dyn = data['Energy_dyn']
else:
    print("\n"+"="*60)
    print(f'[INFO] There is no dataset in this environment, it generates automatically'.center(60,'-'))
    dataset, mean_MC, cov_MC, moment3_MC, moment3_MC_norm,Energy_MC, Energy_dyn = MC_triad_direct(params, m0, var0,
    method = 'Euler',noise_level = args.NOISE_LEVEL)
    np.savez(
    args.DATA_SAVE_PATH,
    dataset=dataset,
    mean_MC=mean_MC,
    cov_MC=cov_MC,
    moment3_MC=moment3_MC,
    moment3_MC_norm=moment3_MC_norm,
    Energy_MC=Energy_MC,
    Energy_dyn=Energy_dyn
    )
    print(f'[INFO] Right now it is ok for data. We use it for the first stage training: FEX'.center(60,'='))

dataset_tensor = torch.from_numpy(dataset).float().to(DEVICE)
dimension = dataset_tensor.shape[1]  # Assuming the second dimension is the number of features
sampler = Sampler()
mse = nn.MSELoss()
l1 = nn.L1Loss()
integratorParams = Body4TrainIntegrationParams(dt=params['Dt'],)
integrator = Body4TrainIntegrator(integratorParams,method=INTEGRATOR_METHOD)
pool = Pool()


PMF_SIZES = tuple([len(unary_ops), len(binary_ops), len(unary_ops), len(binary_ops)] * dimension)
NUM_NODES = len(PMF_SIZES)


controller = Controller(pmf_sizes=PMF_SIZES).to(DEVICE)
controller_optim = torch.optim.Adam(controller.parameters(), CONTROLLER_LR)
        
print(dataset.shape)
print(f'the dimension is {dimension}')
print(f'the PMF_SIZES is {PMF_SIZES}')
print(f'the NUM_NODES is {NUM_NODES}')
print("="*60)

if args.TRAIN_THREE_DIMENSION_INTEGRATED == False:
    print("\n"+"="*60)
    print('[INFO] Start to train the FEX')
    print("[INFO] The idea is first train the FEX for each dimension, and then train the integrated FEX model")
    print("And in this example, we always can get  ground truth operator sequence for each dimension")
    #for dim in range(1, dimension+1):
    print("\n"+"="*60)
    print(f"The dimension is {args.TRAIN_WORKING_DIM}")
    model_save_path = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f"best_candidates_pool_summary_{args.TRAIN_WORKING_DIM}.txt")
    log_file = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}",f'log_dimension_{args.TRAIN_WORKING_DIM}_{args.NOISE_LEVEL}.txt')
    # Always create the log file directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    if os.path.exists(model_save_path) and os.path.exists(log_file): #os.path.exists(model_save_path) and 
        print(f'[INFO] Model for dimension {args.TRAIN_WORKING_DIM} has already generated, just using for the second stage training:FEX'.center(60, '='))
        print("\n Loading the initial training model and log file")
        print('[INFO] Print the initial training model expression')          
        get_score_expression_from_file(model_save_path)
    else:
        print(f'[INFO]No MODEL FOR DIMENSION {args.TRAIN_WORKING_DIM} SAVED IN THIS PATH, it will be generated automatically')        
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
        print("\n")
        print('[INFO] Initialize the best candidates pool')
        best_candidates_pool = []
        best_loss = float('inf')
        MAX_BEST_CANDIDATES = 20

        # dimension 1 need 10 EXPLORATION_ITERS
        for explore_idx in range(EXPLORATION_ITERS):
            print(f'\n[INFO] Exploration {explore_idx + 1}/{EXPLORATION_ITERS}')
            logprint(f'\n[INFO] Exploration {explore_idx + 1}/{EXPLORATION_ITERS}')
                
            controller_optim.zero_grad()
            pmfs = controller(torch.zeros(CONTROLLER_INPUT_SIZE, device=DEVICE))
            scores = torch.zeros(NUM_TREES, device=DEVICE)
                
                # Generate and train operator sequences
            op_seqs = torch.zeros(NUM_TREES, NUM_NODES, dtype=torch.int, device=DEVICE)
            trained_count = 0
                
            for tree_idx in range(NUM_TREES):
                op_seqs[tree_idx, :] = sampler(pmfs, output=torch.zeros(NUM_NODES, dtype=torch.int, device=DEVICE))
                model = FEX(op_seqs[tree_idx,:], dim=3).to(DEVICE)
                model.apply(weights_init)
                expression = model.expression_visualize()
                parts = expression.split(') + (')
                nonlinear_expr = parts[1].strip()
                    
                # Skip trivial expressions
                if ("x1" not in nonlinear_expr and "x2" not in nonlinear_expr and "x3" not in nonlinear_expr) or \
                    ("x1" in nonlinear_expr and "x2" not in nonlinear_expr and "x3" not in nonlinear_expr and "**" not in nonlinear_expr and "sin" not in nonlinear_expr and "cos" not in nonlinear_expr and "exp" not in nonlinear_expr) or \
                    ("x1" not in nonlinear_expr and "x2" in nonlinear_expr and "x3" not in nonlinear_expr and "**" not in nonlinear_expr and "sin" not in nonlinear_expr and "cos" not in nonlinear_expr and "exp" not in nonlinear_expr) or \
                    ("x1" not in nonlinear_expr and "x2" not in nonlinear_expr and "x3" in nonlinear_expr and "**" not in nonlinear_expr and "sin" not in nonlinear_expr and "cos" not in nonlinear_expr and "exp" not in nonlinear_expr) or \
                    ("x1" in nonlinear_expr and "x2" in nonlinear_expr and "x3" in nonlinear_expr and "**" not in nonlinear_expr and "sin" not in nonlinear_expr and "cos" not in nonlinear_expr and "exp" not in nonlinear_expr):
                    print(f"[INFO] Skipping model with trivial nonlinear expression: {expression}")
                    logprint(f"[INFO] Skipping model with trivial nonlinear expression: {expression}")
                    continue
                    
                trained_count += 1
                    
                # Train the model
                model_optim = torch.optim.Adam(model.parameters(), lr=FEX_LR)
                for train_idx in range(TRAIN_EPOCHS_FIRST):
                    model_optim.zero_grad()
                    integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor.to(DEVICE), integration_func=model, index=args.TRAIN_WORKING_DIM)
                        
                    du_pred, du_target = integrator.integrate(integration_args)
                    loss = mse(du_pred, du_target)
                    loss.backward()
                    model_optim.step()
                    
                # LBFGS fine-tuning
                lbfgs_optim = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, max_eval=25,
                                                   tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)
                    
                def lbfgs_closure():
                    lbfgs_optim.zero_grad()
                    integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor.to(DEVICE), integration_func=model, index=args.TRAIN_WORKING_DIM)
                    du_pred, du_target = integrator.integrate(integration_args)
                    loss = mse(du_pred, du_target)
                    if torch.isnan(loss):
                        return torch.tensor(1e6, requires_grad=True)
                    loss.backward()
                    return loss
                    
                # Run LBFGS
                for _ in range(10):
                    try:
                        loss = lbfgs_optim.step(lbfgs_closure)
                        if torch.isnan(loss):
                            break
                    except Exception:
                        break
                    
                # Ensure loss is a tensor for consistent handling
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=DEVICE)
                    
                # Apply noise level penalty if needed
                if args.NOISE_LEVEL == 0:
                    loss = 1e6 * loss
                elif args.NOISE_LEVEL == 0.2:
                    loss = 2e3 * loss
                elif args.NOISE_LEVEL == 1:
                    loss = 80*loss
                # Calculate score and add to pool
                if not math.isnan(loss.item()):
                    scores[tree_idx] = 1 / (1 + torch.sqrt(loss))
                else:
                    scores[tree_idx] = 0.
                    
                pool.add(scores[tree_idx], model, loss.item(), op_seqs[tree_idx,:].tolist())
                    
                # Print current model info
                print("\n"+"="*60)
                print(f"[INFO] Model {tree_idx + 1}")
                print(f"Expression: {model.expression_visualize()}")
                print(f"Expression simplified: {model.expression_visualize_simplified()}")
                print(f"Loss: {loss.item():.6f}")
                print(f"Score: {scores[tree_idx]:.6f}")
                print(f"Operator sequence: {op_seqs[tree_idx,:].tolist()}")
                print("="*60)
                    
                   
                    
                logprint("\n"+"="*60)
                logprint(f"[INFO] Model {tree_idx + 1}")
                logprint(f"Expression: {model.expression_visualize()}")
                logprint(f"Expression simplified: {model.expression_visualize_simplified()}")
                logprint(f"Loss: {loss.item():.6f}")
                logprint(f"Score: {scores[tree_idx]:.6f}")
                logprint(f"Operator sequence: {op_seqs[tree_idx,:].tolist()}")
                logprint("="*60)
            # Controller update
            scores_detached = scores.cpu().detach().numpy()
            scores_upper_quantile = np.percentile(scores_detached, q=(1 - CONTROLLER_TOP_SAMPLES_FRACTION), method=CONTROLLER_QUANTILE_METHOD)
            indicator_upper_quantile = (scores_detached >= scores_upper_quantile).astype(int)
                
            sum_log_probs = torch.zeros(NUM_TREES, device=DEVICE)
            log_pmfs = [torch.log(pmf) for pmf in pmfs]
            for tree_idx, ops in enumerate(op_seqs):
                for pmf_idx, op in enumerate(ops):
                    log_prob = log_pmfs[pmf_idx][op]
                    sum_log_probs[tree_idx] += log_prob
                
            scores_detached = torch.from_numpy(scores_detached).to(DEVICE)
            indicator_upper_quantile = torch.from_numpy(indicator_upper_quantile).to(DEVICE)
                
            controller_loss = -(1 / CONTROLLER_TOP_SAMPLES_FRACTION) * torch.mean((scores_detached - scores_upper_quantile) * indicator_upper_quantile * sum_log_probs)
            controller_loss.backward()
            controller_optim.step()
                
            # Log exploration results
            logprint(f"Trained {trained_count}/{NUM_TREES} sequences")
            logprint(f"Best loss in pool: {min([c.error for c in pool]):.6f}")
            print(f"Trained {trained_count}/{NUM_TREES} sequences")
            print(f"Best loss in pool: {min([c.error for c in pool]):.6f}")
                
            # Update best candidates pool
            for candidate_ in pool:
                current_loss = candidate_.error
                current_expr = candidate_.expression  # This is now already simplified
                current_score = candidate_.score  # assuming .score exists
                            
                # Check if expression follows the allowed terms for this dimension
                check_result = check_allowed_terms(current_expr, args.TRAIN_WORKING_DIM)
                if not check_result['valid']:
                    continue
                elif args.TRAIN_WORKING_DIM == 1 and not ('x2*x3' in check_result['terms_present'] or 'x2x3' in check_result['terms_present']):
                    continue
                elif args.TRAIN_WORKING_DIM == 2 and not ('x1*x3' in check_result['terms_present'] or 'x1x3' in check_result['terms_present']):
                    continue
                elif args.TRAIN_WORKING_DIM == 3 and not ('x1*x2' in check_result['terms_present'] or 'x1x2' in check_result['terms_present']):
                    continue
                else:
                    best_candidates_pool.append(candidate_)
            # Print current pool status
            print("\n"+"="*60)
            print(f"\n[INFO] Current Pool Status ({len(pool)} candidates):")
            print("Pool Selection:")
            pool_list = sorted(list(pool), key=lambda c: c.score, reverse=True)
            # Print top 50
            for idx, candidate_ in enumerate(pool_list[:50]):
                print(f"  {idx + 1}. Score: {candidate_.score:.6f}, Loss: {candidate_.error:.6f}, Seq: {candidate_.action}, Expression={candidate_.expression}")
                
            print("\n")
            print("Best Candidate Pool Selection:")
            for idx,  candidate_ in enumerate(best_candidates_pool):
                print(f"  {idx + 1}. Loss: {candidate_.error:.6f}, Seq: {candidate_.action}, Expression={candidate_.expression}")
            print("=" * 60)
            
        # Select best candidate
        logprint(f"\nBest candidates found: {len(best_candidates_pool)}")
        print(f"\nBest candidates found: {len(best_candidates_pool)}")
        for idx, candidate_ in enumerate(best_candidates_pool):
            logprint(f"Candidate {idx + 1}: Loss={candidate_.error:.6f}, Seq={candidate_.action}")
            print(f"Candidate {idx + 1}: Loss={candidate_.error:.6f}, Seq={candidate_.action}")
            
        # Create save directory if it doesn't exist
        save_dir = os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}")
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, f"best_candidates_pool_summary_{args.TRAIN_WORKING_DIM}.txt")
        # Write summary
        with open(summary_path, "w") as f:
            for idx, candidate_ in enumerate(best_candidates_pool):
                f.write(f"Candidate {idx + 1}: Score={candidate_.score:.6f}, Loss={candidate_.error:.6f}, Seq={candidate_.action}, Expr={candidate_.expression}\n")

        print(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        logprint(f"[INFO] best_candidates_pool_summary saved to {summary_path}")
        # Use best candidate by default (or add user selection if needed)
        best_candidate = min(best_candidates_pool, key=lambda c: c.error)
        optimal_idx = best_candidate.action
        logprint(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.expression}")
        print(f"Selected: Loss={best_candidate.error:.6f}, Expression={best_candidate.expression}")

            
        logprint(f"[INFO] Now we need to train the integrated FEX model")
        print(f"[INFO] Now we need to train the integrated FEX model")
else:
    print("\n"+"="*60)
    print("[INFO] Loading FEX models from previous stage...")
    # print(f"[INFO] get the picture of how the single dimension FEX model works")
    # coefficients = get_coefficients(load_dir= DIR_TRIAD, DEVICE=args.DEVICE)
    # plot_NOISE_LEVEL_EFFECT(coefficients,save_dir=args.LOG_SAVE_PATH)
    # print(f"the coefficients are {coefficients}")
    op_seqs_all = {}
    models = {}
    symbols = [sp.symbols(f'x{i+1}') for i in range(dimension)]
    for dim in range(1, dimension+1):
        print(f'the dimension is {dim}')
        sequence = get_sequence(os.path.join(args.LOG_SAVE_PATH, f"noise_{args.NOISE_LEVEL}", f'best_candidates_pool_summary_{dim}.txt'))
        op_seqs = torch.tensor(sequence, device=DEVICE)
        op_seqs_all[dim] = op_seqs
        print(f"[INFO] {dim} dimensiondata found. Now let us train inetgerated FEX model")
        print("\n")
        model = FEX(op_seqs, dim=dimension).to(DEVICE)
        model.apply(weights_init)
        models[str(dim)] = model
        

    print("="*60)
    # # Replace the hardcoded symbols with a dimension-variable approach
              
    #if dim == 1: # torch.tensor([1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 2, 2], device=DEVICE)
    #elif dim == 2: # torch.tensor([2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2], device=DEVICE)
    #elif dim == 3:#torch.tensor([0, 0, 2, 2, 2, 0, 2, 2, 5, 0, 7, 1], device=DEVICE)
    # print(f"the coefficents_history is {coefficents_history}")
    loss_history = []
    # # Create optimizer for all parameters
    all_params = []
    for model in models.values():
        all_params.extend(model.parameters())
    model_optim = torch.optim.Adam(all_params, lr=FEX_LR)
    
    # # Training loop
    for train_idx in range(TRAIN_EPOCHS_SECOND):
        model_optim.zero_grad()
        total_pred_loss = 0

        # Prediction and extra loss
        for dim in range(1, dimension+1):
            model = models[str(dim)]
            # Step 1: Get coefficients with autograd enabled
            coeff_x1, coeff_x2, coeff_x3 = model.get_all_linear_nonlinear_coeffs_autograd(dim=dim-1)
            # Step 2: Compute rounded values

            integration_args = Body4TrainIntegrationArgs(y0=dataset_tensor, integration_func=model, index=dim)
            current_state = dataset_tensor[:, :, :-1]
            u_current = current_state[:, 0, :]
            u_pred, u_target = integrator.integrate(integration_args)

            du_pred = torch.gradient(u_pred, dim=0)[0]
            loss = mse(u_pred, u_target)
            # if dim == 1:
            #     coeffs_3 = round(float(coeff_x3))
            #     coeffs_2 = round(float(coeff_x2))
            #     extra_loss = torch.abs(model.linear_a[0] + 0.2)**2
                        
            # elif dim == 2:
            #     coeffs_3 = round(float(coeff_x3))
            #     coeffs_1 = round(float(coeff_x1))
            #     extra_loss = torch.abs(model.linear_a[1] + 0.1)**2
            # elif dim == 3:
            #     coeffs_2 = round(float(coeff_x2))
            #     coeffs_1 = round(float(coeff_x1))
            #     extra_loss = torch.abs(model.linear_a[2] + 0.1)**2
            total_pred_loss += loss#+ extra_loss
        
        # Call backward only once after all dimensions are processed
        total_pred_loss.backward(retain_graph=True)
        model_optim.step()

        with torch.no_grad():
            if train_idx % 50 == 0:
                loss_history.append(total_pred_loss.item())
                for dim in range(1, dimension+1):
                    model = models[str(dim)]
                    expr = model.expression_visualize_simplified()
                    coeffs = extract_coefficients_from_expr(expr, dim)
                    for term, value in coeffs.items():
                        coefficents_history[dim][term].append(value)
                #print(f"the coefficents_history is {coefficents_history}")
            if train_idx % 100 == 0:
                print("\n"+"="*60)
                print(f"Training index: {train_idx}")
                print(f"Loss: {total_pred_loss.item():.6f}")
                # Print expressions for each dimension
                expressions = {}
                for dim in range(1, dimension+1):
                    expressions[f'Dimension {dim}'] = models[str(dim)].expression_visualize_simplified()
                print(f"Expression: {expressions}")
                print("="*60)

        if train_idx == TRAIN_EPOCHS_SECOND-1:
            for dim in range(1, dimension+1):
                final_expr = models[str(dim)].expression_visualize_simplified()
            loss_history_dict = {1: loss_history, 2: loss_history, 3: loss_history}
            plot_training_progress_grid(loss_history_dict, coefficents_history, final_expr, args.NOISE_LEVEL,save_dir=args.LOG_SAVE_PATH)



    



    
    
    
    


    
    
    


    





# # Formula_1 = [-1.6100, 0.9751, -0.2461, 0.8654] # −0.2461x1+0.9751x2−1.6100x3+0.8654x2x3−0.0229
# # Formula_2 = [-0.9674, -2.0017, -0.15087, -0.3720]
# # Formula_3 = [1.5229, 1.2813, 0.1577, 1.05]




# # [2, 1, 2, 2, 0, 0, 1, 2, 0, 0, 2, 2] dimension 2  11 epochs
# #  [0, 0, 2, 2, 2, 2, 2, 2, 5, 0, 7, 0] dimension 3 14 epochs
# # Selected candidate 4:
# # Operator sequence: [8, 2, 2, 2, 5, 2, 1, 0, 2, 2, 7, 2]






# # save_path = os.path.join(args.figure_save_path, 'three_comparing.pdf')
# # os.makedirs(os.path.dirname(save_path), exist_ok=True)
# # plot_stats(np.arange(params['Nt']+1), mean_MC_all, cov_MC_all, moment3_MC_all, Energy_MC_all, Energy_dyn,save_path)
# # plot_third_order_moments(np.arange(params['Nt']+1), moment3_MC_all,save_path)
# # plot_deviation_subplots(np.arange(params['Nt']+1), cov_MC_all, moment3_MC_norm_all,save_path)



