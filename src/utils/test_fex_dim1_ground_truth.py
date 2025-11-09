"""
Test script for Ground Truth FEX Result for Dimension 1 with random_cascade_deterministic dataset
This script tests and optimizes FEX_with_random_force models using actual dataset.
"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.FEX_with_random_force import FEX_with_random_force
from utils.Train_Integrator import Body4TrainIntegrationArgs, Body4TrainIntegrationParams, Body4TrainIntegrator
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
from config import DIR_EXAMPLE

def test_fex_dim1_ground_truth(use_ground_truth_tmM=False):
    """
    Test and optimize FEX_with_random_force model for dimension 1 using random_cascade_deterministic dataset
    
    Args:
        use_ground_truth_tmM: If True, use ground truth tmM values for OU loss (easier).
                              If False, learn m_t from state dynamics only (more challenging).
    """
    
    print("="*80)
    print("Ground Truth FEX Test for Dimension 1 - random_cascade_deterministic Dataset")
    print("="*80)
    
    # Load random_cascade_deterministic dataset
    params_name = 'random_cascade_deterministic'
    noise_level = 1.0
    
    # Get data file path
    base_path = os.path.join(DIR_EXAMPLE, 'MC_triad', 'Results', params_name, f'noise_{noise_level}')
    data_file = os.path.join(base_path, f'simulation_results_noise_{noise_level}.npz')
    
    print(f"\n[INFO] Loading dataset from: {data_file}")
    
    if os.path.exists(data_file):
        data = np.load(data_file)
        dataset_full = data['dataset']  # Shape: (50000, 3, 1001)
        print(f"[INFO] Loaded dataset shape: {dataset_full.shape}")
    else:
        print(f"[INFO] Dataset not found, generating new dataset...")
        m0, var0 = MC_triad_initial_value()
        params = params_init(params_name, sample=50000)
        dataset_full, mean_MC, cov_MC, moment3_MC, moment3_MC_norm, Energy_MC, Energy_dyn = MC_triad_direct(
            params, m0, var0, noise_level=noise_level
        )
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.savez(data_file, dataset=dataset_full, mean_MC=mean_MC, cov_MC=cov_MC, 
                moment3_MC=moment3_MC, moment3_MC_norm=moment3_MC_norm, 
                Energy_MC=Energy_MC, Energy_dyn=Energy_dyn)
        print(f"[INFO] Generated and saved dataset shape: {dataset_full.shape}")
    
    # Select subset for testing
    test_samples = 1000
    np.random.seed(42)
    selected_indices = np.random.choice(dataset_full.shape[0], size=min(test_samples, dataset_full.shape[0]), replace=False)
    dataset = dataset_full[selected_indices]  # Shape: (test_samples, 3, time_steps)
    
    print(f"[INFO] Using {test_samples} samples for testing")
    print(f"[INFO] Dataset shape: {dataset.shape}")
    
    # Convert to tensor
    dataset_tensor = torch.from_numpy(dataset).float()
    
    # Get dt from params
    params = params_init(params_name, sample=50000)
    dt = params['Dt']
    print(f"[INFO] Time step dt: {dt}")
    
    # Check ground truth forcing values (tmM) - this is what m(t) should represent
    if 'tmM' in params:
        tmM = params['tmM']  # Shape: (Nt, 3)
        print(f"[INFO] Ground truth tmM shape: {tmM.shape}")
        print(f"[INFO] Ground truth tmM - mean: {tmM.mean():.6f}, std: {tmM.std():.6f}")
    
    # Use single operator sequence for dimension 1 testing (16 operators: 12 state + 4 force)
    test_op_seq = [1, 1, 2, 1,    # x1 operators
                   1, 1, 2, 2,    # x2 operators
                   0, 1, 2, 2,    # x3 operators
                   2, 1, 0, 2]    # m(t) operators
    
    print(f"\n{'='*80}")
    print(f"Testing Operator Sequence: {test_op_seq}")
    print(f"{'='*80}")
    
    op_seqs = torch.tensor(test_op_seq)
    model = FEX_with_random_force(op_seqs, dim=3)
    
    # Setup integrator (same as in 1stage_deterministic.py)
    integrator_params = Body4TrainIntegrationParams(dt=dt)
    integrator = Body4TrainIntegrator(integrator_params, method="integration-based")
    
    integration_args = Body4TrainIntegrationArgs(
        y0=dataset_tensor,
        integration_func=model,
        index=1,  # Test dimension 1
        params_name=params_name
    )
    
    # Print initial network and Force_FEX parameters
    print(f"[INFO] Initial m_network: {sum(p.numel() for p in model.m_network.parameters())} parameters")
    print(f"[INFO] Initial Force_FEX parameters:")
    print(f"  force_weight_1: {model.Force_FEX.force_weight_1.item():.6f}")
    print(f"  force_weight_2: {model.Force_FEX.force_weight_2.item():.6f}")
    print(f"  force_weight_3: {model.Force_FEX.force_weight_3.item():.6f}")
    print(f"  force_bias_1: {model.Force_FEX.force_bias_1.item():.6f}")
    print(f"  force_bias_2: {model.Force_FEX.force_bias_2.item():.6f}")
    print(f"  force_bias_3: {model.Force_FEX.force_bias_3.item():.6f}")
    
    # Check ground truth tmM if available (for reference)
    if 'tmM' in params:
        tmM_gt = params['tmM']  # Shape: (Nt, 3)
        print(f"[INFO] Ground truth tmM available - mean: {tmM_gt.mean():.6f}, std: {tmM_gt.std():.6f}")
        print(f"  Note: m(t) will be learned by the neural network from (u1, u2, u3, t)")
    
    # Initial evaluation
    print(f"\n[Initial Evaluation]")
    with torch.no_grad():
        pred, label, x_current, t_current = integrator.integrate(integration_args)
        
        mse_state = nn.MSELoss()(pred, label)
        
        # Compute m(t) from the network
        m_t = model.m_network(torch.cat([x_current, t_current], dim=1))
        
        # Compute dm/dt using autograd
        dm_dt_autograd = model.compute_dm_dt(x_current, t_current)
        
        # Compute dm/dt from Force_FEX: dm/dt = Force_FEX(m(t))
        dm_dt_pred = model.Force_FEX(m_t)
        
        # OU loss: ||dm/dt (autograd) - Force_FEX(m(t))||
        mse_ou = nn.MSELoss()(dm_dt_autograd, dm_dt_pred)
        total_loss = mse_state + mse_ou
        
        print(f"  Total Loss: {total_loss.item():.6f}")
        print(f"  MSE (State): {mse_state.item():.6f}")
        print(f"  MSE (OU): {mse_ou.item():.6f}")
        print(f"  dt used: {integrator._integratorparams.dt}")
        print(f"  m_t shape: {m_t.shape}")
        print(f"  dm_dt_autograd mean: {dm_dt_autograd.mean().item():.6f}")
        print(f"  dm_dt_pred (Force_FEX) mean: {dm_dt_pred.mean().item():.6f}")
        
        # Show initial formula
        try:
            simplified_expr = model.expression_visualize_simplified()
            print(f"\n  Initial Formula: {simplified_expr}")
        except Exception as e:
            print(f"  [Warning] Could not generate formula: {e}")
    
    # Training loop (following 1stage_deterministic.py pattern)
    print(f"\n{'='*80}")
    print("Training Model with Adam...")
    print(f"{'='*80}")
    
    train_epochs = 50000
    learning_rate_high = 0.5  # Use when loss > 2
    learning_rate_low = 0.01   # Use when loss <= 2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_high)
    mse_loss_fn = nn.MSELoss()
    
    print(f"  Training epochs: {train_epochs}")
    print(f"  Learning rate schedule: {learning_rate_high} when loss > 2, {learning_rate_low} when loss <= 2")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    loss_history = []
    print_every = 100
    current_lr = learning_rate_high
    
    for epoch in range(train_epochs):
        optimizer.zero_grad()
        
        # Forward pass - get predictions, states, and time from model
        pred_train, label_train, x_current, t_current = integrator.integrate(integration_args)
        
        # Compute m(t) from the network: m(t) = m_network(u1, u2, u3, t)
        m_t_train = model.m_network(torch.cat([x_current, t_current], dim=1))
        
        # State dynamics loss: ||ẋ_target - (FEX(x) + m_t)||
        mse_state_train = mse_loss_fn(pred_train, label_train)
        
        # RESIDUAL MATCHING LOSS: Ensure m_t = r_t = ẋ_t - FEX(x_t)
        # According to the formulation: ẋ_t = FEX(x_t) + m_t, so m_t should equal the residual
        # This enforces that m_t captures the unexplained residual, not just any value
        # Compute FEX(x_t) = pred_train - m_t_train (deterministic part, since pred_train = FEX(x_t) + m_t)
        # Then r_t = label_train - FEX(x_t) should equal m_t_train
        fex_deterministic = pred_train - m_t_train  # FEX(x_t) without m_t
        residual_actual = label_train - fex_deterministic  # r_t = ẋ_t - FEX(x_t)
        mse_residual = mse_loss_fn(residual_actual, m_t_train)  # ||r_t - m_t||^2
        
        # Debug: Print residual matching info
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"  [Residual Matching Epoch {epoch+1}]:")
            print(f"    Residual r_t = ẋ_t - FEX(x_t) sample: {residual_actual[:5].detach().cpu().flatten()}")
            print(f"    m_t sample: {m_t_train[:5].detach().cpu().flatten()}")
            print(f"    Residual loss: {mse_residual.item():.6f}")
            print(f"    Residual mean: {residual_actual.mean().item():.6f}, m_t mean: {m_t_train.mean().item():.6f}")
            print(f"    Residual std: {residual_actual.std().item():.6f}, m_t std: {m_t_train.std().item():.6f}")
        
        # OU process evolution loss
        # Use autograd to compute dm/dt from the network: dm/dt = d(m_network(x, t))/dt
        # Then compare with Force_FEX(m(t)): dm/dt should equal Force_FEX(m(t))
        dm_dt_autograd = model.compute_dm_dt(x_current, t_current)
        dm_dt_pred = model.Force_FEX(m_t_train)
        mse_ou_train = mse_loss_fn(dm_dt_autograd, dm_dt_pred)
        
        # Debug output
        if epoch == 0 or (epoch + 1) % 1000 == 0:
            print(f"  [OU Loss Epoch {epoch+1}]:")
            print(f"    m_t sample: {m_t_train[:5].detach().cpu().flatten()}")
            print(f"    dm_dt (autograd) sample: {dm_dt_autograd[:5].detach().cpu().flatten()}")
            print(f"    dm_dt (Force_FEX) sample: {dm_dt_pred[:5].detach().cpu().flatten()}")
            print(f"    OU loss: {mse_ou_train.item():.6f}")
            
            # Check expected OU formula: dm/dt ≈ -0.5*m (deterministic part)
            m_sample = m_t_train[0].item()
            expected_dm_dt = -0.5 * m_sample
            actual_dm_dt_autograd = dm_dt_autograd[0].item()
            actual_dm_dt_force = dm_dt_pred[0].item()
            print(f"    For m={m_sample:.6f}:")
            print(f"      Expected (deterministic): dm/dt ≈ -0.5*m = {expected_dm_dt:.6f}")
            print(f"      Actual (autograd): {actual_dm_dt_autograd:.6f}")
            print(f"      Actual (Force_FEX): {actual_dm_dt_force:.6f}")
            print(f"      Ratio autograd/expected: {actual_dm_dt_autograd/expected_dm_dt if expected_dm_dt != 0 else 'inf':.2f}x")
            print(f"      Ratio Force_FEX/expected: {actual_dm_dt_force/expected_dm_dt if expected_dm_dt != 0 else 'inf':.2f}x")
        
        # Legacy code removed - no longer using ground truth tmM or m_t_next approach
        if False and use_ground_truth_tmM:
            # SCENARIO 1: tmM is KNOWN (ground truth available)
            # Use ground truth tmM values to learn OU process formula directly
            # This is easier and more accurate
            
            # Extract ground truth tmM values matching the time steps
            batch_size = dataset_tensor.shape[0]
            time_steps = dataset_tensor.shape[2]  # e.g., 1001
            tmM_shape = params['tmM'].shape[0]  # e.g., 1000
            
            # We have time_steps-1 pairs from dataset, but tmM has tmM_shape points
            num_pairs = min(time_steps - 1, tmM_shape - 1)
            
            # Extract tmM values for consecutive pairs
            tmM_current_flat = params['tmM'][:num_pairs, 0].flatten()  # Shape: (num_pairs,)
            tmM_next_flat = params['tmM'][1:num_pairs+1, 0].flatten()  # Shape: (num_pairs,)
            
            # Tile for batch dimension to match pred_train shape
            tmM_current_tiled = np.tile(tmM_current_flat, batch_size)  # Shape: (batch*num_pairs,)
            tmM_next_tiled = np.tile(tmM_next_flat, batch_size)
            
            # Convert to tensors
            tmM_current = torch.from_numpy(tmM_current_tiled).float().reshape(-1, 1).to(pred_train.device)
            tmM_next = torch.from_numpy(tmM_next_tiled).float().reshape(-1, 1).to(pred_train.device)
            
            # Truncate pred_train and label_train if needed to match tmM pairs
            if num_pairs < time_steps - 1:
                pairs_per_batch = num_pairs
                pred_train = pred_train[:batch_size * pairs_per_batch]
                label_train = label_train[:batch_size * pairs_per_batch]
                mse_state_train = mse_loss_fn(pred_train, label_train)
            
            # Verify shapes match
            assert tmM_current.shape[0] == tmM_next.shape[0] == pred_train.shape[0], \
                f"Shape mismatch: tmM_current {tmM_current.shape}, tmM_next {tmM_next.shape}, pred_train {pred_train.shape}"
            
            # Compute target dm/dt from ground truth tmM
            # OU process: dm/dt = -theta*m + sigma*dW/dt, where theta = 0.5
            # So (tmM_next - tmM_current)/dt = -0.5*tmM_current + noise
            # 
            # PROBLEM: The noise term has variance sigma^2/dt = 1.0/0.01 = 100
            # This means noise std ≈ 10, which is MUCH larger than the deterministic part
            # So the target is dominated by noise, not the deterministic -0.5*m term
            #
            # SOLUTION: Use a moving average or filter to reduce noise, OR
            # learn from the deterministic part directly by using regression
            dm_dt_actual_train = (tmM_next - tmM_current) / dt_actual
            
            # Debug: Check if target matches expected OU formula
            if epoch == 0 or (epoch + 1) % 1000 == 0:
                # For a few samples, check if dm/dt ≈ -0.5*m
                sample_idx = 0
                m_sample = tmM_current[sample_idx].item()
                dm_dt_expected = -0.5 * m_sample
                dm_dt_actual_sample = dm_dt_actual_train[sample_idx].item()
                noise_std = 1.0 / np.sqrt(dt_actual)  # sigma/sqrt(dt) = 1.0/sqrt(0.01) ≈ 10
                
                # Check the magnitude of noise vs deterministic part
                noise_magnitude = abs(dm_dt_actual_sample - dm_dt_expected)
                deterministic_magnitude = abs(dm_dt_expected)
                
                print(f"  [OU Debug Epoch {epoch+1}] For m={m_sample:.6f}:")
                print(f"    Expected dm/dt (deterministic): {dm_dt_expected:.6f}")
                print(f"    Actual (tmM_next-tmM)/dt: {dm_dt_actual_sample:.6f}")
                print(f"    Noise magnitude: {noise_magnitude:.6f}")
                print(f"    Deterministic magnitude: {deterministic_magnitude:.6f}")
                print(f"    Noise/Deter ratio: {noise_magnitude/deterministic_magnitude if deterministic_magnitude > 0 else 'inf':.2f}x")
                print(f"    Noise std: {noise_std:.2f}")
                
                # Check overall statistics
                dm_dt_actual_np = dm_dt_actual_train.detach().cpu().numpy().flatten()
                dm_dt_expected_np = -0.5 * tmM_current.detach().cpu().numpy().flatten()
                noise_np = dm_dt_actual_np - dm_dt_expected_np
                
                print(f"  [OU Statistics] Across all samples:")
                print(f"    Deterministic part: mean={dm_dt_expected_np.mean():.6f}, std={dm_dt_expected_np.std():.6f}")
                print(f"    Noise part: mean={noise_np.mean():.6f}, std={noise_np.std():.6f}")
                print(f"    Total target: mean={dm_dt_actual_np.mean():.6f}, std={dm_dt_actual_np.std():.6f}")
                print(f"    Noise std / Deterministic std: {noise_np.std()/dm_dt_expected_np.std() if dm_dt_expected_np.std() > 0 else 'inf':.2f}x")
                print(f"  [WARNING] Noise dominates! OU loss will be large because noise is unpredictable.")
            
            # Train Force_FEX to predict dm/dt from ground truth tmM_current
            # PROBLEM: The target (tmM_next - tmM_current)/dt includes large noise (std ≈ 10)
            # This noise dominates the deterministic part (-0.5*m), causing the model
            # to learn the wrong coefficient (e.g., -0.993*m instead of -0.5*m)
            #
            # SOLUTION: Use a regression approach to fit the deterministic part
            # We want to learn: dm/dt = -theta*m, where theta = 0.5
            # Instead of fitting noisy (tmM_next - tmM_current)/dt directly,
            # we can use least squares regression: minimize ||dm/dt + theta*m||^2
            #
            # However, since we're using a neural network (Force_FEX), we still
            # need to train it. The key is that with enough data, the noise should
            # average out and the model should learn -0.5*m.
            #
            # If it's learning -0.993*m instead, it might be because:
            # 1. The noise is too large relative to the deterministic part
            # 2. There's not enough data for the noise to average out
            # 3. The model architecture can't express -0.5*m exactly
            #
            # Let's try using the noisy target but with more training
            dm_dt_pred_train = model.Force_FEX(tmM_current)
            mse_ou_train = mse_loss_fn(dm_dt_actual_train, dm_dt_pred_train)
            
            # Debug: Why is OU loss so large?
            if (epoch + 1) % 1000 == 0:
                # The OU loss is large because:
                # 1. Target includes noise with std ≈ 10
                # 2. Model can only predict deterministic part (-0.5*m)
                # 3. Residual = noise (unpredictable) → large MSE
                with torch.no_grad():
                    residual = dm_dt_actual_train - dm_dt_pred_train
                    residual_np = residual.detach().cpu().numpy().flatten()
                    print(f"  [OU Loss Analysis Epoch {epoch+1}]:")
                    print(f"    OU loss: {mse_ou_train.item():.6f}")
                    print(f"    Residual mean: {residual_np.mean():.6f}, std: {residual_np.std():.6f}")
                    print(f"    Expected noise std: {1.0/np.sqrt(dt_actual):.2f}")
                    print(f"    Residual std ≈ noise std? {abs(residual_np.std() - 1.0/np.sqrt(dt_actual)) < 2.0}")
                    print(f"    [Explanation] OU loss is large because:")
                    print(f"      - Target includes noise (std ≈ 10)")
                    print(f"      - Model predicts deterministic part only")
                    print(f"      - Residual = noise (unpredictable) → large MSE")
                    print(f"    [Note] This is expected! The model can't predict noise.")
            
        
        # Debug: Check Force_FEX parameters and output
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"  [Debug Epoch {epoch+1}] ========================================")
            print(f"  [Debug Epoch {epoch+1}] Force_FEX weights: w1={model.Force_FEX.force_weight_1.item():.4f}, w2={model.Force_FEX.force_weight_2.item():.4f}, w3={model.Force_FEX.force_weight_3.item():.4f}")
            print(f"  [Debug Epoch {epoch+1}] Force_FEX biases: b1={model.Force_FEX.force_bias_1.item():.4f}, b2={model.Force_FEX.force_bias_2.item():.4f}, b3={model.Force_FEX.force_bias_3.item():.4f}")
            print(f"  [Debug Epoch {epoch+1}] m_t (from network) sample: {m_t_train[:5].detach().cpu().flatten()}")
            print(f"  [Debug Epoch {epoch+1}] Expected OU formula: dm/dt ≈ -0.5*m (deterministic part)")
            print(f"  [Debug Epoch {epoch+1}] ========================================")
        
        # Total loss
        # Total loss: state dynamics + residual matching + OU process evolution
        # Weight the residual loss to balance with other terms
        residual_weight = 1.0  # Can be tuned if needed
        total_loss_train = mse_state_train + residual_weight * mse_residual + mse_ou_train
        
        # Adjust learning rate based on loss
        if total_loss_train.item() > 2.0:
            if current_lr != learning_rate_high:
                current_lr = learning_rate_high
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate_high
                print(f"  [LR Change at Epoch {epoch + 1}] Loss > 2, switching to LR = {learning_rate_high}")
        else:
            if current_lr != learning_rate_low:
                current_lr = learning_rate_low
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate_low
                print(f"  [LR Change at Epoch {epoch + 1}] Loss <= 2, switching to LR = {learning_rate_low}")
        
        # Backward pass
        total_loss_train.backward()
        optimizer.step()
        
        # Log loss periodically
        if (epoch + 1) % print_every == 0 or epoch == 0:
            loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss_train.item(),
                'mse_state': mse_state_train.item(),
                'mse_residual': mse_residual.item(),
                'mse_ou': mse_ou_train.item(),
                'learning_rate': current_lr,
            })
            
            print(f"  Epoch {epoch + 1:6d}/{train_epochs}: Total Loss = {total_loss_train.item():.6f} "
                  f"(State: {mse_state_train.item():.6f}, Residual: {mse_residual.item():.6f}, OU: {mse_ou_train.item():.6f}, LR: {current_lr:.4f})")
            
            # Print simplified formula every 100 epochs
            try:
                simplified_expr = model.expression_visualize_simplified()
                print(f"    Formula: {simplified_expr}")
            except Exception as e:
                print(f"    [Warning] Could not generate formula: {e}")
        
        # Store loss for final epoch
        elif epoch == train_epochs - 1:
            loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss_train.item(),
                'mse_state': mse_state_train.item(),
                'mse_ou': mse_ou_train.item(),
                'learning_rate': current_lr,
            })
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation After Training")
    print(f"{'='*80}")
    
    with torch.no_grad():
        pred_final, label_final, x_current_final, t_current_final = integrator.integrate(integration_args)
        
        mse_state_final = mse_loss_fn(pred_final, label_final)
        
        # Compute m(t) from the network
        m_t_final = model.m_network(torch.cat([x_current_final, t_current_final], dim=1))
        
        # Compute dm/dt using autograd
        dm_dt_autograd_final = model.compute_dm_dt(x_current_final, t_current_final)
        
        # Compute dm/dt from Force_FEX
        dm_dt_pred_final = model.Force_FEX(m_t_final)
        
        mse_ou_final = mse_loss_fn(dm_dt_autograd_final, dm_dt_pred_final)
        total_loss_final = mse_state_final + mse_ou_final
        
        relative_error_final = torch.mean(torch.abs(pred_final - label_final) / (torch.abs(label_final) + 1e-8))
        
        print(f"\n[Final Results]")
        print(f"  Total Loss: {total_loss_final.item():.6f} (was {total_loss.item():.6f}, improvement: {total_loss.item() - total_loss_final.item():.6f})")
        print(f"  MSE (State): {mse_state_final.item():.6f} (was {mse_state.item():.6f})")
        print(f"  MSE (OU): {mse_ou_final.item():.6f} (was {mse_ou.item():.6f})")
        print(f"  Relative Error: {relative_error_final.item():.6f}")
        print(f"  dt used: {integrator._integratorparams.dt}")
        print(f"  dm_dt (autograd) mean: {dm_dt_autograd_final.mean():.6f}")
        print(f"  dm_dt (Force_FEX) mean: {dm_dt_pred_final.mean():.6f}")
        
        # Show final formula
        try:
            simplified_expr = model.expression_visualize_simplified()
            print(f"\n  Final Formula: {simplified_expr}")
            
            # Check if OU formula matches expected -0.5*m
            force_expr = model.Force_FEX.expression_visualize_simplified()
            print(f"  OU Process Formula (dm/dt): {force_expr}")
            print(f"  Expected: -0.5*m (or close to it)")
        except Exception as e:
            print(f"  [Warning] Could not generate formula: {e}")
    
    print(f"\n{'='*80}")
    print("Test Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Test with original FEX learning the OU process formula from data
    print("="*80)
    print("Testing with FEX learning OU process formula from data")
    print("Force_FEX will learn dm/dt = Force_FEX(m_t) from (m_t_next - m_t)/dt")
    print("="*80)
    test_fex_dim1_ground_truth(use_ground_truth_tmM=False)
