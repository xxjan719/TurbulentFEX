import torch
import torch.nn as nn
import torch.optim
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.FEX_with_force import FEX_with_force
from utils.Train_Integrator import Body4TrainIntegrationArgs, Body4TrainIntegrationParams, Body4TrainIntegrator
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
from config import DIR_EXAMPLE

def test_fex_dim1_ground_truth():
    """
    Test and optimize FEX_with_force model for dimension 1 using random_cascade_deterministic dataset
    """

    print("="*80)
    print("FEX_with_force Test for Dimension 1 - random_cascade_deterministic Dataset")
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
        
        # Check if tmM follows the expected decay pattern
        time_points = np.arange(tmM.shape[0]) * dt
        tmM_first_component = tmM[:, 0]  # First component
        
        # Expected deterministic solution: 1.5 * exp(-0.5*t)
        expected_deterministic = 1.5 * np.exp(-0.5 * time_points)
        
        print(f"[INFO] Checking tmM decay pattern:")
        print(f"  tmM[0] = {tmM_first_component[0]:.6f} (expected: {expected_deterministic[0]:.6f})")
        print(f"  tmM[100] = {tmM_first_component[100]:.6f} (expected: {expected_deterministic[100]:.6f})")
        print(f"  tmM[500] = {tmM_first_component[500]:.6f} (expected: {expected_deterministic[500]:.6f})")
        print(f"  tmM[-1] = {tmM_first_component[-1]:.6f} (expected: {expected_deterministic[-1]:.6f})")
        
        # Check correlation with expected decay
        correlation = np.corrcoef(tmM_first_component, expected_deterministic)[0, 1]
        print(f"  Correlation with exp(-0.5*t): {correlation:.6f}")
        
        # Try to fit exp(alpha*t) to see what exponent the data suggests
        # Use log-linear regression: log(tmM) = log(A) + alpha*t
        # Only use positive values for log
        positive_mask = tmM_first_component > 0
        if np.sum(positive_mask) > 10:
            log_tmM = np.log(tmM_first_component[positive_mask])
            t_positive = time_points[positive_mask]
            # Linear regression: log(y) = alpha*t + log(A)
            coeffs = np.polyfit(t_positive, log_tmM, 1)
            fitted_alpha = coeffs[0]
            fitted_A = np.exp(coeffs[1])
            print(f"  Fitted to exp(alpha*t): alpha = {fitted_alpha:.6f}, A = {fitted_A:.6f}")
            print(f"  Expected: alpha = -0.5, A = 1.5")

    # Use single operator sequence for dimension 1 testing (13 operators: 12 state + 1 time)
    test_op_seq = [1, 1, 2, 1,    # x1 operators
                   1, 1, 2, 2,    # x2 operators
                   0, 1, 2, 2,    # x3 operators
                   6]             # time operator

    print(f"\n{'='*80}")
    print(f"Testing Operator Sequence: {test_op_seq}")
    print(f"{'='*80}")

    op_seqs = torch.tensor(test_op_seq)
    model = FEX_with_force(op_seqs, dim=3)
    
    # Initialize force parameters to help learn exp(-0.5*t) pattern
    # Expected: 1.5*exp(-0.5*t) = exp(-0.5*t + ln(1.5))
    with torch.no_grad():
        # Use same dtype as model parameters
        model.force_a.data = torch.tensor([-1.0], dtype=model.force_a.dtype, device=model.force_a.device)
        model.force_b.data = torch.tensor([np.log(1.5)], dtype=model.force_b.dtype, device=model.force_b.device)
    print(f"[INFO] Initialized force_a = {model.force_a.item():.6f} (will learn toward -0.5)")
    print(f"[INFO] Initialized force_b = {model.force_b.item():.6f} (expected: {np.log(1.5):.6f})")

    # Setup integrator (same as in 1stage_deterministic.py)
    integrator_params = Body4TrainIntegrationParams(dt=dt)
    integrator = Body4TrainIntegrator(integrator_params, method="integration-based")

    integration_args = Body4TrainIntegrationArgs(
        y0=dataset_tensor,
        integration_func=model,
        index=1,  # Test dimension 1
        params_name=params_name
    )

    # Initial evaluation
    print(f"\n[Initial Evaluation]")
    with torch.no_grad():
        pred, label = integrator.integrate(integration_args)
        mse_state = nn.MSELoss()(pred, label)
        total_loss = mse_state

        print(f"  Total Loss: {total_loss.item():.6f}")
        print(f"  MSE (State): {mse_state.item():.6f}")

        # Show initial formula
        try:
            simplified_expr = model.expression_visualize_simplified()
            print(f"\n  Initial Formula: {simplified_expr}")
        except Exception as e:
            print(f"  [Warning] Could not generate formula: {e}")
    
    # Test with exact ground truth formula: -1*x1 + 2*x2*x3 + 1.5*exp(-0.5*t)
    print(f"\n{'='*80}")
    print("Testing with Exact Ground Truth Formula")
    print(f"{'='*80}")
    print("Expected: -1*x1 + 2*x2*x3 + 0*x2 + 0*x3 + 1.5*exp(-0.5*t)")
    
    # Create a simple function to compute the exact formula
    def exact_formula_derivative(u_flat, t_flat):
        """
        Compute exact derivative: -1*x1 + 2*x2*x3 + 1.5*exp(-0.5*t)
        u_flat: (batch*time, 3) - state variables
        t_flat: (batch*time, 1) - time
        """
        x1 = u_flat[:, 0:1]
        x2 = u_flat[:, 1:2]
        x3 = u_flat[:, 2:3]
        t = t_flat.squeeze(-1) if t_flat.dim() > 1 else t_flat
        
        # Exact formula for dimension 1: du1/dt = -1*x1 + 2*x2*x3 + 1.5*exp(-0.5*t)
        result = -1.0 * x1 + 2.0 * x2 * x3 + 1.5 * torch.exp(-0.5 * t).unsqueeze(-1)
        return result
    
    # Get the input data structure (same as integrator)
    with torch.no_grad():
        current_state = dataset_tensor[:, :, :-1]
        next_state = dataset_tensor[:, :, 1:]
        
        u1 = current_state[:, 0, :]
        u2 = current_state[:, 1, :]
        u3 = current_state[:, 2, :]
        
        u1_flat = u1.reshape(-1, 1)
        u2_flat = u2.reshape(-1, 1)
        u3_flat = u3.reshape(-1, 1)
        
        # Generate time vector (same as integrator)
        num_time_steps = current_state.shape[2]
        time_steps = torch.arange(num_time_steps, dtype=torch.float32) * dt
        time_flat = time_steps.unsqueeze(0).expand(current_state.shape[0], -1).reshape(-1, 1)
        
        u_flat = torch.cat([u1_flat, u2_flat, u3_flat], dim=1)
        
        # Compute exact derivative: f(u_i) = -1*x1 + 2*x2*x3 + 1.5*exp(-0.5*t)
        exact_derivative = exact_formula_derivative(u_flat, time_flat)
        
        # For integration-based method: u_{i+1} = u_i + dt * f(u_i)
        ui_flat = u1_flat  # Current u1 values
        exact_pred_integrated = ui_flat + dt * exact_derivative
        
        # Get the label (next state u1)
        ui_next_flat = next_state[:, 0, :].reshape(-1, 1)
        
        # Compute loss with exact formula
        exact_loss = nn.MSELoss()(exact_pred_integrated, ui_next_flat)
        
        print(f"\n[Exact Formula Test]")
        print(f"  Loss with exact formula: {exact_loss.item():.6f}")
        print(f"  This is the theoretical minimum loss achievable")
        print(f"  Current model loss: {total_loss.item():.6f}")
        print(f"  Difference: {total_loss.item() - exact_loss.item():.6f}")
        if exact_loss.item() > 1e-6:
            print(f"  [WARNING] Exact formula loss is not zero! This suggests:")
            print(f"    - The data might not exactly follow the formula")
            print(f"    - There might be numerical integration errors")
            print(f"    - The forcing term tmM might not be exactly 1.5*exp(-0.5*t)")

    # Skip training - just check exact formula loss
    print(f"\n{'='*80}")
    print("Skipping Training - Only Checking Exact Formula Loss")
    print(f"{'='*80}")
    
    # Exit early - don't train
    
    
    # Training loop (following 1stage_deterministic.py pattern)
    print(f"\n{'='*80}")
    print("Training Model with Adam...")
    print(f"{'='*80}")

    train_epochs = 50000
    learning_rate = 0.01  # Starting learning rate
    min_learning_rate = 0.0  # Minimum learning rate (decay to zero)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()

    print(f"  Training epochs: {train_epochs}")
    print(f"  Learning rate: {learning_rate} (will decay to {min_learning_rate})")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_history = []
    print_every = 100

    for epoch in range(train_epochs):
        optimizer.zero_grad()

        # Forward pass - get predictions
        pred_train, label_train = integrator.integrate(integration_args)

        # State dynamics loss
        mse_state_train = mse_loss_fn(pred_train, label_train)

        # Total loss
        total_loss_train = mse_state_train

        # Learning rate decay: linear decay from learning_rate to min_learning_rate
        current_lr = learning_rate * (1 - epoch / train_epochs) + min_learning_rate * (epoch / train_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Backward pass
        total_loss_train.backward()
        optimizer.step()

        # Log loss periodically
        if (epoch + 1) % print_every == 0 or epoch == 0:
            loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss_train.item(),
                'mse_state': mse_state_train.item(),
                'learning_rate': current_lr,
            })

            print(f"  Epoch {epoch + 1:6d}/{train_epochs}: Total Loss = {total_loss_train.item():.6f} "
                  f"(State: {mse_state_train.item():.6f}, LR: {current_lr:.6f})")

            # Print simplified formula every 100 epochs
            try:
                simplified_expr = model.expression_visualize_simplified()
                print(f"    Formula: {simplified_expr}")
                
                # Debug: Check force parameters
                force_a_val = model.force_a.item()
                force_b_val = model.force_b.item()
                print(f"    Force params: force_a = {force_a_val:.6f} (expected: -0.5), force_b = {force_b_val:.6f} (expected: {np.log(1.5):.6f})")
                print(f"    Force function: exp({force_a_val:.6f}*t + {force_b_val:.6f}) = {np.exp(force_b_val):.6f}*exp({force_a_val:.6f}*t)")
            except Exception as e:
                print(f"    [Warning] Could not generate formula: {e}")

        # Store loss for final epoch
        elif epoch == train_epochs - 1:
            loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss_train.item(),
                'mse_state': mse_state_train.item(),
                'learning_rate': current_lr,
            })

    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation After Training")
    print(f"{'='*80}")

    with torch.no_grad():
        pred_final, label_final = integrator.integrate(integration_args)

        mse_state_final = mse_loss_fn(pred_final, label_final)
        total_loss_final = mse_state_final

        relative_error_final = torch.mean(torch.abs(pred_final - label_final) / (torch.abs(label_final) + 1e-8))

        print(f"\n[Final Results]")
        print(f"  Total Loss: {total_loss_final.item():.6f} (was {total_loss.item():.6f}, improvement: {total_loss.item() - total_loss_final.item():.6f})")
        print(f"  MSE (State): {mse_state_final.item():.6f} (was {mse_state.item():.6f})")
        print(f"  Relative Error: {relative_error_final.item():.6f}")

        # Show final formula
        try:
            simplified_expr = model.expression_visualize_simplified()
            print(f"\n  Final Formula: {simplified_expr}")
        except Exception as e:
            print(f"  [Warning] Could not generate formula: {e}")

    print(f"\n{'='*80}")
    print("Test Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Test with FEX_with_force for random_cascade_deterministic
    print("="*80)
    print("Testing with FEX_with_force for random_cascade_deterministic")
    print("="*80)
    test_fex_dim1_ground_truth()