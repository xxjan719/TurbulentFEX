import torch
import torch.nn as nn
import torch.optim
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.FEX import FEX
from utils.Train_Integrator import Body4TrainIntegrationArgs, Body4TrainIntegrationParams, Body4TrainIntegrator
from Example.MC_triad.MC_triad import params_init, MC_triad_direct, MC_triad_initial_value
from config import DIR_EXAMPLE

def test_fex_dim1_ground_truth():
    """
    Test and optimize FEX model for dimension 1 using random_cascade_deterministic dataset
    """

    print("="*80)
    print("FEX Test for Dimension 1 - random_cascade_deterministic Dataset")
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

    # Use single operator sequence for dimension 1 testing (12 operators for 3 dimensions)
    test_op_seq = [1, 1, 2, 1,    # x1 operators
                   1, 1, 2, 2,    # x2 operators
                   0, 1, 2, 2]    # x3 operators

    print(f"\n{'='*80}")
    print(f"Testing Operator Sequence: {test_op_seq}")
    print(f"{'='*80}")

    op_seqs = torch.tensor(test_op_seq)
    model = FEX(op_seqs, dim=3)

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

        # Forward pass - get predictions
        pred_train, label_train = integrator.integrate(integration_args)

        # State dynamics loss
        mse_state_train = mse_loss_fn(pred_train, label_train)

        # Total loss
        total_loss_train = mse_state_train

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
                'learning_rate': current_lr,
            })

            print(f"  Epoch {epoch + 1:6d}/{train_epochs}: Total Loss = {total_loss_train.item():.6f} "
                  f"(State: {mse_state_train.item():.6f}, LR: {current_lr:.4f})")

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
    # Test with regular FEX for random_cascade_deterministic
    print("="*80)
    print("Testing with FEX for random_cascade_deterministic")
    print("="*80)
    test_fex_dim1_ground_truth()