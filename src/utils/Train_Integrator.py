# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
from typing import Callable, Tuple
import torch
from torch import Tensor

try:
    from .FEX import FEX
except:
    from FEX import FEX

@dataclass
class Body4TrainIntegrationParams:
    dt: float

@dataclass
class Body4TrainIntegrationArgs:
    integration_func: Callable  # Can be a single model or a dict of models
    y0: Tensor
    index: int  # Can be 1, 2, 3 for single dimension, or 'all' for all dimensions

class Body4TrainIntegrator:
    def __init__(self, integratorParams: Body4TrainIntegrationParams, method: str = "integration-based"):
        """
        Initialize the integrator
        Args:
            integratorParams: Integration parameters
            method: Integration method to use. Options:
                   - "derivative-based": Uses derivative-based method (ui_next - ui)/dt
                   - "integration-based": Uses Runge-Kutta 2nd order (RK2) integration
        """
        self._integratorparams = integratorParams
        self.method = method.lower()
        if self.method not in ["derivative-based", "integration-based"]:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
    
    def integrate(self, integrationArgs: Body4TrainIntegrationArgs) -> Tuple[Tensor, Tensor]:
        trainingset = integrationArgs.y0
        integration_func = integrationArgs.integration_func
        index = integrationArgs.index
        state = trainingset.clone()
        next_state = state[:,:,1:]
        current_state = state[:,:,:-1]
        u1 = current_state[:,0,:]     
        u2 = current_state[:,1,:]
        u3 = current_state[:,2,:]

        u1_flat = u1.reshape(-1, 1)
        u2_flat = u2.reshape(-1, 1)
        u3_flat = u3.reshape(-1, 1)
        u_flat = torch.cat([u1_flat, u2_flat, u3_flat], dim=1)
        
        # Check if we're handling all dimensions or a single dimension
        if index == 'all' or index is None:
            # Handle all dimensions simultaneously
            ui_next = next_state  # Shape: (batch, dim, time)
            ui = current_state    # Shape: (batch, dim, time)
            ui_next_flat = ui_next.reshape(-1, 3)  # Shape: (batch*time, dim)
            ui_flat = ui.reshape(-1, 3)            # Shape: (batch*time, dim)
        else:
            # Handle single dimension (original behavior)
            ui_next = next_state[:,index-1,:]
            ui = current_state[:,index-1,:]
            ui_next_flat = ui_next.reshape(-1, 1)
            ui_flat = ui.reshape(-1, 1)
        
        if self.method == "derivative-based":
            # Derivative-based method
            if index == 'all' or index is None:
                # For all dimensions, we need to compute derivatives for each dimension
                label = (ui_next_flat - ui_flat)/self._integratorparams.dt
                # Check if integration_func is a dict of models or a single model
                if isinstance(integration_func, dict):
                    # Use the dictionary of models
                    expression_pred = torch.zeros_like(label)
                    for dim_idx in range(3):
                        model = integration_func[str(dim_idx + 1)]
                        expression_pred[:, dim_idx:dim_idx+1] = model(u_flat)
                else:
                    # Use a single model (fallback)
                    expression_pred = torch.zeros_like(label)
                    for dim_idx in range(3):
                        expression_pred[:, dim_idx:dim_idx+1] = integration_func(u_flat)
            else:
                # Single dimension (original behavior)
                label = (ui_next_flat - ui_flat)/self._integratorparams.dt
                expression_pred = integration_func(u_flat)
                
        elif self.method == "integration-based":  # RK2 method
            dt = self._integratorparams.dt
            derivative_func = integration_func
            
            if index == 'all' or index is None:
                # Proper RK2 for all dimensions simultaneously
                # This is the correct way to handle coupled systems
                
                # Step 1: Compute k1 for all dimensions
                k1_all = torch.zeros_like(u_flat)
                if isinstance(integration_func, dict):
                    # Use the dictionary of models
                    for dim_idx in range(3):
                        model = integration_func[str(dim_idx + 1)]
                        k1_all[:, dim_idx:dim_idx+1] = model(u_flat)
                else:
                    # Use a single model (fallback)
                    for dim_idx in range(3):
                        k1_all[:, dim_idx:dim_idx+1] = integration_func(u_flat)
                
                # Step 2: Update full state vector for k2 calculation
                u_updated = u_flat + dt * k1_all
                
                # Step 3: Compute k2 for all dimensions
                k2_all = torch.zeros_like(u_flat)
                if isinstance(integration_func, dict):
                    # Use the dictionary of models
                    for dim_idx in range(3):
                        model = integration_func[str(dim_idx + 1)]
                        k2_all[:, dim_idx:dim_idx+1] = model(u_updated)
                else:
                    # Use a single model (fallback)
                    for dim_idx in range(3):
                        k2_all[:, dim_idx:dim_idx+1] = integration_func(u_updated)
                
                # Step 4: Apply RK2 formula for all dimensions
                expression_pred = ui_flat + (dt/2.0) * (k1_all + k2_all)
                label = ui_next_flat
                
            else:
                # Single dimension RK2 (previous implementation)
                # Step 1: Compute k1 for the current dimension
                k1 = derivative_func(u_flat)
                
                # Step 2: For k2, we need to estimate the full state update
                # Since we're training one dimension at a time, we'll use a simplified approach
                # that assumes the other dimensions change proportionally to their current values
                # This is a reasonable approximation for coupled systems
                
                # Create updated state by advancing all dimensions proportionally
                u_updated = u_flat.clone()
                
                # Update the current dimension with the computed derivative
                u_updated[:, index-1] = u_flat[:, index-1] + dt * k1.flatten()
                
                # For other dimensions, use a simple proportional update
                # This assumes the coupling is captured by the current state values
                for dim_idx in range(3):
                    if dim_idx != index - 1:
                        # Use a small proportional change based on current values
                        # This is a heuristic for coupled systems
                        u_updated[:, dim_idx] = u_flat[:, dim_idx] * (1.0 + 0.1 * dt)
                
                # Step 3: Compute k2 using the updated state
                k2 = derivative_func(u_updated)
                
                # Step 4: Apply RK2 formula
                expression_pred = ui_flat + (dt/2.0) * (k1 + k2)
                label = ui_next_flat
        else:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
        return expression_pred, label





if __name__ == "__main__":
    op_seqs = torch.tensor([2,0,3,2,
            4,2,5,2,
            6,1,7,2])
    model = FEX(op_seqs, dim=3)
    integratorParams = Body4TrainIntegrationParams(
    dt=10**-2,)
    x = torch.randn(10**4,3,10**2+1)
    integration_args = Body4TrainIntegrationArgs(y0=x, integration_func=model, index=1)
    integrator = Body4TrainIntegrator(integratorParams, method="integration-based")
    
    expression_pred, label = integrator.integrate(integration_args)
    print(expression_pred.shape)
    print(label.shape)
    