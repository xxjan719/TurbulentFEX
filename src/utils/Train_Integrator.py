# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
from typing import Callable, Tuple, Union, Dict
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
    integration_func: Union[Callable, Dict[str, Callable]]  # Can be a single model or a dict of models
    y0: Tensor
    index: Union[int, str]  # Can be 1, 2, 3 for single dimension, or 'all' for all dimensions

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
            index_int = int(index)  # Convert to int for indexing
            ui_next = next_state[:,index_int-1,:]
            ui = current_state[:,index_int-1,:]
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
                index_int = int(index)  # Convert to int for indexing
                label = (ui_next_flat - ui_flat)/self._integratorparams.dt
                if isinstance(integration_func, dict):
                    model = integration_func[str(index_int)]
                    expression_pred = model(u_flat)
                else:
                    expression_pred = integration_func(u_flat)
        elif self.method == "integration-based":  # RK2 method
            dt = self._integratorparams.dt
            
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
                # Heun's method: y_{n+1} = y_n + dt * k2
                expression_pred = ui_flat + dt * k2_all
                label = ui_next_flat
                
            else:
                # RK2 method for single dimension training using ground truth data
                index_int = int(index)  # Convert to int for indexing
                
                # Step 1: Compute k1 using current state
                if isinstance(integration_func, dict):
                    model = integration_func[str(index_int)]
                    k1 = model(u_flat)
                else:
                    k1 = integration_func(u_flat)
                
                # Step 2: Compute midpoint values using ground truth data
                # For the dimension being trained: u_mid = u_n + 0.5*dt*k1
                # For other dimensions: interpolate between current and next time step
                u_mid = u_flat.clone()
                
                # Update the dimension being trained with RK2 midpoint
                u_mid[:, index_int-1] = u_flat[:, index_int-1] + 0.5 * dt * k1.flatten()
                
                # For other dimensions, we need to estimate midpoint values without using future data
                # We can use a simple approximation: assume other dimensions change slowly
                # or use a small time step approximation
                for dim_idx in range(3):
                    if dim_idx != index_int - 1:
                        # Option 1: Assume other dimensions remain approximately constant at midpoint
                        # u_mid ≈ u_n (simple but reasonable for small dt)
                        u_mid[:, dim_idx] = u_flat[:, dim_idx]
                        
                        # Option 2: Use a small proportional change based on current values
                        # This is a heuristic that doesn't require future data
                        # u_mid[:, dim_idx] = u_flat[:, dim_idx] * (1.0 + 0.01 * dt)
                
                # Step 3: Compute k2 using midpoint state
                if isinstance(integration_func, dict):
                    model = integration_func[str(index_int)]
                    k2 = model(u_mid)
                else:
                    k2 = integration_func(u_mid)
                
                # Step 4: RK2 prediction for the dimension being trained
                # u_{n+1} = u_n + dt * k2 (this is the standard RK2 formula)
                expression_pred = ui_flat + dt * k2
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