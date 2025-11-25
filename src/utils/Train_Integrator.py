# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import torch
from torch import Tensor

try:
    from .FEX import FEX
    from .FEX_with_force import FEX_with_force
except:
    from FEX import FEX
    from FEX_with_force import FEX_with_force

@dataclass
class Body4TrainIntegrationParams:
    dt: float

@dataclass
class Body4TrainIntegrationArgs:
    integration_func: Callable
    y0: Tensor
    index: int
    params_name: str = "equipart"  # Add parameter name to determine which model to use

class Body4TrainIntegrator:
    def __init__(self, integratorParams: Body4TrainIntegrationParams, method: str = "integration-based"):
        """
        Initialize the integrator
        Args:
            integratorParams: Integration parameters
            method: Integration method to use. Options:
                   - "derivative-based": Uses derivative-based method (ui_next - ui)/dt
                   - "integration-based": Uses Euler integration
        """
        self._integratorparams = integratorParams
        self.method = method.lower()
        if self.method not in ["derivative-based", "integration-based"]:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
    
    def integrate(self, integrationArgs: Body4TrainIntegrationArgs) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        trainingset = integrationArgs.y0
        integration_func = integrationArgs.integration_func
        index = integrationArgs.index
        params_name = integrationArgs.params_name
        state = trainingset.clone()
        next_state = state[:,:,1:]
        current_state = state[:,:,:-1]
        u1 = current_state[:,0,:]     
        u2 = current_state[:,1,:]
        u3 = current_state[:,2,:]

        u1_flat = u1.reshape(-1, 1)
        u2_flat = u2.reshape(-1, 1)
        u3_flat = u3.reshape(-1, 1)
        
        # Determine which model to use based on params_name
        if params_name in ['cascade', 'equipart', 'dual_cascade']:
            # Use regular FEX for these cases - state variables only
            u_flat = torch.cat([u1_flat, u2_flat, u3_flat], dim=1)
            
        elif params_name == 'periodic_cascade':
            # Use FEX_with_force for periodic_cascade - need to add time dimension
            # Generate time vector using current_state structure - much more efficient
            num_time_steps = current_state.shape[2]
            time_steps = torch.arange(num_time_steps, dtype=torch.float32) * self._integratorparams.dt
            # Use the same structure as current_state for time, then reshape
            time_flat = time_steps.unsqueeze(0).expand(current_state.shape[0], -1).reshape(-1, 1)
            u_flat = torch.cat([u1_flat, u2_flat, u3_flat, time_flat], dim=1)

        
        ui_next = next_state[:,index-1,:]
        ui = current_state[:,index-1,:]
        ui_next_flat = ui_next.reshape(-1, 1)
        ui_flat = ui.reshape(-1, 1)
        
        
        if self.method == "derivative-based":
            # Derivative-based method
            label = (ui_next_flat - ui_flat)/self._integratorparams.dt
            expression_pred = integration_func(u_flat)
        elif self.method == "integration-based":  # Euler method
            dt = self._integratorparams.dt
            derivative_func = integration_func
            
            # Euler method: u_{i+1} = u_i + dt * f(u_i)
            k1 = derivative_func(u_flat)
            expression_pred = ui_flat + dt * k1
            label = ui_next_flat
        else:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
        

        return expression_pred, label





if __name__ == "__main__":
    # # Test with regular FEX for equipart case
    op_seqs = torch.tensor([2,0,3,2,
            4,2,5,2,
            6,1,7,2])
    model = FEX(op_seqs, dim=3)
    integratorParams = Body4TrainIntegrationParams(
    dt=10**-2,)
    x = torch.randn(100,3,101)  # Much smaller test data
    integration_args = Body4TrainIntegrationArgs(y0=x, integration_func=model, index=1, params_name="equipart")
    integrator = Body4TrainIntegrator(integratorParams, method="integration-based")
    
    expression_pred, label = integrator.integrate(integration_args)
    print("Regular FEX (equipart):")
    print(f"Expression pred shape: {expression_pred.shape}")
    print(f"Label shape: {label.shape}")
    
    # Test with FEX_with_force for periodic_cascade case
    op_seqs_with_force = torch.tensor([2,0,3,2,
            4,2,5,2,
            6,1,7,2, 6])  # Add force operator at the end
    model_with_force = FEX_with_force(op_seqs_with_force, dim=3)
    integration_args_force = Body4TrainIntegrationArgs(y0=x, integration_func=model_with_force, index=1, params_name="periodic_cascade")
    
    expression_pred_force, label_force = integrator.integrate(integration_args_force)
    print("\nFEX_with_force (periodic_cascade):")
    print(f"Expression pred shape: {expression_pred_force.shape}")
    print(f"Label shape: {label_force.shape}")
    