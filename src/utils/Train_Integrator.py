# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import torch
from torch import Tensor

try:
    from .FEX import FEX
    from .FEX_with_force import FEX_with_force
    from .FEX_with_random_force import FEX_with_random_force
except:
    from FEX import FEX
    from FEX_with_force import FEX_with_force
    from FEX_with_random_force import FEX_with_random_force

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
        
        
        if params_name == 'random_cascade_deterministic':
            # Use FEX_with_random_force - only state variables (first 12 operators)
            # m(t) is handled internally by the model as a learnable parameter
            
            # Get all states (current and next) - state has shape (batch, dim, time_steps)
            # Flatten to (batch*time_steps, dim) for processing
            # IMPORTANT: Structure should match time steps like periodic_cascade
            # Order: batch 0 (time 0, time 1, ..., time T), batch 1 (time 0, time 1, ..., time T), ...
            u1_all = state[:,0,:]  # Shape: (batch, time_steps)
            u2_all = state[:,1,:]  # Shape: (batch, time_steps)
            u3_all = state[:,2,:]  # Shape: (batch, time_steps)
            
            # Reshape to (batch*time_steps, 1) then concatenate to (batch*time_steps, 3)
            # This preserves the time step ordering: for each batch, all time steps are consecutive
            u1_flat_all = u1_all.reshape(-1, 1)  # Shape: (batch*time_steps, 1)
            u2_flat_all = u2_all.reshape(-1, 1)  # Shape: (batch*time_steps, 1)
            u3_flat_all = u3_all.reshape(-1, 1)  # Shape: (batch*time_steps, 1)
            u_flat_all = torch.cat([u1_flat_all, u2_flat_all, u3_flat_all], dim=1)  # Shape: (batch*time_steps, 3)
            
            # Verify structure: u_flat_all[i] corresponds to batch i//time_steps, time i%time_steps
            # This ensures m_t_expanded[i] corresponds to the same time step
           
            # Call model with all states to create m_t_expanded parameter
            _ = integration_func(u_flat_all)  # Forward pass creates/updates m_t_expanded parameter
            
            # Extract m_t and m_t_next from the parameter
            param_name_all = f'_m_t_expanded_{u_flat_all.shape[0]}'  # Parameter for all states
            
            if hasattr(integration_func, param_name_all):
                m_all = getattr(integration_func, param_name_all)  # Get all m values, shape: (batch*time_steps, 1)
                # Reshape to (batch, time_steps, 1) to properly extract consecutive pairs
                # Structure matches u_flat_all: batch 0 (all time steps), batch 1 (all time steps), ...
                batch_size = state.shape[0]
                time_steps = state.shape[2]
                m_all_reshaped = m_all.reshape(batch_size, time_steps)
                # Extract m_t (time 0 to time_steps-2) and m_t_next (time 1 to time_steps-1)
                # This gives consecutive time step pairs for each batch
                m_t = m_all_reshaped[:, :-1].reshape(-1, 1).clone()  # Shape: (batch*(time_steps-1), 1)
                m_t_next = m_all_reshaped[:, 1:].reshape(-1, 1).clone()  # Shape: (batch*(time_steps-1), 1)
                # Structure: m_t[i] and m_t_next[i] are from the same batch, consecutive time steps
            else:
                # Fallback if parameter doesn't exist
                batch_size = state.shape[0]
                time_steps = state.shape[2]
                m_t = torch.zeros(batch_size * (time_steps - 1), 1, device=u_flat_all.device)
                m_t_next = torch.zeros(batch_size * (time_steps - 1), 1, device=u_flat_all.device)
            
            # Compute expression_pred using current states (all except last time step per batch)
            # Need to match the shape of label: (batch, time_steps-1) -> (batch*(time_steps-1), 1)
            u1_current = u1_all[:, :-1]  # Shape: (batch, time_steps-1)
            u2_current = u2_all[:, :-1]   # Shape: (batch, time_steps-1)
            u3_current = u3_all[:, :-1]  # Shape: (batch, time_steps-1)
            
            # Reshape to (batch*(time_steps-1), 1) then concatenate
            u1_flat_current = u1_current.reshape(-1, 1)
            u2_flat_current = u2_current.reshape(-1, 1)
            u3_flat_current = u3_current.reshape(-1, 1)
            u_flat_current = torch.cat([u1_flat_current, u2_flat_current, u3_flat_current], dim=1)
            # Shape: (batch*(time_steps-1), 3)
            
            expression_pred = integration_func(u_flat_current)
            # Shape: (batch*(time_steps-1), 1)
            
            # Compute label using derivative: (u_next - u_current) / dt
            # Get the specific dimension based on index
            if index == 1:
                ui_current = u1_all[:, :-1]  # Shape: (batch, time_steps-1)
                ui_next = u1_all[:, 1:]      # Shape: (batch, time_steps-1)
            elif index == 2:
                ui_current = u2_all[:, :-1]
                ui_next = u2_all[:, 1:]
            elif index == 3:
                ui_current = u3_all[:, :-1]
                ui_next = u3_all[:, 1:]
            else:
                raise ValueError("Index must be 1, 2, or 3")
            
            # Compute label: (ui_next - ui_current) / dt, then flatten
            label = ((ui_next - ui_current) / self._integratorparams.dt).reshape(-1, 1)
            # Shape: (batch*(time_steps-1), 1)
          
            # Now m_t and m_t_next are from consecutive time steps, so they should be different
            # They can be used to compute (m_t_next - m_t)/dt for training
            return expression_pred, label, m_t, m_t_next
        
        # Determine which model to use based on params_name
        elif params_name in ['cascade', 'equipart', 'dual_cascade']:
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
    
    #Test with FEX_with_random_force for random_cascade_deterministic case
    # 16 operators: 12 for state + 4 for force (m(t) OU process)
    op_seqs_with_random_force = torch.tensor([
        # 12 operators for state variables (4 per dimension)
        1, 1, 2, 1,    # x1 operators
        1, 1, 2, 2,    # x2 operators
        0, 1, 2, 2,    # x3 operators
        # 4 operators for force m(t)
        1, 1, 2, 2     # m(t) operators
    ])
    model_with_random_force = FEX_with_random_force(op_seqs_with_random_force, dim=3)
    
    integration_args_random = Body4TrainIntegrationArgs(
        y0=x, 
        integration_func=model_with_random_force, 
        index=1, 
        params_name="random_cascade_deterministic"
    )
    
    expression_pred_random, label_random, m_t, m_t_next = integrator.integrate(integration_args_random)
    print("\nFEX_with_random_force (random_cascade_deterministic):")
    print(f"Expression pred shape: {expression_pred_random.shape}")
    print(f"Label shape: {label_random.shape}")
    print(f"m_t shape: {m_t.shape if m_t is not None else 'None'}")
    print(f"m_t_next shape: {m_t_next.shape if m_t_next is not None else 'None'}")
    print(f"m_t: {m_t}")
    print(f"m_t_next: {m_t_next}")
    