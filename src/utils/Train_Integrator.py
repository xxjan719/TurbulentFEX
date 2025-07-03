# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
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
    integration_func: callable
    y0: Tensor
    index: int

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
    
    def integrate(self, integrationArgs: Body4TrainIntegrationArgs) -> Tensor:
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
        ui_next = next_state[:,index-1,:]
        ui = current_state[:,index-1,:]
        ui_next_flat = ui_next.reshape(-1, 1)
        ui_flat = ui.reshape(-1, 1)
        
        if self.method == "derivative-based":
            # Derivative-based method
            label = (ui_next_flat - ui_flat)/self._integratorparams.dt
            expression_pred = integration_func(u_flat)
        elif self.method == "integration-based":  # RK2 method
            dt = self._integratorparams.dt
            derivative_func = integration_func
            
            # Compute the two k values for RK2
            k1 = derivative_func(u_flat)
            # print(k1.shape)
            # print(u_flat)
            # print(dt*k1)
            if index ==1:
                u_i_next = ui_flat + dt* k1
                u_k2 = torch.cat([u_i_next,u2_flat,u3_flat],dim=1)
                k2 = derivative_func(u_k2)

            elif index ==2:
                u_i_next = ui_flat + dt* k1
                u_k2 = torch.cat([u1_flat,u_i_next,u3_flat],dim=1)
                k2 = derivative_func(u_k2)

            elif index ==3:
                u_i_next = ui_flat + dt* k1
                u_k2 = torch.cat([u1_flat,u2_flat,u_i_next],dim=1)
                k2 = derivative_func(u_k2)
            
            # Compute the next state using RK2 formula
            expression_pred = ui_flat + (dt/2.0) * (k1 + k2)
            label = ui_next_flat
        else:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
        return expression_pred, label





if __name__ == "__main__":
    op_seqs = [2,0,3,2,
            4,2,5,2,
            6,1,7,2]
    model = FEX(op_seqs, dim=3)
    integratorParams = Body4TrainIntegrationParams(
    dt=10**-2,)
    x = torch.randn(10**4,3,10**2+1)
    integration_args = Body4TrainIntegrationArgs(y0=x, integration_func=model, index=1)
    integrator = Body4TrainIntegrator(integratorParams, method="integration-based")
    
    expression_pred, label = integrator.integrate(integration_args)
    print(expression_pred.shape)
    print(label.shape)
    