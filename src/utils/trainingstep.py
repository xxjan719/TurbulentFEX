# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass
import torch
from torch import Tensor
from utils.FEX import FEX

@dataclass
class Body4TrainIntegrationParams:
    dt: float

@dataclass
class Body4TrainIntegrationArgs:
    integration_func: callable
    y0: Tensor
    index: int

class Body4TrainIntegrator:
    def __init__(self, integratorParams: Body4TrainIntegrationParams):
        self._integratorparams = integratorParams
    
    def integrate(self, integrationArgs: Body4TrainIntegrationArgs) -> Tensor:
        trainingset = integrationArgs.y0
        integration_func = integrationArgs.integration_func
        index = integrationArgs.index
        state = trainingset.clone()
        next_state = trainingset[:,:,1:]
        current_state = trainingset[:,:,:-1]
        u1 = current_state[:,0,:]     
        u2 = current_state[:,1,:]
        u3 = current_state[:,2,:]

        print(u1.shape)
        u1_flat = u1.reshape(-1, 1)
        u2_flat = u2.reshape(-1, 1)
        u3_flat = u3.reshape(-1, 1)
        u_flat = torch.cat([u1_flat, u2_flat, u3_flat], dim=1)
        ui_next = next_state[:,index,:]
        ui = current_state[:,index,:]
        ui_next_flat = ui_next.reshape(-1, 1)
        ui_flat = ui.reshape(-1, 1)
        
        print(u_flat.shape)
        
        label = (ui_next_flat - ui_flat)/self._integratorparams.dt
        expression_pred = integration_func(u_flat)
        print(expression_pred.shape)



        return expression_pred, label





if __name__ == "__main__":
    op_seqs = [2,0,3,2,
            4,2,5,2,
            6,1,7,2]
    model = FEX(op_seqs)
    integratorParams = Body4TrainIntegrationParams(
    dt=10**-3,)
    x = torch.randn(10**4,3,10**4+1)
    integration_args = Body4TrainIntegrationArgs(y0=x, integration_func=model)
    integrator = Body4TrainIntegrator(integratorParams)
    
    integrator.integrate(integration_args)
    