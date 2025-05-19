# a decorator takes a function, extends it and returns.
# a function can return a function
from dataclasses import dataclass

@dataclass
class Body4TrainIntegrationParams:
    df: float
    intermediate_dt: float
    @property
    def num_intermediate_steps(self) -> float:
        result = self.dt/self.intermediate_dt
        if math.isclose(result,round(result), rel_tol = 1e-09):
            return int(result)
        else:
            print(f"dt {self.dt} is not divisible by intermediate_dt {self.intermediate_dt} are not nice multiples.")
            int_result = int(result)
        return int_result

@dataclass
class Body4TrainIntegrationArgs:
    derivative_func: callable
    y0: Tensor


class Body4TrainIntegrator:
    def __init__(self, integratorParams: Body4TrainIntegrationParams):
        self._integratorparams = integratorParams
    
    def integrate(self, integrationArgs: Body4TrainIntegrationArgs) -> Tensor:
        # Unpack the arguments
        derivative_func = integrationArgs.derivative_func
        y0 = integrationArgs.y0
        u1 = y0[:,:,0]
        u1_
        u2 = y0[:,:,1]
        u3 = y0[:,:,2]
        dt = self._integratorparams_intermediate_dt
        num_intermediate_steps = self._integratorparams.num_intermediate_steps

        for j in range(num_intermediate_steps):

        # Initialize the state
        state = y0.clone()



        return state





if __name__ == "__main__":
    