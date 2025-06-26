import numpy as np
import os

CASE = 'equipart'
dt = 0.01
data = np.load(os.path.join(os.getcwd(), 'src', 'Example', 'MC_triad', 'Results', CASE, 'simulation_results.npz'))
u1 = data['dataset'][:,0]
u2 = data['dataset'][:,1]
u3 = data['dataset'][:,2]
u1_next = u1[:,1:].reshape(-1,1)
u2_next = u2[:,1:].reshape(-1,1)
u3_next = u3[:,1:].reshape(-1,1)
u1_current = u1[:,:-1].reshape(-1,1)
u2_current = u2[:,:-1].reshape(-1,1)
u3_current = u3[:,:-1].reshape(-1,1)
u_current = np.concatenate([u1_current,u2_current,u3_current],axis=1)
u_next = np.concatenate([u1_next,u2_next,u3_next],axis=1)

def FEX_model1(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return -0.2*x1 + 1*x2*x3 + 1*x2 + -2*x3 

def FEX_model2(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return  -0.6*x1*x3 + -1*x1 + -0.1*x2 + -3*x3 

def FEX_model3(x):
    x1 = x[:, 0:1].squeeze(-1)
    x2 = x[:, 1:2].squeeze(-1)
    x3 = x[:, 2:3].squeeze(-1)
    return  -0.4*x1*x2 + 2*x1 + 3*x2 + -0.1*x3 

def FEX_model_check(x):
    return np.stack([FEX_model1(x), FEX_model2(x), FEX_model3(x)], axis=1)

def Buu(B, u, v):
    """Bilinear term B[i]*u[i]*v[i]"""
    return np.array([B[0]*u[:,0]*v[:,0], B[1]*u[:,1]*v[:,1], B[2]*u[:,2]*v[:,2]]).T

# Define parameters for RK4
L = np.array([[0, 1, -2], [-1, 0, -3], [2, 3, 0]])
G = np.diag([0.2, 0.1, 0.1])
B = np.array([1, -0.6, -0.4])
MC = u_current.shape[0]  # number of samples
Dt = dt
i = 1  # time step index

# Initialize params for noise
params = {}
params['req'] = 2.5
params['SS'] = params['req']*np.sqrt(2*G)
params['SSt'] = np.zeros((3,3))
params['tmM'] = np.zeros((1000, 3))  # time modulation (assuming 1000 time steps)

# RK4 integration with stochastic noise
u = u_current.copy()  # initial state

# Generate noise for this step
SS = params['SS'] + params['tmM'][i - 1, :] ** 2 * (params['SSt'] - params['SS'])
noise = np.random.multivariate_normal(mean=[0,0,0], cov=SS, size=MC)

# RK4 steps
k1 = (L @ u.T).T - u @ G + Buu(B, u, u) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
u1 = u + 0.5 * Dt * k1
k2 = (L @ u1.T).T - u1 @ G + Buu(B, u1, u1) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
u2 = u + 0.5 * Dt * k2
k3 = (L @ u2.T).T - u2 @ G + Buu(B, u2, u2) + np.ones((MC, 1)) * params['tmM'][i - 1, :]
u3 = u + Dt * k3
k4 = (L @ u3.T).T - u3 @ G + Buu(B, u3, u3) + np.ones((MC, 1)) * params['tmM'][i - 1, :]

# Final update with deterministic AND stochastic parts
u_rk4 = u + Dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) + np.sqrt(Dt) * noise

u_pred = FEX_model_check(u_current)
print(u_pred.shape)



r1 = (u1_next - u_rk4[:,0].reshape(-1,1))
r2 = (u2_next - u_rk4[:,1].reshape(-1,1))
r3 = (u3_next - u_rk4[:,2].reshape(-1,1))
r = np.concatenate([r1,r2,r3],axis=1)
residual_cov_r = np.sqrt(np.cov(r.T)/dt)
print("Residual covariance:")
print(residual_cov_r)

k1 = FEX_model_check(u)
u1 = u + 0.5 * Dt * k1
k2 = FEX_model_check(u1)
u2 = u + 0.5 * Dt * k2
k3 = FEX_model_check(u2)
u3 = u + Dt * k3
k4 = FEX_model_check(u3)
u_rk4_pred = u + Dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
r1 = (u1_next - u_rk4[:,0].reshape(-1,1))
r2 = (u2_next - u_rk4[:,1].reshape(-1,1))
r3 = (u3_next - u_rk4[:,2].reshape(-1,1))
r = np.concatenate([r1,r2,r3],axis=1)
residual_cov_r = np.sqrt(np.cov(r.T)/dt)
print("Residual covariance:")
print(residual_cov_r)
print("\nTheoretical SS:")
print(params['SS'])

MC_samples = 1000  # number of Monte Carlo samples
time_steps = 1000  # number of time steps

# Reshape the data
u1_next_reshaped = u1_next.reshape(MC_samples, time_steps)
u2_next_reshaped = u2_next.reshape(MC_samples, time_steps)
u3_next_reshaped = u3_next.reshape(MC_samples, time_steps)

u1_current_reshaped = u1_current.reshape(MC_samples, time_steps)
u2_current_reshaped = u2_current.reshape(MC_samples, time_steps)
u3_current_reshaped = u3_current.reshape(MC_samples, time_steps)

# Reshape FEX predictions
u_pred_reshaped = u_rk4_pred.reshape(MC_samples, time_steps, 3)

# Calculate residuals for each time step
residuals = np.zeros((MC_samples, 3, time_steps))
for t in range(time_steps):
    residuals[:, 0, t] = u1_next_reshaped[:, t] -  u_pred_reshaped[:, t, 0]
    residuals[:, 1, t] = u2_next_reshaped[:, t] -  u_pred_reshaped[:, t, 1]
    residuals[:, 2, t] = u3_next_reshaped[:, t] - u_pred_reshaped[:, t, 2]

# Compute covariance for each time step
residual_cov_time = np.zeros((time_steps, 3))
for t in range(time_steps):
    residual_cov_time[t,0] = np.std(residuals[:, 0, t].T) / np.sqrt((dt))
    residual_cov_time[t,1] = np.std(residuals[:, 1, t].T) / np.sqrt((dt))
    residual_cov_time[t,2] = np.std(residuals[:, 2, t].T) / np.sqrt((dt))
    if t% 100 == 0:
        print(residual_cov_time[t,0],residual_cov_time[t,1],residual_cov_time[t,2])

print("Residual covariance shape:", residual_cov_time.shape)
print("First time step covariance:")

print("\nTheoretical SS:")
print(params['SS'])




