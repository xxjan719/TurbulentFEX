import numpy as np
import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    model_path = Path(os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart'))
    save_dir = os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart','FN_model')
except:
    model_path = Path(os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart'))
    save_dir = os.path.join(os.getcwd(), 'src','Example','MC_triad','Results', 'equipart','FN_model')
os.makedirs(save_dir,exist_ok=True)

print(os.getcwd())
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.manual_seed(1234)
# np.random.seed(1234)

# data = np.load(os.path.join(model_path, 'equipart.npz')) 
# dt = 0.01

    
#     residuals,u_current = generate_rk4_residue(FEX_model_check, data, dt)
#     print(residuals.shape,u_current.shape)
    
#     scaler = np.array([20,20,20])
#     ODE_Solution,ZT_Solution = generate_second_step(u_current,residuals,scaler,dt,device)
#     print(ODE_Solution.shape)
    
#     mean_value, std_value = generate_mean_and_std(ODE_Solution)
#     print(mean_value.shape, std_value.shape)
#     print(mean_value[0:2,:],std_value[0:2,:])
    
#     # Train ensemble models for better accuracy
#     train_FN_ensemble(ODE_Solution, ZT_Solution, dim=3, device=device, save_dir=save_dir)
    
#     # # Test predictions with ensemble
    
    
#     print("\n=== Ensemble Prediction Results ===")
#     # Load and use ensemble predictions for each dimension
#     for t in range(1):
#         z_test = np.random.randn(1000,3)
#         z_test_tensor = torch.tensor(z_test, dtype=torch.float32).to(device)
#         for dim in range(1, 4):
#             # Load normalization parameters
#             norm_params = np.load(os.path.join(save_dir,f'norm_params_dim{dim}_t{t}.npy'), allow_pickle=True).item()
#             y_mean = norm_params['mean']
#             y_std = norm_params['std']
        
#             # Ensemble prediction
#             ensemble_predictions = []
#             n_models = 5  # Number of models in ensemble
            
#             for model_idx in range(n_models):
#                 FN_dim = FN_Net(1,1,100).to(device)
#                 FN_dim.load_state_dict(torch.load(os.path.join(save_dir,f'FN_dim{dim}_t0_model{model_idx}.pth'), weights_only=True))
                
#                 # Make prediction
#                 pred = (FN_dim(z_test_tensor[:,dim-1:dim].reshape(-1,1))).cpu().detach().numpy()
#                 ensemble_predictions.append(pred)
            
#             # Average ensemble predictions
#             pred = np.mean(ensemble_predictions, axis=0)
            
#             # Denormalize: pred * y_std + y_mean
#             pred = pred * y_std + y_mean
            
#             # Scale back by scaler
#             pred = pred / scaler[dim-1]
            
#             print(f"Dimension {dim}: {np.std(pred)/np.sqrt(dt):.6f}")
    
#     print("\nExpected values should be close to original")
#     print("Ensemble method should provide more accurate results!")




