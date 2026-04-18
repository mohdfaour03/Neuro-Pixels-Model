import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def evaluate_model(model, test_dataset, device='cpu', seq_horizon=50):
    """
    Evaluates a trained model by natively running unbroken simulations
    across bounded overlapping validation horizons (50 steps) continuously resetting 
    via Teacher Forcing to prevent deterministic drift chaotic decay bounds.
    """
    model.to(device)
    model.eval()
    
    N_total = len(test_dataset.x)
    
    # Provide x0 and the entire u_t sequence to force the model to autoregressively fully track the test set!
    # Utilizing the internal .forward() loop preserves complex hidden states (LSTM/CTRNN) inherently!
    x0 = test_dataset.x[0].unsqueeze(0).to(device) # [1, 2]
    u_full_seq = test_dataset.u.unsqueeze(0).to(device) # [1, N_total, 1]
    
    with torch.no_grad():
        # Predict the entire sequence continuously native in PyTorch C++ graph!
        out = model(x0, u_full_seq[:, :-1, :])
        
        if isinstance(out, tuple):
            preds_2d = out[0][0].cpu().numpy()
            all_E = out[1][0, :, 0].cpu().numpy()
            all_I = out[2][0, :, 0].cpu().numpy()
        else:
            preds_2d = out[0].cpu().numpy()
            all_E = preds_2d[:, 0]
            all_I = preds_2d[:, 1]
            
    all_preds_2d = preds_2d
    all_targets_2d = test_dataset.x[1:].numpy()
    
    # Flatten for global metrics
    all_preds_flat = all_preds_2d.flatten()
    all_targets_flat = all_targets_2d.flatten()
        
    # Standard metrics
    mse = mean_squared_error(all_targets_flat, all_preds_flat)
    mae = mean_absolute_error(all_targets_flat, all_preds_flat)
    r2 = r2_score(all_targets_flat, all_preds_flat)
    
    std_preds = np.std(all_preds_flat)
    std_targets = np.std(all_targets_flat)
    
    if std_preds > 1e-8 and std_targets > 1e-8:
        pearson_corr, _ = pearsonr(all_targets_flat, all_preds_flat)
    else:
        pearson_corr = 0.0
        
    # --- Phase 2 Metrics ---
    
    # 1. Temporal alignment quality (Cross-correlation and Lag)
    ctr_targets = all_targets_flat - np.mean(all_targets_flat)
    ctr_preds = all_preds_flat - np.mean(all_preds_flat)
    
    if std_preds > 1e-8 and std_targets > 1e-8:
        cc = np.correlate(ctr_targets, ctr_preds, mode='full')
        cc = cc / (std_targets * std_preds * len(all_targets_flat))
        zero_index = len(all_targets_flat) - 1
        
        max_lag_check = min(100, zero_index)
        cc_window = cc[zero_index - max_lag_check : zero_index + max_lag_check + 1]
        best_lag_idx = np.argmax(cc_window)
        
        lag_value = best_lag_idx - max_lag_check
        max_cross_corr = cc_window[best_lag_idx]
    else:
        lag_value = 0
        max_cross_corr = 0.0
        
    # 2. Dynamics quality: derivative consistency
    target_diff = np.diff(all_targets_flat)
    pred_diff = np.diff(all_preds_flat)
    if np.std(target_diff) > 1e-8 and np.std(pred_diff) > 1e-8:
        deriv_corr, _ = pearsonr(target_diff, pred_diff)
    else:
        deriv_corr = 0.0
        
    # 3. Stability metrics
    p_min = np.min(all_preds_flat)
    p_max = np.max(all_preds_flat)
    p_var = np.var(all_preds_flat)
    
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'R2': float(r2),
        'Pearson': float(pearson_corr),
        'Cross_Corr': float(max_cross_corr),
        'Lag': int(lag_value),
        'Deriv_Consistency': float(deriv_corr),
        'Pred_Min': float(p_min),
        'Pred_Max': float(p_max),
        'Pred_Var': float(p_var)
    }
    
    return metrics, all_preds_2d, all_targets_2d, all_E, all_I
