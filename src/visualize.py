import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(time_vector, true_x, pred_x, u_t, model_name, save_path=None, window_size=500):
    """
    Plots true vs predicted population activity along with the stimulus signal.
    """
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        true_x = true_x[:window_size]
        pred_x = pred_x[:window_size]
        u_t = u_t[:window_size]
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    if len(true_x.shape) > 1 and true_x.shape[1] == 2:
        ax1.plot(time_vector, true_x[:, 0], label='True E(t)', color='black', alpha=0.7)
        ax1.plot(time_vector, pred_x[:, 0], label=f'Pred E(t) - {model_name}', color='blue', linestyle='--')
        
        ax1.plot(time_vector, true_x[:, 1], label='True I(t)', color='gray', alpha=0.7)
        ax1.plot(time_vector, pred_x[:, 1], label=f'Pred I(t) - {model_name}', color='red', linestyle='-.')
    else:
        ax1.plot(time_vector, true_x, label='True x(t)', color='black', alpha=0.7)
        ax1.plot(time_vector, pred_x, label=f'Predicted x(t) - {model_name}', color='red', linestyle='--')
        
    ax1.set_ylabel('Population Activity (Hz/Norm)')
    ax1.set_title(f'{model_name} Next-Step Prediction')
    ax1.legend(loc='upper right', ncol=2)
    
    ax2.fill_between(time_vector, 0, u_t, color='gray', alpha=0.5, step='pre')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Stimulus u(t)')
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_loss_curves(train_losses, val_losses, model_name, save_path=None):
    """Plots training and validation loss curves."""
    if not train_losses:
        return
        
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_phase(E_t, I_t, model_name, save_path=None, window_size=1000):
    """
    Plots E(t) vs I(t) to show the learned vector field trajectory.
    """
    if E_t is None or I_t is None:
        return
        
    if len(E_t) > window_size:
        E_t = E_t[:window_size]
        I_t = I_t[:window_size]
        
    plt.figure(figsize=(6, 6))
    plt.plot(E_t, I_t, alpha=0.5, color='blue', linewidth=0.8)
    
    # Highlight start and end
    plt.scatter(E_t[0], I_t[0], color='green', label='Start', zorder=5)
    plt.scatter(E_t[-1], I_t[-1], color='red', label='End', zorder=5)
    
    plt.xlabel('Excitatory Latent E(t)')
    plt.ylabel('Inhibitory Latent I(t)')
    plt.title(f'Phase Plot - {model_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
def plot_error(time_vector, true_x, pred_x, model_name, save_path=None, window_size=500):
    """Plots absolute prediction error over time."""
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        true_x = true_x[:window_size]
        pred_x = pred_x[:window_size]
        
    error = np.abs(true_x - pred_x)
    
    plt.figure(figsize=(10, 3))
    if len(error.shape) > 1 and error.shape[1] == 2:
        plt.plot(time_vector, error[:, 0], color='blue', alpha=0.8, label='E(t) Error')
        plt.plot(time_vector, error[:, 1], color='red', alpha=0.8, label='I(t) Error')
        plt.legend()
    else:
        plt.plot(time_vector, error, color='red', alpha=0.8)
        
    plt.xlabel('Time (s)')
    plt.ylabel('|Error|')
    plt.title(f'Absolute Error Over Time - {model_name}')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_residual_effect(time_vector, true_x, pred_wc, pred_hybrid, save_path=None, window_size=500):
    """Overlays WC pure vs WC+Residual predictions"""
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        true_x = true_x[:window_size]
        pred_wc = pred_wc[:window_size]
        pred_hybrid = pred_hybrid[:window_size]
        
    plt.figure(figsize=(12, 4))
    
    if len(true_x.shape) > 1 and true_x.shape[1] == 2:
        plt.plot(time_vector, true_x[:, 0], label='True E(t)', color='black', alpha=0.7, linewidth=2)
        plt.plot(time_vector, pred_wc[:, 0], label='WC Pure E', color='blue', linestyle='--', alpha=0.8)
        plt.plot(time_vector, pred_hybrid[:, 0], label='Residual E', color='red', linestyle='-.', alpha=0.8)
    else:
        plt.plot(time_vector, true_x, label='True x(t)', color='black', alpha=0.7, linewidth=2)
        plt.plot(time_vector, pred_wc, label='WC Pure', color='blue', linestyle='--', alpha=0.8)
        plt.plot(time_vector, pred_hybrid, label='WC + Residual', color='red', linestyle='-.', alpha=0.8)
        
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    plt.title('Residual Correction Effect (Excitatory Phase)')
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_latent_trajectories(time_vector, E_t, I_t, model_name, save_path=None, window_size=500):
    """Plots E(t) and I(t) overlayed over time."""
    if E_t is None or I_t is None:
        return
        
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        E_t = E_t[:window_size]
        I_t = I_t[:window_size]
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_vector, E_t, label='E(t) [Excitatory]', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(time_vector, I_t, label='I(t) [Inhibitory]', color='red', alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Latent Activation')
    plt.title(f'Latent E/I Trajectories Over Time - {model_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
