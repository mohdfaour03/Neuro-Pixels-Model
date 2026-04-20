import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config, save_path, device='cpu'):
    """
    Trains a model with early stopping.
    Handles optional regularization on E and I bounds.
    """
    model.to(device)
    
    # Check if model is the persistence baseline, which requires no training
    if type(model).__name__ == "PersistenceBaseline":
        return model, [], []
        
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    # Check for phase 2 regularization (only for Mechanistic/Hybrid)
    reg_strength = config['models'].get('mechanistic', {}).get('regularization_strength', 0.0)
    apply_reg = reg_strength > 0.0 and type(model).__name__ in ["MechanisticModel", "HybridModel"]
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    
    # Wrap epochs in tqdm
    epoch_pbar = tqdm(range(epochs), desc=f"Training {type(model).__name__}", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        epoch_train_loss = 0.0
        for x_seq_in, u_t, target, _ in train_loader:
            x_seq_in, u_t, target = x_seq_in.to(device), u_t.to(device), target.to(device)
            x0 = x_seq_in[:, 0, :]
            
            optimizer.zero_grad()
            out = model(x0=x0, u_seq=u_t, x_seq=x_seq_in)
            
            if isinstance(out, tuple):
                preds, E_seq, I_seq = out
            else:
                preds = out
                
            # Curriculum Learning: dynamically scale the BPTT horizon
            # Forces model to master short-horizon dynamics before punishing it for full-horizon drifting
            seq_len_total = target.size(1)
            active_steps = max(2, int((epoch / epochs) * seq_len_total * 1.5))
            active_steps = min(active_steps, seq_len_total)
            
            pred_slice = preds[:, :active_steps, :]
            target_slice = target[:, :active_steps, :]
            loss = criterion(pred_slice, target_slice)

            # Penalize flat or phase-lagged solutions by matching local trajectory changes too.
            if active_steps > 1:
                pred_delta = pred_slice[:, 1:, :] - pred_slice[:, :-1, :]
                target_delta = target_slice[:, 1:, :] - target_slice[:, :-1, :]
                loss = loss + 0.25 * criterion(pred_delta, target_delta)
            
            # Keep the residual small enough to stay hybrid, but not so harshly regularized
            # that it collapses back to the mechanistic attractor.
            if hasattr(model, 'last_residual_magnitude'):
                loss += 0.0001 * (model.last_residual_magnitude / active_steps)
                
            loss.backward()
            
            # Gradient Clipping is absolutely mandatory to stop exploding parameters in BPTT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item() * x_seq_in.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_seq_in, u_t, target, _ in val_loader:
                x_seq_in, u_t, target = x_seq_in.to(device), u_t.to(device), target.to(device)
                x0 = x_seq_in[:, 0, :]
                
                out = model(x0=x0, u_seq=u_t, x_seq=x_seq_in)
                if isinstance(out, tuple):
                    preds = out[0]
                else:
                    preds = out
                    
                loss = criterion(preds, target)
                if target.size(1) > 1:
                    pred_delta = preds[:, 1:, :] - preds[:, :-1, :]
                    target_delta = target[:, 1:, :] - target[:, :-1, :]
                    loss = loss + 0.25 * criterion(pred_delta, target_delta)
                epoch_val_loss += loss.item() * x_seq_in.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            epoch_pbar.set_postfix({'Train Loss': f'{epoch_train_loss:.4f}', 'Val Loss': f'{epoch_val_loss:.4f}', 'Best': f'{best_val_loss:.4f}'})
        else:
            early_stop_counter += 1
            epoch_pbar.set_postfix({'Train Loss': f'{epoch_train_loss:.4f}', 'Val Loss': f'{epoch_val_loss:.4f}', 'Patience': f'{early_stop_counter}/{patience}'})
            if early_stop_counter >= patience:
                epoch_pbar.write(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break
                
    # Load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model, train_losses, val_losses
