import os

import torch
import torch.optim as optim
from tqdm import tqdm


def weighted_mse(preds, target, region_weights=None):
    error = (preds - target) ** 2
    if region_weights is not None:
        weights = torch.as_tensor(region_weights, dtype=preds.dtype, device=preds.device)
        error = error * weights.view(1, 1, -1)
    return error.mean()


def train_model(model, train_loader, val_loader, config, save_path, device="cpu"):
    """
    Trains a model with early stopping and simple regularization terms suited to the multi-region setup.
    """
    model.to(device)

    if type(model).__name__ == "PersistenceBaseline":
        return model, [], []

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    prediction_mode = config["training"].get("prediction_mode", "rollout")
    teacher_forcing = prediction_mode == "one_step"
    region_weights = config["models"].get("loss_region_weights")

    residual_reg_strength = config["models"].get("hybrid", {}).get(
        "residual_regularization_strength", 0.0
    )
    coupling_reg_strength = config["models"].get("coupling_regularization_strength", 0.0)
    sparsity_reg_strength = config["models"].get("sparsity_regularization_strength", 0.0)
    smoothness_reg_strength = config["models"].get("smoothness_regularization_strength", 0.0)

    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(epochs), desc=f"Training {type(model).__name__}", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        epoch_train_loss = 0.0

        for x_seq_in, u_t, target, _ in train_loader:
            x_seq_in = x_seq_in.to(device)
            u_t = u_t.to(device)
            target = target.to(device)
            x0 = x_seq_in[:, 0, :]

            optimizer.zero_grad()
            out = model(x0=x0, u_seq=u_t, x_seq=x_seq_in, teacher_forcing=teacher_forcing)
            preds = out[0] if isinstance(out, tuple) else out

            seq_len_total = target.size(1)
            if prediction_mode == "rollout":
                active_steps = max(2, int((epoch / max(epochs, 1)) * seq_len_total * 1.5))
                active_steps = min(active_steps, seq_len_total)
            else:
                active_steps = seq_len_total

            pred_slice = preds[:, :active_steps, :]
            target_slice = target[:, :active_steps, :]
            loss = weighted_mse(pred_slice, target_slice, region_weights)

            if active_steps > 1:
                pred_delta = pred_slice[:, 1:, :] - pred_slice[:, :-1, :]
                target_delta = target_slice[:, 1:, :] - target_slice[:, :-1, :]
                loss = loss + 0.25 * weighted_mse(pred_delta, target_delta, region_weights)

            if hasattr(model, "last_residual_magnitude"):
                loss = loss + residual_reg_strength * model.last_residual_magnitude
            if hasattr(model, "coupling_l2_penalty"):
                loss = loss + coupling_reg_strength * model.coupling_l2_penalty()
            if hasattr(model, "coupling_l1_penalty"):
                loss = loss + sparsity_reg_strength * model.coupling_l1_penalty()
            if hasattr(model, "last_latent_smoothness"):
                loss = loss + smoothness_reg_strength * model.last_latent_smoothness

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * x_seq_in.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_seq_in, u_t, target, _ in val_loader:
                x_seq_in = x_seq_in.to(device)
                u_t = u_t.to(device)
                target = target.to(device)
                x0 = x_seq_in[:, 0, :]

                out = model(x0=x0, u_seq=u_t, x_seq=x_seq_in, teacher_forcing=teacher_forcing)
                preds = out[0] if isinstance(out, tuple) else out

                loss = weighted_mse(preds, target, region_weights)
                if target.size(1) > 1:
                    pred_delta = preds[:, 1:, :] - preds[:, :-1, :]
                    target_delta = target[:, 1:, :] - target[:, :-1, :]
                    loss = loss + 0.25 * weighted_mse(pred_delta, target_delta, region_weights)
                epoch_val_loss += loss.item() * x_seq_in.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{epoch_train_loss:.4f}",
                    "Val Loss": f"{epoch_val_loss:.4f}",
                    "Best": f"{best_val_loss:.4f}",
                }
            )
        else:
            early_stop_counter += 1
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{epoch_train_loss:.4f}",
                    "Val Loss": f"{epoch_val_loss:.4f}",
                    "Patience": f"{early_stop_counter}/{patience}",
                }
            )
            if early_stop_counter >= patience:
                epoch_pbar.write(
                    f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}"
                )
                break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model, train_losses, val_losses
