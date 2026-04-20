"""Training loop for the WC-UDE model."""

from __future__ import annotations
import sys
import os
sys.path.append(os.path.expanduser('~/DDM_Project/Proposed_method/Project'))

from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import TrainingConfig
from dataset import PreparedData
from Network import TrajectoryNet, WilsonCowanPhysics


@dataclass
class TrainHistory:
    data_loss: list[float] = field(default_factory=list)
    physics_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def _dynamic_loss(
    dEdt_cd: torch.Tensor,
    dIdt_cd: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    if dEdt_cd.shape[0] >= config.dyn_window:
        dE_win = dEdt_cd.unfold(0, config.dyn_window, config.dyn_stride)
        dI_win = dIdt_cd.unfold(0, config.dyn_window, config.dyn_stride)
        dE_win_mean = dE_win.abs().mean(dim=-1)
        dI_win_mean = dI_win.abs().mean(dim=-1)
        return (
            torch.mean(torch.relu(config.dyn_floor_E - dE_win_mean) ** 2)
            + torch.mean(torch.relu(config.dyn_floor_I - dI_win_mean) ** 2)
        )

    return (
        torch.mean(torch.relu(config.dyn_floor_E - dEdt_cd.abs()) ** 2)
        + torch.mean(torch.relu(config.dyn_floor_I - dIdt_cd.abs()) ** 2)
    )


def train_model(
    model: TrajectoryNet,
    wc: WilsonCowanPhysics,
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
) -> TrainHistory:
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(wc.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min,
    )
    history = TrainHistory()
    best_model_state = deepcopy(model.state_dict())
    best_wc_state = deepcopy(wc.state_dict())

    progress = tqdm(
        range(config.n_epochs),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        model.train()
        wc.train()
        optimizer.zero_grad()
        combined, _, E_int, I_int = model(data.t_train, data.u_train, data.beh_train)

        loss_data = nn.functional.mse_loss(combined, data.y_train)
        loss_ic = nn.functional.mse_loss(combined[0], data.y_train[0])

        dEdt_cd = (E_int[2:] - E_int[:-2]) / (2 * config.dt)
        dIdt_cd = (I_int[2:] - I_int[:-2]) / (2 * config.dt)
        loss_dyn = _dynamic_loss(dEdt_cd, dIdt_cd, config)

        lambda_phys = min(
            config.lambda_phys_max * epoch / config.ramp_epochs,
            config.lambda_phys_max,
        )
        if lambda_phys > 0:
            E_mid = torch.sigmoid(E_int[1:-1])
            I_mid = torch.sigmoid(I_int[1:-1])
            U_mid = data.u_train[1:-1]
            dEdt_wc, dIdt_wc = wc(E_mid, I_mid, U_mid)
            n_phys = min(config.n_phys_samples, dEdt_cd.shape[0])
            idx = torch.randint(0, dEdt_cd.shape[0], (n_phys,), device=device)
            loss_phys = (
                nn.functional.mse_loss(dEdt_cd[idx], dEdt_wc[idx])
                + nn.functional.mse_loss(dIdt_cd[idx], dIdt_wc[idx])
            )
        else:
            loss_phys = torch.tensor(0.0, device=device)

        loss = (
            loss_data
            + config.lambda_ic * loss_ic
            + lambda_phys * loss_phys
            + config.lambda_dyn * loss_dyn
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(wc.parameters()),
            max_norm=config.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        model.eval()
        wc.eval()
        with torch.no_grad():
            combined_val, _, _, _ = model(data.t_val, data.u_val, data.beh_val)
            val_loss = nn.functional.mse_loss(combined_val, data.y_val)

        if val_loss.item() < history.best_val_loss and epoch >= config.checkpoint_warmup:
            history.best_val_loss = val_loss.item()
            history.best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            best_wc_state = deepcopy(wc.state_dict())
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": history.best_val_loss,
                    "model_state_dict": best_model_state,
                    "wc_state_dict": best_wc_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "training_config": config,
                },
                config.checkpoint_path,
            )

        history.data_loss.append(loss_data.item())
        history.physics_loss.append(loss_phys.item())
        history.total_loss.append(loss.item())
        history.val_loss.append(val_loss.item())

        progress.set_postfix_str(
            f"loss={loss.item():.4g} val={val_loss.item():.4g}",
            refresh=False,
        )

        should_log = epoch % config.log_every == 0 or epoch == config.n_epochs - 1
        if should_log:
            tqdm.write(
                f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | "
                f"data: {loss_data.item():.6f} | val: {val_loss.item():.6f} | "
                f"phys: {loss_phys.item():.6f} | "
                f"dyn: {loss_dyn.item():.6f} | "
                # f"|dE/dt|: {dEdt_cd.abs().mean().item():.6f} | "
                # f"|dI/dt|: {dIdt_cd.abs().mean().item():.6f} | "
                # f"E_std: {E_int.std(dim=0).mean().item():.4f} | "
                # f"I_std: {I_int.std(dim=0).mean().item():.4f} | "
                # f"lambda_phys: {lambda_phys:.3f} | lr: {scheduler.get_last_lr()[0]:.6f}"
            )

    if history.best_epoch >= 0:
        model.load_state_dict(best_model_state)
        wc.load_state_dict(best_wc_state)
        print(
            f"\nTraining complete. Best val epoch: {history.best_epoch} | "
            f"Best val MSE: {history.best_val_loss:.6f} | "
            f"Checkpoint: {config.checkpoint_path}"
        )
    else:
        print("\nTraining complete. No validation checkpoint was saved during warmup.")
    return history
