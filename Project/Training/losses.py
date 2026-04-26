"""Loss helpers for baseline and explicit WC-residual models."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import BASELINE_VARIANT


def _dynamic_loss(
    dEdt_cd: torch.Tensor,
    dIdt_cd: torch.Tensor,
    dyn_window: int,
    dyn_stride: int,
    dyn_floor_E: float,
    dyn_floor_I: float,
) -> torch.Tensor:
    if dEdt_cd.shape[0] >= dyn_window:
        dE_win = dEdt_cd.unfold(0, dyn_window, dyn_stride)
        dI_win = dIdt_cd.unfold(0, dyn_window, dyn_stride)
        dE_win_mean = dE_win.abs().mean(dim=-1)
        dI_win_mean = dI_win.abs().mean(dim=-1)
        return (
            torch.mean(torch.relu(torch.tensor(dyn_floor_E, device=dEdt_cd.device) - dE_win_mean) ** 2)
            + torch.mean(torch.relu(torch.tensor(dyn_floor_I, device=dIdt_cd.device) - dI_win_mean) ** 2)
        )

    return (
        torch.mean(torch.relu(torch.tensor(dyn_floor_E, device=dEdt_cd.device) - dEdt_cd.abs()) ** 2)
        + torch.mean(torch.relu(torch.tensor(dyn_floor_I, device=dIdt_cd.device) - dIdt_cd.abs()) ** 2)
    )


def _first_difference(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.shape[0] < 2:
        return sequence.new_zeros((0, *sequence.shape[1:]))
    return sequence[1:] - sequence[:-1]


def _safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return target.new_tensor(0.0)
    return nn.functional.mse_loss(pred, target)


def _compute_cross_regularization(model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "get_cross_parameters"):
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    cross_parameters = tuple(model.get_cross_parameters())
    if len(cross_parameters) == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    cross_l2 = torch.stack([parameter.pow(2).mean() for parameter in cross_parameters]).sum()
    cross_l1 = torch.stack([parameter.abs().mean() for parameter in cross_parameters]).sum()
    return cross_l2, cross_l1


def compute_training_losses(
    model,
    outputs: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    config,
    dt: float,
    epoch: int,
) -> dict[str, torch.Tensor]:
    if getattr(model, "variant", None) == BASELINE_VARIANT:
        return _compute_baseline_losses(model, outputs, y_true, config, dt, epoch)
    return _compute_explicit_losses(model, outputs, y_true, config)


def compute_validation_objective(
    model,
    outputs: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    config,
) -> torch.Tensor:
    if getattr(model, "variant", None) == BASELINE_VARIANT:
        return nn.functional.mse_loss(outputs["pred"], y_true)
    return _compute_explicit_losses(model, outputs, y_true, config)["total"]


def _compute_baseline_losses(
    model,
    outputs: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    config,
    dt: float,
    epoch: int,
) -> dict[str, torch.Tensor]:
    loss_data = nn.functional.mse_loss(outputs["pred"], y_true)
    loss_ic = nn.functional.mse_loss(outputs["pred"][0], y_true[0])

    E_raw = outputs["E_raw"]
    I_raw = outputs["I_raw"]
    dEdt_cd = (E_raw[2:] - E_raw[:-2]) / (2 * dt)
    dIdt_cd = (I_raw[2:] - I_raw[:-2]) / (2 * dt)
    loss_dyn = _dynamic_loss(
        dEdt_cd,
        dIdt_cd,
        dyn_window=config.dyn_window,
        dyn_stride=config.dyn_stride,
        dyn_floor_E=config.dyn_floor_E,
        dyn_floor_I=config.dyn_floor_I,
    )

    lambda_phys = min(
        config.lambda_phys_max * epoch / max(config.ramp_epochs, 1),
        config.lambda_phys_max,
    )
    if lambda_phys > 0 and dEdt_cd.shape[0] > 0:
        E_mid = torch.sigmoid(E_raw[1:-1])
        I_mid = torch.sigmoid(I_raw[1:-1])
        latent_mid = torch.cat([E_mid, I_mid], dim=-1)
        wc_derivative = model.wc(latent_mid, outputs["u_mid"])
        n_regions = model.n_regions
        dEdt_wc = wc_derivative[:, :n_regions]
        dIdt_wc = wc_derivative[:, n_regions:]
        n_phys = min(config.n_phys_samples, dEdt_cd.shape[0])
        idx = torch.randint(0, dEdt_cd.shape[0], (n_phys,), device=y_true.device)
        loss_phys = (
            nn.functional.mse_loss(dEdt_cd[idx], dEdt_wc[idx])
            + nn.functional.mse_loss(dIdt_cd[idx], dIdt_wc[idx])
        )
    else:
        loss_phys = torch.tensor(0.0, device=y_true.device)

    zero = torch.tensor(0.0, device=y_true.device)
    loss_total = (
        loss_data
        + config.lambda_ic * loss_ic
        + lambda_phys * loss_phys
        + config.lambda_dyn * loss_dyn
    )
    return {
        "data": loss_data,
        "delta": zero,
        "ic": loss_ic,
        "physics": loss_phys,
        "dynamic": loss_dyn,
        "residual_reg": zero,
        "state_smoothness": zero,
        "cross_l2": zero,
        "cross_l1": zero,
        "total": loss_total,
    }


def _compute_explicit_losses(
    model,
    outputs: dict[str, torch.Tensor],
    y_true: torch.Tensor,
    config,
) -> dict[str, torch.Tensor]:
    predictions = outputs["pred"]
    loss_data = nn.functional.mse_loss(predictions, y_true)
    loss_ic = nn.functional.mse_loss(predictions[0], y_true[0])

    pred_delta = _first_difference(predictions)
    true_delta = _first_difference(y_true)
    loss_delta = _safe_mse(pred_delta, true_delta)

    residual = outputs["residual_derivative"]
    residual_reg = residual.pow(2).mean() if residual.numel() > 0 else y_true.new_tensor(0.0)

    latent_state = outputs["latent_state"]
    latent_delta = _first_difference(latent_state)
    state_smoothness = (
        latent_delta.pow(2).mean() if latent_delta.numel() > 0 else y_true.new_tensor(0.0)
    )

    cross_l2, cross_l1 = _compute_cross_regularization(model, y_true.device)

    loss_total = (
        loss_data
        + config.delta_loss_weight * loss_delta
        + config.lambda_ic * loss_ic
        + config.residual_reg_weight * residual_reg
        + config.state_smoothness_weight * state_smoothness
        + config.cross_l2_weight * cross_l2
        + config.cross_l1_weight * cross_l1
    )
    return {
        "data": loss_data,
        "delta": loss_delta,
        "ic": loss_ic,
        "physics": y_true.new_tensor(0.0),
        "dynamic": y_true.new_tensor(0.0),
        "residual_reg": residual_reg,
        "state_smoothness": state_smoothness,
        "cross_l2": cross_l2,
        "cross_l1": cross_l1,
        "total": loss_total,
    }
