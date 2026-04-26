"""Training loop for the baseline and explicit WC-residual models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from dataset import PreparedData
from Training.losses import compute_training_losses, compute_validation_objective


@dataclass
class TrainHistory:
    data_loss: list[float] = field(default_factory=list)
    delta_loss: list[float] = field(default_factory=list)
    physics_loss: list[float] = field(default_factory=list)
    dynamic_loss: list[float] = field(default_factory=list)
    residual_reg_loss: list[float] = field(default_factory=list)
    state_smoothness_loss: list[float] = field(default_factory=list)
    cross_l2_loss: list[float] = field(default_factory=list)
    cross_l1_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def _run_model(model, data: PreparedData, split: str, dt: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    t, u, beh, y = data.get_split_tensors(split)
    if getattr(model, "variant", None) == "baseline_latent_wc_regularized":
        outputs = model(t, u, beh)
        outputs["u_mid"] = u[1:-1]
        return outputs, y

    outputs = model(t=t, u=u, beh=beh, y0=y[0], dt=dt)
    return outputs, y


def train_model(
    model,
    data: PreparedData,
    config,
    device: torch.device,
    dt: float,
) -> TrainHistory:
    del device
    torch.manual_seed(config.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.scheduler_t_max, 1),
        eta_min=config.scheduler_eta_min,
    )

    history = TrainHistory()
    best_model_state = deepcopy(model.state_dict())

    progress = tqdm(
        range(config.n_epochs),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in progress:
        model.train()
        optimizer.zero_grad()
        train_outputs, y_train = _run_model(model, data, split="train", dt=dt)
        losses = compute_training_losses(
            model=model,
            outputs=train_outputs,
            y_true=y_train,
            config=config,
            dt=dt,
            epoch=epoch,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs, y_val = _run_model(model, data, split="val", dt=dt)
            val_loss = compute_validation_objective(
                model=model,
                outputs=val_outputs,
                y_true=y_val,
                config=config,
            )

        if val_loss.item() < history.best_val_loss and epoch >= config.checkpoint_warmup:
            history.best_val_loss = val_loss.item()
            history.best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "val_objective": history.best_val_loss,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "training_config": dict(config.__dict__),
                    "model_variant": getattr(model, "variant", type(model).__name__),
                    "dt": dt,
                },
                config.checkpoint_path,
            )

        history.data_loss.append(losses["data"].item())
        history.delta_loss.append(losses["delta"].item())
        history.physics_loss.append(losses["physics"].item())
        history.dynamic_loss.append(losses["dynamic"].item())
        history.residual_reg_loss.append(losses["residual_reg"].item())
        history.state_smoothness_loss.append(losses["state_smoothness"].item())
        history.cross_l2_loss.append(losses["cross_l2"].item())
        history.cross_l1_loss.append(losses["cross_l1"].item())
        history.total_loss.append(losses["total"].item())
        history.val_loss.append(val_loss.item())

        progress.set_postfix_str(
            f"loss={losses['total'].item():.4g} val={val_loss.item():.4g}",
            refresh=False,
        )

        should_log = epoch % config.log_every == 0 or epoch == config.n_epochs - 1
        if should_log:
            tqdm.write(
                f"Epoch {epoch:4d} | Total: {losses['total'].item():.6f} | "
                f"data: {losses['data'].item():.6f} | "
                f"delta: {losses['delta'].item():.6f} | "
                f"val_obj: {val_loss.item():.6f} | "
                f"phys: {losses['physics'].item():.6f} | "
                f"dyn: {losses['dynamic'].item():.6f} | "
                f"res_reg: {losses['residual_reg'].item():.6f} | "
                f"state_smooth: {losses['state_smoothness'].item():.6f} | "
                f"cross_l2: {losses['cross_l2'].item():.6f} | "
                f"cross_l1: {losses['cross_l1'].item():.6f}"
            )

    if history.best_epoch >= 0:
        model.load_state_dict(best_model_state)
        print(
            f"\nTraining complete. Best val epoch: {history.best_epoch} | "
            f"Best val objective: {history.best_val_loss:.6f} | "
            f"Checkpoint: {config.checkpoint_path}"
        )
    else:
        print("\nTraining complete. No validation checkpoint was saved during warmup.")

    return history
