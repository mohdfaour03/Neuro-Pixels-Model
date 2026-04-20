"""Model evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from dataset import PreparedData
from Evaluation.metrics import r_squared
from Network import TrajectoryNet, WilsonCowanPhysics


def evaluate_split(
    model: TrajectoryNet,
    data: PreparedData,
    split: str,
) -> Dict[str, Any]:
    if split == "train":
        t, u, beh, y = data.t_train, data.u_train, data.beh_train, data.y_train
    elif split == "val":
        t, u, beh, y = data.t_val, data.u_val, data.beh_val, data.y_val
    elif split == "test":
        t, u, beh, y = data.t_test, data.u_test, data.beh_test, data.y_test
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

    model.eval()
    with torch.no_grad():
        combined, stim_out, E_int, I_int = model(t, u, beh)
        pred = combined.cpu().numpy()
        stim_pred = torch.sigmoid(stim_out).cpu().numpy()
        E_pred = torch.sigmoid(E_int).cpu().numpy()
        I_pred = torch.sigmoid(I_int).cpu().numpy()
        y_np = y.cpu().numpy()

    print(f"=== {split.title()} R2 Scores ===")
    print(f"{'Region':<10} {'Combined':>10} {'Stim Only':>10} {'Internal E':>10}")
    print("-" * 45)
    r2_scores = {}
    for i, region in enumerate(data.regions):
        combined_r2 = r_squared(y_np[:, i], pred[:, i])
        stim_r2 = r_squared(y_np[:, i], stim_pred[:, i])
        internal_e_r2 = r_squared(y_np[:, i], E_pred[:, i])
        r2_scores[region] = {
            "combined": combined_r2,
            "stim_only": stim_r2,
            "internal_E": internal_e_r2,
        }
        print(
            f"{region:<10} "
            f"{combined_r2:>10.4f} "
            f"{stim_r2:>10.4f} "
            f"{internal_e_r2:>10.4f}"
        )

    mse = nn.functional.mse_loss(combined, y).item()
    print(f"\n{split.title()} MSE: {mse:.6f}")
    return {
        "pred": pred,
        "stim_pred": stim_pred,
        "E_pred": E_pred,
        "I_pred": I_pred,
        "y": y_np,
        "mse": mse,
        "r2": r2_scores,
    }


def print_wc_parameters(wc: WilsonCowanPhysics) -> None:
    print("\n=== Learned WC Parameters ===")
    with torch.no_grad():
        print(f"tau_E: {nn.functional.softplus(wc.tau_E).cpu().numpy()}")
        print(f"tau_I: {nn.functional.softplus(wc.tau_I).cpu().numpy()}")
        print(f"w_EE:  {wc.w_EE.cpu().numpy()}")
        print(f"w_EI:  {wc.w_EI.cpu().numpy()}")
        print(f"w_IE:  {wc.w_IE.cpu().numpy()}")
        print(f"w_II:  {wc.w_II.cpu().numpy()}")
        print(f"a:     {wc.a.cpu().numpy()}")
        print(f"b_E:   {wc.b_E.cpu().numpy()}")
        print(f"b_I:   {wc.b_I.cpu().numpy()}")
