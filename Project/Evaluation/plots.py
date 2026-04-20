"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from dataset import PreparedData
from Training import TrainHistory


def _finish_plot(path: Optional[Path]) -> None:
    if path is None:
        plt.show()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def plot_data_overview(data: PreparedData, save_dir: Optional[Path] = None) -> None:
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data.t_ds_train, data.y_ds_train[:, 0])
    plt.title("Downsampled Neural Activity")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate")
    plt.legend(data.regions)
    plt.subplot(2, 1, 2)
    plt.plot(data.t_ds_train, data.running_ds_train)
    plt.title("Downsampled Behavioral Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Running Speed")
    plt.legend(["Running Speed"], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    _finish_plot(None if save_dir is None else save_dir / "data_overview.svg")


def plot_evaluation(
    data: PreparedData,
    history: TrainHistory,
    train_results: Dict[str, Any],
    test_results: Dict[str, Any],
    save_dir: Optional[Path] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].semilogy(history.data_loss, label="Data loss")
    axes[0].semilogy(history.val_loss, label="Validation loss")
    axes[0].semilogy(history.total_loss, label="Total loss", alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training Loss")
    axes[1].semilogy(history.physics_loss, label="Physics loss", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].set_title("Physics Loss")
    plt.tight_layout()
    _finish_plot(None if save_dir is None else save_dir / "loss_curves.svg")

    _plot_trajectory(
        data.t_ds_train,
        data.regions,
        train_results,
        "Train Set Trajectory Fit",
        None if save_dir is None else save_dir / "train_trajectory.svg",
    )
    _plot_trajectory(
        data.t_ds_test,
        data.regions,
        test_results,
        "Test Set Trajectory Fit",
        None if save_dir is None else save_dir / "test_trajectory.svg",
    )

    _plot_ei_dynamics(
        data.t_ds_train,
        data.regions,
        train_results,
        None if save_dir is None else save_dir / "ei_dynamics.svg",
    )

    beh_np = data.beh_train.cpu().numpy()
    E_pred = train_results["E_pred"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for i, region in enumerate(data.regions):
        axes[0, i].scatter(beh_np[:, 0], E_pred[:, i], s=1, alpha=0.1)
        axes[0, i].set_xlabel("Running (z)")
        axes[0, i].set_ylabel("E")
        axes[0, i].set_title(f"{region} vs Running")
        axes[1, i].scatter(beh_np[:, 1], E_pred[:, i], s=1, alpha=0.1)
        axes[1, i].set_xlabel("Pupil (z)")
        axes[1, i].set_ylabel("E")
        axes[1, i].set_title(f"{region} vs Pupil")
    plt.tight_layout()
    _finish_plot(None if save_dir is None else save_dir / "behavior_vs_e.svg")


def _plot_trajectory(
    t_values: np.ndarray,
    regions: list[str],
    results: Dict[str, Any],
    title: str,
    path: Optional[Path],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t_plot = t_values[:500]
    y = results["y"]
    pred = results["pred"]
    stim_pred = results["stim_pred"]
    E_pred = results["E_pred"]
    for i, region in enumerate(regions):
        axes[i].plot(t_plot, y[:500, i], "k", alpha=0.5, label="Data")
        axes[i].plot(t_plot, pred[:500, i], "b", label="Combined")
        axes[i].plot(t_plot, stim_pred[:500, i], "r--", alpha=0.5, label="Stim only")
        axes[i].plot(t_plot, E_pred[:500, i], "g--", alpha=0.5, label="Internal E")
        axes[i].set_ylabel(region)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)


def _plot_ei_dynamics(
    t_values: np.ndarray,
    regions: list[str],
    results: Dict[str, Any],
    path: Optional[Path],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t_plot = t_values[:500]
    E_pred = results["E_pred"]
    I_pred = results["I_pred"]
    for i, region in enumerate(regions):
        axes[i].plot(t_plot, E_pred[:500, i], "b", label="E (internal)")
        axes[i].plot(t_plot, I_pred[:500, i], "r", label="I (internal)")
        axes[i].set_ylabel(region)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Learned E/I Dynamics")
    plt.tight_layout()
    _finish_plot(path)
