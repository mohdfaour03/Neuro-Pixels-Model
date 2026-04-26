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


def plot_evaluation(
    data: PreparedData,
    history: TrainHistory,
    train_results: Dict[str, Any],
    val_results: Dict[str, Any],
    test_results: Dict[str, Any],
    save_dir: Optional[Path] = None,
) -> None:
    _plot_loss_curves(history, None if save_dir is None else save_dir / "loss_curves.svg")

    _plot_trajectory(
        train_results,
        data.regions,
        "Train Trajectory Fit",
        None if save_dir is None else save_dir / "train_trajectory.svg",
    )
    _plot_trajectory(
        val_results,
        data.regions,
        "Validation Trajectory Fit",
        None if save_dir is None else save_dir / "val_trajectory.svg",
    )
    _plot_trajectory(
        test_results,
        data.regions,
        "Test Trajectory Fit",
        None if save_dir is None else save_dir / "test_trajectory.svg",
    )

    _plot_ei_dynamics(
        train_results,
        data.regions,
        "Train Latent E/I Trajectories",
        None if save_dir is None else save_dir / "train_ei_dynamics.svg",
    )
    _plot_ei_dynamics(
        test_results,
        data.regions,
        "Test Latent E/I Trajectories",
        None if save_dir is None else save_dir / "test_ei_dynamics.svg",
    )

    _plot_derivative_terms(
        train_results,
        data.regions,
        "Train WC vs Residual Derivatives",
        None if save_dir is None else save_dir / "train_derivatives.svg",
    )
    _plot_derivative_terms(
        test_results,
        data.regions,
        "Test WC vs Residual Derivatives",
        None if save_dir is None else save_dir / "test_derivatives.svg",
    )

    _plot_residual_ratio(
        train_results,
        data.regions,
        "Train Residual/WC Magnitude Ratio",
        None if save_dir is None else save_dir / "train_residual_ratio.svg",
    )
    _plot_residual_ratio(
        test_results,
        data.regions,
        "Test Residual/WC Magnitude Ratio",
        None if save_dir is None else save_dir / "test_residual_ratio.svg",
    )

    _plot_residual_stats(
        test_results,
        data.regions,
        None if save_dir is None else save_dir / "test_residual_stats.svg",
    )
    _plot_collapse_diagnostics(
        train_results,
        data.regions,
        "Train Collapse Diagnostics",
        None if save_dir is None else save_dir / "train_collapse_diagnostics.svg",
    )
    _plot_collapse_diagnostics(
        test_results,
        data.regions,
        "Test Collapse Diagnostics",
        None if save_dir is None else save_dir / "test_collapse_diagnostics.svg",
    )


def _plot_loss_curves(history: TrainHistory, path: Optional[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.data_loss, label="Train data")
    axes[0].plot(history.delta_loss, label="Train delta")
    axes[0].plot(history.val_loss, label="Validation objective")
    axes[0].plot(history.total_loss, label="Train total", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Primary Objective Terms")
    axes[0].legend()

    axes[1].plot(history.physics_loss, label="Physics", alpha=0.8)
    axes[1].plot(history.dynamic_loss, label="Dynamic", alpha=0.8)
    axes[1].plot(history.residual_reg_loss, label="Residual magnitude", alpha=0.8)
    axes[1].plot(history.state_smoothness_loss, label="State smoothness", alpha=0.8)
    axes[1].plot(history.cross_l2_loss, label="Cross L2", alpha=0.8)
    axes[1].plot(history.cross_l1_loss, label="Cross L1", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Regularization Terms")
    axes[1].legend()
    plt.tight_layout()
    _finish_plot(path)


def _plot_trajectory(
    results: Dict[str, Any],
    regions: list[str],
    title: str,
    path: Optional[Path],
) -> None:
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 2.5 * n_regions), sharex=True)
    axes = np.atleast_1d(axes)

    t_plot = results["time"][:500]
    y = results["y"]
    pred = results["pred"]
    stim_pred = results.get("stim_pred")
    for idx, region in enumerate(regions):
        axes[idx].plot(t_plot, y[:500, idx], color="black", alpha=0.6, label="Data")
        axes[idx].plot(t_plot, pred[:500, idx], color="tab:blue", label="Prediction")
        if stim_pred is not None:
            axes[idx].plot(
                t_plot,
                stim_pred[:500, idx],
                color="tab:red",
                linestyle="--",
                alpha=0.6,
                label="Stimulus-only baseline",
            )
        if results["per_region_temporal_stats"][region]["likely_collapsed"]:
            axes[idx].text(
                0.01,
                0.88,
                "collapse warning",
                transform=axes[idx].transAxes,
                color="red",
                fontsize=8,
                fontweight="bold",
            )
        axes[idx].set_ylabel(region)
        axes[idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)


def _plot_ei_dynamics(
    results: Dict[str, Any],
    regions: list[str],
    title: str,
    path: Optional[Path],
) -> None:
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 2.5 * n_regions), sharex=True)
    axes = np.atleast_1d(axes)

    t_plot = results["time"][:500]
    E_pred = results["E_pred"]
    I_pred = results["I_pred"]
    for idx, region in enumerate(regions):
        axes[idx].plot(t_plot, E_pred[:500, idx], color="tab:blue", label="E")
        axes[idx].plot(t_plot, I_pred[:500, idx], color="tab:red", label="I")
        axes[idx].set_ylabel(region)
        axes[idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)


def _plot_derivative_terms(
    results: Dict[str, Any],
    regions: list[str],
    title: str,
    path: Optional[Path],
) -> None:
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(16, 2.5 * n_regions), sharex=True)
    axes = np.atleast_2d(axes)

    t_plot = results["time"][:300]
    wc = results["wc_derivative"]
    residual = results["residual_derivative"]
    total = results["total_derivative"]
    for idx, region in enumerate(regions):
        axes[idx, 0].plot(t_plot, wc[:300, idx], label="WC dE/dt", color="tab:blue")
        axes[idx, 0].plot(
            t_plot,
            residual[:300, idx],
            label="Residual dE/dt",
            color="tab:orange",
        )
        axes[idx, 0].plot(
            t_plot,
            total[:300, idx],
            label="Total dE/dt",
            color="tab:green",
            alpha=0.8,
        )
        axes[idx, 0].set_ylabel(region)
        axes[idx, 0].legend(loc="upper right", fontsize=7)
        axes[idx, 1].plot(
            t_plot,
            wc[:300, idx + n_regions],
            label="WC dI/dt",
            color="tab:blue",
        )
        axes[idx, 1].plot(
            t_plot,
            residual[:300, idx + n_regions],
            label="Residual dI/dt",
            color="tab:orange",
        )
        axes[idx, 1].plot(
            t_plot,
            total[:300, idx + n_regions],
            label="Total dI/dt",
            color="tab:green",
            alpha=0.8,
        )
        axes[idx, 1].legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("Excitatory derivatives")
    axes[0, 1].set_title("Inhibitory derivatives")
    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)


def _plot_residual_ratio(
    results: Dict[str, Any],
    regions: list[str],
    title: str,
    path: Optional[Path],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t_plot = results["time"][:500]
    ratio_over_time = results["residual_ratio_over_time"][:500]
    ratio_by_region = results["residual_ratio_by_region"][:500]

    axes[0].plot(t_plot, ratio_over_time, color="tab:purple")
    axes[0].set_ylabel("|res| / |WC|")
    axes[0].set_title("Overall residual-to-WC magnitude ratio")

    for idx, region in enumerate(regions):
        axes[1].plot(t_plot, ratio_by_region[:, idx], label=region)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Per-region ratio")
    axes[1].set_title("Per-region residual-to-WC ratio")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)


def _plot_residual_stats(
    results: Dict[str, Any],
    regions: list[str],
    path: Optional[Path],
) -> None:
    stats = results["per_region_residual_stats"]
    mean_ratio = [stats[region]["mean_ratio"] for region in regions]
    mean_abs_residual = [
        0.5 * (stats[region]["mean_abs_residual_E"] + stats[region]["mean_abs_residual_I"])
        for region in regions
    ]
    mean_abs_wc = [
        0.5 * (stats[region]["mean_abs_wc_E"] + stats[region]["mean_abs_wc_I"])
        for region in regions
    ]

    x = np.arange(len(regions))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(x - width / 2, mean_abs_wc, width=width, label="|WC|")
    axes[0].bar(x + width / 2, mean_abs_residual, width=width, label="|Residual|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions)
    axes[0].set_title("Mean derivative magnitude by region")
    axes[0].legend()

    axes[1].bar(x, mean_ratio, color="tab:purple")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regions)
    axes[1].set_title("Mean residual/WC ratio by region")
    plt.tight_layout()
    _finish_plot(path)


def _plot_collapse_diagnostics(
    results: Dict[str, Any],
    regions: list[str],
    title: str,
    path: Optional[Path],
) -> None:
    temporal_stats = results["per_region_temporal_stats"]
    std_ratio = [temporal_stats[region]["std_ratio"] for region in regions]
    flatness_ratio = [temporal_stats[region]["flatness_ratio"] for region in regions]
    delta_mse = [temporal_stats[region]["delta_mse"] for region in regions]
    target_std = [temporal_stats[region]["target_std"] for region in regions]
    pred_std = [temporal_stats[region]["pred_std"] for region in regions]

    x = np.arange(len(regions))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].bar(x - width / 2, target_std, width=width, label="Target std")
    axes[0].bar(x + width / 2, pred_std, width=width, label="Pred std")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions)
    axes[0].set_title("Prediction vs target std")
    axes[0].legend()

    axes[1].bar(x - width / 2, std_ratio, width=width, label="Std ratio")
    axes[1].bar(x + width / 2, flatness_ratio, width=width, label="Flatness ratio")
    axes[1].axhline(
        results["collapse_summary"]["collapse_threshold"],
        color="red",
        linestyle="--",
        linewidth=1,
        label="Collapse threshold",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regions)
    axes[1].set_title("Collapse ratios")
    axes[1].legend()

    axes[2].bar(x, delta_mse, color="tab:green")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(regions)
    axes[2].set_title("Per-region delta MSE")

    fig.suptitle(title)
    plt.tight_layout()
    _finish_plot(path)
