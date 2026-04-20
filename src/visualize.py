import matplotlib.pyplot as plt
import numpy as np


def select_region_subset(region_names, max_regions):
    return region_names[: min(len(region_names), max_regions)]


def plot_loss_curves(train_losses, val_losses, model_name, save_path=None):
    if not train_losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_predictions(
    time_vector,
    true_x,
    pred_x,
    u_t,
    region_names,
    model_name,
    save_path=None,
    window_size=500,
    max_regions=4,
):
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        true_x = true_x[:window_size]
        pred_x = pred_x[:window_size]
        u_t = u_t[:window_size]

    selected_regions = select_region_subset(region_names, max_regions)
    fig, axes = plt.subplots(len(selected_regions) + 1, 1, figsize=(12, 2.5 * (len(selected_regions) + 1)), sharex=True)

    axes[0].fill_between(time_vector, 0, u_t, color="gray", alpha=0.45, step="pre")
    axes[0].set_ylabel("u(t)")
    axes[0].set_title(f"{model_name} Multi-Region Prediction")

    for idx, region_name in enumerate(selected_regions, start=1):
        region_idx = region_names.index(region_name)
        axes[idx].plot(time_vector, true_x[:, region_idx], color="black", alpha=0.8, label=f"True {region_name}")
        axes[idx].plot(
            time_vector,
            pred_x[:, region_idx],
            color="tab:blue",
            linestyle="--",
            alpha=0.9,
            label=f"Pred {region_name}",
        )
        axes[idx].set_ylabel(region_name)
        axes[idx].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_shared_timeline(
    time_vector,
    true_x,
    pred_x,
    u_t,
    region_names,
    model_name,
    save_path=None,
    window_size=500,
    max_regions=5,
):
    if len(time_vector) > window_size:
        time_vector = time_vector[:window_size]
        true_x = true_x[:window_size]
        pred_x = pred_x[:window_size]
        u_t = u_t[:window_size]

    selected_regions = select_region_subset(region_names, max_regions)
    selected_idx = [region_names.index(region_name) for region_name in selected_regions]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    ax_top.fill_between(time_vector, 0, u_t, color="gray", alpha=0.45, step="pre")
    ax_top.set_ylabel("u(t)")
    ax_top.set_title(f"{model_name} Shared Timeline")

    offset_scale = max(np.std(true_x[:, selected_idx]), 1e-6) * 4.0
    for offset_idx, region_idx in enumerate(selected_idx):
        offset = offset_idx * offset_scale
        region_name = region_names[region_idx]
        ax_bottom.plot(time_vector, true_x[:, region_idx] + offset, color="black", alpha=0.8)
        ax_bottom.plot(time_vector, pred_x[:, region_idx] + offset, linestyle="--", alpha=0.8, label=region_name)
        ax_bottom.text(time_vector[0], offset, region_name, va="bottom")

    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Offset traces")
    ax_bottom.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_region_metric_bars(region_metrics, metric_name, model_name, save_path=None):
    region_names = list(region_metrics.keys())
    metric_values = [region_metrics[region_name][metric_name] for region_name in region_names]

    plt.figure(figsize=(10, 4))
    plt.bar(region_names, metric_values, color="tab:blue", alpha=0.8)
    plt.ylabel(metric_name)
    plt.title(f"{model_name} Per-Region {metric_name}")
    plt.xticks(rotation=30)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_coupling_heatmaps(couplings, region_names, model_name, save_prefix=None):
    if couplings is None:
        return

    matrix_names = ["W_EE_cross", "W_EI_cross", "W_IE_cross", "W_II_cross"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, matrix_name in zip(axes.flat, matrix_names):
        matrix = couplings[matrix_name]
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_title(matrix_name)
        ax.set_xticks(range(len(region_names)))
        ax.set_yticks(range(len(region_names)))
        ax.set_xticklabels(region_names, rotation=45, ha="right")
        ax.set_yticklabels(region_names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} Learned Couplings", y=1.02)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_heatmaps.png")
    plt.close()


def plot_latent_phase(E_t, I_t, region_names, model_name, save_path=None, window_size=1000, max_regions=4):
    if E_t is None or I_t is None:
        return

    if len(E_t) > window_size:
        E_t = E_t[:window_size]
        I_t = I_t[:window_size]

    selected_regions = select_region_subset(region_names, max_regions)
    fig, axes = plt.subplots(1, len(selected_regions), figsize=(4 * len(selected_regions), 4))
    if len(selected_regions) == 1:
        axes = [axes]

    for ax, region_name in zip(axes, selected_regions):
        region_idx = region_names.index(region_name)
        ax.plot(E_t[:, region_idx], I_t[:, region_idx], color="tab:blue", alpha=0.7)
        ax.scatter(E_t[0, region_idx], I_t[0, region_idx], color="green", s=20)
        ax.scatter(E_t[-1, region_idx], I_t[-1, region_idx], color="red", s=20)
        ax.set_title(region_name)
        ax.set_xlabel("E")
        ax.set_ylabel("I")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Latent Phase Plots")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_rollout_comparison(
    time_vector,
    true_x,
    pred_x,
    region_names,
    model_name,
    save_path=None,
    short_window=200,
    long_window=1000,
):
    region_name = region_names[0]
    region_idx = 0

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    short_stop = min(len(time_vector), short_window)
    axes[0].plot(time_vector[:short_stop], true_x[:short_stop, region_idx], color="black", label=f"True {region_name}")
    axes[0].plot(
        time_vector[:short_stop],
        pred_x[:short_stop, region_idx],
        color="tab:orange",
        linestyle="--",
        label=f"Pred {region_name}",
    )
    axes[0].set_title("Short Horizon")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    long_stop = min(len(time_vector), long_window)
    axes[1].plot(time_vector[:long_stop], true_x[:long_stop, region_idx], color="black", label=f"True {region_name}")
    axes[1].plot(
        time_vector[:long_stop],
        pred_x[:long_stop, region_idx],
        color="tab:orange",
        linestyle="--",
        label=f"Pred {region_name}",
    )
    axes[1].set_title("Longer Horizon")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Rollout Comparison")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
