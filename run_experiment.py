import argparse
import csv
import json
import os
import warnings

import numpy as np
import torch

from src.data_access import get_session
from src.datasets import create_dataloaders
from src.evaluate import evaluate_model
from src.features import build_stimulus_signal
from src.models import (
    HybridModel,
    IndependentPerRegionBaseline,
    LinearBaseline,
    MLPBaseline,
    MechanisticModel,
    PersistenceBaseline,
)
from src.preprocessing import (
    build_multi_region_population_activity,
    collect_multi_region_units,
)
from src.train import train_model
from src.utils import load_config, set_seed
from src.visualize import (
    plot_coupling_heatmaps,
    plot_latent_phase,
    plot_loss_curves,
    plot_predictions,
    plot_region_metric_bars,
    plot_rollout_comparison,
    plot_shared_timeline,
)

warnings.filterwarnings("ignore")


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def save_per_region_csv(path, results):
    rows = []
    for model_name, metrics in results.items():
        for region_name, region_metrics in metrics.get("Per_Region", {}).items():
            row = {"Model": model_name, "Region": region_name}
            row.update(region_metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run Neural Dynamics Experiment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lightweight", "heavyweight"],
        default="heavyweight",
        help="Choose lightweight for quick debugging or heavyweight for full training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to run. If not specified, runs all configured models.",
    )
    args = parser.parse_args()

    config = load_config("config/default.yaml")

    if args.mode == "lightweight":
        print("Running in LIGHTWEIGHT mode. Using a smaller training budget for Phase 3 debugging.")
        config["training"]["epochs"] = 2
        config["training"]["sequence_length"] = 12
        config["training"]["batch_size"] = 4096
        config["training"]["early_stopping_patience"] = 2

    set_seed(config["project"]["seed"])

    out_dir = config["project"]["output_dir"]
    for subdir in ["figures", "metrics", "models", "logs"]:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    save_json(os.path.join(out_dir, "logs", "used_config.json"), config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    session = get_session(config["data"]["session_id"], config["project"]["data_dir"])
    stim_table = session.stimulus_presentations
    t_start = float(stim_table["start_time"].min())
    t_stop = float(stim_table["stop_time"].max())
    print(f"Global time window: {t_start:.2f} to {t_stop:.2f} seconds")

    region_unit_map, unit_metadata = collect_multi_region_units(
        session,
        candidate_regions=config["data"]["selected_regions"],
        min_units_per_region=config["data"]["min_units_per_region"],
    )
    retained_regions = list(region_unit_map.keys())
    print(f"Retained regions: {retained_regions}")

    time_vector, x_t_raw, signal_metadata = build_multi_region_population_activity(
        session,
        region_unit_map,
        t_start,
        t_stop,
        bin_size=config["data"]["bin_size"],
        smoothing_sigma=config["data"]["smoothing_sigma"],
        normalize_by_unit_count=config["data"]["normalize_by_unit_count"],
    )

    _, u_t = build_stimulus_signal(
        session,
        t_start,
        t_stop,
        bin_size=config["data"]["bin_size"],
        stimulus_class=config["features"]["stimulus_class"],
    )

    x_mean = x_t_raw.mean(axis=0)
    x_std = x_t_raw.std(axis=0) + 1e-8
    x_t = (x_t_raw - x_mean) / x_std

    preprocessing_summary = {
        "session_id": int(config["data"]["session_id"]),
        "selected_regions_requested": config["data"]["selected_regions"],
        "retained_regions": retained_regions,
        "unit_metadata": unit_metadata,
        "signal_metadata": signal_metadata,
        "bin_size": float(config["data"]["bin_size"]),
        "smoothing_sigma": float(config["data"]["smoothing_sigma"]),
        "normalize_by_unit_count": bool(config["data"]["normalize_by_unit_count"]),
        "x_shape": [int(dim) for dim in x_t.shape],
    }
    save_json(os.path.join(out_dir, "logs", "preprocessing_metadata.json"), preprocessing_summary)

    print(f"Built aligned multi-region signals: x(t) shape={x_t.shape}, u(t) shape={u_t.shape}")

    seq_len = config["training"]["sequence_length"]
    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(
        x_t,
        u_t,
        time_vector,
        train_split=config["training"]["train_split"],
        val_split=config["training"]["val_split"],
        batch_size=config["training"]["batch_size"],
        sequence_length=seq_len,
    )

    obs_dim = x_t.shape[1]
    input_dim = obs_dim + 1

    mechanistic_cfg = config["models"]["mechanistic"]
    models_to_run = {
        "Persistence": PersistenceBaseline(),
        "Independent_Per_Region": IndependentPerRegionBaseline(
            obs_dim=obs_dim,
            hidden_dim=config["models"]["independent_region"]["hidden_dim"],
            dt=mechanistic_cfg["dt"],
        ),
        "Joint_Linear": LinearBaseline(input_dim=input_dim, output_dim=obs_dim, dt=mechanistic_cfg["dt"]),
        "Joint_MLP": MLPBaseline(
            hidden_dims=config["models"]["baseline_mlp"]["hidden_dims"],
            input_dim=input_dim,
            output_dim=obs_dim,
            dt=mechanistic_cfg["dt"],
        ),
    }

    if config["models"].get("use_mechanistic", True):
        models_to_run["MechanisticModel"] = MechanisticModel(
            obs_dim=obs_dim,
            dt=mechanistic_cfg["dt"],
            activation_function=mechanistic_cfg["activation_function"],
            observation_model=mechanistic_cfg["observation_model"],
            use_cross_region_coupling=config["models"]["use_cross_region_coupling"],
            cross_coupling_mode=config["models"]["cross_coupling_mode"],
            local_coupling_init_scale=mechanistic_cfg["local_coupling_init_scale"],
            cross_coupling_init_scale=mechanistic_cfg["cross_coupling_init_scale"],
        )

    if config["models"].get("use_residual", True):
        models_to_run["HybridModel"] = HybridModel(
            obs_dim=obs_dim,
            dt=mechanistic_cfg["dt"],
            activation_function=mechanistic_cfg["activation_function"],
            observation_model=mechanistic_cfg["observation_model"],
            use_cross_region_coupling=config["models"]["use_cross_region_coupling"],
            cross_coupling_mode=config["models"]["cross_coupling_mode"],
            local_coupling_init_scale=mechanistic_cfg["local_coupling_init_scale"],
            cross_coupling_init_scale=mechanistic_cfg["cross_coupling_init_scale"],
            residual_hidden_dims=config["models"]["hybrid"]["residual_hidden_dims"],
            residual_scale_init=config["models"]["hybrid"]["residual_scale_init"],
        )

    if args.model:
        if args.model in models_to_run:
            models_to_run = {args.model: models_to_run[args.model]}
            print(f"Restricted execution to target model: {args.model}")
        else:
            print(f"Warning: Model '{args.model}' not found. Valid models are: {list(models_to_run.keys())}")
            return

    results = {}
    eval_limit = 1500 if args.mode == "lightweight" else None

    for model_name, model in models_to_run.items():
        save_path = os.path.join(out_dir, "models", f"{model_name}.pth")
        print(f"\n--- Training {model_name} ---")

        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, config, save_path, device
        )
        plot_loss_curves(
            train_losses,
            val_losses,
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_loss_curves.png"),
        )

        primary_eval_mode = config["training"].get("eval_mode", "one_step")
        metrics, all_preds, all_targets, all_E, all_I, couplings = evaluate_model(
            trained_model,
            test_dataset,
            device=device,
            prediction_mode=primary_eval_mode,
            max_eval_steps=eval_limit,
            region_names=retained_regions,
        )
        results[model_name] = metrics

        test_time = test_dataset.time[1 : 1 + len(all_preds)]
        test_u_t = test_dataset.u.numpy().flatten()[: len(test_time)]

        rollout_targets = all_targets
        rollout_preds = all_preds
        rollout_time = test_time
        if primary_eval_mode != "rollout":
            rollout_metrics, rollout_preds, rollout_targets, _, _, _ = evaluate_model(
                trained_model,
                test_dataset,
                device=device,
                prediction_mode="rollout",
                max_eval_steps=eval_limit,
                region_names=retained_regions,
            )
            results[model_name]["Rollout_Eval"] = rollout_metrics
            rollout_time = test_dataset.time[1 : 1 + len(rollout_preds)]

        plot_predictions(
            test_time,
            all_targets,
            all_preds,
            test_u_t,
            retained_regions,
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_predictions.png"),
            window_size=500,
        )
        plot_shared_timeline(
            test_time,
            all_targets,
            all_preds,
            test_u_t,
            retained_regions,
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_shared_timeline.png"),
            window_size=500,
        )
        plot_region_metric_bars(
            metrics["Per_Region"],
            "MSE",
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_per_region_mse.png"),
        )
        plot_region_metric_bars(
            metrics["Per_Region"],
            "Pearson",
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_per_region_pearson.png"),
        )
        plot_rollout_comparison(
            rollout_time,
            rollout_targets,
            rollout_preds,
            retained_regions,
            model_name,
            save_path=os.path.join(out_dir, "figures", f"{model_name}_rollout_compare.png"),
        )

        if all_E is not None and all_I is not None:
            plot_latent_phase(
                all_E,
                all_I,
                retained_regions,
                model_name,
                save_path=os.path.join(out_dir, "figures", f"{model_name}_latent_phase.png"),
            )

        if couplings is not None:
            np.savez(os.path.join(out_dir, "metrics", f"{model_name}_couplings.npz"), **couplings)
            plot_coupling_heatmaps(
                couplings,
                retained_regions,
                model_name,
                save_prefix=os.path.join(out_dir, "figures", f"{model_name}_couplings"),
            )

    print("\n--- Final Results ---")
    print(f"Regions used: {retained_regions}")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(
            f"  MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}"
        )
        print(
            f"  Cross_Corr={metrics['Cross_Corr']:.4f}, Lag={metrics['Lag']}, "
            f"dx/dt Corr={metrics['Deriv_Consistency']:.4f}"
        )

    save_json(os.path.join(out_dir, "metrics", "results.json"), results)
    save_per_region_csv(os.path.join(out_dir, "metrics", "per_region_metrics.csv"), results)


if __name__ == "__main__":
    main()
