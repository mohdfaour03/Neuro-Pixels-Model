"""Command-line entry point for the WC-UDE experiment."""

import argparse
import json
from pathlib import Path

import torch

from config import DataConfig, ModelConfig, TrainingConfig
from dataset import load_preprocess_data
from Evaluation import evaluate_split, plot_evaluation, print_wc_parameters
from Network import TrajectoryNet, WilsonCowanPhysics
from Training import train_model
from utils import set_seed


def _jsonify_metrics(results: dict) -> dict:
    return {
        "mse": float(results["mse"]),
        "r2": {
            region: {
                metric_name: float(metric_value)
                for metric_name, metric_value in region_scores.items()
            }
            for region, region_scores in results["r2"].items()
        },
    }


def save_metrics(
    save_dir: Path,
    train_results: dict,
    val_results: dict,
    test_results: dict,
    history,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_validation": {
            "epoch": int(history.best_epoch),
            "mse": float(history.best_val_loss),
        },
        "final_losses": {
            "train_total": float(history.total_loss[-1]) if history.total_loss else None,
            "train_data": float(history.data_loss[-1]) if history.data_loss else None,
            "validation": float(history.val_loss[-1]) if history.val_loss else None,
            "physics": float(history.physics_loss[-1]) if history.physics_loss else None,
        },
        "splits": {
            "train": _jsonify_metrics(train_results),
            "validation": _jsonify_metrics(val_results),
            "test": _jsonify_metrics(test_results),
        },
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the WC-UDE model.")
    parser.add_argument("--data-path", type=Path, default=DataConfig().data_path)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().n_epochs)
    parser.add_argument("--downsample", type=int, default=DataConfig().downsample)
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-path", type=Path, default=TrainingConfig().checkpoint_path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_config = TrainingConfig(n_epochs=args.epochs, checkpoint_path=args.checkpoint_path)
    data_config = DataConfig(data_path=args.data_path, downsample=args.downsample)
    model_config = ModelConfig()
    set_seed(training_config.seed)

    data = load_preprocess_data(
        data_path=data_config.data_path,
        device=device,
        downsample=data_config.downsample,
        train_fraction=data_config.train_fraction,
        val_fraction=data_config.val_fraction,
    )
    print(f"Train: {data.y_ds_train.shape[0]} timesteps")
    print(f"Val:   {data.y_ds_val.shape[0]} timesteps")
    print(f"Test:  {data.y_ds_test.shape[0]} timesteps")
    print(f"Regions: {data.regions}")
    print(
        "Train tensors: "
        f"t={data.t_train.shape}, Y={data.y_train.shape}, "
        f"U={data.u_train.shape}, beh={data.beh_train.shape}"
    )
    print(
        "Val tensors:   "
        f"t={data.t_val.shape}, Y={data.y_val.shape}, "
        f"U={data.u_val.shape}, beh={data.beh_val.shape}"
    )
    print(
        "Test tensors:  "
        f"t={data.t_test.shape}, Y={data.y_test.shape}, "
        f"U={data.u_test.shape}, beh={data.beh_test.shape}"
    )

    model = TrajectoryNet(**model_config.__dict__).to(device)
    wc = WilsonCowanPhysics(
        n_regions=model_config.n_regions,
        n_stim=model_config.n_stim,
    ).to(device)

    print(f"Total TrajectoryNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"WC learnable parameters: {sum(p.numel() for p in wc.parameters())}")

    history = train_model(model, wc, data, training_config, device)
    train_results = evaluate_split(model, data, split="train")
    val_results = evaluate_split(model, data, split="val")
    test_results = evaluate_split(model, data, split="test")
    print_wc_parameters(wc)

    print("\n=== E/I Dynamics Statistics ===")
    print(f"E_int std per region:  {train_results['E_pred'].std(axis=0)}")
    print(f"I_int std per region:  {train_results['I_pred'].std(axis=0)}")
    print(
        "E_int range per region: "
        f"{train_results['E_pred'].min(axis=0)} to {train_results['E_pred'].max(axis=0)}"
    )

    save_metrics(args.save_dir, train_results, val_results, test_results, history)

    if not args.no_plots:
        plot_evaluation(data, history, train_results, test_results, args.save_dir)
        print(f"Saved SVG figures to: {args.save_dir}")


if __name__ == "__main__":
    main()
