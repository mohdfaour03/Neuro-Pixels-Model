"""Command-line entry point for the WC-residual ablation study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from config import (
    BASELINE_VARIANT,
    EXPLICIT_WC_RESIDUAL_ALIAS,
    MODEL_VARIANTS,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    normalize_variant_name,
)
from dataset import load_preprocess_data
from Evaluation import (
    evaluate_split,
    plot_evaluation,
    print_wc_parameters,
    save_metrics_bundle,
    save_split_outputs,
)
from Network import build_model
from Training import train_model
from utils import set_seed


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_run_config(
    save_dir: Path,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dt: float,
) -> None:
    payload = {
        "data": asdict(data_config),
        "model": asdict(model_config),
        "training": asdict(training_config),
        "resolved_integration_dt": dt,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(payload, indent=2, default=_json_default))


def _resolve_save_dir(root: Path, variant: str) -> Path:
    if root.name == variant:
        return root
    return root / variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline and explicit WC-residual variants.",
    )
    parser.add_argument("--data-path", type=Path, default=DataConfig().data_path)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().n_epochs)
    parser.add_argument("--downsample", type=int, default=DataConfig().downsample)
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--variant",
        type=str,
        default=ModelConfig().variant,
        choices=[*MODEL_VARIANTS, EXPLICIT_WC_RESIDUAL_ALIAS],
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--residual-hidden-size", type=int, default=ModelConfig().residual_hidden)
    parser.add_argument("--residual-alpha", type=float, default=ModelConfig().residual_scale)
    parser.add_argument(
        "--delta-loss-weight",
        type=float,
        default=TrainingConfig().delta_loss_weight,
    )
    parser.add_argument(
        "--residual-reg-weight",
        type=float,
        default=TrainingConfig().residual_reg_weight,
    )
    parser.add_argument(
        "--state-smoothness-weight",
        type=float,
        default=TrainingConfig().state_smoothness_weight,
    )
    parser.add_argument("--cross-l2-weight", type=float, default=TrainingConfig().cross_l2_weight)
    parser.add_argument("--cross-l1-weight", type=float, default=TrainingConfig().cross_l1_weight)
    parser.add_argument(
        "--integration-step",
        type=float,
        default=None,
        help="If omitted, use the median downsampled dataset timestep.",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        choices=["euler", "rk4"],
        default=ModelConfig().integrator,
    )
    parser.add_argument(
        "--use-behavior-in-residual",
        action=argparse.BooleanOptionalAction,
        default=ModelConfig().use_behavior_in_residual,
    )
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant = normalize_variant_name(args.variant)
    save_dir = _resolve_save_dir(args.save_dir, variant)
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else save_dir / "checkpoints" / "best_validation.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model variant: {variant}")
    print(f"Save directory: {save_dir}")

    data_config = DataConfig(
        data_path=args.data_path,
        downsample=args.downsample,
    )
    training_config = TrainingConfig(
        n_epochs=args.epochs,
        delta_loss_weight=args.delta_loss_weight,
        residual_reg_weight=args.residual_reg_weight,
        state_smoothness_weight=args.state_smoothness_weight,
        cross_l2_weight=args.cross_l2_weight,
        cross_l1_weight=args.cross_l1_weight,
        integration_step=args.integration_step,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    set_seed(training_config.seed)

    data = load_preprocess_data(
        data_path=data_config.data_path,
        device=device,
        downsample=data_config.downsample,
        train_fraction=data_config.train_fraction,
        val_fraction=data_config.val_fraction,
    )
    dt = training_config.integration_step or data.integration_dt

    model_config = ModelConfig(
        variant=variant,
        n_stim=data.n_stim,
        n_regions=data.n_regions,
        n_behavioral=data.n_behavioral,
        residual_hidden=args.residual_hidden_size,
        residual_scale=args.residual_alpha,
        use_behavior_in_residual=args.use_behavior_in_residual,
        integrator=args.integrator,
    )

    print(f"Train: {data.y_ds_train.shape[0]} timesteps")
    print(f"Val:   {data.y_ds_val.shape[0]} timesteps")
    print(f"Test:  {data.y_ds_test.shape[0]} timesteps")
    print(f"Regions: {data.regions}")
    print(f"Stimulus features: {data.feature_names}")
    print(f"Integration dt: {dt:.6f}")

    model = build_model(model_config).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    if hasattr(model, "wc") and model.wc is not None:
        print(f"WC parameters: {sum(parameter.numel() for parameter in model.wc.parameters()):,}")
    if hasattr(model, "residual"):
        print(
            f"Residual parameters: "
            f"{sum(parameter.numel() for parameter in model.residual.parameters()):,}"
        )
        print(f"Residual alpha: {getattr(model, 'get_residual_alpha', lambda: 0.0)():.6f}")

    _save_run_config(save_dir, data_config, model_config, training_config, dt)

    history = train_model(model=model, data=data, config=training_config, device=device, dt=dt)
    train_results = evaluate_split(model, data, split="train", dt=dt)
    val_results = evaluate_split(model, data, split="val", dt=dt)
    test_results = evaluate_split(model, data, split="test", dt=dt)
    print_wc_parameters(model)

    for split_name, split_results in {
        "train": train_results,
        "val": val_results,
        "test": test_results,
    }.items():
        save_split_outputs(save_dir, split_name, split_results)

    save_metrics_bundle(
        save_dir=save_dir,
        history=history,
        split_results={
            "train": train_results,
            "validation": val_results,
            "test": test_results,
        },
    )

    if not args.no_plots:
        plot_evaluation(
            data=data,
            history=history,
            train_results=train_results,
            val_results=val_results,
            test_results=test_results,
            save_dir=save_dir,
        )
        print(f"Saved figures to: {save_dir}")


if __name__ == "__main__":
    main()
