"""Model evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from dataset import PreparedData
from Evaluation.metrics import r_squared


COLLAPSE_RATIO_THRESHOLD = 0.35


def _run_model(model, data: PreparedData, split: str, dt: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    t, u, beh, y = data.get_split_tensors(split)
    if getattr(model, "variant", None) == "baseline_latent_wc_regularized":
        outputs = model(t, u, beh)
        return outputs, y
    outputs = model(t=t, u=u, beh=beh, y0=y[0], dt=dt)
    return outputs, y


def _tensor_to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def _compute_residual_diagnostics(
    wc_derivative: np.ndarray,
    residual_derivative: np.ndarray,
    regions: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    eps = 1e-8
    n_regions = len(regions)
    wc_norm = np.linalg.norm(wc_derivative, axis=-1)
    residual_norm = np.linalg.norm(residual_derivative, axis=-1)
    ratio_over_time = residual_norm / (wc_norm + eps)

    ratio_by_region = []
    per_region_stats: dict[str, dict[str, float]] = {}
    for idx, region in enumerate(regions):
        wc_pair = wc_derivative[:, [idx, idx + n_regions]]
        residual_pair = residual_derivative[:, [idx, idx + n_regions]]
        region_ratio = np.linalg.norm(residual_pair, axis=-1) / (
            np.linalg.norm(wc_pair, axis=-1) + eps
        )
        ratio_by_region.append(region_ratio)
        per_region_stats[region] = {
            "mean_abs_wc_E": float(np.mean(np.abs(wc_pair[:, 0]))),
            "mean_abs_wc_I": float(np.mean(np.abs(wc_pair[:, 1]))),
            "mean_abs_residual_E": float(np.mean(np.abs(residual_pair[:, 0]))),
            "mean_abs_residual_I": float(np.mean(np.abs(residual_pair[:, 1]))),
            "mean_ratio": float(np.mean(region_ratio)),
            "max_ratio": float(np.max(region_ratio)),
        }

    return ratio_over_time, np.stack(ratio_by_region, axis=1), per_region_stats


def _compute_temporal_diagnostics(
    pred: np.ndarray,
    y_true: np.ndarray,
    regions: list[str],
) -> tuple[float, np.ndarray, np.ndarray, dict[str, dict[str, float | bool]], dict[str, Any]]:
    eps = 1e-8
    pred_delta = np.diff(pred, axis=0)
    target_delta = np.diff(y_true, axis=0)

    if pred_delta.size == 0:
        zero = np.zeros((0, pred.shape[1]), dtype=pred.dtype)
        return 0.0, zero, zero, {
            region: {
                "delta_mse": 0.0,
                "pred_variance": float(np.var(pred[:, idx])),
                "target_variance": float(np.var(y_true[:, idx])),
                "pred_std": float(np.std(pred[:, idx])),
                "target_std": float(np.std(y_true[:, idx])),
                "std_ratio": 1.0,
                "pred_flatness": 0.0,
                "target_flatness": 0.0,
                "flatness_ratio": 1.0,
                "likely_collapsed": False,
            }
            for idx, region in enumerate(regions)
        }, {
            "delta_mse": 0.0,
            "mean_abs_delta_pred": 0.0,
            "mean_abs_delta_target": 0.0,
            "likely_collapsed_regions": [],
            "n_collapsed_regions": 0,
            "collapse_threshold": COLLAPSE_RATIO_THRESHOLD,
        }

    delta_mse = float(np.mean((pred_delta - target_delta) ** 2))
    per_region_temporal_stats: dict[str, dict[str, float | bool]] = {}
    collapsed_regions: list[str] = []
    for idx, region in enumerate(regions):
        region_delta_mse = float(np.mean((pred_delta[:, idx] - target_delta[:, idx]) ** 2))
        pred_variance = float(np.var(pred[:, idx]))
        target_variance = float(np.var(y_true[:, idx]))
        pred_std = float(np.std(pred[:, idx]))
        target_std = float(np.std(y_true[:, idx]))
        pred_flatness = float(np.mean(np.abs(pred_delta[:, idx])))
        target_flatness = float(np.mean(np.abs(target_delta[:, idx])))
        std_ratio = pred_std / (target_std + eps)
        flatness_ratio = pred_flatness / (target_flatness + eps)
        likely_collapsed = (
            std_ratio < COLLAPSE_RATIO_THRESHOLD
            or flatness_ratio < COLLAPSE_RATIO_THRESHOLD
        )
        if likely_collapsed:
            collapsed_regions.append(region)

        per_region_temporal_stats[region] = {
            "delta_mse": region_delta_mse,
            "pred_variance": pred_variance,
            "target_variance": target_variance,
            "pred_std": pred_std,
            "target_std": target_std,
            "std_ratio": float(std_ratio),
            "pred_flatness": pred_flatness,
            "target_flatness": target_flatness,
            "flatness_ratio": float(flatness_ratio),
            "likely_collapsed": bool(likely_collapsed),
        }

    collapse_summary = {
        "delta_mse": delta_mse,
        "mean_abs_delta_pred": float(np.mean(np.abs(pred_delta))),
        "mean_abs_delta_target": float(np.mean(np.abs(target_delta))),
        "likely_collapsed_regions": collapsed_regions,
        "n_collapsed_regions": len(collapsed_regions),
        "collapse_threshold": COLLAPSE_RATIO_THRESHOLD,
    }
    return delta_mse, pred_delta, target_delta, per_region_temporal_stats, collapse_summary


def evaluate_split(
    model,
    data: PreparedData,
    split: str,
    dt: float,
) -> Dict[str, Any]:
    time_values = data.get_split_times(split)
    model.eval()
    with torch.no_grad():
        outputs, y = _run_model(model, data, split, dt)

    pred = _tensor_to_numpy(outputs["pred"])
    y_np = y.detach().cpu().numpy()
    E_pred = _tensor_to_numpy(outputs["E_pred"])
    I_pred = _tensor_to_numpy(outputs["I_pred"])
    latent_state = _tensor_to_numpy(outputs["latent_state"])
    wc_derivative = _tensor_to_numpy(outputs["wc_derivative"])
    residual_derivative = _tensor_to_numpy(outputs["residual_derivative"])
    total_derivative = _tensor_to_numpy(outputs["total_derivative"])
    stim_pred = _tensor_to_numpy(outputs.get("stim_pred"))

    residual_ratio_over_time, residual_ratio_by_region, per_region_residual_stats = (
        _compute_residual_diagnostics(
            wc_derivative=wc_derivative,
            residual_derivative=residual_derivative,
            regions=data.regions,
        )
    )
    (
        delta_mse,
        pred_delta,
        target_delta,
        per_region_temporal_stats,
        collapse_summary,
    ) = _compute_temporal_diagnostics(
        pred=pred,
        y_true=y_np,
        regions=data.regions,
    )

    mse = nn.functional.mse_loss(outputs["pred"], y).item()
    mae = nn.functional.l1_loss(outputs["pred"], y).item()

    print(f"\n=== {split.title()} Metrics ===")
    print(
        f"{'Region':<10} {'R2':>10} {'DeltaMSE':>10} "
        f"{'StdRatio':>10} {'FlatRatio':>10} {'Collapse':>10} {'|res/WC|':>12}"
    )
    print("-" * 86)

    r2_scores = {}
    for idx, region in enumerate(data.regions):
        region_r2 = r_squared(y_np[:, idx], pred[:, idx])
        temporal_stats = per_region_temporal_stats[region]
        r2_scores[region] = {"prediction": region_r2}
        print(
            f"{region:<10} "
            f"{region_r2:>10.4f} "
            f"{temporal_stats['delta_mse']:>10.4f} "
            f"{temporal_stats['std_ratio']:>10.4f} "
            f"{temporal_stats['flatness_ratio']:>10.4f} "
            f"{str(temporal_stats['likely_collapsed']):>10} "
            f"{per_region_residual_stats[region]['mean_ratio']:>12.4f}"
        )

    print(f"{split.title()} MSE: {mse:.6f}")
    print(f"{split.title()} MAE: {mae:.6f}")
    print(f"{split.title()} Delta MSE: {delta_mse:.6f}")
    print(f"{split.title()} mean residual/WC ratio: {float(np.mean(residual_ratio_over_time)):.4f}")
    if collapse_summary["likely_collapsed_regions"]:
        print(
            f"{split.title()} collapse warning regions: "
            f"{collapse_summary['likely_collapsed_regions']}"
        )

    return {
        "variant": getattr(model, "variant", type(model).__name__),
        "residual_alpha": float(getattr(model, "get_residual_alpha", lambda: 0.0)()),
        "time": time_values,
        "pred": pred,
        "pred_delta": pred_delta,
        "target_delta": target_delta,
        "stim_pred": stim_pred,
        "E_pred": E_pred,
        "I_pred": I_pred,
        "latent_state": latent_state,
        "wc_derivative": wc_derivative,
        "residual_derivative": residual_derivative,
        "total_derivative": total_derivative,
        "residual_ratio_over_time": residual_ratio_over_time,
        "residual_ratio_by_region": residual_ratio_by_region,
        "per_region_residual_stats": per_region_residual_stats,
        "per_region_temporal_stats": per_region_temporal_stats,
        "collapse_summary": collapse_summary,
        "y": y_np,
        "mse": mse,
        "mae": mae,
        "delta_mse": delta_mse,
        "r2": r2_scores,
    }


def save_split_outputs(save_dir: Path, split: str, results: Dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    arrays_to_save = {
        key: value
        for key, value in results.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(save_dir / f"{split}_outputs.npz", **arrays_to_save)
    stats_path = save_dir / f"{split}_residual_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "residual_stats": results["per_region_residual_stats"],
                "temporal_stats": results["per_region_temporal_stats"],
                "collapse_summary": results["collapse_summary"],
                "residual_alpha": results["residual_alpha"],
            },
            indent=2,
        )
    )


def _jsonify_history(history) -> dict[str, float | int | None]:
    return {
        "best_epoch": int(history.best_epoch),
        "best_validation_objective": float(history.best_val_loss),
        "train_total_last": float(history.total_loss[-1]) if history.total_loss else None,
        "train_data_last": float(history.data_loss[-1]) if history.data_loss else None,
        "train_delta_last": float(history.delta_loss[-1]) if history.delta_loss else None,
        "train_physics_last": float(history.physics_loss[-1]) if history.physics_loss else None,
        "train_dynamic_last": float(history.dynamic_loss[-1]) if history.dynamic_loss else None,
        "train_residual_reg_last": (
            float(history.residual_reg_loss[-1]) if history.residual_reg_loss else None
        ),
        "train_state_smoothness_last": (
            float(history.state_smoothness_loss[-1])
            if history.state_smoothness_loss
            else None
        ),
        "train_cross_l2_last": (
            float(history.cross_l2_loss[-1]) if history.cross_l2_loss else None
        ),
        "train_cross_l1_last": (
            float(history.cross_l1_loss[-1]) if history.cross_l1_loss else None
        ),
        "validation_objective_last": float(history.val_loss[-1]) if history.val_loss else None,
    }


def save_metrics_bundle(
    save_dir: Path,
    history,
    split_results: dict[str, dict[str, Any]],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_validation": _jsonify_history(history),
        "splits": {
            split_name: {
                "mse": float(results["mse"]),
                "mae": float(results["mae"]),
                "delta_mse": float(results["delta_mse"]),
                "residual_alpha": float(results["residual_alpha"]),
                "mean_residual_to_wc_ratio": float(np.mean(results["residual_ratio_over_time"])),
                "collapse_summary": results["collapse_summary"],
                "r2": {
                    region: {
                        metric_name: float(metric_value)
                        for metric_name, metric_value in region_scores.items()
                    }
                    for region, region_scores in results["r2"].items()
                },
                "per_region_residual_stats": results["per_region_residual_stats"],
                "per_region_temporal_stats": results["per_region_temporal_stats"],
            }
            for split_name, results in split_results.items()
        },
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))


def print_wc_parameters(model) -> None:
    if not hasattr(model, "wc") or model.wc is None:
        return

    wc = model.wc
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
