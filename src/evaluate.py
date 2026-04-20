import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_pearson(x, y):
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr = pearsonr(x, y)[0]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def cross_corr_and_lag(target, pred, max_lag=100):
    if np.std(target) < 1e-8 or np.std(pred) < 1e-8:
        return 0.0, 0

    ctr_target = target - np.mean(target)
    ctr_pred = pred - np.mean(pred)
    cc = np.correlate(ctr_target, ctr_pred, mode="full")
    cc = cc / (np.std(target) * np.std(pred) * len(target))

    zero_index = len(target) - 1
    max_lag = min(max_lag, zero_index)
    cc_window = cc[zero_index - max_lag : zero_index + max_lag + 1]
    best_idx = int(np.argmax(cc_window))
    return float(cc_window[best_idx]), int(best_idx - max_lag)


def summarize_couplings(couplings, region_names):
    if couplings is None:
        return None

    entries = []
    for matrix_name in ["W_EE_cross", "W_EI_cross", "W_IE_cross", "W_II_cross"]:
        matrix = couplings.get(matrix_name)
        if matrix is None:
            continue
        for target_idx, target_region in enumerate(region_names):
            for source_idx, source_region in enumerate(region_names):
                if target_idx == source_idx:
                    continue
                value = float(matrix[target_idx, source_idx])
                entries.append(
                    {
                        "matrix": matrix_name,
                        "source": source_region,
                        "target": target_region,
                        "value": value,
                    }
                )

    if not entries:
        return None

    dense_values = np.array([abs(entry["value"]) for entry in entries], dtype=np.float64)
    density = float(np.mean(dense_values > 1e-3))
    strongest_positive = sorted(entries, key=lambda entry: entry["value"], reverse=True)[:5]
    strongest_negative = sorted(entries, key=lambda entry: entry["value"])[:5]

    return {
        "density": density,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative,
    }


def evaluate_model(
    model,
    test_dataset,
    device="cpu",
    prediction_mode="rollout",
    max_eval_steps=None,
    region_names=None,
):
    """
    Evaluates a trained model on the held-out temporal segment.
    """
    model.to(device)
    model.eval()

    if max_eval_steps is not None:
        eval_stop = min(len(test_dataset.x), max_eval_steps + 1)
        x_source = test_dataset.x[:eval_stop]
        u_source = test_dataset.u[:eval_stop]
    else:
        x_source = test_dataset.x
        u_source = test_dataset.u

    x_full_seq_in = x_source[:-1].unsqueeze(0).to(device)
    u_full_seq = u_source[:-1].unsqueeze(0).to(device)
    teacher_forcing = prediction_mode == "one_step"

    with torch.no_grad():
        x0_stub = x_full_seq_in[:, 0, :]
        out = model(x0=x0_stub, u_seq=u_full_seq, x_seq=x_full_seq_in, teacher_forcing=teacher_forcing)

    if isinstance(out, tuple):
        preds_2d = out[0][0].detach().cpu().numpy()
        all_E = out[1][0].detach().cpu().numpy()
        all_I = out[2][0].detach().cpu().numpy()
    else:
        preds_2d = out[0].detach().cpu().numpy()
        all_E = None
        all_I = None

    all_targets_2d = x_source[1:].cpu().numpy()
    obs_dim = all_targets_2d.shape[1]
    if region_names is None:
        region_names = [f"region_{idx}" for idx in range(obs_dim)]

    all_preds_flat = preds_2d.reshape(-1)
    all_targets_flat = all_targets_2d.reshape(-1)

    metrics = {
        "MSE": float(mean_squared_error(all_targets_flat, all_preds_flat)),
        "MAE": float(mean_absolute_error(all_targets_flat, all_preds_flat)),
        "R2": float(r2_score(all_targets_flat, all_preds_flat)),
        "Pearson": safe_pearson(all_targets_flat, all_preds_flat),
    }

    overall_cc, overall_lag = cross_corr_and_lag(all_targets_flat, all_preds_flat)
    metrics["Cross_Corr"] = float(overall_cc)
    metrics["Lag"] = int(overall_lag)

    target_diff = np.diff(all_targets_flat)
    pred_diff = np.diff(all_preds_flat)
    metrics["Deriv_Consistency"] = safe_pearson(target_diff, pred_diff)
    metrics["NaN_Preds"] = bool(np.isnan(preds_2d).any())
    metrics["Pred_Min"] = float(np.min(preds_2d))
    metrics["Pred_Max"] = float(np.max(preds_2d))
    metrics["Pred_Var"] = float(np.var(preds_2d))

    if all_E is not None and all_I is not None:
        latent_concat = np.concatenate([all_E, all_I], axis=-1)
        metrics["Max_Latent_Abs"] = float(np.max(np.abs(latent_concat)))
        metrics["NaN_Latents"] = bool(np.isnan(latent_concat).any())
    else:
        metrics["Max_Latent_Abs"] = 0.0
        metrics["NaN_Latents"] = False

    per_region = {}
    for region_idx, region_name in enumerate(region_names):
        target_r = all_targets_2d[:, region_idx]
        pred_r = preds_2d[:, region_idx]
        region_cc, region_lag = cross_corr_and_lag(target_r, pred_r)
        per_region[region_name] = {
            "MSE": float(mean_squared_error(target_r, pred_r)),
            "MAE": float(mean_absolute_error(target_r, pred_r)),
            "Pearson": safe_pearson(target_r, pred_r),
            "Cross_Corr": float(region_cc),
            "Lag": int(region_lag),
            "Deriv_Consistency": safe_pearson(np.diff(target_r), np.diff(pred_r)),
            "Pred_Min": float(np.min(pred_r)),
            "Pred_Max": float(np.max(pred_r)),
            "Pred_Var": float(np.var(pred_r)),
        }

    metrics["Per_Region"] = per_region

    couplings = model.get_coupling_matrices() if hasattr(model, "get_coupling_matrices") else None
    metrics["Coupling_Diagnostics"] = summarize_couplings(couplings, region_names)

    return metrics, preds_2d, all_targets_2d, all_E, all_I, couplings
