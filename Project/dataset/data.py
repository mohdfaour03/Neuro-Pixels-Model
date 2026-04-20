"""Dataset loading and preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    t_train: torch.Tensor
    t_val: torch.Tensor
    t_test: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    u_train: torch.Tensor
    u_val: torch.Tensor
    u_test: torch.Tensor
    beh_train: torch.Tensor
    beh_val: torch.Tensor
    beh_test: torch.Tensor
    t_ds_train: np.ndarray
    t_ds_val: np.ndarray
    t_ds_test: np.ndarray
    y_ds_train: np.ndarray
    y_ds_val: np.ndarray
    y_ds_test: np.ndarray
    running_ds_train: np.ndarray
    running_ds_val: np.ndarray
    running_ds_test: np.ndarray
    pupil_ds_train: np.ndarray
    pupil_ds_val: np.ndarray
    pupil_ds_test: np.ndarray
    regions: List[str]
    feature_names: List[str]
    y_min: np.ndarray
    y_max: np.ndarray
    u_scaler: StandardScaler
    beh_scaler: StandardScaler


def _downsample(values: np.ndarray, downsample: int, trailing_shape: tuple) -> np.ndarray:
    n_trim = (len(values) // downsample) * downsample
    return values[:n_trim].reshape(-1, downsample, *trailing_shape).mean(axis=1)


def load_preprocess_data(
    data_path: Union[str, Path],
    device: torch.device,
    downsample: int = 40,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> PreparedData:
    """Load the rich stimulus dataset and return train/validation/test tensors."""
    data = np.load(Path(data_path).expanduser(), allow_pickle=True)

    y_raw = data["Y"]
    u_raw = data["U_rich"]
    t_raw = data["time_bins"]
    regions = list(data["regions"])
    feature_names = list(data["feature_names"])
    running_raw = data["running_speed"]
    pupil_raw = data["pupil_width"]

    train_end = int(train_fraction * len(t_raw))
    val_end = train_end + int(val_fraction * len(t_raw))
    y_raw_train, y_raw_val, y_raw_test = (
        y_raw[:train_end, :],
        y_raw[train_end:val_end, :],
        y_raw[val_end:, :],
    )
    u_raw_train, u_raw_val, u_raw_test = (
        u_raw[:train_end, :],
        u_raw[train_end:val_end, :],
        u_raw[val_end:, :],
    )
    t_raw_train, t_raw_val, t_raw_test = t_raw[:train_end], t_raw[train_end:val_end], t_raw[val_end:]
    running_raw_train, running_raw_val, running_raw_test = (
        running_raw[:train_end],
        running_raw[train_end:val_end],
        running_raw[val_end:],
    )
    pupil_raw_train, pupil_raw_val, pupil_raw_test = (
        pupil_raw[:train_end],
        pupil_raw[train_end:val_end],
        pupil_raw[val_end:],
    )

    n_regions = y_raw.shape[1]
    n_stim = u_raw.shape[1]
    y_ds_train = _downsample(y_raw_train, downsample, (n_regions,))
    y_ds_val = _downsample(y_raw_val, downsample, (n_regions,))
    y_ds_test = _downsample(y_raw_test, downsample, (n_regions,))
    u_ds_train = _downsample(u_raw_train, downsample, (n_stim,))
    u_ds_val = _downsample(u_raw_val, downsample, (n_stim,))
    u_ds_test = _downsample(u_raw_test, downsample, (n_stim,))
    t_ds_train = _downsample(t_raw_train, downsample, ())
    t_ds_val = _downsample(t_raw_val, downsample, ())
    t_ds_test = _downsample(t_raw_test, downsample, ())
    running_ds_train = _downsample(running_raw_train, downsample, ())
    running_ds_val = _downsample(running_raw_val, downsample, ())
    running_ds_test = _downsample(running_raw_test, downsample, ())
    pupil_ds_train = _downsample(pupil_raw_train, downsample, ())
    pupil_ds_val = _downsample(pupil_raw_val, downsample, ())
    pupil_ds_test = _downsample(pupil_raw_test, downsample, ())

    running_ds_train = np.clip(running_ds_train, 0, None)
    running_ds_val = np.clip(running_ds_val, 0, None)
    running_ds_test = np.clip(running_ds_test, 0, None)

    pupil_clip_train = np.percentile(pupil_ds_train, 99)
    pupil_ds_train = np.clip(pupil_ds_train, 0, pupil_clip_train)
    pupil_ds_val = np.clip(pupil_ds_val, 0, pupil_clip_train)
    pupil_ds_test = np.clip(pupil_ds_test, 0, pupil_clip_train)

    t_min, t_max = t_ds_train.min(), t_ds_train.max()
    t_norm_train = (t_ds_train - t_min) / (t_max - t_min)
    t_norm_val = (t_ds_val - t_min) / (t_max - t_min)
    t_norm_test = (t_ds_test - t_min) / (t_max - t_min)

    u_scaler = StandardScaler()
    u_norm_train = u_scaler.fit_transform(u_ds_train)
    u_norm_val = u_scaler.transform(u_ds_val)
    u_norm_test = u_scaler.transform(u_ds_test)

    beh_scaler = StandardScaler()
    beh_raw_train = np.column_stack([running_ds_train, pupil_ds_train])
    beh_raw_val = np.column_stack([running_ds_val, pupil_ds_val])
    beh_raw_test = np.column_stack([running_ds_test, pupil_ds_test])
    beh_norm_train = beh_scaler.fit_transform(beh_raw_train)
    beh_norm_val = beh_scaler.transform(beh_raw_val)
    beh_norm_test = beh_scaler.transform(beh_raw_test)

    y_min = y_ds_train.min(axis=0, keepdims=True)
    y_max = y_ds_train.max(axis=0, keepdims=True)
    y_norm_train = (y_ds_train - y_min) / (y_max - y_min + 1e-10)
    y_norm_val = (y_ds_val - y_min) / (y_max - y_min + 1e-10)
    y_norm_test = (y_ds_test - y_min) / (y_max - y_min + 1e-10)

    return PreparedData(
        t_train=torch.tensor(t_norm_train, dtype=torch.float32, device=device).unsqueeze(1),
        t_val=torch.tensor(t_norm_val, dtype=torch.float32, device=device).unsqueeze(1),
        t_test=torch.tensor(t_norm_test, dtype=torch.float32, device=device).unsqueeze(1),
        y_train=torch.tensor(y_norm_train, dtype=torch.float32, device=device),
        y_val=torch.tensor(y_norm_val, dtype=torch.float32, device=device),
        y_test=torch.tensor(y_norm_test, dtype=torch.float32, device=device),
        u_train=torch.tensor(u_norm_train, dtype=torch.float32, device=device),
        u_val=torch.tensor(u_norm_val, dtype=torch.float32, device=device),
        u_test=torch.tensor(u_norm_test, dtype=torch.float32, device=device),
        beh_train=torch.tensor(beh_norm_train, dtype=torch.float32, device=device),
        beh_val=torch.tensor(beh_norm_val, dtype=torch.float32, device=device),
        beh_test=torch.tensor(beh_norm_test, dtype=torch.float32, device=device),
        t_ds_train=t_ds_train,
        t_ds_val=t_ds_val,
        t_ds_test=t_ds_test,
        y_ds_train=y_ds_train,
        y_ds_val=y_ds_val,
        y_ds_test=y_ds_test,
        running_ds_train=running_ds_train,
        running_ds_val=running_ds_val,
        running_ds_test=running_ds_test,
        pupil_ds_train=pupil_ds_train,
        pupil_ds_val=pupil_ds_val,
        pupil_ds_test=pupil_ds_test,
        regions=regions,
        feature_names=feature_names,
        y_min=y_min,
        y_max=y_max,
        u_scaler=u_scaler,
        beh_scaler=beh_scaler,
    )
