"""Project configuration defaults."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_path: Path = Path(__file__).resolve().parents[1] / "data/region_activity_with_rich_stim.npz"
    downsample: int = 40
    train_fraction: float = 0.70
    val_fraction: float = 0.15


@dataclass
class ModelConfig:
    n_fourier: int = 128
    fourier_freqs: int = 64
    min_freq: float = 1.0
    max_freq: float = 1000.0
    n_stim: int = 15
    n_regions: int = 4
    n_behavioral: int = 2
    hidden: int = 128
    gru_hidden: int = 64
    gru_layers: int = 2


@dataclass
class TrainingConfig:
    n_epochs: int = 500
    lr: float = 1e-3
    scheduler_t_max: int = 2000
    scheduler_eta_min: float = 1e-5
    lambda_phys_max: float = 0.3
    lambda_ic: float = 10.0
    lambda_dyn: float = 5.0
    dyn_floor_E: float = 0.2
    dyn_floor_I: float = 0.1
    ramp_epochs: int = 500
    dt: float = 0.8
    dyn_window: int = 50
    dyn_stride: int = 25
    n_phys_samples: int = 2000
    grad_clip: float = 1.0
    log_every: int = 50
    checkpoint_warmup: int = 0
    checkpoint_path: Path = Path(__file__).resolve().parent / "checkpoints" / "best_validation.pt"
    seed: int = 42


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
