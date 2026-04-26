"""Project configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASELINE_VARIANT = "baseline_latent_wc_regularized"
WC_ONLY_VARIANT = "wc_only"
WC_PLUS_RESIDUAL_VARIANT = "wc_plus_residual"
RESIDUAL_ONLY_VARIANT = "residual_only"
EXPLICIT_WC_RESIDUAL_ALIAS = "explicit_wc_residual"

MODEL_VARIANTS = (
    BASELINE_VARIANT,
    WC_ONLY_VARIANT,
    WC_PLUS_RESIDUAL_VARIANT,
    RESIDUAL_ONLY_VARIANT,
)


def normalize_variant_name(variant: str) -> str:
    if variant == EXPLICIT_WC_RESIDUAL_ALIAS:
        return WC_PLUS_RESIDUAL_VARIANT
    return variant


def is_explicit_rollout_variant(variant: str) -> bool:
    return normalize_variant_name(variant) in {
        WC_ONLY_VARIANT,
        WC_PLUS_RESIDUAL_VARIANT,
        RESIDUAL_ONLY_VARIANT,
    }


@dataclass
class DataConfig:
    data_path: Path = Path(__file__).resolve().parents[1] / "data/region_activity_with_rich_stim.npz"
    downsample: int = 40
    train_fraction: float = 0.70
    val_fraction: float = 0.15


@dataclass
class ModelConfig:
    variant: str = BASELINE_VARIANT
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
    residual_hidden: int = 32
    residual_layers: int = 2
    initial_hidden: int = 16
    residual_scale: float = 0.05
    use_behavior_in_residual: bool = False
    readout_type: str = "excitatory_direct"
    integrator: str = "euler"


@dataclass
class TrainingConfig:
    n_epochs: int = 500
    lr: float = 1e-3
    scheduler_t_max: int = 500
    scheduler_eta_min: float = 1e-5
    lambda_phys_max: float = 0.3
    lambda_ic: float = 10.0
    delta_loss_weight: float = 0.25
    lambda_dyn: float = 5.0
    dyn_floor_E: float = 0.2
    dyn_floor_I: float = 0.1
    ramp_epochs: int = 500
    integration_step: float | None = None
    residual_reg_weight: float = 1e-2
    state_smoothness_weight: float = 1e-3
    cross_l2_weight: float = 0.0
    cross_l1_weight: float = 0.0
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
