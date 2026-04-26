# Wilson-Cowan Backbone + Explicit Residual UDE

This branch now supports two scientifically distinct paths:

- `baseline_latent_wc_regularized`: the original method, where a `TrajectoryNet` predicts activity directly and a separate Wilson-Cowan module regularizes latent derivatives.
- explicit rollout variants: a true state-space Wilson-Cowan backbone with an optional neural residual correction.

The explicit model evolves a latent state

```text
dx/dt = f_WC(x, u; theta) + r_phi(x, u, optional_behavior)
```

where `x = [E_1, ..., E_R, I_1, ..., I_R]`. Observed neural activity is read out explicitly from the excitatory state by default.

## Why This Refactor Matters

The old path was useful as a baseline, but it was still a latent predictor with a physics regularizer. The new path is closer to the intended UDE formulation:

- initialize latent state `x0`
- compute Wilson-Cowan derivative
- compute residual derivative
- combine them explicitly
- integrate the state forward over time
- read out predicted observations from the mechanistic state

That makes the residual a clean object for later symbolic regression, because it is now an explicit correction term rather than something buried inside the whole latent dynamics.

## Variants

- `baseline_latent_wc_regularized`
  - preserved baseline
  - direct predictor + WC derivative matching regularizer
- `wc_only`
  - explicit Wilson-Cowan rollout
  - no neural residual
- `wc_plus_residual`
  - explicit Wilson-Cowan rollout
  - small residual MLP correction
- `residual_only`
  - explicit rollout sanity check with no WC contribution

`explicit_wc_residual` is also accepted as a CLI alias for `wc_plus_residual`.

## Repository Layout

- `Project/dataset/`
  - data loading and preprocessing
- `Project/Network/`
  - baseline model
  - Wilson-Cowan derivative
  - residual derivative
  - latent initialization
  - state integration
  - readout
- `Project/Training/`
  - training loop
  - baseline losses
  - explicit residual losses
- `Project/Evaluation/`
  - metrics
  - saved rollout outputs
  - diagnostic plots

## Environment

This repo was run from the existing conda environment `NN_SP`.

On Windows/Anaconda:

```bash
conda activate NN_SP
```

## Run

Generic pattern:

```bash
python Project/main.py --variant <variant_name> --save-dir results
```

Examples:

```bash
python Project/main.py --variant baseline_latent_wc_regularized --save-dir results
python Project/main.py --variant wc_only --save-dir results
python Project/main.py --variant wc_plus_residual --save-dir results
python Project/main.py --variant residual_only --save-dir results
```

Useful options:

```bash
python Project/main.py --variant wc_plus_residual --epochs 50 --residual-hidden-size 32
python Project/main.py --variant wc_plus_residual --residual-reg-weight 0.05
python Project/main.py --variant wc_plus_residual --integration-step 0.8 --integrator euler
python Project/main.py --variant wc_plus_residual --use-behavior-in-residual
python Project/main.py --variant wc_only --downsample 120
```

## Outputs

Each run now saves into a variant-specific folder:

```text
results/
  baseline_latent_wc_regularized/
  wc_only/
  wc_plus_residual/
  residual_only/
```

Each variant folder contains:

- `config.json`
- `metrics.json`
- `train_outputs.npz`, `val_outputs.npz`, `test_outputs.npz`
- `*_residual_stats.json`
- `loss_curves.svg`
- `train_trajectory.svg`, `val_trajectory.svg`, `test_trajectory.svg`
- `train_ei_dynamics.svg`, `test_ei_dynamics.svg`
- `train_derivatives.svg`, `test_derivatives.svg`
- `train_residual_ratio.svg`, `test_residual_ratio.svg`
- `test_residual_stats.svg`
- `checkpoints/best_validation.pt`

The saved `.npz` files include:

- predicted trajectories
- latent `E` and `I` trajectories
- WC derivative term over time
- residual derivative term over time
- total derivative term over time
- residual-to-WC ratios over time

## Current Scientific Read

This branch is now the right bridge between the old regularized latent predictor and future symbolic regression work because:

- the mechanistic backbone is explicit
- the residual is explicit and separately measurable
- rollout diagnostics are saved automatically
- ablations are now easy to compare

But the branch is not yet fully ready for symbolic regression as a final step if the residual remains too large or if `residual_only` stays too competitive. That is now measurable directly from the saved diagnostics instead of being hidden inside the model.

See `next_step_summary.md` for the concrete run commands and the current ablation results from this branch.
