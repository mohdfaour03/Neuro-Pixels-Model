# Wilson-Cowan UDE for Visual Cortex Dynamics

This repository implements a physics-informed universal differential equation model for neural activity across visual cortical regions. The model combines stimulus-driven neural prediction with latent excitatory/inhibitory Wilson-Cowan dynamics.

## Method

The pipeline trains a `TrajectoryNet` with Fourier time features, rich stimulus inputs, behavioral signals, and a Wilson-Cowan physics regularizer. Data are split into train, validation, and test segments, with the best model selected by validation MSE.

## Repository

- `Project/`: source code for data preparation, network modules, training, and evaluation.
- `Notebooks/`: original experimental notebooks.
- `data/`: input `.npz` dataset.
- `results/`: saved SVG figures and metrics.

## Run

```bash
python Project/main.py
```

Useful options:

```bash
python Project/main.py --epochs 400
python Project/main.py --save-dir results
```

## Outputs

The training script saves the best validation checkpoint, final evaluation metrics, and SVG figures for losses, trajectories, and learned E/I dynamics.
