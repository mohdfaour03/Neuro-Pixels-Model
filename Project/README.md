# WC-UDE Project

This folder converts `Notebooks/WC_UDE_Colab_Suhaib.ipynb` into a small research-style Python project.

## Structure

- `dataset/`: data loading, train/validation/test split, downsampling, clipping, scaling, and tensor conversion.
- `Network/`: Fourier features, `TrajectoryNet`, and Wilson-Cowan physics modules.
- `Training/`: training loop with data, initial-condition, dynamic, physics, tqdm progress, validation, and best-checkpoint saving.
- `Evaluation/`: R2/MSE metrics, learned parameter reporting, and plots.
- `utils/`: seed/reproducibility helpers.
- `config.py`: dataclass defaults for data, model, and training settings.
- `main.py`: end-to-end command-line experiment runner.

## Run

From this directory:

```bash
python main.py
```

From the repository root:

```bash
python Project/main.py
```

Useful options:

```bash
python Project/main.py --epochs 400 --downsample 40 --save-dir Project/results
python Project/main.py --epochs 5 --no-plots
python Project/main.py --checkpoint-path Project/checkpoints/best_validation.pt
```

The default data path points to `../region_activity_with_rich_stim.npz`, matching the current repository layout.
The default split is `70/15/15` for train/validation/test. During training, the best validation weights are saved to `Project/checkpoints/best_validation.pt` and loaded back before final train/validation/test evaluation.

By default, final figures are saved as SVG files in `results/`, and metrics are saved to `results/metrics.json`. These result files are intentionally kept in version control.
