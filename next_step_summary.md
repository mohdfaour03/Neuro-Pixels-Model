# Next Step Summary

## Architectural Changes

- Preserved the original method as `baseline_latent_wc_regularized`.
- Added explicit rollout variants:
  - `wc_only`
  - `wc_plus_residual`
  - `residual_only`
- Refactored the explicit path into separate pieces for:
  - preprocessing
  - latent initialization
  - Wilson-Cowan derivative
  - residual derivative
  - state integration
  - observation readout
  - losses
  - evaluation and plotting
- The new explicit model exposes:
  - WC derivative alone
  - residual derivative alone
  - total derivative
- The default explicit readout is now the excitatory state directly.
- Evaluation now saves rollout artifacts and residual diagnostics for train/val/test.

## Files Added / Modified

Added:

- `Project/dataset/__init__.py`
- `Project/Network/__init__.py`
- `Project/Network/components.py`
- `Project/Training/__init__.py`
- `Project/Training/losses.py`
- `Project/Evaluation/__init__.py`
- `Project/utils/__init__.py`
- `next_step_summary.md`

Modified:

- `Project/config.py`
- `Project/main.py`
- `Project/dataset/data.py`
- `Project/Network/models.py`
- `Project/Training/train.py`
- `Project/Evaluation/evaluate.py`
- `Project/Evaluation/metrics.py`
- `Project/Evaluation/plots.py`
- `README.md`
- `notebook.md`

## Exact Commands Run

Environment check:

```bash
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --help"
```

Short explicit smoke test:

```bash
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --variant wc_plus_residual --epochs 1 --save-dir results --no-plots"
```

Ablation sweep used for saved comparison outputs:

```bash
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --variant baseline_latent_wc_regularized --epochs 3 --downsample 120 --save-dir results"
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --variant wc_only --epochs 3 --downsample 120 --save-dir results"
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --variant wc_plus_residual --epochs 3 --downsample 120 --save-dir results --residual-reg-weight 0.05"
cmd.exe /C "call C:\Users\shafi\anaconda3\Scripts\activate.bat NN_SP && python Project\main.py --variant residual_only --epochs 3 --downsample 120 --save-dir results --residual-reg-weight 0.05"
```

## Key Train / Val / Test Numbers

### `baseline_latent_wc_regularized`

- validation MSE best: `0.017236`
- train MSE: `0.019975`
- val MSE: `0.017236`
- test MSE: `0.014600`

### `wc_only`

- validation MSE best: `0.228354`
- train MSE: `0.314525`
- val MSE: `0.228354`
- test MSE: `0.210745`

### `wc_plus_residual`

- validation MSE best: `0.227401`
- train MSE: `0.313139`
- val MSE: `0.227401`
- test MSE: `0.209370`

### `residual_only`

- validation MSE best: `0.015598`
- train MSE: `0.037116`
- val MSE: `0.015598`
- test MSE: `0.013098`

## Did The Residual Stay Small?

- `wc_plus_residual`: no, not really.
- Mean residual-to-WC ratio:
  - train: `0.7196`
  - val: `0.8403`
  - test: `0.9936`
- Per-region test mean ratio was about `0.99` in all four regions.
- So the explicit residual path is implemented correctly, but in this short run the residual was still roughly the same size as the mechanistic term on the test split.

For the sanity check:

- `residual_only` had effectively zero WC contribution by construction and still fit very well.
- That means the task is still easy enough for the learned correction to dominate if left unconstrained.

## Is The Branch Ready For Symbolic Regression?

- Structurally: yes.
  - The residual is explicit.
  - The saved outputs make symbolic regression dataset extraction well-defined.
  - WC, residual, and total derivatives are separated and logged.
- Scientifically: not yet.
  - `wc_plus_residual` does not yet keep the residual clearly small.
  - `residual_only` is still too competitive.
  - The next step should be to make the WC backbone carry more of the dynamics before trying to distill the residual symbolically.

## Saved Results

Variant-specific outputs are in:

- `results/baseline_latent_wc_regularized/`
- `results/wc_only/`
- `results/wc_plus_residual/`
- `results/residual_only/`

Each folder contains metrics, plots, saved rollout arrays, and the best validation checkpoint.
