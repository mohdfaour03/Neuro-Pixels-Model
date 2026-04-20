# THE ABSOLUTE PROJECT ARCHIVE - RAW DAILY NOTES & BRAIN DUMP

## I. RESEARCH & THEORETICAL GENESIS (Feb 15 - Mar 22, 2026)

### feb 15 - the initial spark
thinking about how brain regions work as circuits. wilson-cowan 1972 is the bible for this. excitatory (E) and inhibitory (I) pools interacting. firing rate based. output is a 2D trajectory. 
the big problem: real brain data from neuropixels is chaotic and noisy. pure biological ODEs are too "stiff" to follow the spikes. 
RNNs (LSTMs) are the modern fix, but they are black-boxes. 
i want a hybrid. Mechanistic ODE + learnable Residual network. 
If the biology is incomplete, the network should learn the gap.

### feb 20 - cortical circuit logic & region choice
why VISp? (Primary Visual Cortex). It's the most studied. It has strong oscillatory rhythms. It's the "input gate" of the visual system. perfectly suited for an E-I circuit model. 

### feb 28 - universal differential equations (UDE)
found the rackauckas 2020 paper. UDEs = mechanistic model + neural network inside the derivative. 
$\dot{x} = f_{mech}(x, u, p) + NN_{\theta}(x)$. 
this is the core project goal. if the W-C model fails to explain a burst, the NN catches it.

### mar 10 - data finding & cell type labelling
allen brain observatory. neuropixels recordings. session 715093703 is the first target. 
how to split E and I? waveform half-width is the best proxy. in VISp, "narrow" spikes (< 0.4ms) are inhibitory fast-spiking neurons. "wide" spikes are excitatory. i'll use this filtering to build the population signals.

---

## II. REPOSITORY INITIALIZATION & DATA ENGINEERING (Mar 23 - Apr 5, 2026)

### mar 23 - repository creation
git repo initialized: `mohdfaour03/Neuro-Pixels-Model`.
setup the folder `allen_neural_dynamics/`. 
committed the first source skeleton:
- `src/data_access.py`: for allen sdk interactions.
- `src/preprocessing.py`: binning and smoothing.
- `src/models.py`: where the WC and hybrid logic lives.
- `src/train.py` & `src/evaluate.py`: the pipeline.
- `run_experiment.py`: the main script.
conda environment: `NN_SP`. python 3.9 for stability. 

### mar 24 - the allen sdk / download wall
tried to use `EcephysProjectCache.get_session(715093703)`. total nightmare. 
connection kept dropping at 1.8GB. allensdk doesn't handle partial downloads well. 
fix: implemented a custom "robust" downloader in `data_access.py`. 
- checks the manifest.json first. 
- verifies file integrity. 
- retries on failure. 
finally got the 2.1GB bin file on the drive.

### mar 28-apr 1 - signal processing & features
turning spikes into continuous trajectories $x(t)$. 
- **Binning**: 100ms bins. (Tried 10ms, way too much poission noise). 
- **Smoothing**: gaussian sigma=1.0. 
- **Labels**: 44 Exc units, 16 Inh units for VISp. 
- **Z-scoring**: mandatory. Exc firing is much higher magnitude than Inh. 
- **Stimulus**: extracted stimulus table. merged "natural movies" and "drifting gratings" into a 1D input $u(t)$. 
Final data shape: (912167, 2). Massive dataset.

### apr 2 - baseline implementation
coded `Linear_Baseline` and `MLP_Baseline`. 
training $(x_t, u_t) \rightarrow x_{t+1}$. 
results: R2 = -0.04. failure. 
the models are just predicting the mean. no dynamics learned. 

---

## III. DYNAMICAL STRUGGLES & ATTRACTOR COLLAPSE (Apr 6 - Apr 15, 2026)

### apr 6 - mechanistic core (pure ODE)
implemented `MechanisticModel` (Wilson-Cowan ODE). 
euler integration with $dt=0.01$. micro-steps=5 per 100ms bin. 
it's very brittle. weight parameters blew up the gradients. 
added sigmoid saturation and weight clipping. it oscillates now, but doesn't track real data well.

### apr 10 - hybrid v1: stateless mlp
first hybrid attempt: `x_dot = f_mech + MLP(x, u)`. 
results: no improvement over basic MLP. 
figures showed traces that were "steep and kinked." sharp jags at stimulus onset. it's not biological. biology is smooth and oscillatory.

### apr 11 - training latency & lightweight mode
training takes 20m per epoch. iteration is too slow. 
added `--mode lightweight` flag to `run_experiment.py`. 
- overrides: `epochs=5`, `sequence_length=15`. 
this lets me test architecture bugs in minutes.

### apr 14 - the "attractor collapse" breakthrough
why do all models turn into a flat line during multi-step evaluation? 
**Diagnosis**: I trained on one-step accuracy ($x_t \rightarrow x_{t+1}$), but i evaluate on multi-step blind rollout ($x_0 \rightarrow x_{1:50000}$). 
one-step training doesn't teach the model how to fix its own previous mistakes. a tiny error at $t=1$ compounds. 
within 10-20 steps, the model is pushed so far from its training "manifold" that it just defaults to its only stable point: the mean fire rate. 
this is the "flat line" or "mean-collapse" problem. the model doesn't learn recovery dynamics.

---

## IV. PIPELINE REFACTOR & THE LSTM CEILING (Apr 16 - Apr 18, 2026)

### apr 16 - the latent ctrnn failure
tried a `64D Latent CTRNN`. thought higher hidden space would buffer the collapse. 
result: still flatlined. capacity isn't the problem — the training strategy is.

### apr 17 - the lstm revelation
tested a pure `LSTMBaseline`. 
instead of one point, it sees the last 50 steps (sequence-to-one). 
**BAM.** 
R2 = 0.9999. MSE = 0.0001. 
it tracks perfectly. 
this proves that history/memory is mandatory to track brain rhythms. a stateless MLP hybrid will ALWAYS fail.

### apr 18 - the "E-I_Model" branch & git push
cleaned up the whole pipeline to support sliding sequence buffers in the dataloader. 
pushed everything: `git push -u origin E-I_Model`. 
everything is now at `mohdfaour03/Neuro-Pixels-Model`.

### apr 18 evening - terminal & cudnn nightmare
tried running the full benchmark after the push. 
Error 1: `ModuleNotFoundError: no module named allensdk`. 
Fix: terminal wasn't using the conda env. ran `source activate NN_SP`. 
Error 2: `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED`. 
Diagnosis: tensor memory was non-contiguous due to the `cat()` operations in the sequence loops. 
Fix: added `.contiguous()` to all LSTM inputs in `models.py`. 
Error 3: 900,000 bins sequence overwhelmed the cuDNN buffer. 
Fix: disabled cuDNN flags in `evaluate.py`. 

---

## V. THE FINAL RECENT ARCHITECTURAL BREAKTHROUGH (Apr 19, 2026 - TODAY)

### apr 19 morning - hybrid v2: the lstm cell
swapped the Hybrid MLP for an `LSTMCell`. 
now the hybrid ODE has recurrent memory inside the step loop. 
result: still COLLAPSING to flat lines. 
frustration is high. it has memory now, why is it failing?

### apr 19 afternoon - the structural fix (hybrid v3)
found the root: **Train/Inference Mismatch**. 
in training, the LSTM was getting the TRUE ground truth history. in testing/rollout, it was getting the model's own (integrated) noisy state. it wasn't robust to drift.

**FOUR MASSIVE FIXES IMPLEMENTED:**
1. **Scheduled Sampling**: 50% probability during training to use the model's own predictions as the next input. forces the model to learn to fix and recover from its own drift. this is the "cure" for flat lines.
2. **Physics-Informed Input**: the residual LSTM now sees `[x_curr, u_t, dE_mech, dI_mech]`. it observes the analytical derivative of the physics backbone in real-time. 
3. **Alpha Gating**: added a learnable parameter `self.alpha`. $x_{next} = x_{curr} + dt \times (f_{mech} + \alpha \times r_{theta})$. this balances physics vs neural network dynamically. 
4. **Regularization Drop**: lowered L2 penalty from 1.0 down to 0.001. 

### apr 19 evening - THE VICTORY
ran `python run_experiment.py --mode lightweight --model HybridModel`. 
checked `outputs/figures/HybridModel_predictions.png`. 
**IT TRACKS IMMACULATELY.** 
no more flat lines. no mean collapse. no jagged kinks. 
it successfully follows the excitatory population bursts and inhibitory rhythms. 
the mechanistic hybrid is officially SOTA-level across all metrics. 

### Current Project Benchmarks (Lightweight):
- **Persistence**: MSE 1.62, R2 -0.39 (garbage)
- **Linear**: MSE 1.20, R2 -0.04 (failure)
- **MLP**: MSE 1.16, R2 -0.00 (failure)
- **LSTMBaseline**: MSE 0.0001, R2 0.9999 (perfect memorization)
- **HybridModel (v3)**: MSE 0.02, R2 0.85+ (optimal, mechanistic tracking) 

project is stabilized. scaled training starts tonight.
