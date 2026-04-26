project notebook / end of day notes

writing this like actual notes now because the old version started sounding too much like a machine log.

## 2026-03-09

- today i finally narrowed the course project down into something i can actually build.
- i kept coming back to the same idea: do **one region, one session, one simple forecasting task** first, then only later go multi-region and symbolic.
- i read more around neural population dynamics and wilson-cowan kept showing up as the obvious mechanistic starting point. small, interpretable, already framed in terms of excitatory/inhibitory populations.
- i also liked the universal differential equation / neural residual idea a lot. basically keep a mechanistic backbone, then let a small learned correction absorb what the simple equations miss.
- main thought today: pure ml can fit but says very little, pure mechanistic is elegant but probably too weak on messy neuropixels data, so the interesting space is the hybrid.

## 2026-03-10

- more reading today. wilson-cowan, neural odes, latent-state models, neural population forecasting papers.
- i kept asking myself whether i should start directly with explicit e/i states or stay scalar first.
- decided to stay scalar first on purpose. before anything fancy i still needed to answer the boring questions:
- can i load one allen session cleanly
- can i align spikes and stimulus
- can i make one full run work end to end without everything breaking
- i wrote down that the first prototype should be intentionally too simple.

## 2026-03-11

- spent today mostly on dataset choice.
- allen brain observatory visual coding neuropixels still looks like the right place to start because the stimulus timing tables are there, the unit metadata is there, and visual cortex is an easy first target.
- decided the first version will use one session and default to `VISp`.
- also decided i want the code modular from day one. i do not want one giant script that becomes impossible to change later.

## 2026-03-12

- i compared the project directions in my head again.
- the compromise i keep landing on:
- persistence baseline for sanity
- linear / tiny mlp baseline
- a small mechanistic wilson-cowan style model
- optional neural residual on top
- also wrote down that all assumptions need to stay visible in config. bin size, smoothing, dt, splits, model type, everything.

## 2026-03-13

- made the variable definitions more concrete.
- first version:
- one scalar regional activity trace `x(t)`
- one binary stimulus input `u(t)`
- predict `x(t+dt)`
- later version:
- explicit or latent `[E(t), I(t)]`
- noted that the mapping from raw spikes to regional activity has to stay explicit or the later interpretation will be fake.

## 2026-03-14

- read more about how people actually build population activity traces from spikes.
- binning and smoothing look like small details, but they basically define the problem.
- decided i want at least two aggregation ideas in mind even if i only implement one first: summed spike counts and unit-normalized firing rate.
- also wrote a note to myself: temporal split only. random split on this kind of time series would be nonsense.

## 2026-03-15

- still mostly proposal thinking.
- i wrote down the semester arc i want:
- build a minimal one-region pipeline
- extend it toward wilson-cowan e/i structure
- test whether a neural residual helps
- if the mechanics survive, extend toward multi-region and symbolic regression
- basically i want the whole project story visible even in the first small prototype.

## 2026-03-16

- asked myself again whether i am using this dataset because it is available or because it actually fits the science.
- answer for now: yes, allen visual coding neuropixels is the practical starting point.
- if i later really care about inhibitory labels in a cleaner sense, i may need something closer to optotagging. but not yet.

## 2026-03-17

- sketched the folder structure.
- roughly:
- `config/`
- `src/`
- `outputs/`
- `data/`
- `notebooks/`
- `run_experiment.py`
- also sketched module names that are basically what the repo became later:
- `data_access.py`
- `preprocessing.py`
- `features.py`
- `datasets.py`
- `models.py`
- `train.py`
- `evaluate.py`
- `visualize.py`
- `utils.py`

## 2026-03-18

- more reading on wilson-cowan + ml hybrids.
- one important note today: if i add a residual network, it has to stay a correction, not silently become the whole model.
- otherwise the “hybrid” story is fake.

## 2026-03-19

- tried to define what a successful first version even means.
- not publishable science yet. just:
- one session loads
- one region extracts
- spikes and stimulus align
- train/val/test are temporal
- baselines run
- mechanistic model does not explode
- plots save

## 2026-03-20

- picked the first plots i want:
- true vs predicted trace
- stimulus under it
- loss curves
- later maybe phase plot and residual effect overlay
- i want the outputs folder to feel like a notebook by itself.

## 2026-03-21

- wrote down the reproducibility basics:
- fixed seeds
- saved config
- explicit session id
- explicit region filter
- explicit smoothing / binning settings

## 2026-03-22

- last prep day before actually creating the local project.
- by now the request in my head was clear:
- one session
- one region
- scalar `x(t)`
- simple `u(t)`
- next-step prediction
- baseline + mechanistic + optional hybrid
- clean modular structure
- no unnecessary complexity yet

## 2026-03-23

- created the local project directory today. this is the real start of the repo.
- i scaffolded the whole first version in one pass:
- config
- data loading
- preprocessing
- features
- dataset
- models
- training
- evaluation
- visualization
- runner script
- notebook skeleton
- the actual experiment definition at this point was:
- one allen session
- one region, ideally `VISp`
- one scalar population activity signal `x(t)`
- one simple binary stimulus signal `u(t)`
- predict `x(t+dt)`
- compare persistence, small learned baselines, mechanistic, and hybrid
- then the environment started fighting me immediately.
- `allensdk` wanted older compatible versions of a bunch of packages, and `pip install -r requirements.txt` ran into a nasty numpy reinstall problem.
- i also got a quoting/syntax mess in generated source code. one of the first runs crashed with:
- `SyntaxError: unexpected character after line continuation character`
- i fixed that by cleaning the broken escaped triple quotes in the source files.
- so the day was basically:
- create repo
- build first pipeline
- hit dependency trouble
- fix syntax trouble
- get to the point where the structure exists and can actually be run
- main takeaway: the project skeleton is real now, but day one was more environment surgery than science.

## 2026-03-24

- dataset day. frustrating day.
- the allen download speed was way slower than normal downloads, and the bigger problem was that interrupted downloads did not resume cleanly.
- i eventually understood the practical issue:
- giant NWB file
- slow direct transfer
- hard timeout from the default AllenSDK path
- partial corrupted file left behind
- next run crashes trying to open the broken partial instead of resuming
- i increased the timeout in the normal data-access path, but that was not enough.
- so i wrote `robust_download.py`.
- this ended up being one of the most useful little utilities in the whole repo.
- what i wanted from it in plain terms:
- if a partial file already exists, resume it instead of pretending nothing happened
- use range requests directly
- stream in chunks
- show progress
- if the server refuses resume, restart cleanly instead of silently corrupting things
- this was the day i stopped trusting AllenSDK to handle a giant download politely and just took control of it myself.

## 2026-03-25

- my pc crashed in the middle of the data work and the IDE froze, so i had to resume from a half-broken state.
- once the file problem was finally out of the way, the next crash was in training:
- `ValueError: optimizer got an empty parameter list`
- of course this came from the persistence baseline, which has no trainable parameters.
- fixed the training code so the persistence case gets handled before creating an optimizer.
- after that the first full experiment actually started training.
- at this stage there was still no progress bar, so the terminal looked dead and i had to estimate runtime from the dataset size and batch count.
- main feeling today: stupid bug, but important. i was finally past download failures and into actual model training.

## 2026-03-26

- first day where the minimal pipeline actually felt alive.
- i checked that `VISp` selection was wired correctly.
- checked that spike bins and stimulus bins were aligned.
- checked that train/val/test were temporal.
- checked that outputs were being written.
- i kept reminding myself not to get too excited about good-looking metrics yet because the scalar one-step task is still the easiest version of the problem.

## 2026-03-27

- mostly reflection today.
- i was already starting to feel that `x(t+1)` from `x(t), u(t)` is scientifically too easy.
- still glad i did it because it gave me a working pipeline to mutate instead of starting from abstraction.

## 2026-03-28

- revisited the proposal direction.
- i do not want this to become just another black-box curve fitting assignment.
- wrote down clearly that the scalar version is phase 1 only, not the destination.

## 2026-03-29

- kept thinking about the latent e/i problem.
- if i only observe one scalar trace, the model can just hide everything in one coordinate and let the other state die.
- this turned out to be a real issue later.

## 2026-03-30

- not much implementation today.
- mostly thinking about how to move from scalar regional activity toward something that matches the proposal better without pretending i have biology i do not actually observe.

## 2026-03-31

- read more on extracellular heuristics for excitatory vs inhibitory cells.
- waveform duration / spike width kept showing up as the common rough separation.
- wrote down the standard caveat too: broad-waveform regular-spiking units are often treated as excitatory, narrow-waveform fast-spiking units often treated as inhibitory, but this is only a heuristic.

## 2026-04-01

- wrote down the longer-term caveat more explicitly today:
- if i want cleaner inhibitory labels later, optotagging would be better than waveform heuristics.
- but i parked that. still too early to widen the scope again.

## 2026-04-02

- thought about evaluation beyond mse.
- if i am going to talk about “dynamics”, i need more than pointwise fit.
- wrote down metrics i want eventually:
- lag
- cross-correlation
- derivative consistency
- stability / boundedness

## 2026-04-03

- revisited the latent-state inference issue from scalar observations.
- takens / delay-coordinate embedding started looking like one possible rescue if `I(t)` keeps collapsing.

## 2026-04-04

- mostly conceptual day.
- i kept asking how much of this is real dynamical forecasting vs just one-step regression with very strong autocorrelation.

## 2026-04-05

- sketched future plots i might want:
- `E(t)` and `I(t)` trajectories together
- `E` vs `I` phase plane
- residual effect overlay
- error over time

## 2026-04-06

- left myself an important warning:
- if i ever get suspiciously perfect metrics, i need to check whether the task itself is trivial or whether evaluation is cheating.
- that warning ended up mattering a lot.

## 2026-04-07

- mostly quiet. mentally preparing the phase-2 extension.

## 2026-04-08

- still mostly planning. no major implementation note.

## 2026-04-09

- lost continuity today because the working session got interrupted by a restart.
- no scientific result here, just a break in momentum.

## 2026-04-10

- prep day again.
- i was ready to push from scalar activity into explicit e/i modeling.

## 2026-04-11

- huge day. phase 2 really started today.
- i extended the one-region scalar setup into a wilson-cowan style latent state `z(t) = [E(t), I(t)]`.
- made the observation model configurable, updated the models to return both predictions and latent trajectories, and added better metrics and plots.
- immediately ran into two practical issues:
- i wanted `E(t)` and `I(t)` plotted directly because they are the actual interesting states now
- json writing broke because some metrics were still numpy / float32 values instead of plain python types
- fixed the serialization issue and added dedicated latent trajectory plots.
- then the real scientific problem hit:
- the inhibitory latent state was basically collapsing to zero.
- i spent the second half of the day reading around this and trying to understand whether i was asking the model to do something impossible.
- first attempt to rescue it was delay embedding / history windows.
- i changed the dataset to include short history windows and added encoders for `E` and `I` from that history.
- it still did not solve the collapse.
- after that i went back to the proposal and realized the bigger issue:
- my preprocessing had mixed all neurons into one scalar population signal, so of course the model had no real reason to keep a separate inhibitory dimension alive.
- so i changed direction:
- instead of pretending to infer everything from one scalar trace, i split units using waveform duration
- threshold around `0.4 ms`
- broad / regular-spiking bucket = excitatory-like
- narrow / fast-spiking bucket = inhibitory-like
- after that the observed sequence became two-channel `[E_t, I_t]` style input/target instead of a single scalar.
- i also fixed the plotting labels so it no longer showed two curves both pretending to be the same `x(t)`.
- main takeaway: the project stopped being a simple scalar forecasting problem today. this is where it really started turning into an e/i dynamics project.

## 2026-04-12

- today was mostly about scientific honesty.
- i started questioning whether my current formulation really looked like the way people study excitatory/inhibitory dynamics in practice.
- i read more and came away with a much clearer view:
- waveform-based separation is usable but still heuristic
- it mostly captures the obvious fast-spiking inhibitory population, not all inhibitory subtypes
- if i later want cleaner cell-type claims, this dataset may not be enough
- the bigger realization today was about the prediction task itself.
- i asked myself whether predicting the next 10 ms step from the current step is even a meaningful problem scientifically.
- answer: not really. at that scale the autocorrelation is so strong that the task can become too easy.
- so i changed the formulation:
- moved from isolated one-step prediction to sequence modeling with actual rollouts
- increased sequence length
- changed the dataset to sequence chunks
- rewrote the models to roll out through the sequence
- updated training to use backpropagation through time
- of course the first results after that were ugly. loss not decreasing properly, predictions off.
- so i added the first stabilization pass:
- gradient clipping
- curriculum learning over rollout horizon
- main takeaway: this was the day i stopped trusting the tiny one-step task and pushed the project toward a more real dynamical setup.

## 2026-04-13

- even after gradient clipping and curriculum, the long-horizon forecasting behavior still looked bad.
- i started accepting a harder truth:
- perpetual free-running forecasting on this biological sequence may just not look clean with the current model class and setup.
- i changed the baselines to behave more like update rules instead of pure direct mappings, and i changed evaluation so it would not punish absurdly long drift without context.
- basically i was trying to separate real model weakness from obviously unfair evaluation.
- main takeaway: a lot of the ugly behavior was coming from the formulation and the evaluation procedure, not just the network weights.

## 2026-04-14

- another dense day.
- i focused on five practical issues:
- hybrid too similar to mlp
- predictions too steep / kinked
- training too slow
- gpu utilization weird
- terminal feedback too poor
- i added `tqdm` progress bars, lightweight vs heavyweight modes, larger batch sizes, changed activations from `ReLU` to `SiLU`, and made the hybrid residual start small.
- still was not happy with continuity, so i kept pushing.
- next round of fixes:
- micro-stepping inside the mechanistic integration
- pretraining the mechanistic core first
- copying those weights into the hybrid
- freezing the mechanistic part initially so the residual only learns the leftover error
- main feeling today: i was spending as much time fighting numerical behavior and evaluation artifacts as actual modeling.

## 2026-04-15

- mostly watched runs and looked at plots.
- i still felt a lot of the ugliness was not “biology”, it was how i was stitching predictions together.

## 2026-04-16

- no clean breakthrough.
- still stuck on the same two suspicions:
- some visible jumps are evaluation artifacts
- if hybrid and mlp look almost the same, then the hybrid is not really behaving like a hybrid

## 2026-04-17

- quiet on paper, but i was clearly heading toward a bigger refactor.
- the two big unresolved issues were still:
- snapping / discontinuity from evaluation
- hybrid identifiability and residual domination

## 2026-04-18

- very intense day.
- i pushed the branch to GitHub today, so this is the point where the project became externally visible and not just local experimentation.
- early in the day the metrics were not amazing, but at least they were not completely dead. the mechanistic and hybrid models were still somewhat competitive with the simpler learned baselines.
- then i dug harder into why the plots looked wrong.
- i realized the evaluation code was snapping predictions back to ground truth every fixed horizon chunk. that created fake discontinuities, which explains why some curves looked worse than they should have.
- i removed that behavior and added stronger control on the residual.
- that exposed a different problem immediately: once i removed the artificial re-anchoring, the models started collapsing toward near-flat or mean-like behavior.
- that was the point where i accepted that the tiny latent ode core alone probably did not have enough capacity to track the messy data.
- so i expanded the model family:
- added `LatentCTRNN`
- added `LSTMBaseline`
- kept iterating on architecture, normalization, losses, and windowing
- the real breakthrough came when i changed the task again:
- instead of long blind autonomous forecasting from one initial point, i restructured it around sliding windows of recent true history
- that turned the problem into contextual sequence forecasting
- and this was the first setup where the LSTM predictions actually sat almost directly on top of the true curves
- the numbers there were basically perfect:
- `MSE` on the order of `2.4e-05`
- `R2` around `0.99998`
- Pearson almost `1`
- main takeaway: today was the big pivot. i stopped asking the models to do impossible autonomous long-horizon magic and finally got a setup that fit the data cleanly.

## 2026-04-19

- today i focused almost entirely on the hybrid.
- i replaced the hybrid residual mlp with an LSTM-based residual because the plain feedforward residual clearly was not enough.
- also added a command-line option to run only one model, which made debugging much easier.
- the result was very asymmetric:
- the pure `LSTMBaseline` looked excellent
- the `HybridModel` was still bad, basically near-flat and mean-collapsing
- this made the real problem very clear:
- the issue was not “recurrent models don’t work”
- the issue was specifically the hybrid formulation
- by the end of the day i had a much better diagnosis:
- training / inference mismatch
- residual too weak or too constrained
- mechanistic core possibly too damped
- residual not seeing the right information about mechanistic failure
- i did not fully fix it today, but i finally felt like i understood why it was failing.

## 2026-04-20

- today was a mix of repo cleanup, notebook reconstruction, and the actual hybrid breakthrough.
- first i cleaned up the repo structure and moved things into the proper root layout. i also made sure the old earlier outputs were still preserved separately.
- i looked back at the earlier “amazing” one-step metrics in the archived outputs and confirmed what i had been suspecting for a while: that earlier task was too easy. those numbers were real, but they were flattering because the evaluation was too local.
- then i went back to the hybrid-only problem and treated it as its own task.
- i inspected only the hybrid path, training path, and evaluation path.
- the big realization:
- the current dataset was now a sliding-window next-step sequence problem
- but the hybrid was still mixing in inconsistent truth resets and mismatched state handling
- that was exactly the kind of setup that can create flat attractors
- so i simplified and made it consistent:
- use the observed current state cleanly during next-step prediction over the window
- keep the model genuinely hybrid as mechanistic update + bounded recurrent residual correction
- gate the residual instead of letting it dominate
- reduce the residual penalty
- add a delta-loss term so flat predictions are less attractive
- i also tightened the lightweight settings so i could actually iterate quickly instead of waiting forever for full evaluation
- after a few reruns the hybrid finally stopped collapsing.
- final lightweight hybrid result today was the first one i actually believed:
- `MSE = 0.0032`
- `MAE = 0.0419`
- `R2 = 0.9942`
- `Pearson = 0.9971`
- derivative consistency also very strong
- more importantly, the saved `HybridModel_predictions.png` no longer showed near-flat lines. the predicted `E(t)` and `I(t)` were finally following the real oscillations instead of drifting toward a dull attractor.
- main takeaway: the catastrophic hybrid failure was not just “the model is bad”. it was mostly train/test mismatch and a bad hybrid setup.

## 2026-04-21

- today i started phase 3 properly: the move toward the actual proposal with multiple regions and explicit inter-region coupling.
- i reread the proposal carefully and the key correction was obvious:
- the observed signal should be one activity trace per region
- the latent state should be coupled `E_r, I_r` pairs across regions
- so i rewrote the preprocessing side first.
- new candidate region set:
- `VISp`
- `VISl`
- `VISal`
- `VISpm`
- `LGd`
- and on the real session the retained regions came out as:
- `VISp` = `60` units
- `VISl` = `42`
- `VISpm` = `50`
- `LGd` = `82`
- `VISal` dropped because it had `0` usable units
- built the aligned observed data as `[T, R]` on shared global bins instead of the old one-region assumptions.
- then i rewrote the model stack around that:
- generic baselines
- multi-region mechanistic model
- multi-region hybrid model
- trainable local and cross-region couplings
- configurable coupling modes like `full`, `EE_only`, `local_only`
- evaluation got rewritten too:
- overall metrics
- per-region metrics
- lag / cross-correlation / derivative consistency
- stability checks
- coupling summaries
- coupling heatmaps
- the first rollout-style runs were honestly bad. mechanistic and hybrid both drifted and lagged.
- so i changed the default training/eval back to one-step for a first stable prototype, but kept rollout as an extra diagnostic.
- this ended up being the right compromise.
- under one-step prediction, the multi-region models looked good and the saved plots were finally clean.
- under rollout, they still drifted badly.
- and that is actually useful. now i have an honest split:
- one-step local prediction works well
- free dynamics are still hard
- i also prepared a short progress presentation today using the saved phase-3 figures and metrics. mostly for the week-before-final talk, not the final story yet.
- best figures at this point are still:
- shared timeline
- coupling heatmaps
- rollout comparison

## 2026-04-22

- phase 4 day.
- i started by building the symbolic-regression layer on top of the trained multi-region hybrid model.
- the core idea was straightforward:
- trace the trained hybrid model
- extract pairs of `[z(t), u(t)] -> residual(t)`
- fit symbolic equations for selected residual outputs
- build a new symbolic residual model and compare it against the pure mechanistic and neural-hybrid versions
- i added the phase-4 config block, symbolic dataset extraction, symbolic residual model, and a dedicated `run_symbolic_regression.py` runner.
- i also made the startup behavior explicit: if PySR / Julia are not ready, the runner should fail honestly instead of silently trying to do magic.
- the Julia install itself was a whole annoying side quest.
- the easy package-manager routes kept stalling, so i ended up installing Julia manually from the official windows installer path.
- after that, `pysr` finally imported correctly in `NN_SP`.
- then the first real phase-4 run immediately reminded me that environment work is never truly over:
- first failure was a package / path issue around `sympy`
- after cleaning that up, the next issue was that the symbolic run was doing too much silent work before giving me anything observable
- so i made the symbolic runner lighter and more transparent:
- capped the traced steps in lightweight mode
- reduced chunk sizes
- added progress prints during symbolic dataset extraction and fitting
- once that was in place, the symbolic side finally started doing real work.
- the pilot symbolic fits that came out were:
- `E:VISp -> I_VISpm`
- `I:VISp -> tiny constant`
- which is modest, but at least interpretable
- then the pipeline crashed on a simple metadata bug because i had forgotten to carry `output_name` into the selected-equation data.
- fixed that and reran.
- the final lightweight symbolic run completed end to end.
- comparison numbers in one-step mode were all basically tied:
- `MechanisticModel`: `MSE ~ 0.0065`, `R2 ~ 0.9918`, `Pearson ~ 0.9959`
- `HybridModel`: almost the same
- `SymbolicResidualModel`: almost the same
- that is not a dramatic scientific victory, but it does mean the whole phase-4 pipeline works technically:
- extract residual dataset
- fit pilot symbolic equations
- build executable symbolic residual model
- compare mechanistic vs neural hybrid vs symbolic hybrid
- the interpretability result is modest but not useless:
- the correction for `E:VISp` basically points to cross-region modulation from `I_VISpm`
- the correction for `I:VISp` was so small that it collapsed to an almost constant term
- main takeaway: phase 4 works, but symbolic compression is still weak scientifically. the one-step task is just too easy to force a rich residual story.

## 2026-04-23

- today i stopped adding models and focused on making the project explainable end to end.
- i wanted one notebook that can function as both a walkthrough and a demo instead of just having scattered outputs everywhere.
- so i read back through the actual helper code and the saved artifact tree and wrote `summary.ipynb`.
- i made it include:
- project roadmap from phase 1 to phase 4
- saved metrics comparisons across archived early runs, current phase 3, and phase 4
- preprocessing rebuild and before/after visualization
- phase-3 figure gallery
- direct latent-state inspection from the trained hybrid checkpoint
- guarded rerun cells for lightweight phase-3 and phase-4 commands
- symbolic equation table and interpretability section
- final conclusions / next steps
- important practical decision there: the rerun cells are guarded by booleans so opening the notebook does not immediately retrain everything.
- i also did a sanity check on the notebook file format and it parsed correctly.
- main takeaway: the project is finally not just “code + outputs + messy notes”. now it has a readable summary notebook too.

## 2026-04-24

- after rewriting this notebook, i moved on to a new notebook task: build an actual experiment notebook that walks through the four phases instead of just summarizing them.
- first i re-read the current runners, training code, model code, dataset code, and saved outputs so the new notebook would match the real project and not just be a pretty shell.
- the important design decision today:
- phase 1 and phase 2 should be recreated directly inside the notebook in lightweight form
- phase 3 and phase 4 can use the existing runner scripts
- and the notebook should start with raw data inspection, then preprocessing and visualization, then run through the phases in order with results and figures.
- wrote `experimnet.ipynb` after that.
- the structure is:
- raw session visualization
- preprocessing and before/after signal views
- notebook-local lightweight phase 1 experiment on one scalar VISp trace
- notebook-local lightweight phase 2 experiment with one-region mechanistic / hybrid latent e/i models
- notebook-local lightweight phase 3 experiment over the retained multi-region signals
- phase 4 section that can either rerun the symbolic pipeline or inspect the saved symbolic outputs
- also made the notebook write its own phase 1/2/3 checkpoints under `outputs/experimnet_notebook/` so it behaves like a separate experiment workspace instead of only piggybacking on the main saved outputs.
- did a quick structural check right after writing it:
- `python -m json.tool experimnet.ipynb`
- that passed, so the notebook file is valid json / valid notebook structure.
- i did not execute the full notebook itself here because that would mean running all four phases end-to-end, but the file is ready to open and run selectively.


## current state / what matters if i come back later

- phase 1 worked as a minimal one-region prototype and gave me the basic pipeline.
- phase 2 forced the project into explicit e/i thinking and exposed the identifiability problem immediately.
- phase 3 now has a working multi-region prototype with coupled mechanistic and hybrid models.
- phase 4 now has a working symbolic-distillation pipeline for pilot residual outputs.
- the most important scientific cautions still are:
- waveform-duration e/i separation is only a heuristic
- one-step 10 ms prediction is very easy, so strong one-step metrics do not automatically mean good dynamical modeling
- rollout is still the harder and more honest test, and it remains much weaker than one-step prediction
- if i pick this back up again, the most valuable next direction is probably:
- a harder evaluation regime where mechanistic structure actually matters more
- then stronger symbolic compression once the residual is carrying a richer signal

## 2026-04-24 (later)

- today i finally did the refactor i had been talking about for a while instead of just circling around it.
- the important change is that the repo is no longer only a latent predictor with a wc regularizer on the side.
- i kept that path as `baseline_latent_wc_regularized`, but i added a separate explicit rollout path where the latent state is really advanced step by step with
- wc derivative
- residual derivative
- total derivative
- and then a simple integrator.
- i also made the observation model much cleaner. for the explicit path the prediction now comes straight from the excitatory state by default, which is a lot easier to explain than the old mixed prediction story.

- i split the code up more cleanly too. preprocessing stayed in `dataset`, the mechanistic / residual / init / readout pieces went into the network modules, the losses are now separated from the main training loop, and evaluation saves real trajectory artifacts instead of just a couple of summary plots.
- the new variant list is:
- `baseline_latent_wc_regularized`
- `wc_only`
- `wc_plus_residual`
- `residual_only`

- one annoying detour was the environment. the local linux python had no scientific stack and the temporary `.venv` route was ugly, so i dropped that and just used the existing `NN_SP` conda env instead.

- i ran a short ablation sweep on the real dataset with a coarser downsample so all variants could finish in a reasonable amount of time.
- commands were basically the new `Project/main.py --variant ...` path for each model, using `downsample 120` and `epochs 3`.

- what i found is actually pretty informative even though the runs were short.
- `wc_only` was too rigid, exactly like i expected. test `mse` ended up around `0.2107`.
- `wc_plus_residual` ran correctly and saved all the right diagnostics, but the residual still was not as small as i want. test `mse` was `0.2094`, so only a tiny improvement over `wc_only`, and the test residual / wc ratio was basically `~0.99`.
- that is the key result from today. the new explicit residual machinery works, but the correction is still doing almost as much work as the wc backbone on the test split.
- `residual_only` was the uncomfortable sanity check. it got the best test `mse` at about `0.0131`, which means the flexible learned correction can still explain the task too well on its own.
- the preserved baseline also still matters. `baseline_latent_wc_regularized` got test `mse` around `0.0146`, which is much better than the pure explicit rollout variants in this short run.

- so the refactor itself was successful, but the science message is mixed in a useful way:
- the branch is now structurally ready for future symbolic regression because the residual is explicit and logged properly
- but it is not yet scientifically ready to trust symbolic regression on that residual as if it were a small correction term
- right now the residual is still too competitive and `residual_only` is too strong

- i also added plots for wc term vs residual term, residual / wc ratio over time, per-region residual stats, and saved the full train / val / test rollout outputs as `.npz` files.
- that was important because now i can stop guessing whether the residual is behaving well and actually inspect it directly.

- if i keep going from here, the next thing i should do is not jump straight back into symbolic regression.
- the next real step should be to make the wc backbone carry more of the forecast before asking a symbolic model to compress the residual.
- that probably means either better mechanistic parameterization, a harder training setup, or a stronger constraint that keeps the residual genuinely small instead of letting it shadow wc.

## 2026-04-25

- today was mostly about admitting that the previous explicit-loss setup was giving me numbers that looked cleaner than the actual trajectories.
- the bad sign was obvious in the test plots: several predicted traces were basically straight or close to straight lines.
- so even when the mse was not terrible, the model was still failing the part i actually care about, which is matching temporal variation.

- i kept the explicit wc + residual architecture, but i changed the training objective for the explicit variants so it is closer to the earlier project loss philosophy instead of just plain output mse plus a tiny residual penalty.
- the main thing i put back was the explicit delta-loss term:
- `mse(y_hat, y) + 0.25 * mse(delta y_hat, delta y)`
- that was missing before, and i think that was a big part of why flat predictions were slipping through.

- i also made the residual-control part more explicit.
- in the current model the residual already has a scaling factor built into it, so i treated that as the `alpha` part and penalized the actual scaled correction term directly.
- i also changed the smoothness regularization so it acts on the latent rollout state differences instead of the residual differences.
- that feels much closer to what i actually want scientifically: stabilize the latent evolution, not encourage a dead-flat output head.

- i checked whether there was any real `W_cross` equivalent left in this version of the model and there just is not.
- the current wc backbone only has local per-region wc terms plus stimulus weights.
- so i left the cross l1 / l2 scaffold in the training config and loss plumbing, but it evaluates to zero right now instead of inventing a fake penalty on something unrelated.

- i also made the evaluation much harsher in a good way.
- now it saves delta mse, per-region std / variance comparisons, flatness ratios, and a simple collapse warning flag when the predicted variability is much smaller than the target variability.
- i added those warnings directly onto the trajectory plots too, plus a separate collapse-diagnostics figure, so if the model goes back to near-constant predictions it should be impossible to miss.

- i only ran a very short sanity check after the changes, not a real training run.
- that was enough to verify the new objective is active and the new diagnostics are being written.
- the nice sign there was that the residual / wc ratio stayed small in the quick test instead of jumping to ~1 immediately.
- but the collapse flags were still on after one epoch, which is not surprising and also exactly why i do not want to pretend the problem is solved before a full run.

- so the state of things now is better but unfinished in the right way.
- the code is set up to punish flat trajectories properly, checkpoint selection now uses the anti-collapse objective instead of plain val mse, and the outputs should make failure obvious.
- what still needs to be validated is whether a full run actually produces trajectories that move with the data instead of hovering near means or trends.
