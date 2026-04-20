# Phase 9: Stabilizing the HybridModel Autoregressive Attractor

You have accurately pinpointed the critical flaw in the HybridModel: we suffer from a massive **Train/Inference Mismatch**, and the WC core is accumulating integration drift which overwhelms the residual. 

### Why it Collapses (The Diagnosis)
1. **The Autoregressive Dissonance**: During training (and currently sliding-window testing), the LSTM residual is fed the perfect `x_seq` (the true states). However, the Mechanistic block's ODE (`self.step`) integrates sequentially using its own *predicted* `x_curr`. By step 100, the WC ODE has drifted into an attractor (a flat line), but the LSTM is outputting microscopic corrections expecting the state to still match `x_seq` perfectly.
2. **Residual Weakness**: Because the LSTM is zero-initialized and L2 penalized, its gradients vanish. It never learns to "pull" the WC core back onto the chaotic trajectory.
3. **Missing Feedback**: The LSTM doesn't even know what the WC core's predictions are, so it can't explicitly correct them.

### Proposed Fixes

1. **Remove the Train/Test Mismatch (Pure Autoregressive Rollout)**
   - In `HybridModel.forward`, we will force the LSTM to ingest `x_curr` (the model's own integrated state) instead of `x_seq`. This forces the LSTM to learn to push the model *out* of bad predicted states, matching the continuous rollout structure perfectly.

2. **Learnable Alpha Gating for Residual Authority**
   - Introduce `self.alpha = nn.Parameter(torch.tensor(0.1))` initialized small.
   - The ODE will become: `x_dot = f_mech + self.alpha * r_theta`. This allows the network to smoothly escalate the authority of the LSTM without starting at exactly `0.0` (which zeroes out gradients entirely due to multiplication if not careful).

3. **Improve Residual Feature Input**
   - Instead of just `[x_curr, u_t]`, the LSTM inputs will now be `[x_curr, u_t, f_mech]`. Providing the LSTM with the analytical WC derivative (`f_mech`) allows it to observe the physical mechanics internally and predict corrections *based* on the physics failure.

4. **Iterative Evaluation Workflow**
   - I will modify `src/models.py` with these architectural constraints, then continuously execute `python run_experiment.py --mode lightweight --model HybridModel` in the background. I will iteratively tune `dt`, sequence lengths, or scaling based strictly on the trajectory plots produced, until the flat-line attractor effect is eliminated and the Hybrid properly shadows $E(t)$ and $I(t)$.

## User Review Required
> [!IMPORTANT]
> The plan is to completely restructure `HybridModel` into a pure autoregressive sequence generator where the LSTM corrects `x_curr` via a learnable alpha limit, instead of being teacher-forced on `x_seq`. 
> Do you approve this rigorous structural stabilization workflow?
