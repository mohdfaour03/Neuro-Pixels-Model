# Hybrid LSTM Architecture Execution

- [x] Re-write `HybridModel.__init__` in `src/models.py` to define `nn.LSTMCell` and `nn.Linear` projection.
- [x] Initialize the final projection weights/biases to purely `0.0`.
- [x] Modify `HybridModel.forward` to unpack `x_seq` optimally and loop standard track states `h_t`, `c_t`. 
- [x] Reroute `dE_res` and `dI_res` out from LSTM macroscopic variables into the `step()` micro-loop seamlessly.
- [x] Reconfigure `run_experiment.py` to ensure `HybridModel` executes in alongside the other benchmark targets.
- [x] Execute continuous `--mode lightweight` sequence. 
