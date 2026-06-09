# var_mean_0c9738dea136


## Measured Timings
- Oracle: 98.05 us
- Compile (CDT): 93.95 us
- Ratio: 0.96x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_0c9738dea136/oracle_window_reverse_layernorm.py`
- Oracle: measured
- Ratio: 0.959x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin window-reverse residual LayerNorm scope in one shape-specialized Triton row kernel, keeping the reconstructed residual-add row tile live through correction=0 Welford var_mean, libdevice.rsqrt, affine output, final contiguous `[401408,128]` view, and `[128,3136,1]` `rsqrt * 1/128` side output, whereas Inductor already emits one fused generic Welford reduction kernel for the same full scope and CUDAGraph timing

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
