# var_mean_0b554bb6615a


## Measured Timings
- Oracle: 30.37 us
- Compile (CDT): 29.54 us
- Ratio: 0.97x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_0b554bb6615a/oracle_swin_window_layernorm.py`
- Oracle: measured
- Ratio: 0.998x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin residual-add LayerNorm plus unshifted 7x7 window partition scope in one Triton row-reduction kernel, preserving Inductor's generated fp32 population mean, centered-variance, libdevice.rsqrt eps=1e-5, affine epilogue, and final contiguous `[25088, 512]` clone/view layout, whereas Inductor already emits the same fused normalization and layout-store region through its generic persistent-reduction scheduler; Indu

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
