# var_mean_5bea7483a354

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_5bea7483a354/oracle_residual_layernorm.py`
- Oracle: measured
- Ratio: 1.003x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full residual-add LayerNorm scope in one shape-specialized Triton row kernel, preserving Inductor's generated fp32 correction=0 mean then centered-variance order, eps-before-rsqrt, libdevice.rsqrt, affine epilogue, and final contiguous view, whereas Inductor already emits a single fused persistent row-reduction kernel for the same graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.003x) within noise threshold
