# var_mean_54ad7896eb18

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_54ad7896eb18/oracle_swin_reverse_layernorm.py`
- Oracle: measured
- Ratio: 1.038x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-grid reverse-window residual LayerNorm scope in one Triton row kernel, preserving the captured view `[6272,1024] -> [128,49,1024] -> [128,7,7,1024] -> [128,1,1,7,7,1024]`, permute `[0,1,3,2,4,5]`, final view, residual-first add with `view_629`, fp32 correction=0 centered-variance `var_mean`, generated `libdevice.rsqrt(var + 1e-5)`, affine scale/bias, and contiguous final `[6272,1024]` output, wherea

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.038x) within noise threshold
