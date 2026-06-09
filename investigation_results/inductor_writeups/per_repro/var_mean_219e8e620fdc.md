# var_mean_219e8e620fdc

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_219e8e620fdc/oracle_swin_singleton_window_layernorm.py`
- Oracle: measured
- Ratio: 1.009x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin hidden-size-1024 population LayerNorm affine scope, including the `[6272,1024] -> [128,7,7,1024]` view, correction=0 fp32 mean plus centered-variance, `var + 1e-5` before `libdevice.rsqrt`, affine epilogue, singleton window metadata views/permute/views, and final contiguous `[6272,1024]` output, whereas Inductor already emits one fused persistent row-reduction kernel for this same full scope but pays a small resi

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.009x) within noise threshold
