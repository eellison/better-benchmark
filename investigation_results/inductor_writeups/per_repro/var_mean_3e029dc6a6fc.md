# var_mean_3e029dc6a6fc

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_3e029dc6a6fc/oracle_class_token_layernorm_side.py`
- Oracle: measured
- Ratio: 0.967x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DINOv2 residual LayerNorm scope, including `[175360,768] -> [128,1370,768]`, gamma multiply before residual add, fp32 population var_mean, eps-before-rsqrt, affine class-token clone, and full `[128,1370,1]` `invstd / 768` side output, but only evaluates affine scale/bias for the token-0 rows that Repro.forward returns; Inductor currently schedules the affine LayerNorm epilogue across all `[128,1370,768]` row

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
