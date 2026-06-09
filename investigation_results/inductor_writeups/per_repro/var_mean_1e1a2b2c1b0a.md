# var_mean_1e1a2b2c1b0a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_1e1a2b2c1b0a/oracle_swin_droppath_layernorm.py`
- Oracle: measured
- Ratio: 1.049x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin batch drop-path residual add, hidden-size-1024 population var_mean LayerNorm, affine epilogue, singleton window view/permute/view aliases, final flatten, and live `rsqrt / 1024` side output with Inductor's generated two-stage schedule: one stateless batch RNG kernel followed by one full-scope normalization/output kernel; it differs only by being hand-written and shape-specialized, while preserving seed index 

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.049x) within noise threshold
