# var_mean_b92e4f945149

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_b92e4f945149/oracle_dropout_residual_layernorm_side.py`
- Correctness: PASS
- Oracle: `89.79 us`
- `torch.compile coordinate_descent_tuning=True`: `87.74 us`
- Ratio: 0.977x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dropout residual layernorm side pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
