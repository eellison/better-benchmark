# var_mean_d22940b03b5c

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_d22940b03b5c/oracle_residual_layernorm.py`
- Correctness: PASS
- Oracle: `7.07 us`
- `torch.compile coordinate_descent_tuning=True`: `6.94 us`
- Ratio: 0.982x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The residual layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
