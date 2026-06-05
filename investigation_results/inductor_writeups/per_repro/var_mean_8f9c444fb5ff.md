# var_mean_8f9c444fb5ff

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_8f9c444fb5ff/oracle_residual_layernorm.py`
- Correctness: PASS
- Oracle: `8.03 us`
- `torch.compile coordinate_descent_tuning=True`: `7.81 us`
- Ratio: 0.972x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The residual layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
