# var_mean_42b3beb8ea30

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_42b3beb8ea30/oracle_dropout_residual_layernorm_transpose.py`
- Correctness: PASS
- Oracle: `12.54 us`
- `torch.compile coordinate_descent_tuning=True`: `12.06 us`
- Ratio: 0.962x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dropout residual layernorm transpose pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
