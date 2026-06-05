# var_mean_55b78000aab0

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_55b78000aab0/oracle_instance_norm_residual.py`
- Correctness: PASS
- Oracle: `8.13 us`
- `torch.compile coordinate_descent_tuning=True`: `8.26 us`
- Ratio: 1.016x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The instance norm residual pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
