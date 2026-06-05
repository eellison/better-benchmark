# var_mean_35394af84f60

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_35394af84f60/oracle_groupnorm_relu.py`
- Correctness: PASS
- Oracle: `6.02 us`
- `torch.compile coordinate_descent_tuning=True`: `5.89 us`
- Ratio: 0.979x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The groupnorm relu pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
