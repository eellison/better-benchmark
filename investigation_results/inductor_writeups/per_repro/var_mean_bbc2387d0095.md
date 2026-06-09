# var_mean_bbc2387d0095

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_bbc2387d0095/oracle_bn_residual_training.py`
- Correctness: PASS
- Oracle: `11.58 us`
- `torch.compile coordinate_descent_tuning=True`: `11.17 us`
- Ratio: 0.964x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The bn residual training pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
