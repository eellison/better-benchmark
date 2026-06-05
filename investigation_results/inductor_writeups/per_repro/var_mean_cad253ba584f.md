# var_mean_cad253ba584f

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_cad253ba584f/oracle_ghostnet_bn_cat_residual.py`
- Correctness: PASS
- Oracle: `21.66 us`
- `torch.compile coordinate_descent_tuning=True`: `22.05 us`
- Ratio: 1.018x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The ghostnet bn cat residual pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
