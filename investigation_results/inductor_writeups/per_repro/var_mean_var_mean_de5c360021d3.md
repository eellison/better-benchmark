# var_mean_var_mean_de5c360021d3

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_var_mean_de5c360021d3/oracle_dual_bn_relu.py`
- Correctness: PASS
- Oracle: `46.66 us`
- `torch.compile coordinate_descent_tuning=True`: `48.1 us`
- Ratio: 1.031x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dual bn relu pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
