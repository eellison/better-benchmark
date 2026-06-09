# var_mean_25898f81ca4d

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_25898f81ca4d/oracle_mobilevit_residual_layernorm_fold.py`
- Correctness: PASS
- Oracle: `16.1 us`
- `torch.compile coordinate_descent_tuning=True`: `16.42 us`
- Ratio: 1.02x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The mobilevit residual layernorm fold pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
