# var_mean_c0993161a0af

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_c0993161a0af/oracle_bn_training_forward.py`
- Correctness: PASS
- Oracle: `15.36 us`
- `torch.compile coordinate_descent_tuning=True`: `15.78 us`
- Ratio: 1.027x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The bn training forward pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
