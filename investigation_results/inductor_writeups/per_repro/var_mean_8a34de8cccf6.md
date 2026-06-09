# var_mean_8a34de8cccf6

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_8a34de8cccf6/oracle_weight_standardization.py`
- Correctness: PASS
- Oracle: `13.18 us`
- `torch.compile coordinate_descent_tuning=True`: `13.15 us`
- Ratio: 0.998x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The weight standardization pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
