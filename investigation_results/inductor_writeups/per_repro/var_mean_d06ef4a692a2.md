# var_mean_d06ef4a692a2

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_d06ef4a692a2/oracle_dropout_layernorm_side.py`
- Correctness: PASS
- Oracle: `28.35 us`
- `torch.compile coordinate_descent_tuning=True`: `28.1 us`
- Ratio: 0.991x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dropout layernorm side pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
