# var_mean_c07ce112a2ac

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_c07ce112a2ac/oracle_dropout_layernorm_side.py`
- Correctness: PASS
- Oracle: `25.41 us`
- `torch.compile coordinate_descent_tuning=True`: `25.41 us`
- Ratio: 1.0x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dropout layernorm side pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
