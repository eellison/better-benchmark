# var_mean_d6e662ef37f6

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_d6e662ef37f6/oracle_dropout_layernorm_side.py`
- Correctness: PASS
- Oracle: `14.14 us`
- `torch.compile coordinate_descent_tuning=True`: `13.95 us`
- Ratio: 0.986x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The dropout layernorm side pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
