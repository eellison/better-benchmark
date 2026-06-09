# var_mean_2f7685c1567b

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_2f7685c1567b/oracle_gpt2_large_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `28.67 us`
- `torch.compile coordinate_descent_tuning=True`: `28.64 us`
- Ratio: 0.999x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The gpt2 large dropout layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
