# var_mean_c3179bad0f03

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_c3179bad0f03/oracle_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `6.91 us`
- `torch.compile coordinate_descent_tuning=True`: `6.98 us`
- Ratio: 1.009x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The embedding layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
