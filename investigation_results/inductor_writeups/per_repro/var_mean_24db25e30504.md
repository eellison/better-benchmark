# var_mean_24db25e30504

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_24db25e30504/oracle_longformer_embedding_position_layernorm.py`
- Correctness: PASS
- Oracle: `9.79 us`
- `torch.compile coordinate_descent_tuning=True`: `9.66 us`
- Ratio: 0.987x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The longformer embedding position layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
