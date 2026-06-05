# var_mean_a5d43c43648a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_a5d43c43648a/oracle_vit_patch_layernorm.py`
- Correctness: PASS
- Oracle: `15.81 us`
- `torch.compile coordinate_descent_tuning=True`: `15.2 us`
- Ratio: 0.962x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The vit patch layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
