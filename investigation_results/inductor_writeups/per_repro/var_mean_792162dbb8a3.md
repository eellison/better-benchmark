# var_mean_792162dbb8a3

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_792162dbb8a3/oracle_vit_patch_layernorm_no_position.py`
- Correctness: PASS
- Oracle: `73.47 us`
- `torch.compile coordinate_descent_tuning=True`: `72.45 us`
- Ratio: 0.986x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The vit patch layernorm no position pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
