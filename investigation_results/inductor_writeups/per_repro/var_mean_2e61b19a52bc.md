# var_mean_2e61b19a52bc

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_2e61b19a52bc/oracle_mobilevit_patch_layernorm.py`
- Correctness: PASS
- Oracle: `8.96 us`
- `torch.compile coordinate_descent_tuning=True`: `8.99 us`
- Ratio: 1.004x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The mobilevit patch layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
