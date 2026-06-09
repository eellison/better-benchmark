# var_mean_7cebda5b34ee

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_7cebda5b34ee/oracle_embedding_layernorm_aliases.py`
- Correctness: PASS
- Oracle: `7.36 us`
- `torch.compile coordinate_descent_tuning=True`: `7.65 us`
- Ratio: 1.039x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The embedding layernorm aliases pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
