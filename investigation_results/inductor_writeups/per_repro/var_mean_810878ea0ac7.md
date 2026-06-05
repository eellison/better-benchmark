# var_mean_810878ea0ac7

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_810878ea0ac7/oracle_albert_embeddings_layernorm.py`
- Correctness: PASS
- Oracle: `7.62 us`
- `torch.compile coordinate_descent_tuning=True`: `7.71 us`
- Ratio: 1.013x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The albert embeddings layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
