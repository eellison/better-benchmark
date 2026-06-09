# var_mean_36eed06607f1

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_36eed06607f1/oracle_bert_large_embedding_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `22.37 us`
- `torch.compile coordinate_descent_tuning=True`: `22.5 us`
- Ratio: 1.006x
- Status: `at_floor` (no meaningful gap)

## Diagnosis

Inductor already matches the oracle within noise (ratio < 1.05x). The bert large embedding dropout layernorm pattern is handled optimally by current Inductor codegen with coordinate_descent_tuning enabled.

## Config exploration results
- Baseline compile already at oracle level; no further config exploration needed.
