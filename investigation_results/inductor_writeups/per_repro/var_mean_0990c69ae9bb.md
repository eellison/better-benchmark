# var_mean_0990c69ae9bb

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_0990c69ae9bb/oracle_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `7.584 us`
- `torch.compile coordinate_descent_tuning=True`: `7.872 us`
- `torch.compile combo_kernels=True,multi_kernel=3`: `7.808 us`
- Ratio: 1.03 (within noise)
- Status: `at_floor`

## Diagnosis

Inductor is within 3% of the oracle for this embedding + layernorm pattern. The gap is within measurement noise. No actionable improvement.

## Config exploration results
- `coordinate_descent_tuning=True`: 7.872 us
- `combo_kernels=True,multi_kernel=3`: 7.808 us (best, marginal improvement)
- Gap is 1.03x -- effectively at floor.
