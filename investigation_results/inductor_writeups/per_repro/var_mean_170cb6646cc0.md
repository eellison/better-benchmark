# var_mean_170cb6646cc0

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_170cb6646cc0/oracle_megatronbert_embedding_dropout_layernorm.py`
- Correctness: PASS
- Oracle: `42.88 us`
- `torch.compile coordinate_descent_tuning=True`: `43.07 us`
- Ratio: 1.004 (effectively at parity)
- Status: `at_floor`

## Diagnosis

Inductor is within 0.4% of the oracle for this MegatronBERT embedding + dropout + layernorm pattern. The gap is within measurement noise. No actionable improvement needed.

## Config exploration results
- Baseline is already at parity with oracle (1.004x).
