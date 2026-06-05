# var_mean_1fad4754680a

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_1fad4754680a/oracle_embedding_layernorm.py`
- Correctness: PASS
- Oracle: `6.94 us`
- `torch.compile coordinate_descent_tuning=True`: `6.88 us`
- Ratio: 0.991 (Inductor slightly faster)
- Status: `at_floor`

## Diagnosis

Inductor is within 1% of the oracle (actually faster) for this embedding + layernorm pattern [512,768] fp16. No gap exists.

## Config exploration results
- No configs needed -- already at floor.
